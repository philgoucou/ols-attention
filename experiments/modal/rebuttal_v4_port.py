"""
Modal port of the submitted Table 1 pipeline (paper_code/real_data_benchmark_v4.py):
the first rebuttal wave. Formerly rebuttal_modal.py.

Splits the v4 Colab cell into one job per (dataset, config, seed) tuple,
runs them across many T4 GPUs concurrently, and prints the same
consolidated report at the end. Protocol of the submission: features
standardized on the training split, RAW targets (the corrected-protocol
reruns are rebuttal_ablation_warmstart.py and rebuttal_uncapped_ystd.py).
Writes, in the working directory (kept under results/rebuttal/):
    ablation_grid_results.csv   A0-A5 and RB, 8 datasets x 5 seeds
    uncapped_results.csv        OLS / RF / FT-T / RB at full N (California, Kin8nm, Protein)
    tabpfn_results.csv          TabPFN on the eight datasets

Usage:
    modal run rebuttal_v4_port.py            # everything
    modal run rebuttal_v4_port.py --part abl # ablation only
    modal run rebuttal_v4_port.py --part unc # uncapped only
    modal run rebuttal_v4_port.py --part tab # tabpfn only
    modal run rebuttal_v4_port.py::patch_protein_rb   # re-run the Protein/RB uncapped seeds, splice them in
"""

from __future__ import annotations

import concurrent.futures
import io
import pickle
import time
from contextlib import redirect_stdout, redirect_stderr

import modal

# ==================================================================
# CONFIG (identical to the v4 notebook)
# ==================================================================
N_REPEATS   = 5
TEST_FRAC   = 0.2
MAX_N       = 5000
SEED        = 42

RF_TREES    = 500
RF_MTRY     = 1 / 3

FTT_D_MODEL = 64
FTT_N_HEADS = 4
FTT_FFN_DIM = 128
FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1
FTT_EPOCHS  = 200
FTT_BS      = 256
FTT_LR      = 1e-3
FTT_WD      = 1e-4

POLY_PCA_COMP = 200
POLY_D_MODEL  = 64
POLY_FFN_DIM  = 128
POLY_EPOCHS   = 200
POLY_BS       = 256
POLY_LR       = 1e-3
POLY_DROPOUT  = 0.1
POLY_WD       = 1e-4

PAPER = {
    'California': {'A0': 0.771,   'RB': 0.769},
    'Yacht':      {'A0': 0.445,   'RB': 0.960},
    'Energy':     {'A0': 0.992,   'RB': 0.995},
    'Concrete':   {'A0': 0.652,   'RB': 0.892},
    'Airfoil':    {'A0': -75.451, 'RB': 0.927},
    'Abalone':    {'A0': 0.314,   'RB': 0.331},
    'Kin8nm':     {'A0': 0.909,   'RB': 0.917},
    'Protein':    {'A0': 0.418,   'RB': 0.453},
}

UNCAPPED = ['California', 'Kin8nm', 'Protein']

ABL = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'RB']
LBL = {
    'A0': 'FT-Transformer (paper)',
    'A1': 'FT-T + polynomial inputs',
    'A2': 'RB w/o polynomial expansion',
    'A3': 'RB w/o PCA compression',
    'A4': 'RB skeleton, softmax instead of regression',
    'A5': 'FT-T parameter-matched (~30K)',
    'RB': 'Regression Block (paper)',
}

OPENML_REGISTRY = {
    'Concrete': (44959, (1000, 1100),   8, 'Concrete compressive strength'),
    'Energy':   (41478, (700, 800),     8, 'Energy efficiency'),
    'Kin8nm':   (189,   (8000, 8300),   8, 'Kin8nm robot arm'),
    'Protein':  (44963, (45000, 46000), 9, 'Protein structure CASP'),
    'Yacht':    (42370, (300, 320),     6, 'Yacht hydrodynamics'),
    'Airfoil':  (44957, (1400, 1600),   5, 'Airfoil self-noise'),
    'Abalone':  (183,   (4100, 4200),   7, 'Abalone age prediction'),
}
DS_ORDER = ['California', 'Yacht', 'Energy', 'Concrete',
            'Airfoil',    'Abalone', 'Kin8nm', 'Protein']

# ==================================================================
# Modal image
# ==================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "torch==2.4.1",
        "tabpfn>=2.0.0",
    )
)

cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"

app = modal.App("neurips-31482-rebuttal")


# ==================================================================
# Dataset loader (verbatim from v4, with cache dir override)
# ==================================================================
def load_all_datasets_impl():
    import numpy as np
    from sklearn.datasets import fetch_california_housing, fetch_openml

    print(f"Loading {len(DS_ORDER)} datasets")
    loaded = {}
    try:
        d = fetch_california_housing()
        assert d.data.shape == (20640, 8)
        loaded['California'] = (d.data.astype(np.float32),
                                d.target.astype(np.float32))
        print("  OK California: (20640, 8)")
    except Exception as e:
        print(f"  FAIL California: {e}")

    for name in DS_ORDER:
        if name == 'California':
            continue
        data_id, (n_min, n_max), expected_p, desc = OPENML_REGISTRY[name]
        try:
            data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
            X = data.data.select_dtypes(include=[np.number]).values
            y_raw = data.target
            if y_raw is None:
                raise ValueError("Target column not set in OpenML metadata")
            if hasattr(y_raw, 'shape') and len(y_raw.shape) > 1 and y_raw.shape[1] > 1:
                y_raw = y_raw.iloc[:, 0]
            elif hasattr(y_raw, 'columns'):
                y_raw = y_raw.iloc[:, 0]
            if hasattr(y_raw, 'cat') and y_raw.dtype.name == 'category':
                y = y_raw.cat.codes.values.astype(float)
            else:
                y = np.array(y_raw).astype(float)
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            N, P = X.shape
            if not (n_min <= N <= n_max):
                print(f"  FAIL {name}: N={N} outside ({n_min}-{n_max})")
                continue
            if P != expected_p:
                print(f"  FAIL {name}: P={P} != {expected_p}")
                continue
            loaded[name] = (X.astype(np.float32), y.astype(np.float32))
            print(f"  OK {name}: ({N}, {P}) - {desc}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    print(f"{len(loaded)}/{len(DS_ORDER)} datasets ready.")
    return {k: loaded[k] for k in DS_ORDER if k in loaded}


# ==================================================================
# FT-Transformer (verbatim from v4)
# ==================================================================
def train_ft_transformer(X_train, y_train, X_test,
                         d_model=64, n_heads=4, ffn_dim=128,
                         n_blocks=3, dropout=0.1,
                         epochs=50, batch_size=256, lr=1e-3,
                         weight_decay=1e-4, seed=42, verbose=False):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model

    class FTTransformer(nn.Module):
        def __init__(self, nf, d, nh, ffn, nblk, drop):
            super().__init__()
            self.feat_embeds = nn.ModuleList([nn.Linear(1, d) for _ in range(nf)])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nh, dim_feedforward=ffn, dropout=drop,
                batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nblk)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            B = x.size(0)
            tokens = torch.stack(
                [self.feat_embeds[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
            z = self.transformer(tokens)
            return self.head(self.norm(z[:, 0, :]))

    model = FTTransformer(nf, D, n_heads, ffn_dim, n_blocks, dropout).to(device)

    N = X_train.shape[0]
    n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)

    tr_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(1, epochs // 10)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return ep / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val_loss = float('inf'); best_state = None
    patience_counter = 0; patience = 10
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xt_val), yt_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    import numpy as _np
    return _np.concatenate(preds), {'n_params': sum(p.numel() for p in model.parameters())}


# ==================================================================
# RB variants (verbatim from v4)
# ==================================================================
def train_rb_variant(X_train, y_train, X_test,
                     use_poly=True, use_pca=True, softmax_mixer=False,
                     n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL,
                     ffn_dim=POLY_FFN_DIM, epochs=POLY_EPOCHS,
                     batch_size=POLY_BS, lr=POLY_LR, dropout=POLY_DROPOUT,
                     weight_decay=POLY_WD, n_heads=4, seed=42,
                     pca_fit_cap=5000):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model

    def poly_cross(x):
        B, S, Dm = x.shape
        per = torch.cat([x, x**2 - 1], dim=-1)
        pairs = torch.cat(
            [x[:, i, :] * x[:, j, :] for i in range(S) for j in range(i, S)], dim=-1)
        return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)

    def features(x):
        return poly_cross(x) if use_poly else x

    class FrozenPCA(nn.Module):
        def __init__(self, mean, components):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer('comp', torch.from_numpy(components.astype(np.float32)))
            self.out_dim = components.shape[0]
        def forward(self, x): return (x - self.mean) @ self.comp.T

    class Identity(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.out_dim = dim
        def forward(self, x): return x

    class RBVariant(nn.Module):
        def __init__(self, nf, red_layer, d, ffn_dim, drop):
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.randn(1, nf, d) * 0.02)
            if softmax_mixer:
                self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=drop)
            else:
                self.red = red_layer
                self.W = nn.Linear(red_layer.out_dim, d)
            self.n1 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, ffn_dim), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(ffn_dim, d), nn.Dropout(drop))
            self.n2 = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def get_feat(self, emb): return self.red(features(emb))

        def forward(self, x):
            emb = self.embed_x(x)
            if softmax_mixer:
                a, _ = self.attn(emb, emb, emb)
                z = self.n1(emb + a)
            else:
                z = self.n1(emb + self.W(self.get_feat(emb)))
            return self.head(self.n2(z + self.ffn(z)).mean(1))

    red_layer = None
    if not softmax_mixer:
        torch.manual_seed(seed)
        tmp_embed = nn.Linear(1, D)
        tmp_pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
        if len(X_train) > pca_fit_cap:
            sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
            X_pca = X_train[sub]
        else:
            X_pca = X_train
        Xt = torch.from_numpy(X_pca.astype(np.float32))
        with torch.no_grad():
            emb = tmp_embed(Xt.unsqueeze(-1)) + tmp_pos
            phi_all = features(emb)
            feat_dim = phi_all.shape[-1]
            phi_all = phi_all.reshape(-1, feat_dim)
        if use_pca:
            n_comp = min(n_components, phi_all.shape[1], phi_all.shape[0])
            p = PCA(n_components=n_comp, random_state=seed); p.fit(phi_all.numpy())
            red_layer = FrozenPCA(p.mean_, p.components_)
        else:
            red_layer = Identity(feat_dim)

    torch.manual_seed(seed)
    model = RBVariant(nf, red_layer, D, ffn_dim, dropout).to(device)

    if not softmax_mixer:
        model.eval()
        WS_N = min(len(X_train), 20000)
        X_ws = X_train[:WS_N]
        phis = []
        with torch.no_grad():
            for i in range(0, WS_N, 4096):
                xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
                phis.append(model.get_feat(model.embed_x(xb)).mean(1).cpu().numpy())
        phi_pooled = np.concatenate(phis)
        reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
        with torch.no_grad():
            model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
            model.W.bias.data.zero_()
            model.head.weight.data.zero_()
            model.head.weight.data[0, 0] = 1.0
            model.head.bias.data.fill_(float(reg.intercept_))

    N = X_train.shape[0]; n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    tr_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    best, best_state, pat = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xt_val), yt_val).item()
        if vl < best:
            best, best_state, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10:
                break
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    import numpy as _np
    return _np.concatenate(preds), {
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


# ==================================================================
# A5 param matching helpers (verbatim from v4)
# ==================================================================
def ftt_param_count(nf, d, nh, ffn, nblk):
    import torch.nn as nn
    m = nn.ModuleDict({
        'emb': nn.ModuleList([nn.Linear(1, d) for _ in range(nf)]),
        'enc': nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d, nhead=nh, dim_feedforward=ffn, dropout=0.1,
            batch_first=True, activation='gelu'), num_layers=nblk),
        'norm': nn.LayerNorm(d),
        'head': nn.Linear(d, 1),
    })
    return sum(p.numel() for p in m.parameters()) + d


def rb_param_count(nf, d=POLY_D_MODEL, ffn=POLY_FFN_DIM, ncomp=POLY_PCA_COMP):
    import torch.nn as nn
    m = nn.ModuleDict({
        'e':  nn.Linear(1, d),
        'W':  nn.Linear(ncomp, d),
        'n1': nn.LayerNorm(d),
        'f':  nn.Sequential(nn.Linear(d, ffn), nn.Linear(ffn, d)),
        'n2': nn.LayerNorm(d),
        'h':  nn.Linear(d, 1),
    })
    return sum(p.numel() for p in m.parameters()) + nf * d


def pick_a5_dmodel(nf, target):
    best_d, best_gap = None, float('inf')
    for d in [16, 20, 24, 28, 32, 36, 40, 48]:
        if d % 4:
            continue
        gap = abs(ftt_param_count(nf, d, 4, 2 * d, FTT_N_BLOCKS) - target)
        if gap < best_gap:
            best_d, best_gap = d, gap
    return best_d


# ==================================================================
# Dataset slice for one repeat (identical to v4's inner loop)
# ==================================================================
def prep_split(X_full, y_full, seed_idx, need_poly=False):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures

    N_full, P = X_full.shape
    if N_full > MAX_N:
        rng = np.random.RandomState(SEED)  # v4 uses np.random.seed(SEED) at module scope
        idx = rng.choice(N_full, MAX_N, replace=False)
        X_use, y_use = X_full[idx], y_full[idx]
    else:
        X_use, y_use = X_full, y_full

    seed = SEED + seed_idx * 1000
    X_train, X_test, y_train, y_test = train_test_split(
        X_use, y_use, test_size=TEST_FRAC, random_state=seed)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_train); X_te = sc.transform(X_test)

    if need_poly:
        pf = PolynomialFeatures(degree=2, include_bias=False)
        X_tr_pf = pf.fit_transform(X_tr); X_te_pf = pf.transform(X_te)
        sc2 = StandardScaler()
        X_tr_pf = sc2.fit_transform(X_tr_pf); X_te_pf = sc2.transform(X_te_pf)
        return X_tr, X_te, y_train, y_test, X_tr_pf, X_te_pf, P, seed

    return X_tr, X_te, y_train, y_test, None, None, P, seed


# ==================================================================
# Modal remote functions
# ==================================================================
@app.function(
    image=image,
    volumes={CACHE_DIR: cache_vol},
    timeout=1200,
)
def fetch_and_cache():
    """Fetch every dataset once and pickle to the shared Volume."""
    import os
    os.environ['SCIKIT_LEARN_DATA'] = CACHE_DIR
    pkl_path = os.path.join(CACHE_DIR, "datasets_v1.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            ds = pickle.load(f)
        print(f"Loaded cached datasets: {list(ds.keys())}")
        return {k: (v[0].shape, v[1].shape) for k, v in ds.items()}
    ds = load_all_datasets_impl()
    with open(pkl_path, "wb") as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache_vol.commit()
    print(f"Cached to {pkl_path}")
    return {k: (v[0].shape, v[1].shape) for k, v in ds.items()}


def _load_cached():
    import os, pickle as _pk
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return _pk.load(f)


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={CACHE_DIR: cache_vol},
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def run_ablation(dataset_name: str, config: str, seed_idx: int):
    """One ablation cell: dataset x config x seed."""
    import time as _time, traceback
    from sklearn.metrics import r2_score
    t0 = _time.time()
    try:
        ds = _load_cached()
        X_full, y_full = ds[dataset_name]
        need_poly = (config == 'A1')
        X_tr, X_te, y_train, y_test, X_tr_pf, X_te_pf, P, seed = prep_split(
            X_full, y_full, seed_idx, need_poly=need_poly)

        if config == 'A0':
            y_pred, info = train_ft_transformer(
                X_tr, y_train, X_te, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                ffn_dim=FTT_FFN_DIM, n_blocks=FTT_N_BLOCKS, dropout=FTT_DROPOUT,
                epochs=FTT_EPOCHS, batch_size=FTT_BS, lr=FTT_LR,
                weight_decay=FTT_WD, seed=seed)
        elif config == 'A1':
            y_pred, info = train_ft_transformer(
                X_tr_pf, y_train, X_te_pf, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                ffn_dim=FTT_FFN_DIM, n_blocks=FTT_N_BLOCKS, dropout=FTT_DROPOUT,
                epochs=FTT_EPOCHS, batch_size=FTT_BS, lr=FTT_LR,
                weight_decay=FTT_WD, seed=seed)
        elif config == 'A2':
            y_pred, info = train_rb_variant(X_tr, y_train, X_te, use_poly=False, seed=seed)
        elif config == 'A3':
            y_pred, info = train_rb_variant(X_tr, y_train, X_te, use_pca=False, seed=seed)
        elif config == 'A4':
            y_pred, info = train_rb_variant(X_tr, y_train, X_te, softmax_mixer=True, seed=seed)
        elif config == 'A5':
            a5_d = pick_a5_dmodel(P, rb_param_count(P))
            y_pred, info = train_ft_transformer(
                X_tr, y_train, X_te, d_model=a5_d, n_heads=4, ffn_dim=2 * a5_d,
                n_blocks=FTT_N_BLOCKS, dropout=FTT_DROPOUT,
                epochs=FTT_EPOCHS, batch_size=FTT_BS, lr=FTT_LR,
                weight_decay=FTT_WD, seed=seed)
        elif config == 'RB':
            y_pred, info = train_rb_variant(X_tr, y_train, X_te, seed=seed)
        else:
            raise ValueError(f"unknown config {config}")

        r2 = float(r2_score(y_test, y_pred))
        return {
            'kind': 'ablation',
            'dataset': dataset_name,
            'config': config,
            'seed_idx': seed_idx,
            'seed': seed,
            'r2': r2,
            'params': int(info.get('n_params', 0)),
            'wall_s': _time.time() - t0,
            'error': None,
        }
    except Exception as e:
        return {
            'kind': 'ablation', 'dataset': dataset_name, 'config': config,
            'seed_idx': seed_idx, 'r2': float('nan'), 'params': None,
            'wall_s': _time.time() - t0,
            'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={CACHE_DIR: cache_vol},
    cpu=4.0, memory=16384,
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def run_uncapped(dataset_name: str, model_name: str, seed_idx: int):
    """One uncapped-N cell: dataset x {OLS,RF,FTT,RB} x seed on FULL data."""
    import time as _time, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    t0 = _time.time()
    try:
        ds = _load_cached()
        X_full, y_full = ds[dataset_name]
        N_full, P = X_full.shape
        seed = SEED + seed_idx * 1000
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=TEST_FRAC, random_state=seed)
        sc = StandardScaler(); X_tr = sc.fit_transform(X_train); X_te = sc.transform(X_test)

        if model_name == 'OLS':
            m = LinearRegression().fit(X_tr, y_train)
            r2 = r2_score(y_test, m.predict(X_te))
        elif model_name == 'RF':
            m = RandomForestRegressor(
                n_estimators=RF_TREES, max_features=RF_MTRY,
                n_jobs=-1, random_state=seed).fit(X_tr, y_train)
            r2 = r2_score(y_test, m.predict(X_te))
        elif model_name == 'FTT':
            y_pred, _ = train_ft_transformer(
                X_tr, y_train, X_te, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                ffn_dim=FTT_FFN_DIM, n_blocks=FTT_N_BLOCKS, dropout=FTT_DROPOUT,
                epochs=FTT_EPOCHS, batch_size=FTT_BS, lr=FTT_LR,
                weight_decay=FTT_WD, seed=seed)
            r2 = r2_score(y_test, y_pred)
        elif model_name == 'RB':
            y_pred, _ = train_rb_variant(X_tr, y_train, X_te, seed=seed)
            r2 = r2_score(y_test, y_pred)
        else:
            raise ValueError(f"unknown model {model_name}")

        return {
            'kind': 'uncapped', 'dataset': dataset_name, 'N': int(N_full),
            'model': model_name, 'seed_idx': seed_idx, 'r2': float(r2),
            'wall_s': _time.time() - t0, 'error': None,
        }
    except Exception as e:
        return {
            'kind': 'uncapped', 'dataset': dataset_name, 'N': None,
            'model': model_name, 'seed_idx': seed_idx, 'r2': float('nan'),
            'wall_s': _time.time() - t0,
            'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={CACHE_DIR: cache_vol},
    memory=16384,
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def run_tabpfn(dataset_name: str, seed_idx: int):
    """One TabPFN reference cell."""
    import time as _time, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    t0 = _time.time()
    try:
        import torch
        from tabpfn import TabPFNRegressor
        ds = _load_cached()
        X_full, y_full = ds[dataset_name]
        N_full, P = X_full.shape
        if N_full > MAX_N:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(N_full, MAX_N, replace=False)
            X_use, y_use = X_full[idx], y_full[idx]
        else:
            X_use, y_use = X_full, y_full
        seed = SEED + seed_idx * 1000
        X_train, X_test, y_train, y_test = train_test_split(
            X_use, y_use, test_size=TEST_FRAC, random_state=seed)
        sc = StandardScaler(); X_tr = sc.fit_transform(X_train); X_te = sc.transform(X_test)
        m = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
        m.fit(X_tr, y_train)
        r2 = float(r2_score(y_test, m.predict(X_te)))
        return {
            'kind': 'tabpfn', 'dataset': dataset_name, 'seed_idx': seed_idx,
            'r2': r2, 'wall_s': _time.time() - t0, 'error': None,
        }
    except Exception as e:
        return {
            'kind': 'tabpfn', 'dataset': dataset_name, 'seed_idx': seed_idx,
            'r2': float('nan'), 'wall_s': _time.time() - t0,
            'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# ==================================================================
# Aggregation / report (same shape as v4 consolidated report)
# ==================================================================
def build_report(abl_results, unc_results, tab_results, wall_s):
    import numpy as np
    import pandas as pd
    lines = []
    add = lines.append

    add("#" * 78)
    add(f"# CONSOLIDATED REPORT ({wall_s/60:.1f} min wall clock)")
    add("#" * 78)

    # Ablation
    if abl_results:
        dfa = pd.DataFrame(abl_results)
        errs = dfa[dfa['error'].notna()]
        if len(errs):
            add(f"\n[ablation errors: {len(errs)}]")
            for _, row in errs.iterrows():
                add(f"  {row['dataset']}/{row['config']}/seed{row['seed_idx']}: "
                    f"{row['error'].splitlines()[0]}")
        agg = dfa.groupby(['dataset', 'config']).agg(
            r2=('r2', 'mean'), sd=('r2', 'std'),
            params=('params', 'max')).reset_index()
        piv = agg.pivot(index='dataset', columns='config', values='r2').reindex(
            [d for d in DS_ORDER if d in agg['dataset'].unique()])[ABL]
        sdp = agg.pivot(index='dataset', columns='config', values='sd').reindex(piv.index)[ABL]

        add("\n--- Ablation grid: mean R2 (all cells) ---")
        add(piv.round(3).to_string())
        add("\n--- Ablation grid: sd across reps ---")
        add(sdp.round(3).to_string())

        add("\n--- Verification vs paper Table 1 (tol 0.02; Airfoil A0 by sign) ---")
        fails = 0
        for ds, tgt in PAPER.items():
            if ds not in piv.index:
                continue
            for c in ['A0', 'RB']:
                got = piv.loc[ds, c]
                ok = (got < 0) if (c == 'A0' and ds == 'Airfoil') \
                    else (abs(got - tgt[c]) <= 0.02)
                fails += (not ok)
                add(f"  {ds:<11} {c}: got {got:+.3f}  "
                    f"target {tgt[c]:+.3f}  {'OK' if ok else '** DRIFT **'}")
        add("VERIFIED: safe to quote." if fails == 0
            else f"{fails} DRIFTED CELLS: reconcile before quoting new numbers.")

        ranks = piv.rank(axis=1, ascending=False)
        p8 = agg[agg['dataset'] == 'California'].set_index('config')['params']
        add("\n--- OpenReview markdown: ablation summary ---")
        add("| Config | Mean R2 | Avg rank | Wins vs FT-T | Params (P=8) |")
        add("|---|---|---|---|---|")
        for c in ABL:
            wins = int((piv[c] > piv['A0']).sum()) if c != 'A0' else '-'
            pv = int(p8[c]) if c in p8.index and pd.notna(p8[c]) else None
            pc = f"{pv:,}" if pv else '-'
            add(f"| {LBL[c]} | {piv[c].mean():.3f} | {ranks[c].mean():.2f} | {wins} | {pc} |")

        add("\n--- OpenReview markdown: ablation per dataset ---")
        add("| Dataset | " + " | ".join(ABL) + " |")
        add("|---" * (len(ABL) + 1) + "|")
        for ds in piv.index:
            cells = " | ".join(
                f"{piv.loc[ds, c]:.3f}" if piv.loc[ds, c] > -1 else "<0" for c in ABL)
            add(f"| {ds} | {cells} |")

        add("\n--- LaTeX: ablation per dataset ---")
        add(r"\begin{tabular}{l" + "c" * len(ABL) + "}")
        add(r"\toprule")
        add("Dataset & " + " & ".join(ABL) + r" \\ \midrule")
        for ds in piv.index:
            cells = " & ".join(
                f"{piv.loc[ds, c]:.3f}" if piv.loc[ds, c] > -1 else "$<$0" for c in ABL)
            add(f"{ds} & {cells}" + r" \\")
        add(r"\bottomrule\end{tabular}")

    # Uncapped
    if unc_results:
        dfu = pd.DataFrame(unc_results)
        errs = dfu[dfu['error'].notna()]
        if len(errs):
            add(f"\n[uncapped errors: {len(errs)}]")
            for _, row in errs.iterrows():
                add(f"  {row['dataset']}/{row['model']}/seed{row['seed_idx']}: "
                    f"{row['error'].splitlines()[0]}")
        agg = dfu.groupby(['dataset', 'model']).agg(
            r2=('r2', 'mean'), sd=('r2', 'std'), N=('N', 'max')).reset_index()
        pivu = agg.pivot(index='dataset', columns='model', values='r2')[['OLS', 'RF', 'FTT', 'RB']]
        add("\n--- Uncapped N: mean R2 ---")
        add(pivu.round(3).to_string())
        add("\n--- OpenReview markdown: uncapped ---")
        add("| Dataset | N | OLS | RF | FT-T | Reg.Blk |")
        add("|---|---|---|---|---|---|")
        for ds in pivu.index:
            Nfull = int(agg[agg['dataset'] == ds]['N'].iloc[0])
            add(f"| {ds} | {Nfull} | " + " | ".join(
                f"{pivu.loc[ds, m]:.3f}" for m in ['OLS', 'RF', 'FTT', 'RB']) + " |")

    # TabPFN
    if tab_results:
        dft = pd.DataFrame(tab_results)
        errs = dft[dft['error'].notna()]
        if len(errs):
            add(f"\n[tabpfn errors: {len(errs)}]")
            for _, row in errs.iterrows():
                add(f"  {row['dataset']}/seed{row['seed_idx']}: "
                    f"{row['error'].splitlines()[0]}")
        agg = dft.groupby('dataset').agg(r2=('r2', 'mean'), sd=('r2', 'std')).reset_index()
        add("\n--- TabPFN reference column ---")
        for _, row in agg.iterrows():
            add(f"  {row['dataset']:<11} TabPFN R2 = {row['r2']:+.4f} (sd {row['sd']:.4f})")

    return "\n".join(lines)


# ==================================================================
# Local entrypoint: fan out everything concurrently
# ==================================================================
@app.local_entrypoint()
def main(part: str = "all"):
    import pandas as pd

    t0 = time.time()
    print(f">> Prewarming dataset cache in the Modal volume")
    shapes = fetch_and_cache.remote()
    for k, sh in shapes.items():
        print(f"   {k}: X {sh[0]}, y {sh[1]}")

    # Build job lists
    abl_args = [(ds, cfg, s) for ds in shapes for cfg in ABL for s in range(N_REPEATS)]
    unc_args = [(ds, m, s) for ds in UNCAPPED if ds in shapes
                for m in ['OLS', 'RF', 'FTT', 'RB'] for s in range(N_REPEATS)]
    tab_args = [(ds, s) for ds in shapes for s in range(N_REPEATS)]

    do_abl = part in ("all", "abl")
    do_unc = part in ("all", "unc")
    do_tab = part in ("all", "tab")

    print(f">> Fanning out: "
          f"{len(abl_args) if do_abl else 0} ablation + "
          f"{len(unc_args) if do_unc else 0} uncapped + "
          f"{len(tab_args) if do_tab else 0} tabpfn jobs")

    abl_results, unc_results, tab_results = [], [], []

    def run_abl():
        return list(run_ablation.starmap(abl_args, order_outputs=False, return_exceptions=False))
    def run_unc():
        return list(run_uncapped.starmap(unc_args, order_outputs=False, return_exceptions=False))
    def run_tab():
        return list(run_tabpfn.starmap(tab_args, order_outputs=False, return_exceptions=False))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futs = {}
        if do_abl: futs['abl'] = ex.submit(run_abl)
        if do_unc: futs['unc'] = ex.submit(run_unc)
        if do_tab: futs['tab'] = ex.submit(run_tab)
        for tag, f in futs.items():
            r = f.result()
            print(f">> {tag} finished: {len(r)} results in {time.time()-t0:.1f}s")
            if tag == 'abl': abl_results = r
            if tag == 'unc': unc_results = r
            if tag == 'tab': tab_results = r

    # Save raw
    if abl_results:
        pd.DataFrame(abl_results).to_csv("ablation_grid_results.csv", index=False)
    if unc_results:
        pd.DataFrame(unc_results).to_csv("uncapped_results.csv", index=False)
    if tab_results:
        pd.DataFrame(tab_results).to_csv("tabpfn_results.csv", index=False)

    # Consolidated report
    report = build_report(abl_results, unc_results, tab_results, time.time() - t0)
    print("\n" + report)
    with open("consolidated_report.txt", "w") as f:
        f.write(report)
    print("\nSaved: ablation_grid_results.csv, uncapped_results.csv, "
          "tabpfn_results.csv, consolidated_report.txt")


@app.local_entrypoint()
def patch_protein_rb():
    """Rerun Protein/RB uncapped seeds after fixing the warmstart batching bug,
    splice into uncapped_results.csv, and rebuild the consolidated report from
    the on-disk CSVs. Everything else is left untouched."""
    import os, pandas as pd
    t0 = time.time()
    ds_name, model_name = 'Protein', 'RB'
    args = [(ds_name, model_name, s) for s in range(N_REPEATS)]
    print(f">> Rerunning {len(args)} Protein/RB uncapped jobs (post-fix)")
    new_results = list(run_uncapped.starmap(args, order_outputs=False))
    for r in new_results:
        print(f"   seed {r['seed_idx']}: r2={r['r2']:+.4f}  wall={r['wall_s']:.1f}s "
              f"err={None if r['error'] is None else r['error'].splitlines()[0]}")

    # Splice into uncapped_results.csv (drop old Protein/RB rows, append fresh)
    csv_path = "uncapped_results.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"expected {csv_path} in cwd from a prior full run")
    df = pd.read_csv(csv_path)
    mask = (df['dataset'] == ds_name) & (df['model'] == model_name)
    print(f">> dropping {int(mask.sum())} old Protein/RB rows from {csv_path}")
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame(new_results)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f">> {csv_path} now has {len(df)} rows")

    # Rebuild consolidated report from all three on-disk CSVs
    abl_rows = pd.read_csv("ablation_grid_results.csv").to_dict('records') \
        if os.path.exists("ablation_grid_results.csv") else []
    tab_rows = pd.read_csv("tabpfn_results.csv").to_dict('records') \
        if os.path.exists("tabpfn_results.csv") else []
    unc_rows = df.to_dict('records')
    # NaN error strings survive CSV round-trip as float nan → coerce to None
    for rows in (abl_rows, unc_rows, tab_rows):
        for row in rows:
            if 'error' in row and isinstance(row['error'], float):
                row['error'] = None

    report = build_report(abl_rows, unc_rows, tab_rows, time.time() - t0)
    print("\n" + report)
    with open("consolidated_report.txt", "w") as f:
        f.write(report)
    print("\nUpdated: uncapped_results.csv, consolidated_report.txt")
