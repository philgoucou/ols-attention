"""
Cold-start twin of row A7 (App. E of the revised paper): the widened Regression
Block of rebuttal_a7_widened.py trained WITHOUT the Ridge warm start.

Formerly rebuttal_a7nowarm.py. Everything is identical to rebuttal_a7_widened.py --
the (d_model, n_components) search to FT-Transformer's ~100K budget, the frozen PCA,
seeds, capped N=5000, training loop, corrected (y-standardized) protocol -- except
that the Ridge fit is computed and then DISCARDED, so the mixer and the head keep
their default initialization. This isolates whether the warm start is what makes
the extra width irrelevant, or whether the polynomial feature space is simply
saturated well below 260 components. The rows it writes keep 'config': 'A6', the
pre-renumbering label of the paper's A7 row.

Deployed-app pattern: `modal deploy rebuttal_a7_widened_nowarm.py` (app
neurips-31482-a7nowarm), spawn run_a6(dataset, seed_idx) over 8 datasets x 5 seeds
with tag 'a7nowarm', then python collectors/a7nowarm_collect.py ->
results/audit/audit_a7nowarm_results.csv.
"""
from __future__ import annotations
import pickle
import modal

N_REPEATS = 5
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42

POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4
FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128; FTT_N_BLOCKS = 3

DS_ORDER = ['California', 'Yacht', 'Energy', 'Concrete',
            'Airfoil',    'Abalone', 'Kin8nm', 'Protein']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-a7nowarm")
CODE_VERSION = "v3-a7-nowarm"


def _load_cached():
    import os
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def prep_capped(X_full, y_full, seed_idx):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    N_full, P = X_full.shape
    if N_full > MAX_N:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(N_full, MAX_N, replace=False)
        X_use, y_use = X_full[idx], y_full[idx]
    else:
        X_use, y_use = X_full, y_full
    seed = SEED + seed_idx * 1000
    Xtr, Xte, ytr, yte = train_test_split(X_use, y_use, test_size=TEST_FRAC, random_state=seed)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    # CORRECTED PROTOCOL: standardize the target on train too. The original A6 run
    # standardized X only, so its numbers were not comparable to the y-std ablation
    # grid. R2 is scale-invariant in principle, but unscaled targets change the
    # optimization, which is exactly what the Table 1 correction was about.
    scy = StandardScaler()
    ytr = scy.fit_transform(np.asarray(ytr, dtype='float64').reshape(-1, 1)).ravel()
    yte = scy.transform(np.asarray(yte, dtype='float64').reshape(-1, 1)).ravel()
    return (Xtr.astype('float32'), ytr.astype('float32'),
            Xte.astype('float32'), yte.astype('float32'), P, seed)


def ftt_param_count(nf, d, nh, ffn, nblk):
    import torch.nn as nn
    m = nn.ModuleDict({
        'emb': nn.ModuleList([nn.Linear(1, d) for _ in range(nf)]),
        'enc': nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d, nhead=nh, dim_feedforward=ffn, dropout=0.1,
            batch_first=True, activation='gelu'), num_layers=nblk),
        'norm': nn.LayerNorm(d), 'head': nn.Linear(d, 1)})
    return sum(p.numel() for p in m.parameters()) + d


def rb_param_count(nf, d, ffn, ncomp):
    import torch.nn as nn
    m = nn.ModuleDict({'e': nn.Linear(1, d), 'W': nn.Linear(ncomp, d),
                       'n1': nn.LayerNorm(d),
                       'f': nn.Sequential(nn.Linear(d, ffn), nn.Linear(ffn, d)),
                       'n2': nn.LayerNorm(d), 'h': nn.Linear(d, 1)})
    return sum(p.numel() for p in m.parameters()) + nf * d


def pick_a6_dims(nf):
    """Search (d_model, n_components), ffn=2d, to match FT-T's budget at this P."""
    target = ftt_param_count(nf, FTT_D_MODEL, FTT_N_HEADS, FTT_FFN_DIM, FTT_N_BLOCKS)
    best, best_gap = None, float('inf')
    for d in (96, 112, 128, 144):
        for ncomp in (200, 230, 260, 290, 320, 360, 400):
            gap = abs(rb_param_count(nf, d, 2 * d, ncomp) - target)
            if gap < best_gap:
                best, best_gap = (d, ncomp), gap
    return best[0], best[1], target


# ---- train_rb_variant: verbatim RB path from rebuttal_v4_port.py (formerly rebuttal_modal.py) ----
def train_rb_variant(X_train, y_train, X_test,
                     n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL,
                     ffn_dim=POLY_FFN_DIM, epochs=POLY_EPOCHS,
                     batch_size=POLY_BS, lr=POLY_LR, dropout=POLY_DROPOUT,
                     weight_decay=POLY_WD, seed=42, pca_fit_cap=5000):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model

    def poly_cross(x):
        B, S, Dm = x.shape
        per = torch.cat([x, x**2 - 1], dim=-1)
        pairs = torch.cat([x[:, i, :] * x[:, j, :] for i in range(S) for j in range(i, S)], dim=-1)
        return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)

    class FrozenPCA(nn.Module):
        def __init__(self, mean, components):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer('comp', torch.from_numpy(components.astype(np.float32)))
            self.out_dim = components.shape[0]
        def forward(self, x): return (x - self.mean) @ self.comp.T

    class RBVariant(nn.Module):
        def __init__(self, nf, red_layer, d, ffn_dim, drop):
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.randn(1, nf, d) * 0.02)
            self.red = red_layer
            self.W = nn.Linear(red_layer.out_dim, d)
            self.n1 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.ReLU(), nn.Dropout(drop),
                                     nn.Linear(ffn_dim, d), nn.Dropout(drop))
            self.n2 = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def get_feat(self, emb): return self.red(poly_cross(emb))
        def forward(self, x):
            emb = self.embed_x(x)
            z = self.n1(emb + self.W(self.get_feat(emb)))
            return self.head(self.n2(z + self.ffn(z)).mean(1))

    torch.manual_seed(seed)
    tmp_embed = nn.Linear(1, D); tmp_pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    Xt = torch.from_numpy(X_pca.astype(np.float32))
    with torch.no_grad():
        emb = tmp_embed(Xt.unsqueeze(-1)) + tmp_pos
        phi_all = poly_cross(emb)
        feat_dim = phi_all.shape[-1]
        phi_all = phi_all.reshape(-1, feat_dim)
    n_comp = min(n_components, phi_all.shape[1], phi_all.shape[0])
    p = PCA(n_components=n_comp, random_state=seed); p.fit(phi_all.numpy())
    red_layer = FrozenPCA(p.mean_, p.components_)

    torch.manual_seed(seed)
    model = RBVariant(nf, red_layer, D, ffn_dim, dropout).to(device)

    model.eval()
    WS_N = min(len(X_train), 20000)
    X_ws = X_train[:WS_N]
    phis = []
    with torch.no_grad():
        for i in range(0, WS_N, 4096):
            xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
            phis.append(model.get_feat(model.embed_x(xb)).mean(1).cpu().numpy())
    phi_pooled = np.concatenate(phis)
    # WARM START DISABLED. Everything else (widened d_model/ncomp, PCA fit, seeds,
    # training loop) is identical to A7. This isolates whether the ridge warm start
    # is what makes extra width irrelevant, or whether the polynomial feature space
    # is simply saturated well below 260 components.
    reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])  # fit but DISCARD

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
    return np.concatenate(preds), {
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_a6(dataset_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        d, ncomp, target = pick_a6_dims(P)
        pred, info = train_rb_variant(Xtr, ytr, Xte, n_components=ncomp,
                                      d_model=d, ffn_dim=2 * d, seed=seed)
        return {'kind': 'a7nowarm', 'code_version': CODE_VERSION, 'dataset': dataset_name, 'config': 'A6',
                'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'params': int(info['n_params']), 'd_model': d, 'ncomp': ncomp,
                'ftt_target': int(target), 'wall_s': _t.time() - t0, 'error': None}
    except Exception as e:
        return {'kind': 'a7nowarm', 'code_version': CODE_VERSION, 'dataset': dataset_name, 'config': 'A6',
                'seed_idx': seed_idx, 'r2': float('nan'), 'params': None,
                'd_model': None, 'ncomp': None, 'ftt_target': None,
                'wall_s': _t.time() - t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
