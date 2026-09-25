"""
Sublayer-depth factorial: is it REGRESSION-MIXER depth that saturates, or depth
of any kind on these tasks?

The depth sweep stacked whole blocks (mixer+FFN together), so "depth doesn't help"
could mean either (a) the regression mixer is idempotent — stacking regressions on
an already-regressed representation adds nothing, which is the paper's App. A.5
argument — or (b) these datasets are simply too small/simple for any extra depth.

This separates them. The block is decoupled into two stages:

    h = embed(x) + pos
    stage 1:  n_mix x [ LN(h + W_i(PCA_i(polycross(h)))) ]     <- regression mixers
    stage 2:  n_ffn x [ LN(h + FFN_j(h)) ]                     <- plain FFN sublayers
    out = head(mean_tokens(h))

(n_mix=1, n_ffn=1) is EXACTLY the shipped Regression Block. The 2x2 over
{1,3} x {1,3} then reads:

    (1,1) baseline        (1,3) FFN depth only  -> does non-mixer depth help?
    (3,1) mixer depth only (3,3) both
       -> if (1,3) > (1,1) but (3,1) ~ (1,1), mixer depth specifically saturates
          and the idempotency argument is confirmed against a real control.
       -> if (1,3) ~ (1,1) too, depth of any kind is useless here and the honest
          claim is about the tasks, not about regression-as-attention.

Only the FIRST mixer is Ridge warm-started (the paper's mechanism); deeper mixers
use standard init. Each mixer's frozen PCA is fit on its own input representation.
Protocol: y-standardized (the going-forward protocol), capped N=5000, 5 seeds.
Base components match the shipped block (shared tokenizer, mean-pool, ReLU) so
(1,1) is directly comparable to the cur_L1 numbers already measured.

Deploy: modal deploy rebuttal_sublayer.py   (app: neurips-31482-sublayer)
"""
from __future__ import annotations
import pickle
import modal

TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42
POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4

DS_ORDER = ['California', 'Yacht', 'Energy', 'Concrete',
            'Airfoil',    'Abalone', 'Kin8nm', 'Protein']
GRID = [(1, 1), (1, 0), (2, 0), (3, 0)]   # FFN-free variants vs baseline

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-sublayer")


def _load_cached():
    import os
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def prep(X_full, y_full, seed_idx):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    N_full, P = X_full.shape
    if N_full > MAX_N:
        idx = np.random.RandomState(SEED).choice(N_full, MAX_N, replace=False)
        X_full, y_full = X_full[idx], y_full[idx]
    seed = SEED + seed_idx * 1000
    Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=TEST_FRAC, random_state=seed)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    scy = StandardScaler()
    ytr = scy.fit_transform(np.asarray(ytr, dtype='float64').reshape(-1, 1)).ravel()
    yte = scy.transform(np.asarray(yte, dtype='float64').reshape(-1, 1)).ravel()
    return (Xtr.astype('float32'), ytr.astype('float32'),
            Xte.astype('float32'), yte.astype('float32'), P, seed)


def train_rb_sublayers(X_train, y_train, X_test, n_mix=1, n_ffn=1,
                       d_model=POLY_D_MODEL, ffn_dim=POLY_FFN_DIM,
                       n_components=POLY_PCA_COMP, epochs=POLY_EPOCHS,
                       batch_size=POLY_BS, lr=POLY_LR, dropout=POLY_DROPOUT,
                       weight_decay=POLY_WD, seed=42, pca_fit_cap=5000):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model

    def polycross(x):
        B, S, Dm = x.shape
        per = torch.cat([x, x ** 2 - 1], dim=-1)
        pairs = torch.cat([x[:, i, :] * x[:, j, :]
                           for i in range(S) for j in range(i, S)], dim=-1)
        return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)

    class FrozenPCA(nn.Module):
        def __init__(self, mean, comp):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer('comp', torch.from_numpy(comp.astype(np.float32)))
            self.out_dim = comp.shape[0]
        def forward(self, x): return (x - self.mean) @ self.comp.T

    class MixerSublayer(nn.Module):
        """LN(h + W(PCA(polycross(h)))) — the regression-as-attention sublayer."""
        def __init__(self, red):
            super().__init__()
            self.red = red; self.W = nn.Linear(red.out_dim, D); self.n = nn.LayerNorm(D)
        def feat(self, h): return self.red(polycross(h))
        def forward(self, h): return self.n(h + self.W(self.feat(h)))

    class FFNSublayer(nn.Module):
        """LN(h + FFN(h)) — plain position-wise feed-forward, no mixing."""
        def __init__(self):
            super().__init__()
            self.ffn = nn.Sequential(nn.Linear(D, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(ffn_dim, D), nn.Dropout(dropout))
            self.n = nn.LayerNorm(D)
        def forward(self, h): return self.n(h + self.ffn(h))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, D)
            self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.mixers = nn.ModuleList()
            self.ffns = nn.ModuleList()
            self.head = nn.Linear(D, 1)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def upto_mixer(self, x, k):
            h = self.embed_x(x)
            for i in range(k): h = self.mixers[i](h)
            return h
        def forward(self, x):
            h = self.embed_x(x)
            for m in self.mixers: h = m(h)
            for f in self.ffns: h = f(h)
            return self.head(h.mean(1))

    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    Xp = torch.from_numpy(X_pca.astype(np.float32)).to(device)

    torch.manual_seed(seed)
    model = Net().to(device)
    with torch.no_grad():
        feat_dim = polycross(model.embed_x(Xp[:8])).shape[-1]

    # build mixers sequentially, each PCA fit on its own input
    for i in range(n_mix):
        model.eval()
        with torch.no_grad():
            h_in = model.upto_mixer(Xp, i)
            phi = polycross(h_in).reshape(-1, feat_dim).cpu().numpy()
        n_comp = min(n_components, phi.shape[1], phi.shape[0])
        p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
        torch.manual_seed(seed + 100 + i)
        model.mixers.append(MixerSublayer(FrozenPCA(p.mean_, p.components_)).to(device))
    for j in range(n_ffn):
        torch.manual_seed(seed + 200 + j)
        model.ffns.append(FFNSublayer().to(device))

    # Ridge warm-start of the FIRST mixer only (paper's mechanism)
    model.eval()
    WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
    with torch.no_grad():
        for i in range(0, WS_N, 4096):
            xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
            phis.append(model.mixers[0].feat(model.embed_x(xb)).mean(1).cpu().numpy())
    phi_pooled = np.concatenate(phis)
    reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
    with torch.no_grad():
        model.mixers[0].W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
        model.mixers[0].W.bias.data.zero_()
        model.head.weight.data.zero_(); model.head.weight.data[0, 0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

    N = X_train.shape[0]; n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    best, best_state, pat = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best:
            best, best_state, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(preds), {
        'n_params': sum(q.numel() for q in model.parameters() if q.requires_grad)}


PCA_VERSION = "v5-pcasweep"


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_pca_sweep(dataset_name: str, n_mix: int, n_ffn: int, ncomp: int, seed_idx: int):
    """Is the 200-component PCA budget binding? Sweep it down."""
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep(X, y, seed_idx)
        pred, info = train_rb_sublayers(Xtr, ytr, Xte, n_mix=n_mix, n_ffn=n_ffn,
                                        n_components=ncomp, seed=seed)
        return {'dataset': dataset_name, 'variant': f'mix{n_mix}_ffn{n_ffn}_pca{ncomp}',
                'n_mix': n_mix, 'n_ffn': n_ffn, 'ncomp': ncomp, 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'params': int(info['n_params']),
                'version': PCA_VERSION, 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'variant': f'mix{n_mix}_ffn{n_ffn}_pca{ncomp}',
                'n_mix': n_mix, 'n_ffn': n_ffn, 'ncomp': ncomp, 'seed_idx': seed_idx,
                'r2': float('nan'), 'params': None, 'version': PCA_VERSION,
                'wall_s': _t.time()-t0, 'error': f"{type(e).__name__}: {e}"}


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_sublayer(dataset_name: str, n_mix: int, n_ffn: int, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep(X, y, seed_idx)
        pred, info = train_rb_sublayers(Xtr, ytr, Xte, n_mix=n_mix, n_ffn=n_ffn, seed=seed)
        return {'dataset': dataset_name, 'n_mix': n_mix, 'n_ffn': n_ffn,
                'variant': f'mix{n_mix}_ffn{n_ffn}', 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'params': int(info['n_params']),
                'wall_s': _t.time() - t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'n_mix': n_mix, 'n_ffn': n_ffn,
                'variant': f'mix{n_mix}_ffn{n_ffn}', 'seed_idx': seed_idx,
                'r2': float('nan'), 'params': None, 'wall_s': _t.time() - t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
