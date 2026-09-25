"""
Sublayer-composition study: mixer / attention / FFN in every combination.

One harness, one tokenizer, one readout (mean-pool), one training loop — so the
only thing that varies is WHICH sublayers are stacked. Each sublayer is wrapped in
the same post-norm scaffolding LN(x + f(x)).

    mixer : LN(h + W(PCA(polycross(h))))   regression-as-attention (the paper's block)
    attn  : LN(h + MultiHeadAttention(h))  softmax attention
    ffn   : LN(h + FFN(h))                 position-wise, mixes nothing

Configs:
    mix_ffn        [mixer, ffn]         = the shipped Regression Block
    mix_attn_ffn   [mixer, attn, ffn]   = RB + attention + MLP  (does attention ADD anything?)
    attn_ffn       [attn, ffn]          = a one-block FT-Transformer
    ffn_only       [ffn]                = NO mixing at all (FT-T with attention removed)
    mix_attn       [mixer, attn]        = both mixers, no MLP

`ffn_only` is the "FT-Transformer without attention" case. Note it is only
meaningful with mean-pooling: with FT-T's native [CLS] readout, attention is the
sole path from feature tokens to CLS, so removing it makes the output a constant
and R^2 ~ 0 by construction rather than by learning. The cls_degenerate runner
verifies that separately.

Protocol: y-standardized, capped N=5000, 8 datasets, 5 seeds.

Deploy: modal deploy rebuttal_mixattn.py   (app: neurips-31482-mixattn)
"""
from __future__ import annotations
import pickle
import modal

TEST_FRAC = 0.2; MAX_N = 5000; SEED = 42
POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4
N_HEADS = 4

DS_ORDER = ['California','Yacht','Energy','Concrete','Airfoil','Abalone','Kin8nm','Protein']
CONFIGS = {
    'mix_ffn':      ['mixer', 'ffn'],
    'mix_attn_ffn': ['mixer', 'attn', 'ffn'],
    'attn_ffn':     ['attn', 'ffn'],
    'ffn_only':     ['ffn'],
    'mix_attn':     ['mixer', 'attn'],
}
CODE_VERSION = "v6-mixattn"

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4","pandas==2.2.3","scikit-learn==1.5.2","torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-mixattn")


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
    sy = StandardScaler()
    ytr = sy.fit_transform(np.asarray(ytr, dtype='float64').reshape(-1,1)).ravel()
    yte = sy.transform(np.asarray(yte, dtype='float64').reshape(-1,1)).ravel()
    return (Xtr.astype('float32'), ytr.astype('float32'),
            Xte.astype('float32'), yte.astype('float32'), P, seed)


def train_stack(X_train, y_train, X_test, layers, seed=42, readout='mean',
                n_components=POLY_PCA_COMP, pca_fit_cap=5000):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = POLY_D_MODEL
    use_cls = (readout == 'cls')

    def polycross(x):
        B, S, Dm = x.shape
        per = torch.cat([x, x**2 - 1], dim=-1)
        pairs = torch.cat([x[:, i, :] * x[:, j, :] for i in range(S) for j in range(i, S)], dim=-1)
        return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)

    class FrozenPCA(nn.Module):
        def __init__(self, mean, comp):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer('comp', torch.from_numpy(comp.astype(np.float32)))
            self.out_dim = comp.shape[0]
        def forward(self, x): return (x - self.mean) @ self.comp.T

    class Mixer(nn.Module):
        def __init__(self, red):
            super().__init__()
            self.red = red; self.W = nn.Linear(red.out_dim, D); self.n = nn.LayerNorm(D)
        def feat(self, h): return self.red(polycross(h))
        def forward(self, h): return self.n(h + self.W(self.feat(h)))

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.MultiheadAttention(D, N_HEADS, dropout=POLY_DROPOUT, batch_first=True)
            self.n = nn.LayerNorm(D)
        def forward(self, h):
            o, _ = self.a(h, h, h, need_weights=False)
            return self.n(h + o)

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(D, POLY_FFN_DIM), nn.ReLU(), nn.Dropout(POLY_DROPOUT),
                                   nn.Linear(POLY_FFN_DIM, D), nn.Dropout(POLY_DROPOUT))
            self.n = nn.LayerNorm(D)
        def forward(self, h): return self.n(h + self.f(h))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, D)
            self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            if use_cls: self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            self.layers = nn.ModuleList()
            self.head = nn.Linear(D, 1)
        def embed_x(self, x):
            tok = self.embed(x.unsqueeze(-1)) + self.pos
            if use_cls: tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
            return tok
        def upto(self, x, k):
            h = self.embed_x(x)
            for i in range(k): h = self.layers[i](h)
            return h
        def forward(self, x):
            h = self.embed_x(x)
            for L in self.layers: h = L(h)
            return self.head(h[:, 0, :] if use_cls else h.mean(1))

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

    first_mixer = None
    for i, kind in enumerate(layers):
        torch.manual_seed(seed + 100 + i)
        if kind == 'mixer':
            model.eval()
            with torch.no_grad():
                h_in = model.upto(Xp, i)
                phi = polycross(h_in).reshape(-1, feat_dim).cpu().numpy()
            nc = min(n_components, phi.shape[1], phi.shape[0])
            p = PCA(n_components=nc, random_state=seed); p.fit(phi)
            mod = Mixer(FrozenPCA(p.mean_, p.components_)).to(device)
            if first_mixer is None: first_mixer = len(model.layers)
        elif kind == 'attn':
            mod = Attn().to(device)
        else:
            mod = FFN().to(device)
        model.layers.append(mod)

    # Ridge warm-start on the FIRST mixer, when one exists (the paper's mechanism)
    if first_mixer is not None:
        model.eval(); WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
        with torch.no_grad():
            for i in range(0, WS_N, 4096):
                xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
                h = model.upto(xb, first_mixer)
                f = model.layers[first_mixer].feat(h)
                phis.append((f[:, 0, :] if use_cls else f.mean(1)).cpu().numpy())
        reg = Ridge(alpha=1.0); reg.fit(np.concatenate(phis), y_train[:WS_N])
        with torch.no_grad():
            model.layers[first_mixer].W.weight.data[0] = torch.from_numpy(
                reg.coef_.astype(np.float32)).to(device)
            model.layers[first_mixer].W.bias.data.zero_()
            model.head.weight.data.zero_(); model.head.weight.data[0, 0] = 1.0
            model.head.bias.data.fill_(float(reg.intercept_))

    N = X_train.shape[0]; n_val = max(1, int(N*0.15)); idx = np.random.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=POLY_BS, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=POLY_LR, weight_decay=POLY_WD)
    crit = nn.MSELoss(); best, bstate, pat = float('inf'), None, 0
    for ep in range(1, POLY_EPOCHS+1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if not np.isfinite(vl): break
        if vl < best: best, bstate, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if bstate: model.load_state_dict({k: v.to(device) for k, v in bstate.items()})
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            preds.append(model(torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)).cpu().numpy().ravel())
    pr = np.concatenate(preds)
    return pr, {'n_params': sum(q.numel() for q in model.parameters() if q.requires_grad),
                'pred_std': float(np.std(pr))}


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_stack(dataset_name: str, config: str, seed_idx: int, readout: str = 'mean'):
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep(X, y, seed_idx)
        pred, info = train_stack(Xtr, ytr, Xte, CONFIGS[config], seed=seed, readout=readout)
        r2 = float(r2_score(yte, pred)) if np.isfinite(pred).all() else float('nan')
        return {'dataset': dataset_name, 'config': config, 'readout': readout,
                'seed_idx': seed_idx, 'r2': r2, 'params': int(info['n_params']),
                'pred_std': info['pred_std'], 'version': CODE_VERSION,
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'config': config, 'readout': readout,
                'seed_idx': seed_idx, 'r2': float('nan'), 'params': None,
                'pred_std': None, 'version': CODE_VERSION, 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
