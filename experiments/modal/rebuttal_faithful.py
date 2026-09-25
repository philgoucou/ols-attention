"""
F1 + F3 fix: make the Regression Block actually match Appendix C, then test it.

Appendix C describes the RB as the FT-Transformer skeleton with ONLY the attention
sublayer swapped: per-feature tokenizer, learned [CLS] token, GELU FFN, L=3 blocks.
The shipped code differs on all four (shared tokenizer, mean-pool, ReLU, L=1).
This runs a ladder from the shipped block to the fully faithful one so we learn
which fidelity fixes are free, which cost, and — the open question — whether depth
actually helps once the block is built the way the paper says.

Variants (all under the y-standardized protocol going forward):
  cur_L1     shared tok  / mean-pool / ReLU / L=1   <- what Table 1 actually ran
  gelu_L1    shared tok  / mean-pool / GELU / L=1   <- isolate activation
  cls_L1     shared tok  / CLS       / ReLU / L=1   <- isolate readout
  pertok_L1  per-feature / mean-pool / ReLU / L=1   <- isolate tokenizer
  faith_L1   per-feature / CLS       / GELU / L=1   <- all fidelity fixes, 1 block
  faith_L2   per-feature / CLS       / GELU / L=2
  faith_L3   per-feature / CLS       / GELU / L=3   <- the architecture App C claims
  cur_L3     shared tok  / mean-pool / ReLU / L=3   <- depth on the shipped block

FT-T (L=3, y-standardized) comparator already exists in audit_ystd_results.csv.

Deploy: modal deploy rebuttal_faithful.py   (app: neurips-31482-faithful)
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

# variant -> (tokenizer, readout, activation, n_layers)
VARIANTS = {
    'cur_L1':    ('shared',      'mean', 'relu', 1),
    'gelu_L1':   ('shared',      'mean', 'gelu', 1),
    'cls_L1':    ('shared',      'cls',  'relu', 1),
    'pertok_L1': ('per_feature', 'mean', 'relu', 1),
    'faith_L1':  ('per_feature', 'cls',  'gelu', 1),
    'faith_L2':  ('per_feature', 'cls',  'gelu', 2),
    'faith_L3':  ('per_feature', 'cls',  'gelu', 3),
    'cur_L3':    ('shared',      'mean', 'relu', 3),
}

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-faithful")


def _load_cached():
    import os
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def prep(X_full, y_full, seed_idx, ystd=True):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    N_full, P = X_full.shape
    if N_full > MAX_N:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(N_full, MAX_N, replace=False)
        X_full, y_full = X_full[idx], y_full[idx]
    seed = SEED + seed_idx * 1000
    Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=TEST_FRAC, random_state=seed)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    ytr = np.asarray(ytr, dtype='float64'); yte = np.asarray(yte, dtype='float64')
    if ystd:
        scy = StandardScaler()
        ytr = scy.fit_transform(ytr.reshape(-1, 1)).ravel()
        yte = scy.transform(yte.reshape(-1, 1)).ravel()
    return (Xtr.astype('float32'), ytr.astype('float32'),
            Xte.astype('float32'), yte.astype('float32'), P, seed)


def train_rb_flexible(X_train, y_train, X_test, tokenizer='shared', readout='mean',
                      activation='relu', n_layers=1, d_model=POLY_D_MODEL,
                      ffn_dim=POLY_FFN_DIM, n_components=POLY_PCA_COMP,
                      epochs=POLY_EPOCHS, batch_size=POLY_BS, lr=POLY_LR,
                      dropout=POLY_DROPOUT, weight_decay=POLY_WD, seed=42,
                      pca_fit_cap=5000):
    """Regression Block with configurable FT-Transformer-skeleton fidelity.

    tokenizer  'shared'      : one Linear(1,d) reused for every feature (shipped code)
               'per_feature' : one Linear(1,d) per feature (FT-Transformer / App C)
    readout    'mean'        : mean over tokens (shipped code)
               'cls'         : learned [CLS] token prepended, read at position 0 (App C)
    activation 'relu' (shipped) | 'gelu' (App C)
    n_layers   number of stacked regression blocks; each block's frozen PCA is fit on
               its own input, obtained by forward-passing through the blocks below it.
    Warm-start: block 0's mixer row 0 <- Ridge fit of that block's pooled features on y,
    head reads channel 0 (identical mechanism to the shipped code; pooling matches the
    readout so the warm-start prediction is what the head actually sees).
    """
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model
    use_cls = (readout == 'cls')
    Act = nn.GELU if activation == 'gelu' else nn.ReLU

    def poly_cross(x):
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

    class Block(nn.Module):
        def __init__(self, red):
            super().__init__()
            self.red = red
            self.W = nn.Linear(red.out_dim, D)
            self.n1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, ffn_dim), Act(), nn.Dropout(dropout),
                                     nn.Linear(ffn_dim, D), nn.Dropout(dropout))
            self.n2 = nn.LayerNorm(D)
        def feat(self, h): return self.red(poly_cross(h))
        def forward(self, h):
            z = self.n1(h + self.W(self.feat(h)))
            return self.n2(z + self.ffn(z))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if tokenizer == 'per_feature':
                self.embeds = nn.ModuleList([nn.Linear(1, D) for _ in range(nf)])
            else:
                self.embed = nn.Linear(1, D)
            self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            if use_cls:
                self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            self.blocks = nn.ModuleList()
            self.norm = nn.LayerNorm(D)
            self.head = nn.Linear(D, 1)
        def embed_x(self, x):
            if tokenizer == 'per_feature':
                tok = torch.stack([self.embeds[i](x[:, i:i+1]) for i in range(nf)], dim=1)
            else:
                tok = self.embed(x.unsqueeze(-1))
            tok = tok + self.pos
            if use_cls:
                tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
            return tok
        def pool(self, h):
            return h[:, 0, :] if use_cls else h.mean(1)
        def forward_upto(self, x, upto):
            h = self.embed_x(x)
            for l in range(upto):
                h = self.blocks[l](h)
            return h
        def forward(self, x):
            h = self.embed_x(x)
            for blk in self.blocks:
                h = blk(h)
            return self.head(self.norm(self.pool(h)))

    # PCA-fit subsample
    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    X_pca_t = torch.from_numpy(X_pca.astype(np.float32)).to(device)

    torch.manual_seed(seed)
    model = Net().to(device)
    with torch.no_grad():
        feat_dim = poly_cross(model.embed_x(X_pca_t[:8])).shape[-1]

    # build blocks sequentially, each PCA fit on its own input representation
    for l in range(n_layers):
        model.eval()
        with torch.no_grad():
            h_in = model.forward_upto(X_pca_t, upto=l)
            phi = poly_cross(h_in).reshape(-1, feat_dim).cpu().numpy()
        n_comp = min(n_components, phi.shape[1], phi.shape[0])
        p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
        torch.manual_seed(seed + 100 + l)
        model.blocks.append(Block(FrozenPCA(p.mean_, p.components_)).to(device))

    # Ridge warm-start of block 0 (pooling matches the readout)
    model.eval()
    WS_N = min(len(X_train), 20000)
    X_ws = X_train[:WS_N]
    phis = []
    with torch.no_grad():
        for i in range(0, WS_N, 4096):
            xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
            f = model.blocks[0].feat(model.embed_x(xb))
            phis.append((f[:, 0, :] if use_cls else f.mean(1)).cpu().numpy())
    phi_pooled = np.concatenate(phis)
    reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
    with torch.no_grad():
        model.blocks[0].W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
        model.blocks[0].W.bias.data.zero_()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        if vl < best:
            best, best_state, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10:
                break
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(preds), {
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_faithful(dataset_name: str, variant: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        tok, ro, act, L = VARIANTS[variant]
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep(X, y, seed_idx, ystd=True)
        pred, info = train_rb_flexible(Xtr, ytr, Xte, tokenizer=tok, readout=ro,
                                       activation=act, n_layers=L, seed=seed)
        return {'kind': 'faithful', 'dataset': dataset_name, 'variant': variant,
                'tokenizer': tok, 'readout': ro, 'activation': act, 'n_layers': L,
                'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'params': int(info['n_params']), 'wall_s': _t.time() - t0, 'error': None}
    except Exception as e:
        return {'kind': 'faithful', 'dataset': dataset_name, 'variant': variant,
                'tokenizer': None, 'readout': None, 'activation': None, 'n_layers': None,
                'seed_idx': seed_idx, 'r2': float('nan'), 'params': None,
                'wall_s': _t.time() - t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
