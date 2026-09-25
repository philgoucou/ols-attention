"""
Friedman control: separate DIMENSIONALITY from FEATURE IRRELEVANCE.

The earlier Friedman MC held the number of informative features at 5 while growing P,
so the share of relevant features fell (25% at P=20, 10% at P=50). The observed
RB degradation could therefore be either effect. This isolates them:

  arm 'frac5'  : exactly 5 informative features at any P, so the relevant SHARE
                 falls as P grows (50% at P=10, 25% at P=20, 10% at P=50) [old design]
  arm 'orig50' : the Friedman-1 layout is tiled in blocks of 10 — 5 informative +
                 5 noise per block — so the relevant SHARE stays at 50% for every P,
                 exactly as in the original Friedman-1 (10 features, first 5 used)

  f_orig50(X) = (1/sqrt(k)) * sum_{g=0}^{k-1} [ 10 sin(pi x_{10g} x_{10g+1})
                                              + 20 (x_{10g+2} - 0.5)^2
                                              + 10 x_{10g+3} + 5 x_{10g+4} ],  k = P/10
  (features 10g+5 ... 10g+9 are present but unused — the original's noise half)

Both arms are SNR-calibrated to the SAME ratio (SNR=24, matching the original
make_friedman1 with noise=1.0, whose f has std ~4.9), so R^2 is comparable across
P and across arms: noise std = std(f_train)/sqrt(SNR). The 1/sqrt(k) factor keeps
Var(f) roughly constant in k (belt-and-braces; SNR calibration already handles scale).

At P=10 the two arms are identical by construction (both are the original
Friedman-1: 10 features, 5 relevant) — they are both run as a
consistency check and must agree within seed noise.

Grid: arms {frac5, orig50} x P {10, 20, 50} x models {OLS, RF, FTT, RB} x 5 seeds.

Deploy: modal deploy rebuttal_friedman.py   (app: neurips-31482-friedman)
"""
from __future__ import annotations
import modal

TEST_FRAC = 0.2
MAX_N = 5000
FR_N = 6250          # -> 5000 after the cap, matching the other MC runs
SEED = 42
SNR = 24.0           # matches original make_friedman1(noise=1.0)
RF_TREES = 500; RF_MTRY = 1/3

FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128; FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1; FTT_EPOCHS = 200; FTT_BS = 256; FTT_LR = 1e-3; FTT_WD = 1e-4
POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4

ARMS = ['frac5', 'orig50', 'all100']
PS = [10, 20, 50]
MODELS = ['OLS', 'RF', 'FTT', 'RB']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
app = modal.App("neurips-31482-friedman")


def rb_mem_settings(P):
    if P <= 25: return dict(pca_fit_cap=2000, batch_size=256)
    if P <= 35: return dict(pca_fit_cap=1200, batch_size=128)
    if P <= 45: return dict(pca_fit_cap=700,  batch_size=96)
    if P <= 60: return dict(pca_fit_cap=400,  batch_size=48)
    return dict(pca_fit_cap=100, batch_size=24)


def friedman_block(X, o):
    """One Friedman-1 term on features [o, o+5)."""
    import numpy as np
    return (10 * np.sin(np.pi * X[:, o] * X[:, o+1])
            + 20 * (X[:, o+2] - 0.5) ** 2
            + 10 * X[:, o+3] + 5 * X[:, o+4])


def make_data(arm, P, n, rng):
    """Return (X, f) with f the noiseless signal."""
    import numpy as np
    X = rng.uniform(0, 1, size=(n, P))
    if arm == 'frac5':
        f = friedman_block(X, 0)                       # exactly 5 informative, any P
    elif arm == 'orig50':
        k = P // 10                                    # blocks of 10: 5 signal + 5 noise
        f = sum(friedman_block(X, 10 * g) for g in range(k)) / np.sqrt(k)
    elif arm == 'all100':
        k = P // 5                                     # blocks of 5, ALL relevant
        f = sum(friedman_block(X, 5 * g) for g in range(k)) / np.sqrt(k)
    else:
        raise ValueError(arm)
    return X, f


def build_split(arm, P, seed_idx):
    """SNR-calibrated train/test split, X and y standardized on train."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    seed = SEED + seed_idx * 1000
    rng = np.random.RandomState(seed)
    X, f = make_data(arm, P, FR_N, rng)
    sigma = np.std(f) / np.sqrt(SNR)          # same SNR in both arms, all P
    y = f + rng.normal(0, sigma, size=len(f))
    if len(X) > MAX_N:
        idx = np.random.RandomState(SEED).choice(len(X), MAX_N, replace=False)
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC, random_state=seed)
    sx = StandardScaler(); Xtr = sx.fit_transform(Xtr); Xte = sx.transform(Xte)
    sy = StandardScaler()
    ytr = sy.fit_transform(ytr.reshape(-1, 1)).ravel()
    yte = sy.transform(yte.reshape(-1, 1)).ravel()
    return (Xtr.astype('float32'), ytr.astype('float32'),
            Xte.astype('float32'), yte.astype('float32'), seed)


# ---------------- FT-Transformer ----------------
def train_ft_transformer(X_train, y_train, X_test, seed=42):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = FTT_D_MODEL
    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Linear(1, D) for _ in range(nf)])
            self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            enc = nn.TransformerEncoderLayer(d_model=D, nhead=FTT_N_HEADS,
                                             dim_feedforward=FTT_FFN_DIM, dropout=FTT_DROPOUT,
                                             batch_first=True, activation='gelu')
            self.tr = nn.TransformerEncoder(enc, num_layers=FTT_N_BLOCKS)
            self.norm = nn.LayerNorm(D); self.head = nn.Linear(D, 1)
        def forward(self, x):
            B = x.size(0)
            tok = torch.stack([self.emb[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            tok = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)
            return self.head(self.norm(self.tr(tok)[:, 0, :]))
    model = FT().to(device)
    N = X_train.shape[0]; n_val = max(1, int(N * 0.15)); idx = np.random.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]]); yt = torch.from_numpy(y_train[idx[n_val:]]).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]]).to(device); yv = torch.from_numpy(y_train[idx[:n_val]]).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=FTT_BS, shuffle=True)
    crit = nn.MSELoss(); opt = torch.optim.AdamW(model.parameters(), lr=FTT_LR, weight_decay=FTT_WD)
    warm = max(1, FTT_EPOCHS // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: ep/warm if ep < warm else 0.5*(1+np.cos(np.pi*(ep-warm)/max(1, FTT_EPOCHS-warm))))
    best, bs, pc = float('inf'), None, 0
    for ep in range(1, FTT_EPOCHS+1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step(); model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best: best, bs, pc = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= 10: break
    if bs: model.load_state_dict({k: v.to(device) for k, v in bs.items()})
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            preds.append(model(torch.from_numpy(X_test[i:i+4096]).to(device)).cpu().numpy().ravel())
    return np.concatenate(preds)


# ---------------- Regression Block ----------------
def train_rb(X_train, y_train, X_test, pca_fit_cap, batch_size, seed=42):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    nf = X_train.shape[1]; D = POLY_D_MODEL

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

    class Net(nn.Module):
        def __init__(self, red):
            super().__init__()
            self.embed = nn.Linear(1, D); self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.red = red; self.W = nn.Linear(red.out_dim, D); self.n1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, POLY_FFN_DIM), nn.ReLU(), nn.Dropout(POLY_DROPOUT),
                                     nn.Linear(POLY_FFN_DIM, D), nn.Dropout(POLY_DROPOUT))
            self.n2 = nn.LayerNorm(D); self.head = nn.Linear(D, 1)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def feat(self, e): return self.red(polycross(e))
        def forward(self, x):
            e = self.embed_x(x)
            z = self.n1(e + self.W(self.feat(e)))
            return self.head(self.n2(z + self.ffn(z)).mean(1))

    X_pca = X_train if len(X_train) <= pca_fit_cap else X_train[
        np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)]
    torch.manual_seed(seed)
    tmp_embed = nn.Linear(1, D); tmp_pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
    Xt = torch.from_numpy(X_pca.astype(np.float32))
    with torch.no_grad():
        phi = polycross(tmp_embed(Xt.unsqueeze(-1)) + tmp_pos)
        feat_dim = phi.shape[-1]; phi = phi.reshape(-1, feat_dim).numpy()
    n_comp = min(POLY_PCA_COMP, phi.shape[1], phi.shape[0])
    p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
    torch.manual_seed(seed)
    model = Net(FrozenPCA(p.mean_, p.components_)).to(device)

    model.eval(); WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
    warm_chunk = max(8, min(batch_size, 256))
    with torch.no_grad():
        for i in range(0, WS_N, warm_chunk):
            xb = torch.from_numpy(X_ws[i:i+warm_chunk].astype(np.float32)).to(device)
            phis.append(model.feat(model.embed_x(xb)).mean(1).cpu().numpy())
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    phi_pooled = np.concatenate(phis)
    reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
    with torch.no_grad():
        model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
        model.W.bias.data.zero_()
        model.head.weight.data.zero_(); model.head.weight.data[0, 0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

    N = X_train.shape[0]; n_val = max(1, int(N * 0.15)); idx = np.random.permutation(N)
    Xt2 = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt2 = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt2, yt2), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=POLY_LR, weight_decay=POLY_WD); crit = nn.MSELoss()
    best, bs, pat = float('inf'), None, 0
    for ep in range(1, POLY_EPOCHS+1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best: best, bs, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if bs: model.load_state_dict({k: v.to(device) for k, v in bs.items()})
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            preds.append(model(torch.from_numpy(X_test[i:i+batch_size].astype(np.float32)).to(device)).cpu().numpy().ravel())
    return np.concatenate(preds)


# ---------------- Modal functions ----------------
@app.function(image=image, timeout=1200, cpu=4.0, memory=8192,
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_sk(arm: str, P: int, model_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        Xtr, ytr, Xte, yte, seed = build_split(arm, P, seed_idx)
        if model_name == 'OLS':
            from sklearn.linear_model import LinearRegression
            pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
        else:
            from sklearn.ensemble import RandomForestRegressor
            pred = RandomForestRegressor(n_estimators=RF_TREES, max_features=RF_MTRY,
                                         n_jobs=-1, random_state=seed).fit(Xtr, ytr).predict(Xte)
        return {'arm': arm, 'P': P, 'model': model_name, 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'arm': arm, 'P': P, 'model': model_name, 'seed_idx': seed_idx,
                'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.function(image=image, gpu="A10G", timeout=3600, memory=32768,
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_nn(arm: str, P: int, model_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        Xtr, ytr, Xte, yte, seed = build_split(arm, P, seed_idx)
        if model_name == 'FTT':
            pred = train_ft_transformer(Xtr, ytr, Xte, seed=seed)
        else:
            ms = rb_mem_settings(P)
            pred = train_rb(Xtr, ytr, Xte, pca_fit_cap=ms['pca_fit_cap'],
                            batch_size=ms['batch_size'], seed=seed)
        return {'arm': arm, 'P': P, 'model': model_name, 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'arm': arm, 'P': P, 'model': model_name, 'seed_idx': seed_idx,
                'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
