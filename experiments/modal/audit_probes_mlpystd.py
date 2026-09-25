"""
AUDIT PROBE SUITE — empirically verifies every checkable claim in the paper.

  ystd    : FT-T & RB with y standardized (is the Airfoil/Yacht FT-T collapse a
            target-scaling artifact?)                                   [80  T4]
  sens    : App C sensitivity sentence "d_model {32,64,128}, dropout [0,.3],
            lr [1e-4,1e-2] moves R^2 < 2pp on every dataset"            [240 T4]
  cpu     : OLS & RF capped — verify Table 1 baseline columns           [80 CPU]
  mlp     : MLP per paper spec (3x200 ReLU, drop .2, Adam 1e-3, batch 64,
            patience 20, 15% val) — verify Table 1 MLP column           [40  T4]
  attreg  : Eq.(24) Attention Regression reimplemented EXACTLY per the
            appendix spec — verify Table 1 Att.Reg column               [40  T4]
  mc      : full Monte Carlo reimplementation per appendix spec —
            6 DGP x 4 N x 4 SNR x 10 reps x 5 models                    [96  T4]

Deploy: modal deploy audit_probes.py   (app: neurips-31482-audit)
"""
from __future__ import annotations
import pickle
import modal

TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42

FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128; FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1; FTT_EPOCHS = 200; FTT_BS = 256; FTT_LR = 1e-3; FTT_WD = 1e-4
POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4

DS_ORDER = ['California', 'Yacht', 'Energy', 'Concrete',
            'Airfoil',    'Abalone', 'Kin8nm', 'Protein']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-mlpystd")
CODE_VERSION = "v2-mlp-ystd"


def _load_cached():
    import os
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def prep_capped(X_full, y_full, seed_idx, ystd=False):
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
    if ystd:
        scy = StandardScaler()
        ytr = scy.fit_transform(np.asarray(ytr).reshape(-1, 1)).ravel()
        yte = scy.transform(np.asarray(yte).reshape(-1, 1)).ravel()
    return (Xtr.astype('float32'), np.asarray(ytr, dtype='float32'),
            Xte.astype('float32'), np.asarray(yte, dtype='float32'), P, seed)


# ================= FT-Transformer (verbatim Table-1 pipeline) =================
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
    for ep in range(1, FTT_EPOCHS + 1):
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
    import numpy as _np
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            preds.append(model(torch.from_numpy(X_test[i:i+4096]).to(device)).cpu().numpy().ravel())
    return _np.concatenate(preds)


# ================= RB (verbatim Table-1 pipeline, configurable) =================
def train_rb(X_train, y_train, X_test, seed=42, d_model=POLY_D_MODEL,
             ffn_dim=None, dropout=POLY_DROPOUT, lr=POLY_LR,
             n_components=POLY_PCA_COMP, epochs=POLY_EPOCHS):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    if ffn_dim is None: ffn_dim = 2 * d_model
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

    class RB(nn.Module):
        def __init__(self, red):
            super().__init__()
            self.embed = nn.Linear(1, D)
            self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.red = red; self.W = nn.Linear(red.out_dim, D)
            self.n1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(ffn_dim, D), nn.Dropout(dropout))
            self.n2 = nn.LayerNorm(D); self.head = nn.Linear(D, 1)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def feat(self, e): return self.red(poly_cross(e))
        def forward(self, x):
            e = self.embed_x(x)
            z = self.n1(e + self.W(self.feat(e)))
            return self.head(self.n2(z + self.ffn(z)).mean(1))

    torch.manual_seed(seed)
    tmp_embed = nn.Linear(1, D); tmp_pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
    X_pca = X_train if len(X_train) <= 5000 else X_train[
        np.random.RandomState(seed).choice(len(X_train), 5000, replace=False)]
    Xt = torch.from_numpy(X_pca.astype(np.float32))
    with torch.no_grad():
        phi = poly_cross(tmp_embed(Xt.unsqueeze(-1)) + tmp_pos)
        feat_dim = phi.shape[-1]; phi = phi.reshape(-1, feat_dim).numpy()
    n_comp = min(n_components, phi.shape[1], phi.shape[0])
    p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
    torch.manual_seed(seed)
    model = RB(FrozenPCA(p.mean_, p.components_)).to(device)

    model.eval(); WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
    with torch.no_grad():
        for i in range(0, WS_N, 4096):
            xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
            phis.append(model.feat(model.embed_x(xb)).mean(1).cpu().numpy())
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
    loader = DataLoader(TensorDataset(Xt2, yt2), batch_size=POLY_BS, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=POLY_WD)
    crit = nn.MSELoss(); best, bs, pat = float('inf'), None, 0
    for ep in range(1, epochs + 1):
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
    import numpy as _np
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            preds.append(model(torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)).cpu().numpy().ravel())
    return _np.concatenate(preds)


# ================= MLP per paper spec =================
def train_mlp(X_train, y_train, X_test, seed=42, layers=3, units=200, drop=0.2,
              lr=1e-3, batch_size=64, max_epochs=1000, patience=20, val_frac=0.15):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = X_train.shape[1]
    mods, prev = [], P
    for _ in range(layers):
        mods += [nn.Linear(prev, units), nn.ReLU(), nn.Dropout(drop)]; prev = units
    mods += [nn.Linear(prev, 1)]
    model = nn.Sequential(*mods).to(device)
    N = X_train.shape[0]; n_val = max(1, int(N * val_frac)); idx = np.random.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr); crit = nn.MSELoss()
    best, bs, pat = float('inf'), None, 0
    for ep in range(max_epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best: best, bs, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= patience: break
    if bs: model.load_state_dict({k: v.to(device) for k, v in bs.items()})
    model.eval()
    import numpy as _np
    with torch.no_grad():
        return model(torch.from_numpy(X_test.astype(np.float32)).to(device)).cpu().numpy().ravel()


# ================= Attention Regression, Eq.(24), per appendix spec =================
def train_attreg(X_train, y_train, X_test, seed=42, M=5, lam=1e-3,
                 val_frac=0.15, outer_rounds=30, lbfgs_iters=20, es_patience=5):
    """y-hat = sum_m alpha_m softmax(X* Omega_m Xtr') ytr, Omega_m = L_m L_m'
    (lower-tri), penalized loss ||y - yhat||^2 + lam*sum||L_m||_F^2, L-BFGS with
    val early stopping. Init: L_m = chol((X'X+lam I)^{-1}) * sqrt(N) + noise.
    Implementation choices where the text is silent are documented in the audit:
    15%% val split for the early stop; keys = the fit split; strong-Wolfe L-BFGS."""
    import numpy as np, torch
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, P = X_train.shape
    n_val = max(1, int(N * val_frac))
    idx = np.random.permutation(N)
    fit_idx, val_idx = idx[n_val:], idx[:n_val]
    Xf = torch.from_numpy(X_train[fit_idx].astype(np.float32)).to(device)
    yf = torch.from_numpy(y_train[fit_idx].astype(np.float32)).to(device)
    Xv = torch.from_numpy(X_train[val_idx].astype(np.float32)).to(device)
    yv = torch.from_numpy(y_train[val_idx].astype(np.float32)).to(device)
    Xte = torch.from_numpy(X_test.astype(np.float32)).to(device)
    nf = Xf.shape[0]

    S = (Xf.T @ Xf + lam * torch.eye(P, device=device)).double()
    prec = torch.linalg.inv(S)
    C = torch.linalg.cholesky((prec + prec.T) / 2).float() * (nf ** 0.5)
    tril_mask = torch.tril(torch.ones(P, P, device=device))
    g = torch.Generator(device='cpu').manual_seed(seed)
    Ls = []
    for m in range(M):
        noise = torch.randn(P, P, generator=g).to(device) * 0.1 * C.abs().mean()
        Ls.append(torch.nn.Parameter((C + noise) * tril_mask))
    alpha = torch.nn.Parameter(torch.full((M,), 1.0 / M, device=device))
    params = Ls + [alpha]

    def predict(Q):
        out = 0.0
        for m in range(M):
            L = Ls[m] * tril_mask
            Om = L @ L.T
            W = torch.softmax(Q @ Om @ Xf.T, dim=1)
            out = out + alpha[m] * (W @ yf)
        return out

    def loss_fn():
        pred = predict(Xf)
        pen = sum(((Ls[m] * tril_mask) ** 2).sum() for m in range(M))
        return ((yf - pred) ** 2).sum() / nf + lam * pen

    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs_iters,
                            line_search_fn='strong_wolfe')
    best, best_state, pat = float('inf'), None, 0
    for rnd in range(outer_rounds):
        def closure():
            opt.zero_grad(); l = loss_fn(); l.backward(); return l
        try:
            opt.step(closure)
        except Exception:
            break
        with torch.no_grad():
            vl = ((yv - predict(Xv)) ** 2).mean().item()
        if not np.isfinite(vl):
            break
        if vl < best - 1e-9:
            best = vl
            best_state = [t.detach().clone() for t in params]
            pat = 0
        else:
            pat += 1
            if pat >= es_patience:
                break
    if best_state is not None:
        with torch.no_grad():
            for t, s in zip(params, best_state):
                t.copy_(s)
    with torch.no_grad():
        return predict(Xte).cpu().numpy().ravel()


# ================= Modal probe functions =================
@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_ystd(dataset_name: str, model_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx, ystd=True)
        pred = (train_ft_transformer if model_name == 'FTT' else train_rb)(Xtr, ytr, Xte, seed=seed)
        return {'kind': 'ystd', 'dataset': dataset_name, 'model': model_name,
                'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'ystd', 'dataset': dataset_name, 'model': model_name,
                'seed_idx': seed_idx, 'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


SENS = {'d32': dict(d_model=32), 'd128': dict(d_model=128),
        'dr00': dict(dropout=0.0), 'dr03': dict(dropout=0.3),
        'lr1em4': dict(lr=1e-4), 'lr1em2': dict(lr=1e-2)}

@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_sens(dataset_name: str, variant: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        kw = dict(SENS[variant])
        if 'd_model' in kw:
            kw['ffn_dim'] = 2 * kw['d_model']
        else:
            kw['ffn_dim'] = POLY_FFN_DIM
        pred = train_rb(Xtr, ytr, Xte, seed=seed, **kw)
        return {'kind': 'sens', 'dataset': dataset_name, 'variant': variant,
                'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'sens', 'dataset': dataset_name, 'variant': variant,
                'seed_idx': seed_idx, 'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.function(image=image, timeout=1800, volumes={CACHE_DIR: cache_vol},
              cpu=4.0, memory=8192, retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_cpu(dataset_name: str, model_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        if model_name == 'OLS':
            from sklearn.linear_model import LinearRegression
            pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
        elif model_name == 'RF':
            from sklearn.ensemble import RandomForestRegressor
            pred = RandomForestRegressor(n_estimators=500, max_features=1/3,
                                         n_jobs=-1, random_state=seed).fit(Xtr, ytr).predict(Xte)
        else:
            raise ValueError(model_name)
        return {'kind': 'cpu', 'dataset': dataset_name, 'model': model_name,
                'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'cpu', 'dataset': dataset_name, 'model': model_name,
                'seed_idx': seed_idx, 'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_mlp(dataset_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        # CORRECTED PROTOCOL: the original MLP column standardized X only.
        # The MLP is a trained network, so unlike OLS/RF it is not scale-equivariant.
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx, ystd=True)
        pred = train_mlp(Xtr, ytr, Xte, seed=seed)
        return {'kind': 'mlpystd', 'code_version': CODE_VERSION,
                'dataset': dataset_name, 'model': 'MLP',
                'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'mlp', 'dataset': dataset_name, 'model': 'MLP',
                'seed_idx': seed_idx, 'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_attreg(dataset_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        pred = train_attreg(Xtr, ytr, Xte, seed=seed)
        return {'kind': 'attreg', 'dataset': dataset_name, 'model': 'AttReg',
                'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'attreg', 'dataset': dataset_name, 'model': 'AttReg',
                'seed_idx': seed_idx, 'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


# ================= Monte Carlo per appendix spec =================
MC_DGPS = ['linear', 'friedman1', 'friedman2', 'friedman3', 'rotated_sine', 'soft_radial']
MC_NS = [500, 1000, 2500, 5000]
MC_SNRS = [0.5, 1.0, 2.0, 3.0]
MC_J = 1000
MC_REPS = 10

def mc_f(dgp, X):
    import numpy as np
    if dgp == 'linear':
        return (2*(X[:,0]-.5) - (X[:,1]-.5) + 3*(X[:,2]-.5) + 1.5*(X[:,3]-.5) + .5*(X[:,4]-.5))
    if dgp == 'friedman1':
        return 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-.5)**2 + 10*X[:,3] + 5*X[:,4]
    if dgp == 'friedman2':
        return np.sin(np.pi*(X[:,0]+X[:,1]+X[:,2])) + np.log1p(X[:,3]**2)
    if dgp == 'friedman3':
        return X[:,0]*X[:,1] + np.log(X[:,2]+X[:,3]+2)
    if dgp == 'rotated_sine':
        return np.sin(3*X[:,:4].sum(1))
    if dgp == 'soft_radial':
        return 1.0/(1.0 + 5*((X-.5)**2).sum(1))
    raise ValueError(dgp)


@app.function(image=image, gpu="T4", timeout=5400, volumes={CACHE_DIR: cache_vol},
              cpu=4.0, memory=16384,
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_mc_cell(dgp: str, N: int, snr: float):
    """One (DGP, N, SNR) cell: 10 reps x 5 models, per appendix spec."""
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    t0 = _t.time()
    out = {'kind': 'mc', 'dgp': dgp, 'N': N, 'snr': snr, 'error': None}
    try:
        scores = {m: [] for m in ['OLS', 'RF', 'GBM', 'MLP', 'AttReg']}
        for rep in range(MC_REPS):
            rs = np.random.RandomState(SEED + 7919 * rep + hash((dgp, N, int(snr*10))) % 100000)
            Xtr = rs.uniform(0, 1, size=(N, 5))
            Xte = rs.uniform(0, 1, size=(MC_J, 5))
            f_tr, f_te = mc_f(dgp, Xtr), mc_f(dgp, Xte)
            sig = np.sqrt(np.var(f_tr) / snr)
            ytr = (f_tr + rs.normal(0, sig, N)).astype('float32')
            yte = (f_te + rs.normal(0, sig, MC_J)).astype('float32')
            Xtr32, Xte32 = Xtr.astype('float32'), Xte.astype('float32')
            seed = SEED + rep * 1000
            scores['OLS'].append(r2_score(yte, LinearRegression().fit(Xtr32, ytr).predict(Xte32)))
            scores['RF'].append(r2_score(yte, RandomForestRegressor(
                n_estimators=500, max_features=max(1, 5 // 3), n_jobs=-1,
                random_state=seed).fit(Xtr32, ytr).predict(Xte32)))
            scores['GBM'].append(r2_score(yte, GradientBoostingRegressor(
                n_estimators=500, learning_rate=0.01, random_state=seed).fit(Xtr32, ytr).predict(Xte32)))
            scores['MLP'].append(r2_score(yte, train_mlp(Xtr32, ytr, Xte32, seed=seed)))
            scores['AttReg'].append(r2_score(yte, train_attreg(Xtr32, ytr, Xte32, seed=seed)))
        for m, v in scores.items():
            out[f'{m}_mean'] = float(np.mean(v)); out[f'{m}_sd'] = float(np.std(v))
        out['wall_s'] = _t.time() - t0
        return out
    except Exception as e:
        out['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        out['wall_s'] = _t.time() - t0
        return out
