"""
Residual / LayerNorm ablation for BOTH the Regression Block and FT-Transformer.

Both models wrap their mixer and FFN in the same post-norm scaffolding:
    z = LN(x + sublayer(x))
This removes that scaffolding, one piece at a time, for each model:

    full     : LN(x + f(x))     <- as shipped
    no_res   : LN(f(x))         <- drop the skip connection
    no_ln    : x + f(x)         <- drop the normalization
    neither  : f(x)             <- bare sublayer

Prediction worth testing: FT-T is 3 stacked blocks and should depend heavily on
residuals; RB is a single block and should be far more robust to their removal.
If so, that is a depth-related property, not an attention-vs-regression property.

FT-Transformer is reimplemented here because nn.TransformerEncoderLayer hardcodes
residual+norm. The `full` variant is built to match nn.TransformerEncoderLayer with
norm_first=False (post-norm, GELU, dropout on both sublayers) and is validated
against the standard implementation before the ablated variants are trusted.

Protocol: y-standardized (the corrected protocol), capped N=5000, 8 datasets, 5 seeds.
Grid: 2 models x 4 variants x 8 datasets x 5 seeds = 320 jobs.

Deploy: modal deploy rebuttal_resln.py   (app: neurips-31482-resln)
"""
from __future__ import annotations
import pickle
import modal

TEST_FRAC = 0.2; MAX_N = 5000; SEED = 42
FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128; FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1; FTT_EPOCHS = 200; FTT_BS = 256; FTT_LR = 1e-3; FTT_WD = 1e-4
POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4

DS_ORDER = ['California','Yacht','Energy','Concrete','Airfoil','Abalone','Kin8nm','Protein']
VARIANTS = ['full', 'no_res', 'no_ln', 'neither']   # (use_res, use_ln)
VFLAGS = {'full': (True, True), 'no_res': (False, True),
          'no_ln': (True, False), 'neither': (False, False)}

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4","pandas==2.2.3","scikit-learn==1.5.2","torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-resln")


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


def _wrap(x, fx, use_res, ln):
    """Post-norm sublayer wrapper. ln is None when normalization is ablated."""
    y = x + fx if use_res else fx
    return ln(y) if ln is not None else y


# ---------------- FT-Transformer with ablatable scaffolding ----------------
def train_ftt_resln(X_train, y_train, X_test, use_res=True, use_ln=True, seed=42):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = FTT_D_MODEL

    class Block(nn.Module):
        """Matches nn.TransformerEncoderLayer(norm_first=False) when use_res/use_ln."""
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(D, FTT_N_HEADS, dropout=FTT_DROPOUT, batch_first=True)
            self.do1 = nn.Dropout(FTT_DROPOUT)
            self.lin1 = nn.Linear(D, FTT_FFN_DIM); self.lin2 = nn.Linear(FTT_FFN_DIM, D)
            self.do_ff = nn.Dropout(FTT_DROPOUT); self.do2 = nn.Dropout(FTT_DROPOUT)
            self.n1 = nn.LayerNorm(D) if use_ln else None
            self.n2 = nn.LayerNorm(D) if use_ln else None
            self.act = nn.GELU()
        def forward(self, x):
            a, _ = self.attn(x, x, x, need_weights=False)
            x = _wrap(x, self.do1(a), use_res, self.n1)
            f = self.do2(self.lin2(self.do_ff(self.act(self.lin1(x)))))
            return _wrap(x, f, use_res, self.n2)

    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Linear(1, D) for _ in range(nf)])
            self.cls = nn.Parameter(torch.randn(1,1,D) * 0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(FTT_N_BLOCKS)])
            self.norm = nn.LayerNorm(D) if use_ln else None
            self.head = nn.Linear(D, 1)
        def forward(self, x):
            B = x.size(0)
            tok = torch.stack([self.emb[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            h = torch.cat([self.cls.expand(B,-1,-1), tok], dim=1)
            for b in self.blocks: h = b(h)
            z = h[:, 0, :]
            if self.norm is not None: z = self.norm(z)
            return self.head(z)

    model = FT().to(device)
    return _fit_predict(model, X_train, y_train, X_test, device, FTT_EPOCHS, FTT_BS,
                        FTT_LR, FTT_WD, adamw=True, cosine=True, seed=seed)


# ---------------- Regression Block with ablatable scaffolding ----------------
def train_rb_resln(X_train, y_train, X_test, use_res=True, use_ln=True, seed=42,
                   pca_fit_cap=5000, lr=POLY_LR, whiten=False, clip=1.0,
                   rezero=False, w_init_scale=1.0):
    import numpy as np, torch, torch.nn as nn
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = POLY_D_MODEL

    def polycross(x):
        B, S, Dm = x.shape
        per = torch.cat([x, x**2 - 1], dim=-1)
        pairs = torch.cat([x[:, i, :] * x[:, j, :] for i in range(S) for j in range(i, S)], dim=-1)
        return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)

    class FrozenPCA(nn.Module):
        """sklearn applies whitening inside transform() by dividing by
        sqrt(explained_variance_); it is NOT folded into components_. Since this
        module reimplements the transform, the scaling must be applied here or
        whiten=True is silently a no-op."""
        def __init__(self, mean, comp, evar=None):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer('comp', torch.from_numpy(comp.astype(np.float32)))
            self.do_whiten = evar is not None
            if self.do_whiten:
                sc = np.sqrt(np.maximum(evar.astype(np.float32), 1e-12))
                self.register_buffer('scale', torch.from_numpy(sc))
            self.out_dim = comp.shape[0]
        def forward(self, x):
            z = (x - self.mean) @ self.comp.T
            return z / self.scale if self.do_whiten else z

    class Net(nn.Module):
        def __init__(self, red):
            super().__init__()
            self.embed = nn.Linear(1, D); self.pos = nn.Parameter(torch.randn(1, nf, D)*0.02)
            self.red = red; self.W = nn.Linear(red.out_dim, D)
            self.n1 = nn.LayerNorm(D) if use_ln else None
            self.ffn = nn.Sequential(nn.Linear(D, POLY_FFN_DIM), nn.ReLU(), nn.Dropout(POLY_DROPOUT),
                                     nn.Linear(POLY_FFN_DIM, D), nn.Dropout(POLY_DROPOUT))
            self.n2 = nn.LayerNorm(D) if use_ln else None
            self.head = nn.Linear(D, 1)
            if rezero:   # ReZero: start as identity, learn how much sublayer to admit
                self.alpha1 = nn.Parameter(torch.zeros(1))
                self.alpha2 = nn.Parameter(torch.zeros(1))
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def feat(self, e): return self.red(polycross(e))
        def forward(self, x):
            e = self.embed_x(x)
            a1 = self.W(self.feat(e))
            if rezero: a1 = self.alpha1 * a1
            z = _wrap(e, a1, use_res, self.n1)
            a2 = self.ffn(z)
            if rezero: a2 = self.alpha2 * a2
            h = _wrap(z, a2, use_res, self.n2)
            return self.head(h.mean(1))

    X_pca = X_train if len(X_train) <= pca_fit_cap else X_train[
        np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)]
    torch.manual_seed(seed)
    tmp_e = nn.Linear(1, D); tmp_p = nn.Parameter(torch.randn(1, nf, D)*0.02)
    Xt = torch.from_numpy(X_pca.astype(np.float32))
    with torch.no_grad():
        phi = polycross(tmp_e(Xt.unsqueeze(-1)) + tmp_p)
        fd = phi.shape[-1]; phi = phi.reshape(-1, fd).numpy()
    nc = min(POLY_PCA_COMP, phi.shape[1], phi.shape[0])
    p = PCA(n_components=nc, random_state=seed, whiten=whiten); p.fit(phi)
    torch.manual_seed(seed)
    model = Net(FrozenPCA(p.mean_, p.components_,
                         p.explained_variance_ if whiten else None)).to(device)
    if w_init_scale != 1.0:
        with torch.no_grad():
            model.W.weight.data.mul_(w_init_scale); model.W.bias.data.mul_(w_init_scale)

    # Ridge warm-start, unchanged
    model.eval(); WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
    with torch.no_grad():
        for i in range(0, WS_N, 4096):
            xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
            phis.append(model.feat(model.embed_x(xb)).mean(1).cpu().numpy())
    reg = Ridge(alpha=1.0); reg.fit(np.concatenate(phis), y_train[:WS_N])
    with torch.no_grad():
        model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
        model.W.bias.data.zero_()
        model.head.weight.data.zero_(); model.head.weight.data[0,0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

    return _fit_predict(model, X_train, y_train, X_test, device, POLY_EPOCHS, POLY_BS,
                        lr, POLY_WD, adamw=False, cosine=False, seed=seed, clip=clip)


def _fit_predict(model, X_train, y_train, X_test, device, epochs, bs, lr, wd,
                 adamw, cosine, seed, clip=1.0):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    N = X_train.shape[0]; n_val = max(1, int(N*0.15)); idx = np.random.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)
    opt = (torch.optim.AdamW if adamw else torch.optim.Adam)(model.parameters(), lr=lr, weight_decay=wd)
    sched = None
    if cosine:
        warm = max(1, epochs//10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda ep: ep/warm if ep < warm else 0.5*(1+np.cos(np.pi*(ep-warm)/max(1, epochs-warm))))
    crit = nn.MSELoss(); best, bstate, pat = float('inf'), None, 0
    diverged = False
    for ep in range(1, epochs+1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
        if sched: sched.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if not np.isfinite(vl):
            diverged = True; break
        if vl < best: best, bstate, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if bstate: model.load_state_dict({k: v.to(device) for k, v in bstate.items()})
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            preds.append(model(torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)).cpu().numpy().ravel())
    return np.concatenate(preds), {'n_params': sum(q.numel() for q in model.parameters() if q.requires_grad),
                                   'diverged': bool(diverged)}


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_resln(dataset_name: str, model_name: str, variant: str, seed_idx: int):
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        use_res, use_ln = VFLAGS[variant]
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep(X, y, seed_idx)
        fn = train_ftt_resln if model_name == 'FTT' else train_rb_resln
        pred, info = fn(Xtr, ytr, Xte, use_res=use_res, use_ln=use_ln, seed=seed)
        r2 = float(r2_score(yte, pred)) if np.isfinite(pred).all() else float('nan')
        return {'dataset': dataset_name, 'model': model_name, 'variant': variant,
                'use_res': use_res, 'use_ln': use_ln, 'seed_idx': seed_idx, 'r2': r2,
                'params': int(info['n_params']), 'diverged': info['diverged'],
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'model': model_name, 'variant': variant,
                'use_res': None, 'use_ln': None, 'seed_idx': seed_idx, 'r2': float('nan'),
                'params': None, 'diverged': None, 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


# ---- Is the no-LayerNorm failure an optimization artifact rather than a
# ---- structural dependence? Same architecture (no LN), different scaling/optimizer.
CODE_VERSION = "v3-rezero"

FIXES = {
    'ln_reference': dict(use_ln=True,  lr=POLY_LR, whiten=False, clip=1.0, rezero=False, w_init_scale=1.0),
    'noln_base':    dict(use_ln=False, lr=POLY_LR, whiten=False, clip=1.0, rezero=False, w_init_scale=1.0),
    'noln_rezero':  dict(use_ln=False, lr=POLY_LR, whiten=False, clip=1.0, rezero=True,  w_init_scale=1.0),
    'noln_smallw':  dict(use_ln=False, lr=POLY_LR, whiten=False, clip=1.0, rezero=False, w_init_scale=0.1),
}


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_lnfix(dataset_name: str, variant: str, seed_idx: int):
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        cfg = FIXES[variant]
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep(X, y, seed_idx)
        pred, info = train_rb_resln(Xtr, ytr, Xte, use_res=True, use_ln=cfg['use_ln'],
                                    seed=seed, lr=cfg['lr'], whiten=cfg['whiten'],
                                    clip=cfg['clip'], rezero=cfg['rezero'],
                                    w_init_scale=cfg['w_init_scale'])
        r2 = float(r2_score(yte, pred)) if np.isfinite(pred).all() else float('nan')
        return {'dataset': dataset_name, 'variant': variant, 'seed_idx': seed_idx,
                'r2': r2, 'code_version': CODE_VERSION, 'rezero': cfg['rezero'],
                'w_init_scale': cfg['w_init_scale'],
                'diverged': info['diverged'], 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'variant': variant, 'seed_idx': seed_idx,
                'r2': float('nan'), 'code_version': CODE_VERSION, 'rezero': None,
                'w_init_scale': None,
                'diverged': None, 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
