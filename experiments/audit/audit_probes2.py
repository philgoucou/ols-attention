"""
AUDIT PROBE ROUND 2 — the charitable Att.Reg variant.

Round-1 verdict: the Eq.(24) estimator AS DESCRIBED (raw targets; raw X in the MC)
scores far below the published Att.Reg numbers, while OLS/RF/GBM/MLP all reproduce
to ±0.009 — so the reconstruction of the protocol is right and the description of
the estimator is what fails. Hypothesis: the lost pipeline standardized inputs AND
targets for Att.Reg (and the MLP — cf. the Airfoil MLP clue). This probe reruns
Att.Reg with full standardization (fit on train; R² computed on the standardized
scale, which is affine-invariant), everything else identical to round 1.

Deploy: modal deploy audit_probes2.py   (app: neurips-31482-audit2)
"""
from __future__ import annotations
import pickle
import modal

TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-audit2")


def _load_cached():
    import os
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def train_attreg(X_train, y_train, X_test, seed=42, M=5, lam=1e-3,
                 val_frac=0.15, outer_rounds=30, lbfgs_iters=20, es_patience=5):
    """Identical to round-1 implementation (Eq. 24 per spec)."""
    import numpy as np, torch
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, P = X_train.shape
    n_val = max(1, int(N * val_frac))
    idx = np.random.permutation(N)
    fit_idx, val_idx = idx[n_val:], idx[:n_val]
    Xf = torch.from_numpy(X_train[fit_idx].astype('float32')).to(device)
    yf = torch.from_numpy(y_train[fit_idx].astype('float32')).to(device)
    Xv = torch.from_numpy(X_train[val_idx].astype('float32')).to(device)
    yv = torch.from_numpy(y_train[val_idx].astype('float32')).to(device)
    Xte = torch.from_numpy(X_test.astype('float32')).to(device)
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
        import numpy as _np
        if not _np.isfinite(vl):
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


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_attreg_std(dataset_name: str, seed_idx: int):
    """Real data, X standardized (as before) AND y standardized."""
    import time as _t, traceback
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        N_full, P = X.shape
        if N_full > MAX_N:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(N_full, MAX_N, replace=False)
            X, y = X[idx], y[idx]
        seed = SEED + seed_idx * 1000
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC, random_state=seed)
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        scy = StandardScaler()
        ytr = scy.fit_transform(np.asarray(ytr).reshape(-1, 1)).ravel()
        yte = scy.transform(np.asarray(yte).reshape(-1, 1)).ravel()
        pred = train_attreg(Xtr.astype('float32'), ytr.astype('float32'),
                            Xte.astype('float32'), seed=seed)
        return {'kind': 'attreg_std', 'dataset': dataset_name, 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'attreg_std', 'dataset': dataset_name, 'seed_idx': seed_idx,
                'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


MC_DGPS = ['linear', 'friedman1', 'friedman2', 'friedman3', 'rotated_sine', 'soft_radial']
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
def run_mc_attreg_std(dgp: str, N: int, snr: float):
    """MC cell, Att.Reg only, X and y standardized (train-fit)."""
    import time as _t, traceback
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    t0 = _t.time()
    out = {'kind': 'mc_attreg_std', 'dgp': dgp, 'N': N, 'snr': snr, 'error': None}
    try:
        vals = []
        for rep in range(MC_REPS):
            rs = np.random.RandomState(SEED + 7919 * rep + hash((dgp, N, int(snr*10))) % 100000)
            Xtr = rs.uniform(0, 1, size=(N, 5)); Xte = rs.uniform(0, 1, size=(MC_J, 5))
            f_tr, f_te = mc_f(dgp, Xtr), mc_f(dgp, Xte)
            sig = np.sqrt(np.var(f_tr) / snr)
            ytr = f_tr + rs.normal(0, sig, N); yte = f_te + rs.normal(0, sig, MC_J)
            sx = StandardScaler(); Xtr2 = sx.fit_transform(Xtr); Xte2 = sx.transform(Xte)
            sy = StandardScaler()
            ytr2 = sy.fit_transform(ytr.reshape(-1, 1)).ravel()
            yte2 = sy.transform(yte.reshape(-1, 1)).ravel()
            pred = train_attreg(Xtr2.astype('float32'), ytr2.astype('float32'),
                                Xte2.astype('float32'), seed=SEED + rep * 1000)
            vals.append(r2_score(yte2, pred))
        out['AttReg_mean'] = float(np.mean(vals)); out['AttReg_sd'] = float(np.std(vals))
        out['wall_s'] = _t.time() - t0
        return out
    except Exception as e:
        out['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        out['wall_s'] = _t.time() - t0
        return out
