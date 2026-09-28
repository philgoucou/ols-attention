"""
AUDIT PROBE ROUND 4 — the legacy attention net of round 3 under the corrected
protocol and a longer step budget.

Same VERBATIM port of colab_full.py::train_torch_model (the generic 2-block
self-attention regressor, M=4 heads, AdamW, lr 1e-3, wd 1e-3) as audit_probes3.py,
but run_legacy_attn takes (steps, ystd) instead of with_retry: the target is
standardized on the training split when ystd is set, and the number of steps is a
parameter. The saved grid is steps in {300, 3000} x ystd=True, 8 datasets x 5 seeds
(audit_legacy4_results.csv); there is no test-conditioned retry arm. run_mc_legacy is
unchanged from round 3. FORENSICS ONLY, to fingerprint the provenance of the submitted
Att.Reg column; never for reporting.

Deploy: modal deploy audit_probes4.py   (app: neurips-31482-audit4)
"""
from __future__ import annotations
import pickle
import modal

TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42
ATTN_M = 4; ATTN_STEPS = 300; ATTN_LR = 1e-3; ATTN_REG = 1e-3

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-audit4")


def _load_cached():
    import os
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def train_torch_model(X_train, y_train, X_test, M=4, steps=300, lr=1e-3,
                      reg_lambda=1e-3, seed=42):
    """VERBATIM port of colab_full.py::train_torch_model (2-block self-attn net)."""
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = 64

    class AttnRegressor(nn.Module):
        def __init__(self, nf, d, nh):
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.randn(1, nf, d) * 0.02)
            self.attn1 = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0.1)
            self.n1 = nn.LayerNorm(d)
            self.ff1 = nn.Sequential(nn.Linear(d, d*2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d*2, d))
            self.n2 = nn.LayerNorm(d)
            self.attn2 = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0.1)
            self.n3 = nn.LayerNorm(d)
            self.ff2 = nn.Sequential(nn.Linear(d, d*2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d*2, d))
            self.n4 = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
        def forward(self, x):
            x = self.embed(x.unsqueeze(-1)) + self.pos
            a, _ = self.attn1(x, x, x); x = self.n1(x + a); x = self.n2(x + self.ff1(x))
            a, _ = self.attn2(x, x, x); x = self.n3(x + a); x = self.n4(x + self.ff2(x))
            return self.head(x.mean(1))

    model = AttnRegressor(nf, D, max(1, M)).to(device)
    N = X_train.shape[0]
    n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    tr_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_lambda)
    crit = nn.MSELoss()
    best_val = float('inf'); best_state = None; patience = 0; step = 0
    done = False
    while not done:
        model.train()
        for Xb, yb in tr_loader:
            if step >= steps:
                done = True; break
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xt_val), yt_val).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10: break
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_test.astype(np.float32)).to(device)).cpu().numpy().ravel()


@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_legacy_attn(dataset_name: str, seed_idx: int, steps: int, ystd: bool):
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
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype('float32'); Xte = sc.transform(Xte).astype('float32')
        ytr = np.asarray(ytr, dtype='float32'); yte = np.asarray(yte, dtype='float32')
        if ystd:
            scy = StandardScaler()
            ytr = scy.fit_transform(ytr.reshape(-1, 1)).ravel().astype('float32')
            yte = scy.transform(yte.reshape(-1, 1)).ravel().astype('float32')
        pred = train_torch_model(Xtr, ytr, Xte, M=ATTN_M, steps=steps,
                                 lr=ATTN_LR, reg_lambda=ATTN_REG, seed=seed)
        r2 = float(r2_score(yte, pred))
        return {'kind': 'legacy_attn4', 'dataset': dataset_name, 'seed_idx': seed_idx,
                'steps': steps, 'ystd': bool(ystd), 'r2': r2, 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'legacy_attn4', 'dataset': dataset_name, 'seed_idx': seed_idx,
                'steps': steps, 'ystd': bool(ystd), 'r2': float('nan'), 'wall_s': _t.time()-t0,
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


@app.function(image=image, gpu="T4", timeout=3600, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_mc_legacy(dgp: str, N: int, snr: float):
    """MC cell, legacy 2-block net only (no retry — honest arm)."""
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    t0 = _t.time()
    out = {'kind': 'mc_legacy', 'dgp': dgp, 'N': N, 'snr': snr, 'error': None}
    try:
        vals = []
        for rep in range(MC_REPS):
            rs = np.random.RandomState(SEED + 7919 * rep + hash((dgp, N, int(snr*10))) % 100000)
            Xtr = rs.uniform(0, 1, size=(N, 5)).astype('float32')
            Xte = rs.uniform(0, 1, size=(MC_J, 5)).astype('float32')
            f_tr, f_te = mc_f(dgp, Xtr), mc_f(dgp, Xte)
            sig = np.sqrt(np.var(f_tr) / snr)
            ytr = (f_tr + rs.normal(0, sig, N)).astype('float32')
            yte = (f_te + rs.normal(0, sig, MC_J)).astype('float32')
            pred = train_torch_model(Xtr, ytr, Xte, M=ATTN_M, steps=ATTN_STEPS,
                                     lr=ATTN_LR, reg_lambda=ATTN_REG, seed=SEED + rep * 1000)
            vals.append(r2_score(yte, pred))
        out['Attn_mean'] = float(np.mean(vals)); out['Attn_sd'] = float(np.std(vals))
        out['wall_s'] = _t.time() - t0
        return out
    except Exception as e:
        out['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        out['wall_s'] = _t.time() - t0
        return out
