"""
Modal fan-out — higher-dimensional data (xZ71-Q2 literal "higher-dimensional datasets").

A P-ladder of real regression datasets:  CPU_act(21) Bank32nh(32) Ailerons(40)
Pol(48) Superconductivity(81), plus a Friedman-1 Monte Carlo at P=20 and P=50
(only 5 features informative; the rest are noise -> tests robustness to irrelevant dims).
Models: OLS, RandomForest, FT-Transformer, Regression Block.  Capped N=5000, 5 seeds.

The RB's poly-cross feature map is O(P^2) in width, so batch size / PCA-fit subsample
are reduced for large P (documented in `rb_mem_settings`) and RB runs on A10G GPUs.
This is itself the honest scalability caveat: run it, report where it degrades.

    modal run rebuttal_highdim.py
"""
from __future__ import annotations
import pickle, time
import modal

N_REPEATS = 5
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42
RF_TREES = 500; RF_MTRY = 1/3

FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128; FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1; FTT_EPOCHS = 200; FTT_BS = 256; FTT_LR = 1e-3; FTT_WD = 1e-4
POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4

# real datasets: name -> (openml_id, P)
REAL = {
    'CPU_act':           (197, 21),
    'Bank32nh':          (558, 32),
    'Ailerons':          (296, 40),
    'Pol':               (201, 48),
    'Superconductivity': (43174, 81),
}
# Friedman MC: name -> P
FRIEDMAN = {'Friedman-P20': 20, 'Friedman-P50': 50}
FR_N = 5500; FR_NOISE = 1.0
DS_ORDER = ['Friedman-P20', 'Friedman-P50', 'CPU_act', 'Bank32nh', 'Ailerons', 'Pol', 'Superconductivity']
MODELS = ['OLS', 'RF', 'FTT', 'RB']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-highdim")


def rb_mem_settings(P):
    if P <= 25: return dict(pca_fit_cap=2000, batch_size=256)
    if P <= 35: return dict(pca_fit_cap=1200, batch_size=128)
    if P <= 45: return dict(pca_fit_cap=700,  batch_size=96)
    if P <= 60: return dict(pca_fit_cap=400,  batch_size=48)
    return dict(pca_fit_cap=100, batch_size=24)


@app.function(image=image, volumes={CACHE_DIR: cache_vol}, timeout=1800)
def fetch_and_cache_highdim():
    import os, numpy as np
    from sklearn.datasets import fetch_openml
    pkl = os.path.join(CACHE_DIR, "highdim_datasets_v1.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            ds = pickle.load(f)
        return {k: v[0].shape for k, v in ds.items()}
    loaded = {}
    for name, (did, P) in REAL.items():
        try:
            d = fetch_openml(data_id=did, as_frame=True, parser='auto')
            X = d.data.select_dtypes(include=[np.number]).values.astype(np.float32)
            y = np.asarray(d.target).astype(np.float32)
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            assert X.shape[1] == P, f"{name} P={X.shape[1]} != {P}"
            loaded[name] = (X, y)
            print(f"  OK {name}: X={X.shape}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
    with open(pkl, "wb") as f:
        pickle.dump(loaded, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache_vol.commit()
    return {k: v[0].shape for k, v in loaded.items()}


def _load_highdim():
    import os
    with open(os.path.join(CACHE_DIR, "highdim_datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def get_dataset(name, seed_idx):
    """Return full (X, y) for a dataset; Friedman is generated per-seed."""
    import numpy as np
    if name in FRIEDMAN:
        from sklearn.datasets import make_friedman1
        P = FRIEDMAN[name]
        X, y = make_friedman1(n_samples=FR_N, n_features=P, noise=FR_NOISE,
                              random_state=SEED + seed_idx * 1000)
        return X.astype(np.float32), y.astype(np.float32)
    X, y = _load_highdim()[name]
    return X, y


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
    # standardize y too: R^2 is invariant to it for well-scaled targets, but it
    # rescues optimization on tiny-magnitude targets (e.g. Ailerons ~1e-4) where
    # unscaled MSE degenerates. Fit on train only.
    scy = StandardScaler(); ytr = scy.fit_transform(ytr.reshape(-1, 1)).ravel()
    yte = scy.transform(yte.reshape(-1, 1)).ravel()
    return Xtr.astype('float32'), ytr.astype('float32'), Xte.astype('float32'), yte.astype('float32'), P, seed


# ---- FT-Transformer (verbatim) ----
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
            enc = nn.TransformerEncoderLayer(d_model=D, nhead=FTT_N_HEADS, dim_feedforward=FTT_FFN_DIM,
                                             dropout=FTT_DROPOUT, batch_first=True, activation='gelu')
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
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: ep/warm if ep < warm else 0.5*(1+np.cos(np.pi*(ep-warm)/max(1, FTT_EPOCHS-warm))))
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
    return np.concatenate(preds), {'n_params': sum(p.numel() for p in model.parameters())}


# ---- Regression Block (memory-adaptive) ----
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

    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    torch.manual_seed(seed)
    tmp_embed = nn.Linear(1, D); tmp_pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
    Xt = torch.from_numpy(X_pca.astype(np.float32))
    with torch.no_grad():
        emb = tmp_embed(Xt.unsqueeze(-1)) + tmp_pos
        phi = polycross(emb); feat_dim = phi.shape[-1]
        phi = phi.reshape(-1, feat_dim).numpy()
    n_comp = min(POLY_PCA_COMP, phi.shape[1], phi.shape[0])
    p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
    red = FrozenPCA(p.mean_, p.components_)
    torch.manual_seed(seed)
    model = Net(red).to(device)

    # Ridge warm-start (chunk sized by batch_size — poly-cross is O(P^2) wide,
    # so a fixed large chunk OOMs at high P)
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
    Xt2 = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32)); yt2 = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device); yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
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
    return np.concatenate(preds), {'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


# ================= Modal functions =================
@app.function(image=image, timeout=900, volumes={CACHE_DIR: cache_vol},
              cpu=4.0, memory=16384, retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_sk(dataset_name: str, model_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = get_dataset(dataset_name, seed_idx)
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        if model_name == 'OLS':
            from sklearn.linear_model import LinearRegression
            m = LinearRegression().fit(Xtr, ytr); pred = m.predict(Xte)
        elif model_name == 'RF':
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=RF_TREES, max_features=RF_MTRY,
                                      n_jobs=-1, random_state=seed).fit(Xtr, ytr)
            pred = m.predict(Xte)
        else:
            raise ValueError(model_name)
        return {'dataset': dataset_name, 'model': model_name, 'P': int(P), 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'model': model_name, 'P': None, 'seed_idx': seed_idx,
                'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.function(image=image, gpu="A10G", timeout=3600, volumes={CACHE_DIR: cache_vol},
              memory=32768, retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_nn(dataset_name: str, model_name: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = get_dataset(dataset_name, seed_idx)
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        if model_name == 'FTT':
            pred, info = train_ft_transformer(Xtr, ytr, Xte, seed=seed)
        elif model_name == 'RB':
            ms = rb_mem_settings(P)
            pred, info = train_rb(Xtr, ytr, Xte, pca_fit_cap=ms['pca_fit_cap'],
                                  batch_size=ms['batch_size'], seed=seed)
        else:
            raise ValueError(model_name)
        return {'dataset': dataset_name, 'model': model_name, 'P': int(P), 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'params': int(info['n_params']),
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'model': model_name, 'P': None, 'seed_idx': seed_idx,
                'r2': float('nan'), 'params': None, 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.local_entrypoint()
def main():
    import pandas as pd
    t0 = time.time()
    shapes = fetch_and_cache_highdim.remote()
    print(">> cached high-dim datasets:")
    for k, v in shapes.items():
        print(f"   {k}: {v}")
    sk_args = [(ds, m, s) for ds in DS_ORDER for m in ['OLS', 'RF'] for s in range(N_REPEATS)]
    nn_args = [(ds, m, s) for ds in DS_ORDER for m in ['FTT', 'RB'] for s in range(N_REPEATS)]
    print(f">> High-dim: {len(sk_args)} sklearn + {len(nn_args)} NN jobs")

    sk_res = list(run_sk.starmap(sk_args, order_outputs=False))
    print(f">> sklearn done in {time.time()-t0:.1f}s")
    nn_res = list(run_nn.starmap(nn_args, order_outputs=False))
    print(f">> NN done in {time.time()-t0:.1f}s")

    allres = sk_res + nn_res
    df = pd.DataFrame(allres)
    df.to_csv("highdim_results.csv", index=False)
    errs = df[df['error'].notna()]
    if len(errs):
        print(f"\n[highdim errors: {len(errs)}]")
        for _, r in errs.iterrows():
            print(f"  {r['dataset']}/{r['model']}/seed{r['seed_idx']}: {r['error'].splitlines()[0]}")

    agg = df.groupby(['dataset', 'model']).agg(r2=('r2', 'mean'), sd=('r2', 'std'),
                                               P=('P', 'max')).reset_index()
    piv = agg.pivot(index='dataset', columns='model', values='r2').reindex(DS_ORDER)[MODELS]
    Pmap = agg.groupby('dataset')['P'].max()
    print("\n" + "#"*70 + "\n# HIGHER-DIM — mean R2 (capped N, 5 seeds)\n" + "#"*70)
    out = piv.copy(); out.insert(0, 'P', [int(Pmap.get(d, -1)) if pd.notna(Pmap.get(d)) else -1 for d in out.index])
    print(out.round(3).to_string())
    print("\n--- RB minus FT-T by P (does RB degrade as P grows?) ---")
    for ds in piv.index:
        P = Pmap.get(ds)
        if pd.isna(piv.loc[ds, 'RB']) or pd.isna(piv.loc[ds, 'FTT']):
            print(f"  {ds:<18} P={int(P) if pd.notna(P) else '?':>3}: (missing)"); continue
        print(f"  {ds:<18} P={int(P):>3}:  RB={piv.loc[ds,'RB']:+.3f}  FT-T={piv.loc[ds,'FTT']:+.3f}  "
              f"RF={piv.loc[ds,'RF']:+.3f}  (RB-FTT {piv.loc[ds,'RB']-piv.loc[ds,'FTT']:+.3f})")
    print("\n--- OpenReview markdown: higher-dim ---")
    print("| Dataset | P | OLS | RF | FT-T | RB |")
    print("|---|---|---|---|---|---|")
    for ds in piv.index:
        P = Pmap.get(ds)
        cells = " | ".join(f"{piv.loc[ds, m]:.3f}" if pd.notna(piv.loc[ds, m]) else "—" for m in MODELS)
        print(f"| {ds} | {int(P) if pd.notna(P) else '?'} | {cells} |")
    print(f"\nSaved: highdim_results.csv  ({time.time()-t0:.1f}s total)")


@app.function(image=image, gpu="A100-80GB", timeout=5400, volumes={CACHE_DIR: cache_vol},
              memory=65536, retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_nn_big(dataset_name: str, model_name: str, seed_idx: int):
    """A100-80GB variant for the largest-P RB (Superconductivity, P=81)."""
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = get_dataset(dataset_name, seed_idx)
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        ms = rb_mem_settings(P)
        pred, info = train_rb(Xtr, ytr, Xte, pca_fit_cap=ms['pca_fit_cap'],
                              batch_size=ms['batch_size'], seed=seed)
        return {'dataset': dataset_name, 'model': model_name, 'P': int(P), 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'params': int(info['n_params']),
                'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'model': model_name, 'P': None, 'seed_idx': seed_idx,
                'r2': float('nan'), 'params': None, 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.local_entrypoint()
def rerun_super():
    """Targeted: Superconductivity (P=81) RB on A100-80GB, splice into highdim_results.csv."""
    import pandas as pd
    t0 = time.time()
    args = [('Superconductivity', 'RB', s) for s in range(N_REPEATS)]
    print(f">> Superconductivity RB on A100-80GB, {len(args)} seeds")
    res = list(run_nn_big.starmap(args, order_outputs=False))
    for r in sorted(res, key=lambda r: r['seed_idx']):
        e = None if r['error'] is None else r['error'].splitlines()[0]
        print(f"   seed{r['seed_idx']}: r2={r['r2']:+.4f} wall={r['wall_s']:.0f}s err={e}")
    ok = [r for r in res if r['error'] is None]
    if ok:
        df = pd.read_csv("highdim_results.csv")
        df = df[~((df['dataset'] == 'Superconductivity') & (df['model'] == 'RB'))]
        df = pd.concat([df, pd.DataFrame(res)], ignore_index=True)
        df.to_csv("highdim_results.csv", index=False)
        import numpy as np
        print(f">> spliced. Superconductivity RB mean R2 = {np.nanmean([r['r2'] for r in ok]):+.4f}")
    else:
        print(">> all seeds still failed — report as scalability limitation")
    print(f">> ({time.time()-t0:.1f}s)")


@app.local_entrypoint()
def rerun_rb():
    """Rerun RB for all datasets with the fixed (P-scaled) warm-start chunk,
    splice into highdim_results.csv, reprint the table."""
    import os, pandas as pd
    t0 = time.time()
    args = [(ds, 'RB', s) for ds in DS_ORDER for s in range(N_REPEATS)]
    print(f">> Rerunning {len(args)} RB high-dim jobs (fixed warm-start)")
    res = list(run_nn.starmap(args, order_outputs=False))
    for r in sorted(res, key=lambda r: (r['dataset'], r['seed_idx'])):
        e = None if r['error'] is None else r['error'].splitlines()[0]
        print(f"   {r['dataset']:<18} seed{r['seed_idx']} P={r.get('P')}: r2={r['r2']:+.4f} "
              f"wall={r['wall_s']:.0f}s err={e}")

    csv = "highdim_results.csv"
    df = pd.read_csv(csv)
    mask = (df['model'] == 'RB')
    print(f">> dropping {int(mask.sum())} old RB rows")
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame(res)], ignore_index=True)
    df.to_csv(csv, index=False)

    errs = pd.DataFrame(res)
    errs = errs[errs['error'].notna()]
    if len(errs):
        print(f"\n[still failing: {len(errs)}]")
        for _, r in errs.iterrows():
            print(f"  {r['dataset']}/seed{r['seed_idx']}: {r['error'].splitlines()[0]}")

    agg = df.groupby(['dataset', 'model']).agg(r2=('r2', 'mean'), P=('P', 'max')).reset_index()
    piv = agg.pivot(index='dataset', columns='model', values='r2').reindex(DS_ORDER)[MODELS]
    Pmap = agg.groupby('dataset')['P'].max()
    print("\n" + "#"*70 + "\n# HIGHER-DIM (RB fixed) — mean R2\n" + "#"*70)
    out = piv.copy(); out.insert(0, 'P', [int(Pmap.get(d, -1)) if pd.notna(Pmap.get(d)) else -1 for d in out.index])
    print(out.round(3).to_string())
    print("\n--- RB minus FT-T by P ---")
    for ds in piv.index:
        Pv = Pmap.get(ds)
        if pd.isna(piv.loc[ds, 'RB']) or pd.isna(piv.loc[ds, 'FTT']):
            print(f"  {ds:<18}: (missing)"); continue
        print(f"  {ds:<18} P={int(Pv):>3}:  RB={piv.loc[ds,'RB']:+.3f}  FT-T={piv.loc[ds,'FTT']:+.3f}  "
              f"RF={piv.loc[ds,'RF']:+.3f}  (RB-FTT {piv.loc[ds,'RB']-piv.loc[ds,'FTT']:+.3f})")
    print(f"\nUpdated: highdim_results.csv  ({time.time()-t0:.1f}s)")
