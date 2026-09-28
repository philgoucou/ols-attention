"""
Modal fan-out — warm-start decomposition and learning curves (raw-target protocol of
the first rebuttal wave). Formerly rebuttal_extras.py.

(A) Warm-start ablation, all 8 datasets, capped N, 5 seeds. Same-seed decomposition
    A4-softmax  ->  RB-nowarm (regression readout, random init)  ->  RB-warm (+ Ridge init)
    Isolates the regression readout from the Ridge warm-start. If RB-nowarm still
    clearly beats softmax, the readout (not the warm-start) is the driver.

(B) Learning curves: R2 vs N in {500,1000,2500,5000,10000,full} for FT-T and RB on
    California / Kin8nm / Protein. Fixed test split per seed; train set subsampled.

    modal run rebuttal_warmstart_lcurve.py
Writes warmstart_results.csv and lcurve_results.csv in the working directory (kept under
results/rebuttal/). The corrected-protocol warm-start rerun is rebuttal_ablation_warmstart.py.
"""
from __future__ import annotations
import time
import modal

N_REPEATS = 5
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42
LC_DATASETS = ['California', 'Kin8nm', 'Protein']
LC_NS = [500, 1000, 2500, 5000, 10000, -1]   # -1 = full train split

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
app = modal.App("neurips-31482-extras")


def _load_cached():
    import os, pickle as _pk
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return _pk.load(f)


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
    return Xtr.astype(np.float32), ytr.astype(np.float32), Xte.astype(np.float32), yte.astype(np.float32), P, seed


def prep_lcurve(X_full, y_full, seed_idx, n_train):
    """Fixed 80/20 test split per seed; subsample train to n_train (-1 = all)."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    seed = SEED + seed_idx * 1000
    Xtr_full, Xte, ytr_full, yte = train_test_split(X_full, y_full, test_size=TEST_FRAC, random_state=seed)
    avail = len(Xtr_full)
    if n_train == -1 or n_train >= avail:
        Xtr, ytr = Xtr_full, ytr_full
        n_used = avail
    else:
        rng = np.random.RandomState(seed + 7)
        idx = rng.choice(avail, n_train, replace=False)
        Xtr, ytr = Xtr_full[idx], ytr_full[idx]
        n_used = n_train
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    return Xtr.astype(np.float32), ytr.astype(np.float32), Xte.astype(np.float32), yte.astype(np.float32), n_used, seed


# ================= FT-Transformer =================
def train_ft_transformer(X_train, y_train, X_test, n_blocks=FTT_N_BLOCKS,
                         d_model=64, n_heads=4, ffn_dim=128, dropout=0.1,
                         epochs=200, batch_size=256, lr=1e-3, weight_decay=1e-4, seed=42):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model

    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_embeds = nn.ModuleList([nn.Linear(1, D) for _ in range(nf)])
            self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            enc = nn.TransformerEncoderLayer(d_model=D, nhead=n_heads, dim_feedforward=ffn_dim,
                                             dropout=dropout, batch_first=True, activation='gelu')
            self.tr = nn.TransformerEncoder(enc, num_layers=n_blocks)
            self.norm = nn.LayerNorm(D); self.head = nn.Linear(D, 1)
        def forward(self, x):
            B = x.size(0)
            tok = torch.stack([self.feat_embeds[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            tok = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)
            return self.head(self.norm(self.tr(tok)[:, 0, :]))

    model = FT().to(device)
    N = X_train.shape[0]; n_val = max(1, int(N * 0.15)); idx = np.random.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]]); yt = torch.from_numpy(y_train[idx[n_val:]]).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]]).to(device); yv = torch.from_numpy(y_train[idx[:n_val]]).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    crit = nn.MSELoss(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = max(1, epochs // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: ep/warm if ep < warm else 0.5*(1+np.cos(np.pi*(ep-warm)/max(1, epochs-warm))))
    best, bs, pc = float('inf'), None, 0
    for ep in range(1, epochs+1):
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


# ================= Flexible Regression Block =================
def train_rb_flex(X_train, y_train, X_test, mixer='regression', warmstart=True,
                  n_layers=1, use_pca=True, n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL,
                  ffn_dim=POLY_FFN_DIM, n_heads=4, epochs=POLY_EPOCHS, batch_size=POLY_BS,
                  lr=POLY_LR, dropout=POLY_DROPOUT, weight_decay=POLY_WD, seed=42):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    nf = X_train.shape[1]; D = d_model; pca_fit_cap = 5000
    is_reg = (mixer == 'regression')

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

    class Identity(nn.Module):
        def __init__(self, dim): super().__init__(); self.out_dim = dim
        def forward(self, x): return x

    class Block(nn.Module):
        def __init__(self, red):
            super().__init__()
            if is_reg:
                self.red = red; self.W = nn.Linear(red.out_dim, D)
            else:
                self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True, dropout=dropout)
            self.n1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(ffn_dim, D), nn.Dropout(dropout))
            self.n2 = nn.LayerNorm(D)
        def feat(self, h): return self.red(polycross(h))
        def forward(self, h):
            if is_reg:
                z = self.n1(h + self.W(self.feat(h)))
            else:
                a, _ = self.attn(h, h, h); z = self.n1(h + a)
            return self.n2(z + self.ffn(z))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, D); self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.blocks = nn.ModuleList(); self.head = nn.Linear(D, 1)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def forward_upto(self, x, upto):
            h = self.embed_x(x)
            for l in range(upto): h = self.blocks[l](h)
            return h
        def forward(self, x):
            h = self.embed_x(x)
            for blk in self.blocks: h = blk(h)
            return self.head(h.mean(1))

    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    X_pca_t = torch.from_numpy(X_pca.astype(np.float32)).to(device)

    torch.manual_seed(seed)
    model = Net().to(device)
    with torch.no_grad():
        feat_dim = polycross(model.embed_x(X_pca_t[:8])).shape[-1]
    for l in range(n_layers):
        red = None
        if is_reg:
            model.eval()
            with torch.no_grad():
                h_in = model.forward_upto(X_pca_t, upto=l)
                phi = polycross(h_in).reshape(-1, feat_dim).cpu().numpy()
            if use_pca:
                n_comp = min(n_components, phi.shape[1], phi.shape[0])
                p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
                red = FrozenPCA(p.mean_, p.components_)
            else:
                red = Identity(feat_dim)
        torch.manual_seed(seed + 100 + l)
        model.blocks.append(Block(red).to(device))

    if is_reg and warmstart:
        model.eval()
        WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
        with torch.no_grad():
            for i in range(0, WS_N, 4096):
                xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
                phis.append(model.blocks[0].feat(model.embed_x(xb)).mean(1).cpu().numpy())
        phi_pooled = np.concatenate(phis)
        reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
        with torch.no_grad():
            model.blocks[0].W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
            model.blocks[0].W.bias.data.zero_()
            model.head.weight.data.zero_(); model.head.weight.data[0, 0] = 1.0
            model.head.bias.data.fill_(float(reg.intercept_))

    N = X_train.shape[0]; n_val = max(1, int(N * 0.15)); idx = np.random.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32)); yt = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device); yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay); crit = nn.MSELoss()
    best, bs, pat = float('inf'), None, 0
    for ep in range(1, epochs+1):
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
        for i in range(0, len(X_test), 4096):
            preds.append(model(torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)).cpu().numpy().ravel())
    return np.concatenate(preds), {'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


# ================= Modal functions =================
@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_warmstart(dataset_name: str, config: str, seed_idx: int):
    """config in {softmax, rb_nowarm, rb_warm}."""
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        if config == 'softmax':
            pred, info = train_rb_flex(Xtr, ytr, Xte, mixer='softmax', seed=seed)
        elif config == 'rb_nowarm':
            pred, info = train_rb_flex(Xtr, ytr, Xte, mixer='regression', warmstart=False, seed=seed)
        elif config == 'rb_warm':
            pred, info = train_rb_flex(Xtr, ytr, Xte, mixer='regression', warmstart=True, seed=seed)
        else:
            raise ValueError(config)
        return {'kind': 'warmstart', 'dataset': dataset_name, 'config': config,
                'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'params': int(info['n_params']), 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'warmstart', 'dataset': dataset_name, 'config': config,
                'seed_idx': seed_idx, 'r2': float('nan'), 'params': None,
                'wall_s': _t.time()-t0, 'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.function(image=image, gpu="T4", timeout=3600, volumes={CACHE_DIR: cache_vol},
              cpu=4.0, memory=16384, retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_lcurve(dataset_name: str, model_name: str, n_train: int, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, n_used, seed = prep_lcurve(X, y, seed_idx, n_train)
        if model_name == 'FTT':
            pred, info = train_ft_transformer(Xtr, ytr, Xte, n_blocks=FTT_N_BLOCKS, seed=seed)
        elif model_name == 'RB':
            pred, info = train_rb_flex(Xtr, ytr, Xte, mixer='regression', warmstart=True, seed=seed)
        else:
            raise ValueError(model_name)
        return {'kind': 'lcurve', 'dataset': dataset_name, 'model': model_name,
                'n_train_req': n_train, 'n_train_used': int(n_used), 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'kind': 'lcurve', 'dataset': dataset_name, 'model': model_name,
                'n_train_req': n_train, 'n_train_used': None, 'seed_idx': seed_idx,
                'r2': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.local_entrypoint()
def main():
    import pandas as pd
    t0 = time.time()

    # (A) warm-start ablation
    ws_cfgs = ['softmax', 'rb_nowarm', 'rb_warm']
    ws_args = [(ds, c, s) for ds in DS_ORDER for c in ws_cfgs for s in range(N_REPEATS)]
    # (B) learning curves
    lc_args = [(ds, m, n, s) for ds in LC_DATASETS for m in ['FTT', 'RB']
               for n in LC_NS for s in range(N_REPEATS)]
    print(f">> Extras: {len(ws_args)} warm-start + {len(lc_args)} learning-curve jobs")

    ws_res = list(run_warmstart.starmap(ws_args, order_outputs=False))
    print(f">> warm-start done in {time.time()-t0:.1f}s")
    lc_res = list(run_lcurve.starmap(lc_args, order_outputs=False))
    print(f">> learning-curve done in {time.time()-t0:.1f}s")

    pd.DataFrame(ws_res).to_csv("warmstart_results.csv", index=False)
    pd.DataFrame(lc_res).to_csv("lcurve_results.csv", index=False)

    # ---- warm-start report ----
    dfw = pd.DataFrame(ws_res)
    ew = dfw[dfw['error'].notna()]
    if len(ew):
        print(f"\n[warm-start errors: {len(ew)}]")
        for _, r in ew.iterrows(): print(f"  {r['dataset']}/{r['config']}: {r['error'].splitlines()[0]}")
    aggw = dfw.groupby(['dataset', 'config']).agg(r2=('r2', 'mean'), sd=('r2', 'std')).reset_index()
    pivw = aggw.pivot(index='dataset', columns='config', values='r2').reindex(
        [d for d in DS_ORDER if d in aggw.dataset.unique()])[ws_cfgs]
    print("\n" + "#"*70 + "\n# WARM-START ABLATION — mean R2\n" + "#"*70)
    print(pivw.round(3).to_string())
    pivw2 = pivw.copy()
    pivw2['readout gain (nowarm-softmax)'] = pivw['rb_nowarm'] - pivw['softmax']
    pivw2['warmstart gain (warm-nowarm)'] = pivw['rb_warm'] - pivw['rb_nowarm']
    print("\n--- decomposition (avg across datasets) ---")
    print(f"  softmax           mean R2 = {pivw['softmax'].mean():+.3f}")
    print(f"  + regression readout      = {pivw['rb_nowarm'].mean():+.3f}   "
          f"(gain {pivw['rb_nowarm'].mean()-pivw['softmax'].mean():+.3f})")
    print(f"  + Ridge warm-start        = {pivw['rb_warm'].mean():+.3f}   "
          f"(gain {pivw['rb_warm'].mean()-pivw['rb_nowarm'].mean():+.3f})")
    nowarm_beats = int((pivw['rb_nowarm'] > pivw['softmax']).sum())
    print(f"  RB-nowarm beats softmax on {nowarm_beats}/{len(pivw)} datasets")
    print("\n--- per-dataset readout vs warm-start gains ---")
    print(pivw2[['readout gain (nowarm-softmax)', 'warmstart gain (warm-nowarm)']].round(3).to_string())
    print("\n--- OpenReview markdown: warm-start ablation ---")
    print("| Dataset | Softmax (A4) | RB no-warmstart | RB warm-start |")
    print("|---|---|---|---|")
    for ds in pivw.index:
        print(f"| {ds} | {pivw.loc[ds,'softmax']:.3f} | {pivw.loc[ds,'rb_nowarm']:.3f} | {pivw.loc[ds,'rb_warm']:.3f} |")

    # ---- learning-curve report ----
    dfl = pd.DataFrame(lc_res)
    el = dfl[dfl['error'].notna()]
    if len(el):
        print(f"\n[lcurve errors: {len(el)}]")
        for _, r in el.iterrows(): print(f"  {r['dataset']}/{r['model']}/N{r['n_train_req']}: {r['error'].splitlines()[0]}")
    aggl = dfl.groupby(['dataset', 'model', 'n_train_req']).agg(
        r2=('r2', 'mean'), sd=('r2', 'std'), n_used=('n_train_used', 'max')).reset_index()
    print("\n" + "#"*70 + "\n# LEARNING CURVES — mean R2 by N\n" + "#"*70)
    for ds in LC_DATASETS:
        sub = aggl[aggl.dataset == ds]
        if not len(sub): continue
        print(f"\n{ds}:")
        piv = sub.pivot(index='n_train_req', columns='model', values='r2').sort_index()
        used = sub.groupby('n_train_req')['n_used'].max()
        for n in piv.index:
            label = 'full' if n == -1 else str(n)
            ftt = piv.loc[n, 'FTT'] if 'FTT' in piv.columns else float('nan')
            rb = piv.loc[n, 'RB'] if 'RB' in piv.columns else float('nan')
            print(f"  N={label:<6} (used {int(used.loc[n]):>6}):  FT-T={ftt:+.3f}  RB={rb:+.3f}  (RB-FTT {rb-ftt:+.3f})")
    print(f"\nSaved: warmstart_results.csv, lcurve_results.csv  ({time.time()-t0:.1f}s total)")
