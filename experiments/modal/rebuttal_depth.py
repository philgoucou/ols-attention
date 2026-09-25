"""
Modal fan-out: depth sweep for the single-layer limitation.
  * Stacked Regression Block at L = 1, 2, 3
  * FT-Transformer at n_blocks = 1, 2, 3  (matched-depth control)
8 datasets, capped N=5000, 5 seeds each. Reuses the cached dataset volume.

    modal run rebuttal_depth.py
"""
from __future__ import annotations
import pickle, time
import modal

N_REPEATS = 5
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42
DEPTHS = [1, 2, 3]

FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128
FTT_DROPOUT = 0.1; FTT_EPOCHS = 200; FTT_BS = 256; FTT_LR = 1e-3; FTT_WD = 1e-4

POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4

DS_ORDER = ['California', 'Yacht', 'Energy', 'Concrete',
            'Airfoil',    'Abalone', 'Kin8nm', 'Protein']

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "pandas==2.2.3",
                 "scikit-learn==1.5.2", "torch==2.4.1")
)
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-depth")


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
    return (Xtr.astype(np.float32), ytr.astype(np.float32),
            Xte.astype(np.float32), yte.astype(np.float32), P, seed)


# ================= FT-Transformer (verbatim, n_blocks param) =================
def train_ft_transformer(X_train, y_train, X_test, n_blocks,
                         d_model=64, n_heads=4, ffn_dim=128, dropout=0.1,
                         epochs=200, batch_size=256, lr=1e-3,
                         weight_decay=1e-4, seed=42):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model

    class FTTransformer(nn.Module):
        def __init__(self, nf, d, nh, ffn, nblk, drop):
            super().__init__()
            self.feat_embeds = nn.ModuleList([nn.Linear(1, d) for _ in range(nf)])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=nh, dim_feedforward=ffn, dropout=drop,
                batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(enc, num_layers=nblk)
            self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, 1)

        def forward(self, x):
            B = x.size(0)
            tok = torch.stack([self.feat_embeds[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            tok = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
            z = self.transformer(tok)
            return self.head(self.norm(z[:, 0, :]))

    model = FTTransformer(nf, D, n_heads, ffn_dim, n_blocks, dropout).to(device)
    N = X_train.shape[0]; n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]])
    yt_tr = torch.from_numpy(y_train[idx[n_val:]]).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]]).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]]).unsqueeze(1).to(device)
    tr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = max(1, epochs // 10)
    def lam(ep):
        if ep < warm: return ep / warm
        prog = (ep - warm) / max(1, epochs - warm)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    best, best_state, pc = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vl = crit(model(Xt_val), yt_val).item()
        if vl < best:
            best, best_state, pc = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= 10: break
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i+4096]).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(preds), {'n_params': sum(p.numel() for p in model.parameters())}


# ================= Stacked Regression Block =================
def train_stacked_rb(X_train, y_train, X_test, n_layers=1, use_pca=True,
                     n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL,
                     ffn_dim=POLY_FFN_DIM, epochs=POLY_EPOCHS, batch_size=POLY_BS,
                     lr=POLY_LR, dropout=POLY_DROPOUT, weight_decay=POLY_WD, seed=42):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    nf = X_train.shape[1]; D = d_model; pca_fit_cap = 5000

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

    class RBBlock(nn.Module):
        def __init__(self, red, d, ffn_dim, drop):
            super().__init__()
            self.red = red; self.W = nn.Linear(red.out_dim, d)
            self.n1 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.ReLU(), nn.Dropout(drop),
                                     nn.Linear(ffn_dim, d), nn.Dropout(drop))
            self.n2 = nn.LayerNorm(d)
        def feat(self, h): return self.red(polycross(h))
        def forward(self, h):
            z = self.n1(h + self.W(self.feat(h)))
            return self.n2(z + self.ffn(z))

    class StackedRB(nn.Module):
        def __init__(self, nf, d, ffn_dim, drop):
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.randn(1, nf, d) * 0.02)
            self.blocks = nn.ModuleList(); self.head = nn.Linear(d, 1)
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
    model = StackedRB(nf, D, ffn_dim, dropout).to(device)
    with torch.no_grad():
        feat_dim = polycross(model.embed_x(X_pca_t[:8])).shape[-1]

    for l in range(n_layers):
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
        model.blocks.append(RBBlock(red, D, ffn_dim, dropout).to(device))

    # warm-start block 0 with Ridge; head reads channel 0
    model.eval()
    WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
    with torch.no_grad():
        for i in range(0, WS_N, 4096):
            xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
            phis.append(model.blocks[0].feat(model.embed_x(xb)).mean(1).cpu().numpy())
    import numpy as _np
    phi_pooled = _np.concatenate(phis)
    reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
    with torch.no_grad():
        model.blocks[0].W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
        model.blocks[0].W.bias.data.zero_()
        model.head.weight.data.zero_(); model.head.weight.data[0, 0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

    N = X_train.shape[0]; n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    tr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss(); best, best_state, pat = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xt_val), yt_val).item()
        if vl < best:
            best, best_state, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(preds), {
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'n_layers': n_layers}


# ================= Modal remote functions =================
@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_depth_rb(dataset_name: str, n_layers: int, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        pred, info = train_stacked_rb(Xtr, ytr, Xte, n_layers=n_layers, seed=seed)
        return {'kind': 'rb', 'dataset': dataset_name, 'model': f'RB-L{n_layers}',
                'depth': n_layers, 'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'params': int(info['n_params']), 'wall_s': _t.time() - t0, 'error': None}
    except Exception as e:
        return {'kind': 'rb', 'dataset': dataset_name, 'model': f'RB-L{n_layers}',
                'depth': n_layers, 'seed_idx': seed_idx, 'r2': float('nan'),
                'params': None, 'wall_s': _t.time() - t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_depth_ftt(dataset_name: str, n_blocks: int, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, P, seed = prep_capped(X, y, seed_idx)
        pred, info = train_ft_transformer(
            Xtr, ytr, Xte, n_blocks=n_blocks, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
            ffn_dim=FTT_FFN_DIM, dropout=FTT_DROPOUT, epochs=FTT_EPOCHS,
            batch_size=FTT_BS, lr=FTT_LR, weight_decay=FTT_WD, seed=seed)
        return {'kind': 'ftt', 'dataset': dataset_name, 'model': f'FTT-nb{n_blocks}',
                'depth': n_blocks, 'seed_idx': seed_idx, 'r2': float(r2_score(yte, pred)),
                'params': int(info['n_params']), 'wall_s': _t.time() - t0, 'error': None}
    except Exception as e:
        return {'kind': 'ftt', 'dataset': dataset_name, 'model': f'FTT-nb{n_blocks}',
                'depth': n_blocks, 'seed_idx': seed_idx, 'r2': float('nan'),
                'params': None, 'wall_s': _t.time() - t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.local_entrypoint()
def main():
    import pandas as pd
    t0 = time.time()
    datasets = list(DS_ORDER)
    rb_args = [(ds, L, s) for ds in datasets for L in DEPTHS for s in range(N_REPEATS)]
    ftt_args = [(ds, nb, s) for ds in datasets for nb in DEPTHS for s in range(N_REPEATS)]
    print(f">> Depth sweep: {len(rb_args)} stacked-RB + {len(ftt_args)} FT-T jobs")

    rb_res = list(run_depth_rb.starmap(rb_args, order_outputs=False))
    print(f">> RB depth done in {time.time()-t0:.1f}s")
    ftt_res = list(run_depth_ftt.starmap(ftt_args, order_outputs=False))
    print(f">> FT-T depth done in {time.time()-t0:.1f}s")

    allres = rb_res + ftt_res
    df = pd.DataFrame(allres)
    df.to_csv("depth_results.csv", index=False)

    errs = df[df['error'].notna()]
    if len(errs):
        print(f"\n[depth errors: {len(errs)}]")
        for _, r in errs.iterrows():
            print(f"  {r['model']}/{r['dataset']}/seed{r['seed_idx']}: {r['error'].splitlines()[0]}")

    agg = df.groupby(['dataset', 'model']).agg(r2=('r2', 'mean'), sd=('r2', 'std'),
                                               params=('params', 'max')).reset_index()
    print("\n" + "#" * 70 + "\n# DEPTH SWEEP — mean R2 (capped N, 5 seeds)\n" + "#" * 70)
    models = [f'RB-L{L}' for L in DEPTHS] + [f'FTT-nb{nb}' for nb in DEPTHS]
    piv = agg.pivot(index='dataset', columns='model', values='r2').reindex(
        [d for d in DS_ORDER if d in agg.dataset.unique()])[models]
    print("\n--- mean R2 ---")
    print(piv.round(3).to_string())
    prm = agg.pivot(index='dataset', columns='model', values='params')[models]
    print("\n--- params (should match across datasets w/ same P) ---")
    print(prm.round(0).astype('Int64').to_string())

    print("\n--- Depth deltas (RB) ---")
    for ds in piv.index:
        d1, d2, d3 = piv.loc[ds, 'RB-L1'], piv.loc[ds, 'RB-L2'], piv.loc[ds, 'RB-L3']
        print(f"  {ds:<11} L1={d1:+.3f}  L2={d2:+.3f} ({d2-d1:+.3f})  L3={d3:+.3f} ({d3-d1:+.3f})")

    print("\n--- OpenReview markdown: depth sweep ---")
    print("| Dataset | " + " | ".join(models) + " |")
    print("|---" * (len(models) + 1) + "|")
    for ds in piv.index:
        print(f"| {ds} | " + " | ".join(f"{piv.loc[ds, m]:.3f}" if piv.loc[ds, m] > -1 else "<0"
                                        for m in models) + " |")
    print(f"\nSaved: depth_results.csv  ({time.time()-t0:.1f}s total)")
