"""
========================================================
FULL COLAB BENCHMARK — Poly PCA+OLS vs OLS vs RF
Paste this ENTIRE cell into Google Colab and run it.
GPU recommended (T4 is fine, ~15-20 min total).
========================================================
"""

# !pip install -q scikit-learn pandas openpyxl xlrd  # uncomment if needed

import time, warnings, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================
DATASETS = {
    'Yacht':       True,    # 308 obs, 6 feat
    'Energy':      True,    # 768 obs, 8 feat
    'Concrete':    True,    # 1030 obs, 8 feat
    'Airfoil':     True,    # 1503 obs, 5 feat
    'Abalone':     True,    # 4177 obs, 8 feat
    'California':  True,    # 20640 obs, 8 feat
    'Kin8nm':      True,    # 8192 obs, 8 feat
    'Protein':     True,    # 45730 obs, 9 feat
}

N_REPEATS = 5
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42

# Models to run (OLS and RF included for reference;
# set False if you already have those numbers from previous runs)
RUN_OLS  = True
RUN_RF   = True
RUN_POLY = True

# RF config
RF_TREES = 500; RF_MTRY = 1/3

# Poly PCA+OLS config
POLY_PCA_COMP = 200
POLY_D_MODEL = 64
POLY_FFN_DIM = 128
POLY_EPOCHS = 50
POLY_BS = 256
POLY_LR = 1e-3
POLY_DROPOUT = 0.1
POLY_WD = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# DATASET LOADING
# ==========================================
OPENML_REGISTRY = {
    'Concrete':  (44959,  (1000, 1100),  8),
    'Energy':    (41478,  (700, 800),     8),
    'Kin8nm':    (189,    (8000, 8300),   8),
    'Protein':   (44963,  (45000, 46000), 9),
    'Yacht':     (42370,  (300, 320),     6),
    'Airfoil':   (44957,  (1400, 1600),   5),
    'Abalone':   (183,    (4100, 4200),   7),
}

def load_datasets():
    enabled = [k for k, v in DATASETS.items() if v]
    print(f"\nLoading {len(enabled)} datasets...\n")
    loaded = {}

    if 'California' in enabled:
        try:
            d = fetch_california_housing()
            loaded['California'] = (d.data, d.target)
            print(f"  California:  {d.data.shape}")
        except Exception as e:
            print(f"  California FAILED: {e}")

    for name in enabled:
        if name == 'California':
            continue
        if name not in OPENML_REGISTRY:
            continue
        data_id, (n_min, n_max), expected_p = OPENML_REGISTRY[name]
        try:
            data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
            X = data.data.select_dtypes(include=[np.number]).values
            y_raw = data.target
            if hasattr(y_raw, 'shape') and len(y_raw.shape) > 1 and y_raw.shape[1] > 1:
                y_raw = y_raw.iloc[:, 0]
            elif hasattr(y_raw, 'columns'):
                y_raw = y_raw.iloc[:, 0]
            if hasattr(y_raw, 'cat') and y_raw.dtype.name == 'category':
                y = y_raw.cat.codes.values.astype(float)
            else:
                y = np.array(y_raw).astype(float)
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            N, P = X.shape
            if not (n_min <= N <= n_max) or P != expected_p:
                print(f"  {name}: shape mismatch ({N},{P}), skipping")
                continue
            loaded[name] = (X, y)
            print(f"  {name}:{' '*(12-len(name))}{X.shape}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")

    print(f"\n{len(loaded)} datasets ready.\n")
    return loaded


# ==========================================
# POLY PCA+OLS MODEL
# ==========================================
def train_poly_model(X_train, y_train, X_test,
                     n_components=200, d_model=64, ffn_dim=128,
                     epochs=50, batch_size=256, lr=1e-3,
                     dropout=0.1, weight_decay=1e-4,
                     seed=42, verbose=False):
    """
    Polynomial Cross-Feature Transformer with PCA compression + OLS warm-start.
    Replaces self-attention with explicit polynomial interaction features.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = DEVICE
    nf = X_train.shape[1]
    D = d_model

    def poly_cross(x):
        B, S, Dm = x.shape
        per = torch.cat([x, x**2 - 1], dim=-1)
        pairs = torch.cat([x[:, i, :] * x[:, j, :]
                           for i in range(S) for j in range(i, S)], dim=-1)
        return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)

    class FrozenPCA(nn.Module):
        def __init__(self, mean, components):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer('comp', torch.from_numpy(components.astype(np.float32)))
            self.out_dim = components.shape[0]
        def forward(self, x):
            return (x - self.mean) @ self.comp.T

    class PolyPCAModel(nn.Module):
        def __init__(self, nf, pca_layer, d, ffn_dim, drop):
            super().__init__()
            self.pca = pca_layer
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.randn(1, nf, d) * 0.02)
            self.W = nn.Linear(pca_layer.out_dim, d)
            self.n1 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, ffn_dim), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(ffn_dim, d), nn.Dropout(drop))
            self.n2 = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
        def embed_x(self, x):
            return self.embed(x.unsqueeze(-1)) + self.pos
        def get_poly(self, emb):
            return self.pca(poly_cross(emb))
        def forward(self, x):
            emb = self.embed_x(x)
            phi = self.get_poly(emb)
            z = self.n1(emb + self.W(phi))
            return self.head(self.n2(z + self.ffn(z)).mean(1))

    # Step 1: Fit PCA on training poly features
    torch.manual_seed(seed)
    tmp_embed = nn.Linear(1, D)
    tmp_pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
    Xt = torch.from_numpy(X_train.astype(np.float32))
    with torch.no_grad():
        emb = tmp_embed(Xt.unsqueeze(-1)) + tmp_pos
        phi_all = poly_cross(emb).reshape(-1, emb.shape[-1] * 2 + D * nf * (nf + 1) // 2)
    n_comp = min(n_components, phi_all.shape[1], phi_all.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(phi_all.numpy())
    pca_layer = FrozenPCA(pca.mean_, pca.components_)

    # Step 2: Build model
    torch.manual_seed(seed)
    model = PolyPCAModel(nf, pca_layer, D, ffn_dim, dropout).to(device)

    # Step 3: OLS warm-start
    model.eval()
    Xt_dev = Xt.to(device)
    with torch.no_grad():
        emb = model.embed_x(Xt_dev)
        phi_pooled = model.get_poly(emb).mean(1).cpu().numpy()
    reg = Ridge(alpha=1.0)
    reg.fit(phi_pooled, y_train)
    with torch.no_grad():
        model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
        model.W.bias.data.zero_()
        model.head.weight.data.zero_()
        model.head.weight.data[0, 0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

    # Step 4: Train
    yt = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    tr_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    # Step 5: Predict
    model.eval()
    Xte = torch.from_numpy(X_test.astype(np.float32)).to(device)
    with torch.no_grad():
        y_pred = model(Xte).cpu().numpy().ravel()

    info = {
        'n_params': sum(p.numel() for p in model.parameters()),
        'pca_components': n_comp,
    }
    return y_pred, info


# ==========================================
# BENCHMARK
# ==========================================
def run_benchmark():
    datasets = load_datasets()
    if not datasets:
        print("No datasets loaded.")
        return None

    models_active = [m for m, on in [
        ('OLS', RUN_OLS), ('RF', RUN_RF), ('Poly', RUN_POLY)] if on]

    print(f"Models: {', '.join(models_active)}")
    print(f"Repeats: {N_REPEATS}, Test: {TEST_FRAC}, Max N: {MAX_N}\n")

    results = []
    timings = []
    t0_global = time.time()

    for ds_name, (X_full, y_full) in datasets.items():
        N_full, P = X_full.shape
        if N_full > MAX_N:
            np.random.seed(SEED)
            idx = np.random.choice(N_full, MAX_N, replace=False)
            X_use, y_use = X_full[idx], y_full[idx]
            N_use = MAX_N
        else:
            X_use, y_use = X_full, y_full
            N_use = N_full

        print(f"{'='*75}")
        print(f"  {ds_name} (N={N_use}, P={P})")
        print(f"{'='*75}")

        scores = {k: [] for k in models_active}
        times  = {k: [] for k in models_active}

        for r in range(N_REPEATS):
            seed = SEED + r * 1000
            print(f"  Rep {r+1}/{N_REPEATS}...", end=" ", flush=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X_use, y_use, test_size=TEST_FRAC, random_state=seed)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train).astype(np.float32)
            X_te = scaler.transform(X_test).astype(np.float32)
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)

            if 'OLS' in models_active:
                t0 = time.time()
                m = LinearRegression().fit(X_tr, y_train)
                scores['OLS'].append(r2_score(y_test, m.predict(X_te)))
                times['OLS'].append(time.time() - t0)

            if 'RF' in models_active:
                t0 = time.time()
                m = RandomForestRegressor(n_estimators=RF_TREES, max_features=RF_MTRY,
                                          n_jobs=-1, random_state=seed).fit(X_tr, y_train)
                scores['RF'].append(r2_score(y_test, m.predict(X_te)))
                times['RF'].append(time.time() - t0)

            if 'Poly' in models_active:
                t0 = time.time()
                try:
                    y_pred, info = train_poly_model(
                        X_tr, y_train, X_te,
                        n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL,
                        ffn_dim=POLY_FFN_DIM, epochs=POLY_EPOCHS,
                        batch_size=POLY_BS, lr=POLY_LR,
                        dropout=POLY_DROPOUT, weight_decay=POLY_WD,
                        seed=seed, verbose=False)
                    scores['Poly'].append(r2_score(y_test, y_pred))
                except Exception as e:
                    print(f"[Poly:{e}]", end=" ")
                    scores['Poly'].append(np.nan)
                times['Poly'].append(time.time() - t0)

            print("done")

        # Report per dataset
        avg = {k: np.nanmean(v) for k, v in scores.items()}
        se  = {k: np.nanstd(v)  for k, v in scores.items()}
        avg_t = {k: np.mean(v) for k, v in times.items()}

        print(f"\n  {'Model':<8} {'R² (mean)':>10} {'± std':>8} {'Time/rep':>10}")
        print(f"  {'-'*40}")
        for m in models_active:
            print(f"  {m:<8} {avg[m]:>10.4f} {se[m]:>8.4f} {avg_t[m]:>9.1f}s")
        print()

        results.append({
            'Dataset': ds_name, 'N': N_use, 'P': P,
            **{m: avg[m] for m in models_active},
            **{f'{m}_se': se[m] for m in models_active},
            **{f'{m}_time': avg_t[m] for m in models_active},
        })

    elapsed = time.time() - t0_global

    # ── Grand summary ─────────────────────────────────────────────
    df = pd.DataFrame(results)

    print(f"\n{'='*90}")
    print(f"  RESULTS ({elapsed/60:.1f} min total, {N_REPEATS} reps, device={DEVICE})")
    print(f"{'='*90}")

    header = f"  {'Dataset':<12} {'N':>5} {'P':>3}"
    for m in models_active:
        header += f"  {'R²_'+m:>9}"
    for m in models_active:
        header += f"  {'t_'+m:>7}"
    print(header)
    print(f"  {'-'*85}")

    for _, row in df.iterrows():
        vals = {m: row[m] for m in models_active}
        best_val = max(vals.values())
        line = f"  {row['Dataset']:<12} {int(row['N']):>5} {int(row['P']):>3}"
        for m in models_active:
            v = row[m]
            marker = " *" if abs(v - best_val) < 0.001 else "  "
            line += f"  {v:>7.4f}{marker}"
        for m in models_active:
            line += f"  {row[f'{m}_time']:>6.1f}s"
        print(line)

    print(f"  {'-'*85}")
    print(f"  (* = best or within 0.001 of best)\n")

    # Win counts
    print("  Win counts:")
    for m in models_active:
        wins = sum(1 for _, r in df.iterrows()
                   if r[m] == max(r[c] for c in models_active))
        print(f"    {m}: {wins}/{len(df)}")

    # Average rank
    print("\n  Average rank (1=best):")
    ranks = df[models_active].rank(axis=1, ascending=False)
    for m in models_active:
        print(f"    {m}: {ranks[m].mean():.2f}")

    print(f"\n  Total wall-clock: {elapsed/60:.1f} min")

    return df


# ==========================================
# RUN
# ==========================================
df = run_benchmark()
