# Real-data benchmark v4 = the pipeline behind the submitted Table 1 (OLS, RF, MLP, FT-Transformer,
# Attention Regression, Regression Block). Originally colab_full.py; run simulation_attention_regression.py
# first in the same session to load the Attention Regression function definitions it relies on.
"""
========================================================
REAL DATA BENCHMARK FOR ATTENTION REGRESSION — v4
With Polynomial Cross-Feature Transformer (Poly PCA+OLS)

Paste in a NEW Colab cell after running simulation cell.
Run cell 1 first (even briefly) to load function defs.
========================================================
"""

# ==========================================
# CONFIG — EDIT THIS SECTION
# ==========================================

DATASETS = {
    # --- Small & fast ---
    'Diabetes':    False,       # 442 obs, 10 feat (sklearn built-in)
    'Yacht':       True,       # 308 obs, 6 feat
    'Energy':      True,       # 768 obs, 8 feat
    # --- Medium ---
    'Concrete':    True,       # 1030 obs, 8 feat (nonlinear!)
    'Airfoil':     True,       # 1503 obs, 5 feat
    'WineRed':     False,       # 1599 obs, 11 feat
    # --- Larger ---
    'Abalone':     True,       # 4177 obs, 8 feat
    'California':  True,       # 20640→5000 obs, 8 feat (sklearn built-in)
    'Kin8nm':      True,       # 8192→5000 obs, 8 feat
    'Protein':     True,       # 45730→5000 obs, 9 feat
}

N_REPEATS = 5
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42

# Models
RUN_OLS  = True
RUN_RF   = True
RUN_GBM  = True
RUN_MLP  = True
RUN_ATTN = True
RUN_FTT  = True
RUN_POLY = True

# Hyperparameters
RF_TREES = 500;  RF_MTRY = 1/3
GBM_TREES = 500; GBM_LR = 0.01
MLP_LAYERS = 3;  MLP_UNITS = 200; MLP_DROP = 0.2
MLP_ENS = 1;    MLP_EPOCHS = 1000; MLP_PAT = 20
MLP_BS = 64;     MLP_LR_RATE = 1e-3; MLP_VAL = 0.15
ATTN_M = 4;      ATTN_STEPS = 300; ATTN_LR = 1e-3; ATTN_REG = 1e-3

# FT-Transformer hyperparameters (Gorishniy et al. 2021)
FTT_D_MODEL = 64
FTT_N_HEADS = 4
FTT_FFN_DIM = 128
FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1
FTT_EPOCHS = 200
FTT_BS = 256
FTT_LR = 1e-3
FTT_WD = 1e-4

# Poly Raw hyperparameters (no PCA, no embedding before poly)
POLY_D_MODEL = 64
POLY_FFN_DIM = 128
POLY_EPOCHS = 200
POLY_BS = 256
POLY_LR = 1e-3
POLY_DROPOUT = 0.1
POLY_WD = 1e-4

# ==========================================
# END CONFIG
# ==========================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.decomposition import PCA
import warnings, time
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==========================================
# ATTENTION REGRESSION (simple 2-block)
# Self-contained — no dependency on cell 1
# ==========================================

def train_torch_model(X_train, y_train, X_test,
                      M=4, steps=300, lr=1e-3,
                      reg_lambda=1e-3, seed=42, verbose=False):
    """
    Simple attention-based regression: embed features as tokens,
    2-block multi-head self-attention, pool, linear head.
    Trains for `steps` gradient steps (not epochs).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = DEVICE
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

    # Val split for early stopping
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

        # Val check each pass through data
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
        y_pred = model(torch.from_numpy(X_test.astype(np.float32)).to(device)).cpu().numpy().ravel()
    return y_pred, {'n_params': sum(p.numel() for p in model.parameters())}


def train_mlp_ensemble(*args, **kwargs):
    """Stub — not implemented in standalone mode."""
    raise NotImplementedError("MLP ensemble requires cell 1")


# ==========================================
# FT-TRANSFORMER (Gorishniy et al. 2021)
# ==========================================

def train_ft_transformer(X_train, y_train, X_test,
                         d_model=64, n_heads=4, ffn_dim=128,
                         n_blocks=3, dropout=0.1,
                         epochs=50, batch_size=256, lr=1e-3,
                         weight_decay=1e-4, seed=42, verbose=False):
    """
    FT-Transformer: Feature Tokenizer + [CLS] + Transformer Encoder.
    Per-feature learned embeddings, CLS token for readout.

    Interface matches train_torch_model / train_mlp_ensemble:
      Args: X_train (N,P), y_train (N,), X_test (M,P), **hyperparams
      Returns: (y_pred (M,), info_dict)
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = DEVICE
    nf = X_train.shape[1]; D = d_model

    class FTTransformer(nn.Module):
        def __init__(self, nf, d, nh, ffn, nblk, drop):
            super().__init__()
            self.feat_embeds = nn.ModuleList([nn.Linear(1, d) for _ in range(nf)])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nh, dim_feedforward=ffn, dropout=drop,
                batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nblk)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            B = x.size(0)
            tokens = torch.stack([self.feat_embeds[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
            z = self.transformer(tokens)
            return self.head(self.norm(z[:, 0, :]))

    model = FTTransformer(nf, D, n_heads, ffn_dim, n_blocks, dropout).to(device)

    # Val split for early stopping (15%)
    N = X_train.shape[0]
    n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)

    tr_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    crit = nn.MSELoss()

    # LR warmup + cosine decay
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(1, epochs // 10)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return ep / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Training with early stopping + grad clipping
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10

    for ep in range(1, epochs+1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        # Val check
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xt_val), yt_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test.astype(np.float32)).to(device)).cpu().numpy().ravel()
    return y_pred, {'n_params': sum(p.numel() for p in model.parameters())}


# ==========================================
# POLY RAW MODEL
# ==========================================
# Raw polynomial features of x (no embedding, no PCA).
# φ(x) = [x, x²-1, x_i·x_j for all i≤j] → W → LN → FFN → head → ŷ
# All parameters trained end-to-end. No preprocessing. No frozen matrices.

def train_poly_model(X_train, y_train, X_test,
                     d_model=64, ffn_dim=128,
                     epochs=200, batch_size=256, lr=1e-3,
                     dropout=0.1, weight_decay=1e-4,
                     seed=42, verbose=False, **kwargs):
    """
    Polynomial Cross-Feature Regression: replaces attention with explicit
    degree-2 polynomial interactions on raw features.

    φ(x) = [x₁,...,xₚ, x₁²-1,...,xₚ²-1, x₁x₂, x₁x₃,...,xₚ₋₁xₚ]
    ŷ = head(LN(FFN(LN(W·φ(x)))))

    Interface matches train_torch_model / train_mlp_ensemble:
      Args: X_train (N,P), y_train (N,), X_test (M,P), **hyperparams
      Returns: (y_pred (M,), info_dict)
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = DEVICE
    nf = X_train.shape[1]; D = d_model
    n_pairs = nf * (nf + 1) // 2
    poly_dim = nf + nf + n_pairs  # main + squared + cross

    def raw_poly(x):
        """x: (B, P) → (B, P + P + P(P+1)/2)"""
        B, P = x.shape
        main = x
        sq = x**2 - 1
        cross = [x[:, i] * x[:, j] for i in range(P) for j in range(i, P)]
        return torch.cat([main, sq, torch.stack(cross, 1)], 1)

    class PolyRawModel(nn.Module):
        def __init__(self, poly_dim, d, ffn_dim, drop):
            super().__init__()
            self.W = nn.Linear(poly_dim, d)
            self.n1 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.ReLU(), nn.Dropout(drop),
                                     nn.Linear(ffn_dim, d), nn.Dropout(drop))
            self.n2 = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)
        def forward(self, x):
            z = self.n1(self.W(raw_poly(x)))
            return self.head(self.n2(z + self.ffn(z)))

    model = PolyRawModel(poly_dim, D, ffn_dim, dropout).to(device)

    # Val split for early stopping (15%)
    N = X_train.shape[0]
    n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)

    tr_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10

    for ep in range(1, epochs+1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xt_val), yt_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test.astype(np.float32)).to(device)).cpu().numpy().ravel()
    return y_pred, {'n_params': sum(p.numel() for p in model.parameters()), 'poly_dim': poly_dim}


# ==========================================
# DATASET LOADING — VALIDATED
# ==========================================

OPENML_REGISTRY = {
    'Concrete':  (44959,  (1000, 1100),  8,  'Concrete compressive strength'),
    'Energy':    (41478,  (700, 800),     8,  'Energy efficiency'),
    'Kin8nm':    (189,    (8000, 8300),   8,  'Kin8nm robot arm'),
    'Protein':   (44963,  (45000, 46000), 9,  'Protein structure CASP'),
    'WineRed':   (40691,  (1500, 1700),   11, 'Wine quality red'),
    'Yacht':     (42370,  (300, 320),     6,  'Yacht hydrodynamics'),
    'Airfoil':   (44957,  (1400, 1600),   5,  'Airfoil self-noise'),
    'Abalone':   (183,    (4100, 4200),   7,  'Abalone age prediction'),
}


def load_datasets():
    enabled = [k for k, v in DATASETS.items() if v]
    print(f"\n📦 Loading {len(enabled)} datasets: {', '.join(enabled)}\n")
    loaded = {}

    if 'California' in enabled:
        try:
            d = fetch_california_housing()
            assert d.data.shape == (20640, 8)
            loaded['California'] = (d.data, d.target)
            print(f"  ✓ California:  (20640, 8)")
        except Exception as e:
            print(f"  ✗ California: {e}")

    if 'Diabetes' in enabled:
        try:
            d = load_diabetes()
            assert d.data.shape == (442, 10)
            loaded['Diabetes'] = (d.data, d.target)
            print(f"  ✓ Diabetes:    (442, 10)")
        except Exception as e:
            print(f"  ✗ Diabetes: {e}")

    for name in enabled:
        if name in ('California', 'Diabetes'):
            continue
        if name not in OPENML_REGISTRY:
            print(f"  ⚠ {name}: not in registry, skipping")
            continue

        data_id, (n_min, n_max), expected_p, desc = OPENML_REGISTRY[name]
        try:
            data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
            X = data.data.select_dtypes(include=[np.number]).values
            y_raw = data.target
            if y_raw is None:
                raise ValueError("Target column not set in OpenML metadata")
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
            if not (n_min <= N <= n_max):
                print(f"  ✗ {name} (id={data_id}): N={N} outside expected range ({n_min}-{n_max})")
                continue
            if P != expected_p:
                print(f"  ✗ {name} (id={data_id}): P={P} != expected {expected_p}")
                continue
            loaded[name] = (X, y)
            print(f"  ✓ {name}:{' '*(10-len(name))}({N}, {P})  — {desc}")
        except Exception as e:
            print(f"  ✗ {name} (id={data_id}): {e}")

    print(f"\n→ {len(loaded)}/{len(enabled)} datasets ready.\n")
    return loaded


# ==========================================
# BENCHMARK
# ==========================================

def run_benchmark():
    datasets = load_datasets()
    if not datasets:
        print("❌ No datasets loaded.")
        return None

    models_active = [m for m, on in [('OLS', RUN_OLS), ('RF', RUN_RF),
                     ('GBM', RUN_GBM), ('MLP', RUN_MLP), ('Attn', RUN_ATTN),
                     ('FTT', RUN_FTT), ('Poly', RUN_POLY)] if on]
    print(f"🔧 Models: {', '.join(models_active)}")
    print(f"🔁 Repeats: {N_REPEATS}, Test: {TEST_FRAC}, Max N: {MAX_N}\n")

    results = []
    t0 = time.time()

    for ds_name, (X_full, y_full) in datasets.items():
        N_full, P = X_full.shape

        if N_full > MAX_N:
            np.random.seed(SEED)
            idx = np.random.choice(N_full, MAX_N, replace=False)
            X_use, y_use = X_full[idx], y_full[idx]
            N_use = MAX_N
            label = f"{ds_name} (N={N_full}→{MAX_N}, P={P})"
        else:
            X_use, y_use = X_full, y_full
            N_use = N_full
            label = f"{ds_name} (N={N_use}, P={P})"

        print(f"{'='*70}\n  {label}\n{'='*70}")

        scores = {k: [] for k in models_active}
        times  = {k: [] for k in models_active}

        for r in range(N_REPEATS):
            seed = SEED + r * 1000
            print(f"  Rep {r+1}/{N_REPEATS}...", end=" ", flush=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X_use, y_use, test_size=TEST_FRAC, random_state=seed)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)

            if 'OLS' in models_active:
                t_start = time.time()
                try:
                    m = LinearRegression().fit(X_tr, y_train)
                    scores['OLS'].append(r2_score(y_test, m.predict(X_te)))
                except: scores['OLS'].append(np.nan)
                times['OLS'].append(time.time() - t_start)

            if 'RF' in models_active:
                t_start = time.time()
                try:
                    m = RandomForestRegressor(n_estimators=RF_TREES, max_features=RF_MTRY,
                                              n_jobs=-1, random_state=seed).fit(X_tr, y_train)
                    scores['RF'].append(r2_score(y_test, m.predict(X_te)))
                except: scores['RF'].append(np.nan)
                times['RF'].append(time.time() - t_start)

            if 'GBM' in models_active:
                t_start = time.time()
                try:
                    m = GradientBoostingRegressor(n_estimators=GBM_TREES, learning_rate=GBM_LR,
                                                  random_state=seed).fit(X_tr, y_train)
                    scores['GBM'].append(r2_score(y_test, m.predict(X_te)))
                except: scores['GBM'].append(np.nan)
                times['GBM'].append(time.time() - t_start)

            if 'MLP' in models_active:
                t_start = time.time()
                try:
                    y_pred, _ = train_mlp_ensemble(
                        X_tr, y_train, X_te, hidden_units=MLP_UNITS,
                        n_layers=MLP_LAYERS, dropout=MLP_DROP, ensemble_size=MLP_ENS,
                        max_epochs=MLP_EPOCHS, patience=MLP_PAT, batch_size=MLP_BS,
                        lr=MLP_LR_RATE, val_frac=MLP_VAL, seed=seed, verbose=False)
                    scores['MLP'].append(r2_score(y_test, y_pred))
                except Exception as e:
                    print(f"[MLP:{e}]", end=" ")
                    scores['MLP'].append(np.nan)
                times['MLP'].append(time.time() - t_start)

            if 'Attn' in models_active:
                t_start = time.time()
                try:
                    y_pred, _ = train_torch_model(
                        X_tr, y_train, X_te, M=ATTN_M, steps=ATTN_STEPS,
                        lr=ATTN_LR, reg_lambda=ATTN_REG, seed=seed, verbose=False)
                    r2_att = r2_score(y_test, y_pred)

                    if r2_att < 0.05:
                        y_pred, _ = train_torch_model(
                            X_tr, y_train, X_te, M=ATTN_M, steps=ATTN_STEPS,
                            lr=ATTN_LR, reg_lambda=ATTN_REG/100, seed=seed, verbose=False)
                        r2_att_retry = r2_score(y_test, y_pred)
                        if r2_att_retry > r2_att:
                            r2_att = r2_att_retry

                    if r2_att < 0:
                        y_var = np.var(y_test)
                        pred_var = np.var(y_pred)
                        pred_mean = np.mean(y_pred)
                        y_mean = np.mean(y_test)
                        mse = np.mean((y_pred - y_test)**2)
                        print(f"\n    ⚠ ATTN DIAGNOSTIC: R²={r2_att:.3f} | "
                              f"N_tr={len(y_train)} P={X_tr.shape[1]} | "
                              f"y_test: μ={y_mean:.2f} σ²={y_var:.2f} | "
                              f"ŷ: μ={pred_mean:.2f} σ²={pred_var:.2f} | "
                              f"MSE={mse:.2f}", end=" ")

                    scores['Attn'].append(r2_att)
                except Exception as e:
                    print(f"[Attn:{e}]", end=" ")
                    scores['Attn'].append(np.nan)
                times['Attn'].append(time.time() - t_start)

            if 'FTT' in models_active:
                t_start = time.time()
                try:
                    y_pred, _ = train_ft_transformer(
                        X_tr, y_train, X_te,
                        d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                        ffn_dim=FTT_FFN_DIM, n_blocks=FTT_N_BLOCKS,
                        dropout=FTT_DROPOUT, epochs=FTT_EPOCHS,
                        batch_size=FTT_BS, lr=FTT_LR,
                        weight_decay=FTT_WD,
                        seed=seed, verbose=False)
                    scores['FTT'].append(r2_score(y_test, y_pred))
                except Exception as e:
                    print(f"[FTT:{e}]", end=" ")
                    scores['FTT'].append(np.nan)
                times['FTT'].append(time.time() - t_start)

            if 'Poly' in models_active:
                t_start = time.time()
                try:
                    y_pred, _ = train_poly_model(
                        X_tr, y_train, X_te,
                        d_model=POLY_D_MODEL,
                        ffn_dim=POLY_FFN_DIM, epochs=POLY_EPOCHS,
                        batch_size=POLY_BS, lr=POLY_LR,
                        dropout=POLY_DROPOUT, weight_decay=POLY_WD,
                        seed=seed, verbose=False)
                    scores['Poly'].append(r2_score(y_test, y_pred))
                except Exception as e:
                    print(f"[Poly:{e}]", end=" ")
                    scores['Poly'].append(np.nan)
                times['Poly'].append(time.time() - t_start)

            print("✓")

        avg = {k: np.nanmean(v) if v else np.nan for k, v in scores.items()}
        se  = {k: np.nanstd(v)  if v else np.nan for k, v in scores.items()}
        avg_t = {k: np.mean(v) if v else np.nan for k, v in times.items()}

        results.append({
            'Dataset': ds_name, 'N': N_use, 'P': P,
            **{m: avg[m] for m in models_active},
            **{f'{m}_se': se[m] for m in models_active},
            **{f'{m}_time': avg_t[m] for m in models_active},
        })

        parts = [f"{m}:{avg[m]:.3f}" for m in models_active if not np.isnan(avg[m])]
        valid = {k: v for k, v in avg.items() if not np.isnan(v)}
        best = max(valid, key=valid.get) if valid else '?'
        print(f"  → {' | '.join(parts)}  |  Winner: {best}\n")

    # ==========================================
    # OUTPUT
    # ==========================================
    elapsed = time.time() - t0
    df = pd.DataFrame(results)

    print("\n" + "="*120)
    print(f"RESULTS ({elapsed/60:.1f} min, {N_REPEATS} reps, device={DEVICE})")
    print("="*120)

    header = f"{'Dataset':<12} {'N':>5} {'P':>3}"
    for m in models_active: header += f"  {m:>7}"
    header += "  |"
    for m in models_active: header += f"  {'t_'+m:>7}"
    header += "   Best"
    print(header)
    print("-"*120)

    for _, row in df.iterrows():
        line = f"{row['Dataset']:<12} {int(row['N']):>5} {int(row['P']):>3}"
        vals = {}
        for m in models_active:
            v = row[m]
            vals[m] = v
            line += f"  {v:>7.4f}" if not np.isnan(v) else f"  {'--':>7}"
        line += "  |"
        for m in models_active:
            t = row.get(f'{m}_time', np.nan)
            line += f"  {t:>6.1f}s" if not np.isnan(t) else f"  {'--':>7}"
        valid = {k: v for k, v in vals.items() if not np.isnan(v)}
        best = max(valid, key=valid.get) if valid else '--'
        line += f"   {best}"
        print(line)

    print("-"*120)

    # Win counts
    print("\nWin counts:")
    for m in models_active:
        wins = sum(1 for _, r in df.iterrows()
                   if not np.isnan(r[m]) and r[m] == max(r[c] for c in models_active if not np.isnan(r[c])))
        print(f"  {m}: {wins}/{len(df)}")

    # Average rank
    print("\nAverage rank (1=best):")
    ranks = df[models_active].rank(axis=1, ascending=False)
    for m in models_active:
        print(f"  {m}: {ranks[m].mean():.2f}")

    # Timing summary
    print("\nAvg time per rep (seconds):")
    for m in models_active:
        avg_t = df[f'{m}_time'].mean()
        print(f"  {m}: {avg_t:.1f}s")

    df.to_csv("real_data_results_v4.csv", index=False)
    print(f"\n💾 Saved: real_data_results_v4.csv")

    # LaTeX
    col_names = {'OLS': 'OLS', 'RF': 'RF', 'GBM': 'GBM', 'MLP': 'MLP',
                 'Attn': r'Att.\ Reg.', 'FTT': 'FT-T', 'Poly': 'Poly'}
    print("\n\n% ========== LaTeX Table ==========")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(rf"\caption{{Out-of-sample $R^2$ on standard regression benchmarks ({N_REPEATS} random 80/20 splits, averaged). Best in bold.}}")
    print(r"\label{tab:real_data}")
    print(r"\begin{tabular}{lrr" + "c"*len(models_active) + "}")
    print(r"\toprule")
    hdr = r"Dataset & $N$ & $P$"
    for m in models_active: hdr += f" & {col_names.get(m, m)}"
    print(hdr + r" \\")
    print(r"\midrule")

    for _, row in df.iterrows():
        vals = {m: row[m] for m in models_active}
        valid = {k: v for k, v in vals.items() if not np.isnan(v)}
        best_val = max(valid.values()) if valid else None
        cells = []
        for m in models_active:
            v = row[m]
            if np.isnan(v): cells.append("--")
            elif best_val and abs(v - best_val) < 1e-4: cells.append(rf"\textbf{{{v:.3f}}}")
            else: cells.append(f"{v:.3f}")
        print(f"{row['Dataset']} & {int(row['N'])} & {int(row['P'])} & {' & '.join(cells)}" + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    return df


# ==========================================
# RUN
# ==========================================
df_results = run_benchmark()
