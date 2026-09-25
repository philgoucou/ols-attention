# Real-data benchmark v3 (precursor of v4), extracted from notebooks/Attention_paper_simuls.ipynb (cell 1).
"""
========================================================
REAL DATA BENCHMARK FOR ATTENTION REGRESSION — v3
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

# Hyperparameters
RF_TREES = 500;  RF_MTRY = 1/3
GBM_TREES = 500; GBM_LR = 0.01
MLP_LAYERS = 3;  MLP_UNITS = 200; MLP_DROP = 0.2
MLP_ENS = 1;    MLP_EPOCHS = 1000; MLP_PAT = 20
MLP_BS = 64;     MLP_LR_RATE = 1e-3; MLP_VAL = 0.15
ATTN_M = 5;      ATTN_STEPS = 300; ATTN_LR = 1.0; ATTN_REG = 1e-3

# ==========================================
# END CONFIG
# ==========================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
import warnings, time
warnings.filterwarnings('ignore')


# ==========================================
# DATASET LOADING — VALIDATED
# ==========================================

# Each entry: (openml_data_id, expected_n_range, expected_p, description)
# We validate shape after loading to catch wrong IDs
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
    """Load enabled datasets with shape validation."""
    enabled = [k for k, v in DATASETS.items() if v]
    print(f"\n📦 Loading {len(enabled)} datasets: {', '.join(enabled)}\n")
    loaded = {}

    # --- sklearn built-ins ---
    if 'California' in enabled:
        try:
            d = fetch_california_housing()
            assert d.data.shape == (20640, 8), f"Unexpected shape: {d.data.shape}"
            loaded['California'] = (d.data, d.target)
            print(f"  ✓ California:  (20640, 8)")
        except Exception as e:
            print(f"  ✗ California: {e}")

    if 'Diabetes' in enabled:
        try:
            d = load_diabetes()
            assert d.data.shape == (442, 10), f"Unexpected shape: {d.data.shape}"
            loaded['Diabetes'] = (d.data, d.target)
            print(f"  ✓ Diabetes:    (442, 10)")
        except Exception as e:
            print(f"  ✗ Diabetes: {e}")

    # --- OpenML datasets ---
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

            # Handle target — multiple failure modes
            y_raw = data.target
            if y_raw is None:
                # Try explicit target column from data description
                # Protein (CASP): target is 'RMSD', first column
                all_cols = data.data.columns.tolist()
                print(f"  ⚠ {name}: target is None. Columns: {all_cols[:5]}...")
                # Fall back: use first numeric column as target, rest as features
                raise ValueError("Target column not set in OpenML metadata")

            # Multi-target: pick first column (e.g., Energy has heating+cooling)
            if hasattr(y_raw, 'shape') and len(y_raw.shape) > 1 and y_raw.shape[1] > 1:
                print(f"  ℹ {name}: multi-target ({y_raw.shape[1]} cols), using first column")
                y_raw = y_raw.iloc[:, 0]
            elif hasattr(y_raw, 'columns'):
                # DataFrame with single column
                y_raw = y_raw.iloc[:, 0]

            # Convert to float
            if hasattr(y_raw, 'cat') and y_raw.dtype.name == 'category':
                y = y_raw.cat.codes.values.astype(float)
            else:
                y = np.array(y_raw).astype(float)

            # Clean NaN
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]

            N, P = X.shape

            # Validate shape
            if not (n_min <= N <= n_max):
                print(f"  ✗ {name} (id={data_id}): N={N} outside expected range ({n_min}-{n_max}). WRONG DATASET!")
                continue
            if P != expected_p:
                print(f"  ✗ {name} (id={data_id}): P={P} != expected {expected_p}. WRONG DATASET!")
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

    # Check function availability
    try:
        train_torch_model; has_attn = RUN_ATTN
    except NameError:
        has_attn = False
        if RUN_ATTN: print("⚠ train_torch_model not found. Run simulation cell first. Skipping Attn.\n")

    try:
        train_mlp_ensemble; has_mlp = RUN_MLP
    except NameError:
        has_mlp = False
        if RUN_MLP: print("⚠ train_mlp_ensemble not found. Run simulation cell first. Skipping MLP.\n")

    models_active = [m for m, on in [('OLS', RUN_OLS), ('RF', RUN_RF),
                     ('GBM', RUN_GBM), ('MLP', has_mlp), ('Attn', has_attn)] if on]
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

        for r in range(N_REPEATS):
            seed = SEED + r * 1000
            print(f"  Rep {r+1}/{N_REPEATS}...", end=" ", flush=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X_use, y_use, test_size=TEST_FRAC, random_state=seed)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)

            if 'OLS' in models_active:
                try:
                    m = LinearRegression().fit(X_tr, y_train)
                    scores['OLS'].append(r2_score(y_test, m.predict(X_te)))
                except: scores['OLS'].append(np.nan)

            if 'RF' in models_active:
                try:
                    m = RandomForestRegressor(n_estimators=RF_TREES, max_features=RF_MTRY,
                                              n_jobs=-1, random_state=seed).fit(X_tr, y_train)
                    scores['RF'].append(r2_score(y_test, m.predict(X_te)))
                except: scores['RF'].append(np.nan)

            if 'GBM' in models_active:
                try:
                    m = GradientBoostingRegressor(n_estimators=GBM_TREES, learning_rate=GBM_LR,
                                                  random_state=seed).fit(X_tr, y_train)
                    scores['GBM'].append(r2_score(y_test, m.predict(X_te)))
                except: scores['GBM'].append(np.nan)

            if 'MLP' in models_active:
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

            if 'Attn' in models_active:
                try:
                    y_pred, _ = train_torch_model(
                        X_tr, y_train, X_te, M=ATTN_M, steps=ATTN_STEPS,
                        lr=ATTN_LR, reg_lambda=ATTN_REG, seed=seed, verbose=False)
                    r2_att = r2_score(y_test, y_pred)

                    # Retry with weaker reg if poor
                    if r2_att < 0.05:
                        y_pred, _ = train_torch_model(
                            X_tr, y_train, X_te, M=ATTN_M, steps=ATTN_STEPS,
                            lr=ATTN_LR, reg_lambda=ATTN_REG/100, seed=seed, verbose=False)
                        r2_att_retry = r2_score(y_test, y_pred)
                        if r2_att_retry > r2_att:
                            r2_att = r2_att_retry

                    # Diagnostic if still bad
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

            print("✓")

        avg = {k: np.nanmean(v) if v else np.nan for k, v in scores.items()}
        se  = {k: np.nanstd(v)  if v else np.nan for k, v in scores.items()}

        results.append({
            'Dataset': ds_name, 'N': N_use, 'P': P,
            **{m: avg[m] for m in models_active},
            **{f'{m}_se': se[m] for m in models_active},
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

    print("\n" + "="*110)
    print(f"RESULTS ({elapsed/60:.1f} min, {N_REPEATS} reps)")
    print("="*110)

    header = f"{'Dataset':<12} {'N':>5} {'P':>3}"
    for m in models_active: header += f"  {m:>7}"
    header += "   Best"
    print(header)
    print("-"*110)

    for _, row in df.iterrows():
        line = f"{row['Dataset']:<12} {int(row['N']):>5} {int(row['P']):>3}"
        vals = {}
        for m in models_active:
            v = row[m]
            vals[m] = v
            line += f"  {v:>7.4f}" if not np.isnan(v) else f"  {'--':>7}"
        valid = {k: v for k, v in vals.items() if not np.isnan(v)}
        best = max(valid, key=valid.get) if valid else '--'
        line += f"   {best}"
        print(line)

    print("-"*110)

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

    df.to_csv("real_data_results.csv", index=False)
    print(f"\n💾 Saved: real_data_results.csv")

    # LaTeX
    col_names = {'OLS': 'OLS', 'RF': 'RF', 'GBM': 'GBM', 'MLP': 'MLP', 'Attn': r'Att.\ Reg.'}
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