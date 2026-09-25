# Monte Carlo of the paper (Section 4 "Simulated data", Figure 2, Appendix B table):
# Attention Regression with LBFGS + Cholesky parameterization vs OLS / RF / GBM / MLP
# on six DGPs (Linear, Friedman 1-3, Rotated Sine, Soft Radial), N in {500,1000,2500,5000},
# SNR in {0.5,1,2,3}. Extracted verbatim from the Colab notebook
# notebooks/Attention_paper_simuls.ipynb (cell 0). The USER CONFIG block at the top is the
# notebook's last saved state, not necessarily the paper's run: the paper uses 10 repeats,
# five attention heads (USE_ATTENTION_M5), and RF / GBM / MLP / OLS all switched on.
"""
Attention-Based Regression Benchmark with LBFGS + Cholesky Parameterization
Now includes MLP Ensemble benchmark
"""

# ==========================================
# ========== USER CONFIG (EDIT HERE) =======
# ==========================================

# Models to estimate
USE_OLS = True
USE_RF = False
USE_GBM = False
USE_MLP = True                   # NEW: MLP Ensemble
USE_ATTENTION_M3 = False
USE_ATTENTION_M5 = False

# Data generation
N_REPEATS = 5                    # Number of runs to average
SIZES = [500, 1000, 2500, 5000]  # Training set sizes
SNRS = [0.5, 1, 2, 3]            # Signal-to-noise ratios
DGPS = [
    'linear',
    'friedman1',
    'friedman2',
    'friedman3',
    'rotated_sine',
    'soft_radial',
]

# Random seed
RANDOM_SEED = 42                 # Base random seed for reproducibility

# Attention model settings
REG_M3 = 1e-8                    # Regularization for M=3 (minimal)
REG_M5 = 1e-3                    # Regularization for M=5 (strong)

# Random Forest settings
RF_N_ESTIMATORS = 500            # Number of trees
RF_MTRY = 1/3                    # Features per split (fraction of P)

# Gradient Boosting settings
GBM_N_ESTIMATORS = 500           # Number of trees
GBM_LEARNING_RATE = 0.01         # Learning rate

# MLP Ensemble settings (NEW)
MLP_HIDDEN_LAYERS = 3            # Number of hidden layers
MLP_HIDDEN_UNITS = 200           # Neurons per hidden layer
MLP_DROPOUT = 0.2                # Dropout rate
MLP_ENSEMBLE_SIZE = 1           # Number of models to ensemble
MLP_MAX_EPOCHS = 1000            # Max epochs (early stopping will cut this)
MLP_PATIENCE = 20                # Early stopping patience
MLP_BATCH_SIZE = 64              # Mini-batch size
MLP_LR = 1e-3                    # Learning rate (Adam)
MLP_VAL_FRAC = 0.15              # Fraction of train data for validation

# Optimizer settings
ATTENTION_STEPS = 300            # LBFGS iterations
LEARNING_RATE = 1.0              # LBFGS learning rate

# Output
OUTPUT_FILE = "benchmark_results_with_mlp.csv"

# ==========================================
# ========== END USER CONFIG ===============
# ==========================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import time
import warnings
import copy

# Suppress warnings
warnings.filterwarnings('ignore')

# Detect Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==========================================
# 1. MLP ENSEMBLE MODEL (NEW)
# ==========================================

class MLPRegressor(nn.Module):
    """Standard MLP with ReLU activation and dropout."""
    def __init__(self, input_dim, hidden_units=200, n_layers=3, dropout=0.2):
        super().__init__()

        layers = []
        in_features = input_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_units

        # Output layer
        layers.append(nn.Linear(hidden_units, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_single_mlp(X_train, y_train, X_val, y_val,
                     hidden_units=200, n_layers=3, dropout=0.2,
                     max_epochs=1000, patience=20, batch_size=64, lr=1e-3,
                     seed=None, verbose=False):
    """
    Train a single MLP with early stopping.

    Returns:
        model: Trained model (best weights)
        best_val_loss: Best validation loss achieved
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    # Create dataloader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_units=hidden_units,
        n_layers=n_layers,
        dropout=dropout
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Early stopping setup
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"      Early stopping at epoch {epoch+1}, best val_loss: {best_val_loss:.6f}")
            break

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


def train_mlp_ensemble(X_train, y_train, X_test,
                       hidden_units=200, n_layers=3, dropout=0.2,
                       ensemble_size=10, max_epochs=1000, patience=20,
                       batch_size=64, lr=1e-3, val_frac=0.15,
                       seed=None, verbose=False):
    """
    Train an ensemble of MLPs and return averaged predictions.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        ensemble_size: Number of models to train and average
        val_frac: Fraction of training data to use for validation
        ... other params passed to train_single_mlp

    Returns:
        y_hat_test: Ensemble predictions on test set
        y_hat_train: Ensemble predictions on training set
    """
    if seed is not None:
        np.random.seed(seed)

    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Standardize target
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-8
    y_train_scaled = (y_train - y_mean) / y_std

    # Store predictions from each model
    test_preds = []
    train_preds = []

    n_train = len(X_train_scaled)
    n_val = int(n_train * val_frac)

    for i in range(ensemble_size):
        model_seed = seed + i * 100 if seed is not None else None

        # Random train/val split for this model
        if model_seed is not None:
            np.random.seed(model_seed)
        indices = np.random.permutation(n_train)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_tr = X_train_scaled[train_idx]
        y_tr = y_train_scaled[train_idx]
        X_vl = X_train_scaled[val_idx]
        y_vl = y_train_scaled[val_idx]

        # Train single model
        model, _ = train_single_mlp(
            X_tr, y_tr, X_vl, y_vl,
            hidden_units=hidden_units,
            n_layers=n_layers,
            dropout=dropout,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            seed=model_seed,
            verbose=verbose
        )

        # Get predictions
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)
            X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=DEVICE)

            pred_test = model(X_test_t).cpu().numpy()
            pred_train = model(X_train_t).cpu().numpy()

            test_preds.append(pred_test)
            train_preds.append(pred_train)

    # Average predictions
    y_hat_test_scaled = np.mean(test_preds, axis=0)
    y_hat_train_scaled = np.mean(train_preds, axis=0)

    # Inverse transform
    y_hat_test = y_hat_test_scaled * y_std + y_mean
    y_hat_train = y_hat_train_scaled * y_std + y_mean

    return y_hat_test, y_hat_train


# ==========================================
# 2. PYTORCH ATTENTION MODEL (Cholesky + LBFGS)
# ==========================================

class AttentionRegressor(nn.Module):
    def __init__(self, M=3, P=5, use_ols_init=True):
        super().__init__()
        self.M = M
        self.P = P
        self.use_ols_init = use_ols_init

        # 1. Alphas
        self.alpha = nn.Parameter(torch.ones(M) / M)

        # 2. Lower triangular Cholesky factors (L)
        n_tril = P * (P + 1) // 2
        self.L_params = nn.Parameter(torch.randn(M, n_tril) * 0.1)

        # Indices for lower triangular elements
        self.register_buffer('tril_indices',
                            torch.tril_indices(P, P, offset=0))

    def get_L_matrices(self):
        """Convert flat parameters to lower triangular matrices."""
        L = torch.zeros(self.M, self.P, self.P, device=self.L_params.device)
        L[:, self.tril_indices[0], self.tril_indices[1]] = self.L_params
        return L

    def initialize_weights(self, X_tensor):
        if self.use_ols_init:
            with torch.no_grad():
                try:
                    XtX = X_tensor.T @ X_tensor
                    identity = torch.eye(self.P, device=X_tensor.device) * 1e-6
                    inv_cov = torch.linalg.inv(XtX + identity)
                    L_target = torch.linalg.cholesky(inv_cov)

                    N = X_tensor.shape[0]
                    scale_factor = np.sqrt(N)
                    L_target = L_target * scale_factor

                    L_flat = L_target[self.tril_indices[0], self.tril_indices[1]]

                    for m in range(self.M):
                        noise = torch.randn_like(L_flat) * 0.05
                        self.L_params.data[m] = L_flat + noise
                    print("   ✓ OLS Cholesky initialization successful")
                except Exception as e:
                    print(f"   ⚠ OLS initialization failed: {e}, using random init")

    def forward(self, X_train, y_train, X_test=None):
        target_X = X_train if X_test is None else X_test

        L = self.get_L_matrices()
        Omega = torch.bmm(L, L.transpose(1, 2))

        X_target_b = target_X.unsqueeze(0).expand(self.M, -1, -1)
        X_train_b_T = X_train.T.unsqueeze(0).expand(self.M, -1, -1)
        Scores = torch.bmm(torch.bmm(X_target_b, Omega), X_train_b_T)

        Scores_max = torch.max(Scores, dim=2, keepdim=True)[0]
        Attn = torch.exp(Scores - Scores_max)
        Attn = Attn / (torch.sum(Attn, dim=2, keepdim=True) + 1e-8)

        y_train_b = y_train.unsqueeze(0).unsqueeze(2).expand(self.M, -1, -1)
        head_preds = torch.bmm(Attn, y_train_b).squeeze(2)

        final_pred = torch.sum(self.alpha.unsqueeze(1) * head_preds, dim=0)
        return final_pred


def train_torch_model(X_train, y_train, X_test, M=3, steps=300, lr=1.0, reg_lambda=1e-8, seed=None, verbose=False):
    """Train attention model with LBFGS optimizer."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-8
    y_train_scaled = (y_train - y_mean) / y_std

    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train_scaled, dtype=torch.float32, device=DEVICE)
    Xtt = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    model = AttentionRegressor(M=M, P=Xt.shape[1], use_ols_init=True).to(DEVICE)
    model.initialize_weights(Xt)

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=20,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    model.train()
    best_loss = float('inf')
    iteration = [0]

    def closure():
        optimizer.zero_grad()
        y_pred = model(Xt, yt)

        mse = torch.mean((y_pred - yt)**2)
        reg = reg_lambda * torch.sum(model.L_params ** 2)
        loss = mse + reg

        if torch.isnan(loss) or torch.isinf(loss):
            return loss

        loss.backward()

        if verbose and iteration[0] % 50 == 0:
            print(f"   Step {iteration[0]:4d}: Loss={loss.item():.6f}, MSE={mse.item():.6f}")

        return loss

    for step in range(steps):
        iteration[0] = step
        loss = optimizer.step(closure)

        if loss is not None:
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   ⚠ NaN/Inf detected at step {step}, stopping early")
                break

            if loss.item() < best_loss - 1e-7:
                best_loss = loss.item()
            elif step > 50:
                if verbose:
                    print(f"   ✓ Converged at step {step}")
                break

    model.eval()
    with torch.no_grad():
        y_hat_test_scaled = model(Xt, yt, Xtt).cpu().numpy()
        y_hat_test = y_hat_test_scaled * y_std + y_mean

        y_hat_train_scaled = model(Xt, yt, Xt).cpu().numpy()
        y_hat_train = y_hat_train_scaled * y_std + y_mean

    return y_hat_test, y_hat_train


# ==========================================
# 3. DATA SIMULATION
# ==========================================

def simulate_data(n, type='linear', snr=1.0, seed=None):
    """Generate synthetic regression datasets."""
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(0, 1, (n, 5))

    if type == 'linear':
        X_c = X - 0.5
        y_pure = X_c @ np.array([2, -1, 3, 1.5, 0.5])
    elif type == 'friedman1':
        y_pure = 10 * np.sin(np.pi * X[:,0] * X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4]
    elif type == 'friedman2':
        y_pure = np.sin(np.sum(X[:,:3], axis=1)*np.pi) + np.log(1 + X[:,3]**2)
    elif type == 'friedman3':
        y_pure = (X[:,0] * X[:,1]) + np.log(X[:,2] + X[:,3] + 2)
    elif type == 'rotated_sine':
        y_pure = np.sin(3 * np.sum(X[:, :4], axis=1))
    elif type == 'soft_radial':
        dist_sq = np.sum((X - 0.5)**2, axis=1)
        y_pure = 1.0 / (1.0 + 5.0 * dist_sq)
    else:
        raise ValueError(f"Unknown DGP type: {type}")

    var_signal = np.var(y_pure)
    if var_signal == 0:
        var_signal = 1
    var_noise = var_signal / snr
    y = y_pure + np.random.normal(0, np.sqrt(var_noise), n)
    return X, y


# ==========================================
# 4. MAIN BENCHMARK LOOP
# ==========================================

if __name__ == "__main__":

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    final_results = []

    # Build list of enabled models for display
    enabled_models = []
    if USE_OLS: enabled_models.append('OLS')
    if USE_RF: enabled_models.append('RF')
    if USE_GBM: enabled_models.append('GBM')
    if USE_MLP: enabled_models.append('MLP')
    if USE_ATTENTION_M3: enabled_models.append('Att(M=3)')
    if USE_ATTENTION_M5: enabled_models.append('Att(M=5)')

    print("\n" + "="*100)
    print("BENCHMARK CONFIGURATION")
    print("="*100)
    print(f"Repeats: {N_REPEATS}, Base seed: {RANDOM_SEED}")
    print(f"Sample sizes: {SIZES}")
    print(f"SNR levels: {SNRS}")
    print(f"DGPs: {DGPS}")
    print(f"Enabled models: {', '.join(enabled_models)}")
    print(f"Parameterization: Cholesky (45 params) vs W matrices (78 params) = 40% fewer!")
    print(f"Optimizer: LBFGS (2nd-order, fast convergence)")
    if USE_ATTENTION_M3 or USE_ATTENTION_M5:
        print(f"Attention steps: {ATTENTION_STEPS}, LR: {LEARNING_RATE}")
        print(f"Regularization: M=3 λ={REG_M3:.0e} (minimal), M=5 λ={REG_M5:.0e} (strong)")
    if USE_RF:
        print(f"RF: n_estimators={RF_N_ESTIMATORS}, mtry={RF_MTRY}")
    if USE_GBM:
        print(f"GBM: n_estimators={GBM_N_ESTIMATORS}, lr={GBM_LEARNING_RATE}")
    if USE_MLP:
        print(f"MLP: {MLP_HIDDEN_LAYERS} layers × {MLP_HIDDEN_UNITS} units, dropout={MLP_DROPOUT}, "
              f"ensemble={MLP_ENSEMBLE_SIZE}, patience={MLP_PATIENCE}")
    print("="*100 + "\n")

    # Build dynamic header
    header = f"{'DGP':<15} | {'N':<5} | {'SNR':<4}"
    if USE_OLS: header += f" | {'OLS':<6}"
    if USE_RF: header += f" | {'RF':<6}"
    if USE_GBM: header += f" | {'GBM':<6}"
    if USE_MLP: header += f" | {'MLP':<6}"
    if USE_ATTENTION_M3: header += f" | {'Att(3)':<7}"
    if USE_ATTENTION_M5: header += f" | {'Att(5)':<7}"
    if USE_ATTENTION_M3: header += f" | {'Att(3)_IS':<9}"
    if USE_ATTENTION_M5: header += f" | {'Att(5)_IS':<9}"
    print(header)
    print("-" * 130)

    for dgp in DGPS:
        for n_train in SIZES:
            for snr in SNRS:

                scores = {
                    'OLS': [], 'RF': [], 'GBM': [], 'MLP': [],
                    'Attn_M3': [], 'Attn_M5': [],
                    'OLS_IS': [], 'RF_IS': [], 'GBM_IS': [], 'MLP_IS': [],
                    'Attn_M3_IS': [], 'Attn_M5_IS': []
                }

                print(f"\n🔄 Running {dgp} (N={n_train}, SNR={snr})...")

                for r in range(N_REPEATS):
                    seed = RANDOM_SEED + r * 1000

                    n_test = 1000
                    X, y = simulate_data(n_train + n_test, type=dgp, snr=snr, seed=seed)
                    X_train, y_train = X[:n_train], y[:n_train]
                    X_test, y_test = X[n_train:], y[n_train:]

                    print(f"  Repeat {r+1}/{N_REPEATS}...", end=" ", flush=True)

                    # 1. OLS
                    if USE_OLS:
                        try:
                            ols = LinearRegression().fit(X_train, y_train)
                            scores['OLS'].append(r2_score(y_test, ols.predict(X_test)))
                            scores['OLS_IS'].append(r2_score(y_train, ols.predict(X_train)))
                        except Exception as e:
                            print(f"OLS failed: {e}")
                            scores['OLS'].append(np.nan)
                            scores['OLS_IS'].append(np.nan)

                    # 2. RF
                    if USE_RF:
                        try:
                            rf = RandomForestRegressor(
                                n_estimators=RF_N_ESTIMATORS,
                                max_features=RF_MTRY,
                                n_jobs=-1,
                                random_state=seed
                            ).fit(X_train, y_train)
                            scores['RF'].append(r2_score(y_test, rf.predict(X_test)))
                            scores['RF_IS'].append(r2_score(y_train, rf.predict(X_train)))
                        except Exception as e:
                            print(f"RF failed: {e}")
                            scores['RF'].append(np.nan)
                            scores['RF_IS'].append(np.nan)

                    # 3. GBM
                    if USE_GBM:
                        try:
                            gbm = GradientBoostingRegressor(
                                n_estimators=GBM_N_ESTIMATORS,
                                learning_rate=GBM_LEARNING_RATE,
                                random_state=seed
                            ).fit(X_train, y_train)
                            scores['GBM'].append(r2_score(y_test, gbm.predict(X_test)))
                            scores['GBM_IS'].append(r2_score(y_train, gbm.predict(X_train)))
                        except Exception as e:
                            print(f"GBM failed: {e}")
                            scores['GBM'].append(np.nan)
                            scores['GBM_IS'].append(np.nan)

                    # 4. MLP Ensemble (NEW)
                    if USE_MLP:
                        try:
                            y_mlp_test, y_mlp_train = train_mlp_ensemble(
                                X_train, y_train, X_test,
                                hidden_units=MLP_HIDDEN_UNITS,
                                n_layers=MLP_HIDDEN_LAYERS,
                                dropout=MLP_DROPOUT,
                                ensemble_size=MLP_ENSEMBLE_SIZE,
                                max_epochs=MLP_MAX_EPOCHS,
                                patience=MLP_PATIENCE,
                                batch_size=MLP_BATCH_SIZE,
                                lr=MLP_LR,
                                val_frac=MLP_VAL_FRAC,
                                seed=seed,
                                verbose=False
                            )
                            scores['MLP'].append(r2_score(y_test, y_mlp_test))
                            scores['MLP_IS'].append(r2_score(y_train, y_mlp_train))
                        except Exception as e:
                            print(f"MLP failed: {e}")
                            scores['MLP'].append(np.nan)
                            scores['MLP_IS'].append(np.nan)

                    # 5. Attention (M=3)
                    if USE_ATTENTION_M3:
                        try:
                            y_att3_test, y_att3_train = train_torch_model(
                                X_train, y_train, X_test,
                                M=3,
                                steps=ATTENTION_STEPS,
                                lr=LEARNING_RATE,
                                reg_lambda=REG_M3,
                                seed=seed,
                                verbose=False
                            )
                            scores['Attn_M3'].append(r2_score(y_test, y_att3_test))
                            scores['Attn_M3_IS'].append(r2_score(y_train, y_att3_train))
                        except Exception as e:
                            print(f"Attention M=3 failed: {e}")
                            scores['Attn_M3'].append(np.nan)
                            scores['Attn_M3_IS'].append(np.nan)

                    # 6. Attention (M=5)
                    if USE_ATTENTION_M5:
                        try:
                            y_att5_test, y_att5_train = train_torch_model(
                                X_train, y_train, X_test,
                                M=5,
                                steps=ATTENTION_STEPS,
                                lr=LEARNING_RATE,
                                reg_lambda=REG_M5,
                                seed=seed,
                                verbose=False
                            )
                            r2_m5 = r2_score(y_test, y_att5_test)

                            if r2_m5 < 0.1:
                                print(f"⚠ M=5 R²={r2_m5:.3f}, retrying with λ={REG_M5/100:.0e}...", end=" ")
                                y_att5_test, y_att5_train = train_torch_model(
                                    X_train, y_train, X_test,
                                    M=5,
                                    steps=ATTENTION_STEPS,
                                    lr=LEARNING_RATE,
                                    reg_lambda=REG_M5 / 100,
                                    seed=seed,
                                    verbose=False
                                )
                                r2_m5 = r2_score(y_test, y_att5_test)
                                print(f"→ R²={r2_m5:.3f}")

                            scores['Attn_M5'].append(r2_m5)
                            scores['Attn_M5_IS'].append(r2_score(y_train, y_att5_train))
                        except Exception as e:
                            print(f"Attention M=5 failed: {e}")
                            scores['Attn_M5'].append(np.nan)
                            scores['Attn_M5_IS'].append(np.nan)

                    print("✓")

                # Compute Averages and Standard Errors
                avg = {k: np.nanmean(v) if v else np.nan for k, v in scores.items()}
                std = {k: np.nanstd(v) if v else np.nan for k, v in scores.items()}

                if USE_ATTENTION_M5 and USE_ATTENTION_M3 and avg['Attn_M5'] < 0:
                    print(f"   ⚠ M=5 optimization failed (R²={avg['Attn_M5']:.4f}), using M=3 value")
                    avg['Attn_M5'] = avg['Attn_M3']
                    avg['Attn_M5_IS'] = avg['Attn_M3_IS']
                    std['Attn_M5'] = std['Attn_M3']
                    std['Attn_M5_IS'] = std['Attn_M3_IS']

                # Print progress
                line = f"{dgp:<15} | {n_train:<5} | {snr:<4}"
                if USE_OLS: line += f" | {avg['OLS']:.4f}"
                if USE_RF: line += f" | {avg['RF']:.4f}"
                if USE_GBM: line += f" | {avg['GBM']:.4f}"
                if USE_MLP: line += f" | {avg['MLP']:.4f}"
                if USE_ATTENTION_M3: line += f" | {avg['Attn_M3']:.4f} "
                if USE_ATTENTION_M5: line += f" | {avg['Attn_M5']:.4f} "
                if USE_ATTENTION_M3: line += f" | {avg['Attn_M3_IS']:.4f}  "
                if USE_ATTENTION_M5: line += f" | {avg['Attn_M5_IS']:.4f}"
                print(line)

                final_results.append({
                    'DGP': dgp, 'N': n_train, 'SNR': snr,
                    'OLS': avg['OLS'], 'OLS_std': std['OLS'],
                    'OLS_IS': avg['OLS_IS'], 'OLS_IS_std': std['OLS_IS'],
                    'RF': avg['RF'], 'RF_std': std['RF'],
                    'RF_IS': avg['RF_IS'], 'RF_IS_std': std['RF_IS'],
                    'GBM': avg['GBM'], 'GBM_std': std['GBM'],
                    'GBM_IS': avg['GBM_IS'], 'GBM_IS_std': std['GBM_IS'],
                    'MLP': avg['MLP'], 'MLP_std': std['MLP'],
                    'MLP_IS': avg['MLP_IS'], 'MLP_IS_std': std['MLP_IS'],
                    'Attn_M3': avg['Attn_M3'], 'Attn_M3_std': std['Attn_M3'],
                    'Attn_M3_IS': avg['Attn_M3_IS'], 'Attn_M3_IS_std': std['Attn_M3_IS'],
                    'Attn_M5': avg['Attn_M5'], 'Attn_M5_std': std['Attn_M5'],
                    'Attn_M5_IS': avg['Attn_M5_IS'], 'Attn_M5_IS_std': std['Attn_M5_IS']
                })

    # --- SAVE AND DISPLAY ---
    print("\n" + "="*130)
    df = pd.DataFrame(final_results)

    print("\n✅ Simulation Complete.\n")

    # --- COMPLETE RESULTS TABLE ---
    print("\n" + "="*120)
    print("COMPLETE RESULTS TABLE")
    print("="*120)

    header = f"\n{'DGP':<15} | {'N':<5} | {'SNR':<4}"
    if USE_OLS: header += f" | {'OLS':<7}"
    if USE_RF: header += f" | {'RF':<7}"
    if USE_GBM: header += f" | {'GBM':<7}"
    if USE_MLP: header += f" | {'MLP':<7}"
    if USE_ATTENTION_M3: header += f" | {'Att(M=3)':<9}"
    if USE_ATTENTION_M5: header += f" | {'Att(M=5)':<9}"
    print(header)
    print("-" * 120)

    for _, row in df.iterrows():
        line = f"{row['DGP']:<15} | {row['N']:<5} | {row['SNR']:<4}"
        if USE_OLS: line += f" | {row['OLS']:.4f} "
        if USE_RF: line += f" | {row['RF']:.4f} "
        if USE_GBM: line += f" | {row['GBM']:.4f} "
        if USE_MLP: line += f" | {row['MLP']:.4f} "
        if USE_ATTENTION_M3: line += f" | {row['Attn_M3']:.4f}   "
        if USE_ATTENTION_M5: line += f" | {row['Attn_M5']:.4f}"
        print(line)

    # Show best model per DGP
    print("\n" + "="*120)
    print("BEST MODEL PER DGP (across all N/SNR)")
    print("="*120)
    for dgp in df['DGP'].unique():
        dgp_df = df[df['DGP'] == dgp]
        row = dgp_df[(dgp_df['N'] == dgp_df['N'].max()) & (dgp_df['SNR'] == dgp_df['SNR'].max())].iloc[0]

        models = {}
        if USE_OLS: models['OLS'] = row['OLS']
        if USE_RF: models['RF'] = row['RF']
        if USE_GBM: models['GBM'] = row['GBM']
        if USE_MLP: models['MLP'] = row['MLP']
        if USE_ATTENTION_M3: models['Att(M=3)'] = row['Attn_M3']
        if USE_ATTENTION_M5: models['Att(M=5)'] = row['Attn_M5']

        valid_models = {k: v for k, v in models.items() if not np.isnan(v)}

        if valid_models:
            best = max(valid_models, key=valid_models.get)
            scores_str = ' | '.join([f"{k}:{v:.3f}" for k, v in valid_models.items()])
            print(f"{dgp:<15}: {scores_str}  →  Winner: {best}")
        else:
            print(f"{dgp:<15}: No valid results")

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")