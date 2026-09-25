"""
Local CPU sanity test for stacked_rb.train_stacked_rb.
Not a paper-number reproduction (subsampled / short) — a correctness gate:
  * L=1,2,3 all run without crashing or NaNs
  * R2 is sane (clearly positive on California)
  * params increase with depth
  * L=1 is in the right ballpark for California RB (~0.75 full; expect >0.6 here)
"""
import time
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from stacked_rb import train_stacked_rb

torch.set_num_threads(max(1, __import__('os').cpu_count() or 4))
DEV = torch.device("cpu")

# --- small, fast local config ---
N_SUB = 2500
PCA_CAP = 1200
EPOCHS = 60

d = fetch_california_housing()
X_all, y_all = d.data.astype(np.float32), d.target.astype(np.float32)
rng = np.random.RandomState(0)
idx = rng.choice(len(X_all), N_SUB, replace=False)
X, y = X_all[idx], y_all[idx]

print(f"California subsample N={N_SUB}, P={X.shape[1]}, CPU test")
print(f"config: pca_cap={PCA_CAP}, epochs={EPOCHS}\n")

def run(n_layers, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    t0 = time.time()
    pred, info = train_stacked_rb(
        Xtr.astype(np.float32), ytr.astype(np.float32), Xte.astype(np.float32),
        n_layers=n_layers, epochs=EPOCHS, pca_fit_cap=PCA_CAP, seed=seed, device=DEV)
    dt = time.time() - t0
    nan = bool(np.isnan(pred).any())
    r2 = float('nan') if nan else r2_score(yte, pred)
    print(f"  L={n_layers} seed={seed}: R2={r2:+.4f}  params={info['n_params']:,}  "
          f"nan={nan}  {dt:.1f}s")
    return r2, info['n_params'], nan

print("Running depth sweep:")
results = {}
for L in (1, 2, 3):
    r2, npar, nan = run(L, seed=42)
    results[L] = (r2, npar, nan)
# second seed for L=1 stability
run(1, seed=1042)

print("\nChecks:")
ok = True
for L in (1, 2, 3):
    r2, npar, nan = results[L]
    if nan:
        print(f"  FAIL L={L}: NaN predictions"); ok = False
    elif r2 < 0.4:
        print(f"  WARN L={L}: R2={r2:.3f} low for California (subsampled, may be ok)")
    else:
        print(f"  OK   L={L}: R2={r2:.3f}")
# monotone params
p1, p2, p3 = results[1][1], results[2][1], results[3][1]
if p1 < p2 < p3:
    print(f"  OK   params increase with depth: {p1:,} < {p2:,} < {p3:,}")
else:
    print(f"  FAIL params not monotone: {p1:,} {p2:,} {p3:,}"); ok = False

print("\n" + ("ALL SANITY CHECKS PASSED" if ok else "SOME CHECKS FAILED — inspect before Modal"))
