"""
Modal fan-out — TabICL reference on the classification suite (Ayqz named it
alongside TabPFN). Same 6 datasets / splits / seeds as rebuttal_classify.py.
TabICL is classification-only, so it joins the classification table.

    modal run rebuttal_tabicl.py
"""
from __future__ import annotations
import pickle, time
import modal

N_REPEATS = 5
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42
CLF_ORDER = ['BreastCancer', 'Diabetes', 'Phoneme', 'Wine', 'Vehicle', 'Segment']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2",
                      "torch==2.4.1", "tabicl"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-tabicl")


def _load_clf():
    import os
    with open(os.path.join(CACHE_DIR, "clf_datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def prep_clf(X_full, y_full, seed_idx):
    """Identical to rebuttal_classify.prep_clf — same splits guaranteed."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    N_full = len(X_full)
    seed = SEED + seed_idx * 1000
    if N_full > MAX_N:
        idx = np.arange(N_full)
        keep, _ = train_test_split(idx, train_size=MAX_N, random_state=SEED, stratify=y_full)
        X_use, y_use = X_full[keep], y_full[keep]
    else:
        X_use, y_use = X_full, y_full
    Xtr, Xte, ytr, yte = train_test_split(X_use, y_use, test_size=TEST_FRAC,
                                          random_state=seed, stratify=y_use)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    return Xtr.astype('float32'), ytr.astype('int64'), Xte.astype('float32'), yte.astype('int64'), seed


@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              memory=16384, retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_tabicl(dataset_name: str, seed_idx: int):
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score
    t0 = _t.time()
    try:
        from tabicl import TabICLClassifier
        X, y = _load_clf()[dataset_name]
        C = int(y.max()) + 1
        Xtr, ytr, Xte, yte, seed = prep_clf(X, y, seed_idx)
        import torch
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        clf = None
        for kwargs in ({'device': dev}, {}):
            try:
                clf = TabICLClassifier(**kwargs); break
            except TypeError:
                continue
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        proba = np.asarray(proba, dtype=float)
        acc = float(accuracy_score(yte, proba.argmax(1)))
        try:
            auc = float(roc_auc_score(yte, proba[:, 1]) if C == 2
                        else roc_auc_score(yte, proba, multi_class='ovr', average='macro'))
        except Exception:
            auc = float('nan')
        return {'dataset': dataset_name, 'model': 'TabICL', 'n_classes': C,
                'seed_idx': seed_idx, 'acc': acc, 'auc': auc,
                'wall_s': _t.time() - t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'model': 'TabICL', 'n_classes': None,
                'seed_idx': seed_idx, 'acc': float('nan'), 'auc': float('nan'),
                'wall_s': _t.time() - t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.local_entrypoint()
def main():
    import pandas as pd
    t0 = time.time()
    args = [(ds, s) for ds in CLF_ORDER for s in range(N_REPEATS)]
    print(f">> TabICL: {len(args)} jobs")
    res = list(run_tabicl.starmap(args, order_outputs=False))
    df = pd.DataFrame(res)
    df.to_csv("tabicl_results.csv", index=False)
    errs = df[df['error'].notna()]
    if len(errs):
        print(f"[tabicl errors: {len(errs)}]")
        for _, r in errs.iterrows():
            print(f"  {r['dataset']}/seed{r['seed_idx']}: {r['error'].splitlines()[0]}")
    agg = df.groupby('dataset').agg(acc=('acc', 'mean'), auc=('auc', 'mean')).reindex(CLF_ORDER)
    print("\n# TABICL — mean accuracy / AUC")
    print(agg.round(3).to_string())
    print(f"\nSaved: tabicl_results.csv  ({time.time()-t0:.1f}s total)")
