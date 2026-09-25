"""
Foundation models on the UNCAPPED regression datasets (California, Kin8nm, Protein).

We have TabPFN/TabICL only at the N<=5000 cap. This lifts the cap so the uncapped-N
table can carry a foundation-model column.

Two things must be established empirically rather than assumed:
  1. Does TabICL have a REGRESSOR at all? (it is a classification ICL model as far as
     we know; the classification suite is where we used it). The probe reports the
     public API instead of guessing.
  2. TabPFN documents a ~10k training-row ceiling. California has 16,512 train rows
     and Protein 36,584 — both above it. So the run must record whether the model
     used all rows, refused, or silently subsampled; otherwise "uncapped" would be
     a claim we cannot support.

The run function therefore reports n_train_available vs n_train_used and never
hides an internal subsample.

    modal run rebuttal_fm_uncapped.py::probe      # API + limits, cheap
    modal deploy rebuttal_fm_uncapped.py          # then spawn the real jobs
"""
from __future__ import annotations
import pickle
import modal

TEST_FRAC = 0.2
SEED = 42
UNCAPPED = ['California', 'Kin8nm', 'Protein']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2",
                      "torch==2.4.1", "tabpfn>=2.0.0", "tabicl>=0.1.0"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-fm-uncapped")


def _load_cached():
    import os
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def prep_uncapped(X, y, seed_idx):
    """Full-sample split, X and y standardized on train (y-std = current protocol)."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    seed = SEED + seed_idx * 1000
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC, random_state=seed)
    sx = StandardScaler(); Xtr = sx.fit_transform(Xtr); Xte = sx.transform(Xte)
    sy = StandardScaler()
    ytr = sy.fit_transform(np.asarray(ytr, dtype='float64').reshape(-1, 1)).ravel()
    yte = sy.transform(np.asarray(yte, dtype='float64').reshape(-1, 1)).ravel()
    return (Xtr.astype('float32'), ytr.astype('float32'),
            Xte.astype('float32'), yte.astype('float32'), seed)


@app.function(image=image, timeout=900)
def probe():
    """Report what the two packages actually expose, and any documented N ceiling."""
    out = {}
    try:
        import tabpfn, inspect
        out['tabpfn_version'] = getattr(tabpfn, '__version__', 'unknown')
        out['tabpfn_public'] = sorted(n for n in dir(tabpfn) if not n.startswith('_'))
        try:
            from tabpfn import TabPFNRegressor
            sig = inspect.signature(TabPFNRegressor.__init__)
            out['TabPFNRegressor_params'] = list(sig.parameters.keys())
            src = inspect.getsource(TabPFNRegressor.fit) if hasattr(TabPFNRegressor, 'fit') else ''
            out['regressor_fit_mentions_limit'] = any(
                k in src for k in ('10000', '10_000', 'max_samples', 'subsample'))
        except Exception as e:
            out['TabPFNRegressor_error'] = f"{type(e).__name__}: {e}"
    except Exception as e:
        out['tabpfn_import_error'] = f"{type(e).__name__}: {e}"
    try:
        import tabicl
        out['tabicl_version'] = getattr(tabicl, '__version__', 'unknown')
        out['tabicl_public'] = sorted(n for n in dir(tabicl) if not n.startswith('_'))
        out['tabicl_has_regressor'] = any('Regress' in n for n in dir(tabicl))
    except Exception as e:
        out['tabicl_import_error'] = f"{type(e).__name__}: {e}"
    return out


@app.function(image=image, gpu="A100-80GB", timeout=5400, memory=65536,
              volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_fm_uncapped(dataset_name: str, model_name: str, seed_idx: int, cap: int = -1):
    """Full training split unless cap>0. Records rows available vs used, and whether
    the model's own pretraining sample limit had to be overridden to fit them."""
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    t0 = _t.time()
    rec = {'dataset': dataset_name, 'model': model_name, 'seed_idx': seed_idx,
           'cap': cap, 'n_train_available': None, 'n_train_used': None,
           'limits_overridden': False, 'default_refused': None,
           'r2': float('nan'), 'wall_s': None, 'error': None}
    try:
        import torch
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, seed = prep_uncapped(X, y, seed_idx)
        rec['n_train_available'] = int(len(Xtr))
        if cap > 0 and len(Xtr) > cap:
            sub = np.random.RandomState(seed).choice(len(Xtr), cap, replace=False)
            Xtr, ytr = Xtr[sub], ytr[sub]
        rec['n_train_used'] = int(len(Xtr))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        def _looks_like_limit(e):
            s = str(e).lower()
            return any(k in s for k in ('pretraining', 'limit', 'too many', 'exceed',
                                        'maximum number of', 'n_samples'))

        if model_name == 'TabPFN':
            from tabpfn import TabPFNRegressor
            try:
                m = TabPFNRegressor(device=dev, memory_saving_mode='auto',
                                    random_state=seed)
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                rec['default_refused'] = False
            except Exception as e_def:
                if not _looks_like_limit(e_def):
                    raise
                rec['default_refused'] = True
                rec['limits_overridden'] = True
                m = TabPFNRegressor(device=dev, memory_saving_mode='auto',
                                    ignore_pretraining_limits=True,
                                    random_state=seed)
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
        elif model_name == 'TabICL':
            from tabicl import TabICLRegressor
            # TabICL validates the device string strictly: it needs an explicit
            # index ('cuda:0'), not the bare 'cuda' that torch/TabPFN accept.
            dev_icl = 'cuda:0' if dev == 'cuda' else dev
            try:
                m = TabICLRegressor(device=dev_icl, random_state=seed)
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                rec['default_refused'] = False
            except TypeError:
                m = TabICLRegressor(device=dev_icl)
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                rec['default_refused'] = False
        else:
            raise ValueError(model_name)
        rec['r2'] = float(r2_score(yte, pred))
        rec['wall_s'] = _t.time() - t0
        return rec
    except Exception as e:
        rec['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        rec['wall_s'] = _t.time() - t0
        return rec


@app.local_entrypoint()
def show_probe():
    import json
    r = probe.remote()
    print("=== FOUNDATION MODEL API PROBE ===")
    for k in sorted(r):
        v = r[k]
        if isinstance(v, list) and len(v) > 14:
            v = v[:14] + ['...']
        print(f"  {k}: {v}")
