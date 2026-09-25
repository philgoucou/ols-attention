"""
Foundation models (TabPFN / TabICL) on the HIGHER-DIMENSIONAL suite.

Gap closed: TabPFN/TabICL were run on the 8 standard regression datasets, the 3
uncapped ones, and the 6 classification ones -- never on the P-ladder that answers
xZ71-Q2. The p.7-8 table therefore has only OLS/RF/FT-T/RB columns.

Protocol fidelity matters more than convenience here: the new columns must be
comparable to the shipped ones, so `prep_capped` and the dataset loaders are
IMPORTED from rebuttal_highdim rather than reimplemented (cap N=5000, X and y
standardized on train, same seeds). `add_local_python_source` is required or the
sibling module is not present in the container -- that bit us on rebuttal_friedman2.

Both packages document a feature ceiling well above P=81, and TabPFN already ran at
N=36,584 in the uncapped job, so neither N nor P should bind. The run records what
it actually used instead of assuming that.

    modal run rebuttal_fm_highdim.py::canary     # 1 cheap cell per model
    modal deploy rebuttal_fm_highdim.py          # then spawn
"""
from __future__ import annotations
import modal

import rebuttal_highdim as HD

CODE_VERSION = "v1-fm-highdim"
DATASETS = ['CPU_act', 'Bank32nh', 'Ailerons', 'Pol', 'Superconductivity',
             'Friedman-P20', 'Friedman-P50']
MODELS = ['TabPFN', 'TabICL']
N_REPEATS = 5

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2",
                      "torch==2.4.1", "tabpfn>=2.0.0", "tabicl>=0.1.0")
         .add_local_python_source("rebuttal_highdim"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-fm-highdim")


@app.function(image=image, gpu="A100-80GB", timeout=5400, memory=65536,
              volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_fm_highdim(dataset: str, model_name: str, seed_idx: int):
    """One (dataset, model, seed) cell on the high-dim protocol."""
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import r2_score
    t0 = _t.time()
    rec = {'dataset': dataset, 'model': model_name, 'seed_idx': seed_idx,
           'P': None, 'n_train': None, 'r2': float('nan'),
           'limits_overridden': False, 'default_refused': None,
           'code_version': CODE_VERSION, 'wall_s': None, 'error': None}
    try:
        import torch
        # identical protocol to the shipped OLS/RF/FT-T/RB columns
        X_full, y_full = HD.get_dataset(dataset, seed_idx)
        Xtr, ytr, Xte, yte, P, seed = HD.prep_capped(X_full, y_full, seed_idx)
        rec['P'] = int(P)
        rec['n_train'] = int(len(Xtr))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        def _looks_like_limit(e):
            s = str(e).lower()
            return any(k in s for k in ('pretraining', 'limit', 'too many', 'exceed',
                                        'maximum number of', 'n_samples', 'n_features'))

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
            # TabICL needs an explicit device index, not the bare 'cuda'.
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
def canary():
    """Cheapest real cell per model: lowest P, one seed. Verifies the import,
    the cache, both APIs and the device handling before spawning 50 jobs."""
    # The highdim volume already holds highdim_datasets_v1.pkl from the earlier
    # fan-out; fetch_and_cache_highdim belongs to that app, so we read the cache
    # from inside our own container instead of calling across apps.
    print(f"code_version={CODE_VERSION}")
    for mod in MODELS:
        r = run_fm_highdim.remote('CPU_act', mod, 0)
        print(f"  {mod:<8} CPU_act r2={r['r2']:+.4f} P={r['P']} n_train={r['n_train']} "
              f"refused={r['default_refused']} override={r['limits_overridden']} "
              f"v={r['code_version']} wall={r['wall_s']:.0f}s")
        if r['error']:
            print(f"    ERROR: {r['error'][:600]}")
