"""
Trimmed configuration of the Regression Block ("how far can the block be cut", App. E of
the revised paper). Formerly rebuttal_aggressive.py; the CSV keeps the tag agg: n_mix mixing sublayers LN(h + W(PCA_50(polycross(h)))), NO feed-forward
sublayer, NO attention, mean-pool + linear head. 7,489 parameters at P = 8 (against 30,401
for the block and 101,697 for the FT-Transformer). Run on four settings:

    std       the eight Table-1 datasets, capped N = 5000, features + targets standardized
    uncapped  California / Kin8nm / Protein at full sample size
    highdim   CPU_act, Bank32nh, Ailerons, Pol, Superconductivity (+ Friedman P=20/50), capped
    classify  the six OpenML classification datasets, cross-entropy head, logistic warm start

RECONSTRUCTION NOTICE. The harness that produced results/audit/audit_agg_results.csv
(version tag "v7-aggressive", July 2026) was lost with a temporary directory after its CSV
had been saved. This file rebuilds it from its two surviving siblings, rebuttal_sublayer.py
(the mixer / FFN sublayer model and regression training loop, whose `run_pca_sweep` with
n_ffn = 0, ncomp = 50 is the same architecture on the `std` setting) and rebuttal_classify.py
(classification prep, cross-entropy loop and multinomial-logistic warm start). Parameter
counts match the CSV exactly (7,489 at P = 8; 7,939 on Wine with C = 3), and the `canary`
entrypoint re-runs one cell per setting against the CSV's per-seed values. Verified 2026-09-25:
params identical on all four settings; scores 0.7258 vs 0.7295 (std/California, n_mix=3),
0.9254 vs 0.9219 (uncapped/Kin8nm), 0.9665 vs 0.9693 (highdim/CPU_act), 0.9444 vs 0.9444
(classify/Wine), i.e. within GPU run-to-run noise.

    modal run   rebuttal_trimmed.py::canary      # 4 cells vs the saved CSV
    modal deploy rebuttal_trimmed.py             # then spawn run_trim(setting, dataset, n_mix, ncomp, seed_idx)
                                                 # with tag 'agg' and harvest with the collector template
"""
from __future__ import annotations
import pickle
import modal

CODE_VERSION = "v8-aggressive-reconstruction"
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42
D_MODEL = 64
NCOMP = 50
EPOCHS = 200; BS = 256; LR = 1e-3; WD = 1e-4

STD_DS = ['California', 'Yacht', 'Energy', 'Concrete', 'Airfoil', 'Abalone', 'Kin8nm', 'Protein']
UNCAPPED_DS = ['California', 'Kin8nm', 'Protein']
HIGHDIM_DS = ['CPU_act', 'Bank32nh', 'Ailerons', 'Pol', 'Superconductivity']
FRIEDMAN = {'Friedman-P20': 20, 'Friedman-P50': 50}
FR_N = 5500; FR_NOISE = 1.0
CLF_DS = ['BreastCancer', 'Diabetes', 'Phoneme', 'Wine', 'Vehicle', 'Segment']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2", "torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-aggressive")


# ----------------------------------------------------------------------------- data
def _load(pkl):
    import os
    with open(os.path.join(CACHE_DIR, pkl), "rb") as f:
        return pickle.load(f)


def get_xy(setting, dataset, seed_idx):
    """Raw (X, y) for a setting; Friedman is generated per seed as in rebuttal_highdim.py."""
    import numpy as np
    if setting in ('std', 'uncapped'):
        return _load("datasets_v1.pkl")[dataset]
    if setting == 'highdim':
        if dataset in FRIEDMAN:
            from sklearn.datasets import make_friedman1
            X, y = make_friedman1(n_samples=FR_N, n_features=FRIEDMAN[dataset], noise=FR_NOISE,
                                  random_state=SEED + seed_idx * 1000)
            return X.astype(np.float32), y.astype(np.float32)
        return _load("highdim_datasets_v1.pkl")[dataset]
    if setting == 'classify':
        return _load("clf_datasets_v1.pkl")[dataset]
    raise ValueError(setting)


def prep_reg(X_full, y_full, seed_idx, cap):
    """Regression prep shared by std / uncapped / highdim: optional cap, then features AND
    targets standardized on the training split (the corrected protocol)."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    N_full, P = X_full.shape
    if cap and N_full > cap:
        idx = np.random.RandomState(SEED).choice(N_full, cap, replace=False)
        X_full, y_full = X_full[idx], y_full[idx]
    seed = SEED + seed_idx * 1000
    Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=TEST_FRAC, random_state=seed)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    scy = StandardScaler()
    ytr = scy.fit_transform(np.asarray(ytr, dtype='float64').reshape(-1, 1)).ravel()
    yte = scy.transform(np.asarray(yte, dtype='float64').reshape(-1, 1)).ravel()
    return (Xtr.astype('float32'), ytr.astype('float32'),
            Xte.astype('float32'), yte.astype('float32'), P, seed)


def prep_clf(X_full, y_full, seed_idx):
    """Verbatim from rebuttal_classify.py: stratified cap, stratified split, features standardized."""
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


def mem_settings(P):
    """Batch size and PCA-fit subsample by dimension (from rebuttal_highdim.py): the degree-2
    cross-feature map is O(P^2) in width."""
    if P <= 25: return dict(pca_fit_cap=2000, batch_size=256)
    if P <= 35: return dict(pca_fit_cap=1200, batch_size=128)
    if P <= 45: return dict(pca_fit_cap=700,  batch_size=96)
    if P <= 60: return dict(pca_fit_cap=400,  batch_size=48)
    return dict(pca_fit_cap=100, batch_size=24)


# ----------------------------------------------------------------------------- model
def train_trimmed(X_train, y_train, X_test, n_mix=2, ncomp=NCOMP, n_classes=None,
                  batch_size=BS, pca_fit_cap=5000, seed=42):
    """n_mix mixing sublayers, no FFN, mean-pool head. n_classes=None -> regression (MSE,
    ridge warm start of the first mixer); otherwise cross-entropy with a multinomial-logistic
    warm start. Model code is that of rebuttal_sublayer.train_rb_sublayers with n_ffn = 0."""
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, LogisticRegression

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = D_MODEL
    C = 1 if n_classes is None else n_classes

    def polycross(x):
        B, S, Dm = x.shape
        per = torch.cat([x, x ** 2 - 1], dim=-1)
        pairs = torch.cat([x[:, i, :] * x[:, j, :]
                           for i in range(S) for j in range(i, S)], dim=-1)
        return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)

    class FrozenPCA(nn.Module):
        def __init__(self, mean, comp):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer('comp', torch.from_numpy(comp.astype(np.float32)))
            self.out_dim = comp.shape[0]
        def forward(self, x): return (x - self.mean) @ self.comp.T

    class MixerSublayer(nn.Module):
        """LN(h + W(PCA(polycross(h))))"""
        def __init__(self, red):
            super().__init__()
            self.red = red; self.W = nn.Linear(red.out_dim, D); self.n = nn.LayerNorm(D)
        def feat(self, h): return self.red(polycross(h))
        def forward(self, h): return self.n(h + self.W(self.feat(h)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, D)
            self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.mixers = nn.ModuleList()
            self.head = nn.Linear(D, C)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def upto_mixer(self, x, k):
            h = self.embed_x(x)
            for i in range(k): h = self.mixers[i](h)
            return h
        def forward(self, x):
            h = self.embed_x(x)
            for m in self.mixers: h = m(h)
            return self.head(h.mean(1))

    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    Xp = torch.from_numpy(X_pca.astype(np.float32)).to(device)

    torch.manual_seed(seed)
    model = Net().to(device)
    with torch.no_grad():
        feat_dim = polycross(model.embed_x(Xp[:8])).shape[-1]

    # mixers built sequentially, each PCA fit on its own input representation
    for i in range(n_mix):
        model.eval()
        with torch.no_grad():
            phi = polycross(model.upto_mixer(Xp, i)).reshape(-1, feat_dim).cpu().numpy()
        n_comp = min(ncomp, phi.shape[1], phi.shape[0])
        p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
        torch.manual_seed(seed + 100 + i)
        model.mixers.append(MixerSublayer(FrozenPCA(p.mean_, p.components_)).to(device))

    # warm start of the FIRST mixer + head. Chunked by the P-scaled batch size (the fixed
    # 2048-row chunk of an earlier version allocated ~30 GB at P = 48).
    model.eval()
    WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
    chunk = max(64, 4 * batch_size)
    with torch.no_grad():
        for i in range(0, WS_N, chunk):
            xb = torch.from_numpy(X_ws[i:i+chunk].astype(np.float32)).to(device)
            phis.append(model.mixers[0].feat(model.embed_x(xb)).mean(1).cpu().numpy())
    phi_pooled = np.concatenate(phis)
    if n_classes is None:
        reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
        with torch.no_grad():
            model.mixers[0].W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
            model.mixers[0].W.bias.data.zero_()
            model.head.weight.data.zero_(); model.head.weight.data[0, 0] = 1.0
            model.head.bias.data.fill_(float(reg.intercept_))
    elif C <= D:
        try:
            lrf = LogisticRegression(max_iter=1000).fit(phi_pooled, y_train[:WS_N])
            coef = lrf.coef_.astype(np.float32); intr = lrf.intercept_.astype(np.float32)
            if coef.shape[0] == 1 and C == 2:
                coef = np.vstack([-coef[0] / 2, coef[0] / 2])
                intr = np.array([-intr[0] / 2, intr[0] / 2], np.float32)
            if coef.shape[0] == C:
                with torch.no_grad():
                    model.mixers[0].W.weight.data[:C] = torch.from_numpy(coef).to(device)
                    model.mixers[0].W.bias.data[:C].zero_()
                    model.head.weight.data.zero_()
                    for c in range(C): model.head.weight.data[c, c] = 1.0
                    model.head.bias.data = torch.from_numpy(intr).to(device)
        except Exception:
            pass

    # training: Adam, grad-clip 1, 15% validation split, early stopping (patience 10)
    N = X_train.shape[0]; n_val = max(1, int(N * 0.15))
    if n_classes is None:
        idx = np.random.permutation(N)
        crit = nn.MSELoss(); ycast = lambda a: torch.from_numpy(a.astype(np.float32)).unsqueeze(1)
    else:
        idx = np.random.RandomState(seed).permutation(N)
        crit = nn.CrossEntropyLoss(); ycast = lambda a: torch.from_numpy(a)
    Xt = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32)); yt = ycast(y_train[idx[n_val:]])
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device); yv = ycast(y_train[idx[:n_val]]).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, best_state, pat = float('inf'), None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best:
            best, best_state, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):   # prediction batched by the same size
            xb = torch.from_numpy(X_test[i:i+batch_size].astype(np.float32)).to(device)
            outs.append(model(xb).cpu().numpy())
    out = np.concatenate(outs)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    return (out.ravel() if n_classes is None else out), n_params


# ----------------------------------------------------------------------------- one cell
def _cell(setting, dataset, n_mix, ncomp, seed_idx):
    import time as _t, traceback
    import numpy as np
    from sklearn.metrics import r2_score, accuracy_score
    t0 = _t.time()
    rec = {'setting': setting, 'dataset': dataset, 'n_mix': n_mix, 'ncomp': ncomp,
           'seed_idx': seed_idx, 'score': float('nan'), 'P': None, 'n_train': None,
           'params': None, 'version': CODE_VERSION, 'wall_s': None, 'error': None}
    try:
        X, y = get_xy(setting, dataset, seed_idx)
        if setting == 'classify':
            Xtr, ytr, Xte, yte, seed = prep_clf(X, y, seed_idx)
            C = int(y.max()) + 1; P = Xtr.shape[1]
            ms = mem_settings(P) if P > 25 else dict(pca_fit_cap=5000, batch_size=BS)
            logits, n_params = train_trimmed(Xtr, ytr, Xte, n_mix=n_mix, ncomp=ncomp,
                                             n_classes=C, seed=seed, **ms)
            score = accuracy_score(yte, logits.argmax(1))
        else:
            cap = None if setting == 'uncapped' else MAX_N
            Xtr, ytr, Xte, yte, P, seed = prep_reg(X, y, seed_idx, cap)
            ms = mem_settings(P) if setting == 'highdim' else dict(pca_fit_cap=5000, batch_size=BS)
            pred, n_params = train_trimmed(Xtr, ytr, Xte, n_mix=n_mix, ncomp=ncomp, seed=seed, **ms)
            score = r2_score(yte, pred)
        rec.update(score=float(score), P=int(P), n_train=int(len(Xtr)), params=int(n_params))
    except Exception as e:
        rec['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    rec['wall_s'] = _t.time() - t0
    return rec


@app.function(image=image, gpu="A10G", timeout=3600, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_trim(setting: str, dataset: str, n_mix: int, ncomp: int, seed_idx: int):
    return _cell(setting, dataset, n_mix, ncomp, seed_idx)


@app.function(image=image, gpu="A100-80GB", timeout=5400, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_trim_a100(setting: str, dataset: str, n_mix: int, ncomp: int, seed_idx: int):
    """Same cell on an A100: the P >= 40 datasets (Ailerons, Pol, Superconductivity)."""
    return _cell(setting, dataset, n_mix, ncomp, seed_idx)


# ----------------------------------------------------------------------------- canary
# Per-seed cells of results/audit/audit_agg_results.csv used to verify the reconstruction.
VERIFY = [('std', 'California', 3, 50, 0, 0.7295), ('uncapped', 'Kin8nm', 2, 50, 0, 0.9219),
          ('highdim', 'CPU_act', 2, 50, 0, 0.9693), ('classify', 'Wine', 2, 50, 0, 0.9444)]


@app.local_entrypoint()
def canary():
    print(f"code_version={CODE_VERSION}")
    handles = [(v, run_trim.spawn(*v[:5])) for v in VERIFY]
    for (setting, ds, nm, nc, s, target), h in handles:
        r = h.get()
        d = r['score'] - target if r['error'] is None else float('nan')
        print(f"  {setting:<9}{ds:<11} n_mix={nm} seed={s}: score={r['score']:+.4f}  saved={target:+.4f}  "
              f"diff={d:+.4f}  params={r['params']} P={r['P']} n_train={r['n_train']} "
              f"{'OK' if abs(d) < 0.01 else 'CHECK'}")
        if r['error']:
            print("    ERROR:", r['error'][:500])
