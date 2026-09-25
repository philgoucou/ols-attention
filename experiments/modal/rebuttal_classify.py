"""
Modal fan-out — classification (answers xZ71-Q2 / Ayqz: "beyond continuous regression?").

6 OpenML/sklearn datasets (moderate P, kept in the poly-cross's tractable regime):
  binary:     BreastCancer(P30) Diabetes(P8) Phoneme(P5)
  multiclass: Wine(3,P13) Vehicle(4,P18) Segment(7,P19)
Models: LogReg, RandomForest, FT-Transformer, Regression Block, TabPFN.
Metric: accuracy + ROC-AUC (macro-OVR for multiclass). Capped N=5000, 5 seeds.

    modal run rebuttal_classify.py
"""
from __future__ import annotations
import pickle, time
import modal

N_REPEATS = 5
TEST_FRAC = 0.2
MAX_N = 5000
SEED = 42
RF_TREES = 500

POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4
FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128; FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1; FTT_EPOCHS = 200; FTT_BS = 256; FTT_LR = 1e-3; FTT_WD = 1e-4

# name -> (source, id_or_none, n_classes, P, note)
CLF_REGISTRY = {
    'BreastCancer': ('sklearn', 'breast_cancer', 2, 30),
    'Diabetes':     ('openml', 37, 2, 8),
    'Phoneme':      ('openml', 1489, 2, 5),
    'Wine':         ('sklearn', 'wine', 3, 13),
    'Vehicle':      ('openml', 54, 4, 18),
    'Segment':      ('openml', 36, 7, 19),
}
CLF_ORDER = ['BreastCancer', 'Diabetes', 'Phoneme', 'Wine', 'Vehicle', 'Segment']
MODELS = ['LogReg', 'RF', 'FTT', 'RB', 'TabPFN']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2",
                      "torch==2.4.1", "tabpfn>=2.0.0"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-classify")


@app.function(image=image, volumes={CACHE_DIR: cache_vol}, timeout=1200)
def fetch_and_cache_clf():
    import os, numpy as np
    from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
    from sklearn.preprocessing import LabelEncoder
    pkl = os.path.join(CACHE_DIR, "clf_datasets_v1.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            ds = pickle.load(f)
        return {k: (v[0].shape, int(v[1].max()) + 1) for k, v in ds.items()}
    loaded = {}
    for name in CLF_ORDER:
        src, ident, C, P = CLF_REGISTRY[name]
        try:
            if src == 'sklearn':
                d = load_breast_cancer() if ident == 'breast_cancer' else load_wine()
                X = d.data.astype(np.float32); y = d.target.astype(np.int64)
            else:
                d = fetch_openml(data_id=ident, as_frame=True, parser='auto')
                X = d.data.select_dtypes(include=[np.number]).values.astype(np.float32)
                y = LabelEncoder().fit_transform(d.target.astype(str)).astype(np.int64)
            mask = ~np.isnan(X).any(axis=1)
            X, y = X[mask], y[mask]
            assert X.shape[1] == P, f"{name} P={X.shape[1]} != {P}"
            assert len(np.unique(y)) == C, f"{name} C={len(np.unique(y))} != {C}"
            loaded[name] = (X, y)
            print(f"  OK {name}: X={X.shape} C={C}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
    with open(pkl, "wb") as f:
        pickle.dump(loaded, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache_vol.commit()
    return {k: (v[0].shape, int(v[1].max()) + 1) for k, v in loaded.items()}


def _load_clf():
    import os
    with open(os.path.join(CACHE_DIR, "clf_datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def prep_clf(X_full, y_full, seed_idx):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    N_full = len(X_full)
    seed = SEED + seed_idx * 1000
    if N_full > MAX_N:
        # stratified cap
        from sklearn.model_selection import train_test_split as tts
        idx = np.arange(N_full)
        keep, _ = tts(idx, train_size=MAX_N, random_state=SEED, stratify=y_full)
        X_use, y_use = X_full[keep], y_full[keep]
    else:
        X_use, y_use = X_full, y_full
    Xtr, Xte, ytr, yte = train_test_split(X_use, y_use, test_size=TEST_FRAC,
                                          random_state=seed, stratify=y_use)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    return Xtr.astype('float32'), ytr.astype('int64'), Xte.astype('float32'), yte.astype('int64'), seed


def _metrics(yte, proba, n_classes):
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(yte, proba.argmax(1))
    try:
        if n_classes == 2:
            auc = roc_auc_score(yte, proba[:, 1])
        else:
            auc = roc_auc_score(yte, proba, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return float(acc), float(auc)


def _softmax_np(z):
    import numpy as np
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---- NN classifiers (inlined from classify.py) ----
def _train_loop(model, X_train, y_train, epochs, batch_size, lr, weight_decay, device, seed, adamw, cosine):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    crit = nn.CrossEntropyLoss()
    N = X_train.shape[0]; n_val = max(1, int(N * 0.15)); rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]]); yt = torch.from_numpy(y_train[idx[n_val:]])
    Xv = torch.from_numpy(X_train[idx[:n_val]]).to(device); yv = torch.from_numpy(y_train[idx[:n_val]]).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = (torch.optim.AdamW if adamw else torch.optim.Adam)(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if cosine:
        warm = max(1, epochs // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: ep/warm if ep < warm else 0.5*(1+np.cos(np.pi*(ep-warm)/max(1, epochs-warm))))
    best, bs, pat = float('inf'), None, 0
    for ep in range(1, epochs+1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if sched: sched.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best: best, bs, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if bs: model.load_state_dict({k: v.to(device) for k, v in bs.items()})
    return model


def _predict_proba(model, X_test, device):
    import numpy as np, torch
    model.eval(); logits = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            logits.append(model(torch.from_numpy(X_test[i:i+4096]).to(device)).cpu().numpy())
    return _softmax_np(np.concatenate(logits))


def train_ftt_clf(X_train, y_train, X_test, n_classes, seed=42):
    import numpy as np, torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = FTT_D_MODEL
    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Linear(1, D) for _ in range(nf)])
            self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            enc = nn.TransformerEncoderLayer(d_model=D, nhead=FTT_N_HEADS, dim_feedforward=FTT_FFN_DIM,
                                             dropout=FTT_DROPOUT, batch_first=True, activation='gelu')
            self.tr = nn.TransformerEncoder(enc, num_layers=FTT_N_BLOCKS)
            self.norm = nn.LayerNorm(D); self.head = nn.Linear(D, n_classes)
        def forward(self, x):
            B = x.size(0)
            tok = torch.stack([self.emb[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            tok = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)
            return self.head(self.norm(self.tr(tok)[:, 0, :]))
    model = FT().to(device)
    model = _train_loop(model, X_train, y_train, FTT_EPOCHS, FTT_BS, FTT_LR, FTT_WD, device, seed, True, True)
    return _predict_proba(model, X_test, device), {'n_params': sum(p.numel() for p in model.parameters())}


def train_rb_clf(X_train, y_train, X_test, n_classes, seed=42):
    import numpy as np, torch, torch.nn as nn
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = POLY_D_MODEL; C = n_classes; pca_fit_cap = 5000

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

    class Block(nn.Module):
        def __init__(self, red):
            super().__init__()
            self.red = red; self.W = nn.Linear(red.out_dim, D); self.n1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, POLY_FFN_DIM), nn.ReLU(), nn.Dropout(POLY_DROPOUT),
                                     nn.Linear(POLY_FFN_DIM, D), nn.Dropout(POLY_DROPOUT))
            self.n2 = nn.LayerNorm(D)
        def feat(self, h): return self.red(polycross(h))
        def forward(self, h):
            z = self.n1(h + self.W(self.feat(h)))
            return self.n2(z + self.ffn(z))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, D); self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.blocks = nn.ModuleList(); self.head = nn.Linear(D, C)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def forward(self, x):
            h = self.embed_x(x)
            for b in self.blocks: h = b(h)
            return self.head(h.mean(1))

    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    torch.manual_seed(seed); model = Net().to(device)
    Xp = torch.from_numpy(X_pca.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        phi = polycross(model.embed_x(Xp))
        feat_dim = phi.shape[-1]
        phi = phi.reshape(-1, feat_dim).cpu().numpy()
    n_comp = min(POLY_PCA_COMP, phi.shape[1], phi.shape[0])
    p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
    torch.manual_seed(seed + 100)
    model.blocks.append(Block(FrozenPCA(p.mean_, p.components_)).to(device))

    # logistic warm-start of block-0 mixer
    if C <= D:
        model.eval(); WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
        with torch.no_grad():
            for i in range(0, WS_N, 4096):
                xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
                phis.append(model.blocks[0].feat(model.embed_x(xb)).mean(1).cpu().numpy())
        phi_pooled = np.concatenate(phis)
        try:
            lrf = LogisticRegression(max_iter=1000).fit(phi_pooled, y_train[:WS_N])
            coef = lrf.coef_.astype(np.float32); intr = lrf.intercept_.astype(np.float32)
            if coef.shape[0] == 1 and C == 2:
                coef = np.vstack([-coef[0]/2, coef[0]/2]); intr = np.array([-intr[0]/2, intr[0]/2], np.float32)
            if coef.shape[0] == C:
                with torch.no_grad():
                    model.blocks[0].W.weight.data[:C] = torch.from_numpy(coef).to(device)
                    model.blocks[0].W.bias.data[:C].zero_()
                    model.head.weight.data.zero_()
                    for c in range(C): model.head.weight.data[c, c] = 1.0
                    model.head.bias.data = torch.from_numpy(intr).to(device)
        except Exception:
            pass
    model = _train_loop(model, X_train, y_train, POLY_EPOCHS, POLY_BS, POLY_LR, POLY_WD, device, seed, False, False)
    return _predict_proba(model, X_test, device), {'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


# ================= Modal run function =================
@app.function(image=image, gpu="T4", timeout=1800, volumes={CACHE_DIR: cache_vol},
              memory=16384, retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_clf(dataset_name: str, model_name: str, seed_idx: int):
    import time as _t, traceback, numpy as np
    t0 = _t.time()
    try:
        X, y = _load_clf()[dataset_name]
        C = int(y.max()) + 1
        Xtr, ytr, Xte, yte, seed = prep_clf(X, y, seed_idx)
        if model_name == 'LogReg':
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
            proba = m.predict_proba(Xte)
        elif model_name == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
            proba = m.predict_proba(Xte)
        elif model_name == 'FTT':
            proba, _ = train_ftt_clf(Xtr, ytr, Xte, C, seed=seed)
        elif model_name == 'RB':
            proba, _ = train_rb_clf(Xtr, ytr, Xte, C, seed=seed)
        elif model_name == 'TabPFN':
            import torch
            from tabpfn import TabPFNClassifier
            m = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
            m.fit(Xtr, ytr); proba = m.predict_proba(Xte)
        else:
            raise ValueError(model_name)
        acc, auc = _metrics(yte, proba, C)
        return {'dataset': dataset_name, 'model': model_name, 'n_classes': C, 'seed_idx': seed_idx,
                'acc': acc, 'auc': auc, 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'dataset': dataset_name, 'model': model_name, 'n_classes': None, 'seed_idx': seed_idx,
                'acc': float('nan'), 'auc': float('nan'), 'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.local_entrypoint()
def main():
    import pandas as pd
    t0 = time.time()
    shapes = fetch_and_cache_clf.remote()
    print(">> cached classification datasets:")
    for k, v in shapes.items():
        print(f"   {k}: X={v[0]} C={v[1]}")
    args = [(ds, m, s) for ds in CLF_ORDER for m in MODELS for s in range(N_REPEATS)]
    print(f">> Classification: {len(args)} jobs")
    res = list(run_clf.starmap(args, order_outputs=False))
    df = pd.DataFrame(res)
    df.to_csv("classify_results.csv", index=False)

    errs = df[df['error'].notna()]
    if len(errs):
        print(f"\n[classify errors: {len(errs)}]")
        for _, r in errs.iterrows():
            print(f"  {r['dataset']}/{r['model']}/seed{r['seed_idx']}: {r['error'].splitlines()[0]}")

    agg = df.groupby(['dataset', 'model']).agg(acc=('acc', 'mean'), acc_sd=('acc', 'std'),
                                               auc=('auc', 'mean')).reset_index()
    print("\n" + "#"*70 + "\n# CLASSIFICATION — mean accuracy\n" + "#"*70)
    pacc = agg.pivot(index='dataset', columns='model', values='acc').reindex(CLF_ORDER)[MODELS]
    print(pacc.round(3).to_string())
    print("\n# mean ROC-AUC")
    pauc = agg.pivot(index='dataset', columns='model', values='auc').reindex(CLF_ORDER)[MODELS]
    print(pauc.round(3).to_string())
    print("\n--- mean over datasets ---")
    for m in MODELS:
        print(f"  {m:<8} acc={pacc[m].mean():.3f}  auc={pauc[m].mean():.3f}")
    print("\n--- OpenReview markdown: classification accuracy ---")
    print("| Dataset | C | " + " | ".join(MODELS) + " |")
    print("|---|---|" + "---|" * len(MODELS))
    for ds in pacc.index:
        C = int(agg[agg.dataset == ds]['acc'].notna().any() and CLF_REGISTRY[ds][2])
        print(f"| {ds} | {CLF_REGISTRY[ds][2]} | " + " | ".join(f"{pacc.loc[ds, m]:.3f}" for m in MODELS) + " |")
    print(f"\nSaved: classify_results.csv  ({time.time()-t0:.1f}s total)")
