"""
Ablation study addressing ChatGPT's critique:
  1. Ridge-only on pooled PCA poly features (is the neural part needed?)
  2. Poly block, NO warm start (does the architecture work on its own?)
  3. Current hybrid (Poly + PCA + OLS warm start)
  4. Simplified: head(mean(W @ phi)) — no FFN, no LayerNorm (is the skeleton needed?)
  5. PCA on pooled features instead of token-level (fixes the mismatch)

Run on all 5 original datasets with 5 repeats.
"""
import time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
D, FFN_DIM, EPOCHS, BS, LR = 64, 128, 50, 256, 1e-3
DROPOUT, WD = 0.1, 1e-4
N_REPEATS = 5
DEVICE = torch.device("cpu")

# ── Datasets ──────────────────────────────────────────────────────
def get_datasets():
    ds = []
    d = fetch_california_housing()
    ds.append((d.data, d.target, "California (20K, 8f)"))
    try:
        import pandas as pd
        df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")
        ds.append((df.iloc[:,:-1].values, df.iloc[:,-1].values, "Concrete (1K, 8f)"))
    except: pass
    try:
        import pandas as pd
        df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx").dropna()
        ds.append((df.iloc[:,:8].values, df.iloc[:,8].values, "Energy (768, 8f)"))
    except: pass
    try:
        import pandas as pd
        cols = ["sex","length","diameter","height","whole","shucked","viscera","shell","rings"]
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", header=None, names=cols)
        df = pd.get_dummies(df, columns=["sex"], drop_first=False)
        ds.append((df.drop("rings",axis=1).values, df["rings"].values, "Abalone (4.2K, 10f)"))
    except: pass
    return ds

# ── Shared poly features ─────────────────────────────────────────
def poly_cross(x):
    B, S, Dm = x.shape
    per = torch.cat([x, x**2 - 1], dim=-1)
    pairs = torch.cat([x[:,i,:]*x[:,j,:] for i in range(S) for j in range(i,S)], dim=-1)
    return torch.cat([per, pairs.unsqueeze(1).expand(B,S,-1)], dim=-1)

class FrozenPCA(nn.Module):
    def __init__(self, mean, components):
        super().__init__()
        self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer('comp', torch.from_numpy(components.astype(np.float32)))
        self.out_dim = components.shape[0]
    def forward(self, x): return (x - self.mean) @ self.comp.T

def compute_pca_and_features(Xtr, nf, n_comp=200, seed=42, pooled_pca=False):
    """Compute PCA basis and return (pca_layer, embed, pos) all from same init."""
    torch.manual_seed(seed)
    embed = nn.Linear(1, D)
    pos = nn.Parameter(torch.randn(1, nf, D)*0.02)
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    with torch.no_grad():
        emb = embed(Xt.unsqueeze(-1)) + pos
        phi = poly_cross(emb)
        if pooled_pca:
            phi_for_pca = phi.mean(dim=1).numpy()      # (N, poly_dim)
        else:
            phi_for_pca = phi.reshape(-1, phi.shape[-1]).numpy()  # (N*S, poly_dim)
    n_comp = min(n_comp, phi_for_pca.shape[1], phi_for_pca.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(phi_for_pca)
    return FrozenPCA(pca.mean_, pca.components_), embed, pos

# ── Model 1: Ridge-only on pooled PCA poly features ──────────────
def train_ridge_only(Xtr, ytr, Xte, seed=42):
    """Pure ridge on the exact same features the neural model sees."""
    nf = Xtr.shape[1]
    pca_layer, embed, pos = compute_pca_and_features(Xtr, nf, seed=seed)
    Xt_tr = torch.from_numpy(Xtr.astype(np.float32))
    Xt_te = torch.from_numpy(Xte.astype(np.float32))
    with torch.no_grad():
        emb_tr = embed(Xt_tr.unsqueeze(-1)) + pos
        phi_tr = pca_layer(poly_cross(emb_tr)).mean(1).numpy()
        emb_te = embed(Xt_te.unsqueeze(-1)) + pos
        phi_te = pca_layer(poly_cross(emb_te)).mean(1).numpy()
    reg = Ridge(alpha=1.0); reg.fit(phi_tr, ytr)
    return reg.predict(phi_te)

# ── Model 2: Poly block, NO warm start ───────────────────────────
class PolyModel(nn.Module):
    def __init__(self, nf, pca_layer, d=D, h=FFN_DIM, drop=DROPOUT):
        super().__init__()
        self.pca = pca_layer
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, nf, d)*0.02)
        self.W = nn.Linear(pca_layer.out_dim, d)
        self.n1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h,d), nn.Dropout(drop))
        self.n2 = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
    def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
    def get_poly(self, emb): return self.pca(poly_cross(emb))
    def forward(self, x):
        emb = self.embed_x(x); phi = self.get_poly(emb)
        z = self.n1(emb + self.W(phi))
        return self.head(self.n2(z + self.ffn(z)).mean(1))

# ── Model 3: Simplified — no FFN, no LayerNorm ───────────────────
class PolySimple(nn.Module):
    """Just head(mean(W @ phi)) — stripped architecture."""
    def __init__(self, nf, pca_layer, d=D):
        super().__init__()
        self.pca = pca_layer
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, nf, d)*0.02)
        self.W = nn.Linear(pca_layer.out_dim, d)
        self.head = nn.Linear(d, 1)
    def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
    def forward(self, x):
        emb = self.embed_x(x)
        phi = self.pca(poly_cross(emb))
        return self.head((emb + self.W(phi)).mean(1))

# ── OLS warm start (the "hack") ──────────────────────────────────
def ols_init(model, Xtr, ytr):
    model.eval()
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    with torch.no_grad():
        phi_pooled = model.get_poly(model.embed_x(Xt)).mean(1).numpy()
    reg = Ridge(alpha=1.0); reg.fit(phi_pooled, ytr)
    with torch.no_grad():
        model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32))
        model.W.bias.data.zero_()
        model.head.weight.data.zero_(); model.head.weight.data[0,0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

# ── Training ──────────────────────────────────────────────────────
def train_nn(model, Xtr, ytr, Xte, epochs=EPOCHS):
    model.to(DEVICE)
    Xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.from_numpy(ytr.astype(np.float32)).unsqueeze(1)

    # Val split for early stopping
    N = Xtr.shape[0]; n_val = max(1, int(N*0.15))
    idx = np.random.permutation(N)
    tr_loader = DataLoader(TensorDataset(Xt[idx[n_val:]], yt[idx[n_val:]]), batch_size=BS, shuffle=True)
    Xv, yv = Xt[idx[:n_val]].to(DEVICE), yt[idx[:n_val]].to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.MSELoss()
    best_vl = float('inf'); best_state = None; patience = 0

    for ep in range(1, epochs+1):
        model.train()
        for Xb, yb in tr_loader:
            loss = crit(model(Xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        if vl < best_vl:
            best_vl = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 10: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(Xte.astype(np.float32))).numpy().ravel()

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = get_datasets()
    models_list = [
        "Ridge-only",          # pure ridge on pooled PCA features
        "Poly (no warmstart)", # full architecture, random init
        "Poly (+ OLS init)",   # full architecture + OLS hack
        "Poly-Simple (no LN/FFN)", # stripped: head(mean(emb + W@phi))
        "Poly (pooled PCA)",   # PCA fit on pooled features instead of token-level
    ]

    print(f"{'='*95}")
    print(f"  ABLATION STUDY — {N_REPEATS} repeats per dataset")
    print(f"{'='*95}\n")

    all_results = []

    for X, y, ds_name in datasets:
        print(f"  {ds_name}")
        scores = {m: [] for m in models_list}

        for r in range(N_REPEATS):
            seed = SEED + r * 1000
            np.random.seed(seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
            ytr, yte = ytr.astype(np.float32), yte.astype(np.float32)
            nf = Xtr.shape[1]

            # 1. Ridge-only
            pred = train_ridge_only(Xtr, ytr, Xte, seed=seed)
            scores["Ridge-only"].append(r2_score(yte, pred))

            # 2. Poly no warmstart
            torch.manual_seed(seed)
            pca_layer, _, _ = compute_pca_and_features(Xtr, nf, seed=seed)
            torch.manual_seed(seed)
            m = PolyModel(nf, pca_layer)
            pred = train_nn(m, Xtr, ytr, Xte)
            scores["Poly (no warmstart)"].append(r2_score(yte, pred))

            # 3. Poly + OLS init (current hybrid)
            torch.manual_seed(seed)
            pca_layer, _, _ = compute_pca_and_features(Xtr, nf, seed=seed)
            torch.manual_seed(seed)
            m = PolyModel(nf, pca_layer)
            ols_init(m, Xtr, ytr)
            pred = train_nn(m, Xtr, ytr, Xte)
            scores["Poly (+ OLS init)"].append(r2_score(yte, pred))

            # 4. Simplified — no FFN, no LayerNorm
            torch.manual_seed(seed)
            pca_layer, _, _ = compute_pca_and_features(Xtr, nf, seed=seed)
            torch.manual_seed(seed)
            m = PolySimple(nf, pca_layer)
            pred = train_nn(m, Xtr, ytr, Xte)
            scores["Poly-Simple (no LN/FFN)"].append(r2_score(yte, pred))

            # 5. Poly with PCA on pooled features (fixes mismatch)
            torch.manual_seed(seed)
            pca_layer, _, _ = compute_pca_and_features(Xtr, nf, seed=seed, pooled_pca=True)
            torch.manual_seed(seed)
            m = PolyModel(nf, pca_layer)
            ols_init(m, Xtr, ytr)
            pred = train_nn(m, Xtr, ytr, Xte)
            scores["Poly (pooled PCA)"].append(r2_score(yte, pred))

            print(f"    rep {r+1} done")

        row = {'Dataset': ds_name}
        for m in models_list:
            row[m] = np.mean(scores[m])
            row[f'{m}_se'] = np.std(scores[m])
        all_results.append(row)

    # Summary
    print(f"\n{'='*105}")
    print(f"  ABLATION RESULTS ({N_REPEATS} reps)")
    print(f"{'='*105}")
    print(f"  {'Dataset':<25} {'Ridge':>8} {'NoWarm':>8} {'+OLS':>8} {'Simple':>8} {'PoolPCA':>8}")
    print(f"  {'-'*100}")
    for row in all_results:
        line = f"  {row['Dataset']:<25}"
        for m in models_list:
            line += f" {row[m]:>8.4f}"
        print(line)

    print(f"\n  Key questions answered:")
    print(f"  • Ridge-only vs +OLS init: is the neural part needed?")
    print(f"  • No warmstart vs +OLS init: does the architecture work alone?")
    print(f"  • Simple vs full: are FFN/LayerNorm doing anything?")
    print(f"  • Pooled PCA vs token PCA: does fixing the mismatch matter?")
