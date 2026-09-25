"""
PCA(200)+OLS vs Attention(2 blocks) across multiple regression datasets.
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

warnings.filterwarnings("ignore")
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
D, H, EPOCHS, BS, LR, NH = 64, 128, 50, 256, 1e-3, 4
DEVICE = torch.device("cpu")

# ── Datasets ──────────────────────────────────────────────────────
def get_california():
    d = fetch_california_housing()
    return d.data, d.target, "California Housing (20K, 8f)"

def get_concrete():
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    df = pd.read_excel(url); return df.iloc[:,:-1].values, df.iloc[:,-1].values, "Concrete (1K, 8f)"

def get_energy():
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    df = pd.read_excel(url).dropna(); return df.iloc[:,:8].values, df.iloc[:,8].values, "Energy Eff. (768, 8f)"

def get_wine():
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";"); return df.iloc[:,:-1].values, df.iloc[:,-1].values, "Wine Quality (1.6K, 11f)"

def get_abalone():
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    cols = ["sex","length","diameter","height","whole","shucked","viscera","shell","rings"]
    df = pd.read_csv(url, header=None, names=cols)
    df = pd.get_dummies(df, columns=["sex"], drop_first=False)
    return df.drop("rings",axis=1).values, df["rings"].values, "Abalone (4.2K, 10f)"

def prepare(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X.astype(np.float64), y.astype(np.float64), test_size=0.2, random_state=SEED)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)
    ytr, yte = ytr.astype(np.float32), yte.astype(np.float32)
    y_var = float(np.var(yte))
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).unsqueeze(1)), batch_size=BS, shuffle=True)
    te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte).unsqueeze(1)), batch_size=BS, shuffle=False)
    return Xtr, Xte, ytr, yte, y_var, tr, te

# ── Building blocks ───────────────────────────────────────────────
def ffn(d,h): return nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,d))

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

# ── Attention model ───────────────────────────────────────────────
class AttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, NH, batch_first=True)
        self.n1 = nn.LayerNorm(D); self.ffn = ffn(D,H); self.n2 = nn.LayerNorm(D)
    def forward(self, x):
        a, _ = self.attn(x,x,x); x = self.n1(x+a); return self.n2(x + self.ffn(x))

class AttnModel(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.embed = nn.Linear(1,D); self.pos = nn.Parameter(torch.randn(1,nf,D)*0.02)
        self.blocks = nn.Sequential(AttnBlock(), AttnBlock())
        self.head = nn.Linear(D,1)
    def forward(self, x):
        x = self.embed(x.unsqueeze(-1)) + self.pos; return self.head(self.blocks(x).mean(1))

# ── Poly + PCA model ─────────────────────────────────────────────
class PolyPCAModel(nn.Module):
    def __init__(self, nf, pca_layer):
        super().__init__()
        self.pca = pca_layer
        self.embed = nn.Linear(1,D); self.pos = nn.Parameter(torch.randn(1,nf,D)*0.02)
        self.W = nn.Linear(pca_layer.out_dim, D)
        self.n1 = nn.LayerNorm(D); self.ffn = ffn(D,H); self.n2 = nn.LayerNorm(D)
        self.head = nn.Linear(D,1)
    def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
    def get_poly(self, emb): return self.pca(poly_cross(emb))
    def forward(self, x):
        emb = self.embed_x(x); phi = self.get_poly(emb)
        z = self.n1(emb + self.W(phi)); return self.head(self.n2(z + self.ffn(z)).mean(1))

def fit_pca_for(Xtr, nf, n_comp=200):
    torch.manual_seed(SEED)
    embed = nn.Linear(1,D); pos = nn.Parameter(torch.randn(1,nf,D)*0.02)
    with torch.no_grad():
        emb = embed(torch.from_numpy(Xtr).unsqueeze(-1)) + pos
        phi = poly_cross(emb).reshape(-1, emb.shape[-1]*2 + D*nf*(nf+1)//2)
    # Cap n_comp at actual feature count
    actual_dim = phi.shape[1]
    n_comp = min(n_comp, actual_dim, phi.shape[0])
    pca = PCA(n_components=n_comp, random_state=SEED); pca.fit(phi.numpy())
    return FrozenPCA(pca.mean_, pca.components_), pca.explained_variance_ratio_.sum()

def ols_init(model, Xtr, ytr):
    model.eval()
    with torch.no_grad():
        emb = model.embed_x(torch.from_numpy(Xtr))
        phi = model.get_poly(emb).mean(1).numpy()
    reg = Ridge(alpha=1.0); reg.fit(phi, ytr)
    with torch.no_grad():
        model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32))
        model.W.bias.data.zero_()
        model.head.weight.data.zero_(); model.head.weight.data[0,0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

# ── Training ──────────────────────────────────────────────────────
def train_model(model, tr, te, y_var):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss(); best = -1e9; t0 = time.time(); hist = []
    for ep in range(1, EPOCHS+1):
        model.train()
        for X, y in tr:
            loss = crit(model(X), y); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mse = sum(crit(model(X),y).item()*X.size(0) for X,y in te) / sum(X.size(0) for X,y in te)
        r2 = 1 - mse/y_var; best = max(best, r2); hist.append((ep, r2, time.time()-t0))
    elapsed = time.time()-t0
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return best, elapsed, npar, hist

def time_to_r2(hist, threshold):
    for _, r2, t in hist:
        if r2 >= threshold:
            return t
    return None

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = [get_california(), get_concrete(), get_energy(), get_wine(), get_abalone()]
    datasets = [d for d in datasets if d is not None]

    all_rows = []
    for X, y, ds_name in datasets:
        nf = X.shape[1]
        Xtr, Xte, ytr, yte, y_var, tr, te = prepare(X, y)
        print(f"\n{'='*70}")
        print(f"  {ds_name}  |  y_var={y_var:.4f}")
        print(f"{'='*70}")

        # Attention
        torch.manual_seed(SEED)
        m = AttnModel(nf)
        ar2, at, ap, ahist = train_model(m, tr, te, y_var)

        # PCA(200) + OLS
        torch.manual_seed(SEED)
        pca_layer, var_exp = fit_pca_for(Xtr, nf, n_comp=200)
        n_comp_actual = pca_layer.out_dim
        torch.manual_seed(SEED)
        m = PolyPCAModel(nf, pca_layer)
        ols_init(m, Xtr, ytr)
        pr2, pt, pp, phist = train_model(m, tr, te, y_var)

        print(f"  {'Model':<30} {'R²':>7} {'Time':>7} {'Params':>8} {'PCA dims':>9}")
        print(f"  {'-'*65}")
        print(f"  {'Attention (2 blocks)':<30} {ar2:>7.4f} {at:>6.1f}s {ap:>8,}")
        print(f"  {'Poly PCA({n})+OLS'.format(n=n_comp_actual):<30} {pr2:>7.4f} {pt:>6.1f}s {pp:>8,} {n_comp_actual:>9}")

        # Time-to thresholds
        t75_a, t75_p = time_to_r2(ahist, 0.50), time_to_r2(phist, 0.50)
        print(f"\n  Time to R²≥0.50: Attn={'%.1fs'%t75_a if t75_a else 'never':>8}  Poly={'%.1fs'%t75_p if t75_p else 'never':>8}")
        t75_a, t75_p = time_to_r2(ahist, 0.60), time_to_r2(phist, 0.60)
        print(f"  Time to R²≥0.60: Attn={'%.1fs'%t75_a if t75_a else 'never':>8}  Poly={'%.1fs'%t75_p if t75_p else 'never':>8}")

        all_rows.append((ds_name, ar2, at, ap, pr2, pt, pp, n_comp_actual))

    # Final summary
    print(f"\n\n{'='*85}")
    print("  CROSS-DATASET SUMMARY")
    print(f"{'='*85}")
    print(f"  {'Dataset':<30} {'Attn R²':>8} {'Poly R²':>8} {'Δ R²':>7} {'Attn(s)':>8} {'Poly(s)':>8} {'Speedup':>8}")
    print(f"  {'-'*80}")
    for ds, ar2, at, ap, pr2, pt, pp, nc in all_rows:
        delta = pr2 - ar2
        speedup = at / pt if pt > 0 else 0
        print(f"  {ds:<30} {ar2:>8.4f} {pr2:>8.4f} {delta:>+7.4f} {at:>7.1f}s {pt:>7.1f}s {speedup:>7.2f}x")
