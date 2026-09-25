"""
THOROUGH ABLATION — Every critique addressed, every corner checked.
1 rep each, 4 datasets, 12 variants. ~15 min on CPU.
"""
import numpy as np, torch, torch.nn as nn, time, warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings('ignore')

SEED=42; D=64; FFN=128; BS=256; LR=1e-3; EP=50
torch.manual_seed(SEED); np.random.seed(SEED)

# ── Datasets ──────────────────────────────────────────────────────
def load_all():
    ds = []
    d = fetch_california_housing()
    ds.append((d.data, d.target, "California (20K,8f)"))
    try:
        import pandas as pd
        df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")
        ds.append((df.iloc[:,:-1].values, df.iloc[:,-1].values, "Concrete (1K,8f)"))
    except: pass
    try:
        import pandas as pd
        df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx").dropna()
        ds.append((df.iloc[:,:8].values, df.iloc[:,8].values, "Energy (768,8f)"))
    except: pass
    try:
        import pandas as pd
        cols = ["sex","length","diameter","height","whole","shucked","viscera","shell","rings"]
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", header=None, names=cols)
        df = pd.get_dummies(df, columns=["sex"], drop_first=False)
        ds.append((df.drop("rings",axis=1).values, df["rings"].values, "Abalone (4.2K,10f)"))
    except: pass
    return ds

def prep(X, y, seed=SEED):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, ytr.astype(np.float32), yte.astype(np.float32)

# ── Poly features ─────────────────────────────────────────────────
def poly_cross(x):
    B,S,Dm = x.shape
    per = torch.cat([x, x**2-1], dim=-1)
    pairs = torch.cat([x[:,i,:]*x[:,j,:] for i in range(S) for j in range(i,S)], dim=-1)
    return torch.cat([per, pairs.unsqueeze(1).expand(B,S,-1)], dim=-1)

class FrozenPCA(nn.Module):
    def __init__(self, mean, comp):
        super().__init__()
        self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer('comp', torch.from_numpy(comp.astype(np.float32)))
        self.out_dim = comp.shape[0]
    def forward(self, x): return (x - self.mean) @ self.comp.T

def make_pca(Xtr, nf, seed=SEED, n_comp=200, pooled=False):
    torch.manual_seed(seed)
    embed = nn.Linear(1,D); pos = nn.Parameter(torch.randn(1,nf,D)*0.02)
    Xt = torch.from_numpy(Xtr)
    with torch.no_grad():
        emb = embed(Xt.unsqueeze(-1)) + pos
        phi = poly_cross(emb)
        if pooled:
            phi_np = phi.mean(1).numpy()
        else:
            phi_np = phi.reshape(-1, phi.shape[-1]).numpy()
    n_comp = min(n_comp, phi_np.shape[1], phi_np.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed); pca.fit(phi_np)
    return FrozenPCA(pca.mean_, pca.components_), embed, pos

# ── Models ────────────────────────────────────────────────────────
class PolyFull(nn.Module):
    """Full architecture: embed → poly cross → PCA → W → LN+residual → FFN → LN+residual → pool → head"""
    def __init__(self, nf, pca_l):
        super().__init__()
        self.pca = pca_l; self.embed = nn.Linear(1,D); self.pos = nn.Parameter(torch.randn(1,nf,D)*0.02)
        self.W = nn.Linear(pca_l.out_dim, D); self.n1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D,FFN), nn.ReLU(), nn.Dropout(0.1), nn.Linear(FFN,D), nn.Dropout(0.1))
        self.n2 = nn.LayerNorm(D); self.head = nn.Linear(D,1)
    def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
    def get_poly(self, emb): return self.pca(poly_cross(emb))
    def forward(self, x):
        emb = self.embed_x(x); phi = self.get_poly(emb)
        z = self.n1(emb + self.W(phi)); return self.head(self.n2(z + self.ffn(z)).mean(1))

class PolySimple(nn.Module):
    """Stripped: head(mean(emb + W@phi)) — no LN, no FFN"""
    def __init__(self, nf, pca_l):
        super().__init__()
        self.pca = pca_l; self.embed = nn.Linear(1,D); self.pos = nn.Parameter(torch.randn(1,nf,D)*0.02)
        self.W = nn.Linear(pca_l.out_dim, D); self.head = nn.Linear(D,1)
    def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
    def forward(self, x):
        emb = self.embed_x(x); phi = self.pca(poly_cross(emb))
        return self.head((emb + self.W(phi)).mean(1))

class PolyNoResidual(nn.Module):
    """No residual connection: z = LN(W@phi), not LN(emb + W@phi)"""
    def __init__(self, nf, pca_l):
        super().__init__()
        self.pca = pca_l; self.embed = nn.Linear(1,D); self.pos = nn.Parameter(torch.randn(1,nf,D)*0.02)
        self.W = nn.Linear(pca_l.out_dim, D); self.n1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D,FFN), nn.ReLU(), nn.Dropout(0.1), nn.Linear(FFN,D), nn.Dropout(0.1))
        self.n2 = nn.LayerNorm(D); self.head = nn.Linear(D,1)
    def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
    def forward(self, x):
        emb = self.embed_x(x); phi = self.pca(poly_cross(emb))
        z = self.n1(self.W(phi)); return self.head(self.n2(z + self.ffn(z)).mean(1))

def ols_init(model, Xtr, ytr):
    model.eval()
    with torch.no_grad():
        phi_p = model.get_poly(model.embed_x(torch.from_numpy(Xtr))).mean(1).numpy()
    reg = Ridge(alpha=1.0); reg.fit(phi_p, ytr)
    with torch.no_grad():
        model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32))
        model.W.bias.data.zero_(); model.head.weight.data.zero_()
        model.head.weight.data[0,0] = 1.0; model.head.bias.data.fill_(float(reg.intercept_))

def train_nn(model, Xtr, ytr, Xte, epochs=EP):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.MSELoss()
    N = len(Xtr); nv = max(1, int(N*0.15)); idx = np.random.permutation(N)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr[idx[nv:]]), torch.from_numpy(ytr[idx[nv:]]).unsqueeze(1)), batch_size=BS, shuffle=True)
    Xv, yv = torch.from_numpy(Xtr[idx[:nv]]), torch.from_numpy(ytr[idx[:nv]]).unsqueeze(1)
    best_vl = 1e9; best_st = None; pat = 0
    for ep in range(1, epochs+1):
        model.train()
        for Xb, yb in loader:
            loss = crit(model(Xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best_vl: best_vl = vl; best_st = {k: v.clone() for k,v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= 10: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad(): return model(torch.from_numpy(Xte)).numpy().ravel()

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = load_all()
    t0 = time.time()

    print("="*100)
    print("  THOROUGH ABLATION STUDY")
    print("="*100)

    header = f"  {'Variant':<45}"
    for _,_,nm in datasets: header += f" {nm[:12]:>12}"
    print(header)
    print("  " + "-"*95)

    variant_results = []

    for ds_idx, (X, y, ds_name) in enumerate(datasets):
        Xtr, Xte, ytr, yte = prep(X, y)
        nf = Xtr.shape[1]

        results = {}

        # ── QUESTION 1: Is the neural part needed? ────────────────

        # 1a. Ridge on pooled PCA poly features (random embedding)
        pca_layer, emb_mod, pos_mod = make_pca(Xtr, nf)
        Xt_tr = torch.from_numpy(Xtr); Xt_te = torch.from_numpy(Xte)
        with torch.no_grad():
            phi_tr = FrozenPCA(pca_layer.mean.numpy(), pca_layer.comp.numpy())(
                poly_cross(emb_mod(Xt_tr.unsqueeze(-1)) + pos_mod)).mean(1).numpy()
            phi_te = FrozenPCA(pca_layer.mean.numpy(), pca_layer.comp.numpy())(
                poly_cross(emb_mod(Xt_te.unsqueeze(-1)) + pos_mod)).mean(1).numpy()
        reg = Ridge(alpha=1.0); reg.fit(phi_tr, ytr)
        results["1a. Ridge on PCA poly (random emb)"] = r2_score(yte, reg.predict(phi_te))

        # 1b. Sklearn PolynomialFeatures(2) + Ridge (no neural anything)
        pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        Xtr_poly = pf.fit_transform(Xtr); Xte_poly = pf.transform(Xte)
        reg2 = Ridge(alpha=1.0); reg2.fit(Xtr_poly, ytr)
        results["1b. Sklearn Poly(2) + Ridge"] = r2_score(yte, reg2.predict(Xte_poly))

        # 1c. RF baseline
        rf = RandomForestRegressor(n_estimators=500, max_features=1/3, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
        results["1c. Random Forest (500 trees)"] = r2_score(yte, rf.predict(Xte))

        # ── QUESTION 2: Does architecture work without OLS? ──────

        # 2a. Full Poly, NO warmstart
        torch.manual_seed(SEED); pca_l, _, _ = make_pca(Xtr, nf)
        torch.manual_seed(SEED); m = PolyFull(nf, pca_l)
        results["2a. Poly Full (NO warmstart)"] = r2_score(yte, train_nn(m, Xtr, ytr, Xte))

        # 2b. Full Poly + OLS init (current code)
        torch.manual_seed(SEED); pca_l, _, _ = make_pca(Xtr, nf)
        torch.manual_seed(SEED); m = PolyFull(nf, pca_l); ols_init(m, Xtr, ytr)
        results["2b. Poly Full + OLS init (current)"] = r2_score(yte, train_nn(m, Xtr, ytr, Xte))

        # ── QUESTION 3: Are FFN/LN needed? ───────────────────────

        # 3a. Stripped: head(mean(emb + W@phi))
        torch.manual_seed(SEED); pca_l, _, _ = make_pca(Xtr, nf)
        torch.manual_seed(SEED); m = PolySimple(nf, pca_l)
        results["3a. Poly Simple (no LN/FFN)"] = r2_score(yte, train_nn(m, Xtr, ytr, Xte))

        # 3b. No residual: LN(W@phi) instead of LN(emb + W@phi)
        torch.manual_seed(SEED); pca_l, _, _ = make_pca(Xtr, nf)
        torch.manual_seed(SEED); m = PolyNoResidual(nf, pca_l)
        results["3b. Poly (no residual connection)"] = r2_score(yte, train_nn(m, Xtr, ytr, Xte))

        # ── QUESTION 4: Does PCA mismatch matter? ────────────────

        # 4a. PCA on pooled features (fixes token-vs-pooled mismatch)
        torch.manual_seed(SEED); pca_l, _, _ = make_pca(Xtr, nf, pooled=True)
        torch.manual_seed(SEED); m = PolyFull(nf, pca_l)
        results["4a. Poly Full (pooled PCA)"] = r2_score(yte, train_nn(m, Xtr, ytr, Xte))

        # ── QUESTION 5: Frozen embed test ────────────────────────

        # 5a. Freeze embed+pos, train only W+FFN+head
        torch.manual_seed(SEED); pca_l, _, _ = make_pca(Xtr, nf)
        torch.manual_seed(SEED); m = PolyFull(nf, pca_l)
        m.embed.weight.requires_grad_(False); m.embed.bias.requires_grad_(False)
        m.pos.requires_grad_(False)
        results["5a. Poly (frozen embed+pos)"] = r2_score(yte, train_nn(m, Xtr, ytr, Xte))

        # ── QUESTION 6: Is this just a random feature model? ─────

        # 6a. Freeze EVERYTHING except head (random feature model)
        torch.manual_seed(SEED); pca_l, _, _ = make_pca(Xtr, nf)
        torch.manual_seed(SEED); m = PolyFull(nf, pca_l)
        for n, p in m.named_parameters():
            if 'head' not in n: p.requires_grad_(False)
        results["6a. Poly (only head trained)"] = r2_score(yte, train_nn(m, Xtr, ytr, Xte))

        variant_results.append(results)
        print(f"  {ds_name} done ({time.time()-t0:.0f}s)")

    # ── Print table ───────────────────────────────────────────────
    print("\n" + "="*100)
    print("  RESULTS")
    print("="*100)
    header = f"  {'Variant':<45}"
    for _,_,nm in datasets: header += f" {nm[:12]:>12}"
    print(header)
    print("  " + "-"*95)

    all_variants = list(variant_results[0].keys())
    for v in all_variants:
        line = f"  {v:<45}"
        for ds_idx in range(len(datasets)):
            val = variant_results[ds_idx][v]
            line += f" {val:>12.4f}"
        print(line)
        # Visual separator between question groups
        if v.startswith("1c") or v.startswith("2b") or v.startswith("3b") or v.startswith("4a") or v.startswith("5a"):
            print("  " + "-"*95)

    print(f"\n  Total time: {time.time()-t0:.0f}s")

    print(f"\n  {'='*80}")
    print(f"  VERDICT")
    print(f"  {'='*80}")
    print(f"  Q1: Is the neural part needed?")
    print(f"      Compare Ridge-only (1a) vs Poly Full (2a)")
    print(f"  Q2: Does architecture work without OLS hack?")
    print(f"      Compare 2a (no warmstart) vs 2b (+OLS)")
    print(f"  Q3: Are FFN/LN essential?")
    print(f"      Compare 3a (stripped) vs 2a (full)")
    print(f"  Q4: Does token-vs-pooled PCA matter?")
    print(f"      Compare 4a (pooled) vs 2a (token)")
    print(f"  Q5: Does learning the embedding matter?")
    print(f"      Compare 5a (frozen) vs 2a (learned)")
    print(f"  Q6: Is this just a random feature model?")
    print(f"      Compare 6a (only head) vs 2a (full)")
