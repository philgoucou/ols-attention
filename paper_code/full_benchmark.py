"""
Full benchmark: Poly PCA(200)+OLS vs Attention(2blk) vs FT-Transformer
All times include preprocessing (PCA fitting, OLS init).
Both attention models get dropout + weight decay for fair comparison.
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
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cpu")

# ── Datasets ──────────────────────────────────────────────────────
def get_california():
    d = fetch_california_housing()
    return d.data, d.target, "California (20K, 8f)"

def get_concrete():
    import pandas as pd
    df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")
    return df.iloc[:,:-1].values, df.iloc[:,-1].values, "Concrete (1K, 8f)"

def get_energy():
    import pandas as pd
    df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx").dropna()
    return df.iloc[:,:8].values, df.iloc[:,8].values, "Energy (768, 8f)"

def get_wine():
    import pandas as pd
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
    return df.iloc[:,:-1].values, df.iloc[:,-1].values, "Wine (1.6K, 11f)"

def get_abalone():
    import pandas as pd
    cols = ["sex","length","diameter","height","whole","shucked","viscera","shell","rings"]
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", header=None, names=cols)
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

# ── Shared ────────────────────────────────────────────────────────
def ffn(d, h, drop=DROPOUT):
    return nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h,d), nn.Dropout(drop))

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

# ── Model 1: Simple Attention (2 blocks) ─────────────────────────
class AttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, NH, batch_first=True, dropout=DROPOUT)
        self.n1 = nn.LayerNorm(D); self.ffn = ffn(D,H); self.n2 = nn.LayerNorm(D)
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        a, _ = self.attn(x,x,x); x = self.n1(x + self.drop(a)); return self.n2(x + self.ffn(x))

class AttnModel(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.embed = nn.Linear(1,D); self.pos = nn.Parameter(torch.randn(1,nf,D)*0.02)
        self.blocks = nn.Sequential(AttnBlock(), AttnBlock())
        self.head = nn.Linear(D,1)
    def forward(self, x):
        x = self.embed(x.unsqueeze(-1)) + self.pos; return self.head(self.blocks(x).mean(1))

# ── Model 2: FT-Transformer ──────────────────────────────────────
# Gorishniy et al. 2021: Feature Tokenizer + [CLS] + Transformer
class FTTransformer(nn.Module):
    def __init__(self, nf, n_blocks=3):
        super().__init__()
        # Feature Tokenizer: per-feature linear embedding + bias
        self.feat_embeds = nn.ModuleList([nn.Linear(1, D) for _ in range(nf)])
        # Learned [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        # Positional is implicit in per-feature embeddings
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=NH, dim_feedforward=H, dropout=DROPOUT,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    def forward(self, x):
        # x: (B, nf)
        B = x.size(0)
        # Per-feature tokenization
        tokens = [self.feat_embeds[i](x[:, i:i+1]) for i in range(x.size(1))]  # list of (B, D)
        tokens = torch.stack(tokens, dim=1)  # (B, nf, D)
        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, nf+1, D)
        # Transformer
        z = self.transformer(tokens)
        # Read from CLS position
        cls_out = self.norm(z[:, 0, :])
        return self.head(cls_out)

# ── Model 3: Poly PCA + OLS ──────────────────────────────────────
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
    n_comp = min(n_comp, phi.shape[1], phi.shape[0])
    pca = PCA(n_components=n_comp, random_state=SEED); pca.fit(phi.numpy())
    return FrozenPCA(pca.mean_, pca.components_)

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
def get_mse(model, loader):
    model.eval(); crit = nn.MSELoss(); tot = n = 0
    with torch.no_grad():
        for X, y in loader:
            tot += crit(model(X), y).item() * X.size(0); n += X.size(0)
    return tot / n

def train_model(model, tr, te, y_var, epochs=EPOCHS):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss(); best = -1e9; t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        for X, y in tr:
            loss = crit(model(X), y); opt.zero_grad(); loss.backward(); opt.step()
        r2 = 1 - get_mse(model, te) / y_var; best = max(best, r2)
    elapsed = time.time() - t0
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return best, elapsed, npar

# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = [get_california(), get_concrete(), get_energy(), get_wine(), get_abalone()]
    datasets = [d for d in datasets if d is not None]

    # Collect all results
    all_results = {}  # ds_name → list of (model_name, r2, total_time, train_time, prep_time, params)

    for X, y, ds_name in datasets:
        nf = X.shape[1]
        Xtr, Xte, ytr, yte, y_var, tr, te = prepare(X, y)
        print(f"\n{'='*70}")
        print(f"  {ds_name}")
        print(f"{'='*70}")
        rows = []

        # ── Attention (2 blocks) ──
        torch.manual_seed(SEED)
        m = AttnModel(nf)
        t_total_start = time.time()
        r2, train_t, p = train_model(m, tr, te, y_var)
        total_t = time.time() - t_total_start
        rows.append(("Attention (2 blk)", r2, total_t, train_t, 0.0, p))
        print(f"  Attention:     R²={r2:.4f}  total={total_t:.1f}s  params={p:,}")

        # ── FT-Transformer (3 blocks) ──
        torch.manual_seed(SEED)
        m = FTTransformer(nf, n_blocks=3)
        t_total_start = time.time()
        r2, train_t, p = train_model(m, tr, te, y_var)
        total_t = time.time() - t_total_start
        rows.append(("FT-Transformer (3 blk)", r2, total_t, train_t, 0.0, p))
        print(f"  FT-Transformer: R²={r2:.4f}  total={total_t:.1f}s  params={p:,}")

        # ── Poly PCA(200) + OLS (all prep time included) ──
        torch.manual_seed(SEED)
        t_total_start = time.time()
        pca_layer = fit_pca_for(Xtr, nf, n_comp=200)
        torch.manual_seed(SEED)
        m = PolyPCAModel(nf, pca_layer)
        ols_init(m, Xtr, ytr)
        prep_t = time.time() - t_total_start
        r2, train_t, p = train_model(m, tr, te, y_var)
        total_t = time.time() - t_total_start
        rows.append(("Poly PCA(200)+OLS", r2, total_t, train_t, prep_t, p))
        print(f"  Poly PCA+OLS:  R²={r2:.4f}  total={total_t:.1f}s (prep={prep_t:.1f}s + train={train_t:.1f}s)  params={p:,}")

        all_results[ds_name] = rows

    # ── Grand summary ─────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("  FULL BENCHMARK RESULTS (dropout={}, weight_decay={}, {} epochs)".format(DROPOUT, WEIGHT_DECAY, EPOCHS))
    print(f"{'='*100}")
    print(f"  {'Dataset':<22} {'Model':<25} {'R²':>7} {'Total(s)':>9} {'Prep(s)':>8} {'Train(s)':>9} {'Params':>9}")
    print(f"  {'-'*95}")
    for ds_name, rows in all_results.items():
        for i, (mname, r2, total_t, train_t, prep_t, p) in enumerate(rows):
            ds_col = ds_name if i == 0 else ""
            print(f"  {ds_col:<22} {mname:<25} {r2:>7.4f} {total_t:>9.1f} {prep_t:>8.1f} {train_t:>9.1f} {p:>9,}")
        print(f"  {'-'*95}")

    # ── Compact comparison table ──────────────────────────────────
    print(f"\n  COMPACT: Poly PCA+OLS vs best attention-based model")
    print(f"  {'Dataset':<22} {'Best Attn R²':>12} {'Poly R²':>9} {'Attn time':>10} {'Poly time':>10} {'Speedup':>8}")
    print(f"  {'-'*75}")
    for ds_name, rows in all_results.items():
        attn_best_r2 = max(r[1] for r in rows[:2])  # best of Attn and FT-T
        attn_best_t = min(r[2] for r in rows[:2] if r[1] == attn_best_r2)
        attn_best_name = [r[0] for r in rows[:2] if r[1] == attn_best_r2][0]
        poly_r2 = rows[2][1]
        poly_t = rows[2][2]
        speedup = attn_best_t / poly_t if poly_t > 0 else 0
        delta = poly_r2 - attn_best_r2
        winner = "Poly" if delta > 0 else "Attn"
        print(f"  {ds_name:<22} {attn_best_r2:>7.4f} ({attn_best_name[:5]:>5}) {poly_r2:>7.4f}  {attn_best_t:>9.1f} {poly_t:>9.1f}  {speedup:>7.2f}x  {winner}")
