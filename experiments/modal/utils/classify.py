"""
Standalone classification analogues of FT-Transformer and the Regression Block.

RB classifier = the exact RB architecture (poly-cross -> PCA -> regression mixer
-> residual+FFN, optionally stacked) with two swaps that mirror the paper's story:
  * squared-error readout  -> cross-entropy head (Linear d -> C)
  * Ridge warm-start       -> multinomial-logistic warm-start of block-1's mixer
Everything else is unchanged. This tests "does the OLS-as-attention construction
extend beyond continuous regression?" (xZ71-Q2, Ayqz).

Both trainers return class probabilities (softmax) so the caller can score
accuracy and ROC-AUC uniformly for binary and multiclass.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4
FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128; FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1; FTT_EPOCHS = 200; FTT_BS = 256; FTT_LR = 1e-3; FTT_WD = 1e-4


def _softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _train_loop(model, X_train, y_train, epochs, batch_size, lr, weight_decay,
                device, seed, adamw=False, cosine=False):
    """Shared classification training loop with val early-stopping. y int64."""
    crit = nn.CrossEntropyLoss()
    N = X_train.shape[0]; n_val = max(1, int(N * 0.15))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.int64))
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.int64)).to(device)
    loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    if adamw:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if cosine:
        warm = max(1, epochs // 10)
        def lam(ep):
            if ep < warm: return ep / warm
            prog = (ep - warm) / max(1, epochs - warm)
            return 0.5 * (1 + np.cos(np.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    best, best_state, pat = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if sched: sched.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xt_val), yt_val).item()
        if vl < best:
            best, best_state, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def _predict_proba(model, X_test, device, n_classes):
    model.eval(); logits = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)
            logits.append(model(xb).cpu().numpy())
    return _softmax_np(np.concatenate(logits))


# ================= FT-Transformer classifier =================
def train_ftt_clf(X_train, y_train, X_test, n_classes, n_blocks=FTT_N_BLOCKS,
                  d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS, ffn_dim=FTT_FFN_DIM,
                  dropout=FTT_DROPOUT, epochs=FTT_EPOCHS, batch_size=FTT_BS,
                  lr=FTT_LR, weight_decay=FTT_WD, seed=42, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    nf = X_train.shape[1]; D = d_model

    class FTTClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_embeds = nn.ModuleList([nn.Linear(1, D) for _ in range(nf)])
            self.cls_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=D, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout,
                batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(enc, num_layers=n_blocks)
            self.norm = nn.LayerNorm(D); self.head = nn.Linear(D, n_classes)
        def forward(self, x):
            B = x.size(0)
            tok = torch.stack([self.feat_embeds[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            tok = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
            z = self.transformer(tok)
            return self.head(self.norm(z[:, 0, :]))

    model = FTTClf().to(device)
    model = _train_loop(model, X_train, y_train, epochs, batch_size, lr, weight_decay,
                        device, seed, adamw=True, cosine=True)
    proba = _predict_proba(model, X_test, device, n_classes)
    return proba, {'n_params': sum(p.numel() for p in model.parameters())}


# ================= Regression Block classifier =================
def train_rb_clf(X_train, y_train, X_test, n_classes, n_layers=1, use_pca=True,
                 n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL, ffn_dim=POLY_FFN_DIM,
                 epochs=POLY_EPOCHS, batch_size=POLY_BS, lr=POLY_LR, dropout=POLY_DROPOUT,
                 weight_decay=POLY_WD, seed=42, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    nf = X_train.shape[1]; D = d_model; C = n_classes; pca_fit_cap = 5000

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

    class Identity(nn.Module):
        def __init__(self, dim): super().__init__(); self.out_dim = dim
        def forward(self, x): return x

    class RBBlock(nn.Module):
        def __init__(self, red):
            super().__init__()
            self.red = red; self.W = nn.Linear(red.out_dim, D)
            self.n1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(ffn_dim, D), nn.Dropout(dropout))
            self.n2 = nn.LayerNorm(D)
        def feat(self, h): return self.red(polycross(h))
        def forward(self, h):
            z = self.n1(h + self.W(self.feat(h)))
            return self.n2(z + self.ffn(z))

    class RBClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, D)
            self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.blocks = nn.ModuleList(); self.head = nn.Linear(D, C)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def forward_upto(self, x, upto):
            h = self.embed_x(x)
            for l in range(upto): h = self.blocks[l](h)
            return h
        def forward(self, x):
            h = self.embed_x(x)
            for blk in self.blocks: h = blk(h)
            return self.head(h.mean(1))

    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    X_pca_t = torch.from_numpy(X_pca.astype(np.float32)).to(device)

    torch.manual_seed(seed)
    model = RBClf().to(device)
    with torch.no_grad():
        feat_dim = polycross(model.embed_x(X_pca_t[:8])).shape[-1]
    for l in range(n_layers):
        model.eval()
        with torch.no_grad():
            h_in = model.forward_upto(X_pca_t, upto=l)
            phi = polycross(h_in).reshape(-1, feat_dim).cpu().numpy()
        if use_pca:
            n_comp = min(n_components, phi.shape[1], phi.shape[0])
            p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
            red = FrozenPCA(p.mean_, p.components_)
        else:
            red = Identity(feat_dim)
        torch.manual_seed(seed + 100 + l)
        model.blocks.append(RBBlock(red).to(device))

    # multinomial-logistic warm-start of block-0 mixer (analog of Ridge warm-start)
    if C <= D:
        model.eval()
        WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
        with torch.no_grad():
            for i in range(0, WS_N, 4096):
                xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
                phis.append(model.blocks[0].feat(model.embed_x(xb)).mean(1).cpu().numpy())
        phi_pooled = np.concatenate(phis)
        try:
            lr_fit = LogisticRegression(max_iter=1000, C=1.0)
            lr_fit.fit(phi_pooled, y_train[:WS_N])
            coef = lr_fit.coef_.astype(np.float32)          # (n_coef, r)
            intr = lr_fit.intercept_.astype(np.float32)     # (n_coef,)
            r = coef.shape[1]
            if coef.shape[0] == 1 and C == 2:               # binary -> symmetric 2-logit
                coef = np.vstack([-coef[0] / 2, coef[0] / 2])
                intr = np.array([-intr[0] / 2, intr[0] / 2], dtype=np.float32)
            if coef.shape[0] == C:
                with torch.no_grad():
                    model.blocks[0].W.weight.data[:C] = torch.from_numpy(coef).to(device)
                    model.blocks[0].W.bias.data[:C].zero_()
                    model.head.weight.data.zero_()
                    for c in range(C):
                        model.head.weight.data[c, c] = 1.0
                    model.head.bias.data = torch.from_numpy(intr).to(device)
        except Exception:
            pass  # warm-start is a nicety; fall back to standard init

    model = _train_loop(model, X_train, y_train, epochs, batch_size, lr, weight_decay,
                        device, seed, adamw=False, cosine=False)
    proba = _predict_proba(model, X_test, device, C)
    return proba, {'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'n_layers': n_layers}
