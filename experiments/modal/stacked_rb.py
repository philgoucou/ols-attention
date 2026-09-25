"""
Standalone multi-layer ("stacked") Regression Block.

Design (faithful composition of the paper's RB block, L times):
  input embedding:  h0 = Embed(x) + pos          (B, S, d)
  block l (1..L):
      phi_l = PCA_l( polycross(h_{l-1}) )         (B, S, r)
      z_l   = LayerNorm( h_{l-1} + W_l(phi_l) )   (B, S, d)   [regression mixer]
      h_l   = LayerNorm( z_l + FFN_l(z_l) )       (B, S, d)
  output:           yhat = Head( mean_S(h_L) )

Pretraining, exactly mirroring the single-layer RB:
  * PCA_l is fit (frozen) on polycross features of block l's INPUT, computed by
    forward-passing a subsample through blocks 1..l-1. Sequential, no grad.
  * ONLY block 1's mixer W_1 is warm-started with a Ridge fit of pooled block-1
    features onto y (the paper's mechanism). Head is set to read channel 0.
    Deeper blocks use standard init and are learned end-to-end.

For n_layers == 1 this is identical in structure to train_rb_variant(...,'RB').
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# Defaults identical to the paper's RB (POLY_* in the main script)
POLY_PCA_COMP = 200
POLY_D_MODEL  = 64
POLY_FFN_DIM  = 128
POLY_EPOCHS   = 200
POLY_BS       = 256
POLY_LR       = 1e-3
POLY_DROPOUT  = 0.1
POLY_WD       = 1e-4


def _polycross(x):
    """(B,S,Dm) -> per-token [x, x^2-1] concatenated with all cross-token products."""
    B, S, Dm = x.shape
    per = torch.cat([x, x ** 2 - 1], dim=-1)
    pairs = torch.cat([x[:, i, :] * x[:, j, :]
                       for i in range(S) for j in range(i, S)], dim=-1)
    return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)


class _FrozenPCA(nn.Module):
    def __init__(self, mean, components):
        super().__init__()
        self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer('comp', torch.from_numpy(components.astype(np.float32)))
        self.out_dim = components.shape[0]

    def forward(self, x):
        return (x - self.mean) @ self.comp.T


class _Identity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out_dim = dim

    def forward(self, x):
        return x


class _RBBlock(nn.Module):
    """One regression-mixer block: h -> LN(h + W(PCA(polycross(h)))) -> +FFN -> LN."""
    def __init__(self, red_layer, d, ffn_dim, drop):
        super().__init__()
        self.red = red_layer
        self.W = nn.Linear(red_layer.out_dim, d)
        self.n1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(ffn_dim, d), nn.Dropout(drop))
        self.n2 = nn.LayerNorm(d)

    def feat(self, h):
        return self.red(_polycross(h))

    def forward(self, h):
        z = self.n1(h + self.W(self.feat(h)))
        return self.n2(z + self.ffn(z))


class StackedRB(nn.Module):
    def __init__(self, nf, red_layers, d, ffn_dim, drop):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, nf, d) * 0.02)
        self.blocks = nn.ModuleList([_RBBlock(rl, d, ffn_dim, drop) for rl in red_layers])
        self.head = nn.Linear(d, 1)

    def embed_x(self, x):
        return self.embed(x.unsqueeze(-1)) + self.pos

    def forward_upto(self, x, upto):
        """Return the input representation to block index `upto` (0-based).
        upto=0 -> embedding; upto=k -> output of block k-1."""
        h = self.embed_x(x)
        for l in range(upto):
            h = self.blocks[l](h)
        return h

    def forward(self, x):
        h = self.embed_x(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.mean(1))


def train_stacked_rb(X_train, y_train, X_test,
                     n_layers=1, use_pca=True,
                     n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL,
                     ffn_dim=POLY_FFN_DIM, epochs=POLY_EPOCHS,
                     batch_size=POLY_BS, lr=POLY_LR, dropout=POLY_DROPOUT,
                     weight_decay=POLY_WD, seed=42, device=None,
                     pca_fit_cap=5000, verbose=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    nf = X_train.shape[1]; D = d_model

    # ---- PCA-fit subsample (same rule as single-layer RB) ----
    if len(X_train) > pca_fit_cap:
        sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
        X_pca = X_train[sub]
    else:
        X_pca = X_train
    X_pca_t = torch.from_numpy(X_pca.astype(np.float32)).to(device)

    # ---- Sequentially build blocks, fitting each block's PCA on its input ----
    torch.manual_seed(seed)
    # temp model grown block-by-block; we build red_layers first via a scaffold
    scaffold = StackedRB(nf, [_Identity(1) for _ in range(0)], D, ffn_dim, dropout).to(device)
    # feature dim is constant across layers (input always (B,S,D))
    with torch.no_grad():
        probe = _polycross(scaffold.embed_x(X_pca_t[:8]))
        feat_dim = probe.shape[-1]

    red_layers = []
    # We must grow the model so forward_upto works while fitting deeper PCAs.
    model = StackedRB(nf, [], D, ffn_dim, dropout).to(device)
    # rebuild embed/pos deterministically to match a fresh init
    torch.manual_seed(seed)
    model = StackedRB(nf, [], D, ffn_dim, dropout).to(device)

    for l in range(n_layers):
        # representation feeding block l: forward through already-built blocks 0..l-1
        model.eval()
        with torch.no_grad():
            h_in = model.forward_upto(X_pca_t, upto=l)          # (B,S,D)
            phi = _polycross(h_in)                               # (B,S,feat_dim)
            phi_flat = phi.reshape(-1, phi.shape[-1]).cpu().numpy()
        if use_pca:
            n_comp = min(n_components, phi_flat.shape[1], phi_flat.shape[0])
            p = PCA(n_components=n_comp, random_state=seed); p.fit(phi_flat)
            rl = _FrozenPCA(p.mean_, p.components_)
        else:
            rl = _Identity(feat_dim)
        # append a new block using this reduction, with deterministic init
        torch.manual_seed(seed + 100 + l)
        new_block = _RBBlock(rl, D, ffn_dim, dropout).to(device)
        model.blocks.append(new_block)

    # ---- Warm-start block 1 with Ridge (paper's mechanism); head reads channel 0 ----
    model.eval()
    WS_N = min(len(X_train), 20000)
    X_ws = X_train[:WS_N]
    phis = []
    with torch.no_grad():
        for i in range(0, WS_N, 4096):
            xb = torch.from_numpy(X_ws[i:i + 4096].astype(np.float32)).to(device)
            phis.append(model.blocks[0].feat(model.embed_x(xb)).mean(1).cpu().numpy())
    phi_pooled = np.concatenate(phis)
    reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
    with torch.no_grad():
        model.blocks[0].W.weight.data[0] = torch.from_numpy(
            reg.coef_.astype(np.float32)).to(device)
        model.blocks[0].W.bias.data.zero_()
        model.head.weight.data.zero_()
        model.head.weight.data[0, 0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

    # ---- Train end-to-end (identical optimizer / early-stop to single-layer RB) ----
    N = X_train.shape[0]; n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    tr_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    best, best_state, pat = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xt_val), yt_val).item()
        if vl < best:
            best, best_state, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10:
                break
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i + 4096].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return np.concatenate(preds), {'n_params': n_params, 'n_layers': n_layers}
