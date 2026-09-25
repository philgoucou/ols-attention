"""
Mechanism test for the Friedman control result.

The control showed RB loses ground to FT-T when the number of RELEVANT features
grows (gap -0.007 at P=10 -> -0.192 at P=50 with the share held at 50%), and loses
LESS when the extra features are pure noise. Hypothesis for why: the Regression
Block compresses its poly-cross features to a FIXED 200 PCA components regardless
of P. The poly-cross width grows as O(P^2), so a fixed budget spans a shrinking
fraction of the interaction space — a capacity bottleneck, not a property of
regression-as-attention.

Test: rerun RB on the 50%-relevant arm at P=20 and P=50 with n_components in
{400, 800} (200 already measured). If the gap closes materially, the limitation is
a tunable design choice and the paper can say so precisely.

Deploy: modal deploy rebuttal_friedman2.py   (app: neurips-31482-friedman2)
"""
from __future__ import annotations
import modal

# Reuse the exact data + model code from the control run. Modal mounts only the
# entrypoint file by default, so the sibling module must be added to the image
# explicitly or the container fails at import.
from rebuttal_friedman import (build_split, train_rb, rb_mem_settings,  # noqa: F401
                               image as _base_image, POLY_D_MODEL, POLY_FFN_DIM,
                               POLY_EPOCHS, POLY_LR, POLY_DROPOUT, POLY_WD)

image = _base_image.add_local_python_source("rebuttal_friedman")
app = modal.App("neurips-31482-friedman2")


def train_rb_ncomp(X_train, y_train, X_test, n_components, pca_fit_cap, batch_size, seed=42):
    """Identical to rebuttal_friedman.train_rb but with n_components exposed."""
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    nf = X_train.shape[1]; D = POLY_D_MODEL

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

    class Net(nn.Module):
        def __init__(self, red):
            super().__init__()
            self.embed = nn.Linear(1, D); self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.red = red; self.W = nn.Linear(red.out_dim, D); self.n1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, POLY_FFN_DIM), nn.ReLU(), nn.Dropout(POLY_DROPOUT),
                                     nn.Linear(POLY_FFN_DIM, D), nn.Dropout(POLY_DROPOUT))
            self.n2 = nn.LayerNorm(D); self.head = nn.Linear(D, 1)
        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def feat(self, e): return self.red(polycross(e))
        def forward(self, x):
            e = self.embed_x(x)
            z = self.n1(e + self.W(self.feat(e)))
            return self.head(self.n2(z + self.ffn(z)).mean(1))

    X_pca = X_train if len(X_train) <= pca_fit_cap else X_train[
        np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)]
    torch.manual_seed(seed)
    tmp_embed = nn.Linear(1, D); tmp_pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
    Xt = torch.from_numpy(X_pca.astype(np.float32))
    with torch.no_grad():
        phi = polycross(tmp_embed(Xt.unsqueeze(-1)) + tmp_pos)
        feat_dim = phi.shape[-1]; phi = phi.reshape(-1, feat_dim).numpy()
    n_comp = min(n_components, phi.shape[1], phi.shape[0])
    p = PCA(n_components=n_comp, random_state=seed); p.fit(phi)
    evr = float(p.explained_variance_ratio_.sum())
    torch.manual_seed(seed)
    model = Net(FrozenPCA(p.mean_, p.components_)).to(device)

    model.eval(); WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
    warm_chunk = max(8, min(batch_size, 256))
    with torch.no_grad():
        for i in range(0, WS_N, warm_chunk):
            xb = torch.from_numpy(X_ws[i:i+warm_chunk].astype(np.float32)).to(device)
            phis.append(model.feat(model.embed_x(xb)).mean(1).cpu().numpy())
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    phi_pooled = np.concatenate(phis)
    reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
    with torch.no_grad():
        model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
        model.W.bias.data.zero_()
        model.head.weight.data.zero_(); model.head.weight.data[0, 0] = 1.0
        model.head.bias.data.fill_(float(reg.intercept_))

    N = X_train.shape[0]; n_val = max(1, int(N * 0.15)); idx = np.random.permutation(N)
    Xt2 = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt2 = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt2, yt2), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=POLY_LR, weight_decay=POLY_WD); crit = nn.MSELoss()
    best, bs, pat = float('inf'), None, 0
    for ep in range(1, POLY_EPOCHS+1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best: best, bs, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    if bs: model.load_state_dict({k: v.to(device) for k, v in bs.items()})
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            preds.append(model(torch.from_numpy(X_test[i:i+batch_size].astype(np.float32)).to(device)).cpu().numpy().ravel())
    n_par = sum(q.numel() for q in model.parameters() if q.requires_grad)
    return np.concatenate(preds), {'n_params': n_par, 'evr': evr, 'n_comp': n_comp}


@app.function(image=image, gpu="A10G", timeout=5400, memory=32768,
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_rb_ncomp(arm: str, P: int, ncomp: int, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        Xtr, ytr, Xte, yte, seed = build_split(arm, P, seed_idx)
        ms = rb_mem_settings(P)
        pred, info = train_rb_ncomp(Xtr, ytr, Xte, n_components=ncomp,
                                    pca_fit_cap=ms['pca_fit_cap'],
                                    batch_size=ms['batch_size'], seed=seed)
        return {'arm': arm, 'P': P, 'ncomp': ncomp, 'n_comp_used': info['n_comp'],
                'evr': info['evr'], 'params': info['n_params'], 'seed_idx': seed_idx,
                'r2': float(r2_score(yte, pred)), 'wall_s': _t.time()-t0, 'error': None}
    except Exception as e:
        return {'arm': arm, 'P': P, 'ncomp': ncomp, 'n_comp_used': None, 'evr': None,
                'params': None, 'seed_idx': seed_idx, 'r2': float('nan'),
                'wall_s': _t.time()-t0,
                'error': f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
