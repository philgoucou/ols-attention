"""
y-STANDARDIZED rerun of the ablation grid (A0-A5, RB) AND the warm-start decomposition
(softmax / rb_nowarm / rb_warm): the corrected-protocol runs behind App. E of the revised
paper. Formerly rebuttal_ystd_grids.py.

Why: the target-scaling artifact does not hit all configurations equally. The Ridge
warm-start fires only when the mixer is a regression (A2/A3/RB); the FT-T configs
(A0/A1/A5) and the softmax mixer (A4) have no warm-start. Since the warm-start is
what supplies the target's location and scale, the artifact's severity is aligned
with the very property the ablation isolates. Both grids were originally run with
raw targets, so both are reruned here under uniform target standardization.

Grids:
  ablation  : 8 datasets x {A0,A1,A2,A3,A4,A5,RB} x 5 seeds = 280
  warmstart : 8 datasets x {softmax, rb_nowarm, rb_warm} x 5 seeds = 120

Deploy: modal deploy rebuttal_ablation_warmstart.py   (app: neurips-31482-ystd-grids)
Spawn run_abl_ystd(dataset, config, seed_idx) with tag 'ablystd' and run_ws_ystd(dataset,
config, seed_idx) with tag 'wsystd', harvest with the collector template of collectors/
-> results/audit/audit_ablystd_results.csv and audit_wsystd_results.csv.
"""
from __future__ import annotations
import pickle
import modal

N_REPEATS = 5; TEST_FRAC = 0.2; MAX_N = 5000; SEED = 42
FTT_D_MODEL = 64; FTT_N_HEADS = 4; FTT_FFN_DIM = 128; FTT_N_BLOCKS = 3
FTT_DROPOUT = 0.1; FTT_EPOCHS = 200; FTT_BS = 256; FTT_LR = 1e-3; FTT_WD = 1e-4
POLY_PCA_COMP = 200; POLY_D_MODEL = 64; POLY_FFN_DIM = 128
POLY_EPOCHS = 200; POLY_BS = 256; POLY_LR = 1e-3; POLY_DROPOUT = 0.1; POLY_WD = 1e-4
DS_ORDER = ['California','Yacht','Energy','Concrete','Airfoil','Abalone','Kin8nm','Protein']
ABL = ['A0','A1','A2','A3','A4','A5','RB']
WS  = ['softmax','rb_nowarm','rb_warm']

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4","pandas==2.2.3","scikit-learn==1.5.2","torch==2.4.1"))
cache_vol = modal.Volume.from_name("neurips-31482-cache", create_if_missing=True)
CACHE_DIR = "/cache"
app = modal.App("neurips-31482-ystd-grids")


def _load_cached():
    import os
    with open(os.path.join(CACHE_DIR, "datasets_v1.pkl"), "rb") as f:
        return pickle.load(f)


def prep_ystd(X_full, y_full, seed_idx, need_poly=False):
    """Capped split with X AND y standardized on train."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    N_full, P = X_full.shape
    if N_full > MAX_N:
        idx = np.random.RandomState(SEED).choice(N_full, MAX_N, replace=False)
        X_full, y_full = X_full[idx], y_full[idx]
    seed = SEED + seed_idx * 1000
    Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=TEST_FRAC, random_state=seed)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    sy = StandardScaler()
    ytr = sy.fit_transform(np.asarray(ytr, dtype='float64').reshape(-1,1)).ravel()
    yte = sy.transform(np.asarray(yte, dtype='float64').reshape(-1,1)).ravel()
    Xtr_pf = Xte_pf = None
    if need_poly:
        pf = PolynomialFeatures(degree=2, include_bias=False)
        Xtr_pf = pf.fit_transform(Xtr); Xte_pf = pf.transform(Xte)
        s2 = StandardScaler(); Xtr_pf = s2.fit_transform(Xtr_pf); Xte_pf = s2.transform(Xte_pf)
        Xtr_pf = Xtr_pf.astype('float32'); Xte_pf = Xte_pf.astype('float32')
    return (Xtr.astype('float32'), ytr.astype('float32'), Xte.astype('float32'),
            yte.astype('float32'), Xtr_pf, Xte_pf, P, seed)


def ftt_param_count(nf, d, nh, ffn, nblk):
    import torch.nn as nn
    m = nn.ModuleDict({'emb': nn.ModuleList([nn.Linear(1,d) for _ in range(nf)]),
        'enc': nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d, nhead=nh,
            dim_feedforward=ffn, dropout=0.1, batch_first=True, activation='gelu'), num_layers=nblk),
        'norm': nn.LayerNorm(d), 'head': nn.Linear(d,1)})
    return sum(p.numel() for p in m.parameters()) + d


def rb_param_count(nf, d=POLY_D_MODEL, ffn=POLY_FFN_DIM, ncomp=POLY_PCA_COMP):
    import torch.nn as nn
    m = nn.ModuleDict({'e': nn.Linear(1,d), 'W': nn.Linear(ncomp,d), 'n1': nn.LayerNorm(d),
        'f': nn.Sequential(nn.Linear(d,ffn), nn.Linear(ffn,d)), 'n2': nn.LayerNorm(d), 'h': nn.Linear(d,1)})
    return sum(p.numel() for p in m.parameters()) + nf*d


def pick_a5_dmodel(nf, target):
    best_d, best_gap = None, float('inf')
    for d in [16,20,24,28,32,36,40,48]:
        gap = abs(ftt_param_count(nf, d, 4, 2*d, FTT_N_BLOCKS) - target)
        if gap < best_gap: best_d, best_gap = d, gap
    return best_d


def train_ft_transformer(X_train, y_train, X_test,
                         d_model=64, n_heads=4, ffn_dim=128,
                         n_blocks=3, dropout=0.1,
                         epochs=50, batch_size=256, lr=1e-3,
                         weight_decay=1e-4, seed=42, verbose=False):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model

    class FTTransformer(nn.Module):
        def __init__(self, nf, d, nh, ffn, nblk, drop):
            super().__init__()
            self.feat_embeds = nn.ModuleList([nn.Linear(1, d) for _ in range(nf)])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nh, dim_feedforward=ffn, dropout=drop,
                batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nblk)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            B = x.size(0)
            tokens = torch.stack(
                [self.feat_embeds[i](x[:, i:i+1]) for i in range(x.size(1))], dim=1)
            tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
            z = self.transformer(tokens)
            return self.head(self.norm(z[:, 0, :]))

    model = FTTransformer(nf, D, n_heads, ffn_dim, n_blocks, dropout).to(device)

    N = X_train.shape[0]
    n_val = max(1, int(N * 0.15))
    idx = np.random.permutation(N)
    Xt_tr = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32))
    yt_tr = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xt_val = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device)
    yt_val = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)

    tr_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size, shuffle=True)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(1, epochs // 10)
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return ep / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val_loss = float('inf'); best_state = None
    patience_counter = 0; patience = 10
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xt_val), yt_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            xb = torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    import numpy as _np
    return _np.concatenate(preds), {'n_params': sum(p.numel() for p in model.parameters())}



def train_rb_variant(X_train, y_train, X_test,
                     use_poly=True, use_pca=True, softmax_mixer=False,
                     n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL,
                     ffn_dim=POLY_FFN_DIM, epochs=POLY_EPOCHS,
                     batch_size=POLY_BS, lr=POLY_LR, dropout=POLY_DROPOUT,
                     weight_decay=POLY_WD, n_heads=4, seed=42,
                     pca_fit_cap=5000):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = X_train.shape[1]; D = d_model

    def poly_cross(x):
        B, S, Dm = x.shape
        per = torch.cat([x, x**2 - 1], dim=-1)
        pairs = torch.cat(
            [x[:, i, :] * x[:, j, :] for i in range(S) for j in range(i, S)], dim=-1)
        return torch.cat([per, pairs.unsqueeze(1).expand(B, S, -1)], dim=-1)

    def features(x):
        return poly_cross(x) if use_poly else x

    class FrozenPCA(nn.Module):
        def __init__(self, mean, components):
            super().__init__()
            self.register_buffer('mean', torch.from_numpy(mean.astype(np.float32)))
            self.register_buffer('comp', torch.from_numpy(components.astype(np.float32)))
            self.out_dim = components.shape[0]
        def forward(self, x): return (x - self.mean) @ self.comp.T

    class Identity(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.out_dim = dim
        def forward(self, x): return x

    class RBVariant(nn.Module):
        def __init__(self, nf, red_layer, d, ffn_dim, drop):
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.pos = nn.Parameter(torch.randn(1, nf, d) * 0.02)
            if softmax_mixer:
                self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=drop)
            else:
                self.red = red_layer
                self.W = nn.Linear(red_layer.out_dim, d)
            self.n1 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, ffn_dim), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(ffn_dim, d), nn.Dropout(drop))
            self.n2 = nn.LayerNorm(d)
            self.head = nn.Linear(d, 1)

        def embed_x(self, x): return self.embed(x.unsqueeze(-1)) + self.pos
        def get_feat(self, emb): return self.red(features(emb))

        def forward(self, x):
            emb = self.embed_x(x)
            if softmax_mixer:
                a, _ = self.attn(emb, emb, emb)
                z = self.n1(emb + a)
            else:
                z = self.n1(emb + self.W(self.get_feat(emb)))
            return self.head(self.n2(z + self.ffn(z)).mean(1))

    red_layer = None
    if not softmax_mixer:
        torch.manual_seed(seed)
        tmp_embed = nn.Linear(1, D)
        tmp_pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
        if len(X_train) > pca_fit_cap:
            sub = np.random.RandomState(seed).choice(len(X_train), pca_fit_cap, replace=False)
            X_pca = X_train[sub]
        else:
            X_pca = X_train
        Xt = torch.from_numpy(X_pca.astype(np.float32))
        with torch.no_grad():
            emb = tmp_embed(Xt.unsqueeze(-1)) + tmp_pos
            phi_all = features(emb)
            feat_dim = phi_all.shape[-1]
            phi_all = phi_all.reshape(-1, feat_dim)
        if use_pca:
            n_comp = min(n_components, phi_all.shape[1], phi_all.shape[0])
            p = PCA(n_components=n_comp, random_state=seed); p.fit(phi_all.numpy())
            red_layer = FrozenPCA(p.mean_, p.components_)
        else:
            red_layer = Identity(feat_dim)

    torch.manual_seed(seed)
    model = RBVariant(nf, red_layer, D, ffn_dim, dropout).to(device)

    if not softmax_mixer:
        model.eval()
        WS_N = min(len(X_train), 20000)
        X_ws = X_train[:WS_N]
        phis = []
        with torch.no_grad():
            for i in range(0, WS_N, 4096):
                xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
                phis.append(model.get_feat(model.embed_x(xb)).mean(1).cpu().numpy())
        phi_pooled = np.concatenate(phis)
        reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
        with torch.no_grad():
            model.W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
            model.W.bias.data.zero_()
            model.head.weight.data.zero_()
            model.head.weight.data[0, 0] = 1.0
            model.head.bias.data.fill_(float(reg.intercept_))

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
            xb = torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    import numpy as _np
    return _np.concatenate(preds), {
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}



def train_rb_flex(X_train, y_train, X_test, mixer='regression', warmstart=True,
                  n_layers=1, use_pca=True, n_components=POLY_PCA_COMP, d_model=POLY_D_MODEL,
                  ffn_dim=POLY_FFN_DIM, n_heads=4, epochs=POLY_EPOCHS, batch_size=POLY_BS,
                  lr=POLY_LR, dropout=POLY_DROPOUT, weight_decay=POLY_WD, seed=42):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    nf = X_train.shape[1]; D = d_model; pca_fit_cap = 5000
    is_reg = (mixer == 'regression')

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

    class Block(nn.Module):
        def __init__(self, red):
            super().__init__()
            if is_reg:
                self.red = red; self.W = nn.Linear(red.out_dim, D)
            else:
                self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True, dropout=dropout)
            self.n1 = nn.LayerNorm(D)
            self.ffn = nn.Sequential(nn.Linear(D, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(ffn_dim, D), nn.Dropout(dropout))
            self.n2 = nn.LayerNorm(D)
        def feat(self, h): return self.red(polycross(h))
        def forward(self, h):
            if is_reg:
                z = self.n1(h + self.W(self.feat(h)))
            else:
                a, _ = self.attn(h, h, h); z = self.n1(h + a)
            return self.n2(z + self.ffn(z))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, D); self.pos = nn.Parameter(torch.randn(1, nf, D) * 0.02)
            self.blocks = nn.ModuleList(); self.head = nn.Linear(D, 1)
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
    model = Net().to(device)
    with torch.no_grad():
        feat_dim = polycross(model.embed_x(X_pca_t[:8])).shape[-1]
    for l in range(n_layers):
        red = None
        if is_reg:
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
        model.blocks.append(Block(red).to(device))

    if is_reg and warmstart:
        model.eval()
        WS_N = min(len(X_train), 20000); X_ws = X_train[:WS_N]; phis = []
        with torch.no_grad():
            for i in range(0, WS_N, 4096):
                xb = torch.from_numpy(X_ws[i:i+4096].astype(np.float32)).to(device)
                phis.append(model.blocks[0].feat(model.embed_x(xb)).mean(1).cpu().numpy())
        phi_pooled = np.concatenate(phis)
        reg = Ridge(alpha=1.0); reg.fit(phi_pooled, y_train[:WS_N])
        with torch.no_grad():
            model.blocks[0].W.weight.data[0] = torch.from_numpy(reg.coef_.astype(np.float32)).to(device)
            model.blocks[0].W.bias.data.zero_()
            model.head.weight.data.zero_(); model.head.weight.data[0, 0] = 1.0
            model.head.bias.data.fill_(float(reg.intercept_))

    N = X_train.shape[0]; n_val = max(1, int(N * 0.15)); idx = np.random.permutation(N)
    Xt = torch.from_numpy(X_train[idx[n_val:]].astype(np.float32)); yt = torch.from_numpy(y_train[idx[n_val:]].astype(np.float32)).unsqueeze(1)
    Xv = torch.from_numpy(X_train[idx[:n_val]].astype(np.float32)).to(device); yv = torch.from_numpy(y_train[idx[:n_val]].astype(np.float32)).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay); crit = nn.MSELoss()
    best, bs, pat = float('inf'), None, 0
    for ep in range(1, epochs+1):
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
        for i in range(0, len(X_test), 4096):
            preds.append(model(torch.from_numpy(X_test[i:i+4096].astype(np.float32)).to(device)).cpu().numpy().ravel())
    return np.concatenate(preds), {'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_abl_ystd(dataset_name: str, config: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, Xtr_pf, Xte_pf, P, seed = prep_ystd(X, y, seed_idx, need_poly=(config=='A1'))
        if config == 'A0':
            pred, info = train_ft_transformer(Xtr, ytr, Xte, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                ffn_dim=FTT_FFN_DIM, n_blocks=FTT_N_BLOCKS, dropout=FTT_DROPOUT,
                epochs=FTT_EPOCHS, batch_size=FTT_BS, lr=FTT_LR, weight_decay=FTT_WD, seed=seed)
        elif config == 'A1':
            pred, info = train_ft_transformer(Xtr_pf, ytr, Xte_pf, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                ffn_dim=FTT_FFN_DIM, n_blocks=FTT_N_BLOCKS, dropout=FTT_DROPOUT,
                epochs=FTT_EPOCHS, batch_size=FTT_BS, lr=FTT_LR, weight_decay=FTT_WD, seed=seed)
        elif config == 'A2':
            pred, info = train_rb_variant(Xtr, ytr, Xte, use_poly=False, seed=seed)
        elif config == 'A3':
            pred, info = train_rb_variant(Xtr, ytr, Xte, use_pca=False, seed=seed)
        elif config == 'A4':
            pred, info = train_rb_variant(Xtr, ytr, Xte, softmax_mixer=True, seed=seed)
        elif config == 'A5':
            a5d = pick_a5_dmodel(P, rb_param_count(P))
            pred, info = train_ft_transformer(Xtr, ytr, Xte, d_model=a5d, n_heads=4, ffn_dim=2*a5d,
                n_blocks=FTT_N_BLOCKS, dropout=FTT_DROPOUT, epochs=FTT_EPOCHS,
                batch_size=FTT_BS, lr=FTT_LR, weight_decay=FTT_WD, seed=seed)
        elif config == 'RB':
            pred, info = train_rb_variant(Xtr, ytr, Xte, seed=seed)
        else:
            raise ValueError(config)
        return {'kind':'abl_ystd','dataset':dataset_name,'config':config,'seed_idx':seed_idx,
                'r2':float(r2_score(yte,pred)),'params':int(info.get('n_params',0)),
                'wall_s':_t.time()-t0,'error':None}
    except Exception as e:
        return {'kind':'abl_ystd','dataset':dataset_name,'config':config,'seed_idx':seed_idx,
                'r2':float('nan'),'params':None,'wall_s':_t.time()-t0,
                'error':f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}


@app.function(image=image, gpu="T4", timeout=2700, volumes={CACHE_DIR: cache_vol},
              retries=modal.Retries(max_retries=1, backoff_coefficient=1.0))
def run_ws_ystd(dataset_name: str, config: str, seed_idx: int):
    import time as _t, traceback
    from sklearn.metrics import r2_score
    t0 = _t.time()
    try:
        X, y = _load_cached()[dataset_name]
        Xtr, ytr, Xte, yte, _, _, P, seed = prep_ystd(X, y, seed_idx)
        if config == 'softmax':
            pred, info = train_rb_flex(Xtr, ytr, Xte, mixer='softmax', seed=seed)
        elif config == 'rb_nowarm':
            pred, info = train_rb_flex(Xtr, ytr, Xte, mixer='regression', warmstart=False, seed=seed)
        elif config == 'rb_warm':
            pred, info = train_rb_flex(Xtr, ytr, Xte, mixer='regression', warmstart=True, seed=seed)
        else:
            raise ValueError(config)
        return {'kind':'ws_ystd','dataset':dataset_name,'config':config,'seed_idx':seed_idx,
                'r2':float(r2_score(yte,pred)),'params':int(info.get('n_params',0)),
                'wall_s':_t.time()-t0,'error':None}
    except Exception as e:
        return {'kind':'ws_ystd','dataset':dataset_name,'config':config,'seed_idx':seed_idx,
                'r2':float('nan'),'params':None,'wall_s':_t.time()-t0,
                'error':f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
