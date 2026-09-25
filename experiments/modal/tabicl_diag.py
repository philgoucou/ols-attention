"""Diagnostic: which device spec (if any) makes TabICLRegressor work?"""
import modal
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==1.26.4","scikit-learn==1.5.2","torch==2.4.1","tabicl>=0.1.0"))
app = modal.App("neurips-31482-tabicl-diag")

@app.function(image=image, gpu="T4", timeout=1800)
def diag():
    import numpy as np, torch, inspect
    from tabicl import TabICLRegressor
    from sklearn.metrics import r2_score
    rng=np.random.RandomState(0)
    Xtr=rng.uniform(size=(800,8)).astype('float32'); ytr=(Xtr[:,0]*2+Xtr[:,1]).astype('float32')
    Xte=rng.uniform(size=(200,8)).astype('float32'); yte=(Xte[:,0]*2+Xte[:,1]).astype('float32')
    out={'ctor_params': list(inspect.signature(TabICLRegressor.__init__).parameters)}
    variants={
      "str 'cuda:0'": lambda: TabICLRegressor(device='cuda:0'),
      "str 'cuda'":   lambda: TabICLRegressor(device='cuda'),
      "torch.device('cuda',0)": lambda: TabICLRegressor(device=torch.device('cuda',0)),
      "default (no device)":    lambda: TabICLRegressor(),
      "cpu":                    lambda: TabICLRegressor(device='cpu'),
    }
    for name,mk in variants.items():
        try:
            m=mk(); m.fit(Xtr,ytr); p=m.predict(Xte)
            out[name]=f"OK r2={r2_score(yte,p):.3f}"
        except Exception as e:
            out[name]=f"{type(e).__name__}: {str(e)[:110]}"
    return out

@app.local_entrypoint()
def main():
    r=diag.remote()
    print("=== TabICLRegressor DEVICE DIAGNOSTIC ===")
    for k,v in r.items(): print(f"  {k}: {v}")
