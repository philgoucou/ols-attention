"""
Full verification pass over all rebuttal experiment CSVs.

Checks:
  A. File integrity — presence, row counts, seed completeness, error columns.
  B. Internal sanity — NaNs only where errors, metric ranges, param invariants.
  C. Cross-experiment consistency — independent runs that must agree:
       C1. depth FTT-nb3  ==  ablation A0        (identical code + seeds -> tight tol)
       C2. depth RB-L1    ~=  ablation RB        (same arch, different init stream)
       C3. warmstart rb_warm ~= ablation RB      (same arch, different init stream)
       C4. warmstart softmax ~= ablation A4      (same arch, different init stream)
       C5. lcurve N=full FTT == uncapped FTT     (identical code + seeds -> tight tol)
       C6. lcurve N=full RB  ~= uncapped RB      (different impl of same arch)
       C7. depth/ablation/warmstart RB param counts identical per dataset
  D. Paper Table 1 verification recomputed from raw per-seed rows.
  E. Headline claims used in the report/drafts recomputed from raw data.

Exit code 0 iff no FAIL. WARNs are tolerated but listed.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def R(n): return os.path.join(HERE, n)

DS_ORDER = ['California', 'Yacht', 'Energy', 'Concrete', 'Airfoil', 'Abalone', 'Kin8nm', 'Protein']
ABL = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'RB']
PAPER = {'California': {'A0': 0.771, 'RB': 0.769}, 'Yacht': {'A0': 0.445, 'RB': 0.960},
         'Energy': {'A0': 0.992, 'RB': 0.995}, 'Concrete': {'A0': 0.652, 'RB': 0.892},
         'Airfoil': {'A0': -75.451, 'RB': 0.927}, 'Abalone': {'A0': 0.314, 'RB': 0.331},
         'Kin8nm': {'A0': 0.909, 'RB': 0.917}, 'Protein': {'A0': 0.418, 'RB': 0.453}}

lines, n_pass, n_warn, n_fail = [], 0, 0, 0
def PASS(name, detail=""):
    global n_pass; n_pass += 1
    lines.append(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
def WARN(name, detail):
    global n_warn; n_warn += 1
    lines.append(f"WARN  {name}  [{detail}]")
def FAIL(name, detail):
    global n_fail; n_fail += 1
    lines.append(f"FAIL  {name}  [{detail}]")

# ---------------- A. file integrity ----------------
EXPECT = {
    'ablation_grid_results.csv': (280, ['dataset', 'config', 'seed_idx', 'r2', 'params', 'error']),
    'uncapped_results.csv':      (60,  ['dataset', 'N', 'model', 'seed_idx', 'r2', 'error']),
    'tabpfn_results.csv':        (40,  ['dataset', 'seed_idx', 'r2', 'error']),
    'depth_results.csv':         (240, ['dataset', 'model', 'depth', 'seed_idx', 'r2', 'params', 'error']),
    'warmstart_results.csv':     (120, ['dataset', 'config', 'seed_idx', 'r2', 'params', 'error']),
    'lcurve_results.csv':        (180, ['dataset', 'model', 'n_train_req', 'n_train_used', 'seed_idx', 'r2', 'error']),
    'classify_results.csv':      (150, ['dataset', 'model', 'n_classes', 'seed_idx', 'acc', 'auc', 'error']),
    'highdim_results.csv':       (140, ['dataset', 'model', 'P', 'seed_idx', 'r2', 'error']),
    'tabicl_results.csv':        (30,  ['dataset', 'model', 'n_classes', 'seed_idx', 'acc', 'auc', 'error']),
    'a6_results.csv':            (40,  ['dataset', 'config', 'seed_idx', 'r2', 'params', 'error']),
    'audit_faithful_results.csv':(320, ['dataset', 'variant', 'seed_idx', 'r2', 'params', 'error']),
    'audit_frall_results.csv':   (200, ['arm', 'P', 'model', 'seed_idx', 'r2', 'error']),
    'audit_sub_results.csv':     (160, ['dataset', 'variant', 'n_mix', 'n_ffn', 'seed_idx', 'r2', 'params', 'error']),
}
dfs = {}
for f, (nrows, cols) in EXPECT.items():
    if not os.path.exists(R(f)):
        FAIL(f"file {f}", "missing"); continue
    df = pd.read_csv(R(f)); dfs[f] = df
    if len(df) == nrows: PASS(f"rows {f}", f"{len(df)}")
    else: FAIL(f"rows {f}", f"{len(df)} != expected {nrows}")
    missing = [c for c in cols if c not in df.columns]
    if missing: FAIL(f"cols {f}", f"missing {missing}")
    else: PASS(f"cols {f}")

def cellkey(f):
    return {'ablation_grid_results.csv': ['dataset', 'config'],
            'uncapped_results.csv': ['dataset', 'model'],
            'tabpfn_results.csv': ['dataset'],
            'depth_results.csv': ['dataset', 'model'],
            'warmstart_results.csv': ['dataset', 'config'],
            'lcurve_results.csv': ['dataset', 'model', 'n_train_req'],
            'classify_results.csv': ['dataset', 'model'],
            'highdim_results.csv': ['dataset', 'model'],
            'tabicl_results.csv': ['dataset'],
            'a6_results.csv': ['dataset', 'config'],
            'audit_faithful_results.csv': ['dataset', 'variant'],
            'audit_frall_results.csv': ['arm', 'P', 'model'],
            'audit_sub_results.csv': ['dataset', 'variant']}[f]

for f, df in dfs.items():
    bad = []
    for key, g in df.groupby(cellkey(f)):
        seeds = sorted(g['seed_idx'].tolist())
        if seeds != [0, 1, 2, 3, 4]:
            bad.append((key, seeds))
    if bad: FAIL(f"seeds {f}", f"{len(bad)} cells incomplete, e.g. {bad[0]}")
    else: PASS(f"seeds {f}", "all cells have seeds 0-4")

# error accounting: every error must be in the allowed set
ALLOWED_ERR = {('highdim_results.csv', 'Superconductivity', 'RB')}  # pre-A100-splice
for f, df in dfs.items():
    errs = df[df['error'].notna()]
    if not len(errs): PASS(f"errors {f}", "0"); continue
    keys = set()
    for _, r in errs.iterrows():
        keys.add((f, r.get('dataset'), r.get('model', r.get('config'))))
    unexpected = keys - ALLOWED_ERR
    if unexpected: FAIL(f"errors {f}", f"{len(errs)} errors incl. unexpected {sorted(unexpected)[:3]}")
    else: WARN(f"errors {f}", f"{len(errs)} errors, all in known-limitation cells {sorted(keys)}")

# NaN r2 only with error
for f, df in dfs.items():
    metric = 'acc' if f in ('classify_results.csv', 'tabicl_results.csv') else 'r2'
    bad = df[df[metric].isna() & df['error'].isna()]
    if len(bad): FAIL(f"nan-{metric} {f}", f"{len(bad)} rows NaN without error")
    else: PASS(f"nan-{metric} {f}")

# ---------------- B. param invariants ----------------
da = dfs['ablation_grid_results.csv']; dd = dfs['depth_results.csv']; dw = dfs['warmstart_results.csv']
pa = da.groupby(['dataset', 'config'])['params'].max().unstack()
pdd = dd.groupby(['dataset', 'model'])['params'].max().unstack()
pw = dw.groupby(['dataset', 'config'])['params'].max().unstack()
if pa.loc['California', 'RB'] == 30401 and pa.loc['California', 'A4'] == 34177 and pa.loc['California', 'A0'] == 101697:
    PASS("param anchors (California)", "RB=30401 A4=34177 A0=101697")
else:
    FAIL("param anchors (California)", f"RB={pa.loc['California','RB']} A4={pa.loc['California','A4']} A0={pa.loc['California','A0']}")
mism = [(ds, pa.loc[ds, 'RB'], pdd.loc[ds, 'RB-L1'], pw.loc[ds, 'rb_warm'])
        for ds in DS_ORDER if not (pa.loc[ds, 'RB'] == pdd.loc[ds, 'RB-L1'] == pw.loc[ds, 'rb_warm'])]
if mism: FAIL("RB param identity across experiments", f"{mism}")
else: PASS("RB param identity across experiments", "ablation == depth-L1 == warmstart, all 8 datasets")
mono = all(pdd.loc[ds, 'RB-L1'] < pdd.loc[ds, 'RB-L2'] < pdd.loc[ds, 'RB-L3'] and
           pdd.loc[ds, 'FTT-nb1'] < pdd.loc[ds, 'FTT-nb2'] < pdd.loc[ds, 'FTT-nb3'] for ds in DS_ORDER)
PASS("depth params monotone") if mono else FAIL("depth params monotone", "violation")

# ---------------- C. cross-experiment consistency ----------------
ma = da.groupby(['dataset', 'config'])['r2'].mean().unstack()
md = dd.groupby(['dataset', 'model'])['r2'].mean().unstack()
mw = dw.groupby(['dataset', 'config'])['r2'].mean().unstack()
du = dfs['uncapped_results.csv']; mu = du.groupby(['dataset', 'model'])['r2'].mean().unstack()
dl = dfs['lcurve_results.csv']
ml = dl[dl.n_train_req == -1].groupby(['dataset', 'model'])['r2'].mean().unstack()

def xcheck(name, sA, sB, tol, rel_for_big=True):
    worst = 0.0; worst_ds = None
    for ds in sA.index.intersection(sB.index):
        a, b = sA.loc[ds], sB.loc[ds]
        if pd.isna(a) or pd.isna(b): continue
        d = abs(a - b)
        if rel_for_big and max(abs(a), abs(b)) > 2:
            d = d / max(abs(a), abs(b))
        if d > worst: worst, worst_ds = d, ds
    if worst <= tol: PASS(name, f"max|Δ|={worst:.4f} @{worst_ds}")
    elif worst <= 2 * tol: WARN(name, f"max|Δ|={worst:.4f} @{worst_ds} (tol {tol})")
    else: FAIL(name, f"max|Δ|={worst:.4f} @{worst_ds} (tol {tol})")

xcheck("C1 depth FTT-nb3 == ablation A0 (identical code+seeds)", md['FTT-nb3'], ma['A0'], 0.02)
xcheck("C2 depth RB-L1 ~= ablation RB", md['RB-L1'], ma['RB'], 0.05)
xcheck("C3 warmstart rb_warm ~= ablation RB", mw['rb_warm'], ma['RB'], 0.05)
xcheck("C4 warmstart softmax ~= ablation A4 (diff init stream)", mw['softmax'], ma['A4'], 0.06)
xcheck("C5 lcurve full FTT == uncapped FTT (identical code+seeds)", ml['FTT'], mu['FTT'], 0.02)
xcheck("C6 lcurve full RB ~= uncapped RB (diff impl same arch)", ml['RB'], mu['RB'], 0.04)

# ---------------- D. Table 1 verification ----------------
fails_t1 = []
for ds, tgt in PAPER.items():
    a0, rb = ma.loc[ds, 'A0'], ma.loc[ds, 'RB']
    okA = (a0 < 0) if ds == 'Airfoil' else abs(a0 - tgt['A0']) <= 0.02
    okR = abs(rb - tgt['RB']) <= 0.02
    if not okA: fails_t1.append((ds, 'A0', a0, tgt['A0']))
    if not okR: fails_t1.append((ds, 'RB', rb, tgt['RB']))
if fails_t1: FAIL("D Table-1 16-cell verification", f"{fails_t1}")
else: PASS("D Table-1 16-cell verification", "16/16 within ±0.02 (Airfoil A0 by sign)")

# ---------------- E. headline claims ----------------
ranks = ma[ABL].rank(axis=1, ascending=False)
rb_rank = ranks['RB'].mean()
if abs(rb_rank - 1.50) < 0.26: PASS("E RB avg rank ~1.50", f"{rb_rank:.2f}")
else: WARN("E RB avg rank ~1.50", f"got {rb_rank:.2f}")
wins_rb = int((ma['RB'] > ma['A0']).sum())
PASS("E RB wins vs A0", f"{wins_rb}/8") if wins_rb >= 6 else WARN("E RB wins vs A0", f"{wins_rb}/8")

g_read = (mw['rb_nowarm'] - mw['softmax']).mean()
g_warm = (mw['rb_warm'] - mw['rb_nowarm']).mean()
nbeat = int((mw['rb_nowarm'] > mw['softmax']).sum())
if nbeat == 8 and g_read > 2 * max(g_warm, 0):
    PASS("E warm-start decomposition", f"readout {g_read:+.3f} vs warmstart {g_warm:+.3f}; nowarm>softmax {nbeat}/8")
else:
    WARN("E warm-start decomposition", f"readout {g_read:+.3f}, warmstart {g_warm:+.3f}, nowarm>softmax {nbeat}/8")

l5 = dl[dl.n_train_req == 500].groupby(['dataset', 'model'])['r2'].mean().unstack()
small_edge = (l5['RB'] - l5['FTT'])
if (small_edge > 0).all():
    PASS("E small-N edge (RB>FTT at N=500, all 3)", ", ".join(f"{d}:{small_edge[d]:+.3f}" for d in small_edge.index))
else:
    WARN("E small-N edge", str(small_edge.round(3).to_dict()))

dc = dfs['classify_results.csv']
mc = dc.groupby(['dataset', 'model'])['acc'].mean().unstack()
mrow = mc.mean()
if mrow['RB'] > mrow['FTT']:
    PASS("E classification RB>FTT mean acc", f"RB {mrow['RB']:.3f} vs FTT {mrow['FTT']:.3f}")
else:
    WARN("E classification RB>FTT mean acc", f"RB {mrow['RB']:.3f} vs FTT {mrow['FTT']:.3f}")
rng_ok = dc['acc'].dropna().between(0, 1).all() and dc['auc'].dropna().between(0.4, 1.0).all()
PASS("E classification metric ranges") if rng_ok else FAIL("E classification metric ranges", "out of range values")

dh = dfs['highdim_results.csv']
mh = dh.groupby(['dataset', 'model'])['r2'].mean().unstack()
if {'FTT', 'RB'} <= set(mh.columns):
    gaps = {}
    for ds in ['Friedman-P20', 'Friedman-P50']:
        if ds in mh.index and pd.notna(mh.loc[ds, 'RB']):
            gaps[ds] = mh.loc[ds, 'RB'] - mh.loc[ds, 'FTT']
    if len(gaps) == 2 and gaps['Friedman-P50'] < gaps['Friedman-P20'] < 0:
        PASS("E Friedman degradation direction", f"P20 {gaps['Friedman-P20']:+.3f} -> P50 {gaps['Friedman-P50']:+.3f}")
    else:
        WARN("E Friedman degradation direction", str({k: round(v, 3) for k, v in gaps.items()}))
sup = dh[(dh.dataset == 'Superconductivity') & (dh.model == 'RB')]
ok_sup = sup['error'].isna().sum()
lines.append(f"INFO  Superconductivity RB status: {ok_sup}/5 seeds clean "
             f"({'A100 splice landed' if ok_sup == 5 else 'OOM limitation stands' if ok_sup == 0 else 'partial'})")

# lcurve monotonicity: full >= N500 per dataset/model
lm_bad = []
for (ds, m), g in dl.groupby(['dataset', 'model']):
    gg = g.groupby('n_train_req')['r2'].mean()
    if -1 in gg.index and 500 in gg.index and not (gg.loc[-1] >= gg.loc[500] - 0.01):
        lm_bad.append((ds, m))
PASS("E lcurve monotonic (full>=N500)") if not lm_bad else FAIL("E lcurve monotonic", str(lm_bad))

# tabicl sanity
if 'tabicl_results.csv' in dfs:
    dti = dfs['tabicl_results.csv']
    mti = dti.groupby('dataset')['acc'].mean()
    if mti.between(0.5, 1.0).all():
        PASS("E tabicl plausible range", f"mean acc {mti.mean():.3f}")
    else:
        WARN("E tabicl range", str(mti.round(3).to_dict()))

# tabpfn range sanity
dt = dfs['tabpfn_results.csv']
mt = dt.groupby('dataset')['r2'].mean()
PASS("E tabpfn plausible range", f"min {mt.min():.3f} max {mt.max():.3f}") \
    if mt.between(0.2, 1.0).all() else WARN("E tabpfn range", str(mt.round(3).to_dict()))

# ---------------- report ----------------
hdr = f"VERIFICATION REPORT — {n_pass} pass, {n_warn} warn, {n_fail} fail"
out = hdr + "\n" + "=" * len(hdr) + "\n" + "\n".join(lines) + "\n"
with open(R("verification_report.txt"), "w") as fh:
    fh.write(out)
print(out)
sys.exit(1 if n_fail else 0)
