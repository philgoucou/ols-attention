"""Generate every LaTeX table body for the revision from the run CSVs, so no
number is retyped by hand. Output: tables_generated.tex (blocks separated by
'%%%% ===== name ====='), plus a compact console summary of the test statistics."""
import os
import pandas as pd, numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'results')   # the repo's results/ folder (audit/ and rebuttal/)
DS = ['California', 'Yacht', 'Energy', 'Concrete', 'Airfoil', 'Abalone', 'Kin8nm', 'Protein']
N = {'California': 5000, 'Yacht': 308, 'Energy': 768, 'Concrete': 1030, 'Airfoil': 1503,
     'Abalone': 4177, 'Kin8nm': 5000, 'Protein': 5000}
P = {'California': 8, 'Yacht': 6, 'Energy': 8, 'Concrete': 8, 'Airfoil': 5, 'Abalone': 7,
     'Kin8nm': 8, 'Protein': 9}


def load(p):
    # Path resolution only. The CSV names below are the working-directory names used
    # while the paper was revised: 'audit/x.csv' -> results/audit/x.csv,
    # 'results/x.csv' (first rebuttal wave) -> results/rebuttal/x.csv, and the bare
    # 'audit_agg_results.csv' -> results/audit/.
    sub, _, name = p.rpartition('/')
    d = pd.read_csv(f"{D}/{'rebuttal' if sub == 'results' else 'audit'}/{name}")
    return d[d.error.isna()] if 'error' in d else d


def m(d, by, val='r2'):
    return d.groupby(['dataset', by])[val].mean().unstack().reindex(DS)


def pt(a, b):
    a, b = a.align(b, join='inner')
    s, p = stats.ttest_rel(a, b)
    return (a - b).mean(), p


out = {}

# ---------- Table 1, corrected protocol ----------
# OLS / RF / Att.Reg are scale-equivariant (verified) -> published values kept.
ys = m(load('audit/audit_ystd_results.csv'), 'model')
mlp = load('audit/audit_mlpystd_results.csv').groupby('dataset')['r2'].mean().reindex(DS)
pub_ols = dict(zip(DS, [.597, .562, .913, .624, .497, .260, .430, .285]))
pub_rf = dict(zip(DS, [.777, .979, .996, .892, .910, .322, .664, .521]))
pub_ar = dict(zip(DS, [.738, .958, .995, .837, .757, .312, .851, .361]))
t1 = pd.DataFrame({'OLS': pd.Series(pub_ols), 'RF': pd.Series(pub_rf), 'MLP': mlp,
                   'FT-T': ys['FTT'], 'Att.Reg': pd.Series(pub_ar), 'Reg.Blk': ys['RB']}).reindex(DS)


def fmt_row(name, row, cols):
    vals = row[cols].astype(float)
    order = vals.sort_values(ascending=False)
    best, second = order.index[0], order.index[1]
    s = []
    for c in cols:
        v = vals[c]
        txt = f"{v:.3f}" if v >= 0 else "$<\\!0$"
        if c == best:
            txt = "\\textbf{\\color{ForestGreen}" + txt + "}"
        elif c == second:
            txt = "\\textbf{" + txt + "}"
        s.append(txt)
    return f"{name} & {N[name]} & {P[name]} & " + " & ".join(s) + " \\\\"


COLS = ['OLS', 'RF', 'MLP', 'FT-T', 'Att.Reg', 'Reg.Blk']
out['t1_corrected'] = "\n".join(fmt_row(d, t1.loc[d], COLS) for d in DS)

# ---------- FT-T vs RB paired uncertainty ----------
pv = load('audit/audit_ystd_results.csv').pivot_table(index=['dataset', 'seed_idx'], columns='model', values='r2')
rows = []
for d in DS:
    dif = (pv.loc[d]['RB'] - pv.loc[d]['FTT']).dropna()
    se = dif.std(ddof=1) / np.sqrt(len(dif))
    p = stats.ttest_1samp(dif, 0)[1]
    rows.append(f"{d} & {ys.loc[d,'FTT']:.3f} & {ys.loc[d,'RB']:.3f} & {dif.mean():+.3f} & {se:.3f} & {p:.2f} \\\\")
allp = (pv['RB'] - pv['FTT']).dropna()
se = allp.std(ddof=1) / np.sqrt(len(allp))
p = stats.ttest_1samp(allp, 0)[1]
rows.append(f"\\midrule\nPooled (40 dataset$\\times$seed pairs) & & & {allp.mean():+.3f} & {se:.3f} & {p:.3f} \\\\")
out['t1_se'] = "\n".join(rows)
out['t1_se_note'] = f"pooled {allp.mean():+.4f} SE {se:.4f} p={p:.4f}; sign test 8/8 two-sided p={2*0.5**8:.4f}; indiv. sig at 5%: " + \
    str([d for d in DS if stats.ttest_1samp((pv.loc[d]['RB'] - pv.loc[d]['FTT']).dropna(), 0)[1] < 0.05])

# ---------- 9-row ablation grid ----------
ab = m(load('audit/audit_ablystd_results.csv'), 'config')
f = m(load('audit/audit_faithful_results.csv'), 'variant')
a7 = load('audit/audit_a6ystd_results.csv').groupby('dataset')['r2'].mean().reindex(DS)
g = ab[['A0', 'A1', 'A2', 'A3', 'A4', 'A5']].copy()
g['A6'] = f['cur_L3']
g['A7'] = a7
g['RB'] = ab['RB']
cols = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'RB']
out['abl_grid'] = "\n".join(d + " & " + " & ".join(f"{g.loc[d, c]:.3f}" for c in cols) + " \\\\" for d in DS)
rk = g.rank(axis=1, ascending=False).mean()
PR = {'A0': 101697, 'A1': 106305, 'A2': 21697, 'A3': 173249, 'A4': 34177, 'A5': 33013,
      'A6': 89985, 'A7': 101377, 'RB': 30401}
LBL = {'A0': 'A0 & FT-Transformer (3 blocks, $d_{\\text{model}}=64$)',
       'A1': 'A1 & FT-Transformer + degree-2 polynomial inputs',
       'A2': 'A2 & Reg.~Blk without the polynomial expansion',
       'A3': 'A3 & Reg.~Blk without PCA',
       'A4': 'A4 & softmax attention in the Reg.~Blk skeleton (1 block)',
       'A5': 'A5 & FT-Transformer shrunk to the Reg.~Blk budget (3 blocks, $d_{\\text{model}}=36$)',
       'A6': 'A6 & Reg.~Blk stacked to 3 blocks',
       'A7': 'A7 & Reg.~Blk widened to the FT-T budget ($d_{\\text{model}}=128$, 260 components)',
       'RB': 'RB & Regression Block (1 block, $d_{\\text{model}}=64$)'}
out['abl_summary'] = "\n".join(f"{LBL[c]} & {g[c].median():.3f} & {g[c].mean():.3f} & {rk[c]:.2f} & {PR[c]:,} \\\\" for c in cols)
pva = load('audit/audit_ablystd_results.csv').pivot_table(index=['dataset', 'seed_idx'], columns='config', values='r2')
ff = load('audit/audit_faithful_results.csv')
pc3 = ff[ff.variant == 'cur_L3'].pivot_table(index=['dataset', 'seed_idx'], values='r2')['r2']
pa7 = load('audit/audit_a6ystd_results.csv').pivot_table(index=['dataset', 'seed_idx'], values='r2')['r2']
tests = {f'RB-{k}': pt(pva['RB'], pva[k]) for k in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']}
tests['A6-RB'] = pt(pc3, pva['RB'])
tests['A7-RB'] = pt(pa7, pva['RB'])
tests['A5-A0'] = pt(pva['A5'], pva['A0'])
tests['A4-A0'] = pt(pva['A4'], pva['A0'])
out['abl_tests'] = "; ".join(f"{k}: {v[0]:+.4f} p={v[1]:.4f}" for k, v in tests.items())
out['abl_wins'] = f"RB>A4 {(g['RB']>g['A4']).sum()}/8; A4>A0 {(g['A4']>g['A0']).sum()}/8; A5<A0 {(g['A5']<g['A0']).sum()}/8; RB>A3 {(g['RB']>g['A3']).sum()}/8; A7>RB {(g['A7']>g['RB']).sum()}/8"

# ---------- warm-start decomposition (corrected protocol) ----------
w = m(load('audit/audit_wsystd_results.csv'), 'config')
pw = load('audit/audit_wsystd_results.csv').pivot_table(index=['dataset', 'seed_idx'], columns='config', values='r2')
out['ws'] = "\n".join(f"{d} & {w.loc[d,'softmax']:.3f} & {w.loc[d,'rb_nowarm']:.3f} & {w.loc[d,'rb_warm']:.3f} \\\\" for d in DS)
out['ws_tests'] = f"readout(nowarm-softmax) {pt(pw['rb_nowarm'], pw['softmax'])}; warm(warm-nowarm) {pt(pw['rb_warm'], pw['rb_nowarm'])}; wins readout {(w['rb_nowarm']>w['softmax']).sum()}/8"
wr = load('results/warmstart_results.csv')
pwr = wr.pivot_table(index=['dataset', 'seed_idx'], columns='config', values='r2')
wraw = m(wr, 'config')
out['ws_raw'] = (f"raw-y warm vs cold {pt(pwr['rb_warm'], pwr['rb_nowarm'])}; "
                 f"cold shift mean|d| {(w['rb_nowarm']-wraw['rb_nowarm']).abs().mean():.4f} max {(w['rb_nowarm']-wraw['rb_nowarm']).abs().max():.4f}; "
                 f"warm shift mean|d| {(w['rb_warm']-wraw['rb_warm']).abs().mean():.4f} max {(w['rb_warm']-wraw['rb_warm']).abs().max():.4f}")

# ---------- classification ----------
cl = load('results/classify_results.csv')
c = cl.groupby(['dataset', 'model'])['acc'].mean().unstack()
tic = load('results/tabicl_results.csv')
tcol = 'acc' if 'acc' in tic else 'r2'
ti = tic.groupby('dataset')[tcol].mean()
CD = ['BreastCancer', 'Diabetes', 'Phoneme', 'Wine', 'Vehicle', 'Segment']
c = c.reindex(CD)
c['TabICL'] = ti.reindex(CD)
CK = ['LogReg', 'RF', 'FTT', 'RB', 'TabPFN', 'TabICL']
out['clf'] = "\n".join(f"{d} & " + " & ".join(f"{c.loc[d, k]:.3f}" for k in CK) + " \\\\" for d in CD) + \
    "\n\\midrule\nMean & " + " & ".join(f"{c[k].mean():.3f}" for k in CK) + " \\\\"
out['clf_rank'] = str(c[['LogReg', 'RF', 'FTT', 'RB']].rank(axis=1, ascending=False).mean().round(2).to_dict())

# ---------- uncapped + foundation models ----------
u = m(load('audit/audit_uy_results.csv'), 'model')
fm = m(load('audit/audit_fm_results.csv'), 'model')
fi = load('audit/audit_fm_tabicl_uncapped.csv').groupby('dataset')['r2'].mean()
nt = load('audit/audit_uy_results.csv').groupby('dataset')['n_train'].max()
out['uncapped'] = "\n".join(
    f"{d} & {nt[d]:,} & {u.loc[d,'OLS']:.3f} & {u.loc[d,'RF']:.3f} & {u.loc[d,'FTT']:.3f} & {u.loc[d,'RB']:.3f} & {fm.loc[d,'TabPFN']:.3f} & {fi[d]:.3f} \\\\"
    for d in ['California', 'Kin8nm', 'Protein'])

# ---------- higher dimensions + foundation models ----------
h = load('results/highdim_results.csv')
hm = h.groupby(['dataset', 'model'])['r2'].mean().unstack()
hp = h.groupby('dataset')['P'].max()
fh = load('audit/audit_fmhd_results.csv').groupby(['dataset', 'model'])['r2'].mean().unstack()
HD = ['CPU_act', 'Bank32nh', 'Ailerons', 'Pol', 'Superconductivity']


def f3(v):
    return f"{v:.3f}" if v > -1 else "$<\\!0$"


out['highdim'] = "\n".join(
    f"{d.replace('_', chr(92)+'_')} & {int(hp[d])} & {f3(hm.loc[d,'OLS'])} & {hm.loc[d,'RF']:.3f} & {hm.loc[d,'FTT']:.3f} & {hm.loc[d,'RB']:.3f} & {fh.loc[d,'TabPFN']:.3f} & {fh.loc[d,'TabICL']:.3f} \\\\"
    for d in HD)

# ---------- TabPFN on the eight datasets ----------
tp = load('results/tabpfn_results.csv').groupby('dataset')['r2'].mean().reindex(DS)
out['tabpfn8'] = "\n".join(f"{d} & {ys.loc[d,'FTT']:.3f} & {ys.loc[d,'RB']:.3f} & {tp[d]:.3f} \\\\" for d in DS)

# ---------- trimmed configuration + ladder ----------
pc = load('audit/audit_pca_results.csv')
tr = pc[pc.variant == 'mix2_ffn0_pca50'].groupby('dataset')['r2'].mean().reindex(DS)
out['trim'] = "\n".join(f"{d} & {ys.loc[d,'FTT']:.3f} & {ys.loc[d,'RB']:.3f} & {tr[d]:.3f} & {tr[d]-ys.loc[d,'FTT']:+.3f} \\\\" for d in DS) + \
    f"\n\\midrule\nMean & {ys['FTT'].mean():.3f} & {ys['RB'].mean():.3f} & {tr.mean():.3f} & {tr.mean()-ys['FTT'].mean():+.3f} \\\\"
lad = [('1 mixer + FFN (= Reg.~Blk)', 200, 'mix1_ffn1_pca200'), ('1 mixer + FFN', 100, 'mix1_ffn1_pca100'),
       ('1 mixer + FFN', 50, 'mix1_ffn1_pca50'), ('1 mixer + FFN', 25, 'mix1_ffn1_pca25'),
       ('2 mixers, no FFN', 200, 'mix2_ffn0_pca200'), ('2 mixers, no FFN', 100, 'mix2_ffn0_pca100'),
       ('2 mixers, no FFN', 50, 'mix2_ffn0_pca50'), ('2 mixers, no FFN', 25, 'mix2_ffn0_pca25')]
out['ladder'] = "\n".join(
    f"{l} & {k} & {pc[pc.variant==v].groupby('dataset')['r2'].mean().reindex(DS).mean():.3f} & {int(pc[(pc.variant==v)&(pc.dataset=='California')].params.max()):,} \\\\"
    for l, k, v in lad)
ptr = pc[pc.variant == 'mix2_ffn0_pca50'].pivot_table(index=['dataset', 'seed_idx'], values='r2')['r2']
out['trim_tests'] = f"trim vs FTT {pt(ptr, pv['FTT'])}; trim vs RB {pt(ptr, pv['RB'])}; trim>FTT {(tr>ys['FTT']).sum()}/8"
# trimmed on scope settings (agg n_mix=2 ncomp=50)
ag = load('audit_agg_results.csv')
ag = ag[(ag.n_mix == 2) & (ag.ncomp == 50)]
sc = ag.groupby(['setting', 'dataset'])['score'].mean()
out['trim_scope'] = "; ".join(f"{s}/{d}: {v:.3f}" for (s, d), v in sc.items() if s in ('uncapped', 'highdim'))

with open(os.path.join(HERE, 'tables_generated.tex'), 'w') as fh_:
    for k, v in out.items():
        fh_.write(f"%%%% ===== {k} =====\n{v}\n\n")
print(f"wrote {os.path.join(HERE, 'tables_generated.tex')}:", ", ".join(out))
print("\nT1 corrected:\n" + out['t1_corrected'])
for k in ['t1_se_note', 'abl_tests', 'abl_wins', 'ws_tests', 'ws_raw', 'clf_rank', 'trim_tests', 'trim_scope']:
    print(f"\n{k}: {out[k]}")
