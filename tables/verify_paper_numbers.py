"""Check every number in the camera-ready against the released CSVs -- without re-running anything.

    python3 tables/verify_paper_numbers.py

It (1) recomputes every table of the revised paper from results/ (by executing gen_tables.py),
(2) parses tables/paper_snapshot.tex, the table bodies exactly as printed in the camera-ready,
and compares cell by cell; (3) checks the submitted-convention table (App. D) against the LaTeX
rows the original Colab notebooks printed; (4) recomputes every statistic quoted in the text
(paired differences, standard errors, p-values, ranks, win counts) and compares it with the
value the paper states. Exit code 0 only if everything matches.
"""
import io, os, re, sys, json, contextlib
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- 1. recompute (run the generator)
ns = {'__file__': os.path.join(HERE, 'gen_tables.py'), '__name__': 'gen_tables'}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(ns['__file__']).read(), ns)

# ---------------------------------------------------------------- 2. parse the snapshot
snap = {}
for block in open(os.path.join(HERE, 'paper_snapshot.tex')).read().split('%% ===== ')[1:]:
    label, _, body = block.partition(' =====\n')
    rows = []
    for line in body.strip().splitlines():
        cells = [c.strip() for c in line.split('&')]
        if 'multicolumn' in line or cells[0] in ('', 'Configuration') or cells[1] in ('Configuration',):
            continue
        rows.append(cells)
    snap[label] = rows

def num(cell):
    c = cell.replace(',', '').replace('\\_', '_').replace('$', '').replace('\\!', '').strip()
    if c in ('<0', '<\\!0'):
        return '<0'
    try:
        return float(c)
    except ValueError:
        return c

report, fails = [], 0
def check(where, got, want, tol):
    """got: computed number; want: cell string from the paper."""
    global fails
    w = num(want)
    if w == '<0':
        ok = got < 0
    elif isinstance(w, str):
        return  # a label cell
    else:
        ok = abs(got - w) <= tol
    if not ok:
        fails += 1
        report.append(f"  FAIL {where}: paper {want!r} vs computed {got:.4f}")
    return ok

n_cells = 0
def table(label, expected_rows):
    """expected_rows: list of (row_key, [computed values...]) aligned with the snapshot rows."""
    global n_cells
    rows = snap[label]
    assert len(rows) == len(expected_rows), (label, len(rows), len(expected_rows))
    for cells, (key, vals) in zip(rows, expected_rows):
        assert num(cells[0]) == key or cells[0].startswith(key), (label, cells[0], key)
        for j, v in enumerate(vals):
            if v is None:
                continue
            cell = cells[j + 1]
            dec = len(cell.split('.')[1]) if '.' in cell else 0          # decimals printed in the paper
            tol = 0.5 if dec == 0 else 0.51 * 10 ** (-dec)               # half a unit in the last printed digit
            check(f"{label} {cells[0]} col{j+1}", v, cell, tol)
            n_cells += 1

DS, CD = ns['DS'], ns['CD']
t1, pv, g, rk, PR = ns['t1'], ns['pv'], ns['g'], ns['rk'], ns['PR']
w, c, u, fm, fi, nt, hm, hp, fh, tp, tr, pc, lad, ys = (ns[k] for k in ['w','c','u','fm','fi','nt','hm','hp','fh','tp','tr','pc','lad','ys'])

# Table 1 (corrected protocol)
table('tab:realdata_main', [(d, [ns['N'][d], ns['P'][d]] + [t1.loc[d, k] for k in ['OLS','RF','MLP','FT-T','Att.Reg','Reg.Blk']]) for d in DS])
# App. D paired SEs
se_rows = []
for d in DS:
    dif = (pv.loc[d]['RB'] - pv.loc[d]['FTT']).dropna()
    se_rows.append((d, [ys.loc[d,'FTT'], ys.loc[d,'RB'], round(ys.loc[d,'RB'],3)-round(ys.loc[d,'FTT'],3), dif.std(ddof=1)/np.sqrt(len(dif)), stats.ttest_1samp(dif,0)[1]]))
allp = (pv['RB'] - pv['FTT']).dropna()
se_rows.append(('Pooled', [None, None, allp.mean(), allp.std(ddof=1)/np.sqrt(len(allp)), stats.ttest_1samp(allp,0)[1]]))
# the pooled row has empty FT-T/RB cells: align columns
rows = snap['tab:se']; pooled = rows[-1]; assert pooled[0].startswith('Pooled')
snap['tab:se'] = rows[:-1] + [[pooled[0], '', '', pooled[3], pooled[4], pooled[5]]]
table('tab:se', se_rows)
# App. E grid, summary, warm start, trimmed, ladder
cols = ['A0','A1','A2','A3','A4','A5','A6','A7','RB']
table('tab:abl', [(d, [g.loc[d, k] for k in cols]) for d in DS])
rows = snap['tab:ablsum']; snap['tab:ablsum'] = [[r[0]] + r[2:] for r in rows]   # drop the description column
table('tab:ablsum', [(k, [g[k].median(), g[k].mean(), rk[k], PR[k]]) for k in cols])
table('tab:ws', [(d, [w.loc[d,'softmax'], w.loc[d,'rb_nowarm'], w.loc[d,'rb_warm']]) for d in DS])
trim_rows = [(d, [ys.loc[d,'FTT'], ys.loc[d,'RB'], tr[d], round(tr[d],3)-round(ys.loc[d,'FTT'],3)]) for d in DS]
trim_rows.append(('Mean', [ys['FTT'].mean(), ys['RB'].mean(), tr.mean(), round(tr.mean(),3)-round(ys['FTT'].mean(),3)]))
trim_rows.append(('Params', [101697, 30401, 7489, None]))
table('tab:trim', trim_rows)
lad_rows = []
for l, k, v in lad:
    mean = pc[pc.variant == v].groupby('dataset')['r2'].mean().reindex(DS).mean()
    prm = int(pc[(pc.variant == v) & (pc.dataset == 'California')].params.max())
    lad_rows.append((l.split(' (')[0].split(' &')[0][:8], [k, mean, prm]))
snap['tab:ladder'] = [[r[0][:8]] + r[1:] for r in snap['tab:ladder']]
table('tab:ladder', lad_rows)
# App. F
table('tab:uncapped', [(d, [nt[d], u.loc[d,'OLS'], u.loc[d,'RF'], u.loc[d,'FTT'], u.loc[d,'RB'], fm.loc[d,'TabPFN'], fi[d]]) for d in ['California','Kin8nm','Protein']])
CK = ['LogReg','RF','FTT','RB','TabPFN','TabICL']
table('tab:clf', [(d, [c.loc[d, k] for k in CK]) for d in CD] + [('Mean', [c[k].mean() for k in CK])])
HD = ['CPU_act','Bank32nh','Ailerons','Pol','Superconductivity']
snap['tab:highdim'] = [[r[0].replace('\\_','_')] + r[1:] for r in snap['tab:highdim']]
table('tab:highdim', [(d, [int(hp[d]), hm.loc[d,'OLS'], hm.loc[d,'RF'], hm.loc[d,'FTT'], hm.loc[d,'RB'], fh.loc[d,'TabPFN'], fh.loc[d,'TabICL']]) for d in HD])
table('tab:tabpfn8', [(d, [ys.loc[d,'FTT'], ys.loc[d,'RB'], tp[d]]) for d in DS])

# ---------------------------------------------------------------- 3. App. D submitted convention vs the notebooks' printed rows
def nb_outputs(path, cell):
    nb = json.load(open(path)); c = nb['cells'][cell]; t = []
    for o in c.get('outputs', []):
        x = o.get('text');  t.append(''.join(x) if isinstance(x, list) else str(x or ''))
    return '\n'.join(t)
def latex_rows(txt):
    rows = {}
    for l in txt.splitlines():
        l2 = re.sub(r"\\textbf\{([^}]*)\}", r"\1", l)
        mm = re.match(r"\s*([A-Za-z0-9]+)\s*&\s*\d+\s*&\s*\d+\s*&(.*)\\\\", l2)
        if mm: rows[mm.group(1)] = [float(x) for x in re.findall(r"-?\d+\.\d+", mm.group(2))]
    return rows
NB = os.path.join(ROOT, 'paper_code', 'notebooks')
v3 = latex_rows(nb_outputs(os.path.join(NB, 'Attention_paper_simuls.ipynb'), 1))   # OLS RF GBM MLP Attn
v4 = latex_rows(nb_outputs(os.path.join(NB, 'Candidate2.ipynb'), 3))               # OLS RF Attn FTT Poly
orig_rows = [(d, [ns['N'][d], ns['P'][d], v3[d][0], v3[d][1], v3[d][3], v4[d][3], v3[d][4], v4[d][4]]) for d in DS]
table('tab:realdata_original', orig_rows)

# ---------------------------------------------------------------- 4. statistics quoted in the text
pva, pw, pwr, ptr = ns['pva'], ns['pw'], ns['pwr'], ns['ptr']
def pt(a, b):
    a, b = a.align(b, join='inner'); return (a - b).mean(), stats.ttest_rel(a, b)[1]
cpu = ns['load']('audit/audit_cpu_results.csv'); mc = cpu.groupby(['dataset','model'])['r2'].mean().unstack().reindex(DS)
uy = u
full = pd.DataFrame({'OLS': mc['OLS'], 'RF': mc['RF'], 'FTT': ys['FTT'], 'RB': ys['RB']})
for d in ['California','Kin8nm','Protein']:
    for k in ['OLS','RF','FTT','RB']: full.loc[d, k] = uy.loc[d, k]
frank = full.rank(axis=1, ascending=False).mean()
sc = ns['sc']
wraw = ns['wraw']
quotes = [
 ("Table 1 notes: pooled RB-FT-T mean", allp.mean(), 0.007, 0.0005),
 ("Table 1 notes: pooled SE", allp.std(ddof=1)/np.sqrt(len(allp)), 0.003, 0.0005),
 ("Table 1 notes: pooled p", stats.ttest_1samp(allp,0)[1], 0.009, 0.0005),
 ("Table 1 notes: sign test 8/8 p", 2*0.5**8, 0.008, 0.0005),
 ("Table 1 notes: RB ahead on 8/8", int((ys['RB']>ys['FTT']).sum()), 8, 0),
 ("Table 1 notes: individually significant datasets", sum(stats.ttest_1samp((pv.loc[d]['RB']-pv.loc[d]['FTT']).dropna(),0)[1]<0.05 for d in DS), 2, 0),
 ("Sec. 4: median RB-FT-T", (ys['RB']-ys['FTT']).median(), 0.004, 0.0005),
 ("Sec. 4: min displayed RB-FT-T (+0.001, differences of the printed 3-dp values)", (ys['RB'].round(3)-ys['FTT'].round(3)).min(), 0.001, 0.0005),
 ("Sec. 4: max RB-FT-T", (ys['RB']-ys['FTT']).max(), 0.029, 0.0005),
 ("Sec. 4 / App. E: RB-A2 mean", pt(pva['RB'], pva['A2'])[0], 0.215, 0.0005),
 ("App. E: RB-A4 mean", pt(pva['RB'], pva['A4'])[0], 0.117, 0.0005),
 ("App. E: RB-A4 p", pt(pva['RB'], pva['A4'])[1], 0.003, 0.0005),
 ("App. E: RB beats A4 on 8/8", int((g['RB']>g['A4']).sum()), 8, 0),
 ("App. E: A4 worse than FT-T on 7/8", int((g['A4']<g['A0']).sum()), 7, 0),
 ("App. E: RB-A3 mean", pt(pva['RB'], pva['A3'])[0], 0.013, 0.0005),
 ("App. E: A3 params / RB params = 5.7x", PR['A3']/PR['RB'], 5.7, 0.05),
 ("App. E: A5-A0 mean (loses 0.008)", -pt(pva['A5'], pva['A0'])[0], 0.008, 0.0005),
 ("App. E: A5-A0 p", pt(pva['A5'], pva['A0'])[1], 0.006, 0.0005),
 ("App. E: A5 median", g['A5'].median(), 0.892, 0.0005),
 ("App. E: A0 median", g['A0'].median(), 0.906, 0.0005),
 ("App. E: A4 median", g['A4'].median(), 0.771, 0.0005),
 ("App. E: A6-RB mean (-0.015)", pt(ns['pc3'], pva['RB'])[0], -0.015, 0.0005),
 ("App. E: A7-RB mean (+0.001)", pt(ns['pa7'], pva['RB'])[0], 0.001, 0.0005),
 ("App. E: A7-RB p (0.57)", pt(ns['pa7'], pva['RB'])[1], 0.57, 0.005),
 ("App. E: A2 loss vs RF-OLS gap, Spearman 0.71", stats.spearmanr(mc['RF']-mc['OLS'], g['RB']-g['A2']).statistic, 0.71, 0.005),
 ("App. E: readout +0.124", pt(pw['rb_nowarm'], pw['softmax'])[0], 0.124, 0.0005),
 ("App. E: readout p 0.003", pt(pw['rb_nowarm'], pw['softmax'])[1], 0.003, 0.0005),
 ("App. E: readout wins 8/8", int((w['rb_nowarm']>w['softmax']).sum()), 8, 0),
 ("App. E: warm start +0.001", pt(pw['rb_warm'], pw['rb_nowarm'])[0], 0.001, 0.0005),
 ("App. E: warm start p 0.59", pt(pw['rb_warm'], pw['rb_nowarm'])[1], 0.59, 0.005),
 ("App. E / App. C: warm start on raw targets +0.021", pt(pwr['rb_warm'], pwr['rb_nowarm'])[0], 0.021, 0.0005),
 ("App. E: raw-target p 0.021", pt(pwr['rb_warm'], pwr['rb_nowarm'])[1], 0.021, 0.0005),
 ("App. C: cold-start shift under rescaling, max 0.18", (w['rb_nowarm']-wraw['rb_nowarm']).abs().max(), 0.18, 0.005),
 ("App. C: warm-start shift under rescaling, max 0.03", (w['rb_warm']-wraw['rb_warm']).abs().max(), 0.03, 0.005),
 ("App. E: trimmed mean 0.774", tr.mean(), 0.774, 0.0005),
 ("App. E: FT-T mean 0.779", ys['FTT'].mean(), 0.779, 0.0005),
 ("App. E: trimmed ahead of FT-T on 3/8", int((tr>ys['FTT']).sum()), 3, 0),
 ("App. E: trimmed vs FT-T p 0.047", pt(ptr, pv['FTT'])[1], 0.047, 0.0005),
 ("App. E: trimmed vs RB gives up 0.011", -pt(ptr, pv['RB'])[0], 0.011, 0.0005),
 ("App. E: trimmed at full N, California 0.787", sc[('uncapped','California')], 0.787, 0.0005),
 ("App. E: trimmed at full N, Kin8nm 0.920", sc[('uncapped','Kin8nm')], 0.920, 0.0005),
 ("App. E: trimmed at full N, Protein 0.614", sc[('uncapped','Protein')], 0.614, 0.0005),
 ("App. F: trimmed CPU_act 0.963", sc[('highdim','CPU_act')], 0.963, 0.0005),
 ("App. F: trimmed Bank32nh 0.473", sc[('highdim','Bank32nh')], 0.473, 0.0005),
 ("App. F: trimmed Ailerons 0.817", sc[('highdim','Ailerons')], 0.817, 0.0005),
 ("App. F: trimmed Pol 0.981", sc[('highdim','Pol')], 0.981, 0.0005),
 ("App. F: full-size avg rank RB 1.62", frank['RB'], 1.62, 0.005),
 ("App. F: full-size avg rank FT-T 2.12", frank['FTT'], 2.12, 0.005),
 ("App. F: full-size avg rank RF 2.25", frank['RF'], 2.25, 0.005),
 ("App. F: full-size avg rank OLS 4.00", frank['OLS'], 4.00, 0.005),
 ("App. F: RB ahead of FT-T on 6/8 at full size", int((full['RB']>full['FTT']).sum()), 6, 0),
 ("App. F: full-size mean difference +0.003", (full['RB']-full['FTT']).mean(), 0.003, 0.0005),
 ("App. F: classification rank RB 2.08", c[['LogReg','RF','FTT','RB']].rank(axis=1, ascending=False).mean()['RB'], 2.08, 0.005),
 ("App. F: classification rank RF 2.17", c[['LogReg','RF','FTT','RB']].rank(axis=1, ascending=False).mean()['RF'], 2.17, 0.005),
 ("App. F: classification rank LogReg 2.58", c[['LogReg','RF','FTT','RB']].rank(axis=1, ascending=False).mean()['LogReg'], 2.58, 0.005),
 ("App. F: classification rank FT-T 3.17", c[['LogReg','RF','FTT','RB']].rank(axis=1, ascending=False).mean()['FTT'], 3.17, 0.005),
 ("App. D: MLP mean 0.758 -> 0.782 (corrected)", ns['mlp'].mean(), 0.782, 0.0005),
 ("App. D: MLP Airfoil corrected 0.917", ns['mlp']['Airfoil'], 0.917, 0.0005),
 ("App. D: RB moves at most +0.029 (Yacht)", (ys['RB'] - pd.Series({d: v4[d][4] for d in DS})).max(), 0.029, 0.0005),
]
ar_raw = ns['load']('audit/audit_attreg_results.csv').set_index(['dataset','seed_idx'])['r2']
ar_std = ns['load']('audit/audit_attreg_std_results.csv').set_index(['dataset','seed_idx'])['r2']
quotes.append(("App. D: Attention Regression max change under standardization <= 3e-7", float((ar_raw-ar_std).abs().max()), 3e-7, 1e-7))
n_quotes = 0
for desc, got, want, tol in quotes:
    n_quotes += 1
    if abs(got - want) > tol:
        fails += 1; report.append(f"  FAIL text: {desc}: paper {want} vs computed {got:.5f}")

# ---------------------------------------------------------------- report
print(f"table cells checked: {n_cells}   quoted statistics checked: {n_quotes}   failures: {fails}")
print("\n".join(report) if report else "ALL MATCH")
sys.exit(1 if fails else 0)
