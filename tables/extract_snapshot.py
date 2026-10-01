#!/usr/bin/env python3
"""Extract the table bodies of the camera-ready into tables/paper_snapshot.tex (numbers only).

    python3 tables/extract_snapshot.py /path/to/main.tex

Every table or apptable environment carrying a label tab:... contributes one block
'%% ===== tab:label =====' followed by its data rows, bold/colour stripped, one row per line;
header rows, panel rows and rules are skipped. verify_paper_numbers.py reads this file.
"""
import re, sys, os
src = open(sys.argv[1], encoding='utf-8').read()
src = re.sub(r'(?m)(?<!\\)%.*$', '', src)   # drop LaTeX comments (a comment in the preamble mentions \begin{table})
out = ['%% Table bodies exactly as printed in the camera-ready (numbers only; bold/colour stripped).',
       '%% Extracted from main.tex by tables/extract_snapshot.py. verify_paper_numbers.py checks every cell against the CSVs.', '']
n = 0
for m in re.finditer(r'\\begin\{(table|apptable)\}(.*?)\\end\{\1\}', src, re.S):
    body = m.group(2)
    lab = re.search(r'\\label\{(tab:[^}]*)\}', body)
    if not lab:
        continue
    rows = []
    for line in body.splitlines():
        t = line.strip()
        if '&' not in t or not t.endswith('\\\\') or '\\multicolumn' in t:
            continue
        raw = [c.strip() for c in t[:-2].split('&')]
        if all((c.startswith('\\textbf') or c.startswith('\\boldmath') or c == '') for c in raw):
            continue   # header row
        t = t[:-2].strip()
        t = re.sub(r'\\textbf\{\\color\{[^}]*\}([^}]*)\}', r'\1', t)
        t = re.sub(r'\\textbf\{([^}]*)\}', r'\1', t)
        t = re.sub(r'\\color\{[^}]*\}', '', t)
        t = t.replace('$<\\!0$', '<0')
        cells = [c.strip() for c in t.split('&')]
        if not any(re.search(r'\d', c) for c in cells[1:]):
            continue
        rows.append(' & '.join(cells))
    out.append('%% ===== ' + lab.group(1) + ' ====='); out.extend(rows); out.append(''); n += 1
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_snapshot_new.tex')
open(dst, 'w', encoding='utf-8').write('\n'.join(out))
print('wrote', dst, 'with', n, 'tables')
