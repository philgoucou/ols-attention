"""Spawn the full audit probe suite on the deployed app; save call IDs."""
import json
import modal

DS = ['California', 'Yacht', 'Energy', 'Concrete', 'Airfoil', 'Abalone', 'Kin8nm', 'Protein']
SENS = ['d32', 'd128', 'dr00', 'dr03', 'lr1em4', 'lr1em2']
MC_DGPS = ['linear', 'friedman1', 'friedman2', 'friedman3', 'rotated_sine', 'soft_radial']
MC_NS = [500, 1000, 2500, 5000]
MC_SNRS = [0.5, 1.0, 2.0, 3.0]

APP = "neurips-31482-audit"
calls = []

def spawn(fn_name, args_list, tag):
    f = modal.Function.from_name(APP, fn_name)
    for a in args_list:
        fc = f.spawn(*a)
        calls.append({'tag': tag, 'args': list(a), 'call_id': fc.object_id})
    print(f"spawned {len(args_list):4d} {tag}")

spawn('run_ystd',   [(d, m, s) for d in DS for m in ('FTT', 'RB') for s in range(5)], 'ystd')
spawn('run_sens',   [(d, v, s) for d in DS for v in SENS for s in range(5)], 'sens')
spawn('run_cpu',    [(d, m, s) for d in DS for m in ('OLS', 'RF') for s in range(5)], 'cpu')
spawn('run_mlp',    [(d, s) for d in DS for s in range(5)], 'mlp')
spawn('run_attreg', [(d, s) for d in DS for s in range(5)], 'attreg')
spawn('run_mc_cell', [(g, n, r) for g in MC_DGPS for n in MC_NS for r in MC_SNRS], 'mc')

with open('audit_call_ids.json', 'w') as fh:
    json.dump(calls, fh, indent=1)
print(f"TOTAL {len(calls)} audit jobs spawned; ids saved")
