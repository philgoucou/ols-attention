"""Spawn A6 (RB @ ~100K, the paper's row A7) on the deployed app; save call IDs for the collector.

Targets the FIRST-WAVE app name neurips-31482-a6 (raw targets -> results/rebuttal/a6_results.csv
via a6_collect.py). The surviving source, rebuttal_a7_widened.py, deploys the corrected-protocol
app neurips-31482-a6ystd (same function name run_a6; collector a6ystd_collect.py), so point the
from_name() call at that app to redo the App. E number."""
import json
import modal

DS_ORDER = ['California', 'Yacht', 'Energy', 'Concrete',
            'Airfoil',    'Abalone', 'Kin8nm', 'Protein']
N_REPEATS = 5

f = modal.Function.from_name("neurips-31482-a6", "run_a6")
calls = []
for ds in DS_ORDER:
    for s in range(N_REPEATS):
        fc = f.spawn(ds, s)
        calls.append({'dataset': ds, 'seed_idx': s, 'call_id': fc.object_id})
with open('a6_call_ids.json', 'w') as fh:
    json.dump(calls, fh, indent=1)
print(f"spawned {len(calls)} A6 jobs; ids saved to a6_call_ids.json")
