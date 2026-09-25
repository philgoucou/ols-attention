"""Spawn A6 (RB @ ~100K) on the deployed app; save call IDs for the collector."""
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
