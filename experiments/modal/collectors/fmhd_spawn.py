"""Spawn the FM high-dim grid against the DEPLOYED app, so the jobs survive this
client. Writes call ids for the resilient collector."""
import json, os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['CPU_act', 'Bank32nh', 'Ailerons', 'Pol', 'Superconductivity',
            'Friedman-P20', 'Friedman-P50']
MODELS = ['TabPFN', 'TabICL']
N_REPEATS = 5

fn = modal.Function.from_name("neurips-31482-fm-highdim", "run_fm_highdim")
calls = []
for ds in DATASETS:
    for mod in MODELS:
        for s in range(N_REPEATS):
            h = fn.spawn(ds, mod, s)
            calls.append({'tag': 'fmhd', 'args': [ds, mod, s], 'call_id': h.object_id})
with open(os.path.join(HERE, 'fmhd_call_ids.json'), 'w') as f:
    json.dump(calls, f, indent=1)
print(f"spawned {len(calls)} jobs -> fmhd_call_ids.json")
