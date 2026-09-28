"""Fire-and-forget: spawn the 5 Superconductivity RB seeds on the DEPLOYED app
(rebuttal_highdim.py::run_nn_big, A100-80GB). Formerly spawn_super.py.
Call IDs go to super_call_ids.json so any later client can collect."""
import json
import modal

f = modal.Function.from_name("neurips-31482-highdim", "run_nn_big")
calls = []
for s in range(5):
    fc = f.spawn('Superconductivity', 'RB', s)
    calls.append({'seed_idx': s, 'call_id': fc.object_id})
    print(f"spawned seed {s}: {fc.object_id}", flush=True)
with open('super_call_ids.json', 'w') as fh:
    json.dump(calls, fh, indent=1)
print("all spawned; ids saved to super_call_ids.json")
