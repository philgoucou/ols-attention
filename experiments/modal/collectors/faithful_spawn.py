"""Spawn the faithful-RB ladder on the deployed app; save call IDs."""
import json, modal
DS = ['California','Yacht','Energy','Concrete','Airfoil','Abalone','Kin8nm','Protein']
VARIANTS = ['cur_L1','gelu_L1','cls_L1','pertok_L1','faith_L1','faith_L2','faith_L3','cur_L3']
f = modal.Function.from_name("neurips-31482-faithful", "run_faithful")
calls = []
for d in DS:
    for v in VARIANTS:
        for s in range(5):
            fc = f.spawn(d, v, s)
            calls.append({'tag': 'faithful', 'args': [d, v, s], 'call_id': fc.object_id})
json.dump(calls, open('faithful_call_ids.json','w'), indent=1)
print(f"spawned {len(calls)} faithful-RB jobs")
