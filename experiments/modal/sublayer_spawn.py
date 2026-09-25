import json, modal
DS=['California','Yacht','Energy','Concrete','Airfoil','Abalone','Kin8nm','Protein']
GRID=[(1,1),(1,3),(3,1),(3,3)]
f=modal.Function.from_name("neurips-31482-sublayer","run_sublayer")
calls=[]
for d in DS:
    for nm,nf in GRID:
        for s in range(5):
            calls.append({'tag':'sub','args':[d,nm,nf,s],'call_id':f.spawn(d,nm,nf,s).object_id})
json.dump(calls, open('sublayer_call_ids.json','w'), indent=1)
print(f"spawned {len(calls)} sublayer jobs")
