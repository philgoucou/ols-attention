import json, modal
ARMS=['frac5','orig50']; PS=[10,20,50]
fsk = modal.Function.from_name("neurips-31482-friedman","run_sk")
fnn = modal.Function.from_name("neurips-31482-friedman","run_nn")
calls=[]
for arm in ARMS:
    for P in PS:
        for s in range(5):
            for m in ['OLS','RF']:
                calls.append({'tag':'fr','args':[arm,P,m,s],'call_id':fsk.spawn(arm,P,m,s).object_id})
            for m in ['FTT','RB']:
                calls.append({'tag':'fr','args':[arm,P,m,s],'call_id':fnn.spawn(arm,P,m,s).object_id})
json.dump(calls, open('friedman_call_ids.json','w'), indent=1)
print(f"spawned {len(calls)} friedman-control jobs")
