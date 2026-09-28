"""Resilient collector for the first-wave (raw-target) A6 spawns of a6_spawn.py, i.e.
the app neurips-31482-a6 -> results/rebuttal/a6_results.csv (same contract as super_collect.py v2):
client-side errors retried with backoff, per-call progress persisted, rerunnable.
Writes a6_results.csv when all calls resolve."""
import json, os, time
import numpy as np
import pandas as pd
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = os.path.join(HERE, 'a6_call_ids.json')
PROGRESS = os.path.join(HERE, 'a6_collected.json')
CSV = os.path.join(HERE, 'a6_results.csv')
OVERALL_DEADLINE_S = 3 * 3600
POLL_SLICE_S = 120

calls = json.load(open(IDS))
done = {}
if os.path.exists(PROGRESS):
    done = json.load(open(PROGRESS))
    print(f"resuming: {len(done)}/{len(calls)} already harvested", flush=True)

t0 = time.time()
for c in calls:
    key = f"{c['dataset']}|{c['seed_idx']}"
    if key in done:
        continue
    fc = modal.FunctionCall.from_id(c['call_id'])
    backoff = 5
    while True:
        if time.time() - t0 > OVERALL_DEADLINE_S:
            print(f"{key}: overall deadline reached, leaving unharvested", flush=True)
            break
        try:
            r = fc.get(timeout=POLL_SLICE_S)
            done[key] = r
            with open(PROGRESS, 'w') as fh:
                json.dump(done, fh)
            err = (r.get('error') or 'None').splitlines()[0]
            print(f"harvested {key}: r2={r.get('r2')} params={r.get('params')} err={err}", flush=True)
            break
        except Exception as e:
            name = type(e).__name__
            if name == 'FunctionTimeoutError':
                done[key] = {'kind': 'a6', 'dataset': c['dataset'], 'config': 'A6',
                             'seed_idx': c['seed_idx'], 'r2': float('nan'), 'params': None,
                             'd_model': None, 'ncomp': None, 'ftt_target': None,
                             'wall_s': float('nan'),
                             'error': 'FunctionTimeoutError: remote limit'}
                with open(PROGRESS, 'w') as fh:
                    json.dump(done, fh)
                print(f"{key}: remote timeout — terminal", flush=True)
                break
            if isinstance(e, TimeoutError):
                continue
            print(f"{key}: client-side {name} — retrying in {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(120, backoff * 2)

rows = list(done.values())
clean = [r for r in rows if r.get('error') is None]
print(f"harvest complete: {len(clean)}/{len(calls)} clean", flush=True)
if rows:
    pd.DataFrame(rows).to_csv(CSV, index=False)
    if clean:
        df = pd.DataFrame(clean)
        agg = df.groupby('dataset')['r2'].mean()
        print("A6 mean R2 by dataset:", flush=True)
        for ds, v in agg.items():
            print(f"  {ds:<11} {v:+.4f}", flush=True)
    print(f"WROTE {CSV} with {len(rows)} rows", flush=True)
print("A6 COLLECTOR DONE", flush=True)
