"""Resilient collector for the audit probe suite. Same contract as the P=81
collector: client-side errors retried with backoff, per-call progress persisted,
rerunnable any time. Writes audit_<tag>_results.csv per probe family."""
import json, os, time
from collections import defaultdict
import pandas as pd
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = os.path.join(HERE, 'sublayer_call_ids.json')
PROGRESS = os.path.join(HERE, 'sublayer_collected.json')
OVERALL_DEADLINE_S = 5 * 3600
POLL_SLICE_S = 120

calls = json.load(open(IDS))
done = {}
if os.path.exists(PROGRESS):
    done = json.load(open(PROGRESS))
    print(f"resuming: {len(done)}/{len(calls)} already harvested", flush=True)

t0 = time.time()
n_since_save = 0
for c in calls:
    key = c['call_id']
    if key in done:
        continue
    fc = modal.FunctionCall.from_id(key)
    backoff = 5
    while True:
        if time.time() - t0 > OVERALL_DEADLINE_S:
            print(f"{c['tag']}{c['args']}: deadline reached, unharvested", flush=True)
            break
        try:
            r = fc.get(timeout=POLL_SLICE_S)
            done[key] = r
            n_since_save += 1
            if n_since_save >= 5:
                with open(PROGRESS, 'w') as fh: json.dump(done, fh)
                n_since_save = 0
            err = (r.get('error') or '')[:60]
            print(f"[{len(done)}/{len(calls)}] {c['tag']}{c['args']}"
                  + (f" ERR {err}" if err else ""), flush=True)
            break
        except Exception as e:
            name = type(e).__name__
            if name == 'FunctionTimeoutError':
                done[key] = {'kind': c['tag'], 'args': c['args'], 'error': 'FunctionTimeoutError'}
                print(f"{c['tag']}{c['args']}: remote timeout — terminal", flush=True)
                break
            if isinstance(e, TimeoutError):
                continue
            print(f"{c['tag']}{c['args']}: client {name} — retry {backoff}s", flush=True)
            time.sleep(backoff); backoff = min(120, backoff * 2)

with open(PROGRESS, 'w') as fh:
    json.dump(done, fh)

by_tag = defaultdict(list)
for c in calls:
    if c['call_id'] in done:
        by_tag[c['tag']].append(done[c['call_id']])
for tag, rows in by_tag.items():
    out = os.path.join(HERE, f"audit_{tag}_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    nerr = sum(1 for r in rows if r.get('error'))
    print(f"WROTE audit_{tag}_results.csv: {len(rows)} rows, {nerr} errors", flush=True)
print("SUBLAYER COLLECTOR DONE", flush=True)
