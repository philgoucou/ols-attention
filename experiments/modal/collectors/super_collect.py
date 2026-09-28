"""Resilient collector for the spawned Superconductivity RB seeds (v2). Formerly
collect_super.py; splices the harvested seeds into highdim_results.csv next to itself.

Key properties:
  * The GPU work runs on the DEPLOYED app, fully server-side — it does not
    depend on this script, this laptop, or this network in any way.
  * Client-side errors (DNS blips, dropped wifi) are RETRIED with backoff,
    never recorded as seed failures. run_nn_big never raises remotely — it
    returns an error dict — so any exception here is infrastructure.
  * Each harvested seed is persisted to collected_results.json immediately;
    rerunning this script skips seeds already harvested. Modal holds spawned
    results server-side for ~24h, so the harvest can happen any time today.
  * Terminal remote failures (FunctionTimeoutError after retries) are the only
    thing recorded as a failed seed.

Splice: replaces highdim_results.csv rows per-seed, only for seeds actually
harvested clean; OOM rows for unharvested seeds are left in place.
"""
import json, os, time
import numpy as np
import pandas as pd
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = os.path.join(HERE, 'super_call_ids.json')
PROGRESS = os.path.join(HERE, 'collected_results.json')
CSV = os.path.join(HERE, 'highdim_results.csv')
OVERALL_DEADLINE_S = 4 * 3600
POLL_SLICE_S = 120

calls = json.load(open(IDS))
done = {}
if os.path.exists(PROGRESS):
    done = {int(k): v for k, v in json.load(open(PROGRESS)).items()}
    print(f"resuming: {len(done)} seeds already harvested {sorted(done)}", flush=True)

t0 = time.time()
for c in calls:
    s = c['seed_idx']
    if s in done:
        continue
    fc = modal.FunctionCall.from_id(c['call_id'])
    backoff = 5
    while True:
        if time.time() - t0 > OVERALL_DEADLINE_S:
            print(f"seed {s}: overall deadline reached, leaving unharvested", flush=True)
            break
        try:
            r = fc.get(timeout=POLL_SLICE_S)
            done[s] = r
            with open(PROGRESS, 'w') as fh:
                json.dump({str(k): v for k, v in done.items()}, fh)
            err = (r.get('error') or 'None').splitlines()[0]
            print(f"harvested seed {s}: r2={r.get('r2')} wall_s={r.get('wall_s')} err={err}", flush=True)
            break
        except Exception as e:
            name = type(e).__name__
            if name == 'FunctionTimeoutError':
                # the REMOTE function exceeded its 5400s limit (after retries) — terminal
                done[s] = {'dataset': 'Superconductivity', 'model': 'RB', 'P': 81,
                           'seed_idx': s, 'r2': float('nan'), 'params': None,
                           'wall_s': float('nan'), 'error': 'FunctionTimeoutError: remote 5400s limit'}
                with open(PROGRESS, 'w') as fh:
                    json.dump({str(k): v for k, v in done.items()}, fh)
                print(f"seed {s}: remote timeout — recorded as terminal failure", flush=True)
                break
            if isinstance(e, TimeoutError):
                # poll slice elapsed, job still running — keep waiting silently
                continue
            # anything else = client-side infrastructure -> backoff and retry
            print(f"seed {s}: client-side {name} ({str(e)[:70]}) — retrying in {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(120, backoff * 2)

clean = {s: r for s, r in done.items() if r.get('error') is None}
print(f"harvest complete: {len(clean)}/5 clean, {len(done) - len(clean)} failed, "
      f"{5 - len(done)} unharvested", flush=True)

if clean:
    df = pd.read_csv(CSV)
    for s, r in done.items():
        mask = ((df['dataset'] == 'Superconductivity') & (df['model'] == 'RB')
                & (df['seed_idx'] == s))
        df = df[~mask]
    df = pd.concat([df, pd.DataFrame(list(done.values()))], ignore_index=True)
    df.to_csv(CSV, index=False)
    print(f"SPLICED per-seed; Superconductivity RB clean-seed mean r2 = "
          f"{np.nanmean([r['r2'] for r in clean.values()]):+.4f}", flush=True)
else:
    print("NO CLEAN SEEDS harvested — csv untouched, limitation framing stands", flush=True)
print("COLLECTOR DONE", flush=True)
