# experiments/modal/collectors/ — spawn and collect helpers

The deployed apps in `experiments/modal/` are run in two steps that survive the client: a
**spawn** script fires every cell of a grid with `modal.Function.from_name(app, fn).spawn(*args)`
and saves the call ids to `<tag>_call_ids.json`; a **collector** harvests those ids with
`FunctionCall.from_id(id).get()`, retrying client-side errors (DNS blips, dropped wifi) with
backoff, persisting progress to `<tag>_collected.json` after every few results so that it can be
re-run at any time, and finally writes one `audit_<tag>_results.csv` per tag. Modal keeps spawned
results server-side for about 24 hours, so the harvest can happen much later than the spawn.

**Paths.** Every collector locates its call-id and progress files through
`HERE = os.path.dirname(__file__)` and writes its CSV next to itself, i.e. **in this folder**.
Most spawn scripts write their `<tag>_call_ids.json` to the current working directory, so run
both from inside this folder:

```bash
modal deploy experiments/modal/rebuttal_faithful.py
cd experiments/modal/collectors
python faithful_spawn.py        # -> faithful_call_ids.json (here)
python faithful_collect.py      # -> audit_faithful_results.csv (here); re-run if interrupted
```

The `*_call_ids.json` and `*_collected.json` files are git-ignored. The CSVs of record were moved
to `results/audit/` (or `results/rebuttal/` for the first wave) after collection.

| Spawn | Collector | Deployed app / function | Writes |
|---|---|---|---|
| `a6_spawn.py` | `a6_collect.py` | `neurips-31482-a6` / `run_a6` — the first-wave, raw-target run of the widened block (the surviving source `rebuttal_a7_widened.py` deploys the corrected-protocol app `neurips-31482-a6ystd`) | `a6_results.csv` (in `results/rebuttal/`) |
| by hand, tag `a6ystd` | `a6ystd_collect.py` | `neurips-31482-a6ystd` / `run_a6` (`rebuttal_a7_widened.py`) | `audit_a6ystd_results.csv` |
| by hand, tag `a7nowarm` | `a7nowarm_collect.py` | `neurips-31482-a7nowarm` / `run_a6` (`rebuttal_a7_widened_nowarm.py`) | `audit_a7nowarm_results.csv` |
| `faithful_spawn.py` | `faithful_collect.py` | `neurips-31482-faithful` / `run_faithful` (`rebuttal_faithful.py`) | `audit_faithful_results.csv` |
| by hand, tag `fm` | `fm_collect.py` | `neurips-31482-fm-uncapped` / `run_fm_uncapped` (`rebuttal_fm_uncapped.py`) | `audit_fm_results.csv` (and, after the TabICL device fix, `audit_fm_tabicl_uncapped.csv`) |
| `fmhd_spawn.py` | `fmhd_collect.py` | `neurips-31482-fm-highdim` / `run_fm_highdim` (`rebuttal_fm_highdim.py`) | `audit_fmhd_results.csv` |
| `friedman_spawn.py` | `friedman_collect.py` | `neurips-31482-friedman` / `run_sk`, `run_nn` (`rebuttal_friedman.py`), arms `frac5`, `orig50` | `audit_fr_results.csv` |
| by hand, tag `fr3` | `friedman3_collect.py` | same app, arm `all100`, P in {5, 10, 20, 50} | `audit_fr3_results.csv` |
| by hand, tag `fr2` | `friedman2_collect.py` | `neurips-31482-friedman2` / `run_rb_ncomp` (`rebuttal_friedman2.py`) | `audit_fr2_results.csv` |
| `sublayer_spawn.py` | `sublayer_collect.py` | `neurips-31482-sublayer` / `run_sublayer` (`rebuttal_sublayer.py`) | `audit_sub_results.csv` |
| `super_spawn.py` (formerly `spawn_super.py`) | `super_collect.py` (formerly `collect_super.py`) | `neurips-31482-highdim` / `run_nn_big` (`rebuttal_highdim.py`): the five Superconductivity (P = 81) Regression Block seeds on an A100-80GB, spliced into `highdim_results.csv` | `highdim_results.csv` (in `results/rebuttal/`) |
| by hand, tag `uy` | `uy_collect.py` | `neurips-31482-uncapped-ystd` / `run_sk_ystd`, `run_nn_ystd` (`rebuttal_uncapped_ystd.py`) | `audit_uy_results.csv` |

"By hand" means the grid was spawned with the same three lines as the spawn scripts, typed at
a Python prompt, and harvested with a copy of the collector template. The template is the same
file every time (only the `IDS` / `PROGRESS` names change), so any family without a helper here —
the corrected-protocol ablation and warm-start grids (`rebuttal_ablation_warmstart.py`, tags
`ablystd` and `wsystd`), the trimmed configuration (`rebuttal_trimmed.py`, tag `agg`), the PCA ladder
and no-FFN variants (`rebuttal_sublayer.py`, tags `pca` and `noffn`), the residual / LayerNorm
ablations (`rebuttal_resln.py`, tags `resln` and `lnfix`), the mixer / attention compositions
(`rebuttal_mixattn.py`, tag `mixattn`) and the corrected MLP column
(`experiments/audit/audit_probes_mlp_ystd.py`, tag `mlpystd`, collector in
`experiments/audit/collectors/`) — is redone as:

```python
import json, modal
f = modal.Function.from_name("neurips-31482-ystd-grids", "run_abl_ystd")
calls = [{'tag': 'ablystd', 'args': [d, c, s], 'call_id': f.spawn(d, c, s).object_id}
         for d in DS for c in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'RB'] for s in range(5)]
json.dump(calls, open('ablystd_call_ids.json', 'w'), indent=1)
```

followed by a copy of, say, `faithful_collect.py` with `IDS = 'ablystd_call_ids.json'` and
`PROGRESS = 'ablystd_collected.json'`.
