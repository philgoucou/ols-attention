# experiments/audit/collectors/

Spawn and collect helpers of the audit probes; same contract as
[`experiments/modal/collectors/`](../../modal/collectors/README.md). Each collector reads
`<tag>_call_ids.json` and `<tag>_collected.json` next to itself (`HERE = os.path.dirname(__file__)`)
and writes `audit_<tag>_results.csv` **in this folder**; `audit_spawn.py` writes
`audit_call_ids.json` to the current working directory, so run it from here.

| File | App | Harvests |
|---|---|---|
| `audit_spawn.py` | `neurips-31482-audit` (`audit_probes.py`) | spawns all six round-1 families: `ystd` (80), `sens` (240), `cpu` (80), `mlp` (40), `attreg` (40), `mc` (96) |
| `audit_collect.py` | `neurips-31482-audit` | `audit_<tag>_results.csv` for the tags `ystd`, `sens`, `cpu`, `mlp`, `attreg`, `mc` |
| `audit2_collect.py` | `neurips-31482-audit2` (`audit_probes2.py`) | `audit_attreg_std_results.csv`, `audit_mc_attreg_std_results.csv` |
| `audit3_collect.py` | `neurips-31482-audit3` (`audit_probes3.py`) | `audit_legacy_attn_results.csv`, `audit_mc_legacy_results.csv` |
| `audit4_collect.py` | `neurips-31482-audit4` (`audit_probes4.py`) | `audit_legacy4_results.csv` |
| `mlpystd_collect.py` | `neurips-31482-mlpystd` (`audit_probes_mlp_ystd.py`) | `audit_mlpystd_results.csv` |
