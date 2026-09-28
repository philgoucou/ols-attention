# experiments/audit/ — text-vs-code audit probes

Four rounds of Modal probes written while auditing the submitted paper against its code, plus
the corrected MLP column. Each probe re-implements a model *from the paper's own description*
and runs it on the Table 1 protocol (eight datasets, N capped at 5000, five random 80/20 splits,
`seed = 42 + 1000 * seed_idx`) or on the Monte Carlo grid (6 DGPs × 4 N × 4 SNR), so that every
checkable claim is checked against an independent implementation. The two rows marked in bold
feed the revised paper directly; the others are verifications and forensics whose conclusions
are in the text and in the README's provenance section.

| Script (Modal app) | Remote function → tag | CSV in `results/audit/` | Role |
|---|---|---|---|
| `audit_probes.py` (`neurips-31482-audit`) | `run_ystd` → `ystd` | `audit_ystd_results.csv` (80 rows) | FT-Transformer and Regression Block under the corrected protocol (target standardized on the training split): **the FT-T and Reg. Block columns of Table 1**, the paired SEs and the sign test of App. D, and the FT-T / RB reference columns of the App. E and F tables |
| `audit_probes_mlp_ystd.py` (`neurips-31482-mlpystd`; formerly `experiments/modal/audit_probes_mlpystd.py`) | `run_mlp` → `mlpystd` | `audit_mlpystd_results.csv` (40) | the MLP per the paper's spec (3 × 200 ReLU, dropout .2, Adam 1e-3, batch 64, patience 20, 15 % validation) with the target standardized: **the MLP column of Table 1**. A copy of `audit_probes.py` whose only functional difference is `ystd=True` in `run_mlp` |
| `audit_probes.py` | `run_cpu` → `cpu` | `audit_cpu_results.csv` (80) | OLS and RF, capped: reproduce the submitted columns (scale-invariant, so the published values were kept) |
| `audit_probes.py` | `run_mlp` → `mlp` | `audit_mlp_results.csv` (40) | the MLP with the features standardized only (the submitted convention) |
| `audit_probes.py` | `run_attreg` → `attreg` | `audit_attreg_results.csv` (40) | the Eq. (24) Attention Regression estimator re-implemented exactly per the appendix, raw targets |
| `audit_probes.py` | `run_sens` → `sens` | `audit_sens_results.csv` (240) | the App. C sensitivity sentence: variants `d32`, `d128`, `dr00`, `dr03`, `lr1em4`, `lr1em2` around the shipped configuration |
| `audit_probes.py` | `run_mc_cell` → `mc` | `audit_mc_results.csv` (96 cells) | the full Monte Carlo per the appendix spec: 6 DGPs × 4 N × 4 SNR × 10 replications, OLS / RF / GBM / MLP / Att. Reg (mean and sd per cell) |
| `audit_probes2.py` (`neurips-31482-audit2`) | `run_attreg_std` → `attreg_std`; `run_mc_attreg_std` → `mc_attreg_std` | `audit_attreg_std_results.csv` (40); `audit_mc_attreg_std_results.csv` (96) | round 2: Att. Reg with features and targets standardized, on Table 1 and on the Monte Carlo grid (the scale-equivariance check behind keeping the published Att. Reg column) |
| `audit_probes3.py` (`neurips-31482-audit3`) | `run_legacy_attn(with_retry)` → `legacy_attn`; `run_mc_legacy` → `mc_legacy` | `audit_legacy_attn_results.csv` (80); `audit_mc_legacy_results.csv` (96) | round 3, forensics only: verbatim port of the legacy `train_torch_model` (a generic 2-block self-attention regressor) with and without the legacy pipeline's test-conditioned retry, to fingerprint the provenance of the submitted Att. Reg numbers |
| `audit_probes4.py` (`neurips-31482-audit4`) | `run_legacy_attn(steps, ystd)` → `legacy_attn4` | `audit_legacy4_results.csv` (80) | round 4, forensics only: the same net with 300 and 3000 steps under standardized targets |

Byte-identical copies of the eleven round 1–3 CSVs also sit in `results/rebuttal/`, where they
were first collected during the July 2026 rebuttal.

## Running

```bash
modal deploy experiments/audit/audit_probes.py
cd experiments/audit/collectors
python audit_spawn.py       # all six round-1 families -> audit_call_ids.json (576 jobs)
python audit_collect.py     # rerunnable; one audit_<tag>_results.csv per family, written here
```

Rounds 2–4 and the MLP rerun were spawned by hand with the same
`modal.Function.from_name(app, fn).spawn(*args)` pattern (see
`experiments/modal/collectors/README.md`) and harvested with `audit2_collect.py`,
`audit3_collect.py`, `audit4_collect.py` and `mlpystd_collect.py`, which live in
[`collectors/`](collectors/README.md). GPUs: T4 for the neural probes, CPU for `run_cpu`.
