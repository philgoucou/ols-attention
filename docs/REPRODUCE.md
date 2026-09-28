# Reproducing the paper's tables

Fastest check, no GPU: `python3 tables/verify_paper_numbers.py` recomputes every table and every quoted statistic from `results/` and compares with the camera-ready (488 cells + 60 statistics; all match on 28 Sept 2026).


Every number in the revised paper comes from a CSV under `results/`, and every LaTeX table body
is regenerated from those CSVs by one command. Re-running the experiments themselves needs a
[Modal](https://modal.com) account and GPU credit. For each table below: the CSV(s) it reads,
the block of `tables/gen_tables.py` that prints it, and the command(s) that produced the CSV.
The per-family details (apps, remote functions, tags, collectors) are in
`experiments/modal/README.md` and `experiments/audit/README.md`.

## 0. Setup and the one command that rebuilds the tables

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, scikit-learn, torch, modal, tabpfn, tabicl
python3 tables/gen_tables.py        # no GPU, a few seconds
```

`gen_tables.py` prints the corrected Table 1 and every test statistic to the console and writes
all table bodies to `tables/tables_generated.tex`, one block per `%%%% ===== <name> =====` marker:
`t1_corrected`, `t1_se`, `t1_se_note`, `abl_grid`, `abl_summary`, `abl_tests`, `abl_wins`, `ws`,
`ws_tests`, `ws_raw`, `clf`, `clf_rank`, `uncapped`, `highdim`, `tabpfn8`, `trim`, `ladder`,
`trim_tests`, `trim_scope`. Rows whose `error` column is non-empty are dropped before averaging.

To re-run experiments: `pip install modal && modal token new`. The eight Table 1 datasets are
cached in the Modal volume `neurips-31482-cache` by the `fetch_and_cache` step at the start of
`modal run experiments/modal/rebuttal_v4_port.py`; every deployed app reads that cache
(`datasets_v1.pkl`). The deployed-app pattern is

```bash
modal run    experiments/modal/rebuttal_fm_highdim.py::canary   # one cheap cell per model: checks code, image and cache
modal deploy experiments/modal/rebuttal_fm_highdim.py           # deployed app: jobs survive the client
cd experiments/modal/collectors
python fmhd_spawn.py                                            # spawn the grid  -> fmhd_call_ids.json
python fmhd_collect.py                                          # harvest, rerunnable -> audit_fmhd_results.csv
```

(`canary` entrypoints exist on `rebuttal_fm_highdim.py` and `rebuttal_trimmed.py`; the other
deployed apps go straight to `modal deploy`. Apps with a `main` entrypoint are run with
`modal run <app>.py` and write their CSV in the current directory.)

## Table 1 — corrected protocol

| | |
|---|---|
| Block | `t1_corrected` |
| CSVs | `results/audit/audit_ystd_results.csv` (FT-T, Reg. Blk), `results/audit/audit_mlpystd_results.csv` (MLP). OLS, RF and Att. Reg are the published values (hard-coded as `pub_ols`, `pub_rf`, `pub_ar` in `gen_tables.py`), kept because they are scale-equivariant: verified by `audit_cpu_results.csv` and `audit_attreg_results.csv` / `audit_attreg_std_results.csv`. |
| Commands | `modal deploy experiments/audit/audit_probes.py`, then in `experiments/audit/collectors/`: `python audit_spawn.py` and `python audit_collect.py` (families `ystd`, `cpu`, `attreg`, and also `sens`, `mlp`, `mc`). Then `modal deploy experiments/audit/audit_probes_mlp_ystd.py`, spawn `run_mlp(dataset, seed_idx)` over the 8 datasets × 5 seeds with tag `mlpystd`, and `python mlpystd_collect.py`. |

## Appendix D — protocol

**Submitted convention (raw targets), FT-T and Reg. Blk.** `results/rebuttal/ablation_grid_results.csv`,
configs `A0` (= the FT-Transformer) and `RB`, produced by
`modal run experiments/modal/rebuttal_v4_port.py --part abl` (the Modal port of
`paper_code/real_data_benchmark_v4.py`; the submitted numbers themselves come from the notebook
run described in the README's provenance section). Not a `gen_tables.py` block; the means are

```python
import pandas as pd
d = pd.read_csv('results/rebuttal/ablation_grid_results.csv')
d[d.config.isin(['A0', 'RB'])].groupby(['dataset', 'config']).r2.mean().unstack()
```

**Paired standard errors and sign test.** Blocks `t1_se` and `t1_se_note`, from the 40 dataset ×
seed pairs of `results/audit/audit_ystd_results.csv`.

## Appendix E — ablations

**Ablation grid (A0–A7, RB).** Block `abl_grid`; tests `abl_tests`, `abl_wins`.

| Rows | CSV | Produced by |
|---|---|---|
| A0–A5, RB | `results/audit/audit_ablystd_results.csv` | `modal deploy experiments/modal/rebuttal_ablation_warmstart.py`; spawn `run_abl_ystd(dataset, config, seed_idx)` for configs A0–A5, RB with tag `ablystd`; collect |
| A6 (three stacked blocks) | `results/audit/audit_faithful_results.csv`, variant `cur_L3` | `modal deploy experiments/modal/rebuttal_faithful.py`; `python faithful_spawn.py`; `python faithful_collect.py` |
| A7 (widened block) | `results/audit/audit_a6ystd_results.csv` (config label `A6` inside the file is this row's pre-renumbering name) | `modal deploy experiments/modal/rebuttal_a7_widened.py`; spawn `run_a6(dataset, seed_idx)` with tag `a6ystd`; `python a6ystd_collect.py` |
| A7 cold-start twin (text) | `results/audit/audit_a7nowarm_results.csv` | same with `rebuttal_a7_widened_nowarm.py`, tag `a7nowarm`, `a7nowarm_collect.py` |

**Summary table (median, mean, mean rank, parameters).** Block `abl_summary`, from the same CSVs.
The parameter counts are the `PR` dict hard-coded in `gen_tables.py`: for A0–A5 and RB they equal the
`params` column of `audit_ablystd_results.csv` at P = 8; for A6 and A7 (89,985 and 101,377) they equal
the P = 9 (Protein) rows of `audit_faithful_results.csv` (`cur_L3`) and `audit_a6ystd_results.csv`,
whose P = 8 rows read 89,921 and 101,249 (see the note in `results/README.md`).

**Readout vs warm start.** Blocks `ws`, `ws_tests` from `results/audit/audit_wsystd_results.csv`
(`rebuttal_ablation_warmstart.py`, `run_ws_ystd(dataset, config, seed_idx)` for configs `softmax`,
`rb_nowarm`, `rb_warm`, tag `wsystd`). Block `ws_raw` compares with the raw-target run
`results/rebuttal/warmstart_results.csv` from `modal run experiments/modal/rebuttal_warmstart_lcurve.py`.

**Trimmed configuration and PCA ladder.** Blocks `trim`, `trim_tests`, `ladder` from
`results/audit/audit_pca_results.csv` (`modal deploy experiments/modal/rebuttal_sublayer.py`;
spawn `run_pca_sweep(dataset, n_mix, n_ffn, ncomp, seed_idx)` for `(1, 1)` and `(2, 0)` mixers /
FFN sublayers × ncomp in {25, 50, 100, 200}, tag `pca`; collect). The trimmed configuration is the
variant `mix2_ffn0_pca50`. Block `trim_scope` (the trimmed configuration at full N and at higher P)
is from `results/audit/audit_agg_results.csv`, `n_mix = 2`, `ncomp = 50`
(`modal run experiments/modal/rebuttal_trimmed.py::canary` re-runs one cell per setting against the
CSV; `modal deploy` it and spawn `run_trim(setting, dataset, n_mix, ncomp, seed_idx)` with tag `agg`
for the full grid).

## Appendix F — scope

**Full sample size.** Block `uncapped`: `results/audit/audit_uy_results.csv` (OLS, RF, FT-T, RB;
`modal deploy experiments/modal/rebuttal_uncapped_ystd.py`, spawn `run_sk_ystd` / `run_nn_ystd`
`(dataset, model, seed_idx)` with tag `uy`, `python uy_collect.py`), `results/audit/audit_fm_results.csv`
(TabPFN) and `results/audit/audit_fm_tabicl_uncapped.csv` (TabICL) from
`experiments/modal/rebuttal_fm_uncapped.py` (`modal run ...::show_probe` first reports the API and
row limits; then `modal deploy`, spawn `run_fm_uncapped(dataset, model, seed_idx)` with tag `fm`,
`python fm_collect.py`).

**Classification.** Blocks `clf`, `clf_rank`: `results/rebuttal/classify_results.csv` from
`modal run experiments/modal/rebuttal_classify.py` (LogReg, RF, FT-T, RB, TabPFN) and
`results/rebuttal/tabicl_results.csv` from `modal run experiments/modal/rebuttal_tabicl.py`.

**Higher dimensions.** Block `highdim`: `results/rebuttal/highdim_results.csv` from
`modal run experiments/modal/rebuttal_highdim.py` (the five Superconductivity RB seeds were re-run
on an A100-80GB with `super_spawn.py` / `super_collect.py`, or equivalently
`modal run experiments/modal/rebuttal_highdim.py::rerun_super`), and
`results/audit/audit_fmhd_results.csv` (TabPFN, TabICL) from `rebuttal_fm_highdim.py` with
`fmhd_spawn.py` / `fmhd_collect.py`. `rebuttal_fm_highdim.py` imports `rebuttal_highdim.py` and
must stay next to it.

**TabPFN on the eight benchmarks.** Block `tabpfn8`: `results/rebuttal/tabpfn_results.csv` from
`modal run experiments/modal/rebuttal_v4_port.py --part tab`, next to the FT-T / RB columns of
`audit_ystd_results.csv`.

## Section 4, Figure 2 and Appendix B — Monte Carlo

`paper_code/simulation_attention_regression.py` is the Colab cell that ran it (standalone; set
the `USER CONFIG` block at the top). As saved in the notebooks, cell 1 of
`paper_code/notebooks/AttReg_simul1.ipynb` and of `AttReg_simul2.ipynb` each ran it with
`N_REPEATS = 5`, `USE_ATTENTION_M5 = True` and RF / GBM on; the paper pools the two runs (its 10
replications). The MLP line came from a separate run with only `USE_OLS` and `USE_MLP` on (a
5-repeat version is cell 2 of `Attention_paper_simuls.ipynb`; the exact 10-repeat run was not
saved). An independent re-implementation of the whole grid per the appendix spec is
`results/audit/audit_mc_results.csv` (`audit_probes.py`, `run_mc_cell(dgp, N, snr)`, tag `mc`:
means and standard deviations of OLS, RF, GBM, MLP and Att. Reg over 10 replications per cell).

## Experiments run but not reported in tables

Learning curves (`results/rebuttal/lcurve_results.csv`), the whole-block depth sweep
(`depth_results.csv`), the sublayer factorial and no-FFN variants (`audit_sub`, `audit_noffn`),
residual / LayerNorm ablations (`audit_resln`, `audit_lnfix`), mixer / attention compositions
(`audit_mixattn`, `audit_fttnoffn`) and the Friedman controls (`audit_fr`, `audit_fr2`,
`audit_fr3`, `audit_frall`): see `experiments/modal/README.md` and `results/README.md`.
