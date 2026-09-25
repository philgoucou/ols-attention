# Ordinary Least Squares as an Attention Mechanism — code and results

Code, per-seed results and table generators for

> P. Goulet Coulombe, *Ordinary Least Squares as an Attention Mechanism*, NeurIPS 2026.
> OpenReview: https://openreview.net/forum?id=MRyrIW18CW

Every number in the paper's tables can be traced to a CSV in `results/` and to the script that
produced it. `tables/gen_tables.py` regenerates every LaTeX table body of the revision from the CSVs.

## Layout

```
paper_code/          the original pipeline of the submitted paper (Colab-era)
  simulation_attention_regression.py   Monte Carlo: Attention Regression (LBFGS + Cholesky heads),
                                       OLS / RF / GBM / MLP, six DGPs, 4 N x 4 SNR   -> Fig. 2, App. B
  real_data_benchmark_v4.py            submitted Table 1 (OLS, RF, MLP, FT-Transformer, Att. Reg, Reg. Block)
  real_data_benchmark_v3.py            precursor of v4 (kept for the record)
  ablation_study.py, thorough_ablation.py, full_benchmark.py, multi_dataset_pca.py   early explorations
  notebooks/                           the Colab notebooks these were pasted from, with saved outputs
experiments/modal/   the rebuttal / revision experiments (Modal fan-out, ~1,500 GPU runs)
experiments/audit/   text-vs-code audit probes (protocol correction, reproduction of Table 1, sensitivity)
results/rebuttal/    per-seed CSVs of the first rebuttal wave (July 2026)
results/audit/       per-seed CSVs of the audit and corrected-protocol runs
tables/              gen_tables.py -> tables_generated.tex (all table bodies of the revision)
```

## Where each table comes from

| Paper | Script | CSV |
|---|---|---|
| Table 1, corrected protocol: FT-T and Reg. Block | `experiments/audit/audit_probes.py` (`run_ystd`) | `results/audit/audit_ystd_results.csv` |
| Table 1: MLP (corrected) | `experiments/modal/audit_probes_mlpystd.py` | `results/audit/audit_mlpystd_results.csv` |
| Table 1: OLS, RF (scale-invariant, verified) | `experiments/audit/audit_probes.py` (`run_cpu`) | `results/audit/audit_cpu_results.csv` |
| Table 1: Att. Reg (scale-equivariance check) | `experiments/audit/audit_probes2.py` | `results/audit/audit_attreg{,_std}_results.csv` |
| Table 1, submitted convention (App. D) | `paper_code/real_data_benchmark_v4.py`; Modal port `experiments/modal/rebuttal_modal.py` | `results/rebuttal/ablation_grid_results.csv` (A0/RB columns) |
| Paired SEs, sign test (App. D) | `tables/gen_tables.py` on the y-std CSV | `results/audit/audit_ystd_results.csv` |
| Ablations A0–A5, RB (App. E) | `experiments/modal/rebuttal_ystd_grids.py` | `results/audit/audit_ablystd_results.csv` |
| A6, three stacked blocks | `experiments/modal/rebuttal_faithful.py` (variant `cur_L3`) | `results/audit/audit_faithful_results.csv` |
| A7, widened block (and its cold-start twin) | `experiments/modal/rebuttal_a6.py`, `rebuttal_a7nowarm.py` | `results/audit/audit_a6ystd_results.csv`, `audit_a7nowarm_results.csv` |
| Readout vs warm start (App. E) | `experiments/modal/rebuttal_ystd_grids.py` (warm-start family); raw-target run in `rebuttal_modal.py` | `results/audit/audit_wsystd_results.csv`, `results/rebuttal/warmstart_results.csv` |
| Trimmed configuration and PCA ladder (App. E) | `experiments/modal/rebuttal_sublayer.py` (`run_pca_sweep`) | `results/audit/audit_pca_results.csv` |
| Trimmed configuration at full N / higher P / classification | `experiments/modal/rebuttal_aggressive.py` (reconstruction, see note) | `results/audit/audit_agg_results.csv` |
| Full sample size (App. F) | `experiments/modal/rebuttal_uncapped_ystd.py`; TabPFN/TabICL: `rebuttal_fm_uncapped.py` | `results/audit/audit_uy_results.csv`, `audit_fm_results.csv`, `audit_fm_tabicl_uncapped.csv` |
| Classification (App. F) | `experiments/modal/rebuttal_classify.py`; TabICL: `rebuttal_tabicl.py` | `results/rebuttal/classify_results.csv`, `tabicl_results.csv` |
| Higher dimensions (App. F) | `experiments/modal/rebuttal_highdim.py`; TabPFN/TabICL: `rebuttal_fm_highdim.py` | `results/rebuttal/highdim_results.csv`, `results/audit/audit_fmhd_results.csv` |
| TabPFN on the eight benchmarks (App. F) | `experiments/modal/rebuttal_modal.py` (TabPFN family) | `results/rebuttal/tabpfn_results.csv` |
| Monte Carlo (Sec. 4, Fig. 2, App. B) | `paper_code/simulation_attention_regression.py` | notebook outputs in `paper_code/notebooks/` |

Other rebuttal experiments not reported in the paper (learning curves, depth of the paper-faithful
variant, residual/LayerNorm ablations, mixer/attention compositions, Friedman controls) are in
`experiments/modal/` with their CSVs under `results/`.

## Running the Modal experiments

Each `rebuttal_*.py` is a [Modal](https://modal.com) app. The pattern used throughout:

```bash
modal run  experiments/modal/rebuttal_ystd_grids.py::canary   # one cheap cell, checks code + data cache
modal deploy experiments/modal/rebuttal_ystd_grids.py         # deployed app, jobs survive the client
python experiments/modal/<name>_spawn.py                      # spawn the grid, writes *_call_ids.json
python experiments/modal/<name>_collect.py                    # resilient collector -> audit_<tag>_results.csv
```

Datasets are fetched from UCI / OpenML by the scripts' `fetch_and_cache` functions into a Modal volume
(`neurips-31482-cache`). Seeds, splits (5 random 80/20), the N = 5000 cap and the standardization of
features and targets on the training split are fixed inside the scripts. GPUs used: T4 / A10G, and
A100-80GB for P >= 40 (the degree-2 cross-feature map is O(P^2)).

`experiments/modal/rebuttal_aggressive.py` is a reconstruction: the original harness was lost with a
temporary directory after its CSV had been saved; the reconstruction is verified against that CSV
cell by cell (see its header).

## The original Colab pipeline

`paper_code/simulation_attention_regression.py` defines the Attention Regression estimator (multi-head,
Cholesky-parameterized embeddings, LBFGS with early stopping, ridge-precision warm start) and the Monte
Carlo. `paper_code/real_data_benchmark_v4.py` is the Table 1 pipeline of the submitted paper and expects
those definitions to be loaded in the same session (it was written as a second Colab cell). The Modal
port in `experiments/modal/rebuttal_modal.py` reproduces its OLS / RF / FT-Transformer / Regression Block
columns within +-0.02, and is the pipeline used for every number in the revised paper.

## License

MIT.
