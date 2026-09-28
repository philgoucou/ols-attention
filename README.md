# Ordinary Least Squares as an Attention Mechanism — code and results

Code, per-seed results and table generators for

> Philippe Goulet Coulombe, *Ordinary Least Squares as an Attention Mechanism*, NeurIPS 2026.
> OpenReview: https://openreview.net/forum?id=MRyrIW18CW

The paper reads the OLS fitted-value map as an attention mechanism and builds two estimators on
it: *Attention Regression* (multi-head, Cholesky-parameterized embeddings fitted by LBFGS),
studied by Monte Carlo against OLS, random forests, boosting and an MLP, and the *Regression
Block*, a Transformer block whose softmax attention is replaced by a regression mixer on degree-2
cross-features (PCA, ridge warm start), benchmarked on eight regression datasets against OLS, RF,
an MLP, the FT-Transformer and tabular foundation models. This repository holds the original
Colab pipeline of the submission, the ~1,500-run Modal pipeline of the rebuttal and revision, the
per-seed CSV of every number in the paper, and the script that regenerates every LaTeX table body
from those CSVs. Nothing needs a GPU except re-running the experiments themselves.

```bibtex
@inproceedings{gouletcoulombe2026ols,
  title     = {Ordinary Least Squares as an Attention Mechanism},
  author    = {Goulet Coulombe, Philippe},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=MRyrIW18CW}
}
```

(`CITATION.cff` carries the same entry.)

## Layout

```
.
├── README.md                this file
├── CITATION.cff
├── LICENSE                  MIT
├── requirements.txt
├── docs/
│   └── REPRODUCE.md         table by table: the CSV, the gen_tables.py block, the command that produced it
├── paper_code/              the original Colab pipeline of the submitted paper       (README inside)
│   ├── simulation_attention_regression.py   Monte Carlo: Attention Regression vs OLS / RF / GBM / MLP -> Fig. 2, App. B
│   ├── real_data_benchmark_v4.py            submitted Table 1 (OLS, RF, MLP, FT-Transformer, Att. Reg, Reg. Block)
│   ├── real_data_benchmark_v3.py            precursor of v4: the OLS / RF / MLP / Att. Reg columns of the submission
│   ├── colab_benchmark.py, ablation_study.py, thorough_ablation.py, full_benchmark.py, multi_dataset_pca.py   early explorations
│   └── notebooks/                           the five Colab notebooks with their saved outputs: the provenance record
├── experiments/             everything run after submission, on Modal                (README inside)
│   ├── modal/               the rebuttal / revision apps, documented family by family (README inside)
│   │   ├── collectors/      *_spawn.py / *_collect.py helpers of the deployed apps     (README inside)
│   │   └── utils/           verification, report and draft generators, diagnostics     (README inside)
│   └── audit/               text-vs-code audit probes: Table 1 reproduced column by column, Monte Carlo per spec, forensics
│       └── collectors/
├── results/                 every per-seed CSV                                         (README inside: one line per file)
│   ├── rebuttal/            first rebuttal wave (July 2026), the submission's raw-target protocol
│   └── audit/               audit and corrected-protocol runs (September 2026)
└── tables/
    ├── gen_tables.py        regenerates every LaTeX table body of the paper from results/
    └── tables_generated.tex its output, as committed
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Regenerate every table body of the paper from the saved CSVs (no GPU, seconds).
python3 tables/gen_tables.py          # -> tables/tables_generated.tex, one block per '%%%% ===== name =====' marker

# 2. Re-run a Modal experiment (needs `modal token new` and GPU credit): canary -> deploy -> spawn -> collect.
modal run    experiments/modal/rebuttal_fm_highdim.py::canary   # one cheap cell per model: checks code, image, data cache
modal deploy experiments/modal/rebuttal_fm_highdim.py           # deployed app: the jobs survive the client
cd experiments/modal/collectors
python fmhd_spawn.py                                            # spawns the grid, saves the call ids to fmhd_call_ids.json
python fmhd_collect.py                                          # resilient collector, rerunnable -> audit_fmhd_results.csv
```

Apps with a `main` entrypoint (`rebuttal_v4_port.py`, `rebuttal_depth.py`, `rebuttal_warmstart_lcurve.py`,
`rebuttal_classify.py`, `rebuttal_tabicl.py`, `rebuttal_highdim.py`) are run with `modal run <app>.py`
and write their CSV in the current directory. `docs/REPRODUCE.md` gives the commands for every
table and figure; `experiments/modal/README.md` documents each experiment family.

## Where each table comes from

Paper references are to the revised paper. `tables/gen_tables.py` reads these CSVs and drops rows
with a non-empty `error` column before averaging.

| Paper | Script | CSV |
|---|---|---|
| Table 1, corrected protocol: FT-T and Reg. Block | `experiments/audit/audit_probes.py` (`run_ystd`) | `results/audit/audit_ystd_results.csv` |
| Table 1: MLP (corrected) | `experiments/audit/audit_probes_mlp_ystd.py` (formerly `experiments/modal/audit_probes_mlpystd.py`) | `results/audit/audit_mlpystd_results.csv` |
| Table 1: OLS, RF (scale-invariant, verified) | `experiments/audit/audit_probes.py` (`run_cpu`) | `results/audit/audit_cpu_results.csv` |
| Table 1: Att. Reg (scale-equivariance check) | `experiments/audit/audit_probes2.py` | `results/audit/audit_attreg{,_std}_results.csv` |
| Table 1, submitted convention (App. D) | `paper_code/real_data_benchmark_v4.py`; Modal port `experiments/modal/rebuttal_v4_port.py` (formerly `rebuttal_modal.py`) | `results/rebuttal/ablation_grid_results.csv` (A0/RB columns) |
| Paired SEs, sign test (App. D) | `tables/gen_tables.py` on the y-std CSV | `results/audit/audit_ystd_results.csv` |
| Ablations A0–A5, RB (App. E) | `experiments/modal/rebuttal_ablation_warmstart.py` (formerly `rebuttal_ystd_grids.py`) | `results/audit/audit_ablystd_results.csv` |
| A6, three stacked blocks | `experiments/modal/rebuttal_faithful.py` (variant `cur_L3`) | `results/audit/audit_faithful_results.csv` |
| A7, widened block (and its cold-start twin) | `experiments/modal/rebuttal_a7_widened.py` (formerly `rebuttal_a6.py`), `rebuttal_a7_widened_nowarm.py` (formerly `rebuttal_a7nowarm.py`) | `results/audit/audit_a6ystd_results.csv`, `audit_a7nowarm_results.csv` |
| Readout vs warm start (App. E) | `experiments/modal/rebuttal_ablation_warmstart.py` (warm-start family); raw-target run in `rebuttal_warmstart_lcurve.py` (formerly `rebuttal_extras.py`) | `results/audit/audit_wsystd_results.csv`, `results/rebuttal/warmstart_results.csv` |
| Trimmed configuration and PCA ladder (App. E) | `experiments/modal/rebuttal_sublayer.py` (`run_pca_sweep`) | `results/audit/audit_pca_results.csv` |
| Trimmed configuration at full N / higher P / classification | `experiments/modal/rebuttal_trimmed.py` (formerly `rebuttal_aggressive.py`; reconstruction, see note) | `results/audit/audit_agg_results.csv` |
| Full sample size (App. F) | `experiments/modal/rebuttal_uncapped_ystd.py`; TabPFN/TabICL: `rebuttal_fm_uncapped.py` | `results/audit/audit_uy_results.csv`, `audit_fm_results.csv`, `audit_fm_tabicl_uncapped.csv` |
| Classification (App. F) | `experiments/modal/rebuttal_classify.py`; TabICL: `rebuttal_tabicl.py` | `results/rebuttal/classify_results.csv`, `tabicl_results.csv` |
| Higher dimensions (App. F) | `experiments/modal/rebuttal_highdim.py`; TabPFN/TabICL: `rebuttal_fm_highdim.py` | `results/rebuttal/highdim_results.csv`, `results/audit/audit_fmhd_results.csv` |
| TabPFN on the eight benchmarks (App. F) | `experiments/modal/rebuttal_v4_port.py` (TabPFN family) | `results/rebuttal/tabpfn_results.csv` |
| Monte Carlo (Sec. 4, Fig. 2, App. B) | `paper_code/simulation_attention_regression.py` | notebook outputs in `paper_code/notebooks/` |

Other rebuttal experiments not reported in the paper (learning curves, depth of the paper-faithful
variant, sublayer factorial, residual/LayerNorm ablations, mixer/attention compositions, Friedman
controls) are in `experiments/modal/` with their CSVs under `results/`; `results/README.md` lists
every file with its columns.

## Checking the paper's numbers without re-running anything

```bash
python3 tables/verify_paper_numbers.py
```

recomputes every table of the paper from the CSVs in `results/`, compares each cell with
`tables/paper_snapshot.tex` (the table bodies exactly as printed in the camera-ready), checks the
submitted-convention table against the rows the original Colab notebooks printed, and recomputes
the 60 statistics quoted in the text (paired differences, standard errors, p-values, average ranks,
win counts). Result on 28 Sept 2026: **488 table cells and 60 quoted statistics, all match**
(exit code 0). Difference columns in the paper are differences of the printed 3-decimal values, so
that a reader subtracting two printed numbers recovers the printed difference; the script applies
the same convention.

## Running the Modal experiments

Each `rebuttal_*.py` is a [Modal](https://modal.com) app. The pattern used throughout:

```bash
modal run    experiments/modal/<app>.py::canary      # where one exists (rebuttal_fm_highdim.py, rebuttal_trimmed.py): one cheap cell
modal deploy experiments/modal/<app>.py              # deployed app, jobs survive the client
python experiments/modal/collectors/<tag>_spawn.py   # spawn the grid, writes <tag>_call_ids.json (run from collectors/)
python experiments/modal/collectors/<tag>_collect.py # resilient collector -> audit_<tag>_results.csv, next to itself
```

Datasets are fetched from UCI / OpenML by the scripts' `fetch_and_cache` functions into a Modal volume
(`neurips-31482-cache`). Seeds, splits (5 random 80/20), the N = 5000 cap and the standardization of
features and targets on the training split are fixed inside the scripts. GPUs used: T4 / A10G, and
A100-80GB for P >= 40 (the degree-2 cross-feature map is O(P^2)).

`experiments/modal/rebuttal_trimmed.py` (formerly `rebuttal_aggressive.py`) is a reconstruction: the
original harness was lost with a temporary directory after its CSV had been saved. It was rebuilt from
`rebuttal_sublayer.py` and `rebuttal_classify.py` and re-run against the CSV's per-seed cells
(`modal run rebuttal_trimmed.py::canary`): parameter counts identical on every setting; scores within
GPU run-to-run noise (std/California n_mix=3 seed 0: 0.7258 vs 0.7295; uncapped/Kin8nm: 0.9254 vs 0.9219;
highdim/CPU_act: 0.9665 vs 0.9693; classify/Wine: 0.9444 vs 0.9444).

## Provenance of the submitted numbers (verified 28 Sept 2026 from the notebooks' saved outputs)

The submitted Table 1 was assembled from two Colab runs whose printed LaTeX rows are still in the
notebooks under `paper_code/notebooks/`:

| Column(s) of the submitted Table 1 | Produced by | Saved output |
|---|---|---|
| OLS, RF, MLP, Att. Reg | `real_data_benchmark_v3.py` (cell 1 of `Attention_paper_simuls.ipynb`), with the Attention Regression estimator of `simulation_attention_regression.py` | the `\\` rows printed at the end of that cell |
| FT-Transformer, Reg. Block | `real_data_benchmark_v4.py` (cell 4 of `Candidate2.ipynb`; the run is cell 3) | the `\\` rows printed at the end of that cell |

All 48 cells (6 columns x 8 datasets) match the submitted table to the printed precision. Note that the
v4 run also printed an Attention Regression column (e.g. California 0.715, Concrete 0.171, Airfoil below
zero) that differs from the v3 run's (0.738, 0.837, 0.757); the submitted table used the v3 values. The
revised paper keeps that column: it is invariant to the target-scaling correction, which is what the
revision changed.

The Monte Carlo of Section 4 / Appendix B is the union of `AttReg_simul1.ipynb` and `AttReg_simul2.ipynb`
(cell 1 of each: 96 conditions x 5 repeats, five attention heads, RF and GBM on), i.e. the paper's 10
replications. Pooled over the two, the saved per-condition rows give OLS 0.279, RF 0.454, GBM 0.418,
Attention Regression 0.476 (paper: 0.28, 0.45, 0.42, 0.48; Linear DGP OLS 0.557, paper 0.56), and the
same per-DGP ordering the paper describes. The MLP column came from a separate run of
`simulation_attention_regression.py` with only OLS and the MLP switched on; the saved 5-repeat run in
cell 2 of `Attention_paper_simuls.ipynb` gives an overall MLP R^2 of 0.534 against 0.52 in the paper, so
the exact run behind the paper's MLP figure was not saved.

## The original Colab pipeline

`paper_code/simulation_attention_regression.py` defines the Attention Regression estimator (multi-head,
Cholesky-parameterized embeddings, LBFGS with early stopping, ridge-precision warm start) and the Monte
Carlo. `paper_code/real_data_benchmark_v4.py` is the Table 1 pipeline of the submitted paper and expects
those definitions to be loaded in the same session (it was written as a second Colab cell). The Modal
port in `experiments/modal/rebuttal_v4_port.py` (formerly `rebuttal_modal.py`) reproduces its OLS / RF /
FT-Transformer / Regression Block columns within +-0.02, and is the pipeline used for every number in
the revised paper.

## License

MIT.
