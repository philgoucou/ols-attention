# paper_code/ — the original Colab pipeline of the submitted paper

These files are cells of the Google Colab notebooks in `notebooks/`, extracted verbatim; a short
header comment at the top of each says which notebook and cell. They are the record of how the
submitted numbers were produced. They are not the pipeline behind the revised paper's tables
(that is `experiments/`; see "Provenance of the submitted numbers" in the top-level README).
They run top to bottom as Colab cells and fetch the datasets from sklearn / UCI / OpenML themselves.

## Which file produced which submitted column

| Submitted number | File | Notebook cell with the saved output |
|---|---|---|
| Table 1: OLS, RF, MLP, Att. Reg | `real_data_benchmark_v3.py`, with the Attention Regression estimator defined in `simulation_attention_regression.py` (run first in the same session) | `Attention_paper_simuls.ipynb`, cell 1 |
| Table 1: FT-Transformer, Reg. Block | `real_data_benchmark_v4.py` (originally `colab_full.py`; also expects the simulation cell's definitions) | `Candidate2.ipynb`, cell 4 (the run whose rows were pasted is cell 3) |
| Sec. 4 / Fig. 2 / App. B: OLS, RF, GBM, Att. Reg | `simulation_attention_regression.py` with `N_REPEATS = 5`, `USE_ATTENTION_M5 = True`, RF and GBM on, run twice and pooled (the paper's 10 replications) | `AttReg_simul1.ipynb` and `AttReg_simul2.ipynb`, cell 1 of each |
| Sec. 4 / App. B: MLP | `simulation_attention_regression.py` with only OLS and the MLP on; the exact 10-repeat run was not saved | `Attention_paper_simuls.ipynb`, cell 2 (a 5-repeat run) |

The v4 run also printed an Attention Regression column that differs from v3's; the submitted
table used the v3 values (details in the top-level README).

## Files

| File | Role |
|---|---|
| `simulation_attention_regression.py` | defines the Attention Regression estimator (multi-head, Cholesky-parameterized embeddings, LBFGS with early stopping, ridge-precision warm start) and runs the Monte Carlo: six DGPs (linear, Friedman 1–3, rotated sine, soft radial) × N ∈ {500, 1000, 2500, 5000} × SNR ∈ {0.5, 1, 2, 3}, against OLS / RF / GBM / MLP. The `USER CONFIG` block at the top is the notebook's last saved state, not necessarily the paper's run (see the header comment) |
| `real_data_benchmark_v4.py` | the Table 1 pipeline of the submission: eight datasets, N capped at 5000, five random 80/20 splits, features standardized on the training split, raw targets; OLS, RF, MLP, FT-Transformer, Attention Regression, Regression Block (the "polynomial cross-feature Transformer": degree-2 cross-features → PCA → regression mixer with a ridge warm start → residual + FFN) |
| `real_data_benchmark_v3.py` | its precursor, without the FT-Transformer and the Regression Block |
| `colab_benchmark.py` | early exploration: Poly-PCA + OLS against OLS and RF (`Candidate2.ipynb`, cell 0) |
| `ablation_study.py`, `thorough_ablation.py`, `full_benchmark.py`, `multi_dataset_pca.py` | early CPU explorations of the block: ridge-only baselines, no warm start, no FFN / LayerNorm, pooled vs token-level PCA; PCA(200) + OLS against a 2-block attention net and the FT-Transformer |

## Notebooks

Unmodified, with their outputs, as pulled from Google Drive. They are the provenance record: the
`\\` LaTeX rows printed at the end of the benchmark cells are what was pasted into the submitted
Table 1, and the per-condition rows of the simulation cells are the paper's Monte Carlo.

| Notebook | Cells |
|---|---|
| `Attention_paper_simuls.ipynb` | 0: the Monte Carlo script (= `simulation_attention_regression.py`); 1: the v3 real-data benchmark (= `real_data_benchmark_v3.py`), the OLS / RF / MLP / Att. Reg rows of Table 1; 2: the Monte Carlo again, the 5-repeat OLS + MLP run |
| `AttReg_simul1.ipynb`, `AttReg_simul2.ipynb` | cell 1 of each: the Monte Carlo run (5 repeats, five heads, RF and GBM on) whose union is the paper's Fig. 2 / App. B |
| `Candidate2.ipynb` | 0: `colab_benchmark.py`; 1–4: successive versions of the v4 benchmark, cell 4 being `real_data_benchmark_v4.py` (the FT-T / Reg. Block rows of Table 1; the run is cell 3); 5–6: the first, single-run Colab versions of the rebuttal experiments, superseded by `experiments/modal/` |
| `AttReg.ipynb` | development history of the Attention Regression benchmark script: six successive versions of the simulation cell, four with outputs |
