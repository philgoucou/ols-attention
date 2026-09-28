# experiments/modal/ — the rebuttal and revision experiments

Each `rebuttal_*.py` is a self-contained [Modal](https://modal.com) app: pinned image, the
dataset cache volume, one remote function per grid cell, every constant at the top of the file
(the conventions shared by all of them are in [`../README.md`](../README.md)). Two ways of running them:

* **`modal run` apps** have a `main` local entrypoint that fans the grid out with `starmap`,
  waits, and writes the CSV in the current directory: `rebuttal_v4_port.py`, `rebuttal_depth.py`,
  `rebuttal_warmstart_lcurve.py`, `rebuttal_classify.py`, `rebuttal_tabicl.py`, `rebuttal_highdim.py`.
* **Deployed apps** (`modal deploy <app>.py`) are spawned with
  `modal.Function.from_name(app, fn).spawn(...)` and harvested later by a resilient collector
  that writes `audit_<tag>_results.csv`; the jobs survive the client. The helpers and the
  pattern are in [`collectors/`](collectors/README.md). Every app not listed above.

Two scripts import a sibling and must stay in this folder next to it: `rebuttal_fm_highdim.py`
imports `rebuttal_highdim.py`, and `rebuttal_friedman2.py` imports `rebuttal_friedman.py` (both
through `import` plus `.add_local_python_source(...)` in the Modal image).

Renamed for this release — the old name is in each file's docstring and the Modal app names are
unchanged: `rebuttal_a6.py` → `rebuttal_a7_widened.py`, `rebuttal_a7nowarm.py` →
`rebuttal_a7_widened_nowarm.py`, `rebuttal_ystd_grids.py` → `rebuttal_ablation_warmstart.py`,
`rebuttal_modal.py` → `rebuttal_v4_port.py`, `rebuttal_extras.py` → `rebuttal_warmstart_lcurve.py`,
`rebuttal_aggressive.py` → `rebuttal_trimmed.py`; `audit_probes_mlpystd.py` moved to
`../audit/audit_probes_mlp_ystd.py`; `spawn_super.py` / `collect_super.py` → `collectors/super_spawn.py`
/ `super_collect.py`.

GPUs: T4 unless stated; A10G for the Regression Block at P ≥ 20 and at full N (the degree-2
cross-feature map is O(P²) wide); A100-80GB for the foundation models and for Superconductivity
(P = 81). Paper references are to the revised paper: Table 1, App. D (protocol), App. E
(ablations), App. F (scope). "Not in the paper" means the run informed the rebuttal or the text,
but no table uses its numbers. Row counts read `datasets × configurations × 5 seeds`.

## Families

### 1. Ablation grid, A0–A5 and RB (App. E)

| | |
|---|---|
| App | `rebuttal_ablation_warmstart.py` (formerly `rebuttal_ystd_grids.py`), app `neurips-31482-ystd-grids`, `run_abl_ystd(dataset, config, seed_idx)` |
| Helpers | deployed; spawned by hand with tag `ablystd`, harvested with the collector template |
| CSV | `results/audit/audit_ablystd_results.csv` — 8 × {A0, A1, A2, A3, A4, A5, RB} × 5 = 280 rows, corrected protocol |
| Paper | App. E grid (`abl_grid`), summary (`abl_summary`), paired tests (`abl_tests`, `abl_wins`) |
| First wave | `rebuttal_v4_port.py --part abl` (formerly `rebuttal_modal.py`; app `neurips-31482-rebuttal`; the Modal port of `paper_code/real_data_benchmark_v4.py`, raw targets) → `results/rebuttal/ablation_grid_results.csv`, the same 280 cells; its A0 and RB columns are the submitted-convention table of App. D |

Configurations: A0 the FT-Transformer (3 blocks, d_model 64; 101,697 parameters at P = 8); A1 the
FT-T on degree-2 polynomial inputs; A2 the Regression Block without the polynomial expansion;
A3 without the PCA; A4 softmax attention in the Regression Block skeleton (1 block); A5 the FT-T
shrunk to the Regression Block's budget (3 blocks, d_model 36); RB the Regression Block (1 block,
d_model 64, 200 PCA components; 30,401 parameters).

### 2. Readout vs warm start (App. E)

| | |
|---|---|
| App | `rebuttal_ablation_warmstart.py`, `run_ws_ystd(dataset, config, seed_idx)` for `softmax`, `rb_nowarm`, `rb_warm` |
| Helpers | deployed; tag `wsystd`, collector template |
| CSV | `results/audit/audit_wsystd_results.csv` — 8 × 3 × 5 = 120 rows |
| Paper | App. E readout-vs-warm-start table (`ws`, `ws_tests`); `ws_raw` measures the shift from the raw-target run |
| First wave | `rebuttal_warmstart_lcurve.py` (formerly `rebuttal_extras.py`; app `neurips-31482-extras`; `modal run`) → `results/rebuttal/warmstart_results.csv` (same grid, raw targets) and `results/rebuttal/lcurve_results.csv` (learning curves of the FT-T and the RB on California / Kin8nm / Protein at N_train ∈ {500, 1000, 2500, 5000, 10000, full}; not in the paper) |

The decomposition, same seeds throughout: A4's softmax mixer → the regression readout with
random initialization (`rb_nowarm`) → plus the Ridge warm start (`rb_warm`, i.e. the RB).

### 3. Widened block, A7, and its cold-start twin (App. E)

| | |
|---|---|
| Apps | `rebuttal_a7_widened.py` (formerly `rebuttal_a6.py`; app `neurips-31482-a6ystd`, `run_a6(dataset, seed_idx)`) and `rebuttal_a7_widened_nowarm.py` (formerly `rebuttal_a7nowarm.py`; app `neurips-31482-a7nowarm`, `run_a6`) |
| Helpers | tag `a6ystd` → `collectors/a6ystd_collect.py`; tag `a7nowarm` → `collectors/a7nowarm_collect.py` |
| CSVs | `results/audit/audit_a6ystd_results.csv` and `audit_a7nowarm_results.csv` — 8 × 5 = 40 rows each. Their `config` field says `A6`: this row's label before the App. E renumbering |
| Paper | row A7 of the App. E grid and summary; the cold-start twin is discussed in the text |
| First wave | app `neurips-31482-a6` (raw targets; `collectors/a6_spawn.py`, `a6_collect.py`) → `results/rebuttal/a6_results.csv`. Only the corrected-protocol source survives |

The block's (d_model, n_components) are searched, with FFN width 2·d_model, to match the
FT-Transformer's budget at each dataset's P (d_model 128 and 260 components at P = 8): the
symmetric counterpart of A5. The twin computes the Ridge warm start and then discards it.

### 4. Depth and sublayers

| Script | Run and output |
|---|---|
| `rebuttal_depth.py` (app `neurips-31482-depth`, `modal run`) | the Regression Block stacked to L ∈ {1, 2, 3} against the FT-Transformer with 1–3 blocks, raw targets → `results/rebuttal/depth_results.csv` (8 × 6 × 5 = 240). Not in the paper: superseded by the corrected-protocol runs below |
| `rebuttal_sublayer.py` (app `neurips-31482-sublayer`), `run_sublayer(dataset, n_mix, n_ffn, seed_idx)` | the block decoupled into n_mix regression-mixer sublayers followed by n_ffn FFN sublayers, only the first mixer warm-started. The 2 × 2 over {1, 3} × {1, 3} → `results/audit/audit_sub_results.csv` (160; `collectors/sublayer_spawn.py`, `sublayer_collect.py`); the no-FFN variants `mix1_ffn0`, `mix2_ffn0`, `mix3_ffn0` (tag `noffn`) → `results/audit/audit_noffn_results.csv` (160, with the `mix1_ffn1` reference rows copied from `audit_sub`). Not in the paper |
| `rebuttal_sublayer.py`, `run_pca_sweep(dataset, n_mix, n_ffn, ncomp, seed_idx)` | the PCA ladder, {1 mixer + FFN, 2 mixers without FFN} × {25, 50, 100, 200} components (tag `pca`) → `results/audit/audit_pca_results.csv` (320). **App. E, trimmed configuration and ladder** (`trim`, `trim_tests`, `ladder`); the trimmed configuration is `mix2_ffn0_pca50` |

### 5. Faithful-variant ladder (App. E, row A6)

| | |
|---|---|
| App | `rebuttal_faithful.py`, app `neurips-31482-faithful`, `run_faithful(dataset, variant, seed_idx)` |
| Helpers | `collectors/faithful_spawn.py`, `faithful_collect.py` |
| CSV | `results/audit/audit_faithful_results.csv` — 8 × 8 variants × 5 = 320 rows |
| Paper | variant `cur_L3` (the shipped block stacked to 3) is row A6 of the App. E grid; the rest of the ladder is discussed in the text |

Variants as (tokenizer / readout / activation / blocks): `cur_L1` shared / mean-pool / ReLU / 1
(the block of Table 1); `gelu_L1`, `cls_L1`, `pertok_L1` (one change each); `faith_L1`, `faith_L2`,
`faith_L3` (per-feature tokenizer / [CLS] / GELU, 1–3 blocks); `cur_L3`.

### 6. Trimmed configuration at scale (App. E)

| | |
|---|---|
| App | `rebuttal_trimmed.py` (formerly `rebuttal_aggressive.py`; app `neurips-31482-aggressive`), `run_trim(setting, dataset, n_mix, ncomp, seed_idx)`; `modal run rebuttal_trimmed.py::canary` re-runs one cell per setting against the CSV |
| Helpers | deployed; tag `agg`, collector template |
| CSV | `results/audit/audit_agg_results.csv` — settings `std` (the eight datasets), `uncapped` (California, Kin8nm, Protein at full N), `highdim` (CPU_act, Bank32nh, Ailerons, Pol, Superconductivity), `classify` (the six classification datasets); n_mix ∈ {2, 3}, 50 components; 135 rows, of which the 10 Superconductivity cells failed (CUDA out of memory) |
| Paper | App. E, the trimmed configuration at full N and at higher P (`trim_scope`, n_mix = 2) |

Reconstruction notice: the harness that produced this CSV was lost with a temporary directory
after the CSV had been saved. The file was rebuilt from `rebuttal_sublayer.py` and
`rebuttal_classify.py` and verified cell by cell against the CSV (its docstring and the top-level
README give the numbers).

### 7. Classification (App. F)

| | |
|---|---|
| Apps | `rebuttal_classify.py` (app `neurips-31482-classify`, `modal run`): LogReg, RF, FT-Transformer, the Regression Block with a cross-entropy head and a multinomial-logistic warm start, TabPFN; `rebuttal_tabicl.py` (app `neurips-31482-tabicl`, `modal run`): TabICL on the same splits |
| CSVs | `results/rebuttal/classify_results.csv` (6 × 5 × 5 = 150; accuracy and ROC-AUC) and `results/rebuttal/tabicl_results.csv` (30) |
| Paper | App. F classification table (`clf`, `clf_rank`) |

Datasets: BreastCancer (P = 30), Diabetes (8), Phoneme (5), Wine (3 classes, P = 13), Vehicle
(4, 18), Segment (7, 19); stratified cap and splits. `utils/classify.py` is the standalone module
that `rebuttal_classify.py` inlines. The trimmed configuration on this suite is in `audit_agg`
(setting `classify`).

### 8. Full sample size (App. F)

| | |
|---|---|
| Apps | `rebuttal_uncapped_ystd.py` (app `neurips-31482-uncapped-ystd`; `run_sk_ystd`, `run_nn_ystd(dataset, model, seed_idx)`; A10G): OLS, RF, FT-T, RB at full N under the corrected protocol. `rebuttal_fm_uncapped.py` (app `neurips-31482-fm-uncapped`; `run_fm_uncapped(dataset, model, seed_idx)`; A100-80GB; `modal run rebuttal_fm_uncapped.py::show_probe` first reports the TabICL API and TabPFN's row limits) |
| Helpers | tag `uy` → `collectors/uy_collect.py`; tag `fm` → `collectors/fm_collect.py` |
| CSVs | `results/audit/audit_uy_results.csv` (3 × 4 × 5 = 60; n_train 16,512 / 6,553 / 36,584); `results/audit/audit_fm_results.csv` (TabPFN, 15 valid rows — its 15 TabICL rows failed on a device-string bug); `results/audit/audit_fm_tabicl_uncapped.csv` (TabICL re-run after the fix found with `utils/tabicl_diag.py`, 15 rows). Every row records `n_train_used == n_train_available`: no internal subsampling |
| Paper | App. F uncapped table (`uncapped`) |
| First wave | `rebuttal_v4_port.py --part unc` → `results/rebuttal/uncapped_results.csv` (raw targets; the Protein / RB seeds were re-run through `::patch_protein_rb` after a warm-start batching fix) |

### 9. Higher dimensions (App. F)

| | |
|---|---|
| Apps | `rebuttal_highdim.py` (app `neurips-31482-highdim`; `modal run`, plus `::rerun_rb` and `::rerun_super`; RB on A10G, `run_nn_big` on A100-80GB): OLS, RF, FT-T, RB on CPU_act (P = 21), Bank32nh (32), Ailerons (40), Pol (48), Superconductivity (81), and a Friedman-1 Monte Carlo at P = 20 and 50. `rebuttal_fm_highdim.py` (app `neurips-31482-fm-highdim`; imports `rebuttal_highdim`; `::canary`): TabPFN and TabICL on the same splits |
| Helpers | `collectors/super_spawn.py`, `super_collect.py` (the five Superconductivity RB seeds, spliced into the CSV); `collectors/fmhd_spawn.py`, `fmhd_collect.py` |
| CSVs | `results/rebuttal/highdim_results.csv` (7 × 4 × 5 = 140; Superconductivity / RB seed 3 timed out at 5,400 s, so that cell averages 4 seeds); `results/audit/audit_fmhd_results.csv` (7 × 2 × 5 = 70) |
| Paper | App. F higher-dimensions table (`highdim`), the five real datasets |

`rb_mem_settings(P)` reduces the batch size and the PCA-fit subsample for large P, the
cross-feature map being O(P²) wide: the scalability caveat the paper reports.

### 10. Foundation models (App. F)

TabPFN and TabICL appear in four places: `results/rebuttal/tabpfn_results.csv` (TabPFN on the
eight Table 1 datasets, `rebuttal_v4_port.py --part tab`; App. F table `tabpfn8`, next to the
corrected FT-T / RB columns of `audit_ystd_results.csv`), the classification suite (family 7),
the uncapped suite (family 8) and the higher-dimensional suite (family 9). `utils/tabicl_diag.py`
is the device diagnostic.

### 11. Friedman controls (not in the paper)

| | |
|---|---|
| Apps | `rebuttal_friedman.py` (app `neurips-31482-friedman`; `run_sk`, `run_nn(arm, P, model, seed_idx)`; A10G): SNR-calibrated Friedman-1 designs that separate dimensionality from feature irrelevance — arm `frac5` (5 informative features at any P), `orig50` (the Friedman-1 layout tiled so that half the features stay relevant), `all100` (all features relevant). `rebuttal_friedman2.py` (app `neurips-31482-friedman2`; imports `rebuttal_friedman`; `run_rb_ncomp(arm, P, ncomp, seed_idx)`): the Regression Block with 400 and 800 PCA components on arm `orig50` |
| Helpers | `collectors/friedman_spawn.py`, `friedman_collect.py` (tag `fr`); `friedman3_collect.py` (tag `fr3`); `friedman2_collect.py` (tag `fr2`) |
| CSVs | `results/audit/audit_fr_results.csv` (`frac5`, `orig50` × P ∈ {10, 20, 50} × 4 models × 5 = 120); `audit_fr3_results.csv` (`all100` × P ∈ {5, 10, 20, 50} × 4 × 5 = 80); `audit_frall_results.csv` (the two concatenated, 200); `audit_fr2_results.csv` (`orig50` × P ∈ {20, 50} × ncomp ∈ {400, 800} × 5 = 20) |

### 12. Residual / LayerNorm ablations (not in the paper)

| | |
|---|---|
| App | `rebuttal_resln.py` (app `neurips-31482-resln`): `run_resln(dataset, model, variant, seed_idx)` removes the post-norm scaffolding LN(x + f(x)) one piece at a time (`full`, `no_res`, `no_ln`, `neither`) for both the FT-Transformer (re-implemented to expose it) and the Regression Block; `run_lnfix(dataset, variant, seed_idx)` asks whether the no-LayerNorm failure is an optimization artifact (`ln_reference`, `noln_base`, `noln_rezero`, `noln_smallw`) |
| Helpers | deployed; tags `resln` and `lnfix`, collector template |
| CSVs | `results/audit/audit_resln_results.csv` (8 × 2 × 4 × 5 = 320); `audit_lnfix_results.csv` (8 × 4 × 5 = 160) |

### 13. Mixer / attention compositions (not in the paper)

| | |
|---|---|
| App | `rebuttal_mixattn.py` (app `neurips-31482-mixattn`), `run_stack(dataset, config, seed_idx)`: one harness, one tokenizer, mean-pool readout, only the stacked sublayers vary — `mix_ffn` (= the Regression Block), `mix_attn_ffn`, `attn_ffn` (a one-block FT-T), `ffn_only` (no mixing at all), `mix_attn` |
| Helpers | deployed; tag `mixattn`, collector template |
| CSVs | `results/audit/audit_mixattn_results.csv` (8 × 5 × 5 = 200). `results/audit/audit_fttnoffn_results.csv` (the FT-Transformer with and without its FFN sublayer, 80 rows, version tag `v4-noffn-ftt`) answers the same question, but the harness that produced it was not preserved |

## Name tokens

`ystd` targets standardized on the training split (the corrected protocol) · `abl` ablation grid ·
`ws` warm start · `a6ystd` the paper's A7 row (pre-renumbering label) · `a7nowarm` its cold-start
twin · `sub` sublayer factorial · `noffn` no FFN sublayer · `pca` PCA ladder · `agg` the trimmed
("aggressive") configuration · `uy` uncapped, y-standardized · `fm` foundation models, uncapped ·
`fmhd` foundation models, higher dimensions · `fr`, `fr2`, `fr3`, `frall` Friedman controls ·
`resln`, `lnfix` residual / LayerNorm · `mixattn` sublayer compositions · `super` Superconductivity.
