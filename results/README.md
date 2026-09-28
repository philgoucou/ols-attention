# results/ — every per-seed CSV

One row per (dataset, configuration, seed) cell as returned by the remote function that ran it;
nothing is aggregated here. `tables/gen_tables.py` averages these files into the paper's tables,
dropping rows whose `error` column is non-empty. File names are the ones the scripts wrote and
are referenced as such by the README map and the table generator, so they are kept unchanged.

Common columns: `dataset`; `seed_idx` 0–4 (`seed = 42 + 1000 * seed_idx`, one random 80/20 split
each); `r2` test R² on the held-out 20 % (on the standardized scale under the corrected protocol);
`acc`, `auc` accuracy and ROC-AUC (macro-OVR for multiclass); `params` trainable parameter count;
`wall_s` wall-clock seconds of the cell; `error` empty, or the traceback of a failed cell;
`kind` the tag of the probe family; `code_version` / `version` the harness version tag.

## results/rebuttal/ — first rebuttal wave (July 2026), raw-target protocol

| File | Rows | Produced by | Contents | Columns |
|---|---|---|---|---|
| `a6_results.csv` | 40 | app `neurips-31482-a6`, the first-wave (raw-target) version of `experiments/modal/rebuttal_a7_widened.py`; `experiments/modal/collectors/a6_spawn.py`, `a6_collect.py` | the widened Regression Block (the paper's A7; `config` reads `A6`): d_model 128, 260 components, matched to the FT-T budget (`ftt_target`) | kind, dataset, config, seed_idx, r2, params, d_model, ncomp, ftt_target, wall_s, error |
| `ablation_grid_results.csv` | 280 | `experiments/modal/rebuttal_v4_port.py --part abl` | A0–A5 and RB on the eight datasets, raw targets; the A0 (FT-T) and RB columns are the submitted-convention table of App. D | kind, dataset, config, seed_idx, seed, r2, params, wall_s, error |
| `classify_results.csv` | 150 | `experiments/modal/rebuttal_classify.py` | LogReg, RF, FTT, RB, TabPFN on BreastCancer, Diabetes, Phoneme, Wine, Vehicle, Segment → App. F | dataset, model, n_classes, seed_idx, acc, auc, wall_s, error |
| `depth_results.csv` | 240 | `experiments/modal/rebuttal_depth.py` | RB stacked to L = 1, 2, 3 (`RB-L1..3`) and the FT-T with 1–3 blocks (`FTT-nb1..3`), raw targets; not in the paper | kind, dataset, model, depth, seed_idx, r2, params, wall_s, error |
| `highdim_results.csv` | 140 (1 error) | `experiments/modal/rebuttal_highdim.py`, Superconductivity RB seeds via `experiments/modal/collectors/super_spawn.py` / `super_collect.py` | OLS, RF, FTT, RB on CPU_act (P 21), Bank32nh (32), Ailerons (40), Pol (48), Superconductivity (81), Friedman-P20, Friedman-P50 → App. F (real datasets). Superconductivity / RB seed 3 timed out (5,400 s), so that cell averages four seeds | dataset, model, P, seed_idx, r2, wall_s, error, params |
| `lcurve_results.csv` | 180 | `experiments/modal/rebuttal_warmstart_lcurve.py` | learning curves: FTT and RB on California, Kin8nm, Protein at `n_train_req` ∈ {500, 1000, 2500, 5000, 10000, −1 = full}; not in the paper | kind, dataset, model, n_train_req, n_train_used, seed_idx, r2, wall_s, error |
| `tabicl_results.csv` | 30 | `experiments/modal/rebuttal_tabicl.py` | TabICL on the six classification datasets → App. F | dataset, model, n_classes, seed_idx, acc, auc, wall_s, error |
| `tabpfn_results.csv` | 40 | `experiments/modal/rebuttal_v4_port.py --part tab` | TabPFN on the eight datasets → App. F | kind, dataset, seed_idx, r2, wall_s, error |
| `uncapped_results.csv` | 60 | `experiments/modal/rebuttal_v4_port.py --part unc` (Protein / RB seeds re-run through `::patch_protein_rb`) | OLS, RF, FTT, RB at full N on California (20,640), Kin8nm (8,192), Protein (45,730), raw targets; superseded by `audit/audit_uy_results.csv` | kind, dataset, N, model, seed_idx, r2, wall_s, error |
| `warmstart_results.csv` | 120 | `experiments/modal/rebuttal_warmstart_lcurve.py` | `softmax` / `rb_nowarm` / `rb_warm` on the eight datasets, raw targets; `gen_tables.py` (`ws_raw`) compares it with the corrected-protocol rerun | kind, dataset, config, seed_idx, r2, params, wall_s, error |
| `audit_attreg_results.csv`, `audit_attreg_std_results.csv`, `audit_cpu_results.csv`, `audit_legacy4_results.csv`, `audit_legacy_attn_results.csv`, `audit_mc_attreg_std_results.csv`, `audit_mc_legacy_results.csv`, `audit_mc_results.csv`, `audit_mlp_results.csv`, `audit_sens_results.csv`, `audit_ystd_results.csv` | | | byte-identical copies of the files of the same name in `audit/`, where they were first collected in July 2026; kept because the rebuttal referred to them at this location | |

## results/audit/ — audit and corrected-protocol runs (every number of the revised paper's neural columns)

| File | Rows | Produced by | Contents | Columns |
|---|---|---|---|---|
| `audit_a6ystd_results.csv` | 40 | `experiments/modal/rebuttal_a7_widened.py`; `experiments/modal/collectors/a6ystd_collect.py` | the widened block, corrected protocol: **row A7** of the App. E grid (`config` reads `A6`, its pre-renumbering label) | kind, code_version, dataset, config, seed_idx, r2, params, d_model, ncomp, ftt_target, wall_s, error |
| `audit_a7nowarm_results.csv` | 40 | `experiments/modal/rebuttal_a7_widened_nowarm.py`; `experiments/modal/collectors/a7nowarm_collect.py` | the same block without the Ridge warm start (App. E text) | same as above |
| `audit_ablystd_results.csv` | 280 | `experiments/modal/rebuttal_ablation_warmstart.py`, `run_abl_ystd` | A0–A5 and RB, corrected protocol: **rows A0–A5 and RB** of the App. E grid | kind, dataset, config, seed_idx, r2, params, wall_s, error |
| `audit_agg_results.csv` | 135 (10 errors) | `experiments/modal/rebuttal_trimmed.py` (reconstruction of the lost harness) | the trimmed configuration (n_mix ∈ {2, 3} mixers, no FFN, 50 components) on four settings: `std` (eight datasets), `uncapped` (California, Kin8nm, Protein), `highdim` (five datasets), `classify` (six datasets); `score` is R², or accuracy for `classify`. The ten Superconductivity cells failed (CUDA out of memory). → App. E (`trim_scope`, n_mix = 2) | setting, dataset, n_mix, ncomp, seed_idx, score, P, n_train, params, version, wall_s, error |
| `audit_attreg_results.csv` | 40 | `experiments/audit/audit_probes.py`, `run_attreg` | the Eq. (24) Attention Regression estimator per the appendix, raw targets | kind, dataset, model, seed_idx, r2, wall_s, error |
| `audit_attreg_std_results.csv` | 40 | `experiments/audit/audit_probes2.py`, `run_attreg_std` | the same with features and targets standardized (scale-equivariance check of Table 1's Att. Reg column) | kind, dataset, seed_idx, r2, wall_s, error |
| `audit_cpu_results.csv` | 80 | `experiments/audit/audit_probes.py`, `run_cpu` | OLS and RF, capped: reproduces the submitted columns | kind, dataset, model, seed_idx, r2, wall_s, error |
| `audit_faithful_results.csv` | 320 | `experiments/modal/rebuttal_faithful.py`; `experiments/modal/collectors/faithful_spawn.py`, `faithful_collect.py` | the faithful-variant ladder `cur_L1`, `gelu_L1`, `cls_L1`, `pertok_L1`, `faith_L1`, `faith_L2`, `faith_L3`, `cur_L3`; **`cur_L3` is row A6** of the App. E grid | kind, dataset, variant, tokenizer, readout, activation, n_layers, seed_idx, r2, params, wall_s, error |
| `audit_fm_results.csv` | 30 (15 errors) | `experiments/modal/rebuttal_fm_uncapped.py`; `experiments/modal/collectors/fm_collect.py` | TabPFN at full N on California, Kin8nm, Protein → App. F; the 15 TabICL rows failed on a device-string bug (see the next file). `n_train_used == n_train_available` on every valid row | dataset, model, seed_idx, cap, n_train_available, n_train_used, limits_overridden, default_refused, r2, wall_s, error |
| `audit_fm_tabicl_uncapped.csv` | 15 | `experiments/modal/rebuttal_fm_uncapped.py`, re-run after the device fix found with `experiments/modal/utils/tabicl_diag.py` | TabICL at full N → App. F | same as above |
| `audit_fmhd_results.csv` | 70 | `experiments/modal/rebuttal_fm_highdim.py`; `experiments/modal/collectors/fmhd_spawn.py`, `fmhd_collect.py` | TabPFN and TabICL on the higher-dimensional suite (five real datasets and Friedman-P20 / P50) → App. F | dataset, model, seed_idx, P, n_train, r2, limits_overridden, default_refused, code_version, wall_s, error |
| `audit_fr_results.csv` | 120 | `experiments/modal/rebuttal_friedman.py`; `experiments/modal/collectors/friedman_spawn.py`, `friedman_collect.py` | Friedman control, arms `frac5` and `orig50` × P ∈ {10, 20, 50} × OLS, RF, FTT, RB; not in the paper | arm, P, model, seed_idx, r2, wall_s, error |
| `audit_fr2_results.csv` | 20 | `experiments/modal/rebuttal_friedman2.py`; `experiments/modal/collectors/friedman2_collect.py` | RB with 400 and 800 PCA components on arm `orig50`, P ∈ {20, 50}; not in the paper | arm, P, ncomp, n_comp_used, evr, params, seed_idx, r2, wall_s, error |
| `audit_fr3_results.csv` | 80 | `experiments/modal/rebuttal_friedman.py`, arm `all100`; `experiments/modal/collectors/friedman3_collect.py` | all features relevant, P ∈ {5, 10, 20, 50}; not in the paper | arm, P, model, seed_idx, r2, wall_s, error |
| `audit_frall_results.csv` | 200 | concatenation of `audit_fr_results.csv` and `audit_fr3_results.csv` | the three Friedman arms together | arm, P, model, seed_idx, r2, wall_s, error |
| `audit_fttnoffn_results.csv` | 80 | harness not preserved (version tag `v4-noffn-ftt`) | the FT-Transformer with (`ffn`) and without (`noffn`) its FFN sublayer, corrected protocol; not in the paper | dataset, model, variant, seed_idx, r2, params, diverged, version, wall_s, error |
| `audit_legacy4_results.csv` | 80 | `experiments/audit/audit_probes4.py` | forensics: the legacy 2-block self-attention net with 300 and 3000 steps, targets standardized | kind, dataset, seed_idx, steps, ystd, r2, wall_s, error |
| `audit_legacy_attn_results.csv` | 80 | `experiments/audit/audit_probes3.py` | forensics: the legacy net with and without the test-conditioned retry | kind, dataset, seed_idx, with_retry, r2, wall_s, error |
| `audit_lnfix_results.csv` | 160 | `experiments/modal/rebuttal_resln.py`, `run_lnfix` | RB without LayerNorm under optimization fixes: `ln_reference`, `noln_base`, `noln_rezero`, `noln_smallw`; not in the paper | dataset, variant, seed_idx, r2, code_version, rezero, w_init_scale, diverged, wall_s, error |
| `audit_mc_attreg_std_results.csv` | 96 | `experiments/audit/audit_probes2.py`, `run_mc_attreg_std` | Monte Carlo cells (6 DGPs × 4 N × 4 SNR): Att. Reg with standardized inputs and targets, mean and sd over replications | kind, dgp, N, snr, error, AttReg_mean, AttReg_sd, wall_s |
| `audit_mc_legacy_results.csv` | 96 | `experiments/audit/audit_probes3.py`, `run_mc_legacy` | Monte Carlo cells: the legacy self-attention net (forensics) | kind, dgp, N, snr, error, Attn_mean, Attn_sd, wall_s |
| `audit_mc_results.csv` | 96 | `experiments/audit/audit_probes.py`, `run_mc_cell` | the full Monte Carlo per the appendix spec, 10 replications per cell: mean and sd of OLS, RF, GBM, MLP, Att. Reg (independent check of Fig. 2 / App. B) | kind, dgp, N, snr, error, OLS_mean, OLS_sd, RF_mean, RF_sd, GBM_mean, GBM_sd, MLP_mean, MLP_sd, AttReg_mean, AttReg_sd, wall_s |
| `audit_mixattn_results.csv` | 200 | `experiments/modal/rebuttal_mixattn.py` | sublayer compositions `mix_ffn`, `mix_attn_ffn`, `attn_ffn`, `ffn_only`, `mix_attn` with mean-pool readout; not in the paper | dataset, config, readout, seed_idx, r2, params, pred_std, version, wall_s, error |
| `audit_mlp_results.csv` | 40 | `experiments/audit/audit_probes.py`, `run_mlp` | the MLP per the paper's spec with features standardized only (the submitted convention) | kind, dataset, model, seed_idx, r2, wall_s, error |
| `audit_mlpystd_results.csv` | 40 | `experiments/audit/audit_probes_mlp_ystd.py`; `experiments/audit/collectors/mlpystd_collect.py` | the MLP under the corrected protocol: **the MLP column of Table 1** | kind, code_version, dataset, model, seed_idx, r2, wall_s, error |
| `audit_noffn_results.csv` | 160 | `experiments/modal/rebuttal_sublayer.py`, `run_sublayer` with `n_ffn = 0` | `mix1_ffn0`, `mix2_ffn0`, `mix3_ffn0`, plus the `mix1_ffn1` reference rows copied from `audit_sub_results.csv`; not in the paper | dataset, n_mix, n_ffn, variant, seed_idx, r2, params, wall_s, error |
| `audit_pca_results.csv` | 320 | `experiments/modal/rebuttal_sublayer.py`, `run_pca_sweep` | the PCA ladder `mix1_ffn1_pca{25,50,100,200}`, `mix2_ffn0_pca{25,50,100,200}`; **`mix2_ffn0_pca50` is the trimmed configuration** of App. E (the `mix1_ffn1_pca200` rows are those of `audit_sub_results.csv`) | dataset, variant, n_mix, n_ffn, ncomp, seed_idx, r2, params, version, wall_s, error |
| `audit_resln_results.csv` | 320 | `experiments/modal/rebuttal_resln.py`, `run_resln` | FTT and RB with the post-norm scaffolding removed piece by piece: `full`, `no_res`, `no_ln`, `neither`; not in the paper | dataset, model, variant, use_res, use_ln, seed_idx, r2, params, diverged, wall_s, error |
| `audit_sens_results.csv` | 240 | `experiments/audit/audit_probes.py`, `run_sens` | the Regression Block's sensitivity: `d32`, `d128`, `dr00`, `dr03`, `lr1em4`, `lr1em2` (App. C sentence) | kind, dataset, variant, seed_idx, r2, wall_s, error |
| `audit_sub_results.csv` | 160 | `experiments/modal/rebuttal_sublayer.py`, `run_sublayer`; `experiments/modal/collectors/sublayer_spawn.py`, `sublayer_collect.py` | the mixer × FFN sublayer factorial `mix{1,3}_ffn{1,3}`; not in the paper | dataset, n_mix, n_ffn, variant, seed_idx, r2, params, wall_s, error |
| `audit_uy_results.csv` | 60 | `experiments/modal/rebuttal_uncapped_ystd.py`; `experiments/modal/collectors/uy_collect.py` | OLS, RF, FTT, RB at full N (n_train 16,512 / 6,553 / 36,584), corrected protocol → App. F | dataset, model, seed_idx, n_train, r2, wall_s, error |
| `audit_wsystd_results.csv` | 120 | `experiments/modal/rebuttal_ablation_warmstart.py`, `run_ws_ystd` | `softmax` / `rb_nowarm` / `rb_warm`, corrected protocol → App. E readout-vs-warm-start table | kind, dataset, config, seed_idx, r2, params, wall_s, error |
| `audit_ystd_results.csv` | 80 | `experiments/audit/audit_probes.py`, `run_ystd`; `experiments/audit/collectors/audit_spawn.py`, `audit_collect.py` | FTT and RB under the corrected protocol: **the FT-T and Reg. Block columns of Table 1**, the paired SEs of App. D, and the reference columns of the App. E and F tables | kind, dataset, model, seed_idx, r2, wall_s, error |

## Notes

* **Failed cells.** `highdim_results.csv`: Superconductivity / RB seed 3 (`FunctionTimeoutError`,
  5,400 s), so the paper's Superconductivity RB number averages four seeds. `audit_agg_results.csv`:
  the ten Superconductivity cells (CUDA out of memory, 38.5 GiB requested), so the trimmed
  configuration has no Superconductivity number. `audit_fm_results.csv`: all fifteen TabICL rows
  (`ValueError` on the device argument), superseded by `audit_fm_tabicl_uncapped.csv`.
* **Parameter counts in the App. E summary.** `tables/gen_tables.py` hard-codes them (`PR`). For
  A0–A5 and RB the values are the `params` column of `audit_ablystd_results.csv` at P = 8 (California,
  Concrete, Energy, Kin8nm). For A6 (89,985) and A7 (101,377) they are the P = 9 (Protein) rows of
  `audit_faithful_results.csv` (`cur_L3`) and `audit_a6ystd_results.csv`; the P = 8 rows of those files
  read 89,921 and 101,249. Left as generated.
* **Copied rows.** `audit_frall` is `audit_fr` + `audit_fr3`; the `mix1_ffn1` rows of `audit_noffn`
  and the `mix1_ffn1_pca200` rows of `audit_pca` are the `mix1_ffn1` rows of `audit_sub` (identical
  values, not re-run).
* **Labels.** `config = A6` inside `audit_a6ystd`, `audit_a7nowarm` and `a6_results` is the paper's
  row A7; the paper's A6 is `cur_L3` of `audit_faithful`.
