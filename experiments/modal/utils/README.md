# experiments/modal/utils/ — rebuttal-time utilities

None of these is imported by an experiment app, and none is needed to regenerate the paper's
tables. They are kept because they document how the rebuttal numbers were checked and written up.

| File | Role |
|---|---|
| `verify_all.py` | full verification pass over the rebuttal CSVs: file integrity and seed completeness, metric ranges and parameter invariants, cross-experiment consistency of cells that independent runs must agree on (e.g. `depth` FT-T with 3 blocks vs ablation A0, learning curve at N = full vs uncapped), Table 1 recomputed from raw rows, and the headline claims of the drafts recomputed from raw data. Exit code 0 iff no FAIL. Writes `verification_report.txt`. |
| `build_report_v2.py` | the consolidated rebuttal report (HTML → PDF) assembled from the result CSVs; sections skip gracefully when a CSV is missing. |
| `make_drafts.py` | OpenReview response drafts (global and per reviewer) with every number pulled from the final CSVs at generation time; writes `response_*.md`. Drafts only, nothing is posted. |
| `stacked_rb.py` | standalone multi-layer ("stacked") Regression Block: L blocks of PCA-of-polycross → regression mixer → residual + FFN, with block 1 warm-started by Ridge. `rebuttal_depth.py` carries its own copy of this logic. |
| `test_stacked_local.py` | CPU sanity test of `stacked_rb.train_stacked_rb` (imports it, so the two stay together): L = 1, 2, 3 run without NaNs, parameter counts grow with depth, California R² is in the right ballpark. Not a paper-number reproduction. |
| `tabicl_diag.py` | Modal diagnostic that found which `device` argument `TabICLRegressor` accepts; the fix is in `rebuttal_fm_uncapped.py` and explains the errored TabICL rows of `audit_fm_results.csv`. |
| `classify.py` | the standalone classification analogues of the FT-Transformer and the Regression Block (cross-entropy head, multinomial-logistic warm start). Inlined verbatim into `rebuttal_classify.py`, which is the file that ran. |

`verify_all.py`, `build_report_v2.py` and `make_drafts.py` look for the CSVs next to themselves
(`HERE`), as they did in the working directory of the rebuttal. To run them today, copy or
symlink the relevant files from `results/rebuttal/` and `results/audit/` into this folder.
