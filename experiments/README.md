# experiments/

Everything that was run after submission: the rebuttal (July 2026) and the revision
(September 2026). All of it runs on [Modal](https://modal.com). None of it is needed to
regenerate the paper's tables from the saved CSVs; `python3 tables/gen_tables.py` does that
from `results/` alone.

| Folder | What it holds |
|---|---|
| [`modal/`](modal/README.md) | the experiment apps (`rebuttal_*.py`), documented by family in that README: ablation grid, warm start, widened block, depth / sublayers, faithful-variant ladder, trimmed configuration, classification, uncapped N, higher dimensions, foundation models, Friedman controls, residual / LayerNorm, mixer / attention compositions |
| [`modal/collectors/`](modal/collectors/README.md) | the `*_spawn.py` / `*_collect.py` helpers of the apps that run as deployed apps |
| [`modal/utils/`](modal/utils/README.md) | rebuttal-time utilities: cross-experiment consistency checks, report and draft generators, a standalone stacked block with its local test, a TabICL device diagnostic, the standalone classification module |
| [`audit/`](audit/README.md) | the text-vs-code audit probes: the submitted Table 1 reproduced column by column from the paper's own descriptions, the target-scaling correction, the sensitivity sentence, the Monte Carlo per the appendix spec, and forensics on the submitted Att. Reg column |

Every CSV these runs produced is under [`results/`](../results/README.md). The per-family
tables in `modal/README.md` and `audit/README.md` name, for each CSV, the script, the Modal app,
the remote function and tag, the collector, and the paper table that uses it.

## The two protocols

Two conventions run through the file names, so they are worth fixing once:

* **Raw targets**, the convention of the submitted paper (`paper_code/real_data_benchmark_v4.py`):
  features standardized on the training split, target left on its original scale. The first
  rebuttal wave (`results/rebuttal/`) uses it.
* **Corrected protocol, "ystd"**: features *and* target standardized on the training split
  (R² is computed on the standardized scale, which is affine-invariant). The revised paper's
  FT-Transformer, Regression Block and MLP numbers all use it, and the CSV names carry the token:
  `audit_ystd`, `audit_ablystd`, `audit_wsystd`, `audit_a6ystd`, `audit_mlpystd`, `audit_uy`
  (uncapped, y-standardized). OLS, RF and the Attention Regression estimator are
  scale-equivariant, so their published values were kept once verified (`audit/`).

## Common conventions of every app

* Pinned image: `numpy 1.26.4`, `pandas 2.2.3`, `scikit-learn 1.5.2`, `torch 2.4.1`
  (plus `tabpfn>=2.0.0`, `tabicl>=0.1.0` where a foundation model is involved).
* Datasets are fetched once from UCI / OpenML into the Modal volume `neurips-31482-cache`
  (`datasets_v1.pkl` for the eight Table 1 datasets, written by `fetch_and_cache` in
  `modal/rebuttal_v4_port.py`; the higher-dimensional and classification suites have their own
  `fetch_and_cache_*` functions in `rebuttal_highdim.py` and `rebuttal_classify.py`).
* One remote function call per cell of a grid, `(dataset, config, seed_idx)`, with
  `seed = 42 + 1000 * seed_idx`, five random 80/20 splits, N capped at 5000 (stratified for
  classification). Remote functions never raise: a failed cell comes back as a row with a
  non-empty `error` column, and `tables/gen_tables.py` drops such rows.
* Seeds, caps, split fractions and every hyperparameter are constants at the top of each script.
