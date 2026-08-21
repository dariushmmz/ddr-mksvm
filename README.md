# DDR-MKSVM — implementation (v2)

Implementation of `DDR-MKSVM_spec.md`, built on top of the corrected,
fully-parameterized deterministic scripts (`DATASET_CONFIG`,
`_apply_transform`, `_parse_kernel`, `_compute_kernel`,
`_compute_kernel_test` in `unit_of_work_deterministic_binary.py` /
`unit_of_work_deterministic_multiclass.py`). DDR-MKSVM **imports these
directly rather than re-deriving them**, so it automatically tracks any
future change to the dataset registry, transforms, or kernel parsing —
there is exactly one place those live.

## What changed from v1

- **Dataset-parameterized**, matching the corrected base scripts: both
  `unit_of_work_ddr_binary.py` and the new `unit_of_work_ddr_multiclass.py`
  take a `dataset_name` argument and look up `DATASET_CONFIG` themselves,
  instead of hardcoding one CSV/kernel.
- **`ddr_mksvm/config_adapter.py`** (new) bridges the base scripts' kernel
  string convention (`"hom_linear"`, `"inhom_quadratic"`, `"inhom_cubic"`,
  `"gaussian_rbf"`, ...) to `ddr_mksvm`'s kernel objects, by calling the
  base scripts' own `_parse_kernel`/`_compute_kernel` — not reimplementing
  the parsing. `tests/test_config_adapter.py` checks this numerically for
  every kernel string in both registries.
- **`unit_of_work_ddr_multiclass.py`** (new) — the previous version's
  README flagged multiclass as an unported gap. It's now implemented:
  one-vs-all over `L` classes, with `f_theta`/`eta` **shared** across all
  `L` binary subproblems (`AlternatingTrainer.fit_one_vs_all`), matching
  how the deterministic script already shares one kernel across all `L`
  subproblems.
- **`main_ddr_binary.py` / `main_ddr_multiclass.py`** now take
  `--dataset`/`--ablation`/`--n_runs`, matching the corrected base
  scripts' CLI, and save to `results/<dataset>/ddr_mksvm/` — same
  Colab-workflow shape as `running_deterministic_version.py`.
- The corrected `DATASET_CONFIG` in `unit_of_work_deterministic_multiclass.py`
  now correctly lists `heart_disease` (5 classes) and `dermatology`
  (6 classes) as multiclass — matching what the earlier reproduction-report
  review flagged as the likely-correct reading of the base paper's Table
  3/5 formatting. No action needed on the DDR-MKSVM side beyond picking
  this registry up automatically (which it does, by import).

## What's here

```
ddr_mksvm/
  config_adapter.py             NEW: bridges DATASET_CONFIG kernel strings
                                 to ddr_mksvm kernel specs, via the base
                                 scripts' own _parse_kernel/_compute_kernel.
  kernels/base_kernels.py       Linear / Polynomial / Gaussian-RBF kernels
                                 with analytic Lipschitz bounds (Lemmas 3-4).
  kernels/deep_kernel.py        f_theta (Definition 3) + Identity fallback.
  kernels/mkl_combination.py    eta on the simplex via softmax (Lemma 1).
  lipschitz/analytic_bounds.py  Lemma 2 (spectral-norm product), Lemma 5.
  lipschitz/empirical_estimator.py  Diagnostic-only empirical Lipschitz estimate.
  dro/wasserstein_risk.py       Theorem 1's closed-form DRO risk.
  optim/convex_subproblem.py    Corollary 1's q=2 SOCP plus explicit legacy q=1 LP.
  optim/alternating_trainer.py  fit() for binary, fit_one_vs_all() for multiclass.
  legacy_reduction/base_paper_mode.py  Standalone Proposition 4.8 check.

unit_of_work_ddr_binary.py      dataset_name-parameterized, binary.
unit_of_work_ddr_multiclass.py  NEW: dataset_name-parameterized, multiclass.
main_ddr_binary.py              --dataset/--ablation/--n_runs CLI, binary.
main_ddr_multiclass.py          NEW: same CLI, multiclass.

tests/                          Maps 1:1 to Part I's lemmas/theorem, plus
                                 test_config_adapter.py for the new bridge.
```

## Install & run

```bash
pip install -r requirements.txt
pytest tests/ -v

# Binary (matches main_deterministic_binary.py's dataset/ csv convention):
python main_ddr_binary.py --dataset mammographicmass_binary --ablation full
python main_ddr_binary.py --dataset parkinson --ablation all --n_runs 96

# Multiclass:
python main_ddr_multiclass.py --dataset iris --ablation full
python main_ddr_multiclass.py --dataset heart_disease --ablation dro_only
python main_ddr_multiclass.py --dataset dermatology --ablation all
```

Long DDR-MKSVM runs write one atomic checkpoint per completed random
split under `results/<dataset>/ddr_mksvm/checkpoints/`.  After an
interruption, repeat the same command with `--resume`; for example:

```bash
python main_ddr_binary.py --dataset mammographicmass_binary --ablation all --resume
```

The input CSV, `--n_runs`, `--seeded` setting, and ablation configuration
must match the checkpoint.  A command without `--resume` intentionally
starts fresh and replaces checkpoints for each selected ablation as it is
reached.  Completed checkpoint files are retained after a successful run,
so the same `--resume` command can also regenerate aggregate `.mat`/`.csv`
outputs without retraining.

Available `--dataset` values come directly from each script's
`DATASET_CONFIG` (`unit_of_work_deterministic_binary.py` /
`unit_of_work_deterministic_multiclass.py`), printed if you pass an
unknown name.

## What was validated in this environment, and what wasn't

Still **no internet access** in this sandbox, so `cvxpy`/`torch`/`scikit-learn`
couldn't be installed here (your environment already has them — your
uploaded scripts import all three). What *was* run and passed here using
only `numpy` (already present):

- **`tests/test_lemma1_psd.py`**, **`tests/test_lemma3_rbf_lipschitz.py`**,
  **`tests/test_theorem1_duality.py`** — 7/7 passed, unchanged from v1.
- **`tests/test_config_adapter.py`'s actual logic** — manually cross-checked
  in the sandbox by reimplementing `_parse_kernel`/`_compute_kernel`
  locally (since importing the real module needs `cvxpy`) and comparing
  against `kernel_spec_from_config` for all five kernel strings that
  appear in the two registries (`hom_linear`, `inhom_linear`,
  `inhom_quadratic`, `inhom_cubic`, `gaussian_rbf`) — all five matched to
  `1e-8`. The actual test file imports the real modules and will re-run
  this properly once `cvxpy` is installed.

**Not executed here** (run these first in your environment):

- `tests/test_lemma2_spectral_bound.py` (needs `torch`)
- `tests/test_reduction_to_base_paper.py` (needs `cvxpy`) — the numeric
  proof of Proposition 4.8 at the LP level.
- `tests/test_config_adapter.py` (needs `cvxpy`) — the numeric proof that
  DDR-MKSVM's legacy path uses byte-identical kernels to the deterministic
  baseline, for every dataset in both registries, not just one hand-picked
  case.
- End-to-end runs of `main_ddr_binary.py` / `main_ddr_multiclass.py`
  against real dataset CSVs (none were uploaded to this conversation —
  place them under `dataset/` per `running_deterministic_version.py`'s
  convention, e.g. via the same `ucimlrepo` fetch shown there).

**Action item before using this in anger:** run
`pytest tests/test_reduction_to_base_paper.py tests/test_config_adapter.py -v`
first. If either fails, don't trust the `legacy` ablation to actually
equal `unit_of_work_deterministic_binary.py`/`_multiclass.py`'s numbers,
and stop before running `--ablation all` — that would just compound the
discrepancy across every dataset and ablation.

## Known simplifications relative to the spec (unchanged from v1)

- `AlternatingTrainer` uses fixed `n_outer`/`n_inner` iteration counts
  rather than a convergence-based stopping rule.
- The polynomial kernel's Lipschitz bound (Lemma 4) uses the simpler of
  two derivations named in the spec's remark — valid but not the
  tightest possible, making `epsilon`'s penalty more conservative than
  necessary for polynomial kernels. Doesn't affect Theorem 1's
  correctness, only classifier tightness.
- Testing-phase kernel evaluation for the DNN/MKL path recomputes each
  base kernel per test batch rather than caching `f_theta`'s forward
  pass — fine at the dataset sizes here (m up to ~1055), would need
  batching for larger data.
- `fit_one_vs_all`'s shared representation is a design choice, not
  something the spec mandates explicitly — an alternative (independent
  `f_theta` per class) is more expensive and wasn't implemented; worth
  an ablation if per-class errors diverge a lot in practice (e.g. the
  Parkinson-style class-imbalance pathology from the earlier
  reproduction report).
