# Project setup and local execution

## 1. Scope and supported environment

The active repository is a local, CPU-oriented scientific Python project. It
does not require Modal, a cloud volume, a GPU, or another orchestration service.
Python 3.11 and 3.12 are supported; Python 3.12 is the tested reference.

Run commands from the repository root unless using the installed
`ddr-mksvm` console command. Windows PowerShell examples are used below.

## 2. Environment setup

```powershell
cd 'F:\Projects\SVM_Improvement\.SVM Version 2\ddr_project'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the editable package plus development and retained
deep-kernel test dependencies. Install `requirements-paper.txt` only when
rebuilding the manuscript. The minimal active model dependencies are declared
under `[project.dependencies]` in `pyproject.toml`; optional groups are `dev`,
`deep`, and `paper`.

```powershell
python -m pip install -r requirements-paper.txt  # optional manuscript tools
```

System tools:

- No external solver is needed for the supported RKHS SVMs; scikit-learn uses
  its bundled LIBSVM implementation.
- SciPy/HiGHS is used by the source-profile LP code.
- Pandoc and Microsoft Word are needed only to rebuild the journal-style DOCX
  and PDF with `paper/final_manuscript/build_manuscript_outputs_revised_3.py`.
- Git is recommended so manifests can record a commit and dirty-state flag.

## 3. Repository structure

| Directory | Purpose |
|---|---|
| `ddr_mksvm/` | installable source package |
| `ddr_mksvm/models/` | scientific-name facades for frozen model modules |
| `ddr_mksvm/data/` | canonical dataset registry/loaders |
| `ddr_mksvm/evaluation/` | metric and paired-statistics interfaces |
| `ddr_mksvm/experiments/` | seed/protocol registry |
| `ddr_mksvm/kernels/`, `optim/`, `dro/`, `robust_solvers/` | mathematical components and retained research implementations |
| `dataset/` | local immutable input CSV files |
| `configs/` | seed and protocol metadata |
| `scripts/` | thin local CLI launcher and script guidance |
| `tests/` | regression, reduction, provenance, and numerical tests |
| `results/` | immutable historical artifacts plus local run outputs |
| `docs/` | research reports, setup, audit, and protocol documentation |
| `paper/` | final manuscript, tables, figures, evidence map, and build tools |
| `archive/modal/` | inactive cloud-specific historical wrappers |
| `archive/research_scripts/` | superseded or rejected experiment drivers retained for provenance |
| `.github/` | concise contribution guidance, templates, and local-equivalent CI |

See `docs/REPOSITORY_MIGRATION.md` for old-to-new path mappings.

## 4. Dataset setup and provenance

CSV files live in `dataset/`. The canonical inventory is
`results/expanded_dataset_study/inventory/dataset_inventory.csv`; file hashes
are stored beside it. Rebuild the inventory without fitting models:

```powershell
python -m ddr_mksvm.cli inventory
```

Loaders find data relative to the repository root. Numeric tables use the
frozen label-last schemas. The Breast Cancer recurrence cohort uses its
documented semantic repair, training-only one-hot vocabulary, unknown-category
handling, and training-only standardization. Missing-data rules and complete-
case exclusions are frozen in `ddr_mksvm/expanded_dataset_study.py`.

To add a dataset safely:

1. place the immutable raw/local CSV under `dataset/`;
2. document its license/source and compute its SHA-256;
3. add a schema entry and loader without modifying existing entries;
4. define training-only preprocessing and class/label semantics;
5. add loader, split, leakage, and hash tests;
6. allocate new prospective seeds in a new protocol; and
7. never use test outcomes to change the architecture or search space.

## 5. Model names and status

| Historical label | Scientific name | Status |
|---|---|---|
| Legacy | Source-Profile q1 Kernel SVM | audited executable/reference baseline |
| V7 | RKHS-RBF-OVO SVM | supported deterministic model |
| V8-C | CS-RKHS-RBF-OVO SVM | supported class-sensitive alternative |
| V8.1 | Training-Only Dual-Kernel Selection SVM | rejected at Gate8; archived |
| Robust V8-C | RKHS-Norm Robust Class-Sensitive SVM | numerically rejected |
| R8/R8.1 | Source-Aligned robust q1 LP/equilibrated LP | rejected |
| R9 | Bounded-RBF Class-Sensitive Robust q1 SVM | rejected at fresh Gate8 |

The robust track is scientifically closed. Its code, solver diagnostics, and
negative results remain available for audit, but it is not an active model
selection path.

## 6. Running tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
```

The environment variable avoids unrelated globally installed pytest plugins.
Scientific assertions, frozen hashes, reduction identities, data semantics,
checkpoint behavior, and consolidated results are all tested.

## 7. Local experiments

List commands:

```powershell
python -m ddr_mksvm.cli --help
```

### 7.1 One-seed smoke

Use a new development seed that is not reserved for confirmation:

```powershell
python -m ddr_mksvm.cli smoke `
  --dataset breast_cancer_recurrence `
  --seed 17000 `
  --n-jobs 1 `
  --run-name local-smoke-17000
```

### 7.2 RKHS-RBF-OVO (historical V7) comparison

```powershell
python -m ddr_mksvm.cli v7 `
  --datasets parkinson wine `
  --seed-start 17000 --n-seeds 1 --n-jobs 1 `
  --run-name local-rkhs-smoke
```

### 7.3 Class-sensitive RKHS-RBF-OVO (historical V8-C)

```powershell
python -m ddr_mksvm.cli v8c `
  --datasets blood_transfusion heart_disease `
  --variants v7 v8_c `
  --seed-start 17000 --n-seeds 1 --n-jobs 1 `
  --run-name local-class-sensitive-smoke
```

### 7.4 Expanded dataset stages

The following commands show the frozen historical stage sizes. **The ranges
16000-16395 are already consumed evidence. Do not rerun them as if they were
new confirmation.** Use a separate output root for an explicit reproduction.

Smoke:

```powershell
python -m ddr_mksvm.cli expanded `
  --datasets breast_cancer_recurrence dermatology `
  --models legacy v7 v8_c --stage smoke `
  --seed-start 16000 --n-seeds 1 --n-jobs 1 `
  --run-name reproduction-smoke-16000 `
  --output-root results/local_reproduction
```

Gate8:

```powershell
python -m ddr_mksvm.cli expanded `
  --datasets breast_cancer_recurrence dermatology `
  --models legacy v7 v8_c --stage gate8 `
  --seed-start 16100 --n-seeds 8 --n-jobs 4 `
  --run-name reproduction-gate8-16100-16107 `
  --output-root results/local_reproduction
```

Gate24, using the frozen Gate8 decisions and only the promoted comparison:

```powershell
python -m ddr_mksvm.cli expanded `
  --datasets breast_cancer_recurrence dermatology `
  --models legacy v7 --stage gate24 `
  --seed-start 16200 --n-seeds 24 --n-jobs 4 `
  --promotion-source results/expanded_dataset_study/gate8/expanded-gate8-16100-16107-v1/analysis/gate_decisions.csv `
  --run-name reproduction-gate24-16200-16223 `
  --output-root results/local_reproduction
```

Untouched-at-the-time 96-seed confirmation, authorized by the frozen Gate24
table:

```powershell
python -m ddr_mksvm.cli expanded `
  --datasets breast_cancer_recurrence dermatology `
  --models legacy v7 --stage confirmation `
  --seed-start 16300 --n-seeds 96 --n-jobs 4 `
  --promotion-source results/expanded_dataset_study/gate24/expanded-gate24-16200-16223-v1/analysis/gate_decisions.csv `
  --run-name reproduction-confirmation-16300-16395 `
  --output-root results/local_reproduction
```

Never consume `14000-14095` or `15400-15423`; they remain reserved for a future
independently derived robust method. The canonical registry is
`configs/seed_registry.toml`.

## 8. Analysis and paper reproduction

Analyze any completed/partial expanded run:

```powershell
python -m ddr_mksvm.cli analyze results/local_smoke/local-smoke-17000
```

Rebuild the final cross-dataset summaries from the already frozen artifacts:

```powershell
python -m ddr_mksvm.cli consolidate
```

Recommended manuscript reproduction sequence, with no model fitting:

```powershell
python -m ddr_mksvm.cli inventory
python -m ddr_mksvm.cli consolidate
python paper/final_manuscript/generate_paper_figures.py
python paper/final_manuscript/build_paper_data_tables.py
python paper/final_manuscript/build_manuscript_outputs_revised_3.py
python paper/final_manuscript/validate_revised_3.py
```

The final PDF build is Windows/Word-specific; the Markdown, data tables, and
figures are portable.

## 9. Checkpoints and results

Each staged run contains, as applicable:

- `config.json` and `manifest.json`;
- atomic per-dataset/per-seed `checkpoints/`;
- explicit `failures/` records;
- `per_run.csv`, `summary.csv`, `paired_comparisons.csv`, class metrics, and
  solver diagnostics;
- `split_registry.json`, selected hyperparameters, and source snapshots; and
- `analysis/` tables generated by the audit command.

Re-running an identical run name resumes valid checkpoints or reports an
already-complete immutable run. An incompatible configuration is rejected.
Use a new run name instead of deleting or overwriting a completed result.

Paper-ready tables and plots are under `paper/final_manuscript/`; consolidated
statistical tables are under
`results/expanded_dataset_study/analysis/cross_dataset/`.

## 10. Troubleshooting

### Missing dataset

Run `python -m ddr_mksvm.cli inventory`, verify the expected CSV name under
`dataset/`, and compare its SHA-256 with the inventory. Do not silently fetch a
different revision during a confirmation run.

### Windows paths

Quote the repository path because it contains spaces. Prefer `pathlib` in new
code. Run commands from the project root or use the installed console command.

### Solver failure

Retain the failure JSON and solver diagnostics. Do not change tolerances,
scaling, kernels, or hyperparameters inside a frozen gate. The robust failure
policies in `docs/robust/` remain authoritative for historical branches.

### Long run

Start with one seed and `--n-jobs 1`. Increase to at most four workers after
measuring memory/runtime. Gate24 and 96-seed runs can take hours locally; the
checkpoint structure is designed for resume.

### Resume or incompatible config

Repeat the identical command to reuse valid checkpoints. If the runner reports
an incompatible fingerprint, inspect the existing `config.json`; choose a new
run name for a genuinely different configuration. Never edit the stored config
to force reuse.

### Corrupt checkpoint

Preserve the file and failure evidence. The active runners reject incompatible
or malformed checkpoints rather than silently accepting them. Copy the run
directory before any forensic repair.
