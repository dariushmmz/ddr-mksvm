# DDR-MKSVM

**Reproducible nonlinear SVM research for imbalanced tabular classification**

[![Tests](https://github.com/dariushmmz/ddr-mksvm/actions/workflows/ci.yml/badge.svg)](https://github.com/dariushmmz/ddr-mksvm/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11–3.12-blue.svg)](https://www.python.org/)

DDR-MKSVM is a research codebase for evaluating kernel support vector machines
under class imbalance. It starts from an audited source-profile baseline,
introduces a conventional RKHS-regularized RBF model, and then evaluates a
class-sensitive extension. The project also preserves robust-optimization
experiments that failed their predefined gates, because negative results are
part of a reproducible scientific record.

The repository is designed as both an executable experiment suite and a
software/research portfolio sample. It demonstrates model implementation,
leakage-safe evaluation, experiment checkpointing, statistical reporting,
regression testing, and evidence-based documentation.

## Research question

Can a carefully tuned nonlinear SVM improve prediction on imbalanced tabular
datasets without hiding class-specific trade-offs?

Three model families are compared:

| Model | Purpose | Status |
|---|---|---|
| Source-Profile q1 Kernel SVM | Audited reference implementation based on the supplied source behavior | Baseline |
| RKHS-RBF-OVO SVM (`V7`) | Gaussian RBF SVM with RKHS-L2 regularization, nested tuning, and one-vs-one multiclass prediction | Supported error-oriented model |
| CS-RKHS-RBF-OVO SVM (`V8-C`) | Adds normalized square-root inverse-frequency penalties and balanced model selection | Supported when balanced performance is the goal |

Several distributionally robust variants were also tested. They were not
promoted: some were numerically unstable, while the solvable bounded-RBF model
failed the fresh predictive gate. Those results remain documented rather than
being removed.

## Key findings

The main study used matched repeated holdouts and frozen protocols. Highlights
from the untouched 96-seed confirmations include:

- RKHS-RBF-OVO achieved lower mean error than the source-profile baseline on
  all seven confirmed datasets.
- Parkinson mean error decreased from **14.24% to 7.08%**.
- Breast Cancer Recurrence mean error decreased from **30.64% to 26.81%**,
  although minority recall fell—an important example of why accuracy alone is
  insufficient.
- On Blood Transfusion, the class-sensitive model increased balanced accuracy
  from **61.35% to 67.17%** and minority recall by **17.24 percentage points**,
  without a confirmed change in overall error.
- On Heart Disease, the class-sensitive model improved balanced accuracy and
  macro-F1 but increased overall error, exposing a clear balance/accuracy
  trade-off.

These results describe the included datasets and fixed protocol. Repeated
holdouts overlap, so reported intervals describe split variation rather than
independent population-level confidence intervals. See the
[final manuscript](paper/final_manuscript/SVM_FINAL_PAPER_REVISED_3.md) for the
complete interpretation and limitations.

## Project structure

```text
ddr_mksvm/                 Core models, optimization, evaluation, and CLI
configs/                   Reproducible experiment and seed configuration
dataset/                   Included tabular benchmark datasets
tests/                     Mathematical, solver, workflow, and regression tests
docs/                      Curated setup, architecture, protocol, and final reports
paper/final_manuscript/    Final paper, figures, evidence map, and build tools
scripts/                   Small local entry points
archive/                   Frozen historical and cloud-specific research code
```

The active package has no cloud-platform dependency. Historical Modal wrappers
are isolated under `archive/modal/` for provenance.

## Quick start

Python 3.11 or 3.12 is supported. The following example uses PowerShell:

```powershell
git clone https://github.com/dariushmmz/ddr-mksvm.git
cd ddr-mksvm
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Run a one-seed smoke experiment across the baseline and both supported models:

```powershell
python -m ddr_mksvm.cli smoke `
  --dataset breast_cancer_recurrence `
  --seed 17000 `
  --run-name quickstart
```

Generated experiment outputs and checkpoints are written under `results/`,
which is intentionally excluded from Git.

## Command-line workflows

```text
python -m ddr_mksvm.cli inventory     # rebuild the dataset inventory
python -m ddr_mksvm.cli smoke         # run a one-seed end-to-end check
python -m ddr_mksvm.cli v7 --help     # compare baseline and RKHS-RBF-OVO
python -m ddr_mksvm.cli v8c --help    # evaluate the class-sensitive extension
python -m ddr_mksvm.cli expanded --help
python -m ddr_mksvm.cli analyze --help
python -m ddr_mksvm.cli consolidate
```

Experiments use train-only preprocessing, stratified matched splits, explicit
seed registries, configuration fingerprints, resumable checkpoints, and
atomic result writes. Frozen-source hash tests protect confirmed model
definitions from accidental edits.

## Documentation

- [Setup and execution guide](docs/PROJECT_SETUP_AND_EXECUTION.md) — installation,
  datasets, experiments, resume behavior, and troubleshooting
- [Architecture history](docs/ARCHITECTURE_HISTORY.md) — design evolution and
  model status
- [Core specification](docs/specifications/DDR-MKSVM_SPECIFICATION.md) —
  mathematical and implementation requirements
- [RKHS-RBF-OVO evaluation](docs/V7_CROSS_DATASET_REPORT.md)
- [Class-sensitive evaluation](docs/V8_CROSS_DATASET_REPORT.md)
- [Robust research conclusion](docs/robust/ROBUST_RESEARCH_FINAL_REPORT.md)
- [Expanded-dataset protocol and results](docs/expanded_dataset_study/)
- [Final manuscript](paper/final_manuscript/SVM_FINAL_PAPER_REVISED_3.md)

Only final or operationally important documents are included in the public
documentation set; client correspondence, working prompts, duplicate exports,
and intermediate drafts are excluded.

## Quality and reproducibility

Run the complete regression suite with:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q
```

The tests cover kernel mathematics, SVM formulations, robust solver policies,
data preparation, checkpoint recovery, CLI-facing workflows, frozen-source
integrity, and experiment analysis. GitHub Actions runs the same suite on each
push and pull request.

## Technology

Python, NumPy, SciPy, pandas, scikit-learn, CVXPY, PyTorch, pytest, TOML, and
GitHub Actions.

## Scientific status

RKHS-RBF-OVO and CS-RKHS-RBF-OVO are the supported deterministic models. The
robust candidates are retained as controlled negative evidence and are not
active production architectures. Reserved confirmation seeds should not be
reused without a new prospective protocol.
