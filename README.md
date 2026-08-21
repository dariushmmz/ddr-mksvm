# DDR-MKSVM

DDR-MKSVM is a Python implementation of a distributionally robust, deep,
multiple-kernel SVM. It supports binary and multiclass classification,
four ablation modes, parallel repeated holdout runs, and crash-safe resume
checkpoints.

The mathematical design is documented in [DDR-MKSVM_spec.md](DDR-MKSVM_spec.md).
The implementation reuses the deterministic dataset registry, transforms, and
kernel parsing so the robust and baseline paths stay aligned.

## Features

- Binary and one-vs-all multiclass training.
- Dataset-specific preprocessing and kernels.
- Deep feature learning, multiple-kernel learning, and Wasserstein-DRO modes.
- The paper protocol of 96 independent 75/25 stratified holdout runs.
- Atomic per-run checkpoints and interrupted-run recovery.
- Configurable, container-quota-aware parallel workers.
- Binary missing-value handling for raw values such as `?` and `NaN`.
- MATLAB and CSV result exports.
- Deterministic baseline scripts for comparison with the `legacy` ablation.

## Requirements

- Python 3.10 or newer is recommended.
- A CPU environment is sufficient. The current implementation does not move
  PyTorch models or CVXPY solvers to CUDA, so attaching a GPU does not currently
  accelerate training.
- Enough memory for the selected worker count. Every worker owns a separate
  kernel matrix and solver process and may also own a PyTorch model.

The Python packages and supported version ranges are listed in
[requirements.txt](requirements.txt).

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/dariushmmz/ddr-mksvm.git
cd ddr-mksvm
```

Create and activate a virtual environment on Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -m pytest -q
```

The current test suite has 30 tests covering the mathematical lemmas, solver
formulations, legacy reduction, checkpoint recovery, preprocessing, and runtime
worker selection.

### Hosted notebook setup

In Jupyter/Colab-style notebooks, the equivalent setup is:

```python
!git clone https://github.com/dariushmmz/ddr-mksvm.git
%cd ddr-mksvm
!python -m pip install -r requirements.txt
!python -m pytest -q
```

Upload the required CSV into `dataset/`, then run the command from the quick
start section below. Set `--n_jobs` explicitly to the notebook's allocated CPU
count; if memory is limited, use half that number. A GPU is not used by the
current implementation.

For long jobs, keep the repository or at least `results/` on persistent
storage. Checkpoints stored only on an ephemeral notebook filesystem disappear
when the notebook instance is deleted.

## Dataset setup

Place each CSV directly in `dataset/`. Feature columns must come first and the
class label must be the final column.

Binary labels must be numeric `0` and `1`. Binary feature columns are converted
to numeric values; missing, non-numeric, and infinite feature values are
mean-imputed. The driver prints the number of replacements per affected column.
Missing labels and entirely empty feature columns are rejected.

Multiclass labels must be integers `1..L`. The final label column is normalized
internally to the name `CLASS`. Multiclass CSVs must already contain finite
numeric features and labels.

Only `parkinson.csv` is currently tracked in Git. Supply the other CSVs before
selecting their dataset names.

### Binary datasets

| `--dataset` value | Expected CSV | Transform | Kernel |
| --- | --- | --- | --- |
| `arrhythmia` | `arrhythmia.csv` | none | Gaussian RBF |
| `parkinson` | `parkinson.csv` | min-max | homogeneous linear |
| `climate_model_crashes` | `climate_model_crashes.csv` | none | homogeneous linear |
| `breast_cancer_diagnostic` | `breast_cancer_diagnostic.csv` | min-max | inhomogeneous quadratic |
| `breast_cancer` | `breast_cancer.csv` | standardization | homogeneous linear |
| `blood_transfusion` | `blood_transfusion.csv` | standardization | inhomogeneous cubic |
| `mammographicmass_binary` | `mammographicmass_binary.csv` | standardization | inhomogeneous quadratic |
| `qsar_biodegradation` | `qsar_biodegradation.csv` | min-max | Gaussian RBF |

### Multiclass datasets

| `--dataset` value | Expected CSV | Classes | Transform | Kernel |
| --- | --- | ---: | --- | --- |
| `iris` | `iris_multiclass.csv` | 3 | none | Gaussian RBF |
| `wine` | `wine.csv` | 3 | standardization | inhomogeneous linear |
| `heart_disease` | `heart_disease.csv` | 5 | standardization | inhomogeneous linear |
| `dermatology` | `dermatology.csv` | 6 | none | inhomogeneous quadratic |

The registries in `unit_of_work_deterministic_binary.py` and
`unit_of_work_deterministic_multiclass.py` are the authoritative source for
dataset filenames, transforms, kernels, and class counts.

## Quick start

Start with a small smoke test before launching the 96-run protocol:

```bash
python main_ddr_binary.py \
  --dataset parkinson \
  --ablation full \
  --n_runs 2 \
  --n_jobs 2 \
  --seeded \
  --resume
```

For the long Mammographic Mass experiment on a 16-vCPU notebook:

```bash
python main_ddr_binary.py \
  --dataset mammographicmass_binary \
  --ablation all \
  --n_runs 96 \
  --n_jobs 16 \
  --resume
```

If the notebook has limited memory, begin with `--n_jobs 8`. Changing
`--n_jobs` between interrupted and resumed invocations is safe.

A multiclass example:

```bash
python main_ddr_multiclass.py \
  --dataset iris \
  --ablation all \
  --n_runs 96 \
  --n_jobs 8 \
  --resume
```

Use `python main_ddr_binary.py --help` or
`python main_ddr_multiclass.py --help` to see the current CLI.

## Ablation modes

| Mode | Deep features | Multiple kernels | DRO penalty | Purpose |
| --- | ---: | ---: | ---: | --- |
| `legacy` | off | off | off | Deterministic q=1 baseline/reduction check |
| `dro_only` | off | off | on | Fixed dataset kernel with Wasserstein-DRO |
| `dnn_dro` | on | off | on | Learned representation with one kernel and DRO |
| `full` | on | on | on | Full DDR-MKSVM with the dataset kernel plus an RBF kernel |
| `all` | — | — | — | Runs `legacy`, `dro_only`, `dnn_dro`, then `full` |

## Command-line options

| Option | Meaning |
| --- | --- |
| `--dataset NAME` | Dataset registry key. Binary defaults to `mammographicmass_binary`; multiclass defaults to `iris`. |
| `--ablation MODE` | One ablation or `all`. Defaults to `full`. |
| `--n_runs N` | Number of independent stratified holdout runs. Defaults to 96. |
| `--n_jobs N` | Concurrent workers. The default and `-1` use the detected CPU/container quota. |
| `--seeded` | Uses run index `i` as the split and model seed, enabling paired comparisons across ablations. |
| `--resume` | Loads compatible completed-run checkpoints and executes only unfinished runs. |

Without `--seeded`, the first invocation creates and stores an independent
random seed for every run. Those seeds are reused by `--resume`, but are not
shared between ablations. With `--seeded`, run `i` uses the same split and
network initialization across scripts and ablations, which is useful for
paired comparisons.

## Resume and checkpoint behavior

Use `--resume` even on the first invocation of a long experiment. If no
manifest exists, it creates one and starts normally. Each worker atomically
saves its result immediately after completing one random split.

If training is interrupted, repeat the same command:

```bash
python main_ddr_binary.py --dataset mammographicmass_binary --ablation all --n_runs 96 --n_jobs 16 --resume
```

Completed runs are loaded and only pending runs are scheduled. Work that was
still in flight at interruption is repeated. Resume requires the same:

- dataset name and unchanged input CSV contents;
- ablation configuration;
- `--n_runs` value;
- `--seeded` setting.

The worker count may change. A configuration mismatch produces an explicit
error instead of combining incompatible results.

Running without `--resume` intentionally starts a fresh experiment and
replaces checkpoint files for each selected ablation as it is reached.

## Results

DDR-MKSVM outputs are written under:

```text
results/<dataset>/ddr_mksvm/
```

For binary experiments:

```text
ddr_mksvm_results.mat
ddr_mksvm_summary.csv
checkpoints/binary/<ablation>/manifest.json
checkpoints/binary/<ablation>/run_00000.npz
...
```

For multiclass experiments:

```text
ddr_mksvm_multiclass_results.mat
ddr_mksvm_multiclass_summary.csv
checkpoints/multiclass/<ablation>/manifest.json
checkpoints/multiclass/<ablation>/run_00000.npz
...
```

Aggregate `.mat` and summary CSV files are refreshed after each completed
ablation. This preserves visible results from early ablations if a later one
is interrupted. The `results/` directory is ignored by Git.

## Running the deterministic baselines

Use the baseline scripts to compare against the `legacy` ablation:

```bash
python main_deterministic_binary.py --dataset parkinson --n_runs 96 --seeded
python main_deterministic_multiclass.py --dataset iris --n_runs 96 --seeded
```

The deterministic drivers predate the DDR checkpoint manager: they do not
support `--resume` or `--n_jobs` and currently use all CPUs visible to joblib.
The DDR drivers are recommended for long ablation runs.

## Project structure

```text
ddr_mksvm/
  checkpointing.py              atomic per-run checkpoints and manifests
  config_adapter.py             dataset-kernel configuration bridge
  runtime.py                    CPU affinity/quota-aware worker selection
  dro/wasserstein_risk.py       closed-form Wasserstein-DRO risk
  kernels/base_kernels.py       linear, polynomial, and RBF kernels
  kernels/deep_kernel.py        learned and identity feature extractors
  kernels/mkl_combination.py    simplex-constrained kernel mixture
  lipschitz/                    analytic and empirical Lipschitz tools
  optim/alternating_trainer.py  binary and one-vs-all alternating training
  optim/convex_subproblem.py    q=2 SOCP and legacy q=1 LP

main_ddr_binary.py              binary DDR-MKSVM CLI
main_ddr_multiclass.py          multiclass DDR-MKSVM CLI
unit_of_work_ddr_binary.py      one binary holdout run
unit_of_work_ddr_multiclass.py  one multiclass holdout run
main_deterministic_*.py         deterministic comparison CLIs
unit_of_work_deterministic_*.py shared registries, transforms, and kernels
tests/                          mathematical, solver, resume, and data tests
dataset/                        input CSV files
results/                        generated outputs; ignored by Git
```

## Troubleshooting

### `ValueError: K/y contain NaN or Inf`

Pull the current version. Binary drivers now coerce and mean-impute invalid
feature values before standardization. The startup log lists affected columns.
For multiclass data, clean or impute the CSV first. Invalid labels and columns
with no usable values must be corrected in the source CSV.

### `optimal_inaccurate` or `solver violated xi >= 0`

`optimal_inaccurate` is a CVXPY fallback status that can occur for an
ill-conditioned learned Gram matrix during `dnn_dro` or `full`. The current
solver reconstructs the minimum feasible slack directly from the returned
margin, so a finite inaccurate candidate no longer terminates every parallel
run because of a small negative `xi`. Pull the latest code and repeat the same
training command with `--resume`; the unfinished split will be retried while
completed checkpoints are retained.

The warning can still appear and records that the preferred solver did not
produce a clean `optimal` status. Occasional warnings are recoverable; frequent
warnings indicate numerical conditioning problems and should be considered
when interpreting the affected run.

### Joblib starts more workers than the notebook CPU allocation

Set the limit explicitly, for example `--n_jobs 16`. The current default also
checks Linux cgroup quotas, but an explicit value is best on hosted platforms
that expose the physical host's CPU count.

### The process is killed or the notebook runs out of memory

Reduce the worker count, for example from `--n_jobs 16` to `--n_jobs 8` or
`--n_jobs 4`, then repeat the same command with `--resume`.

### Checkpoint configuration does not match

Use the same CSV, `--n_runs`, `--seeded`, dataset, and ablation settings as the
original run. If a fresh experiment is intended, omit `--resume`; this replaces
the selected ablation's old checkpoints.

### Dataset CSV not found

Check the dataset table above, place the file under `dataset/`, and make sure
its filename exactly matches the registry entry.

## Known limitations

- Training uses fixed `n_outer=6` and `n_inner=15` iteration counts instead of
  a convergence-based stopping rule.
- Resume granularity is one completed random split, not an inner/outer neural
  optimization iteration. An interrupted in-flight split is recomputed.
- PyTorch and CVXPY currently run on CPU; CUDA execution is not implemented.
- The polynomial-kernel Lipschitz bound is valid but conservative.
- DNN/MKL test evaluation recomputes base kernels instead of batching/caching
  for very large datasets.
- Multiclass training shares one learned representation and kernel mixture
  across all one-vs-all classifiers.
