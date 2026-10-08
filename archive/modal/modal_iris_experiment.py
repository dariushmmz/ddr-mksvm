"""Modal wrapper for staged, CPU-only Iris q=1 research experiments."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import modal


APP_NAME = "ddr-mksvm-iris"
VOLUME_NAME = "ddr-mksvm-iris-results"
REMOTE_PROJECT = "/opt/ddr_project"
REMOTE_RESULTS = "/artifacts"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.24,<2.0", "scipy>=1.10", "pandas>=1.5",
        "scikit-learn>=1.2", "joblib>=1.2", "cvxpy>=1.6.5,<1.7", "cvxopt>=1.3.2",
    )
    .add_local_dir("ddr_mksvm", remote_path=f"{REMOTE_PROJECT}/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path=f"{REMOTE_PROJECT}/dataset", copy=True)
    .add_local_file("run_iris_experiments.py", remote_path=f"{REMOTE_PROJECT}/run_iris_experiments.py", copy=True)
)


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=60 * 60 * 6,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={REMOTE_RESULTS: volume},
)
def run_experiment(models: str, n_seeds: int, n_jobs: int, run_name: str,
                   seed_start: int = 0) -> str:
    output_root = Path(REMOTE_RESULTS)
    command = [
        "python", "run_iris_experiments.py", "--models", models,
        "--n-seeds", str(n_seeds), "--n-jobs", str(n_jobs),
        "--run-name", run_name, "--output-root", str(output_root),
        "--seed-start", str(seed_start),
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "IRIS_MODAL_RESOURCES": "4 CPU; 8192 MiB; no GPU",
    })
    print("[modal] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REMOTE_PROJECT, env=environment, check=True)
    volume.commit()
    return str(output_root / run_name)


@app.local_entrypoint()
def main(models: str = "legacy_grid_ova,exact_b_ova,alpha_cv_ova,nu_cv_ova,legacy_grid_ovo",
         n_seeds: int = 1, n_jobs: int = 1, run_name: str = "iris-smoke",
         seed_start: int = 0):
    path = run_experiment.remote(models, n_seeds, n_jobs, run_name, seed_start)
    print(f"Persistent Modal result path: {path}")
