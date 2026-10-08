"""Modal wrapper for raw-Gram-cached warm-started adaptive V3."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "ddr-mksvm-cached-warm-started-v3"
VOLUME_NAME = "ddr-mksvm-v3-results"
REMOTE_PROJECT = "/opt/ddr_project"
REMOTE_RESULTS = "/artifacts"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("ddr_mksvm", remote_path=f"{REMOTE_PROJECT}/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path=f"{REMOTE_PROJECT}/dataset", copy=True)
    .add_local_file("holdouts_train_test.py", remote_path=f"{REMOTE_PROJECT}/holdouts_train_test.py", copy=True)
    .add_local_file("holdouts_train_test_multiclass.py", remote_path=f"{REMOTE_PROJECT}/holdouts_train_test_multiclass.py", copy=True)
    .add_local_file("unit_of_work_deterministic_binary.py", remote_path=f"{REMOTE_PROJECT}/unit_of_work_deterministic_binary.py", copy=True)
    .add_local_file("unit_of_work_deterministic_multiclass.py", remote_path=f"{REMOTE_PROJECT}/unit_of_work_deterministic_multiclass.py", copy=True)
    .add_local_file("unit_of_work_ddr_binary.py", remote_path=f"{REMOTE_PROJECT}/unit_of_work_ddr_binary.py", copy=True)
    .add_local_file("unit_of_work_ddr_multiclass.py", remote_path=f"{REMOTE_PROJECT}/unit_of_work_ddr_multiclass.py", copy=True)
    .add_local_file("run_v3_experiment.py", remote_path=f"{REMOTE_PROJECT}/run_v3_experiment.py", copy=True)
    .add_local_file("run_cached_warm_started_v3_experiment.py", remote_path=f"{REMOTE_PROJECT}/run_cached_warm_started_v3_experiment.py", copy=True)
)


@app.function(
    image=image,
    cpu=16.0,
    memory=32768,
    timeout=60 * 60 * 12,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={REMOTE_RESULTS: volume},
)
def run_experiment(
    datasets: str = "parkinson,blood_transfusion,mammographicmass_binary,iris",
    n_seeds: int = 1,
    n_jobs: int = 1,
    n_outer: int = 1,
    n_inner: int = 1,
    run_name: str = "smoke-cached-warm-started-v3",
) -> str:
    selected = [value.strip() for value in datasets.split(",") if value.strip()]
    output_root = Path(REMOTE_RESULTS) / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = output_root / "experiment_metadata.txt"
    metadata_text = (
        f"run_name={run_name}\ndatasets={','.join(selected)}\nn_seeds={n_seeds}\n"
        f"n_jobs={n_jobs}\nn_outer={n_outer}\nn_inner={n_inner}\n"
        "variants=legacy,adaptive_warm_start_v3,adaptive_warm_cache_v3\n"
        "change=cache_invariant_raw_anchor_training_gram_once_per_fit\n"
        "cpu=16\nmemory_mb=32768\ngpu=none\n"
    )
    if metadata.exists() and metadata.read_text(encoding="utf-8") != metadata_text:
        raise ValueError(
            f"run name {run_name!r} already has different experiment metadata; "
            "choose a new run name rather than mixing configurations"
        )
    metadata.write_text(metadata_text, encoding="utf-8")
    volume.commit()

    for dataset in selected:
        dataset_output = output_root / dataset
        summary = dataset_output / "summary.csv"
        per_run = dataset_output / "per_run.csv"
        if summary.exists() and per_run.exists():
            print(f"[resume] {dataset}: {summary} already exists")
            continue
        command = [
            "python", "run_cached_warm_started_v3_experiment.py",
            "--datasets", dataset,
            "--n-seeds", str(n_seeds),
            "--n-jobs", str(n_jobs),
            "--n-outer", str(n_outer),
            "--n-inner", str(n_inner),
            "--output", str(dataset_output),
        ]
        print("[modal]", " ".join(command), flush=True)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        log_path = dataset_output / "experiment.log"
        dataset_output.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=REMOTE_PROJECT, check=True, env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        print(log_path.read_text(encoding="utf-8"), flush=True)
        volume.commit()
    return str(output_root)


@app.local_entrypoint()
def main(
    datasets: str = "parkinson,blood_transfusion,mammographicmass_binary,iris",
    n_seeds: int = 1,
    n_jobs: int = 1,
    n_outer: int = 1,
    n_inner: int = 1,
    run_name: str = "smoke-cached-warm-started-v3",
):
    result = run_experiment.remote(
        datasets, n_seeds=n_seeds, n_jobs=n_jobs,
        n_outer=n_outer, n_inner=n_inner, run_name=run_name,
    )
    print(f"Persistent Modal result path: {result}")
