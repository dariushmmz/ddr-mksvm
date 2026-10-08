"""Modal-only runner for the expanded frozen-model dataset study."""
import os
import subprocess

import modal


app = modal.App("ddr-mksvm-expanded-dataset-study")
volume = modal.Volume.from_name("ddr-mksvm-expanded-dataset-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==1.26.4", "scipy==1.17.1", "pandas==3.0.5",
        "scikit-learn==1.9.0", "joblib==1.6.0", "cvxpy==1.6.7",
    )
    .add_local_dir("ddr_mksvm", remote_path="/opt/project/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path="/opt/project/dataset", copy=True)
    .add_local_file("run_expanded_dataset_study.py", remote_path="/opt/project/run_expanded_dataset_study.py", copy=True)
    .add_local_file("modal_expanded_dataset_study.py", remote_path="/opt/project/modal_expanded_dataset_study.py", copy=True)
)


@app.function(image=image, cpu=4, memory=8192, timeout=7200, volumes={"/artifacts": volume})
def run(datasets, models, stage, n_seeds, seed_start, n_jobs, run_name, local_git, promotion_source):
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "EXPANDED_MODAL_RUN": "1", "EXPANDED_LOCAL_GIT": local_git,
        "MODAL_APP_ID": app.app_id,
    })
    command = [
        "python", "run_expanded_dataset_study.py",
        "--datasets", datasets, "--models", models, "--stage", stage,
        "--n-seeds", str(n_seeds), "--seed-start", str(seed_start),
        "--n-jobs", str(n_jobs), "--run-name", run_name,
        "--output-root", "/artifacts",
    ]
    if promotion_source:
        command += ["--promotion-source", "/artifacts/" + promotion_source]
    try:
        subprocess.run(command, cwd="/opt/project", env=env, check=True)
    finally:
        volume.commit()
    return "/artifacts/" + run_name


@app.local_entrypoint()
def main(
    datasets: str = "breast_cancer_recurrence,dermatology",
    models: str = "legacy,v7,v8_c",
    stage: str = "smoke",
    n_seeds: int = 1,
    seed_start: int = 16000,
    n_jobs: int = 2,
    run_name: str = "smoke/expanded-smoke-16000-v1",
    promotion_source: str = "",
):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    print(run.remote(datasets, models, stage, n_seeds, seed_start, n_jobs, run_name,
                     commit + " dirty=" + str(dirty), promotion_source))
