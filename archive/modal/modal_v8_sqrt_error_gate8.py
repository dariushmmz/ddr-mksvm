"""Modal entry point for the requested square-root/error-selection Gate8."""
from __future__ import annotations

import os
import subprocess

import modal


app = modal.App("ddr-mksvm-v8-sqrt-error-gate8")
volume = modal.Volume.from_name("ddr-mksvm-v8-class-sensitive-results", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.17.1",
        "pandas==3.0.5",
        "scikit-learn==1.9.0",
        "joblib==1.6.0",
        "cvxpy==1.6.7",
    )
    .add_local_dir("ddr_mksvm", remote_path="/opt/project/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path="/opt/project/dataset", copy=True)
)


@app.function(image=image, cpu=4, memory=4096, timeout=1800, volumes={"/artifacts": volume})
def run(local_git: str) -> str:
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        V8_LOCAL_GIT=local_git,
        MODAL_APP_ID=app.app_id,
    )
    try:
        subprocess.run(
            [
                "python",
                "-m",
                "ddr_mksvm.experiments.gate8_sqrt_error",
                "--output-root",
                "/artifacts",
                "--prior-root",
                "/artifacts/gate8/v8-gate8-20260909",
                "--workers",
                "4",
            ],
            cwd="/opt/project",
            env=environment,
            check=True,
        )
    finally:
        volume.commit()
    return "/artifacts/gate8/v8-sqrt-error-gate8-20261005-v1"


@app.local_entrypoint()
def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    print(run.remote(f"{commit} dirty={dirty}"))

