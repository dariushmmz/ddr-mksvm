"""One-off Modal diagnostic entry point that preserves child stderr."""
from __future__ import annotations

import os
import subprocess

import modal

app = modal.App("ddr-mksvm-robust-v8-c-diagnose")
volume = modal.Volume.from_name("ddr-mksvm-robust-results", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==1.26.4", "scipy==1.15.2", "pandas==2.2.3",
                 "scikit-learn==1.6.1", "joblib==1.4.2", "cvxpy==1.6.7",
                 "clarabel==0.11.1")
    .add_local_dir("ddr_mksvm", remote_path="/opt/project/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path="/opt/project/dataset", copy=True)
    .add_local_file("run_robust_matlab_parity.py", remote_path="/opt/project/run_robust_matlab_parity.py", copy=True)
    .add_local_file("run_robust_v8_c.py", remote_path="/opt/project/run_robust_v8_c.py", copy=True)
    .add_local_file("modal_robust_v8_c.py", remote_path="/opt/project/modal_robust_v8_c.py", copy=True)
    .add_local_file("analyze_robust_v8_c.py", remote_path="/opt/project/analyze_robust_v8_c.py", copy=True))


@app.function(image=image, cpu=4, memory=8192, timeout=3600,
              volumes={"/artifacts": volume})
def diagnose():
    env = os.environ.copy()
    env.update(MODAL_IS_REMOTE="1", MODAL_APP_ID=app.app_id,
               ROBUST_LOCAL_GIT="diagnostic", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = ["python", "run_robust_v8_c.py", "--datasets", "blood_transfusion",
           "--models", "v8_c,robust_v8_c_p2", "--rhos", "0.01",
           "--n-seeds", "1", "--seed-start", "15000", "--n-jobs", "1",
           "--run-name", "debug/blood-rho001-seed15000-20260917",
           "--output-root", "/artifacts"]
    try:
        subprocess.run(cmd, cwd="/opt/project", env=env, check=True)
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    diagnose.remote()
