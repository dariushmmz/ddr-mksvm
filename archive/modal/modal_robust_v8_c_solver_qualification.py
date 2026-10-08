"""CPU-only Modal launcher for controlled Robust V8-C solver qualification."""
from __future__ import annotations

import json
import os
import subprocess

import modal


app = modal.App("ddr-mksvm-robust-v8-c-solver-qualification")
volume = modal.Volume.from_name("ddr-mksvm-robust-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.15.2",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "joblib==1.4.2",
        "cvxpy==1.6.7",
        "clarabel==0.11.1",
        "scs==3.2.8",
        "cvxopt==1.3.2",
    )
    .add_local_dir("ddr_mksvm", remote_path="/opt/project/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path="/opt/project/dataset", copy=True)
    .add_local_file("run_robust_matlab_parity.py", remote_path="/opt/project/run_robust_matlab_parity.py", copy=True)
    .add_local_file("qualify_robust_v8_c_solver.py", remote_path="/opt/project/qualify_robust_v8_c_solver.py", copy=True)
)


def _git_identity() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return f"{commit} dirty={dirty}"
    except Exception:
        return "unavailable"


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def qualify_seed(dataset: str, seed: int, attempts: str, max_calls: int | None,
                 ordinals: str, factor_mode: str, representation_mode: str,
                 objective_scale_mode: str, phase: str, alpha_rule: str, c: float,
                 run_name: str, local_git: str) -> dict:
    volume.reload()
    env = os.environ.copy()
    env.update(
        MODAL_IS_REMOTE="1",
        MODAL_APP_ID=app.app_id,
        ROBUST_LOCAL_GIT=local_git,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    command = [
        "python", "qualify_robust_v8_c_solver.py",
        "--dataset", dataset,
        "--seed", str(seed),
        "--attempts", attempts,
        "--run-name", run_name,
        "--output-root", "/artifacts/solver_qualification",
    ]
    if max_calls is not None:
        command.extend(["--max-calls", str(max_calls)])
    if ordinals:
        command.extend(["--ordinals", ordinals])
    command.extend(["--factor-mode", factor_mode])
    command.extend(["--representation-mode", representation_mode])
    command.extend(["--objective-scale-mode", objective_scale_mode])
    command.extend(["--phase", phase, "--alpha-rule", alpha_rule, "--C", str(c)])
    try:
        result = subprocess.run(command, cwd="/opt/project", env=env, text=True, check=True)
    finally:
        volume.commit()
    return {"dataset": dataset, "seed": seed, "returncode": result.returncode, "app_id": app.app_id}


@app.local_entrypoint()
def main(
    dataset: str = "blood_transfusion",
    seeds: str = "15000,15001",
    attempts: str = "clarabel_strict_original",
    max_calls: int | None = None,
    ordinals: str = "",
    factor_mode: str = "full",
    representation_mode: str = "sample",
    objective_scale_mode: str = "none",
    phase: str = "inner",
    alpha_rule: str = "paper",
    c: float = 0.1,
    run_name: str = "blood/sq-baseline",
    detach: bool = False,
):
    parsed_seeds = [int(value) for value in seeds.split(",") if value]
    if not parsed_seeds:
        raise ValueError("at least one seed is required")
    if len(parsed_seeds) > 2:
        raise ValueError("solver qualification is limited to two exposed seeds")
    local_git = _git_identity()
    calls = [qualify_seed.spawn(dataset, seed, attempts, max_calls, ordinals, factor_mode, representation_mode, objective_scale_mode, phase, alpha_rule, c, run_name, local_git) for seed in parsed_seeds]
    print(json.dumps({"app_id": app.app_id, "calls": [call.object_id for call in calls]}), flush=True)
    if not detach:
        for call in calls:
            print(call.get(), flush=True)
