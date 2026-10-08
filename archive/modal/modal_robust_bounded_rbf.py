"""Detached Modal entry point for the closed R9 exposed-seed registry."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("ddr-mksvm-robust-bounded-rbf-r9")
volume = modal.Volume.from_name("ddr-mksvm-robust-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==1.26.4", "scipy==1.15.2", "pandas==2.2.3", "scikit-learn==1.6.1", "joblib==1.4.2")
    .add_local_dir("ddr_mksvm", remote_path="/opt/project/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path="/opt/project/dataset", copy=True)
    .add_local_file("run_robust_matlab_parity.py", remote_path="/opt/project/run_robust_matlab_parity.py", copy=True)
    .add_local_file("run_robust_bounded_rbf.py", remote_path="/opt/project/run_robust_bounded_rbf.py", copy=True)
    .add_local_file("analyze_robust_bounded_rbf.py", remote_path="/opt/project/analyze_robust_bounded_rbf.py", copy=True)
    .add_local_file("modal_robust_bounded_rbf.py", remote_path="/opt/project/modal_robust_bounded_rbf.py", copy=True)
)


def execute(mode, run_name, local_git, start=None, stop=None):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MODAL_IS_REMOTE="1", MODAL_APP_ID=app.app_id, ROBUST_LOCAL_GIT=local_git)
    command = ["python", "run_robust_bounded_rbf.py", "--run-name", run_name, "--output-root", "/artifacts", "--mode", mode]
    if start is not None:
        command.extend(["--task-start", str(start), "--task-stop", str(stop)])
    result = subprocess.run(command, cwd="/opt/project", env=environment, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result.stdout.strip().splitlines()[-1]


@app.function(image=image, cpu=1, memory=1024, timeout=300, volumes={"/artifacts": volume}, retries=1)
def prepare_remote(run_name, local_git):
    volume.reload()
    try:
        return json.loads(execute("prepare", run_name, local_git))
    finally:
        volume.commit()


@app.function(image=image, cpu=4, memory=8192, timeout=3600, max_containers=4, volumes={"/artifacts": volume}, retries=1)
def run_lane(run_name, local_git, task, task_count, lane_count):
    volume.reload()
    try:
        output = execute("worker", run_name, local_git, task, min(task + 1, task_count))
    finally:
        volume.commit()
    next_task = task + lane_count
    if next_task < task_count:
        run_lane.spawn(run_name, local_git, next_task, task_count, lane_count)
    else:
        lane = task % lane_count
        marker = Path("/artifacts") / run_name / "lane_checkpoints" / f"lane-{lane}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"lane": lane}) + "\n", encoding="utf-8")
        os.replace(temporary, marker)
        volume.commit()
        volume.reload()
        if all((marker.parent / f"lane-{value}.json").exists() for value in range(lane_count)):
            finalize_remote.spawn(run_name, local_git)
    return output


@app.function(image=image, cpu=1, memory=2048, timeout=600, volumes={"/artifacts": volume}, retries=1)
def finalize_remote(run_name, local_git):
    volume.reload()
    try:
        return execute("finalize", run_name, local_git)
    finally:
        volume.commit()


def git_identity():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return f"{commit} dirty={dirty}"
    except Exception:
        return "unavailable"


@app.local_entrypoint()
def main(
    run_name: str = "bounded_rbf_redesign/exposed_qualification/r9-rbf-q1-p2-rho001-exposed-v1",
    n_jobs: int = 4,
):
    if not 1 <= n_jobs <= 4:
        raise ValueError("n_jobs must be in [1,4]")
    local_git = git_identity()
    state = prepare_remote.remote(run_name, local_git)
    print(state, flush=True)
    task_count = int(state["tasks"])
    lane_count = min(n_jobs, task_count)
    for lane in range(lane_count):
        call = run_lane.spawn(run_name, local_git, lane, task_count, lane_count)
        print(f"spawned lane {lane}: {call.object_id}", flush=True)
    print("Detached R9 exposed qualification is checkpointed and running.", flush=True)
