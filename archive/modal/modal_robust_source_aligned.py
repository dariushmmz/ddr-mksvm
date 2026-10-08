"""Detached Modal entry point for the fixed R8 exposed-seed qualification."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("ddr-mksvm-robust-source-aligned-r8")
volume = modal.Volume.from_name("ddr-mksvm-robust-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.15.2",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "joblib==1.4.2",
    )
    .add_local_dir("ddr_mksvm", remote_path="/opt/project/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path="/opt/project/dataset", copy=True)
    .add_local_file(
        "run_robust_matlab_parity.py",
        remote_path="/opt/project/run_robust_matlab_parity.py",
        copy=True,
    )
    .add_local_file(
        "run_robust_source_aligned.py",
        remote_path="/opt/project/run_robust_source_aligned.py",
        copy=True,
    )
    .add_local_file(
        "analyze_robust_source_aligned.py",
        remote_path="/opt/project/analyze_robust_source_aligned.py",
        copy=True,
    )
    .add_local_file(
        "modal_robust_source_aligned.py",
        remote_path="/opt/project/modal_robust_source_aligned.py",
        copy=True,
    )
)


def execute(mode, run_name, local_git, task_start=None, task_stop=None):
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MODAL_IS_REMOTE="1",
        MODAL_APP_ID=app.app_id,
        ROBUST_LOCAL_GIT=local_git,
    )
    command = [
        "python",
        "run_robust_source_aligned.py",
        "--run-name",
        run_name,
        "--output-root",
        "/artifacts",
        "--mode",
        mode,
    ]
    if task_start is not None:
        command.extend(["--task-start", str(task_start), "--task-stop", str(task_stop)])
    result = subprocess.run(
        command,
        cwd="/opt/project",
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=300,
    volumes={"/artifacts": volume},
    retries=1,
)
def prepare_remote(run_name, local_git):
    volume.reload()
    try:
        result = execute("prepare", run_name, local_git)
    finally:
        volume.commit()
    return json.loads(result)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    max_containers=4,
    volumes={"/artifacts": volume},
    retries=1,
)
def run_lane(run_name, local_git, task_start, task_count, lane, lane_count):
    task_stop = min(task_start + 1, task_count)
    volume.reload()
    try:
        result = execute(
            "worker", run_name, local_git, task_start=task_start, task_stop=task_stop
        )
    finally:
        volume.commit()
    next_start = task_start + lane_count
    if next_start < task_count:
        call = run_lane.spawn(
            run_name, local_git, next_start, task_count, lane, lane_count
        )
        print(f"lane {lane} continued as {call.object_id}", flush=True)
    else:
        marker = Path("/artifacts") / run_name / "lane_checkpoints" / f"lane-{lane}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"lane": lane, "task_count": task_count}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, marker)
        volume.commit()
        volume.reload()
        if all((marker.parent / f"lane-{value}.json").exists() for value in range(lane_count)):
            call = finalize_remote.spawn(run_name, local_git)
            print(f"all lanes complete; finalizer {call.object_id}", flush=True)
    return result


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=600,
    volumes={"/artifacts": volume},
    retries=1,
)
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
    run_name: str = "source_aligned_redesign/exposed_qualification/r8-q1-cs-p2-rho001-exposed-v1",
    n_jobs: int = 4,
):
    if not 1 <= n_jobs <= 4:
        raise ValueError("n_jobs must be in [1,4]")
    local_git = git_identity()
    state = prepare_remote.remote(run_name, local_git)
    print(state, flush=True)
    if state["status"] == "complete":
        return
    task_count = int(state["tasks"])
    lane_count = min(n_jobs, task_count)
    for lane in range(lane_count):
        call = run_lane.spawn(
            run_name, local_git, lane, task_count, lane, lane_count
        )
        print(f"spawned lane {lane}: {call.object_id}", flush=True)
    print("Detached exposed-seed qualification is checkpointed and running.", flush=True)
