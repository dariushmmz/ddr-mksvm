"""CPU-only detached Modal entry point for sharded Robust V8-C runs."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("ddr-mksvm-robust-v8-c")
volume = modal.Volume.from_name("ddr-mksvm-robust-results", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==1.26.4", "scipy==1.15.2", "pandas==2.2.3", "scikit-learn==1.6.1",
                 "joblib==1.4.2", "cvxpy==1.6.7", "clarabel==0.11.1")
    .add_local_dir("ddr_mksvm", remote_path="/opt/project/ddr_mksvm", copy=True)
    .add_local_dir("dataset", remote_path="/opt/project/dataset", copy=True)
    .add_local_file("run_robust_matlab_parity.py", remote_path="/opt/project/run_robust_matlab_parity.py", copy=True)
    .add_local_file("run_robust_v8_c.py", remote_path="/opt/project/run_robust_v8_c.py", copy=True)
    .add_local_file("analyze_robust_v8_c.py", remote_path="/opt/project/analyze_robust_v8_c.py", copy=True)
    .add_local_file("modal_robust_v8_c.py", remote_path="/opt/project/modal_robust_v8_c.py", copy=True))


def _execute(mode, datasets, models, rhos, n_seeds, seed_start, run_name,
             local_git, task_start=None, task_stop=None):
    env = os.environ.copy()
    env.update(MODAL_IS_REMOTE="1", MODAL_APP_ID=app.app_id,
               ROBUST_LOCAL_GIT=local_git, OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = ["python", "run_robust_v8_c.py", "--datasets", datasets,
           "--models", models, "--rhos", rhos, "--n-seeds", str(n_seeds),
           "--seed-start", str(seed_start), "--n-jobs", "1", "--run-name",
           run_name, "--output-root", "/artifacts", "--mode", mode]
    if task_start is not None:
        cmd.extend(["--task-start", str(task_start), "--task-stop", str(task_stop)])
    result = subprocess.run(cmd, cwd="/opt/project", env=env,
                            text=True, capture_output=True)
    if result.returncode:
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", flush=True)
        result.check_returncode()
    if result.stdout:
        print(result.stdout, end="", flush=True)
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


@app.function(image=image, cpu=1, memory=1024, timeout=300,
              volumes={"/artifacts": volume}, retries=1)
def prepare_remote(datasets, models, rhos, n_seeds, seed_start, run_name, local_git):
    volume.reload()
    try:
        last = _execute("prepare", datasets, models, rhos, n_seeds, seed_start,
                        run_name, local_git)
    finally:
        volume.commit()
    return json.loads(last)


@app.function(image=image, cpu=4, memory=8192, timeout=3600, max_containers=4,
              volumes={"/artifacts": volume}, retries=1)
def run_shard(datasets, models, rhos, n_seeds, seed_start, run_name, local_git,
              task_start, shard_size, task_count, lane, lane_count):
    task_stop = min(task_start + shard_size, task_count)
    volume.reload()
    try:
        result = _execute("worker", datasets, models, rhos, n_seeds, seed_start,
                          run_name, local_git, task_start, task_stop)
    finally:
        volume.commit()
    next_start = task_start + shard_size * lane_count
    if next_start < task_count:
        call = run_shard.spawn(datasets, models, rhos, n_seeds, seed_start,
                               run_name, local_git, next_start, shard_size,
                               task_count, lane, lane_count)
        print(f"lane {lane} continued as {call.object_id}", flush=True)
    else:
        marker = Path("/artifacts") / run_name / "lane_checkpoints" / f"lane-{lane}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"lane": lane, "lane_count": lane_count,
                                   "task_count": task_count}, sort_keys=True) + "\n")
        os.replace(tmp, marker)
        volume.commit(); volume.reload()
        if all((marker.parent / f"lane-{value}.json").exists() for value in range(lane_count)):
            call = finalize_remote.spawn(datasets, models, rhos, n_seeds,
                                         seed_start, run_name, local_git)
            print(f"all lanes complete; finalizer {call.object_id} spawned", flush=True)
    return result


@app.function(image=image, cpu=1, memory=2048, timeout=600,
              volumes={"/artifacts": volume}, retries=1)
def finalize_remote(datasets, models, rhos, n_seeds, seed_start, run_name, local_git):
    volume.reload()
    try:
        return _execute("finalize", datasets, models, rhos, n_seeds, seed_start,
                        run_name, local_git)
    finally:
        volume.commit()


def _git_identity():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return f"{commit} dirty={dirty}"
    except Exception:
        return "unavailable"


@app.local_entrypoint()
def main(datasets: str="parkinson,iris", models: str="v8_c,robust_v8_c_p2",
         rhos: str="1e-4", n_seeds: int=1, seed_start: int=13000,
         n_jobs: int=4, shard_size: int=1,
         run_name: str="smoke/robust-v8-c-smoke-20260916"):
    if not 1 <= n_jobs <= 4:
        raise ValueError("n_jobs must be between 1 and 4")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    local_git = _git_identity()
    state = prepare_remote.remote(datasets, models, rhos, n_seeds, seed_start,
                                  run_name, local_git)
    print(state, flush=True)
    if state["status"] == "complete":
        return
    task_count = int(state["tasks"])
    lane_count = min(n_jobs, (task_count + shard_size - 1) // shard_size)
    for lane in range(lane_count):
        call = run_shard.spawn(datasets, models, rhos, n_seeds, seed_start,
                               run_name, local_git, lane * shard_size,
                               shard_size, task_count, lane, lane_count)
        print(f"spawned lane {lane}: {call.object_id}", flush=True)
    print("Detached lane chains will checkpoint, self-continue, and finalize remotely.", flush=True)
