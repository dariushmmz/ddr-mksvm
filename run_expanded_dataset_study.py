"""Checkpointed orchestration for the expanded frozen-model dataset study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ddr_mksvm import expanded_dataset_study as study


CODE = [
    "ddr_mksvm/v7_cross_dataset.py",
    "ddr_mksvm/v8_class_sensitive.py",
    "ddr_mksvm/iris_research.py",
    "ddr_mksvm/expanded_dataset_study.py",
    "run_expanded_dataset_study.py",
]


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _safe_job(name, seed, models, checkpoint_dir, failure_dir, fingerprint):
    checkpoint = Path(checkpoint_dir) / f"{name}-{seed}.json"
    failure = Path(failure_dir) / f"{name}-{seed}.json"
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise RuntimeError(f"incompatible checkpoint {checkpoint}")
        return {"kind": "result", "value": result}
    if failure.exists():
        result = json.loads(failure.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise RuntimeError(f"incompatible failure {failure}")
        return {"kind": "failure", "value": result}
    started = time.perf_counter()
    try:
        X, y, spec = study.load_dataset(name)
        result = study.evaluate_seed(name, X, y, spec, seed, models)
        result.update({
            "fingerprint": fingerprint,
            "models": list(models),
            "app_id": "local",
            "dataset_spec": spec,
        })
        atomic_json(checkpoint, result)
        print(f"complete {name} {seed} {result['actual_job_wall_s']:.3f}s", flush=True)
        return {"kind": "result", "value": result}
    except Exception as exc:
        record = {
            "dataset": name,
            "seed": int(seed),
            "fingerprint": fingerprint,
            "app_id": "local",
            "runtime_s": time.perf_counter() - started,
            "exception_type": type(exc).__name__,
            "exception_text": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(failure, record)
        print(f"FAILED {name} {seed}: {type(exc).__name__}: {exc}", flush=True)
        return {"kind": "failure", "value": record}


def _config(args, names, models, seeds):
    hashes = {path: digest(path) for path in CODE}
    datasets = {name: study.load_dataset(name)[2] for name in names}
    versions = {package: importlib.metadata.version(package) for package in (
        "numpy", "scipy", "pandas", "scikit-learn", "cvxpy", "joblib"
    )}
    promotion = None
    if args.promotion_source:
        source = Path(args.promotion_source)
        promotion = {
            "path": str(source),
            "sha256": digest(source),
            # pandas represents empty CSV cells as NaN; round-trip through its
            # JSON encoder so the immutable scientific config stores JSON null
            # rather than a non-standard NaN token.
            "rows": json.loads(pd.read_csv(source).to_json(orient="records")),
        }
    config = {
        "version": "expanded-dataset-study-1",
        "stage": args.stage,
        "datasets": names,
        "models": list(models),
        "seeds": seeds,
        "code_hashes": hashes,
        "dataset_inventory": datasets,
        "preprocessing_definition": study.PREPROCESSING_DEFINITION,
        "preprocessing_definition_hash": study.preprocessing_definition_hash(),
        "alpha_rules": list(study.ALPHA_RULES),
        "C_grid": list(study.SVC_C_GRID),
        "nu_grid": list(study.PAPER_NU_GRID),
        "outer_split": "sorted stratified 75/25, frozen split_indices",
        "inner_split": "3-fold stratified random_state=20000+outer_seed",
        "resources": f"local CPU; {getattr(args, 'n_jobs', 1)} joblib worker(s); no GPU required",
        "execution_backend": "local",
        "versions": versions,
        "python": platform.python_version(),
        "local_git": os.getenv("EXPANDED_LOCAL_GIT", "unknown"),
        "promotion_source": promotion,
        "robust_track": "closed; no robust models",
    }
    return json.loads(json.dumps(config, default=_json_default)), hashes


def _validate_stage_promotion(stage, names, models, promotion):
    """Reject any stage/model combination not authorized by the frozen gate table."""
    if stage not in ("gate24", "confirmation"):
        return
    if not promotion:
        raise ValueError(f"{stage} requires an immutable promotion source")
    rows = promotion.get("rows", [])
    required = []
    if "v7" in models:
        required.append("v7_vs_legacy")
    if "v8_c" in models:
        required.append("v8c_vs_v7")
    for dataset in names:
        for comparison in required:
            authorized = any(
                row.get("dataset") == dataset
                and row.get("comparison") == comparison
                and str(row.get("passed")).lower() == "true"
                for row in rows
            )
            if not authorized:
                raise ValueError(
                    f"{dataset} {comparison} was not promoted by the frozen decision table"
                )


def run_stage(args):
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    unknown = set(names) - set(study.DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets {sorted(unknown)}")
    if not models or any(model not in study.MODELS for model in models):
        raise ValueError(f"invalid models {models}")
    if "v8_c" in models and "v7" not in models:
        raise ValueError("v8_c requires v7 reference")
    if args.n_jobs < 1 or args.n_jobs > 4:
        raise ValueError("n_jobs must be 1..4")
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    config, hashes = _config(args, names, models, seeds)
    _validate_stage_promotion(args.stage, names, models, config.get("promotion_source"))
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = Path(args.output_root) / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("run name exists with incompatible configuration")
    else:
        atomic_json(config_path, config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "complete":
        print("Already complete; immutable result retained")
        return
    for source, sha in hashes.items():
        target = output / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != sha:
                raise RuntimeError(f"source snapshot mismatch {source}")
        else:
            shutil.copyfile(source, target)
    checkpoints = output / "checkpoints"
    failures = output / "failures"
    checkpoints.mkdir(exist_ok=True)
    failures.mkdir(exist_ok=True)
    atomic_json(manifest_path, {
        "status": "running", "fingerprint": fingerprint,
        "app_id": "local",
        "expected_jobs": len(names) * len(seeds),
    })
    started = time.perf_counter()
    jobs = Parallel(n_jobs=args.n_jobs)(
        delayed(_safe_job)(name, seed, models, checkpoints, failures, fingerprint)
        for name in names for seed in seeds
    )
    completed = [job["value"] for job in jobs if job["kind"] == "result"]
    failed = [job["value"] for job in jobs if job["kind"] == "failure"]
    records = pd.DataFrame([record for job in completed for record in job["records"]])
    diagnostics = pd.DataFrame([record for job in completed for record in job["solver_diagnostics"]])
    failure_frame = pd.DataFrame(failed)
    outputs = {
        "per_run.csv": records,
        "summary.csv": study.summarize(records) if not records.empty else pd.DataFrame(),
        "paired_comparisons.csv": study.paired_comparisons(records) if not records.empty else pd.DataFrame(),
        "class_metrics.csv": study.class_metrics(records) if not records.empty else pd.DataFrame(),
        "solver_diagnostics.csv": diagnostics,
        "failures.csv": failure_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    atomic_json(output / "split_registry.json", [job["split"] for job in completed])
    atomic_json(output / "selected_hyperparameters.json", [
        {"dataset": row["dataset"], "model": row["model"], "seed": row["seed"],
         "selected_hyperparameters": json.loads(row["selected_hyperparameters"])}
        for row in records.to_dict(orient="records")
    ])
    file_hashes = {
        path.name: digest(path) for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    status = "complete" if not failed and len(completed) == len(names) * len(seeds) else "partial"
    atomic_json(manifest_path, {
        "status": status,
        "fingerprint": fingerprint,
        "app_id": "local",
        "expected_jobs": len(names) * len(seeds),
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
        "record_count": len(records),
        "stage_wall_s": time.perf_counter() - started,
        "aggregate_job_wall_s": float(sum(job["actual_job_wall_s"] for job in completed)),
        "sha256": file_hashes,
    })
    with tarfile.open(output / "artifacts.tar.gz", "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "artifacts.tar.gz":
                archive.add(path, arcname=path.relative_to(output).as_posix())
    print(study.summarize(records).to_string(index=False) if not records.empty else "no completed records")
    if failed:
        print(f"stage partial: {len(failed)} failures", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(study.DATASETS))
    parser.add_argument("--models", default=",".join(study.MODELS))
    parser.add_argument("--stage", choices=("smoke", "gate8", "gate24", "confirmation"), required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="results/expanded_dataset_study")
    parser.add_argument("--promotion-source", default="")
    run_stage(parser.parse_args())


if __name__ == "__main__":
    main()
