"""Reproducible staged Iris-only experiments for the paper q=1 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from ddr_mksvm.iris_research import (
    ALPHA_RULES,
    COARSE_RHO_GRID,
    EXPANDED_NU_GRID,
    MODEL_SPECS,
    PAPER_NU_GRID,
    SOURCE_CONTINUED_NU_GRID,
    WIDE_NU_GRID,
    evaluate_model,
    index_hash,
    locked_split_indices,
    numeric_sha256,
    reconstruct_authors_iris,
)


CONFIG_VERSION = "iris-v8-standardized-rkhs-l2"
REFERENCE_MODEL = "matlab_source_legacy_highs"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_metadata() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def _code_hash() -> str:
    digest = hashlib.sha256()
    for name in (Path(__file__), Path("ddr_mksvm/iris_research.py")):
        digest.update(name.read_bytes())
    return digest.hexdigest()


def _split_registry(labels: np.ndarray, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        train, test = locked_split_indices(labels, seed)
        rows.append({
            "seed": seed,
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
            "train_indices_hash": index_hash(train),
            "test_indices_hash": index_hash(test),
            "train_class_counts": {str(v): int(np.sum(labels[train] == v)) for v in (1, 2, 3)},
            "test_class_counts": {str(v): int(np.sum(labels[test] == v)) for v in (1, 2, 3)},
        })
    return rows


def _one(model_name: str, seed: int, values: np.ndarray, run_metadata: dict) -> dict:
    train_idx, test_idx = locked_split_indices(values[:, -1], seed)
    X, labels = values[:, :-1], values[:, -1].astype(int)
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    record, diagnostics = evaluate_model(
        model_name, X[train_idx], labels[train_idx], X[test_idx], labels[test_idx], seed
    )
    record.update({
        "run_name": run_metadata["run_name"],
        "git_commit": run_metadata["git_commit"],
        "git_dirty": run_metadata["git_dirty"],
        "timestamp": started,
        "train_indices_hash": index_hash(train_idx),
        "test_indices_hash": index_hash(test_idx),
        "dataset_numeric_sha256": numeric_sha256(values),
        "modal_resources": run_metadata["modal_resources"],
        "code_config_version": CONFIG_VERSION,
        "code_sha256": run_metadata["code_sha256"],
        "runtime_s": time.perf_counter() - begin,
        "hyperparameters": {
            "paper_nu_grid": PAPER_NU_GRID,
            "source_continued_nu_grid": SOURCE_CONTINUED_NU_GRID,
            "expanded_nu_grid": EXPANDED_NU_GRID,
            "wide_nu_grid": WIDE_NU_GRID,
            "alpha_rules": ALPHA_RULES,
            "coarse_rho_grid": COARSE_RHO_GRID,
        },
    })
    for item in diagnostics:
        item.update({"run_name": run_metadata["run_name"], "architecture": model_name, "seed": seed})
    return {"record": record, "solver_diagnostics": diagnostics}


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, float)
    if len(values) < 2:
        return float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return float(values.mean() - half), float(values.mean() + half)


def _summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for model, group in frame.groupby("architecture", sort=True):
        errors = group["test_error"].to_numpy(float)
        ci_low, ci_high = _mean_ci(errors)
        summary_rows.append({
            "architecture": model, "n": len(group), "mean_error": errors.mean(),
            "std_error": errors.std(ddof=1) if len(errors) > 1 else np.nan,
            "median_error": np.median(errors), "standard_error": stats.sem(errors) if len(errors) > 1 else np.nan,
            "error_ci95_low": ci_low, "error_ci95_high": ci_high,
            "accuracy": group["accuracy"].mean(),
            "balanced_accuracy": group["balanced_accuracy"].mean(),
            "macro_f1": group["macro_f1"].mean(),
            "recall_class_1": group["recall_class_1"].mean(),
            "recall_class_2": group["recall_class_2"].mean(),
            "recall_class_3": group["recall_class_3"].mean(),
            "runtime_mean_s": group["runtime_s"].mean(),
            "runtime_median_s": group["runtime_s"].median(),
            "runtime_total_s": group["runtime_s"].sum(),
            "solver_solve_count": group["solver_solve_count"].sum(),
            "solver_status_ok": int((group["solver_status"] == "ok").sum()),
            "trivial_outer_classifiers": group["trivial_outer_classifiers"].sum(),
        })
    paired_rows = []
    if REFERENCE_MODEL in set(frame["architecture"]):
        reference = frame[frame["architecture"] == REFERENCE_MODEL].set_index("seed")["test_error"]
        for model in sorted(set(frame["architecture"]) - {REFERENCE_MODEL}):
            candidate = frame[frame["architecture"] == model].set_index("seed")["test_error"]
            common = reference.index.intersection(candidate.index)
            delta = candidate.loc[common].to_numpy() - reference.loc[common].to_numpy()
            ci_low, ci_high = _mean_ci(delta)
            nonzero = delta[delta != 0]
            wilcoxon_p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
            paired_rows.append({
                "reference": REFERENCE_MODEL, "candidate": model, "n": len(delta),
                "mean_delta": delta.mean(), "delta_ci95_low": ci_low, "delta_ci95_high": ci_high,
                "wins": int(np.sum(delta < 0)), "ties": int(np.sum(delta == 0)),
                "losses": int(np.sum(delta > 0)), "wilcoxon_p": wilcoxon_p,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODEL_SPECS))
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="results/iris_experiments")
    args = parser.parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(models) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if not 1 <= args.n_seeds <= 96:
        raise ValueError("n-seeds must be between 1 and 96")
    if args.seed_start < 0:
        raise ValueError("seed-start must be non-negative")
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    local_csv = Path("dataset/iris_multiclass.csv")
    local_values = pd.read_csv(local_csv).to_numpy(float)
    values = reconstruct_authors_iris(local_values)
    commit, dirty = _git_metadata()
    output = Path(args.output_root) / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "run_name": args.run_name, "git_commit": commit, "git_dirty": dirty,
        "code_sha256": _code_hash(),
        "modal_resources": os.environ.get("IRIS_MODAL_RESOURCES", "local-test-only"),
    }
    config = {
        **run_metadata,
        "config_version": CONFIG_VERSION,
        "models": models, "n_seeds": args.n_seeds, "seed_start": args.seed_start,
        "seeds": seeds,
        "locked_seed_registry": list(range(args.seed_start, args.seed_start + 96)),
        "n_jobs": args.n_jobs, "outer_protocol": "stratified 75/25",
        "inner_protocol": "3-fold stratified CV on outer training only",
        "paper_nu_grid": PAPER_NU_GRID, "expanded_nu_grid": EXPANDED_NU_GRID,
        "source_continued_nu_grid": SOURCE_CONTINUED_NU_GRID,
        "wide_nu_grid": WIDE_NU_GRID,
        "alpha_rules": ALPHA_RULES, "coarse_rho_grid": COARSE_RHO_GRID,
        "local_csv_sha256": _file_sha256(local_csv),
        "authors_numeric_sha256": numeric_sha256(values),
    }
    config_hash = hashlib.sha256(json.dumps(_jsonable(config), sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config_hash:
            raise ValueError("run name already contains an incompatible configuration")
    else:
        _atomic_json(output / "config.json", config)
        locked_seeds = list(range(args.seed_start, args.seed_start + 96))
        _atomic_json(output / "splits.json", {"splits": _split_registry(values[:, -1], locked_seeds)})
        _atomic_json(manifest_path, {
            "run_name": args.run_name, "config_sha256": config_hash,
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        })

    checkpoint_dir = output / "checkpoints"
    tasks, payloads = [], []
    for seed in seeds:
        for model in models:
            checkpoint = checkpoint_dir / f"seed_{seed:03d}__{model}.json"
            if checkpoint.exists():
                payloads.append(json.loads(checkpoint.read_text(encoding="utf-8")))
            else:
                tasks.append((model, seed, checkpoint))
    print(f"[iris] run={args.run_name} cached={len(payloads)} pending={len(tasks)} jobs={args.n_jobs}", flush=True)
    completed = Parallel(n_jobs=min(args.n_jobs, max(1, len(tasks))))(
        delayed(_one)(model, seed, values, run_metadata) for model, seed, _ in tasks
    ) if tasks else []
    for (_, _, checkpoint), payload in zip(tasks, completed):
        _atomic_json(checkpoint, payload)
        payloads.append(payload)

    records = [payload["record"] for payload in payloads]
    records.sort(key=lambda row: (row["seed"], row["architecture"]))
    diagnostics = [item for payload in payloads for item in payload["solver_diagnostics"]]
    frame = pd.DataFrame(records)
    for column in (
        "prediction_vector", "selected_nu_by_task", "alpha_scales", "kernel_weights",
        "kernel_weight_diagnostics", "alpha_cv_scores", "nu_cv_scores", "rho_cv_scores",
        "preprocessing_diagnostics", "hyperparameters", "modal_resources",
        "selected_feature_indices_1based", "feature_cv_scores",
        "outer_nu_grid",
    ):
        if column in frame:
            frame[column] = frame[column].map(lambda value: json.dumps(_jsonable(value), sort_keys=True))
    frame.to_csv(output / "per_run.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "solver_diagnostics.csv", index=False)
    summary, paired = _summaries(frame)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    _atomic_json(manifest_path, {
        "run_name": args.run_name, "config_sha256": config_hash,
        "completed_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "completed_model_seeds": len(records), "solver_solve_count": len(diagnostics),
    })
    print(summary.to_string(index=False), flush=True)
    if not paired.empty:
        print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
