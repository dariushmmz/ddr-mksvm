"""Targeted Gate8: square-root class weights with error-based selection.

This requested ablation reuses the immutable square-root inner-validation banks
from the original V8 Gate8 and fits only the previously missing outer model.
The original V7 and square-root/balanced (V8-C) records are carried forward as
byte-identified references; neither reference model is refitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import t

from ddr_mksvm import v7_cross_dataset as v7
from ddr_mksvm.experiments.class_sensitive_runner import load
from ddr_mksvm.v8_class_sensitive import fit_selected


DATASETS = (
    "blood_transfusion",
    "mammographicmass_binary",
    "heart_disease",
    "parkinson",
    "wine",
)
SEEDS = tuple(range(9000, 9008))
PRIOR_FINGERPRINT = "b6195144643b8d372efa1122e584800a0e1a634582afdc2dbf97823cdeeb99fa"
PRIOR_RUN = "gate8/v8-gate8-20260909"
RUN_NAME = "gate8/v8-sqrt-error-gate8-20261005-v1"
CANDIDATE = "sqrt_error"
CODE = (
    "ddr_mksvm/v7_cross_dataset.py",
    "ddr_mksvm/iris_research.py",
    "ddr_mksvm/v8_class_sensitive.py",
    "ddr_mksvm/experiments/class_sensitive_runner.py",
    "ddr_mksvm/experiments/gate8_sqrt_error.py",
)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_prior(job: dict, dataset: str, seed: int, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if job.get("fingerprint") != PRIOR_FINGERPRINT:
        raise RuntimeError(f"prior Gate8 fingerprint mismatch for {dataset}-{seed}")
    if not {"v7", "v8_c"}.issubset(set(job.get("variants", []))):
        raise RuntimeError(f"required reference variants absent for {dataset}-{seed}")
    if "sqrt" not in job.get("banks", {}):
        raise RuntimeError(f"square-root validation bank absent for {dataset}-{seed}")
    split = job["split"]
    if split["dataset"] != dataset or int(split["seed"]) != seed:
        raise RuntimeError(f"prior split identity mismatch for {dataset}-{seed}")
    tr = np.asarray(split["train_indices"], dtype=int)
    te = np.asarray(split["test_indices"], dtype=int)
    expected_tr, expected_te = v7.split_indices(y, seed)
    if not np.array_equal(tr, expected_tr) or not np.array_equal(te, expected_te):
        raise RuntimeError(f"reconstructed split mismatch for {dataset}-{seed}")
    if split.get("y_test") != y[te].tolist():
        raise RuntimeError(f"stored test labels mismatch for {dataset}-{seed}")
    return tr, te


def run_job(dataset: str, seed: int, prior_root: Path, output: Path, fingerprint: str) -> dict:
    target = output / "checkpoints" / f"{dataset}-{seed}.json"
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise RuntimeError(f"incompatible checkpoint at {target}")
        return saved

    prior_path = prior_root / "checkpoints" / f"{dataset}-{seed}.json"
    if not prior_path.is_file():
        raise FileNotFoundError(f"missing immutable prior checkpoint: {prior_path}")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    X, y, spec = load(dataset)
    tr, te = _validate_prior(prior, dataset, seed, X, y)
    bank = prior["banks"]["sqrt"]

    started = time.perf_counter()
    prediction, parameters, trace, reported_runtime = fit_selected(
        X[tr],
        y[tr],
        X[te],
        spec["transform"],
        "sqrt",
        "error",
        bank["records"],
        float(bank["selection_time_s"]),
    )
    compute_wall = time.perf_counter() - started
    candidate = v7.metric_record(
        dataset,
        CANDIDATE,
        seed,
        y[te],
        prediction,
        reported_runtime,
        parameters,
        tr,
        te,
    )
    references = [row for row in prior["records"] if row["model"] in {"v7", "v8_c"}]
    if {row["model"] for row in references} != {"v7", "v8_c"}:
        raise RuntimeError(f"reference record mismatch for {dataset}-{seed}")
    for row in references:
        if row["train_indices_hash"] != candidate["train_indices_hash"] or row["test_indices_hash"] != candidate["test_indices_hash"]:
            raise RuntimeError(f"reference/candidate split hash mismatch for {dataset}-{seed}")

    result = {
        "fingerprint": fingerprint,
        "dataset": dataset,
        "seed": seed,
        "records": references + [candidate],
        "candidate_trace": trace,
        "split": prior["split"],
        "prior_checkpoint": {
            "logical_path": f"{PRIOR_RUN}/checkpoints/{dataset}-{seed}.json",
            "sha256": sha256(prior_path),
            "fingerprint": prior["fingerprint"],
            "app_id": prior.get("app_id"),
        },
        "selection_bank": {
            "family": "sqrt",
            "records": len(bank["records"]),
            "folds": len(bank["folds"]),
            "original_selection_time_s": bank["selection_time_s"],
            "reused_without_refit": True,
        },
        "new_outer_fit_compute_wall_s": compute_wall,
    }
    json_write(target, result)
    print(f"complete {dataset} {seed}: new outer fit {compute_wall:.3f}s", flush=True)
    return result


def paired_comparisons(frame: pd.DataFrame, reference: str, candidate: str) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, group in frame[frame.model.isin([reference, candidate])].groupby("dataset"):
        ref = group[group.model == reference].set_index("seed").sort_index()
        cand = group[group.model == candidate].set_index("seed").sort_index()
        if list(ref.index) != list(cand.index):
            raise RuntimeError(f"unmatched paired seeds for {dataset}: {reference} vs {candidate}")
        for key in ("train_indices_hash", "test_indices_hash"):
            if not (ref[key] == cand[key]).all():
                raise RuntimeError(f"unmatched {key} for {dataset}: {reference} vs {candidate}")
        for metric in ("test_error", "balanced_accuracy", "macro_f1"):
            delta = cand[metric].to_numpy(float) - ref[metric].to_numpy(float)
            n = len(delta)
            sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
            half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            corrected_half = float(t.ppf(0.975, n - 1) * sd * np.sqrt(1 / n + 1 / 3)) if n > 1 else np.nan
            favorable = -delta if metric == "test_error" else delta
            rows.append(
                {
                    "dataset": dataset,
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "seeds": n,
                    "mean_reference": float(ref[metric].mean()),
                    "mean_candidate": float(cand[metric].mean()),
                    "mean_delta": float(delta.mean()),
                    "nominal_95_split_interval_low": float(delta.mean() - half),
                    "nominal_95_split_interval_high": float(delta.mean() + half),
                    "resampled_sensitivity_low": float(delta.mean() - corrected_half),
                    "resampled_sensitivity_high": float(delta.mean() + corrected_half),
                    "wins": int((favorable > 1e-14).sum()),
                    "ties": int((np.abs(favorable) <= 1e-14).sum()),
                    "losses": int((favorable < -1e-14).sum()),
                    "standardized_paired_effect": float(delta.mean() / sd) if sd else np.nan,
                    "runtime_ratio": float(cand.runtime_s.sum() / ref.runtime_s.sum()),
                    "interval_scope": "nominal/descriptive; dependent repeated holdouts; not a formal independent-sample confidence interval",
                }
            )
    return pd.DataFrame(rows)


def class_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        labels = json.loads(group.iloc[0].labels)
        matrix = sum(np.asarray(value, dtype=int) for value in group.confusion_matrix.map(json.loads))
        for index, label in enumerate(labels):
            tp = int(matrix[index, index])
            precision = tp / matrix[:, index].sum() if matrix[:, index].sum() else 0.0
            recall = tp / matrix[index].sum() if matrix[index].sum() else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "label": label,
                    "support": int(matrix[index].sum()),
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                    "confusion_row": json.dumps(matrix[index].tolist()),
                }
            )
    return pd.DataFrame(rows)


def run_stage(output_root: Path, prior_root: Path, workers: int) -> Path:
    output = output_root / RUN_NAME
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print("Already complete; no artifact overwritten", flush=True)
            return output

    if not (prior_root / "manifest.json").is_file():
        raise FileNotFoundError(f"prior Gate8 run is unavailable: {prior_root}")
    prior_manifest = json.loads((prior_root / "manifest.json").read_text(encoding="utf-8"))
    if prior_manifest.get("fingerprint") != PRIOR_FINGERPRINT or prior_manifest.get("status") != "complete":
        raise RuntimeError("prior Gate8 manifest is not the expected completed run")

    versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "scipy", "pandas", "scikit-learn", "joblib")
    }
    hashes = {path: sha256(path) for path in CODE}
    config = {
        "version": "v8-sqrt-error-gate8-1",
        "run_name": RUN_NAME,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "candidate": {
            "model": CANDIDATE,
            "weighting": "mean-one square-root inverse-frequency within each OVO pair",
            "family": "sqrt",
            "selection_metric": "error",
        },
        "references": ["v7", "v8_c"],
        "prior_run": {
            "logical_path": PRIOR_RUN,
            "fingerprint": PRIOR_FINGERPRINT,
            "manifest_sha256": sha256(prior_root / "manifest.json"),
            "app_id": prior_manifest.get("app_id"),
        },
        "reuse_policy": "reuse immutable sqrt inner-validation records and folds; fit only the missing error-selected outer model",
        "outer": "sorted stratified 75/25; exact original Gate8 indices",
        "inner": "3-fold stratified seed=20000+outer; equal-fold mean error; original alpha/C order tie break",
        "alpha_rules": list(v7.ALPHA_RULES),
        "C_grid": list(v7.SVC_C_GRID),
        "dataset_inventory": {name: load(name)[2] for name in DATASETS},
        "metrics": ["test_error", "balanced_accuracy", "macro_f1", "classwise pooled precision/recall/f1"],
        "interval": {
            "label": "nominal/descriptive 95% paired split-variation interval",
            "interpretation": "Repeated holdouts overlap and are dependent; this is not a formal independent-sample confidence interval.",
        },
        "compute": "Modal CPU; 4 CPUs; no GPU; at most four joblib workers",
        "python": platform.python_version(),
        "versions": versions,
        "code_hashes": hashes,
        "modal_app_id": os.getenv("MODAL_APP_ID", "unknown"),
        "local_git": os.getenv("V8_LOCAL_GIT", "unknown"),
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["fingerprint"] = fingerprint
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("incompatible existing experiment configuration")
    json_write(config_path, config)

    for source in CODE:
        target = output / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != hashes[source]:
            raise RuntimeError(f"incompatible source snapshot: {target}")
        if not target.exists():
            shutil.copyfile(source, target)

    started = time.perf_counter()
    json_write(
        manifest_path,
        {"status": "running", "fingerprint": fingerprint, "app_id": os.getenv("MODAL_APP_ID", "unknown")},
    )
    jobs = Parallel(n_jobs=workers)(
        delayed(run_job)(dataset, seed, prior_root, output, fingerprint)
        for dataset in DATASETS
        for seed in SEEDS
    )
    frame = pd.DataFrame([record for job in jobs for record in job["records"]])
    frame.to_csv(output / "per_run.csv", index=False)
    v7.summarize(frame).to_csv(output / "summary.csv", index=False)
    paired_comparisons(frame, "v7", CANDIDATE).to_csv(output / "paired_vs_v7.csv", index=False)
    paired_comparisons(frame, "v8_c", CANDIDATE).to_csv(output / "paired_vs_v8_c.csv", index=False)
    class_metrics(frame).to_csv(output / "class_metrics.csv", index=False)

    selections = []
    for row in frame[frame.model == CANDIDATE].itertuples(index=False):
        selected = json.loads(row.selected_hyperparameters)
        selections.append(
            {
                "dataset": row.dataset,
                "seed": row.seed,
                "alpha_rule": selected["alpha_rule"],
                "alpha": selected["alpha"],
                "C": selected["C"],
                "inner_error": selected["losses"]["error"],
                "inner_balanced_loss": selected["losses"]["balanced"],
                "support_vectors": selected["support_vectors"],
            }
        )
    pd.DataFrame(selections).to_csv(output / "selected_hyperparameters.csv", index=False)

    files = [path for path in output.rglob("*") if path.is_file() and path != manifest_path]
    manifest = {
        "status": "complete",
        "fingerprint": fingerprint,
        "app_id": os.getenv("MODAL_APP_ID", "unknown"),
        "stage_wall_s": time.perf_counter() - started,
        "new_outer_fit_compute_wall_s": sum(job["new_outer_fit_compute_wall_s"] for job in jobs),
        "new_outer_fits": len(jobs),
        "reused_inner_banks": len(jobs),
        "rows": len(frame),
        "sha256": {str(path.relative_to(output)).replace("\\", "/"): sha256(path) for path in files},
    }
    json_write(manifest_path, manifest)
    print(v7.summarize(frame).to_string(index=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("results/v8"))
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    prior = args.prior_root or args.output_root / PRIOR_RUN
    result = run_stage(args.output_root, prior, args.workers)
    print(result, flush=True)


if __name__ == "__main__":
    main()

