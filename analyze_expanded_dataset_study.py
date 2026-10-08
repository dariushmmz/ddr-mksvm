"""Read-only audit and prospective gate analysis for expanded dataset runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ddr_mksvm import expanded_dataset_study as study
from run_expanded_dataset_study import atomic_json, digest


def _gate8_pass(rows: pd.DataFrame, comparison: str):
    indexed = rows.set_index("metric")
    runtime_ratio = float(indexed.loc["test_error", "runtime_ratio"])
    if comparison == "v7_vs_legacy":
        error = float(indexed.loc["test_error", "mean_delta"])
        candidates = []
        for metric in ("balanced_accuracy", "macro_f1"):
            row = indexed.loc[metric]
            candidates.append((metric, float(row.mean_delta), int(row.wins), int(row.losses)))
        pass_error = error <= 0
        qualifying = [x for x in candidates if x[1] >= .005 and x[2] >= x[3]]
        passed = runtime_ratio <= 10 and error <= .005 and (pass_error or bool(qualifying))
        endpoint = "test_error" if pass_error else (qualifying[0][0] if qualifying else "")
    else:
        error = float(indexed.loc["test_error", "mean_delta"])
        candidates = []
        for metric in ("balanced_accuracy", "macro_f1", "rare_class_recall"):
            row = indexed.loc[metric]
            candidates.append((metric, float(row.mean_delta), int(row.wins), int(row.losses)))
        qualifying = [x for x in candidates if x[1] >= .005 and x[2] >= x[3]]
        passed = runtime_ratio <= 10 and error <= .01 and bool(qualifying)
        endpoint = qualifying[0][0] if qualifying else ""
    return passed, endpoint, runtime_ratio


def _gate24_pass(rows: pd.DataFrame, comparison: str, endpoint: str, class_rows: pd.DataFrame):
    indexed = rows.set_index("metric")
    runtime_ratio = float(indexed.loc["test_error", "runtime_ratio"])
    error = float(indexed.loc["test_error", "mean_delta"])
    if comparison == "v7_vs_legacy":
        error_row = indexed.loc["test_error"]
        balance = float(indexed.loc["balanced_accuracy", "mean_delta"])
        macro = float(indexed.loc["macro_f1", "mean_delta"])
        passed = runtime_ratio <= 10 and (
            (error < 0 and int(error_row.wins) >= int(error_row.losses)) or
            (error <= .0025 and balance >= .005 and macro >= .005)
        )
    else:
        selected = indexed.loc[endpoint]
        threshold = .05 if endpoint == "rare_class_recall" else .01
        recall_ok = True
        if not class_rows.empty:
            recalls = class_rows.pivot(index="label", columns="model", values="recall")
            if {"v7", "v8_c"}.issubset(recalls.columns):
                recall_ok = bool(((recalls.v8_c - recalls.v7) >= -.05).all())
        passed = (runtime_ratio <= 10 and error <= .01 and float(selected.mean_delta) >= threshold
                  and int(selected.wins) > int(selected.losses) and recall_ok)
    return passed, runtime_ratio


def audit(root: Path):
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest.get("sha256", {}).items():
        if digest(root / name) != sha:
            raise AssertionError(f"artifact hash mismatch {name}")
    frame = pd.read_csv(root / "per_run.csv")
    paired = study.paired_comparisons(frame)
    classes = study.class_metrics(frame)
    paired_classes = study.paired_class_recall(frame)
    expected_jobs = len(config["datasets"]) * len(config["seeds"])
    checkpoints = sorted((root / "checkpoints").glob("*.json"))
    failures = sorted((root / "failures").glob("*.json"))
    if len(checkpoints) + len(failures) != expected_jobs:
        raise AssertionError("job accounting mismatch")
    for path in checkpoints:
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["fingerprint"] != manifest["fingerprint"]:
            raise AssertionError("checkpoint fingerprint mismatch")
        split = job["split"]
        if set(split["train_indices"]) & set(split["test_indices"]):
            raise AssertionError("split overlap")
        y = np.asarray(split["y_test"], dtype="<i8")
        for record in job["records"]:
            pred = np.asarray(json.loads(record["prediction_vector"]), dtype="<i8")
            if hashlib.sha256(pred.tobytes()).hexdigest() != record["prediction_hash"]:
                raise AssertionError("prediction hash mismatch")
            if hashlib.sha256(y.tobytes()).hexdigest() != record["true_label_hash"]:
                raise AssertionError("label hash mismatch")
            if int(np.sum(pred != y)) != int(record["errors"]):
                raise AssertionError("error count mismatch")
    numerical_complete = len(failures) == 0 and len(checkpoints) == expected_jobs
    decisions = []
    for dataset in config["datasets"]:
        dataset_frame = frame[frame.dataset == dataset]
        expected_records = len(config["seeds"]) * len(config["models"])
        finite = bool(np.isfinite(dataset_frame[["test_error", "balanced_accuracy", "macro_f1", "runtime_s"]]).all().all())
        complete = numerical_complete and len(dataset_frame) == expected_records
        if config["stage"] == "smoke":
            decisions.append({
                "dataset": dataset, "comparison": "all_models", "stage": "smoke",
                "passed": complete and finite, "endpoint": "numerical_validity",
                "reason": "all models finite and accounted" if complete and finite else "missing/failing/nonfinite task",
            })
            continue
        for reference, candidate, label in (("legacy", "v7", "v7_vs_legacy"), ("v7", "v8_c", "v8c_vs_v7")):
            if reference not in config["models"] or candidate not in config["models"]:
                continue
            rows = paired[(paired.dataset == dataset) & (paired.reference == reference) & (paired.candidate == candidate)]
            if not complete or not finite:
                passed, endpoint, ratio = False, "", None
                reason = "numerical completion failed"
            elif config["stage"] == "gate8":
                passed, endpoint, ratio = _gate8_pass(rows, label)
                reason = "prospective Gate8 rule passed" if passed else "prospective Gate8 rule failed"
            elif config["stage"] == "gate24":
                source_rows = config.get("promotion_source", {}).get("rows", []) if config.get("promotion_source") else []
                endpoint = next((r["endpoint"] for r in source_rows if r["dataset"] == dataset and r["comparison"] == label and str(r["passed"]).lower() == "true"), "")
                class_rows = classes[classes.dataset == dataset]
                passed, ratio = _gate24_pass(rows, label, endpoint, class_rows) if endpoint else (False, None)
                reason = "prospective Gate24 rule passed" if passed else "prospective Gate24 rule failed"
            else:
                source_rows = config.get("promotion_source", {}).get("rows", []) if config.get("promotion_source") else []
                endpoint = next((r["endpoint"] for r in source_rows
                                 if r["dataset"] == dataset and r["comparison"] == label
                                 and str(r["passed"]).lower() == "true"), "")
                ratio = float(rows.iloc[0].runtime_ratio)
                passed = bool(endpoint)
                reason = ("authorized untouched confirmation completed; report without another promotion gate"
                          if passed else "comparison was not authorized by Gate24")
            decisions.append({
                "dataset": dataset, "comparison": label, "stage": config["stage"],
                "passed": bool(passed), "endpoint": endpoint,
                "runtime_ratio": ratio, "reason": reason,
            })
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    paired.to_csv(analysis / "paired_metrics.csv", index=False)
    classes.to_csv(analysis / "class_metrics.csv", index=False)
    paired_classes.to_csv(analysis / "paired_class_recall.csv", index=False)
    pd.DataFrame(decisions).to_csv(analysis / "gate_decisions.csv", index=False)
    runtime = frame.groupby(["dataset", "model"]).runtime_s.agg(["count", "mean", "median", lambda x: x.quantile(.95), "max", "sum"]).reset_index()
    runtime.columns = ["dataset", "model", "count", "mean_s", "median_s", "p95_s", "max_s", "total_s"]
    runtime.to_csv(analysis / "runtime.csv", index=False)
    selection = []
    for row in frame.itertuples():
        params = json.loads(row.selected_hyperparameters)
        selection.append({"dataset": row.dataset, "model": row.model, "seed": row.seed,
                          "alpha_rule": params.get("alpha_rule"), "C": params.get("C"),
                          "selected_nu": json.dumps(params.get("selected_nu")),
                          "preprocessing_hash": row.preprocessing_hash})
    pd.DataFrame(selection).to_csv(analysis / "selected_hyperparameters.csv", index=False)
    diagnostics = pd.read_csv(root / "solver_diagnostics.csv", low_memory=False)
    status_summary = diagnostics.groupby(
        ["dataset", "model", "solver", "solver_status"], dropna=False
    ).size().reset_index(name="calls")
    status_summary.to_csv(analysis / "solver_status_summary.csv", index=False)
    record = {
        "status": "passed",
        "stage": config["stage"],
        "expected_jobs": expected_jobs,
        "completed_jobs": len(checkpoints),
        "failed_jobs": len(failures),
        "records": len(frame),
        "numerical_complete": numerical_complete,
        "fingerprint": manifest["fingerprint"],
        "app_id": manifest["app_id"],
    }
    atomic_json(analysis / "audit.json", record)
    print(pd.DataFrame(decisions).to_string(index=False))
    print(paired.to_string(index=False))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    audit(Path(parser.parse_args().root))
