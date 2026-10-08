"""Checkpointed runner for the isolated MATLAB-source robust q=1 baseline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ddr_mksvm.iris_research import reconstruct_authors_iris
from ddr_mksvm.robust_matlab_parity import (KernelSpec, MODEL_NAMES, NU_GRID, array_hash,
    class_eta, fit_binary, fit_ova, predict_ova)
from ddr_mksvm.v7_cross_dataset import DATASETS, fit_transform, load_dataset, split_indices

SOURCE_HASHES = {
    "binary_main": "9559f56af3832799603297f5fe7d54905b38438076ea7845da4bb322be1132e9",
    "binary_unit": "d2e77e25cb0dcf8b74bffa127a03c457094fa3435ebb36a63dccee0274dd4f7d",
    "binary_split": "7bcbb898e8b85d10cae176cba7ddd9a49ae65bfaa74d11f2685c86c647a719c9",
    "multiclass_main": "6b3136548d191882ccac4e0427a4c33a0380784a5ea74e339bbaeed01c4ec815",
    "multiclass_unit": "dad087b29ee3b3c033ceb3f79b0dddc5b7c37a181a391277efa501f056f0b5f2",
    "multiclass_split": "0ce24ed343ed8f0790675d63bfc5d07974db33cf7f473d93f6973dc5c50b661f",
}
CODE_FILES = ["ddr_mksvm/robust_matlab_parity.py", "run_robust_matlab_parity.py",
              "modal_robust_matlab_parity.py", "analyze_robust_matlab_parity.py"]

PAPER_KERNELS = {
    "parkinson": ("poly", 1, 0.0), "blood_transfusion": ("poly", 3, "std"),
    "mammographicmass_binary": ("poly", 2, "std"),
    "breast_cancer_diagnostic": ("poly", 2, "std"),
    "wine": ("poly", 1, "std"), "heart_disease": ("poly", 1, "std"),
    "dermatology": ("poly", 2, "std"), "iris": ("rbf", 0, 0.0),
}


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load(name: str):
    if name != "iris":
        return load_dataset(name)
    raw = pd.read_csv("dataset/iris_multiclass.csv").to_numpy(float)
    values = reconstruct_authors_iris(raw)
    return values[:, :-1], values[:, -1].astype(int), {
        "rows": 150, "features": 4, "classes": [1, 2, 3],
        "class_counts": {"1": 50, "2": 50, "3": 50}, "transform": "none",
        "source_numeric_sha256": array_hash(values), "project_csv_sha256": digest("dataset/iris_multiclass.csv"),
    }


def profile(name: str, requested: str) -> str:
    if requested == "auto":
        return "matlab_active" if name in {"mammographicmass_binary", "iris"} else "paper_configured"
    if requested == "matlab_active" and name not in {"mammographicmass_binary", "iris"}:
        raise ValueError(f"no unchanged supplied MATLAB robust script exists for {name}")
    return requested


def prepare(name: str, seed: int, requested_profile: str):
    X, y, inventory = load(name)
    selected_profile = profile(name, requested_profile)
    transform = "none" if selected_profile == "matlab_active" else inventory.get("transform", DATASETS.get(name, {}).get("transform", "none"))
    # Adjacent MATLAB experiment mains transform the full table before cvpartition.
    # Preserve that paper-configured convention and label the resulting leakage.
    if transform != "none":
        if transform == "minmax":
            offset=np.min(X,axis=0);scale=np.max(X,axis=0)-offset
        elif transform == "standardization":
            offset=np.mean(X,axis=0);scale=np.std(X,axis=0,ddof=1)
        else:raise ValueError(transform)
        scale=np.where(scale>0,scale,1.0);X=(X-offset)/scale
        prep={"offset":offset.tolist(),"scale":scale.tolist(),"matlab_std_ddof":1 if transform=="standardization" else None}
        preprocessing_scope = "full_dataset_before_split"
    else:
        X = np.asarray(X, float).copy(); prep = {}; preprocessing_scope = "none"
    tr, te = split_indices(y, seed)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    if selected_profile == "matlab_active":
        if name == "mammographicmass_binary":
            kernel = KernelSpec("poly", degree=2, offset=0.0)
        else:
            kernel = KernelSpec("poly", degree=1, offset=0.0)
    else:
        kind, degree, offset = PAPER_KERNELS[name]
        if kind == "rbf":
            alpha = float(np.max(np.std(Xtr, axis=0, ddof=1)))
            kernel = KernelSpec("rbf", alpha=alpha)
        else:
            c = float(np.max(np.std(Xtr, axis=0, ddof=1))) if offset == "std" else float(offset)
            kernel = KernelSpec("poly", degree=degree, offset=c)
    split = {"dataset": name, "seed": seed, "train_indices": tr.tolist(), "test_indices": te.tolist(),
             "train_hash": array_hash(tr, "<i8"), "test_hash": array_hash(te, "<i8"),
             "profile": selected_profile, "preprocessing": transform,
             "preprocessing_scope": preprocessing_scope, "preprocessing_parameters": prep}
    return Xtr, ytr, Xte, yte, kernel, split, inventory


def metric_record(name, seed, model_name, p, rho, ytrue, pred, runtime, kernel, split, diag):
    labels = sorted(np.unique(ytrue).tolist())
    pr, re, f1, support = precision_recall_fscore_support(ytrue, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(ytrue, pred, labels=labels)
    return {"dataset": name, "seed": seed, "model": model_name, "p": p, "rho": rho,
        "q": 1, "kernel": kernel.kind, "kernel_parameters": json.dumps(kernel.__dict__, sort_keys=True),
        "preprocessing": split["preprocessing"], "preprocessing_scope": split["preprocessing_scope"],
        "solver": diag["solver"], "solver_status": diag["solver_status"],
        "objective_value": diag["objective_value"], "runtime_s": runtime,
        "selected_nu": diag["nu"], "threshold": diag["b"], "gamma": diag["gamma"],
        "uncertainty_radius": diag["max_delta"], "robust_penalty_scalar": diag["robust_penalty_scalar"],
        "constraint_max_violation": diag["max_constraint_violation"], "solver_iterations": diag["solver_iterations"],
        "primal_status": diag["primal_status"], "dual_status": diag["dual_status"],
        "test_error": float(np.mean(pred != ytrue)), "balanced_accuracy": float(balanced_accuracy_score(ytrue, pred)),
        "macro_f1": float(f1_score(ytrue, pred, average="macro", zero_division=0)),
        "per_class_recall": json.dumps(dict(zip(map(str, labels), map(float, re))), sort_keys=True),
        "class_precision": json.dumps(dict(zip(map(str, labels), map(float, pr))), sort_keys=True),
        "class_f1": json.dumps(dict(zip(map(str, labels), map(float, f1))), sort_keys=True),
        "class_support": json.dumps(dict(zip(map(str, labels), map(int, support))), sort_keys=True),
        "confusion_matrix": json.dumps(cm.tolist()), "predictions": json.dumps(np.asarray(pred, int).tolist()),
        "prediction_hash": array_hash(pred, "<i8"), "true_labels": json.dumps(np.asarray(ytrue, int).tolist()),
        "true_label_hash": array_hash(ytrue, "<i8"), "split_hash": hashlib.sha256(
            (split["train_hash"] + split["test_hash"]).encode()).hexdigest(),
        "train_indices_hash": split["train_hash"], "test_indices_hash": split["test_hash"]}


def evaluate_one(name: str, seed: int, model_name: str, rho: float, requested_profile: str,
                 solver: str, fingerprint: str, checkpoint: Path):
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise RuntimeError(f"incompatible checkpoint {checkpoint}")
        return result
    Xtr, ytr, Xte, yte, kernel, split, inventory = prepare(name, seed, requested_profile)
    started = time.perf_counter()
    p = {"deterministic": math.inf, "robust_p1": 1.0, "robust_p2": 2.0, "robust_pinf": math.inf}[model_name]
    effective_rho = 0.0 if model_name == "deterministic" else float(rho)
    if len(np.unique(ytr)) == 2:
        classes = sorted(np.unique(ytr).tolist()); negative, positive = classes[0], classes[-1]
        # MATLAB binary table extraction produces the positive block first.
        order = np.concatenate((np.flatnonzero(ytr == positive), np.flatnonzero(ytr == negative)))
        bx, original = Xtr[order], ytr[order]
        by = np.where(original == positive, 1.0, -1.0)
        fitted, diagnostics = fit_binary(bx, by, original, kernel, effective_rho, p, solver=solver)
        fitted.positive_class = int(positive)
        scores = fitted.decision(Xte)
        pred = np.where(scores > 0, positive, negative)
        tie_count = int(np.sum(scores == 0))
        selected = fitted.diagnostics
        fitted_parameters = [{"positive_class": int(positive), "negative_class": int(negative),
            "training_order_within_split": order.tolist(), "u": fitted.u.tolist(), "xi": fitted.xi.tolist(),
            "gamma": fitted.gamma, "b": fitted.b, "delta": fitted.delta.tolist()}]
    else:
        fitted, diagnostics = fit_ova(Xtr, ytr, kernel, effective_rho, p, solver=solver)
        pred, scores, tie_count = predict_ova(fitted, Xte, tie_policy="matlab_error")
        selected_rows = [row for row in diagnostics if row["selected"]]
        fitted_parameters = [{"positive_class": model.positive_class, "u": model.u.tolist(),
            "xi": model.xi.tolist(), "gamma": model.gamma, "b": model.b,
            "delta": model.delta.tolist()} for model in fitted]
        selected = {"solver": "+".join(sorted(set(row["solver"] for row in selected_rows))),
            "solver_status": "+".join(sorted(set(row["solver_status"] for row in selected_rows))),
            "objective_value": float(sum(row["objective_value"] for row in selected_rows)),
            "nu": json.dumps([row["nu"] for row in selected_rows]), "b": json.dumps([row["b"] for row in selected_rows]),
            "gamma": json.dumps([row["gamma"] for row in selected_rows]), "max_delta": max(row["max_delta"] for row in selected_rows),
            "robust_penalty_scalar": float(sum(row["robust_penalty_scalar"] for row in selected_rows)),
            "max_constraint_violation": max(row["max_constraint_violation"] for row in selected_rows),
            "solver_iterations": int(sum(row["solver_iterations"] for row in selected_rows)),
            "primal_status": "+".join(sorted(set(row["primal_status"] for row in selected_rows))),
            "dual_status": "+".join(sorted(set(row["dual_status"] for row in selected_rows)))}
    runtime = time.perf_counter() - started
    record = metric_record(name, seed, model_name, "none" if model_name == "deterministic" else ("inf" if math.isinf(p) else int(p)),
                           effective_rho, yte, pred, runtime, kernel, split, selected)
    record["tie_count"] = tie_count
    result = {"fingerprint": fingerprint, "record": record, "split": split,
              "dataset_inventory": inventory, "solver_diagnostics": diagnostics,
              "fitted_parameters": fitted_parameters, "scores": np.asarray(scores).tolist(),
              "app_id": os.getenv("MODAL_APP_ID", "local")}
    atomic_json(checkpoint, result)
    print(f"complete {name} seed={seed} {model_name} rho={effective_rho:g} {runtime:.3f}s", flush=True)
    return result


def aggregate(run_dir: Path, allow_partial: bool = True, output_dir: Path | None = None):
    checkpoints = sorted((run_dir / "checkpoints").glob("*.json"))
    if not checkpoints and not allow_partial:
        raise RuntimeError("no checkpoints")
    jobs = [json.loads(path.read_text(encoding="utf-8")) for path in checkpoints]
    records = pd.DataFrame([job["record"] for job in jobs])
    diagnostics = pd.DataFrame([{"dataset": job["record"]["dataset"], "seed": job["record"]["seed"],
        "model": job["record"]["model"], "rho": job["record"]["rho"], "p": job["record"]["p"], **row}
        for job in jobs for row in job["solver_diagnostics"]])
    split_rows = [job["split"] for job in jobs]
    if records.empty:
        summary = paired = class_metrics = rho_selection = pd.DataFrame()
    else:
        summary = records.groupby(["dataset", "model", "p", "rho"], dropna=False).agg(
            seeds=("seed", "nunique"), error=("test_error", "mean"), balanced_acc=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"), runtime=("runtime_s", "mean"),
            solver_status=("solver_status", lambda x: "+".join(sorted(set(x))))).reset_index()
        class_rows = []
        for _, row in records.iterrows():
            recalls = json.loads(row.per_class_recall)
            for label, recall in recalls.items():
                class_rows.append({"dataset": row.dataset, "seed": row.seed, "model": row.model,
                                   "p": row.p, "rho": row.rho, "class": label, "recall": recall})
        class_metrics = pd.DataFrame(class_rows)
        paired_rows = []
        for (ds, seed), group in records.groupby(["dataset", "seed"]):
            base = group[group.model == "deterministic"]
            if base.empty: continue
            base = base.iloc[0]
            for _, row in group[group.model != "deterministic"].iterrows():
                paired_rows.append({"dataset": ds, "seed": seed, "model": row.model, "p": row.p, "rho": row.rho,
                    "deterministic_error": base.test_error, "robust_error": row.test_error,
                    "delta_error": row.test_error-base.test_error,
                    "delta_balanced_accuracy": row.balanced_accuracy-base.balanced_accuracy,
                    "delta_macro_f1": row.macro_f1-base.macro_f1})
        paired = pd.DataFrame(paired_rows)
        robust = summary[summary.model != "deterministic"]
        rho_rows = []
        for (ds, model), group in robust.groupby(["dataset", "model"]):
            best = group.sort_values(["error", "rho"], kind="stable").iloc[0]
            rho_rows.append({"dataset": ds, "model": model, "selected_rho": best.rho,
                "selection_basis": "analysis-only minimum test mean; mirrors paper reporting; not deployable validation",
                "candidate_count": len(group)})
        rho_selection = pd.DataFrame(rho_rows)
    target_dir = run_dir if output_dir is None else output_dir
    for name, frame in (("per_run.csv", records), ("summary.csv", summary),
                        ("paired_comparisons.csv", paired), ("class_metrics.csv", class_metrics),
                        ("solver_diagnostics.csv", diagnostics), ("rho_selection.csv", rho_selection),
                        ("split_registry.csv", pd.DataFrame(split_rows))):
        atomic_csv(target_dir / name, frame)
    return {"jobs": jobs, "records": records, "summary": summary}


def experiment_state(args):
    if not os.getenv("ROBUST_ALLOW_LOCAL_FIT") and not os.getenv("MODAL_IS_REMOTE"):
        raise RuntimeError("real fitting is Modal-only; set ROBUST_ALLOW_LOCAL_FIT=1 only for a tiny debugging run")
    run_name_path = Path(args.run_name)
    if run_name_path.is_absolute() or ".." in run_name_path.parts:
        raise ValueError("run_name must be a relative path without '..'")
    names = [v for v in args.datasets.split(",") if v]
    models = [v for v in args.models.split(",") if v]
    if not set(models) <= set(MODEL_NAMES) or "deterministic" not in models:
        raise ValueError("models must include deterministic and use known names")
    rhos = [float(v) for v in args.rhos.split(",") if v]
    if not rhos:
        raise ValueError("at least one rho is required")
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = {path: digest(path) for path in CODE_FILES if Path(path).exists()}
    cfg = {"version": "robust-matlab-parity-1", "datasets": names, "models": models, "rhos": rhos,
        "seeds": seeds, "profile": args.profile, "solver": args.solver, "nu_grid": list(NU_GRID),
        "source_hashes": SOURCE_HASHES, "code_hashes": code_hashes,
        "dataset_inventory": {name: load(name)[2] for name in names},
        "resources": "4 CPU, 8192 MiB, no GPU, <=4 workers", "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "joblib")},
        "local_git": os.getenv("ROBUST_LOCAL_GIT", "unknown")}
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise RuntimeError("run name already has an incompatible configuration")
    if not config_path.exists(): atomic_json(config_path, cfg)
    manifest_path = run_dir / "manifest.json"
    complete = manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete"
    for path, sha in code_hashes.items():
        target = run_dir / "source_snapshot" / path; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != sha: raise RuntimeError(f"immutable snapshot mismatch {target}")
        if not target.exists(): shutil.copyfile(path, target)
    checkpoint_dir = run_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    tasks = []
    for name in names:
        for seed in seeds:
            tasks.append((name, seed, "deterministic", 0.0))
            for model in models:
                if model != "deterministic":
                    tasks.extend((name, seed, model, rho) for rho in rhos)
    return run_dir, manifest_path, checkpoint_dir, fingerprint, tasks, complete


def run(args):
    run_dir, manifest_path, checkpoint_dir, fingerprint, tasks, complete = experiment_state(args)
    if complete:
        print(json.dumps({"status": "complete", "tasks": len(tasks), "run_name": args.run_name}))
        return
    if args.mode == "prepare":
        atomic_json(manifest_path, {"status": "running", "fingerprint": fingerprint,
                                    "app_id": os.getenv("MODAL_APP_ID", "local"),
                                    "expected_jobs": len(tasks)})
        print(json.dumps({"status": "prepared", "tasks": len(tasks), "run_name": args.run_name}))
        return
    if not manifest_path.exists():
        atomic_json(manifest_path, {"status": "running", "fingerprint": fingerprint,
                                    "app_id": os.getenv("MODAL_APP_ID", "local"),
                                    "expected_jobs": len(tasks)})
    elif json.loads(manifest_path.read_text()).get("fingerprint") != fingerprint:
        raise RuntimeError("manifest fingerprint does not match configuration")
    if args.mode == "finalize":
        completed = len(list(checkpoint_dir.glob("*.json")))
        if completed != len(tasks):
            raise RuntimeError(f"cannot finalize: {completed}/{len(tasks)} checkpoints complete")
        started = time.perf_counter()
        output = aggregate(run_dir, allow_partial=False)
        hashes = {path.name: digest(path) for path in run_dir.iterdir() if path.is_file() and path.name != "manifest.json"}
        atomic_json(manifest_path, {"status": "complete", "fingerprint": fingerprint,
            "app_id": os.getenv("MODAL_APP_ID", "local"), "jobs": completed, "rows": len(output["records"]),
            "finalize_wall_s": time.perf_counter()-started, "artifact_sha256": hashes})
        print(output["summary"].to_string(index=False))
        return
    task_start = 0 if args.task_start is None else args.task_start
    task_stop = len(tasks) if args.task_stop is None else min(args.task_stop, len(tasks))
    if task_start < 0 or task_start > task_stop or task_stop > len(tasks):
        raise ValueError(f"invalid task slice [{task_start}, {task_stop}) for {len(tasks)} tasks")
    selected_tasks = tasks[task_start:task_stop]
    started = time.perf_counter()
    results = Parallel(n_jobs=min(args.n_jobs, 4))(delayed(evaluate_one)(name, seed, model, rho,
        args.profile, args.solver, fingerprint, checkpoint_dir / f"{name}-{seed}-{model}-rho-{rho:.12g}.json")
        for name, seed, model, rho in selected_tasks)
    if args.mode == "worker":
        print(json.dumps({"status": "worker_complete", "task_start": task_start,
                          "task_stop": task_stop, "jobs": len(results),
                          "wall_s": time.perf_counter()-started}))
        return
    output = aggregate(run_dir, allow_partial=False)
    hashes = {path.name: digest(path) for path in run_dir.iterdir() if path.is_file() and path.name != "manifest.json"}
    atomic_json(manifest_path, {"status": "complete", "fingerprint": fingerprint,
        "app_id": os.getenv("MODAL_APP_ID", "local"), "jobs": len(tasks), "rows": len(output["records"]),
        "wall_s": time.perf_counter()-started, "artifact_sha256": hashes})
    print(output["summary"].to_string(index=False))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="parkinson,iris")
    p.add_argument("--models", default=",".join(MODEL_NAMES))
    p.add_argument("--rhos", default="1e-4")
    p.add_argument("--n-seeds", type=int, default=1); p.add_argument("--seed-start", type=int, default=13000)
    p.add_argument("--profile", choices=("auto", "matlab_active", "paper_configured"), default="auto")
    p.add_argument("--solver", default="highs"); p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--run-name", required=True); p.add_argument("--output-root", default="results/robust")
    p.add_argument("--mode", choices=("normal", "prepare", "worker", "finalize"), default="normal")
    p.add_argument("--task-start", type=int); p.add_argument("--task-stop", type=int)
    return p


if __name__ == "__main__": run(parser().parse_args())
