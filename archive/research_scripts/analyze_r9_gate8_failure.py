"""R10 deterministic reconstruction of the failed R9 Gate8.

This script never calls an optimizer.  It reconstructs scores, margins and
counterfactual threshold swaps from the selected primal vectors stored in the
immutable Gate8 checkpoints and fails closed if predictions do not reproduce.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from ddr_mksvm.robust_matlab_parity import KernelSpec, array_hash, class_eta, gram
from ddr_mksvm.robust_source_aligned import sign_group_weights
from archive.research_scripts.run_robust_matlab_parity import digest, prepare

REFERENCE = "weighted_deterministic_rbf_q1"
CANDIDATE = "weighted_robust_rbf_q1"
DATASETS = ("blood_transfusion", "parkinson", "mammographicmass_binary", "iris")
SEEDS = tuple(range(15300, 15308))
SUPPORT_TOLERANCES = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)


def atomic_csv(path: Path, rows) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def distribution(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {f"{prefix}_{name}": np.nan for name in (
            "mean", "std", "min", "q05", "q25", "median", "q75", "q95", "max"
        )}
    quantiles = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q95": float(quantiles[4]),
        f"{prefix}_max": float(np.max(values)),
    }


def metric_bundle(ytrue: np.ndarray, prediction: np.ndarray, training_labels: np.ndarray) -> dict:
    classes = sorted(np.unique(ytrue).tolist())
    recalls = []
    for label in classes:
        mask = ytrue == label
        recalls.append(float(np.mean(prediction[mask] == label)))
    counts = {label: int(np.sum(training_labels == label)) for label in classes}
    tracked = min(classes, key=lambda label: (counts[label], label))
    tracked_recall = float(np.mean(prediction[ytrue == tracked] == tracked))
    safety = tracked_recall if len(classes) == 2 else float(min(recalls))
    return {
        "error": float(np.mean(prediction != ytrue)),
        "balanced_accuracy": float(balanced_accuracy_score(ytrue, prediction)),
        "macro_f1": float(f1_score(ytrue, prediction, average="macro", zero_division=0)),
        "safety_recall": safety,
        "tracked_class": int(tracked),
        "tracked_class_recall": tracked_recall,
        "per_class_recall": json.dumps(dict(zip(map(str, classes), recalls)), sort_keys=True),
    }


def selected_diagnostic(checkpoint: dict, task_index: int, multiclass: bool) -> dict:
    rows = [row for row in checkpoint["solver_diagnostics"] if row["selected"]]
    if multiclass:
        rows = [row for row in rows if int(row["task"]) == task_index]
    if len(rows) != 1:
        raise RuntimeError("selected diagnostic is not unique")
    return rows[0]


def reconstruct_model(
    checkpoint: dict,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    model_name: str,
    model_rows: list,
    uncertainty_rows: list,
    train_rows: list,
    test_rows: list,
):
    parameters = checkpoint["fitted_parameters"]
    multiclass = len(parameters) > 1
    alpha = float(checkpoint["record"]["alpha"])
    kernel = KernelSpec("rbf", alpha=alpha)
    task_objects = []
    final_train_columns = []
    final_test_columns = []
    raw_train_columns = []
    raw_test_columns = []

    for task_index, parameter in enumerate(parameters):
        positive = int(parameter["positive_class"])
        if multiclass:
            Xbasis = Xtr
            original_labels = ytr
        else:
            order = np.asarray(parameter["training_order_within_split"], dtype=int)
            Xbasis = Xtr[order]
            original_labels = ytr[order]
        binary = np.where(original_labels == positive, 1.0, -1.0)
        u = np.asarray(parameter["u"], dtype=float)
        xi = np.asarray(parameter["xi"], dtype=float)
        delta = np.asarray(parameter["delta"], dtype=float)
        gamma = float(parameter["gamma"])
        threshold = float(parameter["b"])
        Ktrain = gram(Xbasis, None, kernel)
        Ktest = gram(Xbasis, Xte, kernel)
        raw_train = Ktrain @ (binary * u)
        raw_test = Ktest.T @ (binary * u)
        final_train = raw_train - threshold
        final_test = raw_test - threshold
        l1 = float(np.sum(np.abs(u)))
        robust_tightening = delta * l1
        nominal_optimization_margin = binary * (raw_train - gamma)
        adjusted_optimization_margin = nominal_optimization_margin - robust_tightening
        final_signed_margin = binary * final_train
        diagnostic = selected_diagnostic(checkpoint, task_index, multiclass)
        weights = sign_group_weights(binary, tau=0.5)
        weighted_slack = float(float(diagnostic["nu"]) * (weights @ xi))
        objective = l1 + weighted_slack
        if abs(objective - float(diagnostic["objective_value"])) > 1e-7 * max(1.0, abs(objective)):
            raise RuntimeError("stored objective decomposition does not reproduce")

        eta = class_eta(Xbasis, original_labels, float(checkpoint["record"]["rho"]))
        class_delta = {}
        for label in sorted(np.unique(original_labels).tolist()):
            unique = np.unique(delta[original_labels == label])
            if unique.size != 1:
                raise RuntimeError("R9 RBF delta is not class-constant")
            class_delta[int(label)] = float(unique[0])
            uncertainty_rows.append({
                "dataset": checkpoint["record"]["dataset"],
                "seed": int(checkpoint["record"]["seed"]),
                "model": model_name,
                "task": task_index,
                "positive_class": positive,
                "original_class": int(label),
                "alpha": alpha,
                "eta_c": float(eta[int(label)]),
                "delta_c": class_delta[int(label)],
                "u_l1": l1,
                "effective_robust_penalty": class_delta[int(label)] * l1,
            })

        row = {
            "dataset": checkpoint["record"]["dataset"],
            "seed": int(checkpoint["record"]["seed"]),
            "model": model_name,
            "task": task_index,
            "positive_class": positive,
            "selected_nu": float(diagnostic["nu"]),
            "alpha": alpha,
            "gamma": gamma,
            "threshold": threshold,
            "threshold_minus_gamma": threshold - gamma,
            "u_l1": l1,
            "l1_coefficient_term": l1,
            "weighted_slack_term": weighted_slack,
            "objective_reconstructed": objective,
            "objective_stored": float(diagnostic["objective_value"]),
            "objective_reconstruction_residual": objective - float(diagnostic["objective_value"]),
            "robust_margin_contribution_sum": float(np.sum(robust_tightening)),
            "robust_margin_contribution_mean": float(np.mean(robust_tightening)),
            "robust_margin_contribution_max": float(np.max(robust_tightening)),
            "xi_sum": float(np.sum(xi)),
            "xi_weighted_sum": float(weights @ xi),
            "coefficient_abs_mean": float(np.mean(np.abs(u))),
            "coefficient_abs_max": float(np.max(np.abs(u))),
        }
        for tolerance in SUPPORT_TOLERANCES:
            name = "exact" if tolerance == 0 else f"gt_{tolerance:.0e}".replace("-", "m")
            row[f"support_count_{name}"] = int(np.sum(np.abs(u) > tolerance))
        model_rows.append(row)

        for label in sorted(np.unique(original_labels).tolist()):
            mask = original_labels == label
            group = {
                "dataset": row["dataset"], "seed": row["seed"], "model": model_name,
                "task": task_index, "positive_class": positive,
                "original_class": int(label), "sign_group": int(binary[mask][0]),
                "n": int(np.sum(mask)),
                "xi_positive_count": int(np.sum(xi[mask] > 1e-12)),
                "final_margin_nonpositive_count": int(np.sum(final_signed_margin[mask] <= 0)),
                "adjusted_margin_below_one_count": int(np.sum(adjusted_optimization_margin[mask] < 1 - 1e-9)),
            }
            for values, prefix in (
                (raw_train[mask], "raw_score"),
                (final_train[mask], "final_score"),
                (nominal_optimization_margin[mask], "nominal_optimization_margin"),
                (adjusted_optimization_margin[mask], "robust_adjusted_margin"),
                (final_signed_margin[mask], "final_signed_margin"),
                (xi[mask], "slack"),
                (robust_tightening[mask], "robust_tightening"),
            ):
                group.update(distribution(values, prefix))
            train_rows.append(group)

        for label in sorted(np.unique(yte).tolist()):
            mask = yte == label
            group = {
                "dataset": row["dataset"], "seed": row["seed"], "model": model_name,
                "task": task_index, "positive_class": positive,
                "true_class": int(label), "n": int(np.sum(mask)),
            }
            group.update(distribution(raw_test[mask], "raw_score"))
            group.update(distribution(final_test[mask], "final_score"))
            test_rows.append(group)

        raw_train_columns.append(raw_train)
        raw_test_columns.append(raw_test)
        final_train_columns.append(final_train)
        final_test_columns.append(final_test)
        task_objects.append({
            "positive": positive, "b": threshold, "gamma": gamma, "u": u,
            "raw_train": raw_train, "raw_test": raw_test,
        })

    raw_train_matrix = np.column_stack(raw_train_columns)
    raw_test_matrix = np.column_stack(raw_test_columns)
    final_train_matrix = np.column_stack(final_train_columns)
    final_test_matrix = np.column_stack(final_test_columns)
    if multiclass:
        classes = np.asarray([task["positive"] for task in task_objects])
        train_prediction = classes[np.argmax(final_train_matrix, axis=1)]
        test_prediction = classes[np.argmax(final_test_matrix, axis=1)]
        training_labels = ytr
        score_check = final_test_matrix
    else:
        negative = int(parameters[0]["negative_class"])
        positive = int(parameters[0]["positive_class"])
        train_prediction = np.where(final_train_matrix[:, 0] > 0, positive, negative)
        test_prediction = np.where(final_test_matrix[:, 0] > 0, positive, negative)
        training_labels = ytr[np.asarray(parameters[0]["training_order_within_split"], dtype=int)]
        score_check = final_test_matrix[:, 0]
    stored_prediction = np.asarray(checkpoint["predictions"], dtype=int)
    stored_scores = np.asarray(checkpoint["scores"], dtype=float)
    if not np.array_equal(test_prediction, stored_prediction):
        raise RuntimeError("reconstructed predictions differ from checkpoint")
    if not np.allclose(score_check, stored_scores, rtol=1e-12, atol=1e-12):
        raise RuntimeError("reconstructed scores differ from checkpoint")
    if array_hash(test_prediction, "<i8") != checkpoint["record"]["prediction_hash"]:
        raise RuntimeError("reconstructed prediction hash differs")
    return {
        "tasks": task_objects,
        "multiclass": multiclass,
        "raw_train": raw_train_matrix,
        "raw_test": raw_test_matrix,
        "final_train": final_train_matrix,
        "final_test": final_test_matrix,
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
        "training_labels": training_labels,
        "test_labels": yte,
        "checkpoint": checkpoint,
    }


def predict_with_thresholds(model: dict, thresholds: np.ndarray) -> np.ndarray:
    scores = model["raw_test"] - thresholds[None, :]
    if model["multiclass"]:
        classes = np.asarray([task["positive"] for task in model["tasks"]])
        return classes[np.argmax(scores, axis=1)]
    parameter = model["checkpoint"]["fitted_parameters"][0]
    return np.where(scores[:, 0] > 0, int(parameter["positive_class"]), int(parameter["negative_class"]))


def run(input_dir: Path, output_dir: Path) -> dict:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Gate8 input manifest is not complete")
    model_rows, uncertainty_rows, train_rows, test_rows = [], [], [], []
    flip_rows, confusion_rows, threshold_rows, pair_rows, train_prediction_rows = [], [], [], [], []
    missing_rows = []

    for dataset in DATASETS:
        for seed in SEEDS:
            paths = {
                model: input_dir / "checkpoints" / f"{dataset}-{seed}-{model}.json"
                for model in (REFERENCE, CANDIDATE)
            }
            absent = [model for model, path in paths.items() if not path.exists()]
            if absent:
                missing_rows.append({"dataset": dataset, "seed": seed, "missing_models": json.dumps(absent)})
                continue
            checkpoints = {model: json.loads(path.read_text(encoding="utf-8")) for model, path in paths.items()}
            Xtr, ytr, Xte, yte, _, split, _ = prepare(dataset, seed, "paper_configured")
            for checkpoint in checkpoints.values():
                if checkpoint["split"]["train_hash"] != split["train_hash"] or checkpoint["split"]["test_hash"] != split["test_hash"]:
                    raise RuntimeError("reconstructed split differs from checkpoint")
            models = {
                name: reconstruct_model(checkpoint, Xtr, ytr, Xte, yte, name,
                    model_rows, uncertainty_rows, train_rows, test_rows)
                for name, checkpoint in checkpoints.items()
            }
            reference, candidate = models[REFERENCE], models[CANDIDATE]
            if reference["multiclass"] != candidate["multiclass"] or len(reference["tasks"]) != len(candidate["tasks"]):
                raise RuntimeError("paired task construction differs")

            for name, model in models.items():
                for true_label in sorted(np.unique(model["training_labels"]).tolist()):
                    for predicted_label in sorted(np.unique(model["training_labels"]).tolist()):
                        train_prediction_rows.append({
                            "dataset": dataset, "seed": seed, "model": name,
                            "true_class": int(true_label), "predicted_class": int(predicted_label),
                            "count": int(np.sum((model["training_labels"] == true_label) & (model["train_prediction"] == predicted_label))),
                        })

            reference_prediction = reference["test_prediction"]
            candidate_prediction = candidate["test_prediction"]
            reference_correct = reference_prediction == yte
            candidate_correct = candidate_prediction == yte
            tracked_counts = {label: int(np.sum(ytr == label)) for label in np.unique(ytr)}
            tracked_class = min(tracked_counts, key=lambda label: (tracked_counts[label], label))
            for label in [None] + sorted(np.unique(yte).tolist()):
                mask = np.ones(len(yte), dtype=bool) if label is None else yte == label
                flip_rows.append({
                    "dataset": dataset, "seed": seed,
                    "scope": "overall" if label is None else "class",
                    "true_class": None if label is None else int(label),
                    "tracked_safety_class": int(tracked_class),
                    "n": int(np.sum(mask)),
                    "prediction_flips": int(np.sum(mask & (reference_prediction != candidate_prediction))),
                    "deterministic_errors_corrected": int(np.sum(mask & ~reference_correct & candidate_correct)),
                    "new_errors_introduced": int(np.sum(mask & reference_correct & ~candidate_correct)),
                    "both_correct": int(np.sum(mask & reference_correct & candidate_correct)),
                    "both_wrong": int(np.sum(mask & ~reference_correct & ~candidate_correct)),
                })

            labels = sorted(np.unique(yte).tolist())
            cm_reference = confusion_matrix(yte, reference_prediction, labels=labels)
            cm_candidate = confusion_matrix(yte, candidate_prediction, labels=labels)
            for row_index, true_label in enumerate(labels):
                for column_index, predicted_label in enumerate(labels):
                    confusion_rows.append({
                        "dataset": dataset, "seed": seed,
                        "true_class": int(true_label), "predicted_class": int(predicted_label),
                        "reference_count": int(cm_reference[row_index, column_index]),
                        "candidate_count": int(cm_candidate[row_index, column_index]),
                        "delta_count": int(cm_candidate[row_index, column_index] - cm_reference[row_index, column_index]),
                    })

            for task_index, (reference_task, candidate_task) in enumerate(zip(reference["tasks"], candidate["tasks"])):
                threshold_rows.append({
                    "dataset": dataset, "seed": seed, "task": task_index,
                    "positive_class": int(reference_task["positive"]),
                    "reference_gamma": float(reference_task["gamma"]),
                    "candidate_gamma": float(candidate_task["gamma"]),
                    "delta_gamma": float(candidate_task["gamma"] - reference_task["gamma"]),
                    "reference_threshold": float(reference_task["b"]),
                    "candidate_threshold": float(candidate_task["b"]),
                    "delta_threshold": float(candidate_task["b"] - reference_task["b"]),
                    **distribution(candidate_task["raw_test"] - reference_task["raw_test"], "raw_test_score_difference"),
                })

            reference_thresholds = np.asarray([task["b"] for task in reference["tasks"]])
            candidate_thresholds = np.asarray([task["b"] for task in candidate["tasks"]])
            function_only_prediction = predict_with_thresholds(candidate, reference_thresholds)
            threshold_only_prediction = predict_with_thresholds(reference, candidate_thresholds)
            bundles = {
                "reference": metric_bundle(yte, reference_prediction, ytr),
                "candidate": metric_bundle(yte, candidate_prediction, ytr),
                "function_only": metric_bundle(yte, function_only_prediction, ytr),
                "threshold_only": metric_bundle(yte, threshold_only_prediction, ytr),
            }
            row = {
                "dataset": dataset, "seed": seed,
                "n_test": len(yte),
                "actual_prediction_flips": int(np.sum(reference_prediction != candidate_prediction)),
                "function_only_flips_vs_reference": int(np.sum(function_only_prediction != reference_prediction)),
                "threshold_only_flips_vs_reference": int(np.sum(threshold_only_prediction != reference_prediction)),
                "function_only_agreement_with_candidate": float(np.mean(function_only_prediction == candidate_prediction)),
                "threshold_only_agreement_with_candidate": float(np.mean(threshold_only_prediction == candidate_prediction)),
                "mean_abs_threshold_shift": float(np.mean(np.abs(candidate_thresholds - reference_thresholds))),
                "mean_abs_raw_test_score_shift": float(np.mean(np.abs(candidate["raw_test"] - reference["raw_test"]))),
                "reference_selected_nu": json.dumps([float(selected_diagnostic(checkpoints[REFERENCE], i, reference["multiclass"])["nu"]) for i in range(len(reference["tasks"]))]),
                "candidate_selected_nu": json.dumps([float(selected_diagnostic(checkpoints[CANDIDATE], i, candidate["multiclass"])["nu"]) for i in range(len(candidate["tasks"]))]),
                "selected_nu_equal": all(
                    float(selected_diagnostic(checkpoints[REFERENCE], i, reference["multiclass"])["nu"])
                    == float(selected_diagnostic(checkpoints[CANDIDATE], i, candidate["multiclass"])["nu"])
                    for i in range(len(reference["tasks"]))
                ),
            }
            for prefix, bundle in bundles.items():
                for key, value in bundle.items():
                    row[f"{prefix}_{key}"] = value
            pair_rows.append(row)

    atomic_csv(output_dir / "paired_model_objective_diagnostics.csv", model_rows)
    atomic_csv(output_dir / "class_uncertainty_penalties.csv", uncertainty_rows)
    atomic_csv(output_dir / "training_margin_slack_distributions.csv", train_rows)
    atomic_csv(output_dir / "test_score_distributions.csv", test_rows)
    atomic_csv(output_dir / "train_prediction_counts.csv", train_prediction_rows)
    atomic_csv(output_dir / "prediction_flips.csv", flip_rows)
    atomic_csv(output_dir / "confusion_matrix_changes.csv", confusion_rows)
    atomic_csv(output_dir / "threshold_score_movements.csv", threshold_rows)
    atomic_csv(output_dir / "paired_seed_mechanisms.csv", pair_rows)
    atomic_csv(output_dir / "missing_pairs.csv", missing_rows)

    objective_frame = pd.DataFrame(model_rows)
    objective_metrics = [
        "selected_nu", "alpha", "gamma", "threshold", "u_l1",
        "weighted_slack_term", "objective_reconstructed", "xi_sum",
        "support_count_gt_1em10", "support_count_gt_1em08",
        "robust_margin_contribution_mean",
    ]
    objective_pairs = objective_frame.pivot(
        index=["dataset", "seed", "task", "positive_class"],
        columns="model", values=objective_metrics,
    )
    objective_pairs.columns = [f"{metric}_{'reference' if model == REFERENCE else 'candidate'}" for metric, model in objective_pairs.columns]
    objective_pairs = objective_pairs.reset_index()
    for metric in objective_metrics:
        objective_pairs[f"{metric}_delta"] = objective_pairs[f"{metric}_candidate"] - objective_pairs[f"{metric}_reference"]
    atomic_csv(output_dir / "paired_objective_deltas.csv", objective_pairs)

    training_frame = pd.DataFrame(train_rows)
    training_keys = ["dataset", "seed", "task", "positive_class", "original_class", "sign_group"]
    training_metrics = [
        "final_signed_margin_mean", "nominal_optimization_margin_mean",
        "robust_adjusted_margin_mean", "slack_mean", "slack_q95",
        "xi_positive_count", "final_margin_nonpositive_count",
    ]
    training_pairs = training_frame.pivot(index=training_keys, columns="model", values=training_metrics)
    training_pairs.columns = [f"{metric}_{'reference' if model == REFERENCE else 'candidate'}" for metric, model in training_pairs.columns]
    training_pairs = training_pairs.reset_index()
    for metric in training_metrics:
        training_pairs[f"{metric}_delta"] = training_pairs[f"{metric}_candidate"] - training_pairs[f"{metric}_reference"]
    atomic_csv(output_dir / "paired_training_class_deltas.csv", training_pairs)

    mechanism_frame = pd.DataFrame(pair_rows)
    numeric_mechanisms = mechanism_frame.select_dtypes(include=[np.number]).columns.difference(["seed"])
    mechanism_frame.groupby("dataset", sort=False)[list(numeric_mechanisms)].mean().reset_index().to_csv(
        output_dir / "dataset_mechanism_summary.csv", index=False
    )
    flip_frame = pd.DataFrame(flip_rows)
    flip_frame[flip_frame.scope == "overall"].groupby("dataset", sort=False)[
        ["n", "prediction_flips", "deterministic_errors_corrected", "new_errors_introduced", "both_correct", "both_wrong"]
    ].sum().reset_index().to_csv(output_dir / "dataset_flip_summary.csv", index=False)

    checkpoint_paths = sorted((input_dir / "checkpoints").glob("*.json"))
    provenance = {
        "analysis": "R10 deterministic checkpoint reconstruction; no fitting or solver calls",
        "input_run": str(input_dir),
        "input_manifest_sha256": digest(input_dir / "manifest.json"),
        "input_fingerprint": manifest["fingerprint"],
        "checkpoint_count_in_input": len(checkpoint_paths),
        "paired_seed_count": len(pair_rows),
        "missing_pair_count": len(missing_rows),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "models": [REFERENCE, CANDIDATE],
        "solver_calls": 0,
        "new_model_fits": 0,
        "gate24_or_confirmation_accessed": False,
        "analyzer_sha256": digest(__file__),
        "input_checkpoint_set_sha256": hashlib.sha256("".join(digest(path) for path in checkpoint_paths).encode()).hexdigest(),
    }
    atomic_json(output_dir / "provenance.json", provenance)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/robust/bounded_rbf_redesign/gates/r9-gate8-15300-15307-v1")
    parser.add_argument("--output-dir", default="results/robust/bounded_rbf_redesign/r10_failure_analysis")
    arguments = parser.parse_args()
    print(json.dumps(run(Path(arguments.input_dir), Path(arguments.output_dir)), indent=2))
