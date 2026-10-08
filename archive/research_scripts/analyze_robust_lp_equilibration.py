"""Create the immutable combined R8.1 Parkinson/Blood qualification tables."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ddr_mksvm.lp_equilibration import equilibrate_lp, positive_dynamic_range
from ddr_mksvm.robust_matlab_parity import gram, uncertainty_delta
from ddr_mksvm.robust_source_aligned import FROZEN_P, FROZEN_TAU, build_weighted_q1_lp
from archive.research_scripts.run_robust_lp_equilibration import MODEL_CONFIG, MODELS, problem_coefficients
from archive.research_scripts.run_robust_matlab_parity import atomic_csv, atomic_json


def failed_scaled_range(dataset: str, seed: int, model: str) -> dict:
    inputs = problem_coefficients(dataset, seed, model)
    cfg = MODEL_CONFIG[model]
    K = gram(inputs["Xfit"], None, inputs["kernel"])
    delta = uncertainty_delta(
        inputs["Xfit"],
        inputs["labels"],
        cfg["rho"],
        FROZEN_P,
        inputs["kernel"],
        matlab_semantics=True,
    )
    c, A, b, bounds, _ = build_weighted_q1_lp(
        K,
        inputs["yfit"],
        1e-3,
        delta,
        tau=FROZEN_TAU,
        class_sensitive=cfg["class_sensitive"],
    )
    transformed = equilibrate_lp(c, A, b, bounds)
    return {
        "unscaled_dynamic_range": positive_dynamic_range(A, b, c),
        "scaled_dynamic_range": positive_dynamic_range(
            transformed.A, transformed.b, transformed.c
        ),
        "row_scale_min": float(np.min(transformed.row_scale)),
        "row_scale_max": float(np.max(transformed.row_scale)),
        "variable_scale_min": float(np.min(transformed.variable_scale)),
        "variable_scale_max": float(np.max(transformed.variable_scale)),
    }


def analyze(root: Path) -> dict:
    parkinson_dir = root / "r8-1-parkinson-equivalence-v1"
    blood_dir = root / "r8-1-blood-exposed-v1"
    historical = root.parent / "exposed_qualification" / "r8-q1-cs-p2-rho001-exposed-v3"
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    parkinson = pd.read_csv(parkinson_dir / "per_run.csv")
    equivalence = pd.read_csv(parkinson_dir / "equivalence.csv")
    blood = pd.read_csv(blood_dir / "per_run.csv")
    blood_calls = pd.read_csv(blood_dir / "solver_calls.csv")
    blood_failures = pd.read_csv(blood_dir / "failures.csv")
    historical_failures = pd.read_csv(historical / "solver_failures.csv")

    rows = []
    scaled_lookup = {(int(row.seed), row.model): row for _, row in blood.iterrows()}
    failed_lookup = {(int(row.seed), row.model): row for _, row in blood_failures.iterrows()}
    original_lookup = {
        (int(row.seed), row.model): row for _, row in historical_failures.iterrows()
        if row.dataset == "blood_transfusion"
    }
    first_calls = blood_calls[blood_calls.nu_order == 0].set_index(["seed", "model"])
    selected_calls = blood_calls[blood_calls.selected == True].set_index(["seed", "model"])
    for seed in (15000, 15001):
        for model in MODELS:
            key = (seed, model)
            original = original_lookup[key]
            result = scaled_lookup.get(key)
            if result is not None:
                first = first_calls.loc[key]
                selected = selected_calls.loc[key]
                scale = {
                    "unscaled_dynamic_range": float(first.unscaled_canonical_dynamic_range),
                    "scaled_dynamic_range": float(first.scaled_canonical_dynamic_range),
                    "row_scale_min": float(first.row_scale_min),
                    "row_scale_max": float(first.row_scale_max),
                    "variable_scale_min": float(first.variable_scale_min),
                    "variable_scale_max": float(first.variable_scale_max),
                }
                status = result.solver_status
                status_code = int(result.solver_status_code)
                solver_iterations = int(result.solver_iterations)
                selected_iterations = int(selected.solver_iterations)
                objective = float(result.objective_value)
                violation = float(result.all_calls_max_constraint_violation)
                selected_nu = float(result.selected_nu)
                threshold = float(result.threshold)
                prediction_hash = result.prediction_hash
                runtime = float(result.runtime_s)
                error = float(result.test_error)
                balanced_accuracy = float(result.balanced_accuracy)
                macro_f1 = float(result.macro_f1)
                minority_recall = float(result.minority_recall)
                class_recall = result.per_class_recall
                accepted = bool(result.all_solver_calls_accepted)
            else:
                scale = failed_scaled_range("blood_transfusion", seed, model)
                failure = failed_lookup[key]
                status = failure.exception_text
                status_code = 4
                solver_iterations = None
                selected_iterations = None
                objective = violation = selected_nu = threshold = None
                prediction_hash = None
                runtime = error = balanced_accuracy = macro_f1 = minority_recall = None
                class_recall = None
                accepted = False
            rows.append({
                "dataset": "blood_transfusion",
                "seed": seed,
                "model": model,
                "unscaled_status_code": int(original.solver_status_code),
                "unscaled_status": original.solver_status,
                "unscaled_dynamic_range": scale["unscaled_dynamic_range"],
                "scaled_dynamic_range": scale["scaled_dynamic_range"],
                "dynamic_range_reduction_factor": scale["unscaled_dynamic_range"] / scale["scaled_dynamic_range"],
                "row_scale_min": scale["row_scale_min"],
                "row_scale_max": scale["row_scale_max"],
                "variable_scale_min": scale["variable_scale_min"],
                "variable_scale_max": scale["variable_scale_max"],
                "scaled_status_code": status_code,
                "scaled_status": status,
                "accepted": accepted,
                "total_iterations": solver_iterations,
                "selected_iterations": selected_iterations,
                "objective_original_coordinates": objective,
                "max_original_coordinate_violation": violation,
                "selected_nu": selected_nu,
                "threshold": threshold,
                "prediction_hash": prediction_hash,
                "runtime_s": runtime,
                "test_error": error,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "minority_recall": minority_recall,
                "per_class_recall": class_recall,
            })
    blood_table = pd.DataFrame(rows)
    atomic_csv(output / "blood_qualification.csv", blood_table)
    atomic_csv(output / "parkinson_equivalence.csv", equivalence)
    atomic_csv(output / "parkinson_runs.csv", parkinson)

    decision = bool(
        len(equivalence) == 4
        and equivalence[[
            "selected_nu_equal",
            "prediction_equal",
            "tie_count_equal",
            "threshold_equivalent",
            "objective_equivalent",
            "scaled_feasible",
            "status_no_regression",
        ]].all(axis=None)
        and len(blood_table) == 8
        and blood_table.accepted.all()
        and blood_table.max_original_coordinate_violation.max() <= 1e-7
    )
    summary = {
        "verdict": "R8.1 NUMERICALLY QUALIFIED" if decision else "R8.1 NOT QUALIFIED",
        "parkinson_equivalence_pass": bool(len(equivalence) == 4 and equivalence.objective_equivalent.all() and equivalence.prediction_equal.all()),
        "blood_expected_models": 8,
        "blood_accepted_models": int(blood_table.accepted.sum()),
        "blood_failed_models": int((~blood_table.accepted).sum()),
        "blood_max_returned_original_coordinate_violation": float(blood_table.max_original_coordinate_violation.dropna().max()),
        "fresh_seeds_consumed": [],
        "gate8_launched": False,
        "gate24_launched": False,
        "confirmation_launched": False,
    }
    atomic_json(output / "qualification_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/robust/source_aligned_redesign/lp_equilibration",
    )
    print(json.dumps(analyze(Path(parser.parse_args().root)), indent=2))
