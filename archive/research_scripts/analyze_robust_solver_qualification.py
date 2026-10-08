"""Aggregate immutable Robust V8-C solver-qualification artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from archive.research_scripts.run_robust_matlab_parity import atomic_csv, atomic_json


def _read_tree(root: Path, label: str) -> pd.DataFrame:
    files = sorted(root.glob("**/seed-*/solver_calls.csv"))
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["experiment"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _accepted(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return bool(json.loads(value).get("accepted", False))


_CVXOPT_ITERATION = re.compile(
    r"^\s*(\d+):\s+([+\-\d.eE]+)\s+([+\-\d.eE]+)\s+"
    r"([+\-\d.eE]+)\s+([+\-\d.eE]+)\s+([+\-\d.eE]+)\s+([+\-\d.eE]+)\s*$"
)


def _cvxopt_trace_summary(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("**/seed-*/solver_logs/*cvxopt*.log")):
        iterations = []
        content = path.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            match = _CVXOPT_ITERATION.match(line)
            if match:
                iteration = int(match.group(1))
                pcost, dcost, gap, pres, dres, kt = map(float, match.groups()[1:])
                relative_gap = gap / max(1.0, abs(pcost), abs(dcost))
                iterations.append((iteration, pcost, dcost, gap, pres, dres, kt, relative_gap))
        if not iterations:
            continue
        best = min(iterations, key=lambda item: max(item[4], item[5], item[7]))
        terminal = iterations[-1]
        seed = int(next(part for part in path.parts if part.startswith("seed-")).split("-")[1])
        rows.append({
            "seed": seed,
            "iterations_recorded": len(iterations),
            "terminal_iteration": terminal[0],
            "terminal_primal_objective": terminal[1],
            "terminal_dual_objective": terminal[2],
            "terminal_gap": terminal[3],
            "terminal_primal_residual": terminal[4],
            "terminal_dual_residual": terminal[5],
            "terminal_relative_gap": terminal[7],
            "best_joint_iteration": best[0],
            "best_joint_primal_residual": best[4],
            "best_joint_dual_residual": best[5],
            "best_joint_relative_gap": best[7],
            "terminated_at_max_iterations": "maximum number of iterations reached" in content.lower(),
            "returned_solution": False,
            "log_path": str(path),
        })
    return pd.DataFrame(rows)


def analyze(result_root: Path, output: Path) -> dict:
    paths = {
        "blood_original": result_root / "blood/sq-003-boundary-complete",
        "blood_alternatives": result_root / "blood/sq-004-representative-comparison",
        "blood_zero_compression": result_root / "blood/sq-005-zero-row-compression",
        "parkinson_original": result_root / "comparisons/sq-006-parkinson-conditioning",
        "blood_qdldl": result_root / "blood/sq-007-qdldl-representative",
        "blood_deduplicated_representative": result_root / "blood/sq-008-deduplicated-representative",
        "blood_deduplicated_inner_full": result_root / "blood/sq-009-deduplicated-full-boundary",
        "blood_deduplicated_outer": result_root / "blood/sq-011-selected-outer",
        "blood_scaled_outer": result_root / "blood/sq-012-scaled-selected-outer",
        "blood_cvxopt_outer": result_root / "blood/sq-013-cvxopt-high-accuracy-outer-v2",
    }
    frames = [_read_tree(path, label) for label, path in paths.items()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("no solver qualification records found")
    calls = pd.concat(frames, ignore_index=True)
    calls["accepted"] = calls["feasibility"].map(_accepted)
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "all_solver_calls.csv", calls)

    original = calls[calls.experiment == "blood_original"].copy()
    boundary = (
        original.groupby(["seed", "alpha_rule", "C"], dropna=False)
        .agg(
            calls=("ordinal", "size"),
            exceptions=("exception_type", lambda x: int(x.notna().sum())),
            optimal=("status", lambda x: int((x == "optimal").sum())),
            optimal_inaccurate=("status", lambda x: int((x == "optimal_inaccurate").sum())),
            accepted=("accepted", "sum"),
            max_violation=("max_constraint_violation", "max"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    atomic_csv(output / "failure_boundary.csv", boundary)

    comparison = (
        calls.groupby(["experiment", "attempt", "solver"], dropna=False)
        .agg(
            calls=("ordinal", "size"),
            exceptions=("exception_type", lambda x: int(x.notna().sum())),
            optimal=("status", lambda x: int((x == "optimal").sum())),
            optimal_inaccurate=("status", lambda x: int((x == "optimal_inaccurate").sum())),
            accepted=("accepted", "sum"),
            max_violation=("max_constraint_violation", "max"),
            median_runtime_sec=("runtime_sec", "median"),
            p95_runtime_sec=("runtime_sec", lambda x: float(np.nanpercentile(x, 95))),
            max_runtime_sec=("runtime_sec", "max"),
            total_runtime_sec=("runtime_sec", "sum"),
        )
        .reset_index()
    )
    atomic_csv(output / "solver_comparison.csv", comparison)

    representatives = {0, 12, 24, 36, 48, 60, 72, 84}
    reference = original[original.ordinal.isin(representatives)].set_index(["seed", "ordinal"])
    equivalence_rows = []
    alternatives = calls[(calls.experiment != "blood_original") & (calls.phase == "inner")]
    alternatives = alternatives[alternatives.dataset == "blood_transfusion"]
    for _, row in alternatives.iterrows():
        key = (row.seed, row.ordinal)
        ref = reference.loc[key] if key in reference.index else None
        if isinstance(ref, pd.DataFrame):
            ref = ref.iloc[0]
        objective_relative = prediction_equal = loss_difference = None
        if ref is not None and pd.notna(ref.objective) and pd.notna(row.objective):
            denominator = max(1.0, abs(float(ref.objective)), abs(float(row.objective)))
            objective_relative = abs(float(ref.objective) - float(row.objective)) / denominator
            prediction_equal = bool(ref.validation_prediction_hash == row.validation_prediction_hash)
            loss_difference = float(row.validation_balanced_loss - ref.validation_balanced_loss)
        equivalence_rows.append({
            "experiment": row.experiment,
            "seed": int(row.seed),
            "ordinal": int(row.ordinal),
            "alpha_rule": row.alpha_rule,
            "attempt": row.attempt,
            "reference_status": None if ref is None else ref.status,
            "alternative_status": row.status,
            "reference_accepted": False if ref is None else bool(ref.accepted),
            "alternative_accepted": bool(row.accepted),
            "relative_objective_difference": objective_relative,
            "prediction_equal": prediction_equal,
            "balanced_loss_difference": loss_difference,
        })
    equivalence = pd.DataFrame(equivalence_rows)
    atomic_csv(output / "solver_equivalence.csv", equivalence)

    blood_rep = original[(original.seed == 15000) & original.ordinal.isin(representatives)].copy()
    park = calls[calls.experiment == "parkinson_original"].copy()
    conditioning = pd.concat([blood_rep, park], ignore_index=True)
    conditioning_columns = [
        "dataset", "seed", "ordinal", "alpha_rule", "alpha", "feature_abs_max",
        "near_zero_variance_features", "centered_feature_effective_rank",
        "centered_feature_condition_effective", "covariance_condition_effective",
        "gram_effective_rank", "gram_condition_effective", "gram_eigenvalue_min",
        "gram_eigenvalue_max", "canonical_A_abs_min_nonzero", "canonical_A_abs_max",
        "canonical_n_variables", "canonical_n_constraints", "cone_nonneg", "cone_soc",
    ]
    conditioning = conditioning[[column for column in conditioning_columns if column in conditioning]]
    atomic_csv(output / "conditioning_comparison.csv", conditioning)

    runtime = (
        calls.groupby(["experiment", "attempt", "status"], dropna=False)["runtime_sec"]
        .agg(count="size", mean="mean", median="median", maximum="max", total="sum")
        .reset_index()
    )
    runtime["p95"] = calls.groupby(["experiment", "attempt", "status"], dropna=False)["runtime_sec"].quantile(.95).values
    atomic_csv(output / "runtime_summary.csv", runtime)

    cvxopt_trace = _cvxopt_trace_summary(paths["blood_cvxopt_outer"])
    atomic_csv(output / "cvxopt_trace_summary.csv", cvxopt_trace)

    successful_original = original[original.status.notna()]
    dedup_full = calls[calls.experiment == "blood_deduplicated_inner_full"]
    dedup_outer = calls[calls.experiment == "blood_deduplicated_outer"]
    scaled_outer = calls[calls.experiment == "blood_scaled_outer"]
    cvxopt_outer = calls[calls.experiment == "blood_cvxopt_outer"]
    summary = {
        "blood_original_calls": int(len(original)),
        "blood_original_exceptions": int(original.exception_type.notna().sum()),
        "blood_original_failure_rate": float(original.exception_type.notna().mean()),
        "blood_original_optimal_inaccurate": int((original.status == "optimal_inaccurate").sum()),
        "blood_original_returned_solutions": int(len(successful_original)),
        "blood_original_optimal_inaccurate_among_returned": float((successful_original.status == "optimal_inaccurate").mean()),
        "blood_seeds_completed": 0,
        "deduplicated_inner_calls": int(len(dedup_full)),
        "deduplicated_inner_exceptions": int(dedup_full.exception_type.notna().sum()),
        "deduplicated_inner_optimal_inaccurate": int((dedup_full.status == "optimal_inaccurate").sum()),
        "deduplicated_inner_accepted": int(dedup_full.accepted.sum()),
        "selected_outer_calls": int(len(dedup_outer)),
        "selected_outer_exceptions": int(dedup_outer.exception_type.notna().sum()),
        "scaled_outer_calls": int(len(scaled_outer)),
        "scaled_outer_exceptions": int(scaled_outer.exception_type.notna().sum()),
        "cvxopt_outer_calls": int(len(cvxopt_outer)),
        "cvxopt_outer_exceptions": int(cvxopt_outer.exception_type.notna().sum()),
        "cvxopt_outer_accepted": int(cvxopt_outer.accepted.sum()),
        "scs_representative_calls": int((calls.attempt == "scs_1e-6").sum()),
        "scs_representative_accepted": int(calls[calls.attempt == "scs_1e-6"].accepted.sum()),
        "parkinson_representative_calls": int(len(park)),
        "parkinson_representative_accepted": int(park.accepted.sum()),
        "max_accepted_constraint_violation": float(calls.loc[calls.accepted, "max_constraint_violation"].max()),
        "qualification_decision": "NOT YET QUALIFIED",
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", nargs="?", default="results/robust/solver_qualification")
    parser.add_argument("--output-dir", default="results/robust/solver_qualification/comparisons/final")
    args = parser.parse_args()
    summary = analyze(Path(args.result_root), Path(args.output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
