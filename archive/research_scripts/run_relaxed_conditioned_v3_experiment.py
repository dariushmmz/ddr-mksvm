"""Matched experiment including the separate relaxed-conditioned V3 variant."""

import argparse
import contextlib
import io
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ddr_mksvm.optim.conditioned_convex_subproblem import (
    clear_conditioning_records,
    get_conditioning_records,
)
from archive.research_scripts.run_conditioned_v3_experiment import (
    VARIANTS as CONDITIONED_VARIANTS,
    _one as _conditioned_one,
)
from archive.research_scripts.run_v3_experiment import _planned_solver_calls
from archive.research_scripts.unit_of_work_ddr_binary import unit_of_work_ddr_binary
from archive.research_scripts.unit_of_work_ddr_multiclass import unit_of_work_ddr_multiclass
from archive.research_scripts.unit_of_work_deterministic_binary import DATASET_CONFIG as BINARY_CONFIG
from archive.research_scripts.unit_of_work_deterministic_multiclass import DATASET_CONFIG as MULTI_CONFIG


VARIANTS = {
    **CONDITIONED_VARIANTS,
    "relaxed_conditioned_v3": dict(
        dnn_on=True,
        mkl_on=False,
        dro_on=True,
        epsilon=0.001,
        model_version="v3_relaxed_conditioned",
    ),
}


def _one(dataset, variant, seed, n_outer, n_inner):
    if variant in CONDITIONED_VARIANTS:
        return _conditioned_one(dataset, variant, seed, n_outer, n_inner)

    cfg = BINARY_CONFIG.get(dataset) or MULTI_CONFIG.get(dataset)
    data = pd.read_csv(os.path.join("dataset", cfg["csv_name"]))
    kwargs = dict(VARIANTS[variant])
    kwargs.update(n_outer=n_outer, n_inner=n_inner)
    fn = unit_of_work_ddr_binary if dataset in BINARY_CONFIG else unit_of_work_ddr_multiclass
    planned_calls = _planned_solver_calls(dataset, variant, n_outer)
    clear_conditioning_records()
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("always")
            value = fn(data, dataset_name=dataset, seed=seed, **kwargs)
        error = float(value[0] if isinstance(value, tuple) else value)
        status, failure_type, failure_message = "ok", "", ""
        completed_calls = planned_calls
    except Exception as exc:
        caught = []
        error = np.nan
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc).replace("\n", " ")[:500]
        completed_calls = np.nan

    inaccurate_warnings = sum(
        "accepting optimal_inaccurate solver result" in str(item.message)
        for item in caught
    )
    conditioning = get_conditioning_records()
    shifts = [record["diagonal_shift"] for record in conditioning]
    original_conditions = [record["original_condition_number"] for record in conditioning]
    conditioned_conditions = [record["conditioned_condition_number"] for record in conditioning]
    return dict(
        dataset=dataset,
        model=variant,
        seed=seed,
        test_error=error,
        runtime_s=time.perf_counter() - start,
        n_outer=n_outer,
        n_inner=n_inner,
        planned_solver_calls=planned_calls,
        completed_solver_calls=completed_calls,
        optimal_inaccurate_warnings=inaccurate_warnings,
        conditioning_events=len(conditioning),
        mean_diagonal_shift=(float(np.mean(shifts)) if shifts else np.nan),
        max_diagonal_shift=(float(np.max(shifts)) if shifts else np.nan),
        max_original_condition=(float(np.max(original_conditions)) if original_conditions else np.nan),
        max_conditioned_condition=(float(np.max(conditioned_conditions)) if conditioned_conditions else np.nan),
        reported_solver_time_s=(
            float(sum(record["reported_solver_time"] for record in conditioning))
            if conditioning else np.nan
        ),
        status=status,
        failure_type=failure_type,
        failure_message=failure_message,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["parkinson", "blood_transfusion", "mammographicmass_binary", "iris"],
    )
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-outer", type=int, default=1)
    parser.add_argument("--n-inner", type=int, default=2)
    parser.add_argument("--output", default=os.path.join("results", "relaxed_conditioned_v3"))
    args = parser.parse_args()

    jobs = [
        (dataset, variant, seed)
        for dataset in args.datasets
        for seed in range(args.n_seeds)
        for variant in VARIANTS
    ]
    rows = Parallel(n_jobs=args.n_jobs)(
        delayed(_one)(dataset, variant, seed, args.n_outer, args.n_inner)
        for dataset, variant, seed in jobs
    )
    per_run = pd.DataFrame(rows).sort_values(["dataset", "seed", "model"])
    summary = per_run.groupby(["dataset", "model"], as_index=False).agg(
        attempted=("status", "size"),
        n=("test_error", "count"),
        failures=("status", lambda values: int((values != "ok").sum())),
        mean=("test_error", "mean"),
        std=("test_error", "std"),
        median=("test_error", "median"),
        mean_runtime_s=("runtime_s", "mean"),
        total_runtime_s=("runtime_s", "sum"),
        planned_solver_calls=("planned_solver_calls", "sum"),
        completed_solver_calls=("completed_solver_calls", "sum"),
        optimal_inaccurate_warnings=("optimal_inaccurate_warnings", "sum"),
        conditioning_events=("conditioning_events", "sum"),
        max_diagonal_shift=("max_diagonal_shift", "max"),
        max_original_condition=("max_original_condition", "max"),
        max_conditioned_condition=("max_conditioned_condition", "max"),
        reported_solver_time_s=("reported_solver_time_s", "sum"),
    )
    os.makedirs(args.output, exist_ok=True)
    per_run.to_csv(os.path.join(args.output, "per_run.csv"), index=False)
    summary.to_csv(os.path.join(args.output, "summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
