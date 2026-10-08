"""Minimal paired experiment for legacy, V2 full, and residual DDR-MKSVM V3."""

import argparse
import contextlib
import io
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from archive.research_scripts.unit_of_work_ddr_binary import unit_of_work_ddr_binary
from archive.research_scripts.unit_of_work_ddr_multiclass import unit_of_work_ddr_multiclass
from archive.research_scripts.unit_of_work_deterministic_binary import DATASET_CONFIG as BINARY_CONFIG
from archive.research_scripts.unit_of_work_deterministic_multiclass import DATASET_CONFIG as MULTI_CONFIG


VARIANTS = {
    "legacy": dict(dnn_on=False, mkl_on=False, dro_on=False),
    "current_v2": dict(dnn_on=True, mkl_on=True, dro_on=True, epsilon=0.001,
                       extra_kernel_specs=[dict(kind="rbf", alpha=1.0)]),
    "residual_v3": dict(dnn_on=True, mkl_on=False, dro_on=True, epsilon=0.001,
                        model_version="v3_residual"),
}


def _planned_solver_calls(dataset, variant, n_outer):
    """Number of nu-grid convex programs implied by a successful run.

    This counts calls to ``solve_svm_dro`` rather than internal fallback
    attempts by CLARABEL/ECOS/SCS.  Legacy evaluates five nu values once;
    V2/V3 evaluate seven values at every outer step and once more for the
    final, training-consistent convex re-solve.  Multiclass repeats the grid
    for each one-vs-all classifier.
    """
    n_classes = 1 if dataset in BINARY_CONFIG else int(MULTI_CONFIG[dataset]["classes"])
    grid_size = 5 if variant == "legacy" else 7 * (n_outer + 1)
    return n_classes * grid_size


def _one(dataset, variant, seed, n_outer, n_inner):
    cfg = BINARY_CONFIG.get(dataset) or MULTI_CONFIG.get(dataset)
    data = pd.read_csv(os.path.join("dataset", cfg["csv_name"]))
    kwargs = dict(VARIANTS[variant])
    if variant != "legacy":
        kwargs.update(n_outer=n_outer, n_inner=n_inner)
    fn = unit_of_work_ddr_binary if dataset in BINARY_CONFIG else unit_of_work_ddr_multiclass
    start = time.perf_counter()
    planned_calls = _planned_solver_calls(dataset, variant, n_outer)
    try:
        # Solver traces remain available in direct model runs; suppress them
        # here so the controlled experiment has a compact, auditable table.
        with contextlib.redirect_stdout(io.StringIO()):
            value = fn(data, dataset_name=dataset, seed=seed, **kwargs)
        error = float(value[0] if isinstance(value, tuple) else value)
        status, failure_type, failure_message = "ok", "", ""
        completed_calls = planned_calls
    except Exception as exc:  # keep the paired pilot analyzable after one failed fit
        error = np.nan
        status = "failed"
        failure_type = type(exc).__name__
        failure_message = str(exc).replace("\n", " ")[:500]
        completed_calls = np.nan  # exact progress inside a failed grid is unknown
    return dict(
        dataset=dataset, model=variant, seed=seed, test_error=error,
        runtime_s=time.perf_counter() - start,
        n_outer=(0 if variant == "legacy" else n_outer),
        n_inner=(0 if variant == "legacy" else n_inner),
        planned_solver_calls=planned_calls,
        completed_solver_calls=completed_calls,
        status=status, failure_type=failure_type, failure_message=failure_message,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["parkinson", "blood_transfusion",
                                                           "mammographicmass_binary", "iris"])
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-outer", type=int, default=3)
    parser.add_argument("--n-inner", type=int, default=10)
    parser.add_argument("--output", default=os.path.join("results", "v3_controlled"))
    args = parser.parse_args()
    jobs = [(d, v, s) for d in args.datasets for s in range(args.n_seeds) for v in VARIANTS]
    rows = Parallel(n_jobs=args.n_jobs)(
        delayed(_one)(d, v, s, args.n_outer, args.n_inner) for d, v, s in jobs)
    per_run = pd.DataFrame(rows).sort_values(["dataset", "seed", "model"])
    summary = per_run.groupby(["dataset", "model"], as_index=False).agg(
        attempted=("status", "size"), n=("test_error", "count"),
        failures=("status", lambda values: int((values != "ok").sum())),
        mean=("test_error", "mean"), std=("test_error", "std"),
        median=("test_error", "median"), best=("test_error", "min"), worst=("test_error", "max"),
        mean_runtime_s=("runtime_s", "mean"), total_runtime_s=("runtime_s", "sum"),
        planned_solver_calls=("planned_solver_calls", "sum"),
        completed_solver_calls=("completed_solver_calls", "sum"))
    os.makedirs(args.output, exist_ok=True)
    per_run.to_csv(os.path.join(args.output, "per_run.csv"), index=False)
    summary.to_csv(os.path.join(args.output, "summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
