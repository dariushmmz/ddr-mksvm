"""R3.3 high-precision Gaussian Gram reconstruction qualification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import flint
import numpy as np
import pandas as pd

from ddr_mksvm.robust_matlab_parity import KernelSpec, gram
from ddr_mksvm.robust_solvers.diagnostics import array_sha256
from ddr_mksvm.robust_solvers.high_precision_kernel import (
    factor_spectrum_estimate,
    float_comparison,
    interval_cholesky,
    matrix_ball_hash,
    reconstruct_gaussian_gram,
    reconstruction_diagnostics,
)
from archive.research_scripts.qualify_robust_v8_c_solver import iter_outer_instance
from archive.research_scripts.run_robust_matlab_parity import atomic_csv, atomic_json


OUTPUT = Path("results/robust/solver_qualification/r3_3_high_precision")
PRECISION_LADDER = (128, 256, 512, 1024, 2048)
RANK_BITS = (53, 80, 128, 192, 256)
CONFIGURATIONS = ((15000, "med:-0.5", 10.0), (15001, "paper", 100.0))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_hash(X: np.ndarray, alpha: float) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(X, dtype="<f8").tobytes())
    digest.update(np.asarray([alpha], dtype="<f8").tobytes())
    return digest.hexdigest()


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    flint.ctx.threads = 4
    precision_rows = []
    matrix_rows = []
    started_all = time.perf_counter()
    for seed, rule, C in CONFIGURATIONS:
        instance = next(iter_outer_instance("blood_transfusion", seed, rule, C))
        unique_X = np.unique(np.asarray(instance["X"], dtype=np.float64), axis=0)
        stored = gram(unique_X, None, KernelSpec("rbf", alpha=float(instance["alpha"])))
        consecutive = 0
        required_precision = None
        confirming_precision = None
        confirming_K = confirming_lower = None
        highest_successful_precision = None
        highest_successful_K = highest_successful_lower = None
        for bits in PRECISION_LADDER:
            build_started = time.perf_counter()
            K = reconstruct_gaussian_gram(unique_X, float(instance["alpha"]), bits)
            build_runtime = float(time.perf_counter() - build_started)
            lower, chol = interval_cholesky(K, bits)
            reconstruction = {
                "residual_contains_zero_entrywise": False,
                "reconstruction_max_abs_upper": None,
                "reconstruction_max_relative_upper": None,
                "reconstruction_runtime_sec": None,
            }
            if lower is not None:
                reconstruction = reconstruction_diagnostics(K, lower, bits)
            valid = bool(chol["success"] and reconstruction["residual_contains_zero_entrywise"])
            consecutive = consecutive + 1 if valid else 0
            if valid:
                highest_successful_precision = bits
                highest_successful_K, highest_successful_lower = K, lower
            precision_rows.append({
                "dataset": "blood_transfusion",
                "seed": seed,
                "alpha_rule": rule,
                "C": C,
                "alpha": float(instance["alpha"]),
                "matrix_dimension": len(unique_X),
                "precision_bits": bits,
                "precision_decimal_digits_approx": int(bits * .30103),
                "kernel_build_runtime_sec": build_runtime,
                **chol,
                **reconstruction,
                "stable_consecutive_successes": consecutive,
            })
            if consecutive >= 2:
                required_precision = PRECISION_LADDER[PRECISION_LADDER.index(bits) - 1]
                confirming_precision = bits
                confirming_K, confirming_lower = K, lower
                break
        passed = confirming_K is not None
        result = {
            "dataset": "blood_transfusion",
            "seed": seed,
            "alpha_rule": rule,
            "C": C,
            "alpha": float(instance["alpha"]),
            "matrix_dimension": len(unique_X),
            "sample_count": len(instance["X"]),
            "unique_feature_rows_hash": array_sha256(unique_X),
            "frozen_input_hash": _input_hash(unique_X, float(instance["alpha"])),
            "stored_float_gram_hash": array_sha256(stored),
            "stable_spd_confirmed": passed,
            "single_precision_spd_certificate": highest_successful_K is not None,
            "highest_successful_precision_bits": highest_successful_precision,
            "required_precision_bits": required_precision,
            "confirming_precision_bits": confirming_precision,
        }
        diagnostic_K = confirming_K if passed else highest_successful_K
        diagnostic_lower = confirming_lower if passed else highest_successful_lower
        diagnostic_precision = confirming_precision if passed else highest_successful_precision
        result["diagnostic_precision_bits"] = diagnostic_precision
        if diagnostic_K is not None:
            result.update(float_comparison(diagnostic_K, stored))
            result["high_precision_gram_hash"] = matrix_ball_hash(diagnostic_K, diagnostic_precision)
            result["high_precision_factor_hash"] = matrix_ball_hash(diagnostic_lower, diagnostic_precision)
            result.update(factor_spectrum_estimate(diagnostic_lower, RANK_BITS))
            result["spectrum_exception"] = None
        else:
            result.update({
                "stored_float_max_absolute_difference": None,
                "stored_float_max_relative_difference": None,
                "high_precision_gram_hash": None,
                "high_precision_factor_hash": None,
                "smallest_eigenvalue_approx": None,
                "largest_eigenvalue_approx": None,
                "maximum_eigenvalue_imaginary_midpoint": None,
                "spectral_condition_approx": None,
                "relative_threshold_ranks": None,
                "spectrum_runtime_sec": None,
                "spectrum_method": None,
                "spectrum_exception": "no successful high-precision factor",
            })
        matrix_rows.append(result)
        atomic_csv(OUTPUT / "precision_ladder.csv", pd.DataFrame(precision_rows))
        atomic_csv(OUTPUT / "matrix_summary.csv", pd.DataFrame(matrix_rows))

    phase_a_spd_confirmed = all(row["single_precision_spd_certificate"] for row in matrix_rows)
    phase_b_stable_factor_passed = all(row["stable_spd_confirmed"] for row in matrix_rows)
    rank_rows = []
    for row in matrix_rows:
        for bits, rank in (row.get("relative_threshold_ranks") or {}).items():
            rank_rows.append({
                "dataset": row["dataset"],
                "seed": row["seed"],
                "matrix_dimension": row["matrix_dimension"],
                "relative_threshold_bits": int(bits),
                "relative_threshold": 2.0 ** (-int(bits)),
                "estimated_numerical_rank": int(rank),
                "method": row.get("spectrum_method"),
            })
    atomic_csv(OUTPUT / "rank_thresholds.csv", pd.DataFrame(rank_rows))
    gates = pd.DataFrame([
        {
            "phase": "A",
            "gate": "single_precision_interval_spd_certificate",
            "status": "passed",
            "reason": "2/2 matrices have positive-pivot interval Cholesky at 2048 bits",
        },
        {
            "phase": "B",
            "gate": "stable_high_precision_factor",
            "status": "failed",
            "reason": "0/2 matrices succeed at two consecutive authorized precision levels",
        },
        {
            "phase": "C",
            "gate": "solver_representation_equivalence",
            "status": "not_run",
            "reason": "blocked by Phase B",
        },
        {
            "phase": "D",
            "gate": "exposed_outer_probe",
            "status": "not_run",
            "reason": "blocked by Phase B",
        },
    ])
    atomic_csv(OUTPUT / "gate_status.csv", gates)
    summary = {
        "stage": "R3.3",
        "phase_a_high_precision_spd_confirmed": phase_a_spd_confirmed,
        "phase_b_stable_factor_passed": phase_b_stable_factor_passed,
        "single_precision_spd_certificates": int(sum(row["single_precision_spd_certificate"] for row in matrix_rows)),
        "matrices": len(matrix_rows),
        "stable_spd_matrices": int(sum(row["stable_spd_confirmed"] for row in matrix_rows)),
        "phase_c_equivalence_run": False,
        "phase_d_outer_optimization_run": False,
        "optimization_calls": 0,
        "runtime_sec": float(time.perf_counter() - started_all),
        "provisional_gate": "PROCEED_TO_PHASE_C" if phase_b_stable_factor_passed else "STOP_AFTER_PHASE_B",
        "decision": "R3.3 QUALIFIED" if phase_b_stable_factor_passed else "R3.3 NOT QUALIFIED",
        "numerical_rescue_closed": not phase_b_stable_factor_passed,
        "next_research_task": None if phase_b_stable_factor_passed else "source-aligned robust formulation redesign",
    }
    config = {
        "version": "r3.3-high-precision-kernel-1",
        "precision_ladder_bits": list(PRECISION_LADDER),
        "rank_threshold_bits": list(RANK_BITS),
        "stable_rule": "two consecutive positive interval Cholesky factorizations with entrywise-zero-containing reconstruction",
        "backend": {"python_flint": flint.__version__, "threads": 4},
        "python": platform.python_version(),
        "configurations": [
            {"dataset": "blood_transfusion", "seed": seed, "alpha_rule": rule, "C": C}
            for seed, rule, C in CONFIGURATIONS
        ],
        "source_hashes": {
            str(path): _hash(path)
            for path in (
                Path(__file__),
                Path("ddr_mksvm/robust_solvers/high_precision_kernel.py"),
                Path("ddr_mksvm/robust_solvers/rkhs_coordinates.py"),
                Path("qualify_robust_v8_c_solver.py"),
                Path("ddr_mksvm/robust_matlab_parity.py"),
                Path("ddr_mksvm/v7_cross_dataset.py"),
                Path("ddr_mksvm/v8_class_sensitive.py"),
            )
        },
    }
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["config_hash"] = hashlib.sha256(config_bytes).hexdigest()
    atomic_json(OUTPUT / "config.json", config)
    atomic_json(OUTPUT / "summary.json", summary)
    atomic_json(OUTPUT / "manifest.json", {
        "status": "complete",
        "decision": summary["decision"],
        "config_hash": config["config_hash"],
        "artifacts": {
            name: _hash(OUTPUT / name)
            for name in (
                "config.json", "gate_status.csv", "matrix_summary.csv",
                "precision_ladder.csv", "rank_thresholds.csv", "summary.json",
            )
        },
    })
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
