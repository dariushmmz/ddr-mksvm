import json

import numpy as np
import pytest

from ddr_mksvm.robust_solvers.feasibility import FeasibilityThresholds, assess_feasibility
from ddr_mksvm.robust_solvers.policy import AttemptResult, SolverAttempt, SolverPolicy, run_solver_policy
from ddr_mksvm.robust_solvers.registry import qualification_attempts
from ddr_mksvm.robust_solvers.registry import production_policy
from ddr_mksvm.robust_solvers.diagnostics import canonical_problem_summary


def test_qualification_attempt_order_is_deterministic():
    attempts = qualification_attempts()
    assert [item.name for item in attempts] == [
        "clarabel_strict_original",
        "clarabel_relaxed_1e-7",
        "scs_1e-6",
        "clarabel_qdldl_strict",
        "cvxopt_high_accuracy_robust",
    ]
    assert attempts[0].options == {
        "tol_gap_abs": 1e-8,
        "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8,
    }


def test_policy_falls_back_after_exception_and_stops_after_acceptance():
    policy = SolverPolicy(qualification_attempts())
    seen = []

    def solve(attempt, retry):
        seen.append((attempt.name, retry))
        if retry == 0:
            return AttemptResult(attempt, None, {"max_constraint_violation": None},
                                 exception_type="SolverError", exception_text="failed")
        return AttemptResult(attempt, "optimal", {"max_constraint_violation": 2e-8}, payload="model")

    result, history = run_solver_policy(policy, solve)
    assert result is not None and result.payload == "model"
    assert seen == [("clarabel_strict_original", 0), ("clarabel_relaxed_1e-7", 1)]
    assert [entry["classification"] for entry in history] == ["reject", "accept"]


def test_policy_rejects_invalid_status_and_bad_feasibility():
    bad_status = assess_feasibility("unknown", {"max_constraint_violation": 0.0})
    bad_residual = assess_feasibility("optimal", {"max_constraint_violation": 2e-6})
    conditional = assess_feasibility("optimal_inaccurate", {"max_constraint_violation": 2e-8})
    assert not bad_status["accepted"]
    assert not bad_residual["accepted"]
    assert conditional["accepted"] and conditional["classification"] == "conditional"


def test_diagnostics_are_json_serializable_without_fabricated_residuals():
    result = assess_feasibility("optimal", {
        "max_constraint_violation": 0.0,
        "primal_residual": None,
        "dual_residual": None,
        "relative_gap": None,
    }, FeasibilityThresholds())
    encoded = json.dumps(result, sort_keys=True)
    assert '"accepted": true' in encoded
    assert "primal_residual" not in result


def test_policy_exhaustion_is_bounded_and_preserves_history():
    attempts = qualification_attempts()
    policy = SolverPolicy(attempts)

    def solve(attempt, retry):
        return AttemptResult(attempt, "optimal_inaccurate", {"max_constraint_violation": 1e-2})

    result, history = run_solver_policy(policy, solve)
    assert result is None
    assert len(history) == len(attempts)
    assert [entry["retry"] for entry in history] == [0, 1, 2, 3, 4]


def test_unqualified_production_policy_has_no_silent_fallback():
    policy = production_policy()
    assert [attempt.name for attempt in policy.attempts] == ["clarabel_strict_original"]


def test_call_checkpoints_survive_failure_aggregation(tmp_path):
    from archive.research_scripts.qualify_robust_v8_c_solver import aggregate

    calls = tmp_path / "calls"
    calls.mkdir()
    success = {
        "dataset": "tiny", "seed": 1, "ordinal": 0, "phase": "inner", "fold": 0,
        "alpha_rule": "paper", "C": 1.0, "alpha": 1.0,
        "attempt": "clarabel_strict_original", "solver": "CLARABEL",
        "status": "optimal", "runtime_sec": .01, "max_constraint_violation": 0.0,
        "exception_type": None, "exception_text": None,
        "feasibility": {"accepted": True},
    }
    failure = {
        **success, "ordinal": 1, "status": None,
        "exception_type": "cvxpy.error.SolverError", "exception_text": "failed",
        "feasibility": {"accepted": False},
    }
    (calls / "000.json").write_text(json.dumps(success))
    (calls / "001.json").write_text(json.dumps(failure))
    summary = aggregate(tmp_path)
    assert summary["calls"] == 2 and summary["exceptions"] == 1
    assert (calls / "000.json").exists() and (calls / "001.json").exists()
    failures = json.loads((tmp_path / "failures.json").read_text())
    assert len(failures["failures"]) == 1


def test_cvxopt_trace_parser_preserves_terminal_and_best_residuals(tmp_path):
    from archive.research_scripts.analyze_robust_solver_qualification import _cvxopt_trace_summary

    log_dir = tmp_path / "blood" / "seed-15000" / "solver_logs"
    log_dir.mkdir(parents=True)
    (log_dir / "000-cvxopt.log").write_text(
        " 0:  1.0e+01  2.0e+01  3e+01  5e+00  1e+02  1e+00\n"
        " 1:  1.5e+01  1.5e+01  2e-05  1e-09  5e-09  2e-08\n"
        " 2:  1.5e+01  1.5e+01  1e-06  4e+03  8e-06  4e-17\n"
    )
    parsed = _cvxopt_trace_summary(tmp_path)
    assert parsed.loc[0, "terminal_iteration"] == 2
    assert parsed.loc[0, "terminal_primal_residual"] == 4e3
    assert parsed.loc[0, "best_joint_iteration"] == 1


def test_rkhs_factorization_audit_never_clamps_indefinite_matrix():
    from archive.research_scripts.audit_rkhs_coordinate_factorization import factorization_diagnostics

    valid = factorization_diagnostics(np.asarray([[2.0, 0.5], [0.5, 1.0]]))
    invalid = factorization_diagnostics(np.asarray([[1.0, 2.0], [2.0, 1.0]]))
    assert valid["cholesky_success"]
    assert valid["gram_reconstruction_error_relative"] < 1e-14
    assert invalid["gram_eigenvalue_min"] < 0
    assert not invalid["cholesky_success"]
    assert invalid["gram_reconstruction_error_relative"] is None


def test_exact_rkhs_coordinate_identities_and_test_scores():
    from ddr_mksvm.robust_solvers.rkhs_coordinates import (
        coefficients_from_coordinates,
        coordinates_from_coefficients,
        factor_full_rank_gram,
        test_scores_from_coordinates,
        training_scores_from_coordinates,
    )

    K = np.asarray([[2.0, 0.5], [0.5, 1.0]])
    cross = np.asarray([[0.4, 0.2, 0.7], [0.1, 0.8, 0.3]])
    coefficients = np.asarray([0.75, -0.25])
    R = factor_full_rank_gram(K)
    z = coordinates_from_coefficients(R, coefficients)
    assert np.allclose(R.T @ R, K, rtol=0, atol=1e-14)
    assert np.dot(z, z) == pytest.approx(coefficients @ K @ coefficients)
    assert np.allclose(training_scores_from_coordinates(R, z), K @ coefficients)
    assert np.allclose(coefficients_from_coordinates(R, z), coefficients)
    assert np.allclose(test_scores_from_coordinates(cross, R, z), cross.T @ coefficients)


def test_exact_rkhs_factor_rejects_non_symmetric_or_indefinite_input():
    from ddr_mksvm.robust_solvers.rkhs_coordinates import factor_full_rank_gram

    with pytest.raises(ValueError, match="exactly symmetric"):
        factor_full_rank_gram(np.asarray([[1.0, 0.1], [0.2, 1.0]]))
    with pytest.raises(np.linalg.LinAlgError):
        factor_full_rank_gram(np.asarray([[1.0, 2.0], [2.0, 1.0]]))


def test_high_precision_gaussian_is_symmetric_and_interval_cholesky_certifies_spd():
    pytest.importorskip("flint")
    from ddr_mksvm.robust_solvers.high_precision_kernel import (
        interval_cholesky,
        reconstruct_gaussian_gram,
        reconstruction_diagnostics,
    )

    X = np.asarray([[-1.0, 0.25], [0.0, -0.5], [1.0, 0.75]])
    K = reconstruct_gaussian_gram(X, alpha=0.75, precision_bits=128)
    assert all(
        K[i, j].str(45, radius=True) == K[j, i].str(45, radius=True)
        for i in range(3) for j in range(3)
    )
    lower, diagnostics = interval_cholesky(K, precision_bits=128)
    assert diagnostics["success"] and diagnostics["minimum_pivot_lower"] > 0
    residual = reconstruction_diagnostics(K, lower, precision_bits=128)
    assert residual["residual_contains_zero_entrywise"]


def test_high_precision_cholesky_rejects_indefinite_matrix_without_repair():
    flint = pytest.importorskip("flint")
    from ddr_mksvm.robust_solvers.high_precision_kernel import interval_cholesky

    matrix = flint.arb_mat([[1, 2], [2, 1]])
    lower, diagnostics = interval_cholesky(matrix, precision_bits=128)
    assert lower is None
    assert not diagnostics["success"]


def test_high_precision_factor_spectrum_reports_full_rank_for_small_spd_case():
    pytest.importorskip("flint")
    from ddr_mksvm.robust_solvers.high_precision_kernel import (
        factor_spectrum_estimate,
        interval_cholesky,
        reconstruct_gaussian_gram,
    )

    X = np.asarray([[-1.0], [0.0], [1.0]])
    K = reconstruct_gaussian_gram(X, alpha=0.5, precision_bits=128)
    lower, _ = interval_cholesky(K, precision_bits=128)
    spectrum = factor_spectrum_estimate(lower, (53, 80))
    assert spectrum["smallest_eigenvalue_approx"] > 0
    assert spectrum["relative_threshold_ranks"] == {"53": 3, "80": 3}


def test_tiny_cvxpy_problem_can_use_original_attempt_when_available():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("CLARABEL unavailable")
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.norm(x - np.ones(2), 2) <= 1])
    original = qualification_attempts()[0]
    problem.solve(solver=original.solver, **dict(original.options))
    assert problem.status == cp.OPTIMAL
    assert max(constraint.violation() for constraint in problem.constraints) < 1e-6


def test_tiny_soc_problem_can_use_cvxopt_qualification_attempt_when_available():
    cp = pytest.importorskip("cvxpy")
    if "CVXOPT" not in cp.installed_solvers():
        pytest.skip("CVXOPT unavailable")
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.norm(x - np.ones(2), 2) <= 1])
    attempt = next(item for item in qualification_attempts()
                   if item.name == "cvxopt_high_accuracy_robust")
    problem.solve(solver=attempt.solver, **dict(attempt.options))
    assert problem.status == cp.OPTIMAL
    assert max(constraint.violation() for constraint in problem.constraints) < 1e-8
    summary = canonical_problem_summary(problem, "CVXOPT")
    assert summary["canonical_n_variables"] > 0
    assert summary["canonical_n_equalities"] == 0
    assert summary["canonical_n_inequalities"] > 0
