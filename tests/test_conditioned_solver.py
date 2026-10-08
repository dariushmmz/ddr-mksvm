"""Focused tests for the opt-in conditioned Residual V3 solver path."""

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
torch = pytest.importorskip("torch")

from ddr_mksvm.optim.alternating_trainer import AlternatingTrainer
from ddr_mksvm.optim.conditioned_convex_subproblem import (
    clear_conditioning_records,
    condition_gram_matrix,
    get_conditioning_records,
    solve_svm_dro_conditioned,
    train_with_nu_search_conditioned,
)
from ddr_mksvm.optim.conditioned_trainer import ConditionedResidualTrainer
from ddr_mksvm.optim.convex_subproblem import train_with_nu_search
from archive.research_scripts.run_v3_experiment import _planned_solver_calls


def test_rank_deficient_gram_receives_minimal_conditioning_shift():
    X = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    K = X.T @ X
    cap = 1e6

    conditioned, diagnostics = condition_gram_matrix(K, cap)
    eigenvalues = np.linalg.eigvalsh(conditioned)

    assert diagnostics["diagonal_shift"] > 0.0
    assert diagnostics["original_condition_number"] == np.inf
    assert diagnostics["conditioned_condition_number"] == pytest.approx(cap, rel=1e-8)
    assert eigenvalues[-1] / eigenvalues[0] <= cap * (1.0 + 1e-8)
    np.testing.assert_allclose(
        conditioned - K,
        diagnostics["diagonal_shift"] * np.eye(K.shape[0]),
        rtol=1e-8,
        atol=1e-12,
    )


def test_well_conditioned_gram_is_not_changed():
    K = np.diag([1.0, 2.0, 3.0])

    conditioned, diagnostics = condition_gram_matrix(K, condition_number_cap=10.0)

    np.testing.assert_array_equal(conditioned, K)
    assert diagnostics["diagonal_shift"] == 0.0
    assert diagnostics["original_condition_number"] == pytest.approx(3.0)
    assert diagnostics["conditioned_condition_number"] == pytest.approx(3.0)


def test_materially_indefinite_gram_is_rejected():
    with pytest.raises(ValueError, match="not PSD"):
        condition_gram_matrix(np.diag([1.0, -0.1]))


def test_conditioned_socp_returns_solver_and_conditioning_diagnostics():
    X = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    K = X.T @ X + 1.0
    y = np.array([-1.0, -1.0, 1.0, 1.0])

    solution = solve_svm_dro_conditioned(
        K,
        y,
        nu=0.1,
        epsilon=0.01,
        L_theta_eta=1.0,
        condition_number_cap=1e6,
    )

    assert solution is not None
    assert solution["status"] in {"optimal", "optimal_inaccurate"}
    assert solution["solver_name"]
    assert solution["solve_time"] is None or solution["solve_time"] >= 0.0
    assert solution["gram_conditioning"]["diagonal_shift"] > 0.0
    assert solution["gram_conditioning"]["conditioned_condition_number"] <= 1e6 * (1.0 + 1e-8)


def test_conditioned_nu_search_records_diagnostics_for_every_candidate():
    K = np.ones((4, 4))
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    clear_conditioning_records()

    result = train_with_nu_search_conditioned(
        K,
        y,
        nu_grid=[0.01, 0.1],
        epsilon=0.01,
        L_theta_eta=1.0,
        verbose=False,
        condition_number_cap=1e6,
    )

    assert result is not None
    assert len(result["_nu_candidates"]) == 2
    assert "gram_conditioning" in result
    assert all("gram_conditioning" in candidate["solution"] for candidate in result["_nu_candidates"])
    records = get_conditioning_records()
    assert len(records) == 1
    assert records[0]["candidate_count"] == 2
    assert records[0]["optimal_count"] + records[0]["optimal_inaccurate_count"] == 2
    assert records[0]["selected_solver"]


def test_conditioned_trainer_is_opt_in_and_preserves_v3_kernel_initialization():
    kwargs = dict(
        base_kernel_specs=[dict(kind="rbf", alpha=1.0)],
        in_dim=2,
        n_outer=0,
        n_inner=0,
        seed=17,
        raw_anchor_spec=dict(kind="linear", _lip_bound=1.0),
    )
    frozen_v3 = AlternatingTrainer(**kwargs)
    conditioned_v3 = ConditionedResidualTrainer(**kwargs)
    X = np.arange(10, dtype=float).reshape(2, 5) / 10.0

    assert frozen_v3._convex_train_fn is train_with_nu_search
    assert conditioned_v3._convex_train_fn.func is train_with_nu_search_conditioned
    np.testing.assert_allclose(
        conditioned_v3._combined_gram_numpy(X),
        frozen_v3._combined_gram_numpy(X),
        rtol=0.0,
        atol=0.0,
    )


def test_conditioned_variant_preserves_v3_solver_call_count():
    assert _planned_solver_calls("parkinson", "conditioned_v3", n_outer=1) == 14
    assert _planned_solver_calls("iris", "conditioned_v3", n_outer=1) == 42
