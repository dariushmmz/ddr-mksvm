"""Focused tests for the opt-in warm-started adaptive-bandwidth V3 path."""

import numpy as np
import pytest

from ddr_mksvm.optim.convex_subproblem import train_with_nu_search
from ddr_mksvm.optim.warm_started_adaptive_trainer import (
    WarmStartedAdaptiveBandwidthTrainer,
)
from ddr_mksvm.optim.warm_started_convex_subproblem import (
    clear_warm_start_records,
    get_warm_start_records,
    train_with_nu_search_warm_started,
)


def test_warm_started_grid_matches_frozen_solver_decision_and_selected_nu():
    X = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.5, 2.0]])
    K = X.T @ X + np.eye(X.shape[1])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    grid = np.array([1e-3, 1e-2, 1e-1])
    frozen = train_with_nu_search(K, y, grid, epsilon=0.001, L_theta_eta=1.0, verbose=False)
    warm = train_with_nu_search_warm_started(
        K, y, grid, epsilon=0.001, L_theta_eta=1.0, verbose=False
    )
    assert warm["selected_nu"] == pytest.approx(frozen["selected_nu"])
    assert warm["training_error"] == frozen["training_error"]
    np.testing.assert_allclose(
        K @ (y * warm["u"]) - warm["b"],
        K @ (y * frozen["u"]) - frozen["b"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_warm_started_grid_records_one_cold_start_then_reuse():
    K = np.array([[2.0, 0.5], [0.5, 2.0]])
    y = np.array([-1.0, 1.0])
    clear_warm_start_records()
    result = train_with_nu_search_warm_started(
        K, y, [1e-3, 1e-2, 1e-1], verbose=False
    )
    assert result is not None
    records = get_warm_start_records()
    assert len(records) == 1
    assert records[0]["grid_size"] == 3
    assert records[0]["completed_candidates"] == 3
    assert records[0]["cold_starts"] == 1
    assert records[0]["warm_starts"] == 2


def test_warm_started_trainer_is_distinct_and_uses_opt_in_solver():
    trainer = WarmStartedAdaptiveBandwidthTrainer(
        base_kernel_specs=[dict(kind="rbf", alpha=1.0)],
        in_dim=2,
        dnn_on=True,
        mkl_on=False,
        dro_on=True,
        epsilon=0.001,
        n_outer=1,
        n_inner=1,
        seed=0,
        raw_anchor_spec=dict(kind="linear", _lip_bound=1.0),
    )
    assert type(trainer) is WarmStartedAdaptiveBandwidthTrainer
    assert trainer._convex_train_fn is train_with_nu_search_warm_started
