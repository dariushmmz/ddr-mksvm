"""Focused tests for the opt-in adaptive-bandwidth Residual V3 path."""

import numpy as np
import pytest

from ddr_mksvm.optim.adaptive_bandwidth_trainer import (
    AdaptiveBandwidthResidualTrainer,
    clear_bandwidth_records,
    get_bandwidth_records,
    median_distance_bandwidth,
)


def _trainer(seed=0):
    return AdaptiveBandwidthResidualTrainer(
        base_kernel_specs=[dict(kind="rbf", alpha=1.0)],
        in_dim=2,
        dnn_on=True,
        mkl_on=False,
        dro_on=True,
        epsilon=0.001,
        n_outer=1,
        n_inner=1,
        seed=seed,
        raw_anchor_spec=dict(kind="linear", _lip_bound=1.0),
    )


def test_median_distance_bandwidth_sets_median_similarity_to_exp_minus_half():
    Z = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    alpha = median_distance_bandwidth(Z)
    distances = np.array([5.0, 10.0, 5.0])
    similarities = np.exp(-(distances ** 2) / (2.0 * alpha ** 2))
    assert alpha == pytest.approx(5.0)
    assert np.median(similarities) == pytest.approx(np.exp(-0.5))


def test_bandwidth_is_scale_equivariant_and_rejects_collapsed_features():
    Z = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 6.0]])
    assert median_distance_bandwidth(7.0 * Z) == pytest.approx(
        7.0 * median_distance_bandwidth(Z)
    )
    with pytest.raises(ValueError, match="collapsed"):
        median_distance_bandwidth(np.ones((3, 2)))


def test_trainer_calibrates_once_from_training_features_and_freezes_alpha():
    X = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 4.0]])
    clear_bandwidth_records()
    trainer = _trainer()
    trainer._calibrate_bandwidth(X)
    first = trainer.adaptive_bandwidth
    trainer._calibrate_bandwidth(100.0 * X)
    assert first > 0.0
    assert trainer.base_kernel_specs[0]["alpha"] == first
    records = get_bandwidth_records()
    assert len(records) == 1
    assert records[0]["adaptive_bandwidth"] == first
    assert records[0]["n_training_samples"] == 3


def test_adaptive_trainer_remains_distinct_from_frozen_residual_v3():
    from ddr_mksvm.optim.alternating_trainer import AlternatingTrainer

    adaptive = _trainer()
    frozen = AlternatingTrainer(
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
    assert type(adaptive) is AdaptiveBandwidthResidualTrainer
    assert type(frozen) is AlternatingTrainer
    assert frozen.base_kernel_specs[0]["alpha"] == 1.0
