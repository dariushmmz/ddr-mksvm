"""Tests for the opt-in cached warm-started adaptive V3 variant."""

import numpy as np

from ddr_mksvm.optim.alternating_trainer import _numpy_gram
from ddr_mksvm.optim.cached_warm_started_adaptive_trainer import (
    CachedWarmStartedAdaptiveBandwidthTrainer,
    clear_raw_gram_cache_records,
    get_raw_gram_cache_records,
)
from ddr_mksvm.optim.warm_started_adaptive_trainer import (
    WarmStartedAdaptiveBandwidthTrainer,
)


def _kwargs():
    return dict(
        base_kernel_specs=[dict(kind="rbf", alpha=1.0)],
        in_dim=2,
        dnn_on=True,
        mkl_on=False,
        dro_on=True,
        epsilon=0.001,
        n_outer=1,
        n_inner=1,
        seed=0,
        raw_anchor_spec=dict(kind="poly", degree=2, c=1.0, _lip_bound=2.0),
    )


def test_opt_in_cached_gram_is_reused_without_changing_values():
    X = np.array([[0.0, 1.0, 2.0], [1.0, 1.5, 3.0]])
    trainer = CachedWarmStartedAdaptiveBandwidthTrainer(**_kwargs())
    clear_raw_gram_cache_records()
    trainer._prepare_raw_gram_cache(X)
    cached = trainer.raw_anchor_spec["_cached_train_gram"]
    expected = (X.T @ X + 1.0) ** 2
    np.testing.assert_array_equal(cached, expected)
    assert _numpy_gram(trainer.raw_anchor_spec, X) is cached
    trainer._prepare_raw_gram_cache(10.0 * X)
    assert trainer.raw_anchor_spec["_cached_train_gram"] is cached
    records = get_raw_gram_cache_records()
    assert len(records) == 1
    assert records[0]["cache_bytes"] == expected.nbytes


def test_frozen_warm_started_variant_has_no_cache_hook_or_cached_spec():
    frozen = WarmStartedAdaptiveBandwidthTrainer(**_kwargs())
    cached = CachedWarmStartedAdaptiveBandwidthTrainer(**_kwargs())
    assert type(frozen) is WarmStartedAdaptiveBandwidthTrainer
    assert type(cached) is CachedWarmStartedAdaptiveBandwidthTrainer
    assert "_cached_train_gram" not in frozen.raw_anchor_spec
    assert "_cached_train_gram" not in cached.raw_anchor_spec


def test_uncached_numpy_gram_behavior_is_unchanged():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    spec = dict(kind="poly", degree=3, c=0.5)
    np.testing.assert_array_equal(_numpy_gram(spec, X), (X.T @ X + 0.5) ** 3)
