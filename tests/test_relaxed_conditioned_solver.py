"""Tests for the separate Residual V3 variant with a relaxed 1e10 cap."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")
pytest.importorskip("torch")

from ddr_mksvm.optim.conditioned_convex_subproblem import (
    DEFAULT_CONDITION_NUMBER_CAP,
    condition_gram_matrix,
    train_with_nu_search_conditioned,
)
from ddr_mksvm.optim.conditioned_trainer import ConditionedResidualTrainer
from ddr_mksvm.optim.relaxed_conditioned_trainer import (
    RELAXED_CONDITION_NUMBER_CAP,
    RelaxedConditionedResidualTrainer,
)
from archive.research_scripts.run_relaxed_conditioned_v3_experiment import VARIANTS


def _trainer_kwargs():
    return dict(
        base_kernel_specs=[dict(kind="rbf", alpha=1.0)],
        in_dim=2,
        n_outer=0,
        n_inner=0,
        seed=19,
        raw_anchor_spec=dict(kind="linear", _lip_bound=1.0),
    )


def test_relaxed_variant_uses_distinct_1e10_cap():
    original = ConditionedResidualTrainer(**_trainer_kwargs())
    relaxed = RelaxedConditionedResidualTrainer(**_trainer_kwargs())

    assert DEFAULT_CONDITION_NUMBER_CAP == 1e8
    assert original.condition_number_cap == 1e8
    assert RELAXED_CONDITION_NUMBER_CAP == 1e10
    assert relaxed.condition_number_cap == 1e10
    assert relaxed._convex_train_fn.func is train_with_nu_search_conditioned
    assert relaxed._convex_train_fn.keywords["condition_number_cap"] == 1e10


def test_relaxed_cap_keeps_formerly_null_direction_at_solver_threshold():
    # The frozen solver retains only eigenvalues strictly greater than
    # max(1, lambda_max) * 1e-10.  A zero mode conditioned to a 1e10 ratio
    # must remain at, rather than be lifted above, that cutoff.
    K = np.diag([10.0, 0.0])

    conditioned, diagnostics = condition_gram_matrix(
        K,
        condition_number_cap=RELAXED_CONDITION_NUMBER_CAP,
    )
    eigenvalues = np.linalg.eigvalsh(conditioned)
    solver_threshold = max(1.0, float(eigenvalues[-1])) * 1e-10
    retained_by_solver = eigenvalues > solver_threshold

    assert diagnostics["original_condition_number"] == np.inf
    assert diagnostics["conditioned_condition_number"] == pytest.approx(1e10)
    assert eigenvalues[0] <= solver_threshold * (1.0 + 1e-12)
    assert not retained_by_solver[0]


def test_relaxed_experiment_variant_is_separately_named():
    assert VARIANTS["conditioned_v3"]["model_version"] == "v3_conditioned"
    assert (
        VARIANTS["relaxed_conditioned_v3"]["model_version"]
        == "v3_relaxed_conditioned"
    )
