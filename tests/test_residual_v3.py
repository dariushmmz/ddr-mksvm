"""Cheap structural checks for the versioned Residual V3 kernel path."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ddr_mksvm.optim.alternating_trainer import AlternatingTrainer
from archive.research_scripts.run_v3_experiment import _planned_solver_calls


def _trainer(seed=7):
    return AlternatingTrainer(
        [dict(kind="rbf", alpha=1.0)],
        in_dim=3,
        n_outer=0,
        n_inner=0,
        seed=seed,
        raw_anchor_spec=dict(kind="linear", _lip_bound=1.0),
    )


def test_residual_v3_train_and_cross_kernels_match():
    X = np.array([[0.0, 1.0, 2.0], [1.0, -1.0, 0.5], [2.0, 0.0, -2.0]])
    trainer = _trainer()
    gram = trainer._combined_gram_numpy(X)
    cross = trainer.combined_cross_numpy(X, X)

    assert gram.shape == (3, 3)
    np.testing.assert_allclose(gram, gram.T, atol=1e-12)
    np.testing.assert_allclose(cross, gram, rtol=1e-6, atol=1e-7)
    assert np.linalg.eigvalsh(gram).min() >= -1e-8


def test_residual_v3_initialization_is_seed_deterministic():
    X = np.arange(15, dtype=float).reshape(3, 5) / 10.0
    first = _trainer(seed=11)
    second = _trainer(seed=11)

    np.testing.assert_allclose(
        first._combined_gram_numpy(X), second._combined_gram_numpy(X), atol=0.0
    )
    assert first.residual_gate() == pytest.approx(0.10, abs=1e-7)


def test_controlled_pilot_solver_call_accounting():
    assert _planned_solver_calls("parkinson", "legacy", n_outer=1) == 5
    assert _planned_solver_calls("parkinson", "residual_v3", n_outer=1) == 14
    assert _planned_solver_calls("iris", "legacy", n_outer=1) == 15
    assert _planned_solver_calls("iris", "current_v2", n_outer=1) == 42
