import inspect
from pathlib import Path

import numpy as np

from ddr_mksvm.robust_matlab_parity import (
    KernelSpec,
    build_q1_lp,
    fit_binary as fit_source_binary,
    matlab_binary_threshold,
    predict_ova,
    uncertainty_delta,
)
from ddr_mksvm.robust_source_aligned import (
    build_weighted_q1_lp,
    fit_binary,
    fit_ova,
    sign_group_weights,
)


def fixture():
    X = np.array([[-2.0], [-1.0], [0.8], [2.0]])
    labels = np.array([0, 0, 1, 1])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    kernel = KernelSpec("poly", degree=1, offset=0.0)
    K = X @ X.T
    return X, labels, y, kernel, K


def test_sign_group_weights_mean_one_and_tau_zero_exact():
    y = np.array([-1.0] * 9 + [1.0] * 4)
    weights = sign_group_weights(y, tau=0.5)
    assert np.all(weights > 0)
    assert np.isclose(np.sum(weights), len(y), rtol=0, atol=2e-15)
    assert weights[y == 1][0] > weights[y == -1][0]
    assert np.array_equal(sign_group_weights(y, tau=0), np.ones(len(y)))


def test_weight_disabled_is_exact_original_lp():
    X, labels, y, kernel, K = fixture()
    delta = uncertainty_delta(X, labels, 0.01, 2, kernel)
    original = build_q1_lp(K, y, 0.1, delta)
    weighted = build_weighted_q1_lp(
        K, y, 0.1, delta, tau=0.5, class_sensitive=False
    )
    for left, right in zip(original[:3], weighted[:3]):
        assert np.array_equal(left, right)
    assert original[3] == weighted[3]
    assert np.array_equal(weighted[4], np.ones(len(y)))


def test_only_slack_objective_coefficients_change():
    X, labels, y, kernel, K = fixture()
    delta = uncertainty_delta(X, labels, 0.01, 2, kernel)
    c0, A0, b0, bounds0 = build_q1_lp(K, y, 0.1, delta)
    c1, A1, b1, bounds1, weights = build_weighted_q1_lp(K, y, 0.1, delta)
    m = len(y)
    assert np.array_equal(A0, A1)
    assert np.array_equal(b0, b1)
    assert bounds0 == bounds1
    assert np.array_equal(c0[: m + 1], c1[: m + 1])
    assert np.array_equal(c0[2 * m + 1 :], c1[2 * m + 1 :])
    assert np.allclose(c1[m + 1 : 2 * m + 1], 0.1 * weights)


def test_rho_zero_is_weighted_deterministic_same_profile():
    X, labels, y, kernel, K = fixture()
    delta = uncertainty_delta(X, labels, 0.0, 2, kernel)
    assert np.array_equal(delta, np.zeros(len(y)))
    _, A, _, _, _ = build_weighted_q1_lp(K, y, 0.1, delta)
    m = len(y)
    assert np.count_nonzero(A[:m, 2 * m + 1 :]) == 0


def test_disabled_fit_dispatch_matches_source_predictions_and_selection():
    X, labels, y, kernel, _ = fixture()
    source, source_rows = fit_source_binary(X, y, labels, kernel, 0.01, 2)
    reduced, reduced_rows = fit_binary(
        X, y, labels, kernel, 0.01, 2, class_sensitive=False
    )
    query = np.array([[-1.5], [0.0], [1.5]])
    assert np.array_equal(source.decision(query), reduced.decision(query))
    assert source.b == reduced.b
    assert source.diagnostics["nu"] == reduced.diagnostics["nu"]
    assert source.diagnostics["training_error"] == reduced.diagnostics["training_error"]
    for source_row, reduced_row in zip(source_rows, reduced_rows):
        assert source_row.keys() == reduced_row.keys()
        for key in source_row:
            if key != "runtime_s":
                assert source_row[key] == reduced_row[key]


def test_source_threshold_order_and_first_minimum_is_reused():
    source = inspect.getsource(fit_binary)
    assert "matlab_binary_threshold" in source
    assert "min(candidates, key=lambda row: (row[0], row[1]))" in source
    assert "np.linspace" not in source
    assert matlab_binary_threshold.__module__ == "ddr_mksvm.robust_matlab_parity"


def test_ova_construction_and_fail_closed_tie_policy_unchanged():
    X = np.array([[-3.0], [-2.0], [0.0], [0.5], [2.0], [3.0]])
    labels = np.array([1, 1, 2, 2, 3, 3])
    kernel = KernelSpec("poly", degree=1, offset=0.0)
    models, rows = fit_ova(X, labels, kernel, 0.0, 2, tau=0.5)
    assert [model.positive_class for model in models] == [1, 2, 3]
    for model, label in zip(models, [1, 2, 3]):
        assert np.array_equal(model.y, np.where(labels == label, 1.0, -1.0))
    assert sorted(set(row["task"] for row in rows)) == [0, 1, 2]
    # The source predictor is retained; exact score ties fail closed.
    for model in models:
        model.u[:] = 0
        model.b = 0
    try:
        predict_ova(models, np.array([[0.0]]), tie_policy="matlab_error")
    except RuntimeError as exc:
        assert "undefined" in str(exc)
    else:
        raise AssertionError("source OVA ties must fail closed")


def test_candidate_contains_no_soc_or_factorization_machinery():
    import ddr_mksvm.robust_source_aligned as module

    source = inspect.getsource(module).lower()
    forbidden = ("import cvxpy", "cp.soc", "cholesky(", "eigh(", "svd(", "sqrtm(")
    assert all(token not in source for token in forbidden)


def test_frozen_v7_v8_hashes_unchanged():
    import hashlib

    expected = {
        "ddr_mksvm/v7_cross_dataset.py": "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d",
        "ddr_mksvm/v8_class_sensitive.py": "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
