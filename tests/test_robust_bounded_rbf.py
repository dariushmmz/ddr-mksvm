import hashlib
from pathlib import Path

import numpy as np

from ddr_mksvm.robust_bounded_rbf import (
    R9_P,
    R9_RHO,
    R9_TAU,
    build_r9_q1_lp,
    fit_ova,
    rbf_spec,
    rbf_uncertainty_delta,
    source_rbf_alpha,
)
from ddr_mksvm.robust_matlab_parity import (
    KernelSpec,
    build_q1_lp,
    gram,
    matlab_binary_threshold,
    matlab_multiclass_threshold,
)
from ddr_mksvm.robust_source_aligned import sign_group_weights

ROOT = Path(__file__).resolve().parents[1]


def test_rbf_kernel_is_symmetric_bounded_and_unit_diagonal():
    X = np.array([[0.0, 1.0], [2.0, -1.0], [0.5, 0.25]])
    kernel = rbf_spec(X)
    K = gram(X, None, kernel)
    np.testing.assert_allclose(K, K.T, rtol=0, atol=0)
    np.testing.assert_allclose(np.diag(K), 1.0, rtol=0, atol=0)
    assert np.all((K > 0) & (K <= 1))


def test_source_bandwidth_and_rbf_uncertainty_formula():
    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])
    labels = np.array([0, 0, 1, 1])
    expected_alpha = np.max(np.std(X, axis=0, ddof=1))
    assert source_rbf_alpha(X) == expected_alpha
    delta = rbf_uncertainty_delta(X, labels, R9_RHO, R9_P, alpha=expected_alpha)
    for label in (0, 1):
        eta = R9_RHO * np.max(np.std(X[labels == label], axis=0, ddof=1))
        expected = np.sqrt(2 - 2 * np.exp(-(eta**2) / (2 * expected_alpha**2)))
        np.testing.assert_allclose(delta[labels == label], expected, rtol=1e-13, atol=1e-15)


def test_rho_zero_removes_robust_coefficients_exactly():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    K = gram(X, None, rbf_spec(X))
    delta = rbf_uncertainty_delta(X, labels, 0.0, R9_P)
    assert np.array_equal(delta, np.zeros(len(X)))
    weighted = build_r9_q1_lp(K, y, 0.1, delta, tau=R9_TAU, class_sensitive=True)
    assert np.count_nonzero(weighted[1][: len(y), 2 * len(y) + 1 :]) == 0


def test_disabled_weighting_is_exact_source_q1_reduction():
    K = np.array([[1.0, 0.2], [0.2, 1.0]])
    y = np.array([1.0, -1.0])
    delta = np.array([0.02, 0.03])
    r9 = build_r9_q1_lp(K, y, 0.1, delta, class_sensitive=False)
    source = build_q1_lp(K, y, 0.1, delta)
    for got, expected in zip(r9[:3], source[:3]):
        np.testing.assert_array_equal(got, expected)
    assert r9[3] == source[3]
    np.testing.assert_array_equal(r9[4], np.ones(2))


def test_r9_weights_are_positive_and_mean_one():
    y = np.array([1.0] * 3 + [-1.0] * 11)
    weights = sign_group_weights(y, R9_TAU)
    assert np.all(weights > 0)
    np.testing.assert_allclose(np.mean(weights), 1.0, rtol=0, atol=3e-16)


def test_robust_matrix_matches_frozen_rbf_equation():
    K = np.array([[1.0, 0.4], [0.4, 1.0]])
    y = np.array([1.0, -1.0])
    delta = np.array([0.1, 0.2])
    c, A, b, bounds, weights = build_r9_q1_lp(
        K, y, 0.5, delta, class_sensitive=True
    )
    m = len(y)
    np.testing.assert_array_equal(A[:m, :m], -(np.outer(y, y) * K))
    np.testing.assert_array_equal(A[:m, m], y)
    np.testing.assert_array_equal(A[:m, m + 1 : 2 * m + 1], -np.eye(m))
    np.testing.assert_array_equal(A[:m, 2 * m + 1 :], np.outer(delta, np.ones(m)))
    np.testing.assert_array_equal(c[m + 1 : 2 * m + 1], 0.5 * weights)
    np.testing.assert_array_equal(b[:m], -np.ones(m))
    assert bounds[: m + 1] == [(None, None)] * (m + 1)


def test_source_threshold_functions_are_used_directly():
    import ddr_mksvm.robust_bounded_rbf as module

    assert module.matlab_binary_threshold is matlab_binary_threshold
    assert module.matlab_multiclass_threshold is matlab_multiclass_threshold


def test_ova_uses_sorted_source_class_construction(monkeypatch):
    import ddr_mksvm.robust_bounded_rbf as module

    seen = []

    class Stub:
        positive_class = None

    def fake_fit(X, y, original_labels, rho, p, **kwargs):
        seen.append((y.copy(), original_labels.copy(), kwargs["multiclass_bug"]))
        return Stub(), [{"selected": True}]

    monkeypatch.setattr(module, "fit_binary", fake_fit)
    labels = np.array([3, 1, 2, 1])
    models, rows = fit_ova(
        np.arange(8.0).reshape(4, 2), labels, R9_RHO, class_sensitive=True
    )
    assert [model.positive_class for model in models] == [1, 2, 3]
    for label, (binary, original, bug) in zip((1, 2, 3), seen):
        np.testing.assert_array_equal(binary, np.where(labels == label, 1.0, -1.0))
        np.testing.assert_array_equal(original, labels)
        assert bug is True


def test_no_soc_factorization_equilibration_or_repair_path():
    source = (ROOT / "ddr_mksvm" / "robust_bounded_rbf.py").read_text(encoding="utf-8").lower()
    for forbidden in (
        "import cvxpy",
        "np.linalg.cholesky(",
        "np.linalg.eigh(",
        "kernel_jitter",
        "equilibrate_lp",
        "secondordercone",
    ):
        assert forbidden not in source


def test_frozen_v7_v8_hashes_unchanged():
    expected = {
        "v7_cross_dataset.py": "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d",
        "v8_class_sensitive.py": "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448",
    }
    for name, digest in expected.items():
        data = (ROOT / "ddr_mksvm" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
