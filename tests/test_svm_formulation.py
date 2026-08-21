"""Small controlled tests for the exact DDR Corollary-1 convex block."""
import numpy as np
import pytest

pytest.importorskip("cvxpy")
from ddr_mksvm.optim.convex_subproblem import solve_svm_dro


def test_linearly_separable_data_learns_two_classes_without_slack():
    X = np.array([[-2., -1., 1., 2.], [-1., -2., 2., 1.]])
    y = np.array([-1., -1., 1., 1.])
    sol = solve_svm_dro(X.T @ X, y, nu=1e-2)
    d = sol["diagnostics"]
    assert d["training_error"] == 0
    assert d["w_norm_H"] > 1e-6
    assert d["sum_xi"] < 1e-5
    assert set(d["predicted_class_counts"]) == {-1, 1}


def test_nonseparable_duplicate_with_opposite_labels_uses_slack():
    X = np.array([[-1., 0., 0., 1.], [0., 0., 0., 0.]])
    y = np.array([-1., -1., 1., 1.])
    sol = solve_svm_dro(X.T @ X, y, nu=1e-2)
    assert sol["diagnostics"]["sum_xi"] > 0.5


def test_label_encoding_is_rejected_instead_of_silently_collapsing():
    X = np.array([[-1., 1.], [0., 0.]])
    with pytest.raises(ValueError, match="labels"):
        solve_svm_dro(X.T @ X, np.array([0., 1.]), nu=1e-2)


def test_nu_has_specified_regularization_direction():
    X = np.array([[-2., -1., .2, 1., 2.], [-1., -2., .1, 2., 1.]])
    y = np.array([-1., -1., 1., 1., 1.])
    low = solve_svm_dro(X.T @ X, y, nu=1e-4)["diagnostics"]
    high = solve_svm_dro(X.T @ X, y, nu=1e3)["diagnostics"]
    assert high["w_norm_H"] < low["w_norm_H"]
    assert high["sum_xi"] > low["sum_xi"]
