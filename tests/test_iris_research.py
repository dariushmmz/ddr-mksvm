import copy

import numpy as np
import pandas as pd
import pytest

from ddr_mksvm.iris_research import (
    PAPER_NUMERIC_SHA256,
    PAPER_NU_GRID,
    SOURCE_CONTINUED_NU_GRID,
    MODEL_SPECS,
    WIDE_NU_GRID,
    alpha_from_rule,
    build_q1_lp,
    centered_target_nnls_weights,
    exact_threshold,
    fit_multiclass,
    fixed_grid_threshold,
    index_hash,
    locked_split_indices,
    multiscale_rbf_gram,
    optimize_ova_biases,
    polynomial_gram,
    numeric_sha256,
    predict_multiclass,
    rbf_gram,
    reconstruct_authors_iris,
    robust_delta,
    select_rho,
    select_feature_subset,
    solve_binary_q1,
    standardize_train_test,
)


def test_authors_iris_reconstruction_and_locked_split():
    local = pd.read_csv("dataset/iris_multiclass.csv").to_numpy(float)
    paper = reconstruct_authors_iris(local)
    assert numeric_sha256(paper) == PAPER_NUMERIC_SHA256
    assert paper[34].tolist() == [4.9, 3.1, 1.5, 0.2, 1.0]
    assert paper[37].tolist() == [4.9, 3.6, 1.4, 0.1, 1.0]
    first = locked_split_indices(paper[:, -1], 7)
    second = locked_split_indices(paper[:, -1], 7)
    assert index_hash(first[0]) == index_hash(second[0])
    assert len(first[0]) == 112 and len(first[1]) == 38
    assert np.all(np.diff(first[0]) > 0) and np.all(np.diff(first[1]) > 0)
    assert not set(first[0]) & set(first[1])


def test_rbf_parameterization_and_psd():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    K = rbf_gram(X, alpha=5.0)
    assert K[0, 1] == pytest.approx(np.exp(-0.5))
    np.testing.assert_allclose(K, K.T)
    np.testing.assert_allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() >= -1e-12
    assert alpha_from_rule(X, "med:0") == pytest.approx(5.0)


def test_standardization_is_fit_on_training_rows_only():
    train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    test = np.array([[100.0, 1000.0]])
    transformed_train, transformed_test, diag = standardize_train_test(train, test)
    np.testing.assert_allclose(transformed_train.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(transformed_train.std(axis=0, ddof=1), 1.0)
    assert diag["preprocess_mean"] == [3.0, 20.0]
    assert transformed_test[0, 0] == pytest.approx((100.0 - 3.0) / 2.0)


def test_feature_subset_selection_is_training_only_and_returns_valid_coordinates():
    X = np.array([
        [-2.0, 5.0], [-1.8, -5.0], [-1.6, 5.0], [-1.4, -5.0],
        [-0.2, 5.0], [0.0, -5.0], [0.2, 5.0], [0.4, -5.0],
        [1.4, 5.0], [1.6, -5.0], [1.8, 5.0], [2.0, -5.0],
    ])
    labels = np.repeat([1, 2, 3], 4)
    subset, scores, diagnostics = select_feature_subset(
        X, labels, seed=5, subsets=((0,), (1,))
    )
    assert subset in {(0,), (1,)}
    assert set(scores) == {"1", "2"}
    assert len(diagnostics) == 90
    assert all(row["phase"] == "inner_feature_subset" for row in diagnostics)


def test_literal_matlab_polynomial_kernel_and_q1_solve():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0]])
    K = polynomial_gram(X, degree=2, offset=0.0)
    assert K[0, 1] == pytest.approx((1.0 * 3.0 + 2.0 * 4.0) ** 2)
    np.testing.assert_allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() >= -1e-10
    y = np.array([-1.0, 1.0, -1.0])
    model, diag = solve_binary_q1(
        X, y, alpha=1.0, nu=1.0, kernel_kind="poly",
        polynomial_degree=2, polynomial_offset=0.0,
    )
    assert model.kernel_kind == "poly"
    assert diag["kernel_kind"] == "poly"
    assert diag["solver_status_code"] == 0


def test_training_only_multiscale_weights_form_psd_kernel():
    X = np.array([[-2.0, 0.0], [-1.0, 0.2], [0.0, 1.0], [0.2, 1.2], [2.0, 0.0], [2.2, 0.2]])
    labels = np.array([1, 1, 2, 2, 3, 3])
    scales = (0.5, 1.0, 2.0)
    weights, diag = centered_target_nnls_weights(X, labels, alpha=1.0, scales=scales)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert diag["kernel_weight_method"] == "centered-target-nnls"
    K = multiscale_rbf_gram(X, 1.0, scales, weights)
    np.testing.assert_allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() >= -1e-12


def test_exact_threshold_never_worse_than_finite_grid():
    rng = np.random.default_rng(4)
    for _ in range(20):
        scores = rng.normal(size=30)
        y = np.r_[np.ones(15), -np.ones(15)]
        xi = rng.uniform(0.0, 2.0, size=30)
        gamma = float(rng.normal())
        exact_b, exact_error, exact_diag = exact_threshold(scores, y, gamma, xi)
        grid_b, _, _ = fixed_grid_threshold(scores, y, gamma, xi, points=101)
        grid_error = np.mean(np.where(scores - grid_b > 0.0, 1.0, -1.0) != y)
        assert exact_error == np.mean(np.where(scores - exact_b > 0.0, 1.0, -1.0) != y)
        assert exact_error <= grid_error + 1e-12
        assert exact_diag["threshold_candidates"] <= 2 * len(y) + 1


def test_legacy_grid_preserves_descending_matlab_tie_order():
    b, error, diag = fixed_grid_threshold(
        scores=np.array([0.0, 0.0]), y=np.array([-1.0, -1.0]),
        gamma=0.0, xi=np.zeros(2), points=3,
    )
    assert diag["strip_reversed"]
    assert b == pytest.approx(1.0)
    assert error == 0.0


def test_matlab_threshold_training_count_treats_equality_as_not_misclassified():
    b, matlab_error, _ = fixed_grid_threshold(
        scores=np.array([0.0, 0.0]), y=np.array([-1.0, 1.0]),
        gamma=0.0, xi=np.ones(2), points=3,
    )
    assert b == pytest.approx(0.0)
    assert matlab_error == 0.0
    # A deployed binary classifier must assign equality to one side, so this
    # deliberate source behavior cannot be interpreted as prediction error.
    assert np.mean(np.where(np.array([0.0, 0.0]) - b > 0, 1.0, -1.0)
                   != np.array([-1.0, 1.0])) == 0.5


def test_q1_lp_matrix_is_literal_matlab_inequality():
    K = np.array([[1.0, 0.25], [0.25, 1.0]])
    y = np.array([1.0, -1.0])
    objective, A_ub, b_ub, bounds = build_q1_lp(K, y, nu=0.3)
    u = np.array([0.4, -0.2])
    gamma = 0.1
    xi = np.array([0.8, 1.2])
    s = np.abs(u)
    z = np.r_[u, gamma, xi, s]
    D = np.diag(y)
    matlab_margin = D @ (K @ D @ u - np.ones(2) * gamma) + xi
    np.testing.assert_allclose(A_ub[:2] @ z, -matlab_margin)
    np.testing.assert_allclose(b_ub[:2], -np.ones(2))
    assert objective @ z == pytest.approx(np.abs(u).sum() + 0.3 * xi.sum())
    assert bounds[2 + 1] == (0.0, None)


def test_exact_threshold_respects_negative_tie_convention():
    scores = np.array([0.0, 1.0])
    y = np.array([-1.0, 1.0])
    b, error, _ = exact_threshold(scores, y, gamma=0.5, xi=np.zeros(2))
    prediction = np.where(scores - b > 0.0, 1.0, -1.0)
    assert error == np.mean(prediction != y)


def test_q1_nu_transition_and_diagnostics():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    low, low_diag = solve_binary_q1(X, y, alpha=1.0, nu=1e-5)
    high, high_diag = solve_binary_q1(X, y, alpha=1.0, nu=10.0)
    assert low_diag["trivial_u"]
    assert not high_diag["trivial_u"]
    assert high_diag["training_error"] <= low_diag["training_error"]
    assert low.nu < high.nu
    assert WIDE_NU_GRID[-1] == pytest.approx(100.0)
    assert MODEL_SPECS["fixed_nu_sqrt10_ova"]["fixed_nu"] == pytest.approx(np.sqrt(10.0))
    assert MODEL_SPECS["expanded_per_class_nu_rbf"]["outer_nu_grid"][-1] == pytest.approx(10.0)
    np.testing.assert_allclose(SOURCE_CONTINUED_NU_GRID[:5], PAPER_NU_GRID)
    assert SOURCE_CONTINUED_NU_GRID[-2] == pytest.approx(10 ** 0.75)
    assert SOURCE_CONTINUED_NU_GRID[-1] == pytest.approx(10 ** 1.5)
    assert MODEL_SPECS["nu_cv_ova_robust_tiemax_pinf"]["rho_tie_break"] == "largest"


def test_ovo_training_subset_and_tie_rule_do_not_need_test_labels():
    X = np.array([[-2.0], [-1.5], [0.0], [0.5], [2.0], [2.5]])
    labels = np.array([1, 1, 2, 2, 3, 3])
    models = []
    for positive, negative in [(1, 2), (1, 3), (2, 3)]:
        mask = (labels == positive) | (labels == negative)
        binary = np.where(labels[mask] == positive, 1.0, -1.0)
        model, _ = solve_binary_q1(X[mask], binary, alpha=1.0, nu=10.0)
        model.positive_class, model.negative_class = positive, negative
        assert set(labels[mask]) == {positive, negative}
        models.append(model)
    first = predict_multiclass(models, X, "ovo")
    second = predict_multiclass(models, X, "ovo")
    np.testing.assert_array_equal(first, second)
    assert set(first) <= {1, 2, 3}


def test_joint_nu_selection_uses_one_model_per_class_and_training_only_scores():
    X = np.array([
        [-2.0], [-1.8], [-1.6], [-1.4],
        [-0.2], [0.0], [0.2], [0.4],
        [1.4], [1.6], [1.8], [2.0],
    ])
    labels = np.repeat([1, 2, 3], 4)
    models, diagnostics = fit_multiclass(
        X, labels, alpha=0.7, decomposition="ova", threshold_mode="grid",
        nu_grid=(0.01, 0.1), nu_selection="joint_multiclass_train",
    )
    assert len(models) == 3
    assert len(diagnostics) == 6
    assert sum(row["selected_for_task"] for row in diagnostics) == 3
    assert all("joint_multiclass_training_error" in model.diagnostics for model in models)
    assert np.mean(predict_multiclass(models, X, "ova") != labels) <= 1 / 3


def test_joint_bias_search_never_worsens_training_and_stays_in_strips():
    X = np.array([
        [-2.0], [-1.8], [-1.6], [-1.4],
        [-0.2], [0.0], [0.2], [0.4],
        [1.4], [1.6], [1.8], [2.0],
    ])
    labels = np.repeat([1, 2, 3], 4)
    models, _ = fit_multiclass(
        X, labels, alpha=0.7, decomposition="ova", threshold_mode="grid",
        nu_grid=(0.1, 1.0),
    )
    before = np.mean(predict_multiclass(models, X, "ova") != labels)
    calibrated, diag = optimize_ova_biases(models, X, labels)
    after = np.mean(predict_multiclass(calibrated, X, "ova") != labels)
    assert after <= before
    assert diag["orders_evaluated"] == 6
    for model in calibrated:
        assert model.diagnostics["strip_left"] - 1e-12 <= model.b
        assert model.b <= model.diagnostics["strip_right"] + 1e-12


def test_rkhs_normalized_ova_is_invariant_to_positive_classifier_rescaling():
    X = np.array([[-2.0], [-1.5], [0.0], [0.5], [2.0], [2.5]])
    labels = np.array([1, 1, 2, 2, 3, 3])
    models, _ = fit_multiclass(
        X, labels, alpha=1.0, decomposition="ova", threshold_mode="exact", nu=10.0
    )
    expected = predict_multiclass(models, X, "ova", normalize_ova=True)
    scaled = copy.deepcopy(models)
    scaled[0].u *= 7.0
    scaled[0].b *= 7.0
    scaled[0].diagnostics["w_norm_H"] *= 7.0
    actual = predict_multiclass(scaled, X, "ova", normalize_ova=True)
    np.testing.assert_array_equal(actual, expected)


def test_pinf_robust_radius_and_lp_are_active():
    X = np.array([[-2.0, 0.0], [-1.0, 0.5], [1.0, -0.5], [2.0, 0.0]])
    labels = np.array([1, 1, 2, 2])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    small = robust_delta(X, labels, rho=0.01, alpha=1.0)
    large = robust_delta(X, labels, rho=0.1, alpha=1.0)
    assert np.all(small > 0.0)
    assert np.all(large > small)
    _, diag = solve_binary_q1(
        X, y, alpha=1.0, nu=10.0, rho=0.01, original_labels=labels
    )
    assert diag["p"] == "inf"
    assert diag["max_delta"] > 0.0
    assert diag["solver_status_code"] == 0


def test_rho_selection_can_retain_source_per_class_nu_search():
    X = np.array([
        [-2.0], [-1.8], [-1.6], [-1.4],
        [-0.2], [0.0], [0.2], [0.4],
        [1.4], [1.6], [1.8], [2.0],
    ])
    labels = np.repeat([1, 2, 3], 4)
    rho, scores, diagnostics = select_rho(
        X, labels, "ova", "grid", "paper", nu=None, seed=9, grid=(0.0, 1e-4)
    )
    assert rho in {0.0, 1e-4}
    assert set(scores) == {"0", "0.0001"}
    # 2 rho values * 3 folds * 3 classes * 5 source nu values.
    assert len(diagnostics) == 90
    assert {row["nu"] for row in diagnostics} == set(PAPER_NU_GRID)
