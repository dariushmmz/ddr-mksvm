import numpy as np
from scipy.optimize import linprog

from ddr_mksvm.lp_equilibration import (
    EQUILIBRATION_PASSES,
    UPDATE_MAX,
    UPDATE_MIN,
    bound_violation,
    equilibrate_lp,
    to_original_coordinates,
    to_scaled_coordinates,
)
from ddr_mksvm.robust_source_aligned import build_weighted_q1_lp


def example_lp():
    c = np.array([3.0, -2.0, 0.5])
    A = np.array([[1.0e8, -2.0, 0.0], [-3.0e-8, 0.0, 4.0]])
    b = np.array([2.0e8, 5.0])
    bounds = [(None, None), (0.0, None), (-2.0, 7.0)]
    return c, A, b, bounds


def test_coordinate_map_is_bijective_and_residuals_scale_exactly():
    c, A, b, bounds = example_lp()
    transformed = equilibrate_lp(c, A, b, bounds)
    x = np.array([1.0, 2.0, -1.0])
    z = to_scaled_coordinates(x, transformed.variable_scale)
    np.testing.assert_allclose(to_original_coordinates(z, transformed.variable_scale), x)
    np.testing.assert_allclose(
        transformed.b - transformed.A @ z,
        transformed.row_scale * (b - A @ x),
        rtol=2e-15,
        atol=1e-14,
    )


def test_objective_and_bounds_map_without_changing_signs():
    c, A, b, bounds = example_lp()
    transformed = equilibrate_lp(c, A, b, bounds)
    x = np.array([1.0, 2.0, -1.0])
    z = to_scaled_coordinates(x, transformed.variable_scale)
    np.testing.assert_allclose(transformed.c @ z, c @ x, rtol=2e-15, atol=1e-14)
    assert transformed.bounds[0] == (None, None)
    assert transformed.bounds[1][0] == 0.0
    assert transformed.bounds[1][1] is None
    assert transformed.bounds[2][0] < 0 < transformed.bounds[2][1]
    assert bound_violation(x, bounds) == 0.0
    assert bound_violation(z, transformed.bounds) == 0.0


def test_scaling_rule_is_frozen_positive_and_does_not_mutate_inputs():
    c, A, b, bounds = example_lp()
    originals = (c.copy(), A.copy(), b.copy())
    transformed = equilibrate_lp(c, A, b, bounds)
    np.testing.assert_array_equal(c, originals[0])
    np.testing.assert_array_equal(A, originals[1])
    np.testing.assert_array_equal(b, originals[2])
    assert transformed.passes == EQUILIBRATION_PASSES == 8
    assert transformed.update_min == UPDATE_MIN == 1e-4
    assert transformed.update_max == UPDATE_MAX == 1e4
    assert np.all(transformed.row_scale > 0)
    assert np.all(transformed.variable_scale > 0)
    assert np.count_nonzero(A) == np.count_nonzero(transformed.A)
    assert np.count_nonzero(c) == np.count_nonzero(transformed.c)


def test_pre_scaling_q1_construction_is_identical_for_both_weight_modes():
    K = np.array([[2.0, 0.25], [0.25, 3.0]])
    y = np.array([1.0, -1.0])
    delta = np.array([0.1, 0.2])
    for weighted in (False, True):
        original = build_weighted_q1_lp(K, y, 0.01, delta, class_sensitive=weighted)
        before = tuple(value.copy() if isinstance(value, np.ndarray) else list(value) for value in original[:4])
        transformed = equilibrate_lp(*original[:4])
        for got, expected in zip(original[:3], before[:3]):
            np.testing.assert_array_equal(got, expected)
        assert original[3] == before[3]
        np.testing.assert_allclose(
            transformed.A,
            transformed.row_scale[:, None] * original[1] * transformed.variable_scale[None, :],
            rtol=2e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(
            transformed.b, transformed.row_scale * original[2], rtol=2e-15, atol=0.0
        )
        np.testing.assert_allclose(
            transformed.c, original[0] * transformed.variable_scale, rtol=2e-15, atol=0.0
        )


def test_scaled_solution_maps_to_same_synthetic_optimum():
    c = np.array([1.0, 2.0])
    A = np.array([[-1.0e6, -1.0], [1.0, 1.0e-6]])
    b = np.array([-1.0e6, 3.0])
    bounds = [(0.0, None), (0.0, None)]
    direct = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    transformed = equilibrate_lp(c, A, b, bounds)
    scaled = linprog(
        transformed.c,
        A_ub=transformed.A,
        b_ub=transformed.b,
        bounds=transformed.bounds,
        method="highs",
    )
    assert direct.success and scaled.success
    mapped = to_original_coordinates(scaled.x, transformed.variable_scale)
    np.testing.assert_allclose(c @ mapped, direct.fun, rtol=1e-10, atol=1e-10)
    assert np.max(A @ mapped - b) <= 1e-8
    assert bound_violation(mapped, bounds) <= 1e-8
