"""Arbitrary-precision reconstruction of frozen Gaussian Gram matrices."""
from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np


def _flint():
    try:
        from flint import arb, arb_mat, ctx
    except ImportError as exc:  # pragma: no cover - optional audit dependency
        raise ImportError("R3.3 requires python-flint") from exc
    return arb, arb_mat, ctx


def exact_binary_arb(value: float):
    """Represent one finite binary64 value exactly as an Arb rational."""

    arb, _, _ = _flint()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("high-precision kernel inputs must be finite")
    numerator, denominator = value.as_integer_ratio()
    return arb(numerator) / arb(denominator)


def reconstruct_gaussian_gram(X: np.ndarray, alpha: float, precision_bits: int):
    """Evaluate each mathematical kernel entry once and mirror it exactly."""

    arb, arb_mat, ctx = _flint()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite matrix")
    if not float(alpha) > 0 or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite and positive")
    ctx.prec = int(precision_bits)
    exact_X = [[exact_binary_arb(value) for value in row] for row in X]
    exact_alpha = exact_binary_arb(alpha)
    denominator = arb(2) * exact_alpha * exact_alpha
    n, d = X.shape
    K = arb_mat(n, n)
    one = arb(1)
    for i in range(n):
        K[i, i] = one
        for j in range(i + 1, n):
            distance = arb(0)
            for column in range(d):
                difference = exact_X[i][column] - exact_X[j][column]
                distance += difference * difference
            value = (-distance / denominator).exp()
            K[i, j] = value
            K[j, i] = value
    return K


def interval_cholesky(K, precision_bits: int) -> tuple[Any | None, dict[str, Any]]:
    """Unpivoted interval Cholesky; every pivot must be provably positive."""

    arb, arb_mat, ctx = _flint()
    ctx.prec = int(precision_bits)
    n = K.nrows()
    lower = arb_mat(n, n)
    minimum_pivot_lower = math.inf
    minimum_pivot_midpoint = math.inf
    started = time.perf_counter()
    for i in range(n):
        for j in range(i + 1):
            residual = K[i, j]
            for k in range(j):
                residual -= lower[i, k] * lower[j, k]
            if i == j:
                lower_bound = float(residual.lower())
                midpoint = float(residual.mid())
                minimum_pivot_lower = min(minimum_pivot_lower, lower_bound)
                minimum_pivot_midpoint = min(minimum_pivot_midpoint, midpoint)
                if not lower_bound > 0.0:
                    return None, {
                        "success": False,
                        "failed_pivot": i,
                        "failed_pivot_interval": residual.str(max(20, int(precision_bits * .302) + 5), radius=True),
                        "minimum_pivot_lower": minimum_pivot_lower,
                        "minimum_pivot_midpoint": minimum_pivot_midpoint,
                        "runtime_sec": float(time.perf_counter() - started),
                    }
                lower[i, j] = residual.sqrt()
            else:
                lower[i, j] = residual / lower[j, j]
    return lower, {
        "success": True,
        "failed_pivot": None,
        "failed_pivot_interval": None,
        "minimum_pivot_lower": minimum_pivot_lower,
        "minimum_pivot_midpoint": minimum_pivot_midpoint,
        "runtime_sec": float(time.perf_counter() - started),
    }


def reconstruction_diagnostics(K, lower, precision_bits: int) -> dict[str, Any]:
    """Certify that every reconstructed-entry residual interval contains zero."""

    _, arb_mat, ctx = _flint()
    ctx.prec = int(precision_bits)
    started = time.perf_counter()
    residual = K - lower * lower.transpose()
    zero = arb_mat(K.nrows(), K.ncols())
    contains_zero = bool(residual.contains(zero))
    maximum_upper = 0.0
    for i in range(K.nrows()):
        for j in range(K.ncols()):
            maximum_upper = max(maximum_upper, float(residual[i, j].abs_upper()))
    return {
        "residual_contains_zero_entrywise": contains_zero,
        "reconstruction_max_abs_upper": maximum_upper,
        # K_ii=1 exactly, so the entrywise relative scale is one.
        "reconstruction_max_relative_upper": maximum_upper,
        "reconstruction_runtime_sec": float(time.perf_counter() - started),
    }


def matrix_ball_hash(matrix, precision_bits: int) -> str:
    digits = max(20, int(precision_bits * .302) + 5)
    digest = hashlib.sha256()
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            digest.update(matrix[i, j].str(digits, radius=True).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def float_comparison(high_precision, stored: np.ndarray) -> dict[str, Any]:
    stored = np.asarray(stored, dtype=np.float64)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for i in range(stored.shape[0]):
        for j in range(stored.shape[1]):
            midpoint = float(high_precision[i, j].mid())
            difference = abs(midpoint - float(stored[i, j]))
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(
                maximum_relative,
                difference / max(abs(midpoint), np.finfo(float).tiny),
            )
    return {
        "stored_float_max_absolute_difference": maximum_absolute,
        "stored_float_max_relative_difference": maximum_relative,
    }


def approximate_spectrum(K, precision_bits: int, rank_bits: tuple[int, ...]):
    """Compute approximate high-precision eigenvalues for reporting only."""

    _, _, ctx = _flint()
    ctx.prec = int(precision_bits)
    started = time.perf_counter()
    eigenvalues = K.eig(algorithm="approx")
    values = sorted(float(value.real.mid()) for value in eigenvalues)
    maximum_imaginary = max(abs(float(value.imag.mid())) for value in eigenvalues)
    largest = values[-1]
    smallest = values[0]
    ranks = {
        str(bits): int(sum(value > largest * 2.0 ** (-bits) for value in values))
        for bits in rank_bits
    }
    return {
        "smallest_eigenvalue_approx": smallest,
        "largest_eigenvalue_approx": largest,
        "maximum_eigenvalue_imaginary_midpoint": maximum_imaginary,
        "spectral_condition_approx": largest / smallest if smallest > 0 else math.inf,
        "relative_threshold_ranks": ranks,
        "spectrum_runtime_sec": float(time.perf_counter() - started),
    }


def factor_spectrum_estimate(lower, rank_bits: tuple[int, ...]) -> dict[str, Any]:
    """Estimate K's spectrum from the much better-conditioned Cholesky factor.

    This is reporting-only: singular values of L are computed in float64 after
    taking midpoints of a certified arbitrary-precision factor. Since K=L L^T,
    its eigenvalues are the squared singular values of L.
    """

    started = time.perf_counter()
    n = lower.nrows()
    factor = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            factor[i, j] = float(lower[i, j].mid())
    singular = np.linalg.svd(factor, compute_uv=False)
    eigen_estimates = np.square(singular)
    largest = float(eigen_estimates[0])
    smallest = float(eigen_estimates[-1])
    ranks = {
        str(bits): int(np.sum(eigen_estimates > largest * 2.0 ** (-bits)))
        for bits in rank_bits
    }
    return {
        "spectrum_method": "float64_svd_of_certified_high_precision_cholesky_midpoint",
        "smallest_eigenvalue_approx": smallest,
        "largest_eigenvalue_approx": largest,
        "maximum_eigenvalue_imaginary_midpoint": 0.0,
        "spectral_condition_approx": largest / smallest if smallest > 0 else math.inf,
        "relative_threshold_ranks": ranks,
        "spectrum_runtime_sec": float(time.perf_counter() - started),
    }
