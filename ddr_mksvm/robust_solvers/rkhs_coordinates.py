"""Exact full-rank RKHS coordinate transformations.

No regularization, symmetrization, eigenvalue clipping, truncation, or
pseudoinverse is permitted here. Callers must handle ``LinAlgError`` when the
stored Gram matrix has no usable unmodified Cholesky factor.
"""
from __future__ import annotations

import numpy as np


def factor_full_rank_gram(K: np.ndarray) -> np.ndarray:
    """Return upper-triangular R satisfying K = R.T @ R."""

    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be square")
    if not np.all(np.isfinite(K)):
        raise ValueError("K must be finite")
    if not np.array_equal(K, K.T):
        raise ValueError("K must be exactly symmetric; implicit symmetrization is prohibited")
    return np.linalg.cholesky(K).T


def coordinates_from_coefficients(R: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(R, float) @ np.asarray(coefficients, float)


def coefficients_from_coordinates(R: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Recover c from z=Rc without forming an inverse or pseudoinverse."""

    return np.linalg.solve(np.asarray(R, float), np.asarray(coordinates, float))


def training_scores_from_coordinates(R: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Return Kc as R.T@z for K=R.T@R and z=Rc."""

    return np.asarray(R, float).T @ np.asarray(coordinates, float)


def test_scores_from_coordinates(
    cross_kernel: np.ndarray, R: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """Return K(U,X_test).T c using a triangular solve for c."""

    coefficients = coefficients_from_coordinates(R, coordinates)
    return np.asarray(cross_kernel, float).T @ coefficients
