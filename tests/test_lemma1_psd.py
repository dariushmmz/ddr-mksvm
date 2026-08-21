"""
Validates Lemma 1: a conic (nonnegative, simplex-weighted) combination of
PSD kernel matrices is PSD. Pure numpy -- no cvxpy/torch dependency.
"""

import numpy as np


def _random_gram(m, rank, rng):
    A = rng.standard_normal((m, rank))
    return A @ A.T  # PSD by construction


def test_conic_combination_is_psd():
    rng = np.random.default_rng(0)
    m = 25
    K1 = _random_gram(m, 5, rng)
    K2 = _random_gram(m, 9, rng)
    K3 = _random_gram(m, 3, rng)

    for _ in range(25):
        eta = rng.dirichlet(np.ones(3))  # simplex sample
        K = eta[0] * K1 + eta[1] * K2 + eta[2] * K3
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() > -1e-8, f"combined kernel not PSD, min eig={eigvals.min()}"


def test_two_kernel_edge_cases():
    rng = np.random.default_rng(1)
    m = 15
    K1 = _random_gram(m, 4, rng)
    K2 = _random_gram(m, 4, rng)

    for eta1 in (0.0, 1.0, 0.5):
        eta = np.array([eta1, 1 - eta1])
        K = eta[0] * K1 + eta[1] * K2
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() > -1e-8
