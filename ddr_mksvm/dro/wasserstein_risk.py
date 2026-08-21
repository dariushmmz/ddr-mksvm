"""
Theorem 1's closed-form Wasserstein-DRO risk:

    R_eps(w,gamma,theta,eta) = eps * L_ell + (1/m) sum_i hinge_i

with L_ell <= ||w||_H * L_{theta,eta} (Lemma 5's composite bound).

All quantities here are expressed in the DUAL / kernel representation
already used throughout the uploaded codebase: w = sum_j u_j y_j phi(x_j),
so ||w||_H^2 = u^T M u with M = outer(y,y) * K (M is exactly the matrix
already built in unit_of_work_deterministic_binary.py's training loop).
"""

import numpy as np


def hinge_losses(u, gamma, M, y):
    """Per-point hinge loss, vectorized (mirrors the existing
    `-(M @ u) + y*b` misclassification check, but as a soft hinge)."""
    scores = M @ u - y * gamma
    return np.maximum(0.0, 1.0 - scores)


def w_norm_H(u, M):
    """||w||_H = sqrt(u^T M u), M = D K D (Definition 6 / Theorem 1 proof, Step 1)."""
    val = float(u @ (M @ u))
    return float(np.sqrt(max(val, 0.0)))


def training_objective(u, gamma, M, y, epsilon, L_theta_eta):
    """
    (T-obj), Corollary 1: empirical hinge (mean, not sum, per Definition 7's
    E_Q[...] expectation) plus the DRO penalty eps * L_theta_eta * ||w||_H.

    Returns (total, empirical_term, penalty_term) for logging.
    """
    losses = hinge_losses(u, gamma, M, y)
    empirical = float(losses.mean())
    penalty = float(epsilon * L_theta_eta * w_norm_H(u, M))
    return empirical + penalty, empirical, penalty


def dro_risk_closed_form(u, gamma, M, y, epsilon, L_theta_eta):
    """
    R_eps via Theorem 1's closed form (T1): eps*L_ell + mean empirical hinge,
    with L_ell taken at its Lemma-5-derived bound ||w||_H * L_theta_eta.
    Numerically identical to training_objective's total (kept as a
    separate, clearly-named entry point matching the theorem statement,
    for use in tests/test_theorem1_duality.py-style validation).
    """
    total, _, _ = training_objective(u, gamma, M, y, epsilon, L_theta_eta)
    return total
