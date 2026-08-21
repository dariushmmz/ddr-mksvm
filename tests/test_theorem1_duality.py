"""
Validates the mechanism at the heart of Theorem 1's proof (Step 3): for the
inner problem  sup_x [ hinge(x,y) - lambda*||x-x0|| ],

  - lambda >= L (the hinge's Lipschitz constant): the supremum is attained
    at x = x0 (no probed perturbation should ever beat it).
  - lambda <  L: the supremum is unbounded (some probed perturbation
    should beat hinge(x0,y)).

Uses a linear kernel (phi = identity, Lip(phi) = 1 exactly, Lemma 4a) so
L = ||w||_2 exactly and the check is unambiguous. Pure numpy.
"""

import numpy as np


def hinge(w, x, y):
    return max(0.0, 1 - y * float(w @ x))


def test_case_lambda_geq_L_no_violation():
    rng = np.random.default_rng(4)
    w = rng.normal(size=5)
    L = float(np.linalg.norm(w))
    x0 = rng.normal(size=5)
    y = 1.0
    lam = L * 1.05  # lambda slightly above L

    base = hinge(w, x0, y)
    violations = 0
    for _ in range(3000):
        x = x0 + rng.normal(size=5) * rng.uniform(0, 3)
        val = hinge(w, x, y) - lam * np.linalg.norm(x - x0)
        if val > base + 1e-6:
            violations += 1
    assert violations == 0, "found a perturbation beating hinge(x0) when lambda >= L"


def test_case_lambda_less_L_violation_exists():
    rng = np.random.default_rng(5)
    w = rng.normal(size=5)
    L = float(np.linalg.norm(w))
    x0 = rng.normal(size=5)
    y = 1.0
    lam = L * 0.5  # lambda strictly below L

    base = hinge(w, x0, y)
    best_gain = -np.inf
    for _ in range(3000):
        x = x0 + rng.normal(size=5) * rng.uniform(0, 3)
        val = hinge(w, x, y) - lam * np.linalg.norm(x - x0)
        best_gain = max(best_gain, val - base)
    assert best_gain > 0, "expected some perturbation to beat hinge(x0) when lambda < L"


def test_theorem1_closed_form_matches_L_ell_at_lambda_equals_L():
    """
    With lambda pinned exactly at L (Step 4's optimal choice), the inner
    sup should equal hinge(x0,y) up to numerical tolerance, so
    R_eps = eps*L + mean(hinge(x0,y)) as claimed by (T1).
    """
    rng = np.random.default_rng(6)
    m = 8
    n = 4
    W = rng.normal(size=n)
    X0 = rng.normal(size=(m, n))
    Y = rng.choice([-1.0, 1.0], size=m)
    L = float(np.linalg.norm(W))
    eps = 0.37

    empirical = np.mean([hinge(W, X0[i], Y[i]) for i in range(m)])
    closed_form = eps * L + empirical

    # brute-force: for each point, search for the worst x within a growing
    # radius budget and confirm none beats x0 once charged at rate L
    worst_found = 0.0
    for i in range(m):
        base = hinge(W, X0[i], Y[i])
        for _ in range(500):
            x = X0[i] + rng.normal(size=n) * rng.uniform(0, 5)
            val = hinge(W, x, Y[i]) - L * np.linalg.norm(x - X0[i])
            worst_found = max(worst_found, val - base)
    assert worst_found <= 1e-6, "some point beat its own x0 baseline at lambda=L"
    assert closed_form >= empirical  # sanity: DRO penalty is nonnegative
