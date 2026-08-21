"""
Validates Lemma 3: ||phi(u)-phi(u')||_H <= sqrt(2*a) * ||u-u'||_2 GLOBALLY
for the Gaussian kernel k(u,u') = exp(-a||u-u'||^2), with equality
approached as ||u-u'|| -> 0. Computed entirely via the kernel trick
(no explicit feature vectors needed) -- pure numpy.
"""

import numpy as np


def rbf_k(u, up, a):
    return np.exp(-a * np.sum((u - up) ** 2))


def feature_dist_sq(u, up, a):
    # ||phi(u)-phi(u')||_H^2 = k(u,u) + k(u',u') - 2k(u,u')
    return rbf_k(u, u, a) + rbf_k(up, up, a) - 2 * rbf_k(u, up, a)


def test_global_bound_never_violated():
    rng = np.random.default_rng(2)
    for _ in range(500):
        a = rng.uniform(0.01, 5.0)
        d = rng.integers(1, 15)
        u = rng.normal(size=d)
        up = u + rng.normal(size=d) * rng.uniform(0.001, 20.0)

        lhs = feature_dist_sq(u, up, a)
        rhs = 2 * a * np.sum((u - up) ** 2)
        assert lhs <= rhs + 1e-9, f"Lemma 3 violated: lhs={lhs}, rhs={rhs}, a={a}"


def test_bound_is_asymptotically_tight_near_zero():
    """As ||u-u'|| -> 0, ||phi(u)-phi(u')||_H / ||u-u'|| -> sqrt(2a) exactly
    (confirms the bound isn't just valid but the CORRECT constant, not an
    arbitrarily loose one)."""
    rng = np.random.default_rng(3)
    a = 1.3
    u = rng.normal(size=6)
    direction = rng.normal(size=6)
    direction /= np.linalg.norm(direction)

    ratios = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        up = u + eps * direction
        d = feature_dist_sq(u, up, a) ** 0.5
        ratios.append(d / eps)

    target = np.sqrt(2 * a)
    assert abs(ratios[-1] - target) < abs(ratios[0] - target), \
        "ratio should converge toward sqrt(2a) as eps shrinks"
    assert ratios[-1] <= target + 1e-6
    assert ratios[-1] > target - 1e-2
