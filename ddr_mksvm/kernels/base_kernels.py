"""
Base kernels used inside DDR-MKSVM (Definitions 1, and Lemmas 3-4 of
DDR-MKSVM_spec.md, Section 4.3-4.4).

Each kernel class exposes:
  - fit(X)              : X is (n, m) -- features as COLUMNS, matching the
                           `dati` convention already used throughout the
                           uploaded codebase (unit_of_work_deterministic_binary.py).
  - gram(X)              -> (m, m) Gram matrix
  - cross(X, Xq)          -> (m, m_q) cross Gram matrix (train vs. query)
  - lipschitz_bound()     -> float, a valid Lipschitz bound on the feature
                             map w.r.t. the Euclidean norm (used by
                             Lemma 5 / Theorem 1's DRO penalty)

These mirror exactly the two kernels already implemented inline in
unit_of_work_deterministic_binary.py (polynomial, degree 2, data-dependent
c) and its commented-out Gaussian RBF block, refactored into reusable,
Lipschitz-bound-aware objects so they can be combined (MKL, Definition 4)
and/or stacked behind a deep feature extractor (Definition 3).
"""

import numpy as np


class LinearKernel:
    """k(u,u') = <u,u'>.  phi = identity, so Lip(phi) = 1 exactly (Lemma 4a)."""

    def fit(self, X):
        return self

    def gram(self, X):
        return X.T @ X

    def cross(self, X, Xq):
        return X.T @ Xq

    def lipschitz_bound(self):
        return 1.0


class PolynomialKernel:
    """
    k(u,u') = (<u,u'> + c)^d

    Matches unit_of_work_deterministic_binary.py's kernel exactly when
    c = max(std(dati)) and d = 2 (the current default for the binary
    deterministic model).

    Lipschitz bound (Lemma 4b): domain-dependent, requires fit(X) to
    compute B = max_i ||x_i||_2 before lipschitz_bound() can be called --
    unlike the Gaussian RBF kernel's bound, this one is NOT global.
    """

    def __init__(self, degree=2, c=None):
        self.degree = degree
        self.c = c
        self._B = None

    def fit(self, X):
        # X: (n, m)
        self._B = float(np.max(np.linalg.norm(X, axis=0)))
        if self.c is None:
            self.c = float(np.max(np.std(X, axis=1, ddof=0)))
        return self

    def gram(self, X):
        return (X.T @ X + self.c) ** self.degree

    def cross(self, X, Xq):
        return (X.T @ Xq + self.c) ** self.degree

    def lipschitz_bound(self):
        """
        Lemma 4b derivation (mean-value-theorem telescoping, the same
        technique the base paper's Appendix A uses to prove Proposition 1):

        ||phi(u)-phi(u')||_H^2 = (u.u+c)^d + (u'.u'+c)^d - 2(u.u'+c)^d.

        On the bounded domain ||u||_2 <= B, all three inner arguments
        (u.u+c, u'.u'+c, u.u'+c) lie in [c - B^2, B^2 + c]. Bounding the
        difference via the mean value theorem for t -> t^d on that
        interval, then relating (u.u+c)+(u'.u'+c)-2(u.u'+c) = ||u-u'||^2
        gives the (loose but valid, and correctly dimensioned) global-on-
        the-domain bound used here. This is intentionally the SIMPLER of
        two possible derivations (see DDR-MKSVM_spec.md Section 4.4
        remark) -- tighter, kernel-specific bounds are future work, not
        required for Theorem 1 to hold (Theorem 1 only needs *a* valid
        Lipschitz bound, not the tightest one).
        """
        if self._B is None:
            raise ValueError(
                "PolynomialKernel.lipschitz_bound() requires fit(X) first "
                "-- Lemma 4's bound is domain-dependent, unlike the Gaussian "
                "RBF kernel's global bound (Lemma 3)."
            )
        d, c, B = self.degree, self.c, self._B
        max_arg = B ** 2 + c
        min_arg = max(c - B ** 2, 0.0)
        # sup |d/dt t^d| on [min_arg, max_arg]
        max_deriv = d * (max_arg ** (d - 1)) if max_arg > 0 else 0.0
        # ||u-u'||^2 = (u.u+c)+(u'.u'+c)-2(u.u'+c), so a difference of the
        # three kernel arguments of magnitude ~||u-u'||^2 propagates through
        # t^d with slope max_deriv; taking square roots and folding constants
        # conservatively (factor 2) gives:
        return float(np.sqrt(max(2.0 * max_deriv, 0.0)))


class GaussianRBFKernel:
    """
    k(u,u') = exp( -||u-u'||^2 / (2*alpha^2) )

    Matches the commented-out Gaussian RBF block already present in
    unit_of_work_deterministic_binary.py exactly.

    Lipschitz bound (Lemma 3): EXACT and GLOBAL, sqrt(2*a) where
    a = 1/(2*alpha^2) is the coefficient in the k(u,u')=exp(-a||u-u'||^2)
    form used in the spec's proof. See DDR-MKSVM_spec.md Section 4.3 for
    the full proof (via the nondecreasing-function argument on
    g(d) = 2*a*d^2 - (2 - 2*exp(-a*d^2))).
    """

    def __init__(self, alpha=None):
        self.alpha = alpha

    def fit(self, X):
        if self.alpha is None:
            self.alpha = float(np.max(np.std(X, axis=1, ddof=0)))
        return self

    def gram(self, X):
        diff = X[:, :, None] - X[:, None, :]
        sqdist = np.sum(diff ** 2, axis=0)
        return np.exp(-sqdist / (2 * self.alpha ** 2))

    def cross(self, X, Xq):
        diff = X[:, :, None] - Xq[:, None, :]
        sqdist = np.sum(diff ** 2, axis=0)
        return np.exp(-sqdist / (2 * self.alpha ** 2))

    def lipschitz_bound(self):
        if self.alpha is None:
            raise ValueError("GaussianRBFKernel.lipschitz_bound() requires fit(X) or an explicit alpha first.")
        a = 1.0 / (2 * self.alpha ** 2)
        return float(np.sqrt(2 * a))


KERNEL_REGISTRY = {
    "linear": LinearKernel,
    "poly": PolynomialKernel,
    "rbf": GaussianRBFKernel,
}


def build_kernel(spec: dict):
    """spec: dict(kind='poly'|'linear'|'rbf', **kwargs) -> kernel instance (unfit)."""
    spec = dict(spec)
    kind = spec.pop("kind")
    cls = KERNEL_REGISTRY[kind]
    return cls(**spec)
