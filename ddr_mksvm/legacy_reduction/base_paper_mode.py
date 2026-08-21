"""
Proposition 4.8: f_theta = Identity, a single fixed kernel, epsilon = 0
must reduce DDR-MKSVM exactly to the base paper's original deterministic
LP (i.e. to unit_of_work_deterministic_binary.py's training phase).

This module is the "new code path with every extension disabled" side
of that equivalence -- tests/test_reduction_to_base_paper.py is the
"old code path" side. They must agree numerically.
"""

import numpy as np

from ddr_mksvm.optim.convex_subproblem import train_with_nu_search


def run_legacy_binary(dati, y, degree=2, c=None, nu_grid=None):
    """
    dati : (n, m) -- features as columns, exactly as built in
           unit_of_work_deterministic_binary.py (np.hstack([Atrain.T, Btrain.T])).
    y    : (m,) in {-1, +1}
    """
    nu_grid = nu_grid if nu_grid is not None else np.logspace(-3, 0, 5)
    if c is None:
        c = float(np.max(np.std(dati, axis=1, ddof=0)))

    K = (dati.T @ dati + c) ** degree
    best = train_with_nu_search(K, y, nu_grid, epsilon=0.0, L_theta_eta=0.0,
                                formulation="legacy_q1")
    return best, K
