"""
Analytic Lipschitz bounds (Lemma 2: network spectral-norm product bound;
Lemma 5: composite bound for the deep multiple kernel).

network_lipschitz_bound is pure numpy/torch (no cvxpy dependency), so it
is usable wherever torch is available, independent of the convex solver.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def network_lipschitz_bound(model) -> float:
    """
    Lemma 2: L_f <= prod_t ||W_t||_2 (spectral norm of each Linear layer's
    weight matrix), now also folding in DeepFeatureExtractor's learnable
    output gain (see deep_kernel.py) -- scalar multiplication by c has
    Lipschitz constant |c|, so this is a direct, trivial extension of the
    same product, not new theory. Returns 1.0 for a non-torch / identity
    model (theta empty, matching f_theta = Identity having Lip = 1 exactly).
    """
    if not _TORCH_AVAILABLE or model is None:
        return 1.0
    bound = 1.0
    found_scale = False
    for module in model.modules():
        if isinstance(module, nn.Linear):
            found_scale = True
            w = module.weight.detach()
            sv = torch.linalg.matrix_norm(w, ord=2).item()
            bound *= sv
    if hasattr(model, "log_gain"):
        found_scale = True
        bound *= float(torch.exp(model.log_gain.detach()).item())
    return float(bound) if found_scale else 1.0


def composite_lipschitz_bound(L_f: float, kernel_lipschitz_list, eta) -> float:
    """
    Lemma 5: L_{theta,eta} = L_f * sqrt( sum_l eta_l * L_phi_l^2 ).

    kernel_lipschitz_list : list[float], one Lipschitz bound per base kernel
                             (Lemma 3 for Gaussian RBF, Lemma 4 for polynomial,
                             exactly 1.0 for linear)
    eta                   : array-like on the simplex (Lemma 1), length == len(kernel_lipschitz_list)
    """
    eta = np.asarray(eta, dtype=float)
    Lphi = np.asarray(kernel_lipschitz_list, dtype=float)
    if eta.shape != Lphi.shape:
        raise ValueError(f"eta shape {eta.shape} does not match kernel_lipschitz_list shape {Lphi.shape}")
    return float(L_f * np.sqrt(np.sum(eta * (Lphi ** 2))))