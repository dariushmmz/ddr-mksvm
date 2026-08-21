"""
Multiple-kernel mixture (Definition 4) with eta constrained to the
simplex via softmax reparameterization -- this satisfies Lemma 1's
precondition (eta >= 0, sum(eta) = 1) automatically, by construction,
for every value of the underlying unconstrained parameter, so no
constrained optimizer / projection step is needed.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class KernelMixture(nn.Module):
        def __init__(self, n_kernels, init="uniform"):
            super().__init__()
            self.n_kernels = n_kernels
            self.raw = nn.Parameter(torch.zeros(n_kernels))  # softmax(0,...,0) = uniform

        def eta(self):
            """Differentiable torch tensor on the simplex (Lemma 1's precondition)."""
            return torch.softmax(self.raw, dim=0)

        def eta_numpy(self):
            return self.eta().detach().numpy()

else:  # pragma: no cover
    class KernelMixture:  # type: ignore
        def __init__(self, n_kernels, init="uniform"):
            raise ImportError("torch is required for KernelMixture / mkl_on=True with >1 kernel.")


def combine_grams(gram_list, eta):
    """
    Numpy-only combination K = sum_l eta_l * K_l (Definition 4), used both
    by the pure-numpy legacy path and for logging/validation of the torch
    path's combined Gram matrix.
    """
    eta = np.asarray(eta, dtype=float)
    assert np.all(eta >= -1e-9) and abs(eta.sum() - 1.0) < 1e-6, \
        "eta must lie on the probability simplex (Lemma 1's precondition)"
    K = np.zeros_like(gram_list[0])
    for w, G in zip(eta, gram_list):
        K = K + w * G
    return K
