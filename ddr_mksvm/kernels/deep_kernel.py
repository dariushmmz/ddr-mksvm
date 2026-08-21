"""
Deep feature extractor f_theta (Definition 3, DDR-MKSVM_spec.md).

Requires torch. Import is deferred / guarded so the rest of the package
(base kernels, Lipschitz lemma checks) remains usable in environments
without torch installed.
"""

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.parametrizations import spectral_norm
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for DeepFeatureExtractor / dnn_on=True. "
            "Install it (see requirements.txt) or run with dnn_on=False "
            "(Proposition 4.8 legacy/reduction mode)."
        )


if _TORCH_AVAILABLE:

    class DeepFeatureExtractor(nn.Module):
        """
        f_theta: R^n -> R^d, a small spectral-normalized MLP.

        Spectral normalization is used (rather than plain nn.Linear) so
        that each layer's spectral norm ||W_t||_2 is directly available
        for Lemma 2's analytic Lipschitz bound (product of per-layer
        spectral norms) without needing a separate power-iteration pass
        at inference time.

        Sized conservatively (small hidden width, depth <= 3) because the
        target datasets are small (m as low as ~68, per the reproduction
        report) -- a large network here would both overfit and inflate
        Lemma 2's already-loose analytic bound (see spec Section 6.4).
        """

        def __init__(self, in_dim, out_dim=None, hidden=None, depth=2):
            super().__init__()
            out_dim = out_dim or max(4, in_dim)
            hidden = hidden or min(64, max(8, 4 * in_dim))
            layers = []
            d_in = in_dim
            for _ in range(max(depth - 1, 0)):
                layers.append(spectral_norm(nn.Linear(d_in, hidden)))
                layers.append(nn.ReLU())
                d_in = hidden
            layers.append(spectral_norm(nn.Linear(d_in, out_dim)))
            self.net = nn.Sequential(*layers)
            self.in_dim = in_dim
            self.out_dim = out_dim

            # Learnable output GAIN -- a single scalar, equivalent to one
            # more "layer" that's just scalar multiplication. This is the
            # fix for a real bug found via debugging on the Parkinson
            # dataset: a spectral-normalized network at random init can
            # produce outputs orders of magnitude smaller than the raw
            # input features, and the fixed nu_grid used by the convex
            # subproblem is calibrated against the RAW feature scale (it's
            # the exact same grid used by the deterministic baseline). A
            # kernel built on artificially tiny features makes the LP's
            # regularization-vs-margin tradeoff prefer u=0 (majority-class
            # classifier) for every nu in the grid -- confirmed
            # numerically: scaling a normal-scale K down by 1e-4 makes
            # u=0 persist across nu in [0.001, 1.0], matching what was
            # observed. Calibrated once at the start of training (see
            # AlternatingTrainer._calibrate_output_gain) and then further
            # tunable by gradient descent like any other parameter in theta.
            # network_lipschitz_bound() (ddr_mksvm/lipschitz/analytic_bounds.py)
            # folds exp(log_gain) into the same spectral-norm product it
            # already computes -- no new theory needed, since the
            # Lipschitz constant of scalar multiplication by c is |c|.
            self.log_gain = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            # x: (batch, in_dim) -> (batch, out_dim)
            return self.net(x) * torch.exp(self.log_gain)

    class IdentityFeatureExtractor(nn.Module):
        """
        f_theta = Identity (theta is the empty parameter set).

        Used for dnn_on=False / the "DNN off" ablation and for
        legacy_reduction/base_paper_mode.py (Proposition 4.8).
        """

        def forward(self, x):
            return x

else:  # pragma: no cover
    class DeepFeatureExtractor:  # type: ignore
        def __init__(self, *a, **k):
            _require_torch()

    class IdentityFeatureExtractor:  # type: ignore
        def __init__(self, *a, **k):
            _require_torch()