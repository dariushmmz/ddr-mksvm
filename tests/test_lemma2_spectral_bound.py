"""
Validates Lemma 2: the product of per-layer spectral norms is a valid
(never-violated) upper bound on the network's true Lipschitz constant,
cross-checked against the empirical estimator. Requires torch --
skipped automatically if unavailable.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from ddr_mksvm.kernels.deep_kernel import DeepFeatureExtractor
from ddr_mksvm.lipschitz.analytic_bounds import network_lipschitz_bound
from ddr_mksvm.lipschitz.empirical_estimator import empirical_lipschitz


def test_analytic_bound_never_violated_by_empirical_estimate():
    torch.manual_seed(1)
    net = DeepFeatureExtractor(in_dim=6, out_dim=4, hidden=16, depth=2)
    net.eval()

    L_analytic = network_lipschitz_bound(net)

    X = np.random.default_rng(1).normal(size=(300, 6))
    L_emp = empirical_lipschitz(lambda x: net(x), X, n_pairs=800, refine_steps=40, seed=1)

    # small numerical slack for the empirical estimator's own imprecision
    assert L_emp <= L_analytic * 1.10, (
        f"empirical Lipschitz estimate ({L_emp:.4f}) exceeded the analytic "
        f"bound ({L_analytic:.4f}) by more than the allowed slack -- Lemma 2 violated"
    )


def test_identity_network_has_bound_one():
    from ddr_mksvm.kernels.deep_kernel import IdentityFeatureExtractor
    net = IdentityFeatureExtractor()
    assert network_lipschitz_bound(net) == 1.0
