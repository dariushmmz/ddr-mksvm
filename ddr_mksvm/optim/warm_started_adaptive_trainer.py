"""Adaptive-bandwidth Residual V3 with opt-in warm-started nu grids."""

from ddr_mksvm.optim.adaptive_bandwidth_trainer import (
    AdaptiveBandwidthResidualTrainer,
)
from ddr_mksvm.optim.warm_started_convex_subproblem import (
    train_with_nu_search_warm_started,
)


class WarmStartedAdaptiveBandwidthTrainer(AdaptiveBandwidthResidualTrainer):
    """Adaptive-bandwidth V3 using one reusable CVXPY template per grid."""

    def __init__(self, *args, **kwargs):
        kwargs["convex_train_fn"] = train_with_nu_search_warm_started
        super().__init__(*args, **kwargs)
