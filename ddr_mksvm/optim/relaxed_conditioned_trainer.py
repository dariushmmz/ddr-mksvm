"""Residual V3 conditioning aligned with the solver's null-space cutoff.

This is a separate opt-in variant.  It reuses the frozen conditioning
implementation but relaxes the condition-number cap from 1e8 to 1e10 so
formerly-null directions are not lifted above the convex solver's existing
1e-10 relative spectral threshold.
"""

from functools import partial

from ddr_mksvm.optim.alternating_trainer import AlternatingTrainer
from ddr_mksvm.optim.conditioned_convex_subproblem import (
    train_with_nu_search_conditioned,
)


RELAXED_CONDITION_NUMBER_CAP = 1e10


class RelaxedConditionedResidualTrainer(AlternatingTrainer):
    """Residual V3 with the distinct, relaxed 1e10 conditioning cap."""

    def __init__(
        self,
        *args,
        condition_number_cap=RELAXED_CONDITION_NUMBER_CAP,
        **kwargs,
    ):
        self.condition_number_cap = float(condition_number_cap)
        conditioned_solver = partial(
            train_with_nu_search_conditioned,
            condition_number_cap=self.condition_number_cap,
        )
        super().__init__(*args, convex_train_fn=conditioned_solver, **kwargs)
