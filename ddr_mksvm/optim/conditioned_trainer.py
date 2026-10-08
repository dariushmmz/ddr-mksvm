"""Conditioned Residual V3 trainer kept separate from frozen V3.0."""

from functools import partial

from ddr_mksvm.optim.alternating_trainer import AlternatingTrainer
from ddr_mksvm.optim.conditioned_convex_subproblem import (
    DEFAULT_CONDITION_NUMBER_CAP,
    train_with_nu_search_conditioned,
)


class ConditionedResidualTrainer(AlternatingTrainer):
    """Residual V3 with adaptive diagonal conditioning before each SOCP."""

    def __init__(self, *args, condition_number_cap=DEFAULT_CONDITION_NUMBER_CAP, **kwargs):
        self.condition_number_cap = float(condition_number_cap)
        conditioned_solver = partial(
            train_with_nu_search_conditioned,
            condition_number_cap=self.condition_number_cap,
        )
        super().__init__(*args, convex_train_fn=conditioned_solver, **kwargs)
