"""Scientific-name facade for the frozen RKHS-RBF-OVO SVM (internal V7)."""

from ..v7_cross_dataset import (  # noqa: F401
    ALPHA_RULES,
    DATASETS,
    SVC_C_GRID,
    evaluate_seed,
    load_dataset,
    paired,
    split_indices,
    summarize,
)

__all__ = [
    "ALPHA_RULES",
    "DATASETS",
    "SVC_C_GRID",
    "evaluate_seed",
    "load_dataset",
    "paired",
    "split_indices",
    "summarize",
]
