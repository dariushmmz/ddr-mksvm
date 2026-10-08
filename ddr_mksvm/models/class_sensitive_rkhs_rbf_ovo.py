"""Scientific-name facade for the frozen class-sensitive RKHS-RBF-OVO SVM."""

from ..v8_class_sensitive import (  # noqa: F401
    RATIOS,
    VARIANTS,
    disabled_fit_predict,
    evaluate_variants,
    fit_model,
    fit_selected,
    pair_weights,
    predict_model,
    select_family,
    vote,
)

__all__ = [
    "RATIOS",
    "VARIANTS",
    "disabled_fit_predict",
    "evaluate_variants",
    "fit_model",
    "fit_selected",
    "pair_weights",
    "predict_model",
    "select_family",
    "vote",
]
