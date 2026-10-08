"""Stable scientific model interfaces.

The historical implementation files remain at their frozen paths so their
published SHA-256 fingerprints remain verifiable.  Modules in this package
provide descriptive scientific import names without copying the algorithms.
"""

from .class_sensitive_rkhs_rbf_ovo import evaluate_variants
from .rkhs_rbf_ovo import evaluate_seed

__all__ = ["evaluate_seed", "evaluate_variants"]
