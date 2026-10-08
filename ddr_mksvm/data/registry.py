"""Canonical local dataset registry.

Loading delegates to the frozen expanded-study implementation so the refactor
does not alter preprocessing, semantic repairs, or dataset fingerprints.
"""

from __future__ import annotations

from ..expanded_dataset_study import DATASETS, load_dataset

__all__ = ["DATASETS", "load_dataset"]
