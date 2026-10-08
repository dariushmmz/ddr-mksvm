"""Repository-relative paths used by local runners and documentation tools."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULTS_ROOT = PROJECT_ROOT / "results"
DOCS_ROOT = PROJECT_ROOT / "docs"
PAPER_ROOT = PROJECT_ROOT / "paper"
CONFIG_ROOT = PROJECT_ROOT / "configs"


def project_path(*parts: str) -> Path:
    """Return an absolute path below the repository root."""

    return PROJECT_ROOT.joinpath(*parts)
