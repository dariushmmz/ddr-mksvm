"""Repository-boundary and frozen-source regression checks."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FROZEN_HASHES = {
    "ddr_mksvm/v7_cross_dataset.py": "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d",
    "ddr_mksvm/iris_research.py": "a77ecbfba5109272a6b306ad6beef169734cb17d7edc45bd3d4cfb33ba575604",
    "ddr_mksvm/v8_class_sensitive.py": "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448",
    "run_v7_cross_dataset.py": "bf738847f4e0597c40c742cbdf86a954f2a286d12a65d8761953927c7555ce13",
    "archive/modal/modal_v7_cross_dataset.py": "8fd3551327c2dd5906c29273944cbddcd8630e3803bc3733c1c128927b0515f7",
    "archive/frozen_sources/run_v8_class_sensitive.py": "3d7d11a98df32024d2f47800dc5d48b220153993c19fdb5f8464d2c8f8f7cc13",
    "archive/modal/modal_v8_class_sensitive.py": "67be8d8555212adbef0a3678b2c1d2a5f9e194be37d65c8da2a63c1c3061ae45",
    "archive/frozen_sources/run_v8_1.py": "6ec9b5ef941fd7b9a9c0db797a41961cbd55d189b81b4c8aca4c8c490a62793b",
    "archive/frozen_sources/v8_1_provenance.py": "be5ea84d478b09803b13ab9d0362d87454db94dc578ab795cf7f98d2e42d6efd",
    "archive/modal/modal_v8_1.py": "cb330acdaca6a42fef08444d06c410adf214e9232a581066eeaa66da6c39c957",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_model_and_provenance_hashes_are_unchanged():
    for relative, expected in FROZEN_HASHES.items():
        assert _sha256(ROOT / relative) == expected, relative


def test_active_python_has_no_modal_import():
    excluded = {"archive", "results", ".venv", "build", "dist"}
    offenders = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] in excluded:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "modal" or name.startswith("modal.") for name in names):
                offenders.append(relative.as_posix())
    assert offenders == []


def test_cloud_launchers_are_archived_not_active_root_scripts():
    assert not list(ROOT.glob("modal_*.py"))
    assert not list(ROOT.glob("sync_*_artifacts.py"))
    assert (ROOT / "archive/modal/modal_v7_cross_dataset.py").is_file()
    assert (ROOT / "archive/modal/modal_v8_class_sensitive.py").is_file()
