# Repository path migration

The refactor changed organization only. Frozen datasets, model mathematics,
result artifacts, and seed semantics were not changed.

| Previous path | Current path | Status |
|---|---|---|
| `modal_*.py` | `archive/modal/modal_*.py` | historical cloud wrappers; inactive |
| `sync_robust_artifacts.py` | `archive/modal/sync_robust_artifacts.py` | historical volume synchronizer |
| `sync_v8_artifacts.py` | `archive/modal/sync_v8_artifacts.py` | historical volume synchronizer |
| superseded `run_*.py`, `analyze_*.py`, `audit_*.py` | `archive/research_scripts/` | historical/rejected research drivers |
| legacy `main_*.py`, `unit_of_work_*.py`, `holdouts_*.py` | `archive/research_scripts/` | pre-frozen pipeline retained for tests/provenance |
| `test_ddr_mksvm.ipynb` | `notebooks/historical_ddr_mksvm.ipynb` | historical notebook |
| `DDR-MKSVM_spec.md` | `docs/specifications/DDR-MKSVM_SPECIFICATION.md` | original specification |
| supported root commands | `python -m ddr_mksvm.cli ...` | preferred local interface |
| frozen V8/V8.1 logical source paths | `archive/frozen_sources/` | byte-identical manifest evidence |

The following paths intentionally remain unchanged because they are recorded
in frozen manifests or are primary active entry points:

- `ddr_mksvm/v7_cross_dataset.py`
- `ddr_mksvm/v8_class_sensitive.py`
- `run_v7_cross_dataset.py`
- `archive/frozen_sources/run_v8_class_sensitive.py` retains the byte-identical frozen runner; the active local runner is `ddr_mksvm/experiments/class_sensitive_runner.py`
- `run_expanded_dataset_study.py`
- `analyze_expanded_dataset_study.py`
- `consolidate_expanded_dataset_study.py`
- `dataset/`, `results/`, `docs/`, and `paper/`

The archived `modal_v7_cross_dataset.py` remains byte-identical. The V7 freeze
verification maps its historical manifest key to
`archive/modal/modal_v7_cross_dataset.py`, so the original fingerprint remains
auditable without making Modal an active dependency.

The V8.1 logical paths `run_v8_1.py`, `modal_v8_1.py`, and
`v8_1_provenance.py` similarly resolve to byte-identical files under
`archive/frozen_sources/` and `archive/modal/`. This relocation preserves the
recorded hashes; it does not regenerate or reinterpret historical evidence.
