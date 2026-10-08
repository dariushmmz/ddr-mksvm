# Repository refactor verification report

Date: 2026-09-27

## Outcome

The project is now an installable, local-first scientific Python repository.
Cloud launchers are inactive and preserved only for provenance. The refactor
did not change frozen model mathematics, datasets, seed registries, or result
artifacts.

## Verification

| Check | Result |
|---|---|
| Regression suite before path migration | 184 passed, 1 external deprecation warning |
| Interim targeted suite | 52 passed, 2 V8.1 logical-path failures |
| V8.1 provenance repair | exact snapshot mapping; no assertion relaxation |
| Targeted suite after repair | 54 passed, 1 external deprecation warning |
| Final suite (including three repository-boundary tests) | 187 passed, 1 external deprecation warning |
| Editable installation | passed (`python -m pip install -e . --no-deps`) |
| Import outside repository working directory | passed |
| Active Python compilation | passed |
| Active Modal-import AST audit | passed; zero imports |
| Core Markdown link check | passed |

The warning originates in the installed `pytz` package and does not indicate a
project test failure.

## Frozen hash verification

| Artifact | SHA-256 status |
|---|---|
| `ddr_mksvm/v7_cross_dataset.py` | unchanged: `39a2a2c5...6746e06d` |
| `ddr_mksvm/v8_class_sensitive.py` | unchanged: `fd17cce0...100b448` |
| frozen V8 runner | unchanged: `3d7d11a9...8f7cc13` |
| frozen V8.1 runner | unchanged: `6ec9b5ef...a62793b` |
| frozen V8.1 provenance helper | unchanged: `be5ea84d...42d6efd` |
| archived V8.1 Modal launcher | unchanged: `cb330acd...39c957` |

Logical names in immutable manifests are resolved through the explicit map in
`archive/research_scripts/v8_1_provenance.py`. The historical files themselves
remain byte-identical.

## Local smoke experiment

- Run: `results/local_smoke/refactor-smoke-17000`
- Dataset/seed: Breast Cancer Recurrence / 17000
- Models: Source-Profile q1, RKHS-RBF-OVO, CS-RKHS-RBF-OVO
- Jobs: 1 expected, 1 complete, 0 failed
- Records: 3
- Numerical-validity decision: passed
- Job wall time: 2.097 s; stage wall time: 2.209 s

| Model | Error | Balanced accuracy | Macro F1 | Runtime (s) |
|---|---:|---:|---:|---:|
| Legacy | 0.2500 | 0.6415 | 0.6535 | 1.3954 |
| RKHS-RBF-OVO (`v7`) | 0.2500 | 0.6555 | 0.6667 | 0.2566 |
| CS-RKHS-RBF-OVO (`v8_c`) | 0.2778 | 0.6499 | 0.6538 | 0.4284 |

This is an execution smoke, not new scientific confirmation evidence.

## Active repository boundaries

The active package contains no `import modal` or `from modal` statement. One
active provenance check reads the archived `modal_v7_cross_dataset.py` bytes to
verify its historical hash; it does not import or execute Modal. The frozen V7
runner also retains a legacy `modal_resources` metadata field because changing
that file would invalidate its recorded hash. Neither item creates a runtime
dependency.

The archived `running_deterministic_version.py` contains notebook-style `!pip`
syntax and is intentionally not an importable Python module. It is historical
source evidence, excluded from the active package and compilation path.

## Primary migration map

See `docs/REPOSITORY_MIGRATION.md` for the complete mapping. In summary:

- cloud wrappers and synchronizers -> `archive/modal/`;
- superseded research drivers -> `archive/research_scripts/`;
- byte-identical manifest sources -> `archive/frozen_sources/`;
- supported local orchestration -> `python -m ddr_mksvm.cli`;
- scientific package interfaces -> `ddr_mksvm/{models,data,evaluation,experiments}`.

## Remaining migration notes

- Historical reports intentionally retain their original Modal commands and
  paths; the historical Modal guide is prominently marked inactive.
- Result manifests retain original logical filenames. Provenance maps resolve
  those names rather than rewriting immutable evidence.
- Long confirmation reproductions are now local and checkpointed, but their
  elapsed time depends on the workstation; no new confirmation was run during
  this structural refactor.
