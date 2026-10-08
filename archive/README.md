# Historical code archive

This directory preserves code that is scientifically relevant but not part of
the supported execution path.

- `modal/`: byte-preserved cloud launch/synchronization wrappers used for the
  historical campaigns. Modal is not an active dependency.
- `research_scripts/`: superseded, rejected, or one-off architecture and robust
  experiment drivers. Core mathematical implementations and tests remain in
  `ddr_mksvm/` and `tests/`.
- `frozen_sources/`: byte-identical source copies required to resolve logical
  paths recorded by immutable V8/V8.1 manifests after the repository move.

Historical documentation and result manifests may refer to the original root
paths. See `docs/REPOSITORY_MIGRATION.md`. Archived scripts are retained for
auditability, not advertised as current commands.
