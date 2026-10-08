# Frozen source snapshots

These files are byte-identical copies of sources named in immutable experiment
manifests. They are not active entry points.

| Logical historical path | Archived source | SHA-256 |
|---|---|---|
| `run_v8_class_sensitive.py` | `run_v8_class_sensitive.py` | `3d7d11a98df32024d2f47800dc5d48b220153993c19fdb5f8464d2c8f8f7cc13` |
| `run_v8_1.py` | `run_v8_1.py` | `6ec9b5ef941fd7b9a9c0db797a41961cbd55d189b81b4c8aca4c8c490a62793b` |
| `v8_1_provenance.py` | `v8_1_provenance.py` | `be5ea84d478b09803b13ab9d0362d87454db94dc578ab795cf7f98d2e42d6efd` |

The corresponding historical Modal launchers are under `archive/modal/`.
Provenance helpers map logical names to these locations before comparing
hashes, so frozen assertions remain exact after the path migration.
