"""Artifact-only V8.1 regression checks; no training or remote calls."""
import ast
import json
from pathlib import Path
import pytest
from archive.research_scripts.manage_v8_1 import analyze


def test_wrapper_keeps_local_provenance_import_local():
    tree=ast.parse(Path('archive/modal/modal_v8_1.py').read_text())
    assert not any(isinstance(n,ast.ImportFrom) and n.module=='v8_1_provenance' for n in tree.body)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assert any(isinstance(n,ast.ImportFrom) and n.module=='v8_1_provenance' for n in main.body)


@pytest.mark.parametrize('run_name',['smoke/v8-1-smoke-20260915','gate8/v8-1-gate8-20260915'])
def test_existing_evidence_independent_replay(run_name):
    root=Path('results/v8_1')/run_name
    if not ((root/'manifest.json').exists() or (root/'analysis/pruning_manifest.json').exists()):pytest.skip('Evidence not synced')
    analyze(run_name)
    cfg=json.loads((root/'config.json').read_text())
    assert all(11000<=s<=11023 for s in cfg['seeds'])
    for path in (root/'checkpoints').glob('*.json'):
        j=json.loads(path.read_text())
        if 'reused_from' in j:assert j['new_job_wall_s']==0
