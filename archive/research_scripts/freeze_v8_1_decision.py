"""Preserve a rejected development candidate; never authorize confirmation."""
import json
from pathlib import Path
from archive.research_scripts.v8_1_provenance import digest,verify_baselines,write_new


def main():
    count=verify_baselines()
    root=Path('results/v8_1');gate_root=root/'gate8/v8-1-gate8-20260915'
    gate=json.loads((gate_root/'analysis/gate_decision.json').read_text())
    partial=json.loads((gate_root/'analysis/pruning_manifest.json').read_text())
    assert gate['stage']=='gate8' and not gate['passes'] and gate['failed']
    assert not (root/'analysis/candidate_freeze.json').exists()
    assert not list((root/'gate24').rglob('config.json'))
    assert not list((root/'final96').rglob('config.json'))
    consumed=set()
    for path in root.rglob('config.json'):
        cfg=json.loads(path.read_text())
        assert cfg['seeds']==list(range(11000,11000+len(cfg['seeds'])))
        consumed.update(cfg['seeds'])
    target=root/'analysis/negative_decision_freeze.json'
    documents=['docs/V8_1_ASTRA_INVESTIGATION.md','docs/V8_1_MATHEMATICAL_DESIGN.md',
        'docs/V8_1_EXPERIMENT_PLAN.md','docs/V8_1_EXPERIMENT_LOG.md','docs/V8_1_CROSS_DATASET_REPORT.md']
    code=list(gate['source_code_hashes'])+['audit_v8_1.py','manage_v8_1.py','freeze_v8_1_decision.py',
        'tests/test_v8_1_kernel_choice.py','tests/test_v8_1_evidence.py']
    evidence={p.as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=target}
    value=dict(date='2026-09-15',status='rejected_at_gate8_no_confirmation_authorized',
        verdict='no deterministic candidate improves enough; stop architecture search',
        candidate='v8_1_kernel_choice',failed_gates=gate['failed'],baseline_files_verified=count,
        gate8_status=partial['status'],gate8_records=partial['rows'],missing_jobs=partial['missing_jobs'],
        consumed_development_seeds=sorted(consumed),confirmation_seeds=list(range(12000,12096)),
        confirmation_status='untouched',robust_work='blocked; none executed',
        failed_import_app='ap-mxp5vYcZ5Pu79W5SyBdCCP',
        smoke_app='ap-Uy2y7kGAnuMHYKR5wBvumZ',gate8_app=gate['app_id'],
        documents={p:digest(p) for p in documents},code_hashes={p:digest(p) for p in code},
        result_hashes=evidence)
    write_new(target,value)
    print('Negative decision preserved:',digest(target),'evidence files:',len(evidence))


if __name__=='__main__':main()
