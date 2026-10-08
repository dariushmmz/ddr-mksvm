"""Freeze the already-completed, verified deterministic V8-C result; no fitting."""
import argparse
import json
from pathlib import Path
import pandas as pd
from archive.research_scripts.verify_v8_confirmation import sha, save_json


def freeze(root):
    root=Path(root)
    analysis=Path('results/v8/analysis/confirmation_verification')
    cfg=json.loads((root/'config.json').read_text())
    pre=Path('results/v8/analysis/deterministic_candidate_freeze.json')
    candidate=json.loads(pre.read_text())
    audit=json.loads((analysis/'verification.json').read_text())
    qualification=json.loads((root/'qualification.json').read_text())
    assert audit['status']=='passed' and audit['records']==1344
    assert qualification['passes_prespecified_point_screen']
    assert qualification['primary_pass'] and qualification['secondary_pass']
    assert cfg['code_hashes']==candidate['code_hashes']
    for name,expected in cfg['code_hashes'].items():
        assert sha(name)==expected
    hashes=json.loads((analysis/'artifact_hashes.json').read_text())
    for name,expected in hashes.items():
        assert sha(root/name)==expected
    paired=pd.read_csv(analysis/'verified_paired_metrics.csv')
    assert paired.runtime_ratio.max()<=5
    guards={'parkinson':.01,'wine':.005,'iris':.005,'breast_cancer_diagnostic':.005}
    for ds,bound in guards.items():
        row=paired[(paired.dataset==ds)&(paired.metric=='test_error')].iloc[0]
        assert row.mean_delta<=bound
    payload=dict(status='confirmed_and_frozen',date='2026-09-09',variant='v8_c',
        verdict='V8-C qualifies as the new deterministic architecture',
        scope='Predeclared class-sensitive target and mean-error guard qualification; not universal accuracy superiority or formal noninferiority.',
        candidate_manifest=pre.as_posix(),candidate_manifest_sha256=sha(pre),
        architecture=candidate,code_hashes=cfg['code_hashes'],
        result_root=root.as_posix(),configuration=cfg,
        seed_registry=dict(development=list(range(9000,9024)),confirmation=cfg['seeds'],
            confirmation_status='consumed; cannot be reused as untouched confirmation'),
        source_result_hashes=hashes,
        analysis_hashes={p.as_posix():sha(p) for p in sorted(analysis.iterdir()) if p.is_file()},
        analysis_code_hashes={p:sha(p) for p in ['verify_v8_confirmation.py','freeze_v8_confirmation.py']},
        metrics=pd.read_csv(root/'summary.csv').to_dict('records'),
        qualification=qualification,
        vote_audit=dict(v8_c_ties=295,heart=294,wine=1,confirmed_voting_bug=False,v7_traces_available=False),
        limitations=['Heart total error increases 1.875 pp; rarest recall remains only 6.25%.',
            'Mammographic target not repaired.',
            'Wine point guard passes but paired interval upper bound exceeds +0.5 pp.',
            'Overlapping row holdouts do not establish patient-population or subject-level noninferiority.',
            'Standard weighted SVM with balanced selection is not a demonstrated novel theorem.'],
        reproduce_analysis=[
            'python run_v8_class_sensitive.py --freeze',
            f'python verify_v8_confirmation.py {root.as_posix()}',
            f'python analyze_v8_confirmation.py {root.as_posix()}',
            f'python freeze_v8_confirmation.py {root.as_posix()}',
            'python -m pytest -q'],
        training_policy='Completed final96 must not be rerun; source_snapshot and config preserve the historical Modal command. No robust or kernel experiment was run in this analysis continuation.')
    target=Path('results/v8/analysis/deterministic_v8_c_final_freeze.json')
    save_json(target,payload)
    print(f'Frozen {len(hashes)} result files; {target}; SHA256 {sha(target)}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();freeze(a.root)
