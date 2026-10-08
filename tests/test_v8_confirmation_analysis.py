"""Cheap analysis-only regressions. No experimental fitting."""
import json
from pathlib import Path
import numpy as np
import pytest
from archive.research_scripts.verify_v8_confirmation import replay_votes, paired_stats, sha


def test_independent_cycle_replay():
    pred,votes,cycles=replay_votes([1,2,3],[[-1,1,-1],[0,0,0]])
    assert pred.tolist()==[1,3]
    assert votes.tolist()==[[1,1,1],[0,1,2]]
    assert cycles.tolist()==[1,0]


def test_replay_rejects_nonfinite_scores():
    with pytest.raises(AssertionError):
        replay_votes([0,1],[[float('nan')]])


def test_paired_metric_direction_and_degenerate_case():
    s=paired_stats([-.2,0,.1,.1],lower_better=True)
    assert (s['wins'],s['ties'],s['losses'])==(1,1,2)
    assert s['mean_delta']==pytest.approx(0)
    s=paired_stats(np.zeros(8))
    assert s['wilcoxon_p']==1 and s['standardized_paired_effect'] is None
    assert s['ci_low']==s['ci_high']==0


def test_confirmation_hashes_and_every_saved_vote():
    root=Path('results/v8/final96/v8-final96-20260909')
    hashes=Path('results/v8/analysis/confirmation_verification/artifact_hashes.json')
    if not hashes.exists():
        pytest.skip('Local final96 evidence not installed')
    for name,expected in json.loads(hashes.read_text()).items():
        assert sha(root/name)==expected,name
    count=0
    for path in (root/'checkpoints').glob('*.json'):
        job=json.loads(path.read_text())
        row=next(r for r in job['records'] if r['model']=='v8_c')
        trace=job['traces']['v8_c']
        pred,votes,cycles=replay_votes(json.loads(row['labels']),trace['margins_positive_second'])
        np.testing.assert_array_equal(pred,json.loads(row['prediction_vector']))
        np.testing.assert_array_equal(votes,trace['votes'])
        tied=(votes==votes.max(1)[:,None]).sum(1)>1
        assert np.all(cycles[tied]>0)
        count+=int(tied.sum())
    assert count==295


def test_v8_model_code_still_matches_preconfirmation_freeze():
    freeze=json.loads(Path('results/v8/analysis/deterministic_candidate_freeze.json').read_text())
    for name,expected in freeze['code_hashes'].items():
        assert sha(name)==expected,name
