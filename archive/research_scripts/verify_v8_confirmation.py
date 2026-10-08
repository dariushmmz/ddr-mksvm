"""Independent replay of saved confirmation evidence. Never fits a model.

Writes new analysis artifacts only; frozen experiment files are read-only.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t, wilcoxon


HISTORICAL_PATHS = {
    "modal_v7_cross_dataset.py": Path("archive/modal/modal_v7_cross_dataset.py"),
    "modal_v8_class_sensitive.py": Path("archive/modal/modal_v8_class_sensitive.py"),
    "run_v8_class_sensitive.py": Path("archive/frozen_sources/run_v8_class_sensitive.py"),
}


def sha(path):
    logical = str(path).replace("\\", "/")
    actual = HISTORICAL_PATHS.get(logical, Path(path))
    return hashlib.sha256(actual.read_bytes()).hexdigest()


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == payload, f'Existing analysis differs: {path}'
    else:
        path.write_bytes(payload)


def save_json(path, value):
    save(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def replay_votes(classes, margins):
    """Deliberately independent scalar replay, positive/zero => second class."""
    margins = np.asarray(margins, float)
    assert np.isfinite(margins).all()
    pairs = list(combinations(range(len(classes)), 2))
    assert margins.ndim == 2 and margins.shape[1] == len(pairs)
    votes = np.zeros((len(margins), len(classes)), dtype=int)
    cycles = []
    for row, scores in enumerate(margins):
        beats = np.zeros((len(classes), len(classes)), bool)
        for score, (a, b) in zip(scores, pairs):
            winner, loser = (b, a) if score >= 0 else (a, b)
            votes[row, winner] += 1
            beats[winner, loser] = True
        cycles.append(sum(bool((beats[a,b] and beats[b,c] and beats[c,a]) or
                               (beats[a,c] and beats[c,b] and beats[b,a]))
                          for a,b,c in combinations(range(len(classes)), 3)))
    prediction = np.asarray([classes[list(v).index(max(v))] for v in votes])
    return prediction, votes, np.asarray(cycles)


def paired_stats(delta, lower_better=False, test_train_ratio=1/3):
    delta = np.asarray(delta, float)
    n = len(delta)
    mean, sd = float(delta.mean()), float(delta.std(ddof=1))
    half = float(t.ppf(.975, n-1) * sd / np.sqrt(n))
    corrected = float(t.ppf(.975, n-1) * sd * np.sqrt(1/n+test_train_ratio))
    favorable = -delta if lower_better else delta
    nz = delta[np.abs(delta) > 1e-14]
    # Machine-precision rank ties can otherwise depend on CSV/library versions.
    rounded = np.round(nz, 12)
    return dict(seeds=n, mean_delta=mean, ci_low=mean-half, ci_high=mean+half,
        resampled_ci_low=mean-corrected, resampled_ci_high=mean+corrected,
        wins=int((favorable > 1e-14).sum()), ties=int((np.abs(favorable) <= 1e-14).sum()),
        losses=int((favorable < -1e-14).sum()),
        wilcoxon_p=float(wilcoxon(nz).pvalue) if len(nz) else 1.,
        wilcoxon_rounded12_p=float(wilcoxon(rounded).pvalue) if len(nz) else 1.,
        standardized_paired_effect=mean/sd if sd else None,
        test_train_ratio=test_train_ratio)


def verify(root, out):
    root, out = Path(root), Path(out)
    cfg = json.loads((root/'config.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert cfg['seeds'] == list(range(10000,10096))
    assert cfg['variants'] == ['v7','v8_c']
    files = {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    for name, expected in manifest['sha256'].items():
        assert files[name] == expected
    for name, expected in cfg['code_hashes'].items():
        assert sha(name) == expected
        assert files['source_snapshot/'+name] == expected
    for name, expected in json.loads(Path('results/v8/analysis/v7_freeze.json').read_text()).items():
        assert sha(name) == expected
    records, class_rows, tied_rows, votes_summary, ratios, split_map = [], [], [], [], {}, {}
    v7_diagnostics = weighted_diagnostics = 0
    for ds in cfg['datasets']:
        tie_count = exact_zeros = near_zeros = predictions = 0
        for seed in cfg['seeds']:
            job = json.loads((root/'checkpoints'/f'{ds}-{seed}.json').read_text())
            assert job['fingerprint'] == manifest['fingerprint']
            split = job['split']
            assert (split['dataset'], split['seed']) == (ds,seed)
            split_map[ds,seed] = split
            labels = cfg['dataset_inventory'][ds]['classes']
            y = np.asarray(split['y_test'])
            ratios[ds] = len(y)/len(split['train_indices'])
            assert not set(split['train_indices']) & set(split['test_indices'])
            assert len(job['records']) == 2
            for record in job['records']:
                assert (record['dataset'],record['seed']) == (ds,seed)
                pred = np.asarray(json.loads(record['prediction_vector']), dtype='<i8')
                assert hashlib.sha256(pred.tobytes()).hexdigest() == record['prediction_hash']
                cm = np.asarray([[np.sum((y==a)&(pred==b)) for b in labels] for a in labels])
                np.testing.assert_array_equal(cm,json.loads(record['confusion_matrix']))
                support, predicted = cm.sum(1),cm.sum(0)
                recall = cm.diagonal()/support
                precision = np.divide(cm.diagonal(),predicted,out=np.zeros(len(labels)),where=predicted>0)
                f1 = 2*cm.diagonal()/(support+predicted)
                error = float((pred != y).mean())
                assert int((pred!=y).sum()) == record['errors']
                for key,value in dict(test_error=error,accuracy=1-error,
                    balanced_accuracy=recall.mean(),macro_f1=f1.mean()).items():
                    assert abs(record[key]-value) < 2e-15,(ds,seed,key)
                for i,label in enumerate(labels):
                    for key,values in [('recall',recall),('precision',precision),('f1',f1),('support',support)]:
                        assert abs(json.loads(record['class_'+key])[str(label)]-values[i]) < 2e-15
                    class_rows.append(dict(dataset=ds,seed=seed,model=record['model'],label=label,
                        recall=recall[i],precision=precision[i],f1=f1[i],support=int(support[i])))
                records.append(record)
            for d in job['traces']['v7']['solver_diagnostics']:
                assert d['solver_status'] == 'ok'
                v7_diagnostics += 1
            for bank in job['banks'].values():
                for rec in bank['records']:
                    for fold in rec['folds']:
                        for d in fold['solver_diagnostics']:
                            assert d['status'] == 0
                            weighted_diagnostics += 1
            trace = job['traces']['v8_c']
            for d in trace['solver_diagnostics']:
                assert d['status'] == 0
                weighted_diagnostics += 1
            margins = np.asarray(trace['margins_positive_second'])
            pred, votes, cycles = replay_votes(labels,margins)
            np.testing.assert_array_equal(votes,trace['votes'])
            saved = next(r for r in job['records'] if r['model']=='v8_c')
            np.testing.assert_array_equal(pred,json.loads(saved['prediction_vector']))
            ties = np.flatnonzero((votes==votes.max(1)[:,None]).sum(1)>1)
            assert len(ties) == trace['vote_ties']
            tie_count += len(ties)
            predictions += len(y)
            exact_zeros += int((margins==0).sum())
            near_zeros += int((np.abs(margins)<=1e-9).sum())
            for i in ties:
                tied_labels = np.asarray(labels)[votes[i]==votes[i].max()].tolist()
                assert pred[i] == min(tied_labels) and cycles[i]>0
                tied_rows.append(dict(dataset=ds,seed=seed,sample_index=split['test_indices'][i],
                    truth=int(y[i]),prediction=int(pred[i]),correct=bool(y[i]==pred[i]),
                    truth_in_tie=bool(y[i] in tied_labels),tied_labels=json.dumps(tied_labels),
                    votes=json.dumps(votes[i].tolist()),cycles=int(cycles[i]),
                    minimum_absolute_pair_margin=float(np.min(np.abs(margins[i])))))
        votes_summary.append(dict(dataset=ds,v8_c_predictions=predictions,ties=tie_count,
            exact_zero_pair_margins=exact_zeros,near_zero_pair_margins_at_1e_9=near_zeros))
    assert len(split_map)==672 and len(records)==1344
    assert len(list((root/'checkpoints').glob('*.json')))==672
    saved_splits = json.loads((root/'splits.json').read_text())
    assert len(saved_splits)==672
    for split in saved_splits:
        assert split == split_map[split['dataset'],split['seed']]
    frame = pd.DataFrame(records).sort_values(['dataset','model','seed']).reset_index(drop=True)
    csv = pd.read_csv(root/'per_run.csv',float_precision='round_trip').sort_values(['dataset','model','seed']).reset_index(drop=True)
    for col in frame:
        if pd.api.types.is_numeric_dtype(frame[col]):
            np.testing.assert_allclose(frame[col],csv[col],rtol=1e-14,atol=1e-15,equal_nan=True)
        else:
            assert frame[col].tolist()==csv[col].tolist(),col
    pairs = pd.read_csv(root/'paired_comparisons.csv',float_precision='round_trip')
    verified, numerical_notes = [], []
    for ds in cfg['datasets']:
        ref = frame[(frame.dataset==ds)&(frame.model=='v7')].set_index('seed').sort_index()
        cand = frame[(frame.dataset==ds)&(frame.model=='v8_c')].set_index('seed').sort_index()
        for metric in ('test_error','balanced_accuracy','macro_f1','runtime_s'):
            delta = cand[metric].to_numpy()-ref[metric].to_numpy()
            s = paired_stats(delta,metric in ('test_error','runtime_s'),ratios[ds])
            ratio = float(cand.runtime_s.sum()/ref.runtime_s.sum())
            verified.append(dict(dataset=ds,metric=metric,runtime_ratio=ratio,**s))
            if metric != 'runtime_s':
                original = pairs[(pairs.dataset==ds)&(pairs.metric==metric)].iloc[0]
                for key in ('mean_delta','ci_low','ci_high','wins','ties','losses','standardized_paired_effect'):
                    if s[key] is None:
                        assert pd.isna(original[key])
                    else:
                        assert abs(original[key]-s[key])<1e-12,(ds,metric,key)
                assert abs(original.runtime_ratio-ratio)<1e-12
                if abs(original.wilcoxon_p-s['wilcoxon_p'])>1e-12:
                    numerical_notes.append(dict(dataset=ds,metric=metric,saved_p=float(original.wilcoxon_p),recomputed_p=s['wilcoxon_p']))
    classes = pd.DataFrame(class_rows)
    class_pairs = []
    for (ds,label),group in classes.groupby(['dataset','label']):
        for metric in ('recall','precision','f1'):
            p = group.pivot(index='seed',columns='model',values=metric).sort_index()
            class_pairs.append(dict(dataset=ds,label=int(label),metric=metric,
                v7_mean=float(p.v7.mean()),v8_c_mean=float(p.v8_c.mean()),
                **paired_stats(p.v8_c-p.v7,False,ratios[ds])))
    pooled = pd.read_csv(root/'class_metrics.csv',float_precision='round_trip')
    for (ds,model),g in frame.groupby(['dataset','model']):
        cm = sum(np.asarray(json.loads(s)) for s in g.confusion_matrix)
        labels = json.loads(g.iloc[0].labels)
        for i,label in enumerate(labels):
            row = pooled[(pooled.dataset==ds)&(pooled.model==model)&(pooled.label==label)].iloc[0]
            assert row.support==cm[i].sum()
            np.testing.assert_array_equal(json.loads(row.confusion_row),cm[i])
            np.testing.assert_allclose([row.recall,row.precision,row.f1],
                [cm[i,i]/cm[i].sum(),cm[i,i]/cm[:,i].sum() if cm[:,i].sum() else 0,
                 2*cm[i,i]/(cm[i].sum()+cm[:,i].sum())],atol=2e-15)
    summary = pd.read_csv(root/'summary.csv',float_precision='round_trip')
    for row in summary.itertuples():
        g = frame[(frame.dataset==row.dataset)&(frame.model==row.model)]
        assert row.errors==g.errors.sum() and row.predictions==g.n_test.sum()
        for output, source in [('mean_error','test_error'),('balanced_accuracy','balanced_accuracy'),
            ('macro_f1','macro_f1'),('mean_runtime_s','runtime_s')]:
            assert abs(getattr(row,output)-g[source].mean())<1e-12
    ties = pd.DataFrame(tied_rows)
    audit = dict(status='passed',fitting_performed=False,records=1344,checkpoints=672,
        files_read_and_hashed=len(files),v7_successful_estimator_diagnostics=v7_diagnostics,
        v8_successful_binary_fit_diagnostics=weighted_diagnostics,
        v8_vote_summary=votes_summary,ties=int(len(ties)),ties_correct=int(ties.correct.sum()),
        ties_truth_in_tied_set=int(ties.truth_in_tie.sum()),
        minimum_absolute_pair_margin_on_ties=float(ties.minimum_absolute_pair_margin.min()),
        every_tie_contains_a_directed_cycle=True,v7_vote_traces_available=False,
        wilcoxon_environment_differences=numerical_notes,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__),
        interpretation='Conditional matched-split statistics; not independent-patient population inference.')
    for name,table in [('verified_paired_metrics.csv',pd.DataFrame(verified)),
        ('per_seed_class_metrics.csv',classes),('paired_class_metrics.csv',pd.DataFrame(class_pairs)),
        ('vote_ties.csv',ties),('vote_summary.csv',pd.DataFrame(votes_summary))]:
        save(out/name,table.to_csv(index=False).encode())
    save_json(out/'artifact_hashes.json',files)
    save_json(out/'verification.json',audit)
    print(json.dumps(audit,indent=2))
    print(pd.DataFrame(class_pairs).query("metric == 'recall'").to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--output',default='results/v8/analysis/confirmation_verification')
    a=p.parse_args()
    verify(a.root,a.output)
