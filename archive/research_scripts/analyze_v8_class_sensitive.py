"""Read-only scientific validation, with new derived audit artifacts."""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from ddr_mksvm.experiments.class_sensitive_runner import digest, json_write, paired_tables

def derived_tables(root,frame):
    """Additional raw parameter and runtime tables, without any refitting."""
    rows=[];pairs=[];tails=[]
    for row in frame.itertuples():
        par=json.loads(row.selected_hyperparameters)
        rows.append(dict(dataset=row.dataset,model=row.model,seed=row.seed,
            alpha_rule=par['alpha_rule'],alpha=par['alpha'],C=par['C'],
            ratio=par.get('ratio',1.),selection_metric=par.get('selection_metric','error')))
        for pair in par.get('pairs',[]):
            pairs.append(dict(dataset=row.dataset,model=row.model,seed=row.seed,**pair))
    for (ds,model),g in frame.groupby(['dataset','model']):
        tails.append(dict(dataset=ds,model=model,mean_s=g.runtime_s.mean(),
            median_s=g.runtime_s.median(),p90_s=g.runtime_s.quantile(.9),
            p95_s=g.runtime_s.quantile(.95),maximum_s=g.runtime_s.max(),
            total_standalone_s=g.runtime_s.sum()))
    selected=pd.DataFrame(rows)
    outputs={'selected_hyperparameters.csv':selected,'pair_weights.csv':pd.DataFrame(pairs),
        'runtime_distribution.csv':pd.DataFrame(tails),
        'selection_frequencies.csv':selected.groupby(['dataset','model','alpha_rule','C','ratio']).size().rename('count').reset_index()}
    out=root/'derived';out.mkdir(exist_ok=True)
    for name,table in outputs.items():
        payload=table.to_csv(index=False).encode();path=out/name
        if path.exists():assert path.read_bytes()==payload,'derived artifact mismatch'
        else:path.write_bytes(payload)

def audit(root):
    root=Path(root);config=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest['sha256'].items():assert digest(root/name)==sha
    frame=pd.read_csv(root/'per_run.csv');count=0;fits=0;ties=0
    for path in (root/'checkpoints').glob('*.json'):
        job=json.loads(path.read_text());assert job['fingerprint']==manifest['fingerprint']
        split=job['split'];y=np.array(split['y_test'])
        assert not set(split['train_indices'])&set(split['test_indices'])
        for row in job['records']:
            pred=np.asarray(json.loads(row['prediction_vector']),dtype='<i8')
            assert hashlib.sha256(pred.tobytes()).hexdigest()==row['prediction_hash']
            assert int((pred!=y).sum())==row['errors']
            for kind in ('train','test'):
                assert hashlib.sha256(np.asarray(split[kind+'_indices'],dtype='<i8').tobytes()).hexdigest()==row[kind+'_indices_hash']
            count+=1
        for family,bank in job['banks'].items():
            for fold in bank['folds']:
                fi,va=fold['train_indices'],fold['validation_indices']
                assert not set(fi)&set(va)
                assert sorted(fi+va)==list(range(len(split['train_indices'])))
            for rec in bank['records']:
                for fold in rec['folds']:
                    for d in fold['solver_diagnostics']:
                        assert d['status']==0;fits+=1
                        if family in ('inverse','sqrt'):
                            assert abs(sum(d['counts'][c]*w for c,w in d['weights'].items())/sum(d['counts'].values())-1)<1e-12
        for trace in job['traces'].values():ties+=trace.get('vote_ties',0)
    assert count==len(frame)
    source=root/'source_snapshot';source.mkdir(exist_ok=True)
    for name,sha in config['code_hashes'].items():
        target=source/name
        if target.exists(): assert digest(target)==sha
        elif digest(name)==sha:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
        else:raise RuntimeError(f'Code changed before archive: {name}')
    pair,classes=paired_tables(frame)
    derived_tables(root,frame)
    print(pair[['dataset','model','metric','mean_delta','ci_low','ci_high','wins','ties','losses','runtime_ratio']].to_string(index=False))
    audit_record=dict(status='passed',records=count,weighted_bank_binary_fits=fits,vote_ties=ties,app_id=manifest['app_id'],fingerprint=manifest['fingerprint'])
    target=root/'audit.json'
    if target.exists():assert json.loads(target.read_text())==audit_record
    else:json_write(target,audit_record)
    print(audit_record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();audit(a.root)
