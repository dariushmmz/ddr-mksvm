"""Mirror and analyze deterministic V8.1 evidence; never trains a model."""
import argparse
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd
from archive.research_scripts.v8_1_provenance import digest,write_new
from archive.research_scripts.verify_v8_confirmation import paired_stats,replay_votes


def sync(run_name):
    import modal
    volume=modal.Volume.from_name('ddr-mksvm-v8-1-results')
    raw=b''.join(volume.read_file(run_name+'/artifacts.tar.gz'))
    root=(Path('results/v8_1')/run_name).resolve();root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        for member in archive.getmembers():
            target=(root/member.name).resolve()
            assert target.is_relative_to(root) and member.isfile()
            payload=archive.extractfile(member).read();target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==payload
            else:target.write_bytes(payload)
    print('Mirrored',run_name,len(raw),'compressed bytes')


def sync_pruned(run_name):
    """Preserve partial remote evidence without fabricating a complete manifest."""
    import modal
    from modal.volume import FileEntryType
    from pathlib import PurePosixPath
    from concurrent.futures import ThreadPoolExecutor
    volume=modal.Volume.from_name('ddr-mksvm-v8-1-results')
    root=(Path('results/v8_1')/run_name).resolve();root.mkdir(parents=True,exist_ok=True)
    entries=[e for e in volume.iterdir(run_name,recursive=True) if e.type==FileEntryType.FILE]
    def copy(entry):
        relative=PurePosixPath(entry.path.lstrip('/')).relative_to(run_name)
        target=(root/str(relative)).resolve();assert target.is_relative_to(root)
        data=b''.join(volume.read_file(entry.path));assert len(data)==entry.size
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(copy,entries))
    assert not (root/'manifest.json').exists(),'Run completed: use normal analysis'
    jobs=[json.loads(p.read_text()) for p in (root/'checkpoints').glob('*.json')]
    cfg=json.loads((root/'config.json').read_text())
    present={(j['split']['dataset'],j['split']['seed']) for j in jobs}
    missing=sorted(set((d,s) for d in cfg['datasets'] for s in cfg['seeds'])-present)
    assert missing==[('blood_transfusion',11003)],missing
    fingerprints={j['fingerprint'] for j in jobs};assert len(fingerprints)==1
    write_new(root/'analysis/pruning_manifest.json',dict(status='pruned_after_complete_primary_gate_failure',
        app_id='ap-YyFhAPUf39GsX763SUIUPO',fingerprint=next(iter(fingerprints)),
        rows=sum(len(j['records']) for j in jobs),missing_jobs=missing,
        reason='Complete Mammographic gate8 failed magnitude and strict-win gates; stop remaining slow Blood fit.',
        actual_completed_job_wall_s=sum(j['actual_job_wall_s'] for j in jobs),
        new_completed_job_wall_s=sum(j['new_job_wall_s'] for j in jobs),
        censored_fit_runtime='unmeasured; excluded from completed-job sums, not zero cost',
        sha256={p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and 'analysis' not in p.relative_to(root).parts}))
    from ddr_mksvm.v7_cross_dataset import summarize
    frame=pd.DataFrame([r for j in jobs for r in j['records']])
    for name,table in [('per_run.csv',frame),('summary.csv',summarize(frame))]:
        payload=table.to_csv(index=False).encode();path=root/'analysis'/name
        if path.exists():assert path.read_bytes()==payload
        else:path.write_bytes(payload)
    print('Preserved partial gate:',len(jobs),'checkpoints; missing',missing)


def metrics_delta(d,lower=False,ratio=1/3):
    if len(d)>1:return paired_stats(d,lower,ratio)
    value=float(np.asarray(d)[0]);fav=-value if lower else value
    return dict(seeds=1,mean_delta=value,ci_low=None,ci_high=None,wilcoxon_p=None,
        standardized_paired_effect=None,wins=int(fav>1e-14),ties=int(abs(fav)<=1e-14),losses=int(fav < -1e-14))


def analyze(run_name):
    root=Path('results/v8_1')/run_name;cfg=json.loads((root/'config.json').read_text())
    partial=not (root/'manifest.json').exists()
    mf=json.loads((root/('analysis/pruning_manifest.json' if partial else 'manifest.json')).read_text())
    for name,h in mf['sha256'].items():assert digest(root/name)==h,name
    for name,h in cfg['code_hashes'].items():assert digest(name)==h,name
    rows=[];classes=[];selections=[];total=0;ratio={};diagnostics=[];confusions=[];kernel_evidence=[];disagreements=[]
    for path in sorted((root/'checkpoints').glob('*.json')):
        j=json.loads(path.read_text());assert j['fingerprint']==mf['fingerprint']
        split=j['split'];ds=split['dataset'];seed=split['seed'];y=np.asarray(split['y_test'])
        assert seed in cfg['seeds'] and ds in cfg['datasets']
        assert not set(split['train_indices'])&set(split['test_indices'])
        ratio[ds]=len(y)/len(split['train_indices'])
        for row in j['records']:
            labels=json.loads(row['labels']);pred=np.asarray(json.loads(row['prediction_vector']),dtype='<i8')
            import hashlib
            assert hashlib.sha256(pred.tobytes()).hexdigest()==row['prediction_hash']
            for kind in ('train','test'):
                ix=np.asarray(split[kind+'_indices'],dtype='<i8')
                assert hashlib.sha256(ix.tobytes()).hexdigest()==row[kind+'_indices_hash']
            cm=np.asarray([[int(((y==a)&(pred==b)).sum()) for b in labels] for a in labels])
            np.testing.assert_array_equal(cm,json.loads(row['confusion_matrix']))
            for i,a in enumerate(labels):
                for k,b in enumerate(labels):confusions.append(dict(dataset=ds,model=row['model'],true_label=a,predicted_label=b,count=int(cm[i,k])))
            assert int((pred!=y).sum())==row['errors']
            assert abs(float((pred!=y).mean())-row['test_error'])<1e-14
            assert abs(float((cm.diagonal()/cm.sum(1)).mean())-row['balanced_accuracy'])<1e-14
            assert abs(float((2*cm.diagonal()/(cm.sum(0)+cm.sum(1))).mean())-row['macro_f1'])<1e-14
            precision=np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(len(labels)),where=cm.sum(0)>0)
            for metric,values in [('recall',cm.diagonal()/cm.sum(1)),('precision',precision),
                ('f1',2*cm.diagonal()/(cm.sum(0)+cm.sum(1)))]:
                np.testing.assert_allclose([json.loads(row['class_'+metric])[str(c)] for c in labels],values,atol=1e-14,rtol=0)
            for label in labels:
                classes.append(dict(dataset=ds,seed=seed,model=row['model'],label=label,
                    **{m:json.loads(row['class_'+m])[str(label)] for m in ('recall','precision','f1')}))
            rows.append(row)
        by={r['model']:r for r in j['records']}
        for model in ('quadratic_only','v8_1_kernel_choice'):
            pred=np.asarray(json.loads(by[model]['prediction_vector']))
            ref=np.asarray(json.loads(by['v8_c']['prediction_vector']))
            disagreements.append(dict(dataset=ds,seed=seed,model=model,reference='v8_c',
                disagreements=int((pred!=ref).sum()),corrected_errors=int(((pred==y)&(ref!=y)).sum()),
                introduced_errors=int(((pred!=y)&(ref==y)).sum())))
        sel=j['kernel_selection'];source=sel['source_outer_prediction']
        assert by['v8_1_kernel_choice']['prediction_hash']==by[source]['prediction_hash']
        r=min(j['banks']['sqrt']['records'],key=lambda r:(r['losses']['balanced'],*r['order']))
        q=min(j['banks']['quadratic']['records'],key=lambda r:(r['losses']['balanced'],r['order']))
        expected='quadratic' if q['losses']['balanced']<r['losses']['balanced'] else 'rbf'
        assert sel['winner']==expected
        selections.append(dict(dataset=ds,seed=seed,**sel))
        kernel_evidence.append(dict(dataset=ds,seed=seed,**sel,
            rbf_inner_balanced_loss=r['losses']['balanced'],quadratic_inner_balanced_loss=q['losses']['balanced'],
            selected_rbf_parameters={k:v for k,v in r.items() if k!='folds'},quadratic_C=q['C']))
        for family,bank in j['banks'].items():
            for fold in bank['folds']:
                assert not set(fold['train_indices'])&set(fold['validation_indices'])
                assert sorted(fold['train_indices']+fold['validation_indices'])==list(range(len(split['train_indices'])))
            for rec in bank['records']:
                for fold in rec['folds']:
                    for diag in fold['solver_diagnostics']:
                        assert diag['status']==0;total+=1
                        diagnostics.append(dict(dataset=ds,seed=seed,family=family,C=rec['C'],
                            pair=str(diag['pair']),iterations=sum(diag['iterations']),status=diag['status']))
                        n=sum(diag['counts'].values())
                        assert abs(sum(diag['counts'][c]*w for c,w in diag['weights'].items())/n-1)<1e-12
        for model_name,trace in j['traces'].items():
            if 'margins_positive_second' in trace:
                p,v,_=replay_votes(json.loads(by[model_name]['labels']),trace['margins_positive_second'])
                np.testing.assert_array_equal(p,json.loads(by[model_name]['prediction_vector']))
                np.testing.assert_array_equal(v,trace['votes'])
                assert trace['vote_ties']==int(((v==v.max(1)[:,None]).sum(1)>1).sum())
            for d in trace['solver_diagnostics']:
                assert d.get('status',0)==0 and d.get('solver_status','ok')=='ok'
    frame=pd.DataFrame(rows);cr=pd.DataFrame(classes)
    assert len(frame)==mf['rows']
    assert len(frame)==(len(cfg['seeds'])*len(cfg['datasets'])-len(mf.get('missing_jobs',[])))*4
    assert not frame.duplicated(['dataset','seed','model']).any()
    paired=[];class_pair=[]
    for ds in cfg['datasets']:
        g=frame[frame.dataset==ds]
        for ref in ('v7','v8_c'):
            base=g[g.model==ref].set_index('seed').sort_index()
            for model in ('quadratic_only','v8_1_kernel_choice'):
                cand=g[g.model==model].set_index('seed').sort_index()
                for metric in ('test_error','balanced_accuracy','macro_f1','runtime_s'):
                    paired.append(dict(dataset=ds,reference=ref,model=model,metric=metric,
                        runtime_ratio=float(cand.runtime_s.sum()/base.runtime_s.sum()),
                        **metrics_delta(cand[metric]-base[metric],metric in ('test_error','runtime_s'),ratio[ds])))
                for label,c in cr[(cr.dataset==ds)&cr.model.isin([ref,model])].groupby('label'):
                    for metric in ('recall','precision','f1'):
                        p=c.pivot(index='seed',columns='model',values=metric)
                        class_pair.append(dict(dataset=ds,reference=ref,model=model,label=int(label),metric=metric,
                            **metrics_delta(p[model]-p[ref],False,ratio[ds])))
    output=root/'analysis';output.mkdir(exist_ok=True)
    for name,table in [('paired_metrics.csv',pd.DataFrame(paired)),('paired_class_metrics.csv',pd.DataFrame(class_pair)),
        ('kernel_selections.csv',pd.DataFrame(selections)),('inner_solver_diagnostics.csv',pd.DataFrame(diagnostics)),
        ('kernel_evidence.csv',pd.DataFrame(kernel_evidence)),
        ('prediction_disagreements.csv',pd.DataFrame(disagreements)),
        ('pooled_confusions.csv',pd.DataFrame(confusions).groupby(['dataset','model','true_label','predicted_label'],as_index=False)['count'].sum())]:
        payload=table.to_csv(index=False).encode();p=output/name
        if p.exists():assert p.read_bytes()==payload
        else:p.write_bytes(payload)
    audit=dict(status='passed',records=len(frame),inner_weighted_binary_diagnostics=total,source_fingerprint=mf['fingerprint'])
    write_new(output/'audit.json',audit)
    def mean(ds,model,metric):return float(frame[(frame.dataset==ds)&(frame.model==model)][metric].mean())
    def delta(ds,metric,ref='v8_c'):return mean(ds,'v8_1_kernel_choice',metric)-mean(ds,ref,metric)
    def recall(ds,model,label):return float(cr[(cr.dataset==ds)&(cr.model==model)&(cr.label==label)].recall.mean())
    checks={}
    if partial:checks['complete_expected_jobs']=False
    if cfg['stage']!='smoke':
        n=len(cfg['seeds']);mam='mammographicmass_binary';blood='blood_transfusion';heart='heart_disease'
        primary=next(r for r in paired if r['dataset']==mam and r['reference']=='v8_c' and r['model']=='v8_1_kernel_choice' and r['metric']=='test_error')
        checks['mammographic_error_magnitude']=delta(mam,'test_error')<=-.005
        checks['mammographic_strict_wins']=primary['wins']>=(5 if n==8 else n//2)
        if n>=24:checks['mammographic_positive_evidence']=primary['ci_high']<0
        checks['mammographic_class1_guard']=recall(mam,'v8_1_kernel_choice',1)-recall(mam,'v8_c',1)>=-.01
        checks['blood_BA_vs_v7']=delta(blood,'balanced_accuracy','v7')>=(.03 if n==8 else .04)
        checks['blood_minority_vs_v7']=recall(blood,'v8_1_kernel_choice',1)>recall(blood,'v7',1)
        checks['blood_BA_retention']=delta(blood,'balanced_accuracy')>=-.01
        checks['blood_recall_retention']=recall(blood,'v8_1_kernel_choice',1)-recall(blood,'v8_c',1)>=-.03
        checks['blood_error_guard']=delta(blood,'test_error')<=.005
        checks['heart_error_guard']=delta(heart,'test_error')<=.005
        for metric in ('balanced_accuracy','macro_f1'):checks['heart_'+metric+'_guard']=delta(heart,metric)>=-.005
        for ds in ('parkinson','wine','iris','breast_cancer_diagnostic'):
            if ds not in cfg['datasets']:continue
            bound=.01 if ds=='parkinson' else .005
            for ref in ('v7','v8_c'):checks[ds+'_error_vs_'+ref]=delta(ds,'test_error',ref)<=bound
        for ds in cfg['datasets']:
            checks[ds+'_runtime']=mean(ds,'v8_1_kernel_choice','runtime_s')/mean(ds,'v8_c','runtime_s')<=1.5
            for label in cr[cr.dataset==ds].label.unique():
                checks[f'{ds}_class{label}_no_collapse']=not (recall(ds,'v8_c',label)>0 and recall(ds,'v8_1_kernel_choice',label)==0)
    gate=dict(stage=cfg['stage'],candidate='v8_1_kernel_choice',passes=bool(all(checks.values())),checks=checks,
        failed=[k for k,v in checks.items() if not v],next_stage={'smoke':'gate8','gate8':'gate24','gate24':'final96','final96':None}[cfg['stage']],
        source_code_hashes=cfg['code_hashes'],fingerprint=mf['fingerprint'],app_id=mf['app_id'])
    write_new(output/'gate_decision.json',gate)
    print(json.dumps(gate,indent=2))
    print(pd.read_csv(root/('analysis/summary.csv' if partial else 'summary.csv'))[['dataset','model','mean_error','balanced_accuracy','macro_f1','mean_runtime_s']].to_string(index=False))
    print(pd.DataFrame(selections).groupby(['dataset','winner']).size().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sync','sync-pruned','analyze']);p.add_argument('run_name');a=p.parse_args()
    {'sync':sync,'sync-pruned':sync_pruned,'analyze':analyze}[a.action](a.run_name)
