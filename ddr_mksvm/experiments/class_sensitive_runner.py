"""Local, checkpointed runner for the frozen class-sensitive architecture."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, shutil, subprocess, tarfile, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
from joblib import Parallel, delayed
from ddr_mksvm import v7_cross_dataset as v7
from ddr_mksvm.iris_research import reconstruct_authors_iris
from ddr_mksvm.v8_class_sensitive import VARIANTS, evaluate_variants

CODE = ['ddr_mksvm/v7_cross_dataset.py','ddr_mksvm/iris_research.py',
        'ddr_mksvm/v8_class_sensitive.py','ddr_mksvm/experiments/class_sensitive_runner.py']
FROZEN_DIRS = ['results/iris_experiments/iris-v7-rkhs-l2-final96-20260908',
              'results/v7_cross_dataset/v7-cross-final96-20260909',
              'results/v7_cross_dataset/analysis']

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_write(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)

def freeze_reference():
    path=Path('results/v8/analysis/v7_freeze.json');path.parent.mkdir(parents=True,exist_ok=True)
    # Preserve the historical manifest key while reading the byte-identical
    # cloud wrapper from its provenance archive.
    files={p:Path(p) for p in CODE[:2]}
    files['run_v7_cross_dataset.py']=Path('run_v7_cross_dataset.py')
    files['modal_v7_cross_dataset.py']=Path('archive/modal/modal_v7_cross_dataset.py')
    for root in FROZEN_DIRS:
        for item in sorted(Path(root).rglob('*')):
            if item.is_file():files[item.as_posix()]=item
    current={logical:digest(actual) for logical,actual in files.items()}
    if path.exists():
        assert json.loads(path.read_text())==current,'FROZEN V7 CHANGED'
    else: json_write(path,current)
    print(f'Frozen V7 verified: {len(files)} files')

def load(name):
    if name!='iris': return v7.load_dataset(name)
    raw=pd.read_csv('dataset/iris_multiclass.csv').to_numpy(float)
    values=reconstruct_authors_iris(raw)
    return values[:,:4],values[:,4].astype(int),dict(transform='none',rows=150,features=4,
        classes=[1,2,3],class_counts={'1':50,'2':50,'3':50},
        numeric_sha256=hashlib.sha256(values.astype('<f8').tobytes()).hexdigest(),
        csv_sha256=digest('dataset/iris_multiclass.csv'))

def paired_tables(frame):
    rows=[];classes=[]
    for (ds,model),g in frame.groupby(['dataset','model']):
        labels=json.loads(g.iloc[0].labels)
        cm=sum(np.asarray(json.loads(s),int) for s in g.confusion_matrix)
        for i,c in enumerate(labels):
            tp=int(cm[i,i]);precision=tp/cm[:,i].sum() if cm[:,i].sum() else 0.
            recall=tp/cm[i].sum() if cm[i].sum() else 0.
            classes.append(dict(dataset=ds,model=model,label=c,support=int(cm[i].sum()),
                precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,confusion_row=json.dumps(cm[i].tolist())))
        if model=='v7':continue
        ref=frame[(frame.dataset==ds)&(frame.model=='v7')].set_index('seed')
        cand=g.set_index('seed');assert set(ref.index)==set(cand.index)
        ref=ref.loc[cand.index]
        for col in ('train_indices_hash','test_indices_hash'):assert (cand[col]==ref[col]).all()
        for metric in ('test_error','balanced_accuracy','macro_f1'):
            d=(cand[metric]-ref[metric]).to_numpy();n=len(d);sd=np.std(d,ddof=1) if n>1 else 0.
            half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
            corrected=float(t.ppf(.975,n-1)*sd*np.sqrt(1/n+1/3)) if n>1 else None
            favorable=-d if metric=='test_error' else d
            nz=d[np.abs(d)>1e-14]
            rows.append(dict(dataset=ds,model=model,metric=metric,seeds=n,mean_delta=d.mean(),
                ci_low=d.mean()-half if half is not None else None,ci_high=d.mean()+half if half is not None else None,
                resampled_ci_low=d.mean()-corrected if corrected is not None else None,
                resampled_ci_high=d.mean()+corrected if corrected is not None else None,
                wins=int((favorable>1e-14).sum()),ties=int((np.abs(favorable)<=1e-14).sum()),losses=int((favorable< -1e-14).sum()),
                wilcoxon_p=float(wilcoxon(nz).pvalue) if len(nz) else 1.,
                standardized_paired_effect=float(d.mean()/sd) if sd else None,
                runtime_ratio=float(cand.runtime_s.sum()/ref.runtime_s.sum())))
    return pd.DataFrame(rows),pd.DataFrame(classes)

def run_job(name, seed, variants, cache, fingerprint, reuse_root=None):
    path=Path(cache)/f'{name}-{seed}.json'
    if path.exists():
        result=json.loads(path.read_text());assert result['fingerprint']==fingerprint
        assert result['variants']==variants
        return result
    source=Path(reuse_root)/'checkpoints'/f'{name}-{seed}.json' if reuse_root else None
    if source and source.exists():
        result=json.loads(source.read_text())
        assert set(variants)<=set(result['variants'])
        result['reused_from']={'path':str(source),'sha256':digest(source),'fingerprint':result['fingerprint'],'app_id':result['app_id']}
        result['records']=[r for r in result['records'] if r['model'] in variants]
        result['traces']={k:v for k,v in result['traces'].items() if k in variants}
        result.update(fingerprint=fingerprint,variants=variants,new_compute_wall_s=0.)
        json_write(path,result);print(f'reused {name} {seed}',flush=True)
        return result
    X,y,spec=load(name);result=evaluate_variants(name,X,y,spec,seed,variants)
    result.update(fingerprint=fingerprint,variants=variants,app_id='local')
    json_write(path,result)
    print(f'complete {name} {seed} {result["actual_job_wall_s"]:.2f}s',flush=True)
    return result

def run_stage(a):
    names=a.datasets.split(',');variants=a.variants.split(',')
    assert variants[0]=='v7' and set(variants)<=set(VARIANTS)
    seeds=list(range(a.seed_start,a.seed_start+a.n_seeds))
    out=Path(a.output_root)/a.run_name;out.mkdir(parents=True,exist_ok=True)
    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','scikit-learn','cvxpy','joblib')}
    hashes={p:digest(p) for p in CODE}
    cfg=dict(version='v8-class-sensitive-1',datasets=names,variants=variants,seeds=seeds,
        code_hashes=hashes,versions=versions,python=platform.python_version(),
        dataset_inventory={n:load(n)[2] for n in names},alpha_rules=list(v7.ALPHA_RULES),C_grid=list(v7.SVC_C_GRID),
        ratio_grid=[1,.5,2,.25,4],variants_definition=VARIANTS,
        outer='sorted stratified 75/25',inner='3fold stratified seed=20000+outer; equal fold mean; alpha,C,ratio order ties',
        resources=f'local CPU; {a.n_jobs} joblib worker(s); no GPU required',
        execution_backend='local',local_git=os.getenv('V8_LOCAL_GIT','unknown'))
    cfg=json.loads(json.dumps(cfg))  # canonical JSON types make resume comparison exact
    if a.reuse_root:
        prior=json.loads((Path(a.reuse_root)/'config.json').read_text())
        for key in ('versions','python','alpha_rules','C_grid','ratio_grid','variants_definition','outer','inner'):
            assert prior[key]==cfg[key],f'incompatible reuse {key}'
        for name in set(names)&set(prior['datasets']):assert prior['dataset_inventory'][name]==cfg['dataset_inventory'][name]
        for path in CODE[:3]:assert prior['code_hashes'][path]==hashes[path],f'scientific code changed {path}'
        assert set(variants)<=set(prior['variants'])
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg,'incompatible resume'
    else:json_write(out/'config.json',cfg)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())['status']=='complete':
        print('Already complete; no artifact overwritten');return
    for name,sha in hashes.items():
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert digest(target)==sha
        else:shutil.copyfile(name,target)
    cache=out/'checkpoints';cache.mkdir(exist_ok=True);start=time.perf_counter()
    json_write(out/'manifest.json',dict(status='running',fingerprint=fingerprint,app_id='local'))
    results=Parallel(n_jobs=a.n_jobs)(delayed(run_job)(n,s,variants,cache,fingerprint,a.reuse_root) for n in names for s in seeds)
    frame=pd.DataFrame([r for job in results for r in job['records']])
    pair,classes=paired_tables(frame)
    for filename,df in [('per_run.csv',frame),('summary.csv',v7.summarize(frame)),('paired_comparisons.csv',pair),('class_metrics.csv',classes)]:
        df.to_csv(out/filename,index=False)
    json_write(out/'splits.json',[r['split'] for r in results])
    json_write(out/'manifest.json',dict(status='complete',fingerprint=fingerprint,app_id='local',
        stage_wall_s=time.perf_counter()-start,actual_job_wall_s=sum(r['actual_job_wall_s'] for r in results),
        new_compute_wall_s=sum(r.get('new_compute_wall_s',r['actual_job_wall_s']) for r in results),
        rows=len(frame),sha256={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    files=[p for p in out.rglob('*') if p.is_file()]
    with tarfile.open(out/'artifacts.tar.gz','w:gz') as archive:
        for path in files:archive.add(path,arcname=path.relative_to(out).as_posix())
    print(v7.summarize(frame).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--datasets');p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--seed-start',type=int,default=9000);p.add_argument('--n-seeds',type=int,default=1)
    p.add_argument('--n-jobs',type=int,default=1,choices=range(1,5));p.add_argument('--run-name');p.add_argument('--output-root',default='results/v8');p.add_argument('--reuse-root');a=p.parse_args()
    if a.freeze:freeze_reference()
    else:run_stage(a)
