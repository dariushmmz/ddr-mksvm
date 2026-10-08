"""Strict deterministic V8.1 staged runner. Real fitting requires Modal."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from run_v8_class_sensitive import load, paired_tables
from ddr_mksvm.v7_cross_dataset import summarize
from ddr_mksvm.v8_1_kernel_choice import evaluate
from v8_1_provenance import digest,write_new

CODE=['ddr_mksvm/v7_cross_dataset.py','ddr_mksvm/iris_research.py','ddr_mksvm/v8_class_sensitive.py',
      'run_v8_class_sensitive.py','modal_v8_class_sensitive.py','ddr_mksvm/v8_1_kernel_choice.py',
      'run_v8_1.py','modal_v8_1.py','v8_1_provenance.py']
MODELS=['v7','v8_c','quadratic_only','v8_1_kernel_choice']
TARGETS=['blood_transfusion','mammographicmass_binary','heart_disease','parkinson','wine']


def job(name,seed,out,fingerprint,reuse):
    path=out/'checkpoints'/f'{name}-{seed}.json'
    if path.exists():
        value=json.loads(path.read_text());assert value['fingerprint']==fingerprint
        return value
    source=reuse/'checkpoints'/path.name if reuse else None
    if source and source.exists():
        value=json.loads(source.read_text());value['reused_from']=dict(path=str(source),sha256=digest(source),fingerprint=value['fingerprint'])
        value['fingerprint']=fingerprint;value['new_job_wall_s']=0.
    else:
        X,y,spec=load(name);value=evaluate(name,X,y,spec,seed)
        value.update(fingerprint=fingerprint,app_id=os.environ['MODAL_APP_ID'])
        value['new_job_wall_s']=value['actual_job_wall_s']
    write_new(path,value)
    print(f'Complete {name} {seed}; {value["new_job_wall_s"]:.3f}s new work; choice={value["kernel_selection"]["winner"]}',flush=True)
    return value


def analysis_tables(frame):
    outputs=[];class_rows=None
    for reference in ('v7','v8_c'):
        g=frame.copy()
        if reference=='v8_c':
            g=g[g.model!='v7'].copy();g.loc[g.model=='v8_c','model']='v7'
        pair,classes=paired_tables(g);pair['reference']=reference;outputs.append(pair)
        if reference=='v7':class_rows=classes
    return pd.concat(outputs,ignore_index=True),class_rows


def main(a):
    if not os.getenv('V8_1_MODAL_RUN'):raise RuntimeError('Real experiments require Modal')
    design=json.loads(Path('results/v8_1/analysis/design_freeze.json').read_text())
    for path,h in design['document_hashes'].items():assert digest(path)==h,path
    for path,h in design['baseline_code_hashes'].items():assert digest(path)==h,path
    n={'smoke':1,'gate8':8,'gate24':24,'final96':96}[a.stage]
    datasets=TARGETS+(['iris','breast_cancer_diagnostic'] if n>=24 else [])
    seeds=list(range(12000,12096)) if n==96 else list(range(11000,11000+n))
    if a.stage in ('gate24','final96'):
        gate=Path(a.gate_authorization)
        authorization=json.loads(gate.read_text())
        assert authorization['passes'] and authorization['next_stage']==a.stage
        assert authorization['candidate']=='v8_1_kernel_choice'
        assert authorization['source_code_hashes']=={p:digest(p) for p in CODE}
    if n==96:
        freeze=json.loads((Path(a.output_root)/'analysis/candidate_freeze.json').read_text())
        assert freeze['code_hashes']=={p:digest(p) for p in CODE}
        assert freeze['seeds']==seeds
    out=Path(a.output_root)/a.run_name
    out.mkdir(parents=True,exist_ok=True)
    versions={p:importlib.metadata.version(p) for p in design['versions']}
    assert versions==design['versions'],(versions,design['versions'])
    inventory={d:load(d)[2] for d in datasets}
    for ds in datasets:assert inventory[ds]==design['dataset_inventory'][ds],ds
    cfg=dict(stage=a.stage,datasets=datasets,seeds=seeds,models=MODELS,code_hashes={p:digest(p) for p in CODE},
        design_sha256=digest('results/v8_1/analysis/design_freeze.json'),design=design,
        versions=versions,dataset_inventory=inventory,resources='4CPU 4096MiB max4 workers no GPU',
        local_git=os.getenv('V8_1_LOCAL_GIT','unknown'))
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    write_new(out/'config.json',cfg)
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text())['status']=='complete'
        print('Already complete; no refit/overwrite');return
    reuse=Path(a.reuse_root) if a.reuse_root else None
    if reuse:
        prior=json.loads((reuse/'config.json').read_text())
        for key in ('models','code_hashes','design_sha256','versions'):assert prior[key]==cfg[key],key
        for ds in set(prior['datasets'])&set(datasets):assert prior['dataset_inventory'][ds]==inventory[ds]
    for path in CODE+list(design['document_hashes'])+['results/v8_1/analysis/design_freeze.json']:
        target=out/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert digest(target)==digest(path)
        else:shutil.copyfile(path,target)
    (out/'checkpoints').mkdir(exist_ok=True)
    started=time.perf_counter()
    jobs=Parallel(n_jobs=4)(delayed(job)(ds,s,out,fingerprint,reuse) for ds in datasets for s in seeds)
    frame=pd.DataFrame([r for j in jobs for r in j['records']]);paired,classes=analysis_tables(frame)
    for name,df in [('per_run.csv',frame),('summary.csv',summarize(frame)),('paired_comparisons.csv',paired),('class_metrics.csv',classes)]:
        path=out/name;payload=df.to_csv(index=False).encode()
        if path.exists():assert path.read_bytes()==payload
        else:path.write_bytes(payload)
    write_new(out/'splits.json',[j['split'] for j in jobs])
    write_new(out/'manifest.json',dict(status='complete',app_id=os.environ['MODAL_APP_ID'],fingerprint=fingerprint,
        rows=len(frame),stage_wall_s=time.perf_counter()-started,actual_job_wall_s=sum(j['actual_job_wall_s'] for j in jobs),
        new_job_wall_s=sum(j['new_job_wall_s'] for j in jobs),
        sha256={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    files=[p for p in out.rglob('*') if p.is_file()]
    with tarfile.open(out/'artifacts.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=p.relative_to(out).as_posix())
    print(summarize(frame).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','gate8','gate24','final96'],required=True)
    p.add_argument('--run-name',required=True);p.add_argument('--output-root',default='results/v8_1')
    p.add_argument('--reuse-root');p.add_argument('--gate-authorization');main(p.parse_args())
