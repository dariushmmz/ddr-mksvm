"""Run immutable V7 versus corrected executable Legacy on matched datasets."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from ddr_mksvm.v7_cross_dataset import DATASETS,evaluate_seed,load_dataset,paired,summarize

def git_value(*args):
    try:return subprocess.check_output(["git",*args],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "unavailable"
def code_hash(paths):
    h=hashlib.sha256()
    for p in paths:h.update(Path(p).read_bytes())
    return h.hexdigest()
def atomic_csv(df,path):
    tmp=path.with_suffix(path.suffix+".tmp");df.to_csv(tmp,index=False);os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',default=','.join(DATASETS));p.add_argument('--n-seeds',type=int,default=1);p.add_argument('--seed-start',type=int,default=7000)
 p.add_argument('--n-jobs',type=int,default=1);p.add_argument('--run-name',required=True);p.add_argument('--output-root',default='results/v7_cross_dataset');a=p.parse_args()
 names=a.datasets.split(',');unknown=set(names)-set(DATASETS)
 if unknown:raise ValueError(f'unknown datasets {sorted(unknown)}')
 out=Path(a.output_root)/a.run_name
 if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
 out.mkdir(parents=True);codefiles=['ddr_mksvm/v7_cross_dataset.py','ddr_mksvm/iris_research.py','run_v7_cross_dataset.py']
 inventories={};loaded={}
 for name in names:
  X,y,inv=load_dataset(name);loaded[name]=(X,y,inv);inventories[name]=inv
 cfg={'run_name':a.run_name,'datasets':names,'n_seeds':a.n_seeds,'seed_start':a.seed_start,'seeds':list(range(a.seed_start,a.seed_start+a.n_seeds)),'n_jobs':a.n_jobs,
      'outer_protocol':'sorted stratified 75/25','v7_inner':'3-fold stratified; random_state=20000+outer_seed; mean error; alpha-rule then C order ties',
      'v7_alpha_rules':['paper','med:-2','med:-1','med:-0.5','med:0','med:0.5','med:1','med:2'],'v7_C_grid':[.1,1,10,100],
      'legacy_nu_grid':[.001,.005623413251903491,.03162277660168379,.1778279410038923,1.0],'legacy_threshold':'10000-point MATLAB-order first minimum',
      'git_commit':git_value('rev-parse','HEAD'),'git_dirty':bool(git_value('status','--porcelain')),'code_sha256':code_hash(codefiles),'dataset_inventory':inventories,
      'modal_resources':os.getenv('V7_MODAL_RESOURCES','local/unset'),'created_utc':datetime.now(timezone.utc).isoformat()}
 (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 records=[];diags=[];splits=[];start=time.time()
 for name in names:
  X,y,inv=loaded[name];dsout=out/name;dsout.mkdir()
  results=Parallel(n_jobs=a.n_jobs)(delayed(evaluate_seed)(name,X,y,inv,s) for s in cfg['seeds'])
  for rr,dd,ss in results:records.extend(rr);diags.extend(dd);ss['dataset']=name;splits.append(ss)
  atomic_csv(pd.DataFrame([r for r in records if r['dataset']==name]),dsout/'per_run.csv')
  atomic_csv(pd.DataFrame([d for d in diags if d.get('dataset')==name]),dsout/'solver_diagnostics.csv')
 frame=pd.DataFrame(records);summary=summarize(frame);pair=paired(frame)
 atomic_csv(frame,out/'per_run.csv');atomic_csv(pd.DataFrame(diags),out/'solver_diagnostics.csv');atomic_csv(summary,out/'summary.csv');atomic_csv(pair,out/'paired_comparisons.csv')
 (out/'splits.json').write_text(json.dumps({'splits':splits},indent=2)+'\n')
 manifest={'status':'complete','rows':len(frame),'diagnostics':len(diags),'elapsed_wall_s':time.time()-start,'finished_utc':datetime.now(timezone.utc).isoformat()}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(summary.to_string(index=False));print(pair.to_string(index=False))
if __name__=='__main__':main()
