"""Checkpointed smoke/full runner for the isolated Robust V8-C model."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, math, os, platform, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ddr_mksvm import v7_cross_dataset as v7
from ddr_mksvm.robust_v8_c import select_and_fit
from archive.research_scripts.run_robust_matlab_parity import atomic_csv, atomic_json, array_hash, digest, load

CODE_FILES=["ddr_mksvm/v7_cross_dataset.py","ddr_mksvm/v8_class_sensitive.py","ddr_mksvm/robust_v8_c.py",
            "run_robust_v8_c.py","modal_robust_v8_c.py","analyze_robust_v8_c.py"]

def record(name,seed,model,p,rho,y,pred,runtime,params,trace,tr,te):
    labels=sorted(np.unique(y).tolist());pr,re,cf1,sup=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    diags=trace.get("solver_diagnostics",[])
    p_value="inf" if isinstance(p,(float,int)) and math.isinf(p) else p
    return {"dataset":name,"seed":seed,"model":model,"p":p_value,"rho":rho,"q":"RKHS-L2",
      "kernel":"Gaussian RBF","kernel_parameters":json.dumps({"alpha":params.get("alpha"),"alpha_rule":params.get("alpha_rule")}),
      "preprocessing":json.dumps(params.get("preprocessing",{}),sort_keys=True),"solver":"libsvm" if rho==0 else "CVXPY/CLARABEL",
      "solver_status":"ok" if rho==0 else "+".join(sorted(set(d["solver_status"] for d in diags))),
      "objective_value":None if rho==0 else float(sum(d["objective_value"] for d in diags)),"runtime_s":runtime,
      "selected_hyperparameters":json.dumps(params,sort_keys=True),"threshold_intercept":"pair-specific",
      "uncertainty_radius":0. if rho==0 else float(max(d["max_delta"] for d in diags)),
      "robust_penalty_terms":json.dumps([] if rho==0 else [{"pair":d["pair"],"regularization":d["regularization_term"],"weighted_slack":d["weighted_slack_term"]} for d in diags]),
      "constraint_max_violation":0. if rho==0 else float(max(d["max_constraint_violation"] for d in diags)),
      "solver_iterations":None if rho==0 else int(sum(d["solver_iterations"] for d in diags)),
      "test_error":float(np.mean(pred!=y)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "per_class_recall":json.dumps(dict(zip(map(str,labels),map(float,re))),sort_keys=True),
      "class_precision":json.dumps(dict(zip(map(str,labels),map(float,pr))),sort_keys=True),
      "class_f1":json.dumps(dict(zip(map(str,labels),map(float,cf1))),sort_keys=True),
      "class_support":json.dumps(dict(zip(map(str,labels),map(int,sup))),sort_keys=True),
      "confusion_matrix":json.dumps(confusion_matrix(y,pred,labels=labels).tolist()),
      "predictions":json.dumps(np.asarray(pred,int).tolist()),"prediction_hash":array_hash(pred,"<i8"),
      "true_labels":json.dumps(np.asarray(y,int).tolist()),"true_label_hash":array_hash(y,"<i8"),
      "train_indices_hash":array_hash(tr,"<i8"),"test_indices_hash":array_hash(te,"<i8"),
      "split_hash":hashlib.sha256((array_hash(tr,"<i8")+array_hash(te,"<i8")).encode()).hexdigest(),
      "vote_ties":trace.get("vote_ties",0),"reduction":params.get("reduction")}

def job(name,seed,model,rho,p,fingerprint,path):
    if path.exists():
        value=json.loads(path.read_text());
        if value["fingerprint"]!=fingerprint:raise RuntimeError(f"incompatible {path}")
        return value
    X,y,spec=load(name);tr,te=v7.split_indices(y,seed)
    started=time.perf_counter();pred,params,trace,runtime=select_and_fit(X[tr],y[tr],X[te],spec.get("transform","none"),seed,rho,p)
    rec=record(name,seed,model,p,rho,y[te],pred,runtime,params,trace,tr,te)
    if rho>0:
        all_diags=list(trace.get("solver_diagnostics",[]))
        for candidate in trace.get("bank",[]):
            for fold in candidate.get("folds",[]):all_diags.extend(fold.get("solver_diagnostics",[]))
        rec["solver_status"]="+".join(sorted(set(d["solver_status"] for d in all_diags)))
        rec["solver_iterations"]=int(sum(d["solver_iterations"] for d in all_diags))
    value={"fingerprint":fingerprint,"record":rec,"dataset_inventory":spec,"split":{"dataset":name,"seed":seed,"train_indices":tr.tolist(),"test_indices":te.tolist()},
           "trace":trace,"app_id":os.getenv("MODAL_APP_ID","local"),"job_wall_s":time.perf_counter()-started}
    atomic_json(path,value);print(f"complete {name} {seed} {model} {value['job_wall_s']:.2f}s",flush=True);return value

def aggregate(root:Path,output_dir:Path|None=None):
    values=[json.loads(p.read_text()) for p in sorted((root/"checkpoints").glob("*.json"))]
    frame=pd.DataFrame([v["record"] for v in values]);splits=pd.DataFrame([v["split"] for v in values])
    if frame.empty:summary=paired=classes=diags=rhos=pd.DataFrame()
    else:
      summary=frame.groupby(["dataset","model","p","rho"],dropna=False).agg(seeds=("seed","nunique"),error=("test_error","mean"),balanced_acc=("balanced_accuracy","mean"),macro_f1=("macro_f1","mean"),runtime=("runtime_s","mean"),solver_status=("solver_status",lambda x:"+".join(sorted(set(x))))).reset_index()
      rows=[];classrows=[];diagrows=[]
      for _,r in frame.iterrows():
        for c,v in json.loads(r.per_class_recall).items():classrows.append({"dataset":r.dataset,"seed":r.seed,"model":r.model,"class":c,"recall":v})
      for (ds,seed),g in frame.groupby(["dataset","seed"]):
        b=g[g.model=="v8_c"]
        if b.empty:continue
        b=b.iloc[0]
        for _,r in g[g.model!="v8_c"].iterrows():rows.append({"dataset":ds,"seed":seed,"model":r.model,"p":r.p,"rho":r.rho,"deterministic":b.test_error,"robust":r.test_error,"delta_vs_deterministic":r.test_error-b.test_error})
      for v in values:
        prefix={"dataset":v["record"]["dataset"],"seed":v["record"]["seed"],"model":v["record"]["model"]}
        for d in v["trace"].get("solver_diagnostics",[]):diagrows.append({**prefix,"phase":"outer","fold":-1,**d})
        for candidate in v["trace"].get("bank",[]):
          for fold in candidate.get("folds",[]):
            for d in fold.get("solver_diagnostics",[]):diagrows.append({**prefix,"phase":"inner","fold":fold["fold"],"alpha_rule":candidate["alpha_rule"],"candidate_C":candidate["C"],**d})
      paired=pd.DataFrame(rows);classes=pd.DataFrame(classrows);diags=pd.DataFrame(diagrows)
      rhos=pd.DataFrame([{"dataset":ds,"model":m,"rho":g.sort_values(["error","rho"],kind="stable").iloc[0].rho,"selection_basis":"analysis only; fixed-rho smoke, not outer-test deployment selection"} for (ds,m),g in summary[summary.model!="v8_c"].groupby(["dataset","model"])])
    target=root if output_dir is None else output_dir
    for name,table in (("per_run.csv",frame),("summary.csv",summary),("paired_comparisons.csv",paired),("class_metrics.csv",classes),("solver_diagnostics.csv",diags),("rho_selection.csv",rhos),("split_registry.csv",splits)):atomic_csv(target/name,table)
    return values,frame,summary

def experiment_state(a):
    if not os.getenv("MODAL_IS_REMOTE") and not os.getenv("ROBUST_ALLOW_LOCAL_FIT"):raise RuntimeError("real fitting is Modal-only")
    run_name_path=Path(a.run_name)
    if run_name_path.is_absolute() or ".." in run_name_path.parts:raise ValueError("run_name must be a relative path without '..'")
    names=a.datasets.split(",");models=a.models.split(",");seeds=range(a.seed_start,a.seed_start+a.n_seeds);rhos=[float(v) for v in a.rhos.split(",")]
    if not set(models)<={"v8_c","robust_v8_c_p1","robust_v8_c_p2","robust_v8_c_pinf"} or "v8_c" not in models:raise ValueError("models must include v8_c and use known names")
    if not rhos:raise ValueError("at least one rho is required")
    root=Path(a.output_root)/a.run_name;root.mkdir(parents=True,exist_ok=True)
    hashes={p:digest(p) for p in CODE_FILES if Path(p).exists()};cfg={"version":"robust-v8-c-1","datasets":names,"models":models,"rhos":rhos,"seeds":list(seeds),"code_hashes":hashes,"dataset_inventory":{n:load(n)[2] for n in names},"python":platform.python_version(),"versions":{p:importlib.metadata.version(p) for p in ("numpy","scipy","pandas","scikit-learn","cvxpy","clarabel","joblib")},"resources":"4 CPU, 8192 MiB, no GPU, <=4 workers","local_git":os.getenv("ROBUST_LOCAL_GIT","unknown")}
    fp=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();cp=root/"config.json"
    if cp.exists() and json.loads(cp.read_text())!=cfg:raise RuntimeError("incompatible run-name configuration")
    if not cp.exists():atomic_json(cp,cfg)
    mp=root/"manifest.json";complete=mp.exists() and json.loads(mp.read_text()).get("status")=="complete"
    for p,h in hashes.items():
      target=root/"source_snapshot"/p;target.parent.mkdir(parents=True,exist_ok=True)
      if target.exists() and digest(target)!=h:raise RuntimeError("snapshot mismatch")
      if not target.exists():shutil.copyfile(p,target)
    check=root/"checkpoints";check.mkdir(exist_ok=True)
    tasks=[]
    for n in names:
      for s in seeds:
        tasks.append((n,s,"v8_c",0.,math.inf))
        for model in models:
          if model=="v8_c":continue
          p={"robust_v8_c_p1":1.,"robust_v8_c_p2":2.,"robust_v8_c_pinf":math.inf}[model]
          tasks += [(n,s,model,r,p) for r in rhos]
    return root,mp,check,fp,tasks,complete

def run(a):
    root,mp,check,fp,tasks,complete=experiment_state(a)
    if complete:print(json.dumps({"status":"complete","tasks":len(tasks),"run_name":a.run_name}));return
    if a.mode=="prepare":
      atomic_json(mp,{"status":"running","fingerprint":fp,"app_id":os.getenv("MODAL_APP_ID","local"),"expected_jobs":len(tasks)})
      print(json.dumps({"status":"prepared","tasks":len(tasks),"run_name":a.run_name}));return
    if not mp.exists():atomic_json(mp,{"status":"running","fingerprint":fp,"app_id":os.getenv("MODAL_APP_ID","local"),"expected_jobs":len(tasks)})
    elif json.loads(mp.read_text()).get("fingerprint")!=fp:raise RuntimeError("manifest fingerprint does not match configuration")
    if a.mode=="finalize":
      completed=len(list(check.glob("*.json")))
      if completed!=len(tasks):raise RuntimeError(f"cannot finalize: {completed}/{len(tasks)} checkpoints complete")
      start=time.perf_counter();_,frame,summary=aggregate(root);atomic_json(mp,{"status":"complete","fingerprint":fp,"app_id":os.getenv("MODAL_APP_ID","local"),"jobs":completed,"rows":len(frame),"finalize_wall_s":time.perf_counter()-start,"artifact_sha256":{p.name:digest(p) for p in root.iterdir() if p.is_file() and p.name!="manifest.json"}});print(summary.to_string(index=False));return
    task_start=0 if a.task_start is None else a.task_start;task_stop=len(tasks) if a.task_stop is None else min(a.task_stop,len(tasks))
    if task_start<0 or task_start>task_stop or task_stop>len(tasks):raise ValueError(f"invalid task slice [{task_start}, {task_stop}) for {len(tasks)} tasks")
    selected=tasks[task_start:task_stop]
    start=time.perf_counter();values=Parallel(n_jobs=min(a.n_jobs,4))(delayed(job)(n,s,m,r,p,fp,check/f"{n}-{s}-{m}-rho-{r:.12g}.json") for n,s,m,r,p in selected)
    if a.mode=="worker":print(json.dumps({"status":"worker_complete","task_start":task_start,"task_stop":task_stop,"jobs":len(values),"wall_s":time.perf_counter()-start}));return
    _,frame,summary=aggregate(root);atomic_json(mp,{"status":"complete","fingerprint":fp,"app_id":os.getenv("MODAL_APP_ID","local"),"jobs":len(values),"rows":len(frame),"wall_s":time.perf_counter()-start,"artifact_sha256":{p.name:digest(p) for p in root.iterdir() if p.is_file() and p.name!="manifest.json"}});print(summary.to_string(index=False))

def parser():
    p=argparse.ArgumentParser();p.add_argument("--datasets",default="parkinson,iris");p.add_argument("--models",default="v8_c,robust_v8_c_p2");p.add_argument("--rhos",default="1e-4");p.add_argument("--n-seeds",type=int,default=1);p.add_argument("--seed-start",type=int,default=13000);p.add_argument("--n-jobs",type=int,default=2);p.add_argument("--run-name",required=True);p.add_argument("--output-root",default="results/robust");p.add_argument("--mode",choices=("normal","prepare","worker","finalize"),default="normal");p.add_argument("--task-start",type=int);p.add_argument("--task-stop",type=int);return p
if __name__=="__main__":run(parser().parse_args())
