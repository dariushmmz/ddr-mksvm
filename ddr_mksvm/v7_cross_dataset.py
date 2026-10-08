"""Frozen V7 and corrected executable-Legacy cross-dataset primitives."""
from __future__ import annotations

import hashlib, itertools, json, time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from .iris_research import (ALPHA_RULES, PAPER_NU_GRID, SVC_C_GRID,
    alpha_from_rule, solve_binary_q1)

DATASETS = {
 "parkinson": dict(file="parkinson.csv", transform="minmax", kernel="hom_linear", paper=0.1319),
 "blood_transfusion": dict(file="blood_transfusion.csv", transform="standardization", kernel="inhom_cubic", paper=0.2072),
 "mammographicmass_binary": dict(file="mammographicmass_binary.csv", transform="standardization", kernel="inhom_quadratic", paper=0.1571),
 "breast_cancer_diagnostic": dict(file="breast_cancer_diagnostic.csv", transform="minmax", kernel="inhom_quadratic", paper=0.0302),
 "wine": dict(file="wine.csv", transform="standardization", kernel="inhom_linear", paper=0.0277),
 "heart_disease": dict(file="heart_disease.csv", transform="standardization", kernel="inhom_linear", paper=0.1748, complete_case=True),
 "dermatology": dict(file="dermatology.csv", transform="none", kernel="inhom_quadratic", paper=0.0164, complete_case=True),
}

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def load_dataset(name, root="dataset"):
    spec=DATASETS[name]; path=f"{root}/{spec['file']}"; raw=pd.read_csv(path)
    X=raw.iloc[:,:-1].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    y=pd.to_numeric(raw.iloc[:,-1],errors="raise").to_numpy(int)
    finite=np.isfinite(X).all(axis=1)
    if not finite.all():
        if not spec.get("complete_case"): raise ValueError(f"{name} has nonfinite features")
        X,y=X[finite],y[finite]
    classes=np.unique(y)
    if len(classes)<2: raise ValueError("need at least two classes")
    return X,y,{"csv_sha256":sha256_file(path),"raw_rows":len(raw),"rows":len(y),
        "features":X.shape[1],"classes":classes.tolist(),"class_counts":{str(c):int((y==c).sum()) for c in classes},
        "excluded_incomplete_rows":int((~finite).sum()),**spec}

def split_indices(y,seed):
    idx=np.arange(len(y));tr,te=train_test_split(idx,test_size=.25,stratify=y,random_state=int(seed))
    return np.sort(tr),np.sort(te)

def fit_transform(A,B,mode):
    A=np.asarray(A,float);B=np.asarray(B,float)
    if mode=="none": return A.copy(),B.copy(),{}
    if mode=="standardization": offset=A.mean(0);scale=A.std(0,ddof=0)
    elif mode=="minmax": offset=A.min(0);scale=A.max(0)-offset
    else: raise ValueError(mode)
    scale=np.where(scale>0,scale,1.)
    return (A-offset)/scale,(B-offset)/scale,{"offset":offset.tolist(),"scale":scale.tolist()}

def _kernel_spec(X,key):
    if key=="hom_linear": return "poly",1,0.
    degree={"inhom_linear":1,"inhom_quadratic":2,"inhom_cubic":3}[key]
    return "poly",degree,float(np.max(np.std(X,axis=0,ddof=0)))

def _legacy_fit_predict(Xtr,ytr,Xte,spec,seed):
    started=time.perf_counter(); kind,degree,offset=_kernel_spec(Xtr,spec["kernel"])
    classes=sorted(np.unique(ytr).tolist()); models=[]; diagnostics=[]
    tasks=[(classes[-1],classes[0])] if len(classes)==2 else [(c,None) for c in classes]
    for ti,(pos,neg) in enumerate(tasks):
        mask=np.ones(len(ytr),bool) if neg is None else np.isin(ytr,[pos,neg])
        yy=np.where(ytr[mask]==pos,1.,-1.); solved=[]
        for oi,nu in enumerate(PAPER_NU_GRID):
            context={"model":"legacy","seed":seed,"task":ti,"positive":pos,"negative":neg,"nu_order":oi}
            try:
                model,d=solve_binary_q1(Xtr[mask],yy,1.,nu,"grid",context=context,
                    kernel_kind=kind,polynomial_degree=degree,polynomial_offset=offset,solver_backend="scipy-highs")
                d["solver_fallback_from"]=""
            except RuntimeError as first:
                # Use the project's validated LP fallback rather than a new solver path.
                model,d=solve_binary_q1(Xtr[mask],yy,1.,nu,"grid",context=context,
                    kernel_kind=kind,polynomial_degree=degree,polynomial_offset=offset,solver_backend="cvxpy-clarabel")
                d["solver_fallback_from"]=str(first)
            d["selected_for_task"]=False;diagnostics.append(d);solved.append((d["training_error"],oi,model))
        chosen=min(solved,key=lambda z:(z[0],z[1]))[2];chosen.positive_class=pos;chosen.negative_class=neg;chosen.diagnostics["selected_for_task"]=True;models.append(chosen)
    if len(classes)==2: pred=np.where(models[0].decision(Xte)>0,classes[-1],classes[0])
    else:
        scores=np.column_stack([m.decision(Xte) for m in models]);pred=np.asarray(classes)[np.argmax(scores,axis=1)]
    return pred,np.where([d["selected_for_task"] for d in diagnostics])[0].tolist(),diagnostics,time.perf_counter()-started,{"degree":degree,"offset":offset}

def _v7_fit_predict(rawXtr,ytr,rawXte,spec,seed):
    started=time.perf_counter();cv=list(StratifiedKFold(3,shuffle=True,random_state=20000+int(seed)).split(rawXtr,ytr));out=[];diag=[]
    for ai,rule in enumerate(ALPHA_RULES):
      for ci,C in enumerate(SVC_C_GRID):
        errors=[]
        for fold,(fi,va) in enumerate(cv):
          A,B,_=fit_transform(rawXtr[fi],rawXtr[va],spec["transform"]);alpha=alpha_from_rule(A,rule)
          cl=SVC(C=C,kernel="rbf",gamma=1/(2*alpha**2),decision_function_shape="ovo",shrinking=True,tol=1e-9).fit(A,ytr[fi])
          err=float(np.mean(cl.predict(B)!=ytr[va]));errors.append(err)
          diag.append({"model":"v7","seed":seed,"phase":"inner","fold":fold,"alpha_rule":rule,"alpha":alpha,"C":C,"validation_error":err,
                       "solver":"libsvm-smo","solver_status":"ok" if cl.fit_status_==0 else "failed","solver_iterations":int(np.sum(cl.n_iter_)),"selected":False})
        out.append((float(np.mean(errors)),ai,ci,rule,float(C)))
    best=min(out);rule,C=best[3],best[4];A,B,prep=fit_transform(rawXtr,rawXte,spec["transform"]);alpha=alpha_from_rule(A,rule)
    cl=SVC(C=C,kernel="rbf",gamma=1/(2*alpha**2),decision_function_shape="ovo",shrinking=True,tol=1e-9).fit(A,ytr);pred=cl.predict(B)
    diag.append({"model":"v7","seed":seed,"phase":"outer","fold":-1,"alpha_rule":rule,"alpha":alpha,"C":C,"validation_error":best[0],
                 "solver":"libsvm-smo","solver_status":"ok" if cl.fit_status_==0 else "failed","solver_iterations":int(np.sum(cl.n_iter_)),"selected":True,"support_vectors":int(cl.support_.size)})
    return pred,diag,time.perf_counter()-started,{"alpha_rule":rule,"alpha":alpha,"C":C,"inner_error":best[0],"preprocessing":prep,"support_vectors":int(cl.support_.size)}

def metric_record(name,model,seed,ytrue,pred,runtime,params,tr,te):
    labels=sorted(np.unique(ytrue).tolist());pr,re,f1,sup=precision_recall_fscore_support(ytrue,pred,labels=labels,zero_division=0)
    rec={"dataset":name,"model":model,"seed":seed,"n_test":len(ytrue),"errors":int(np.sum(pred!=ytrue)),"test_error":float(np.mean(pred!=ytrue)),
         "accuracy":float(accuracy_score(ytrue,pred)),"balanced_accuracy":float(balanced_accuracy_score(ytrue,pred)),"macro_f1":float(f1_score(ytrue,pred,average="macro")),
         "runtime_s":runtime,"labels":json.dumps(labels),"confusion_matrix":json.dumps(confusion_matrix(ytrue,pred,labels=labels).tolist()),
         "class_precision":json.dumps(dict(zip(map(str,labels),map(float,pr)))),"class_recall":json.dumps(dict(zip(map(str,labels),map(float,re)))),
         "class_f1":json.dumps(dict(zip(map(str,labels),map(float,f1)))),"class_support":json.dumps(dict(zip(map(str,labels),map(int,sup)))),
         "prediction_vector":json.dumps(np.asarray(pred,int).tolist()),"prediction_hash":hashlib.sha256(np.asarray(pred,dtype="<i8").tobytes()).hexdigest(),
         "train_indices_hash":hashlib.sha256(np.asarray(tr,dtype="<i8").tobytes()).hexdigest(),"test_indices_hash":hashlib.sha256(np.asarray(te,dtype="<i8").tobytes()).hexdigest(),
         "selected_hyperparameters":json.dumps(params,sort_keys=True)}
    if len(labels)==2: rec["sensitivity_class_1"]=float(re[labels.index(1)]) if 1 in labels else float(re[-1]);rec["specificity_class_0"]=float(re[labels.index(0)]) if 0 in labels else float(re[0])
    return rec

def evaluate_seed(name,X,y,spec,seed):
    tr,te=split_indices(y,seed);rawtr,rawte=X[tr],X[te]
    Ltr,Lte,prep=fit_transform(rawtr,rawte,spec["transform"])
    lp,_,ld,lt,lpar=_legacy_fit_predict(Ltr,y[tr],Lte,spec,seed);vp,vd,vt,vpar=_v7_fit_predict(rawtr,y[tr],rawte,spec,seed)
    lpar.update({"preprocessing":prep,"nu_grid":list(PAPER_NU_GRID)})
    diagnostics=ld+vd
    for row in diagnostics: row["dataset"]=name
    return [metric_record(name,"legacy",seed,y[te],lp,lt,lpar,tr,te),metric_record(name,"v7",seed,y[te],vp,vt,vpar,tr,te)],diagnostics,{"seed":seed,"train_indices":tr.tolist(),"test_indices":te.tolist()}

def summarize(frame):
    rows=[]
    for (ds,m),g in frame.groupby(["dataset","model"]):
        x=g.test_error.to_numpy(float);n=len(x);se=x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan;q=t.ppf(.975,n-1) if n>1 else np.nan
        rows.append({"dataset":ds,"model":m,"seeds":n,"errors":int(g.errors.sum()),"predictions":int(g.n_test.sum()),"mean_error":x.mean(),"sd_error":x.std(ddof=1) if n>1 else np.nan,
          "se_error":se,"ci_low":x.mean()-q*se if n>1 else np.nan,"ci_high":x.mean()+q*se if n>1 else np.nan,"median_error":np.median(x),"accuracy":g.accuracy.mean(),
          "balanced_accuracy":g.balanced_accuracy.mean(),"macro_f1":g.macro_f1.mean(),"mean_runtime_s":g.runtime_s.mean(),"median_runtime_s":g.runtime_s.median(),"total_runtime_s":g.runtime_s.sum()})
    return pd.DataFrame(rows)

def paired(frame):
    rows=[]
    for ds,g in frame.groupby("dataset"):
      p=g.pivot(index="seed",columns="model",values="test_error");d=(p.v7-p.legacy).to_numpy();n=len(d);se=d.std(ddof=1)/np.sqrt(n) if n>1 else np.nan;q=t.ppf(.975,n-1) if n>1 else np.nan
      nz=d[d!=0];wp=float(wilcoxon(nz).pvalue) if len(nz)>0 else 1.
      rt=g.pivot(index="seed",columns="model",values="runtime_s")
      rows.append({"dataset":ds,"seeds":n,"mean_delta":d.mean(),"ci_low":d.mean()-q*se if n>1 else np.nan,"ci_high":d.mean()+q*se if n>1 else np.nan,
       "wins":int((d<0).sum()),"ties":int((d==0).sum()),"losses":int((d>0).sum()),"wilcoxon_p":wp,"standardized_effect":float(d.mean()/d.std(ddof=1)) if n>1 and d.std(ddof=1)>0 else 0.,
       "runtime_ratio_v7_over_legacy":float(rt.v7.sum()/rt.legacy.sum()),"runtime_saved_s":float(rt.legacy.sum()-rt.v7.sum())})
    return pd.DataFrame(rows)
