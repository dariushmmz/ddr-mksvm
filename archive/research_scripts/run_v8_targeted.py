"""Leakage-safe targeted V8 development experiments on observed Iris seeds."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from ddr_mksvm.iris_research import ALPHA_RULES,SVC_C_GRID,alpha_from_rule,locked_split_indices,reconstruct_authors_iris

PAIRS=list(combinations([1,2,3],2))
def transform_fit_test(a,b,mode):
    if mode=='none': return a,b
    if mode=='standard':
        loc=a.mean(0); scale=a.std(0,ddof=1); scale=np.where(scale>0,scale,1)
    elif mode=='minmax':
        loc=a.min(0); scale=a.max(0)-loc; scale=np.where(scale>0,scale,1)
    else: raise ValueError(mode)
    return (a-loc)/scale,(b-loc)/scale
def svc(C,alpha): return SVC(C=C,kernel='rbf',gamma=1/(2*alpha**2),decision_function_shape='ovo',shrinking=True,tol=1e-9)
def select_global(X,y,seed,mode):
    cv=list(StratifiedKFold(3,shuffle=True,random_state=20000+seed).split(X,y)); scores=[]
    for ai,rule in enumerate(ALPHA_RULES):
      for ci,C in enumerate(SVC_C_GRID):
        es=[]
        for fi,va in cv:
          A,B=transform_fit_test(X[fi],X[va],mode); alpha=alpha_from_rule(A,rule)
          es.append(np.mean(svc(C,alpha).fit(A,y[fi]).predict(B)!=y[va]))
        scores.append((np.mean(es),ai,ci,rule,float(C)))
    return min(scores),cv
def select_pair(X,y,cv,mode,pair,base_rule,base_C,which):
    rules=ALPHA_RULES if which in ('alpha','both') else (base_rule,)
    Cs=SVC_C_GRID if which in ('C','both') else (base_C,)
    scores=[]
    for ai,rule in enumerate(rules):
      for ci,C in enumerate(Cs):
        es=[]
        for fi,va in cv:
          fm=np.isin(y[fi],pair); vm=np.isin(y[va],pair)
          # For C-only, preserve V7's all-class bandwidth statistic exactly.
          if mode != 'none': raise ValueError('pair-specific variants currently require raw preprocessing')
          fitbase=X[fi] if which=='C' else X[fi][fm]
          A=X[fi][fm]; Ball=X[va][vm]
          alpha=alpha_from_rule(fitbase,rule)
          es.append(np.mean(svc(C,alpha).fit(A,y[fi][fm]).predict(Ball)!=y[va][vm]))
        scores.append((np.mean(es),ai,ci,rule,float(C)))
    return min(scores)
def pair_predict(models,X):
    votes=np.zeros((len(X),3),int); margins=[]
    for (i,j),cl in models:
      pr=cl.predict(X).astype(int); votes[np.arange(len(X)),pr-1]+=1; margins.append(cl.decision_function(X))
    return np.argmax(votes,1)+1,np.column_stack(margins),votes
def ard_candidates():
    vals=[]
    for g in range(-3,4):
      for h in range(-2,3):
        logs=np.array([-g/2,-g/2,g/2-h/2,g/2+h/2],float)
        vals.append((float(np.linalg.norm(logs)),abs(g)+abs(h),g,h,np.power(2.0,logs)))
    return [z[-1] for z in sorted(vals,key=lambda z:(z[0],z[1],z[2],z[3]))]
def select_pair_ard(X,y,cv,pair,base_rule,base_C):
    scores=[]
    for order,w in enumerate(ard_candidates()):
      es=[]
      for fi,va in cv:
        fm=np.isin(y[fi],pair);vm=np.isin(y[va],pair);alpha=alpha_from_rule(X[fi],base_rule)
        es.append(np.mean(svc(base_C,alpha).fit(X[fi][fm]*w,y[fi][fm]).predict(X[va][vm]*w)!=y[va][vm]))
      scores.append((np.mean(es),order,w))
    return min(scores,key=lambda z:(z[0],z[1]))
def select_pair_weight(X,y,cv,pair,base_rule,base_C):
    ratios=[1.0,2**-.5,2**.5,0.5,2.0,0.25,4.0];scores=[]
    for order,r in enumerate(ratios):
      es=[];cw={pair[0]:float(np.sqrt(r)),pair[1]:float(1/np.sqrt(r))}
      for fi,va in cv:
        fm=np.isin(y[fi],pair);vm=np.isin(y[va],pair);alpha=alpha_from_rule(X[fi],base_rule)
        cl=SVC(C=base_C,kernel='rbf',gamma=1/(2*alpha**2),class_weight=cw,decision_function_shape='ovo',shrinking=True,tol=1e-9)
        es.append(np.mean(cl.fit(X[fi][fm],y[fi][fm]).predict(X[va][vm])!=y[va][vm]))
      scores.append((np.mean(es),order,r))
    return min(scores)
def evaluate(Xtr,ytr,Xte,yte,seed,variant):
    mode={'raw':'none','standard':'standard','minmax':'minmax'}.get(variant,'none')
    best,cv=select_global(Xtr,ytr,seed,mode); _,_,_,base_rule,base_C=best
    Atr,Ate=transform_fit_test(Xtr,Xte,mode); outer_alpha=alpha_from_rule(Atr,base_rule)
    selected={}
    if variant in ('raw','standard','minmax'):
      cl=svc(base_C,outer_alpha).fit(Atr,ytr); pred=cl.predict(Ate); margins=cl.decision_function(Ate)
    elif variant not in ('pair_ard','pair_weighted_C'):
      which={'pair_alpha':'alpha','pair_C':'C','pair_both':'both'}[variant]; models=[]
      for pair in PAIRS:
        sel=select_pair(Xtr,ytr,cv,mode,pair,base_rule,base_C,which); _,_,_,rule,C=sel
        mask=np.isin(ytr,pair); fitbase=Xtr if which=='C' else Xtr[mask]
        alpha=alpha_from_rule(fitbase,rule); models.append((pair,svc(C,alpha).fit(Xtr[mask],ytr[mask])))
        selected[f'{pair[0]}-{pair[1]}']={'rule':rule,'alpha':alpha,'C':C,'cv_error':sel[0]}
      pred,margins,votes=pair_predict(models,Ate if mode=='none' else transform_fit_test(Xtr,Xte,mode)[1])
    elif variant == 'pair_ard':
      models=[]
      for pair in PAIRS:
        score,_,w=select_pair_ard(Xtr,ytr,cv,pair,base_rule,base_C);mask=np.isin(ytr,pair)
        alpha=alpha_from_rule(Xtr,base_rule); cl=svc(base_C,alpha).fit(Xtr[mask]*w,ytr[mask])
        models.append((pair,(cl,w)));selected[f'{pair[0]}-{pair[1]}']={'weights':w.tolist(),'alpha':alpha,'C':base_C,'cv_error':score}
      votes=np.zeros((len(Xte),3),int); margins=[]
      for (i,j),(cl,w) in models:
        pr=cl.predict(Xte*w).astype(int);votes[np.arange(len(Xte)),pr-1]+=1;margins.append(cl.decision_function(Xte*w))
      pred=np.argmax(votes,1)+1;margins=np.column_stack(margins)
    else:
      models=[]
      for pair in PAIRS:
        score,_,ratio=select_pair_weight(Xtr,ytr,cv,pair,base_rule,base_C);mask=np.isin(ytr,pair)
        cw={pair[0]:float(np.sqrt(ratio)),pair[1]:float(1/np.sqrt(ratio))};alpha=alpha_from_rule(Xtr,base_rule)
        cl=SVC(C=base_C,kernel='rbf',gamma=1/(2*alpha**2),class_weight=cw,decision_function_shape='ovo',shrinking=True,tol=1e-9).fit(Xtr[mask],ytr[mask])
        models.append((pair,cl));selected[f'{pair[0]}-{pair[1]}']={'C_ratio_first_over_second':ratio,'alpha':alpha,'C':base_C,'cv_error':score}
      pred,margins,votes=pair_predict(models,Xte)
    return {'seed':seed,'variant':variant,'errors':int(np.sum(pred!=yte)),'n':len(yte),'error':float(np.mean(pred!=yte)),
            'base_rule':base_rule,'base_alpha':outer_alpha,'base_C':base_C,'inner_error':best[0],
            'pair_selection':json.dumps(selected,sort_keys=True),'predictions':json.dumps(pred.tolist()),
            'prediction_hash':hashlib.sha256(np.asarray(pred,dtype='<i8').tobytes()).hexdigest()}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed-start',type=int,default=4000);p.add_argument('--n-seeds',type=int,default=8)
 p.add_argument('--variants',default='raw,standard,minmax,pair_alpha,pair_C,pair_both');p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output); 
 if out.exists(): raise FileExistsError(out)
 vals=reconstruct_authors_iris(pd.read_csv('dataset/iris_multiclass.csv').to_numpy(float));X,y=vals[:,:4],vals[:,4].astype(int)
 rows=[]; start=time.time()
 for seed in range(a.seed_start,a.seed_start+a.n_seeds):
  tr,te=locked_split_indices(y,seed)
  for v in a.variants.split(','): rows.append(evaluate(X[tr],y[tr],X[te],y[te],seed,v))
 frame=pd.DataFrame(rows);out.mkdir(parents=True);frame.to_csv(out/'per_run.csv',index=False)
 sm=frame.groupby('variant').agg(seeds=('seed','size'),errors=('errors','sum'),n=('n','sum'),mean_split_error=('error','mean')).reset_index();sm['pooled_error']=sm.errors/sm.n
 sm.to_csv(out/'summary.csv',index=False); (out/'config.json').write_text(json.dumps(vars(a)|{'elapsed_s':time.time()-start},indent=2)+'\n');print(sm.to_string(index=False))
if __name__=='__main__':main()
