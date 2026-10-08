"""Create consolidated class, hyperparameter, and solver tables."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def class_rows(frame,stage):
 out=[]
 for (ds,model),g in frame.groupby(['dataset','model']):
  labels=json.loads(g.iloc[0].labels); cm=sum((np.array(json.loads(x),int) for x in g.confusion_matrix),np.zeros((len(labels),len(labels)),int))
  for i,label in enumerate(labels):
   tp=cm[i,i];fn=cm[i].sum()-tp;fp=cm[:,i].sum()-tp
   precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0;f1=2*precision*recall/(precision+recall) if precision+recall else 0
   out.append({'stage':stage,'dataset':ds,'model':model,'class':label,'support':int(cm[i].sum()),'precision':precision,'recall':recall,'f1':f1,'errors':int(fn)})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='results/v7_cross_dataset');a=p.parse_args();root=Path(a.root);out=root/'analysis';out.mkdir(exist_ok=True)
 runs={'gate8':'v7-cross-gate8-fallback-20260909','final96':'v7-cross-final96-20260909'};classes=[];hyp=[];sol=[]
 for stage,run in runs.items():
  base=root/run;f=pd.read_csv(base/'per_run.csv');d=pd.read_csv(base/'solver_diagnostics.csv');classes+=class_rows(f,stage)
  for _,r in f[f.model=='v7'].iterrows():
   q=json.loads(r.selected_hyperparameters);hyp.append({'stage':stage,'dataset':r.dataset,'seed':r.seed,'alpha_rule':q['alpha_rule'],'alpha':q['alpha'],'C':q['C'],'inner_error':q['inner_error'],'support_vectors':q['support_vectors']})
  for keys,g in d.groupby(['dataset','model','solver','solver_status'],dropna=False):
   sol.append({'stage':stage,'dataset':keys[0],'model':keys[1],'solver':keys[2],'status':keys[3],'fits':len(g),'warnings':int((g.solver_status_code.fillna(0)!=0).sum()),
    'fallbacks':int(g.get('solver_fallback_from',pd.Series('',index=g.index)).fillna('').astype(str).ne('').sum()),'selected_trivial':int(((g.get('selected_for_task',False)==True)&(g.get('trivial_u',False)==True)).sum())})
 pd.DataFrame(classes).to_csv(out/'class_metrics.csv',index=False);pd.DataFrame(hyp).to_csv(out/'v7_hyperparameters.csv',index=False);pd.DataFrame(sol).to_csv(out/'solver_summary.csv',index=False)
 h=pd.DataFrame(hyp); freq=h.groupby(['stage','dataset','alpha_rule']).size().rename('count').reset_index();freq.to_csv(out/'alpha_rule_frequency.csv',index=False)
 cf=h.groupby(['stage','dataset','C']).size().rename('count').reset_index();cf.to_csv(out/'C_frequency.csv',index=False)
 print(pd.DataFrame(classes).query("stage=='final96'").to_string(index=False));print(pd.DataFrame(sol).to_string(index=False))
if __name__=='__main__':main()
