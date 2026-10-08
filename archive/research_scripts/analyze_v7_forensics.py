"""Replay the immutable V7 result and emit split/sample-level error forensics.

This script is deliberately read-only with respect to the V7 artifact directory.
It refits only the recorded outer SVCs on the recorded splits, then requires exact
prediction parity before writing a separate forensic artifact.
"""
from __future__ import annotations

import argparse, ast, hashlib, json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from ddr_mksvm.iris_research import reconstruct_authors_iris


def qstats(x):
    x = np.asarray(x, float)
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "q05": float(np.quantile(x,.05)), "q25": float(np.quantile(x,.25)),
            "q75": float(np.quantile(x,.75)), "q95": float(np.quantile(x,.95))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v7-dir", default="results/iris_experiments/iris-v7-rkhs-l2-final96-20260908")
    p.add_argument("--output", default="results/iris_v8_forensics/v7-96-20260908")
    p.add_argument("--dataset", default="dataset/iris_multiclass.csv")
    a=p.parse_args(); src=Path(a.v7_dir); out=Path(a.output)
    if out.exists(): raise FileExistsError(f"refusing to overwrite {out}")
    values=reconstruct_authors_iris(pd.read_csv(a.dataset).to_numpy(float))
    X,y=values[:,:4],values[:,4].astype(int)
    splits={int(s["seed"]):s for s in json.loads((src/"splits.json").read_text())["splits"]}
    allrows=pd.read_csv(src/"per_run.csv")
    v7=allrows[allrows.architecture.eq("rkhs_l2_rbf_svc_cv_ovo")].copy()
    comparisons={m:allrows[allrows.architecture.eq(m)].set_index("seed") for m in
                 ["legacy_grid_ova","matlab_source_legacy_highs"]}
    samples=[]; pairs=[]; parity=[]
    pair_names=list(combinations([1,2,3],2))
    for _,r in v7.sort_values("seed").iterrows():
        seed=int(r.seed); tr=np.array(splits[seed]["train_indices"],int); te=np.array(splits[seed]["test_indices"],int)
        clf=SVC(C=float(r.selected_C),kernel="rbf",gamma=1/(2*float(r.selected_alpha)**2),
                decision_function_shape="ovo",shrinking=True,tol=1e-9).fit(X[tr],y[tr])
        pred=clf.predict(X[te]).astype(int); stored=np.array(ast.literal_eval(r.prediction_vector),int)
        digest=hashlib.sha256(np.asarray(pred,dtype="<i8").tobytes()).hexdigest()
        if not np.array_equal(pred,stored) or digest != r.prediction_hash: raise RuntimeError(f"V7 parity failure seed {seed}")
        dec=clf.decision_function(X[te]); votes=np.zeros((len(te),3),int)
        true_support=np.full((len(te),2),np.nan); true_n=np.zeros(len(te),int)
        for k,(ci,cj) in enumerate(pair_names):
            # sklearn OVO decision >0 votes for the first class; <0 for second.
            win=np.where(dec[:,k]>0,ci,cj); votes[np.arange(len(te)),win-1]+=1
            relevant=np.isin(y[te],[ci,cj]); pair_pred=win[relevant]
            for pos,dval,tval,wval in zip(np.where(relevant)[0],dec[relevant,k],y[te][relevant],pair_pred):
                signed=float(dval if tval==ci else -dval)
                pairs.append(dict(seed=seed,sample_index=int(te[pos]),pair=f"{ci}-{cj}",true_class=int(tval),
                                  pair_prediction=int(wval),pair_correct=bool(wval==tval),decision=float(dval),
                                  true_signed_margin=signed,alpha=float(r.selected_alpha),C=float(r.selected_C)))
                true_support[pos,true_n[pos]]=signed; true_n[pos]+=1
        if not np.array_equal(np.argmax(votes,axis=1)+1,pred): raise RuntimeError(f"vote replay failure seed {seed}")
        other_preds={m:np.array(ast.literal_eval(df.loc[seed].prediction_vector),int) for m,df in comparisons.items()}
        for j,idx in enumerate(te):
            sv=np.sort(votes[j]); topgap=int(sv[-1]-sv[-2]); tied=int(np.sum(votes[j]==sv[-1]))
            row=dict(seed=seed,sample_index=int(idx),true_class=int(y[idx]),prediction=int(pred[j]),correct=bool(pred[j]==y[idx]),
                     alpha=float(r.selected_alpha),C=float(r.selected_C),votes_1=int(votes[j,0]),votes_2=int(votes[j,1]),votes_3=int(votes[j,2]),
                     vote_top_gap=topgap,vote_top_tied=tied,min_abs_pair_margin=float(np.min(np.abs(dec[j]))),
                     true_min_margin=float(np.min(true_support[j])),true_mean_margin=float(np.mean(true_support[j])))
            for k,(ci,cj) in enumerate(pair_names): row[f"margin_{ci}_{cj}"]=float(dec[j,k])
            for m,op in other_preds.items(): row[f"prediction_{m}"]=int(op[j]); row[f"agree_{m}"]=bool(op[j]==pred[j])
            samples.append(row)
        parity.append(dict(seed=seed,prediction_hash=digest,parity=True,alpha=float(r.selected_alpha),C=float(r.selected_C),
                           support_vectors=int(clf.support_.size),iterations=int(np.sum(clf.n_iter_))))
    sdf=pd.DataFrame(samples); pdf=pd.DataFrame(pairs); par=pd.DataFrame(parity)
    class_summary=(sdf.groupby("true_class").correct.agg(["count","sum"]).reset_index().rename(columns={"count":"predictions","sum":"correct"}))
    class_summary["errors"]=class_summary.predictions-class_summary.correct; class_summary["error_rate"]=class_summary.errors/class_summary.predictions
    errs=sdf[~sdf.correct].copy(); errs["error_pair"]=errs.apply(lambda z:f"{min(z.true_class,z.prediction)}-{max(z.true_class,z.prediction)}",axis=1)
    pair_error=errs.groupby("error_pair").size().rename("errors").reset_index()
    recurrence=(sdf.groupby(["sample_index","true_class"]).agg(test_appearances=("correct","size"),errors=("correct",lambda z:int((~z).sum())),
                mean_true_min_margin=("true_min_margin","mean"),min_true_margin=("true_min_margin","min")).reset_index())
    recurrence["error_rate"]=recurrence.errors/recurrence.test_appearances
    margins={k:{"correct":qstats(sdf.loc[sdf.correct,k]),"incorrect":qstats(sdf.loc[~sdf.correct,k])}
             for k in ["min_abs_pair_margin","true_min_margin","true_mean_margin"]}
    ties={"exact_vote_ties":int((sdf.vote_top_tied>1).sum()),"vote_gap_zero":int((sdf.vote_top_gap==0).sum()),
          "vote_gap_one":int((sdf.vote_top_gap==1).sum()),"abs_margin_le_0.05":int((sdf.min_abs_pair_margin<=.05).sum()),
          "abs_margin_le_0.10":int((sdf.min_abs_pair_margin<=.10).sum())}
    disagree={}
    for m in comparisons:
        op=sdf[f"prediction_{m}"]; oc=op.eq(sdf.true_class)
        disagree[m]={"disagreements":int((op!=sdf.prediction).sum()),"v7_only_correct":int((sdf.correct & ~oc).sum()),
                      "other_only_correct":int((~sdf.correct & oc).sum()),"both_wrong":int((~sdf.correct & ~oc).sum())}
    summary={"source_v7":str(src),"immutable_source_files_used":["config.json","splits.json","per_run.csv"],
             "prediction_parity":{"splits":len(par),"all_exact":bool(par.parity.all())},"total_predictions":len(sdf),
             "errors":int((~sdf.correct).sum()),"error_rate":float((~sdf.correct).mean()),"margin_distributions":margins,
             "ties_and_near_ties":ties,"disagreements":disagree,
             "selection":{"alpha_unique":int(par.alpha.nunique()),"C_counts":{str(k):int(v) for k,v in par.C.value_counts().sort_index().items()}},
             "definitions":{"true_min_margin":"minimum signed margin for true class over its two OVO classifiers",
                            "min_abs_pair_margin":"minimum absolute value over all three OVO decisions","near_tie":"descriptive thresholds only; not used for tuning"}}
    out.mkdir(parents=True)
    sdf.to_csv(out/"sample_predictions.csv",index=False); pdf.to_csv(out/"pair_decisions.csv",index=False)
    par.to_csv(out/"split_hyperparameters.csv",index=False); class_summary.to_csv(out/"class_error_summary.csv",index=False)
    pair_error.to_csv(out/"error_pair_summary.csv",index=False); recurrence.sort_values(["errors","error_rate"],ascending=False).to_csv(out/"sample_recurrence.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
