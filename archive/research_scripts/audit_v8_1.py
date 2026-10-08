"""Deterministic V8.1 hypothesis audit: saved evidence and Gram algebra, NO fits."""
import json
from pathlib import Path
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from ddr_mksvm.experiments.class_sensitive_runner import load, digest
from ddr_mksvm.v7_cross_dataset import fit_transform
from archive.research_scripts.verify_v8_confirmation import save_json, save

OUT=Path('results/v8_1/analysis')
FINAL=Path('results/v8/final96/v8-final96-20260909')


def integrity():
    frozen=json.loads(Path('results/v8/analysis/deterministic_v8_c_final_freeze.json').read_text())
    hashes=json.loads(Path('results/v8/analysis/v7_freeze.json').read_text())
    hashes.update(frozen['code_hashes'])
    hashes.update({str(FINAL/k):v for k,v in frozen['source_result_hashes'].items()})
    hashes.update(frozen['analysis_hashes'])
    for p in ('results/v8/analysis/deterministic_v8_c_final_freeze.json',
              'results/v8/analysis/deterministic_candidate_freeze.json','results/v8/analysis/v7_freeze.json'):
        hashes[p]=digest(p)
    for name,expected in hashes.items():
        assert digest(name)==expected, f'FROZEN BASELINE CHANGED: {name}'
    # Inventory additional historical evidence without changing it.
    for p in Path('results/v7_cross_dataset').rglob('*'):
        if p.is_file():hashes[p.as_posix()]=digest(p)
    registries=[]
    for p in Path('results').rglob('config.json'):
        cfg=json.loads(p.read_text(encoding='utf-8-sig'))
        seeds=cfg.get('seeds',[])
        if not isinstance(seeds,list):continue
        overlap=[s for s in seeds if isinstance(s,int) and (11000<=s<=11023 or 12000<=s<=12095)]
        if overlap:registries.append(dict(file=p.as_posix(),seeds=overlap))
    assert not registries, f'Fresh registry overlap: {registries}'
    return dict(status='passed',hashes=hashes,fresh_registry_overlap=registries)


def gram_summary(K,y):
    H=K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean()
    T=(y[:,None]==y[None,:]).astype(float)
    T=T-T.mean(0)[None,:]-T.mean(1)[:,None]+T.mean()
    e=np.linalg.eigvalsh((H+H.T)/2)
    pos=e[e>1e-10*max(e.max(),1.)]
    return dict(centered_alignment=float(np.sum(H*T)/(np.linalg.norm(H)*np.linalg.norm(T))),
        numerical_rank=int(len(pos)),participation_rank=float(pos.sum()**2/np.square(pos).sum()),
        trace_mean=float(np.trace(K)/len(K)),minimum_eigenvalue=float(e.min()))


def audit():
    protection=integrity()
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'baseline_integrity.json',protection)
    cfg=json.loads((FINAL/'config.json').read_text())
    selections=[]; margins=[]; flows=[]; pairs=[]; cycles=[]; geometry=[]
    for ds in cfg['datasets']:
        for seed in cfg['seeds']:
            job=json.loads((FINAL/'checkpoints'/f'{ds}-{seed}.json').read_text())
            y=np.asarray(job['split']['y_test']);classes=np.unique(y)
            rr={r['model']:r for r in job['records']}
            pred={m:np.asarray(json.loads(r['prediction_vector'])) for m,r in rr.items()}
            for m,r in rr.items():
                h=json.loads(r['selected_hyperparameters'])
                selections.append(dict(dataset=ds,seed=seed,model=m,alpha=h['alpha'],alpha_rule=h['alpha_rule'],C=h['C'],runtime_s=r['runtime_s']))
            trace=job['traces']['v8_c'];scores=np.asarray(trace['margins_positive_second']);v=np.asarray(trace['votes'])
            tied=(v==v.max(1)[:,None]).sum(1)>1
            for label in classes:
                for old in classes:
                    for new in classes:
                        n=int(((y==label)&(pred['v7']==old)&(pred['v8_c']==new)).sum())
                        if n:flows.append(dict(dataset=ds,seed=seed,truth=int(label),v7=int(old),v8_c=int(new),count=n))
            for i,label in enumerate(y):
                margins.append(dict(dataset=ds,seed=seed,label=int(label),correct=bool(pred['v8_c'][i]==label),
                    minimum_absolute_margin=float(np.abs(scores[i]).min()),vote_tie=bool(tied[i])))
            for col,(a,b) in enumerate(combinations(classes,2)):
                diag=trace['solver_diagnostics'][col]
                for label in (a,b):
                    mask=y==label; signed=scores[mask,col]*(1 if label==b else -1)
                    pairs.append(dict(dataset=ds,seed=seed,pair=f'{a}-{b}',label=int(label),n=int(mask.sum()),
                        correct=int((signed>0).sum()),margin_median=float(np.median(signed)),
                        support_fraction=diag['support_vectors']/sum(diag['counts'].values()),
                        weight=diag['weights'][str(label)],C_penalty=diag['penalties'][str(label)]))
            truth_in_tie=np.asarray([v[i,list(classes).index(label)]==v[i].max() for i,label in enumerate(y)])
            cycles.append(dict(dataset=ds,seed=seed,ties=int(tied.sum()),errors=int((pred['v8_c']!=y).sum()),
                tie_errors=int((tied&(pred['v8_c']!=y)).sum()),
                tied_true_class_oracle_repairs=int((tied&truth_in_tie&(pred['v8_c']!=y)).sum()),
                new_majority_errors=int(((y==classes[0])&(pred['v7']==y)&(pred['v8_c']!=y)).sum()),
                new_majority_errors_tied=int(((y==classes[0])&(pred['v7']==y)&(pred['v8_c']!=y)&tied).sum())))
        # Single consumed split, algebra only: no optimizer and no fresh seeds.
        X,yy,spec=load(ds)
        job=json.loads((FINAL/'checkpoints'/f'{ds}-10000.json').read_text())
        tr=np.asarray(job['split']['train_indices']);A,_,_=fit_transform(X[tr],X[tr],spec['transform'])
        yy=yy[tr];d=pdist(A);D=squareform(d);h={r['model']:json.loads(r['selected_hyperparameters']) for r in job['records']}
        Q=(1+A@A.T)**2;Q/=np.diag(Q).mean()
        same=yy[:,None]==yy[None,:];upper=np.triu(np.ones_like(same,bool),1)
        _,inv=np.unique(A,axis=0,return_inverse=True)
        unavoidable=0;conflict_groups=0
        for k in np.unique(inv):
            counts=Counter(yy[inv==k]);unavoidable+=sum(counts.values())-max(counts.values())
            conflict_groups+=len(counts)>1
        kernels={'quadratic_trace_normalized':Q}
        for model,params in h.items():kernels[model+'_selected_rbf']=np.exp(-D**2/(2*params['alpha']**2))
        for name,K in kernels.items():
            geometry.append(dict(dataset=ds,seed=10000,kernel=name,n_train=len(A),features=A.shape[1],
                median_within_distance=float(np.median(D[upper&same])),median_between_distance=float(np.median(D[upper&~same])),
                unique_training_rows=int(len(np.unique(inv))),conflicting_duplicate_groups=int(conflict_groups),
                empirical_duplicate_minimum_errors=int(unavoidable),**gram_summary(K,yy)))
    tables=dict(selections=pd.DataFrame(selections),margin_observations=pd.DataFrame(margins),
        confusion_flows=pd.DataFrame(flows),pair_diagnostics=pd.DataFrame(pairs),
        cycle_impact=pd.DataFrame(cycles),single_split_geometry=pd.DataFrame(geometry))
    for name,table in tables.items():save(OUT/(name+'.csv'),table.to_csv(index=False).encode())
    selected=tables['selections'];p=tables['pair_diagnostics'];cy=tables['cycle_impact'];ma=tables['margin_observations']
    print('Freeze verified:',len(protection['hashes']),'files; fresh registries unused')
    print(cy.groupby('dataset').sum(numeric_only=True).drop(columns='seed').to_string())
    print(p[p.dataset=='heart_disease'].groupby(['pair','label']).agg(n=('n','sum'),correct=('correct','sum'),margin_median=('margin_median','median'),support_fraction=('support_fraction','mean'),weight=('weight','mean')).to_string())
    print(tables['single_split_geometry'].to_string(index=False))
    print(ma.groupby(['dataset','correct']).minimum_absolute_margin.median().to_string())
    print(selected[selected.dataset.isin(['heart_disease','mammographicmass_binary'])].groupby(['dataset','model','alpha_rule']).size().to_string())


if __name__=='__main__':audit()
