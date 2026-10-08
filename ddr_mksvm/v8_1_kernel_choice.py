"""One deterministic geometry candidate, isolated from frozen V7 and V8-C."""
from copy import deepcopy
from itertools import combinations
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from . import v8_class_sensitive as base
from .v7_cross_dataset import fit_transform, metric_record
from .iris_research import SVC_C_GRID


def quadratic_scale(X):
    value=float(np.mean(np.square(1+np.einsum('ij,ij->i',X,X))))
    if not np.isfinite(value) or value<=0:raise ValueError('Invalid kernel normalizer')
    return value


def fit_quadratic(X,y,C):
    q=quadratic_scale(X);classes=np.unique(y);models=[];diag=[]
    for a,b in combinations(classes,2):
        mask=np.isin(y,[a,b]);yy=y[mask];weights=base.pair_weights(yy,'sqrt')
        cl=SVC(C=C/q,kernel='poly',degree=2,coef0=1.,gamma=1.,
            class_weight=weights,tol=1e-9,shrinking=True,decision_function_shape='ovo').fit(X[mask],yy)
        if cl.fit_status_:raise RuntimeError('Quadratic SVC failed to converge')
        models.append(cl)
        diag.append(dict(pair=[int(a),int(b)],counts={str(c):int((yy==c).sum()) for c in (a,b)},
            weights={str(c):float(w) for c,w in weights.items()},
            normalized_kernel_penalties={str(c):float(C*w) for c,w in weights.items()},
            solver_penalties={str(c):float(C*w/q) for c,w in weights.items()},
            status=int(cl.fit_status_),solver='libsvm-smo',iterations=cl.n_iter_.tolist(),
            support_vectors=int(cl.support_.size)))
    return dict(models=models,classes=classes,diagnostics=diag,quadratic_scale=q)


def quadratic_bank(X,y,transform,seed):
    """TRAIN-ONLY: no outer-test arguments or labels."""
    start=time.perf_counter();prepared=[];folds=[]
    for fi,va in StratifiedKFold(3,shuffle=True,random_state=20000+int(seed)).split(X,y):
        A,B,prep=fit_transform(X[fi],X[va],transform)
        prepared.append((A,B,y[fi],y[va]));folds.append(dict(train_indices=fi.tolist(),
            validation_indices=va.tolist(),preprocessing=prep,quadratic_scale=quadratic_scale(A)))
    records=[]
    for ci,C in enumerate(SVC_C_GRID):
        details=[]
        for k,(A,B,ya,yb) in enumerate(prepared):
            model=fit_quadratic(A,ya,C);pred,_,_=base.predict_model(model,B)
            details.append(dict(fold=k,losses=base.losses(yb,pred),predictions=pred.tolist(),
                solver_diagnostics=model['diagnostics'],quadratic_scale=model['quadratic_scale']))
        records.append(dict(kernel='quadratic',C=float(C),order=ci,
            losses={m:float(np.mean([r['losses'][m] for r in details])) for m in ('error','balanced','f1')},folds=details))
    return dict(records=records,folds=folds,selection_time_s=time.perf_counter()-start)


def choose(rbf_records,quad_records,include_quadratic=True):
    r=min(rbf_records,key=lambda r:(r['losses']['balanced'],*r['order']))
    if not include_quadratic:return 'rbf',r
    q=min(quad_records,key=lambda r:(r['losses']['balanced'],r['order']))
    # Exact ties prefer EVERY pre-existing RBF configuration over quadratic.
    return ('quadratic',q) if q['losses']['balanced']<r['losses']['balanced'] else ('rbf',r)


def evaluate(name,X,y,spec,seed):
    start=time.perf_counter()
    job=base.evaluate_variants(name,X,y,spec,seed,['v7','v8_c'])
    tr=np.asarray(job['split']['train_indices']);te=np.asarray(job['split']['test_indices'])
    bank=quadratic_bank(X[tr],y[tr],spec['transform'],seed);job['banks']['quadratic']=bank
    selected=min(bank['records'],key=lambda r:(r['losses']['balanced'],r['order']))
    began=time.perf_counter();A,B,prep=fit_transform(X[tr],X[te],spec['transform'])
    model=fit_quadratic(A,y[tr],selected['C']);pred,margins,votes=base.predict_model(model,B)
    refit=time.perf_counter()-began
    params=dict(kernel='quadratic_trace_normalized',C=selected['C'],quadratic_scale=model['quadratic_scale'],
        effective_solver_C=selected['C']/model['quadratic_scale'],preprocessing=prep,
        selection_metric='balanced',inner_losses=selected['losses'],pairs=model['diagnostics'])
    trace=dict(margins_positive_second=margins.tolist(),votes=votes.tolist(),
        vote_ties=int(((votes==votes.max(1)[:,None]).sum(1)>1).sum()),solver_diagnostics=model['diagnostics'])
    qr=metric_record(name,'quadratic_only',seed,y[te],pred,bank['selection_time_s']+refit,params,tr,te)
    job['records'].append(qr);job['traces']['quadratic_only']=trace
    winner,sel=choose(job['banks']['sqrt']['records'],bank['records'])
    if winner=='rbf':
        cr=deepcopy(next(r for r in job['records'] if r['model']=='v8_c'))
        cr['runtime_s']+=bank['selection_time_s']
        chosen_trace=deepcopy(job['traces']['v8_c'])
    else:
        cr=deepcopy(qr);cr['runtime_s']+=job['banks']['sqrt']['selection_time_s']
        chosen_trace=deepcopy(trace)
    cr['model']='v8_1_kernel_choice';job['records'].append(cr)
    job['traces']['v8_1_kernel_choice']=chosen_trace
    job['kernel_selection']=dict(winner=winner,selected_C=sel['C'],inner_balanced_loss=sel['losses']['balanced'],
        comparison='exact minimum; all frozen RBF configurations precede quadratic',
        source_outer_prediction='v8_c' if winner=='rbf' else 'quadratic_only')
    job['actual_job_wall_s']=time.perf_counter()-start
    return job
