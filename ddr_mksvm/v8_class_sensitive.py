"""Isolated weighted RKHS-l2 OVO research; never modifies frozen V7."""
from __future__ import annotations
import itertools
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from . import v7_cross_dataset as frozen
from .iris_research import ALPHA_RULES, SVC_C_GRID, alpha_from_rule

VARIANTS = {
    'v7': ('none', 'error'),
    'selection_balanced': ('none', 'balanced'),
    'selection_f1': ('none', 'f1'),
    'v8_a': ('inverse', 'error'),
    'v8_b': ('inverse', 'balanced'),
    'inverse_f1': ('inverse', 'f1'),
    'v8_c': ('sqrt', 'balanced'),
    'v8_d': ('ratio', 'balanced'),
}
RATIOS = (1., .5, 2., .25, 4.)

def pair_weights(y, family, ratio=1.):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or ratio <= 0:
        raise ValueError('requires two classes and positive ratio')
    if family == 'none':
        weights = np.ones(2)
    elif family in ('inverse', 'sqrt'):
        tau = 1. if family == 'inverse' else .5
        weights = len(y)*counts.astype(float)**(-tau)/np.sum(counts**(1-tau))
    elif family == 'ratio':
        minority = 0 if counts[0] < counts[1] else 1
        weights = np.full(2, 1/ratio); weights[minority] = ratio
    else:
        raise ValueError(family)
    return dict(zip(classes.tolist(), weights.tolist()))

def losses(y, pred):
    return {'error': float(np.mean(pred != y)),
            'balanced': float(1-balanced_accuracy_score(y, pred)),
            'f1': float(1-f1_score(y, pred, average='macro', zero_division=0))}

def vote(classes, margins):
    """Margins positive for second class; exact-zero votes for second."""
    votes = np.zeros((len(margins), len(classes)), dtype=int)
    for col, (i, j) in enumerate(itertools.combinations(range(len(classes)), 2)):
        votes[:, i] += margins[:, col] < 0
        votes[:, j] += margins[:, col] >= 0
    return np.asarray(classes)[np.argmax(votes, axis=1)], votes

def fit_model(X, y, C, alpha, family, ratio=1., force_pairs=False):
    kwargs = dict(C=C, kernel='rbf', gamma=1/(2*alpha**2),
                  decision_function_shape='ovo', shrinking=True, tol=1e-9)
    classes = np.unique(y)
    if family == 'none' and not force_pairs:
        cl = SVC(**kwargs).fit(X, y)
        if cl.fit_status_: raise RuntimeError('native SVC did not converge')
        return {'native': cl, 'classes': classes, 'diagnostics': [{
            'solver': 'libsvm-smo', 'status': int(cl.fit_status_),
            'iterations': cl.n_iter_.tolist(), 'support_vectors': int(cl.support_.size),
            'weights': {str(c): 1. for c in classes}}]}
    models, diagnostics = [], []
    for a, b in itertools.combinations(classes, 2):
        mask = np.isin(y, [a, b]); yy = y[mask]; weights = pair_weights(yy, family, ratio)
        cl = SVC(**kwargs, class_weight=weights).fit(X[mask], yy)
        if cl.fit_status_: raise RuntimeError('pair SVC did not converge')
        models.append(cl)
        diagnostics.append({'pair': [int(a), int(b)], 'counts': {str(c): int((yy==c).sum()) for c in (a,b)},
            'weights': {str(c): w for c,w in weights.items()},
            'penalties': {str(c): C*w for c,w in weights.items()},
            'solver': 'libsvm-smo', 'status': int(cl.fit_status_),
            'iterations': cl.n_iter_.tolist(), 'support_vectors': int(cl.support_.size)})
    return {'models': models, 'classes': classes, 'diagnostics': diagnostics}

def predict_model(model, X):
    if 'native' in model:
        cl = model['native']; margins = cl.decision_function(X)
        margins = margins[:, None] if margins.ndim == 1 else -margins
        _, votes = vote(model['classes'], margins)
        return cl.predict(X), margins, votes
    margins = np.column_stack([cl.decision_function(X) for cl in model['models']])
    pred, votes = vote(model['classes'], margins)
    return pred, margins, votes

def select_family(X, y, transform, seed, family):
    """TRAIN-ONLY API: no outer test inputs exist in this function."""
    start = time.perf_counter()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=20000+int(seed)).split(X,y))
    prepared, fold_records = [], []
    for fi, va in folds:
        A, B, prep = frozen.fit_transform(X[fi], X[va], transform)
        prepared.append((A, B, y[fi], y[va]))
        fold_records.append({'train_indices': fi.tolist(), 'validation_indices': va.tolist(), 'preprocessing': prep})
    records = []
    for ai, rule in enumerate(ALPHA_RULES):
        alphas = [alpha_from_rule(p[0], rule) for p in prepared]
        for ci, C in enumerate(SVC_C_GRID):
            for ri, ratio in enumerate(RATIOS if family=='ratio' else (1.,)):
                details = []
                for fold, (A,B,ya,yb) in enumerate(prepared):
                    cl = fit_model(A, ya, C, alphas[fold], family, ratio)
                    pred, _, _ = predict_model(cl, B)
                    details.append({'fold': fold, 'alpha': alphas[fold], 'losses': losses(yb,pred),
                                    'predictions': pred.tolist(), 'solver_diagnostics': cl['diagnostics']})
                records.append({'alpha_rule': rule, 'C': float(C), 'ratio': ratio,
                    'order': [ai,ci,ri], 'losses': {m: float(np.mean([d['losses'][m] for d in details])) for m in ('error','balanced','f1')},
                    'folds': details})
    return records, fold_records, time.perf_counter()-start

def fit_selected(X, y, Z, transform, family, metric, records, selection_time):
    selected = min(records, key=lambda r: (r['losses'][metric], *r['order']))
    start = time.perf_counter()
    A,B,prep = frozen.fit_transform(X,Z,transform); alpha = alpha_from_rule(A,selected['alpha_rule'])
    cl = fit_model(A,y,selected['C'],alpha,family,selected['ratio'])
    pred,margins,votes = predict_model(cl,B)
    params = {k:selected[k] for k in ('alpha_rule','C','ratio','losses','order')}
    params.update(alpha=alpha, family=family, selection_metric=metric, preprocessing=prep,
                  pairs=cl['diagnostics'], support_vectors=sum(d['support_vectors'] for d in cl['diagnostics']))
    trace = {'margins_positive_second': margins.tolist(), 'votes': votes.tolist(),
             'vote_ties': int(np.sum((votes==votes.max(axis=1)[:,None]).sum(axis=1)>1)),
             'solver_diagnostics': cl['diagnostics']}
    return pred, params, trace, selection_time+time.perf_counter()-start

def evaluate_variants(name, X, y, spec, seed, variants):
    start=time.perf_counter(); tr,te=frozen.split_indices(y,seed)
    records, traces, banks = [], {}, {}
    for variant in variants:
        family,metric=VARIANTS[variant]
        if variant=='v7':
            pred,diag,runtime,params=frozen._v7_fit_predict(X[tr],y[tr],X[te],spec,seed)
            traces[variant]={'solver_diagnostics': diag}
        else:
            if family not in banks:
                inner,folds,elapsed=select_family(X[tr],y[tr],spec['transform'],seed,family)
                banks[family]={'records':inner,'folds':folds,'selection_time_s':elapsed}
            bank=banks[family]
            pred,params,trace,runtime=fit_selected(X[tr],y[tr],X[te],spec['transform'],family,metric,bank['records'],bank['selection_time_s'])
            traces[variant]=trace
        records.append(frozen.metric_record(name,variant,seed,y[te],pred,runtime,params,tr,te))
    return {'records':records,'traces':traces,'banks':banks,
        'split':{'dataset':name,'seed':seed,'train_indices':tr.tolist(),'test_indices':te.tolist(), 'y_test':y[te].tolist()},
        'actual_job_wall_s':time.perf_counter()-start}

def disabled_fit_predict(X,y,Z,spec,seed):
    return frozen._v7_fit_predict(X,y,Z,spec,seed)
