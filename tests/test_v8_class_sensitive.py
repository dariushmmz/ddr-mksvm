import json
import numpy as np
import pytest
import pandas as pd
from ddr_mksvm import v8_class_sensitive as v8
from ddr_mksvm import v7_cross_dataset as v7
from ddr_mksvm.experiments.class_sensitive_runner import freeze_reference, load

@pytest.mark.parametrize('family',['none','inverse','sqrt'])
def test_weights_normalized(family):
    y=np.array([2]*30+[7]*6);w=v8.pair_weights(y,family)
    assert np.mean([w[c] for c in y])==pytest.approx(1.)
    if family=='inverse':assert w[7]/w[2]==pytest.approx(5.)
    if family=='sqrt':assert w[7]/w[2]==pytest.approx(np.sqrt(5))

def test_ratio_orientation():
    assert v8.pair_weights(np.array([0]*6+[1]*2),'ratio',2)=={0:.5,1:2.}
    assert v8.pair_weights(np.array([0,1]),'ratio',1)=={0:1.,1:1.}

@pytest.mark.parametrize('nclasses',[2,3,5])
def test_manual_neutral_pairs_reduce_to_native(nclasses):
    rng=np.random.default_rng(73);X=rng.normal(size=(nclasses*15,4));y=np.repeat(np.arange(nclasses),15)
    Z=rng.normal(size=(40,4))
    native=v8.fit_model(X,y,1.,1.,'none')
    pair=v8.fit_model(X,y,1.,1.,'none',force_pairs=True)
    pn,mn,vn=v8.predict_model(native,Z);pp,mp,vp=v8.predict_model(pair,Z)
    np.testing.assert_array_equal(pn,pp);np.testing.assert_array_equal(vn,vp)
    np.testing.assert_allclose(mn,mp,atol=2e-6)

def test_vote_ties_and_zero():
    p,v=v8.vote([0,1,2],np.array([[-1,1,-1],[0,0,0]]))
    assert p.tolist()==[0,2];assert v[0].tolist()==[1,1,1]

def test_disabled_exact_and_seeding(monkeypatch):
    rng=np.random.default_rng(8);X=rng.normal(size=(36,3));y=np.repeat([0,1,2],12)
    monkeypatch.setattr(v7,'ALPHA_RULES',('paper',));monkeypatch.setattr(v7,'SVC_C_GRID',(1.,))
    a=v7._v7_fit_predict(X,y,X[:3],{'transform':'none'},9)
    b=v8.disabled_fit_predict(X,y,X[:3],{'transform':'none'},9)
    np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1] and a[3]==b[3]

def test_training_only_selection_and_fold_weights(monkeypatch):
    rng=np.random.default_rng(5);X=rng.normal(size=(36,2));y=np.repeat([0,1],[24,12])
    monkeypatch.setattr(v8,'ALPHA_RULES',('paper',));monkeypatch.setattr(v8,'SVC_C_GRID',(1.,))
    records,folds,_=v8.select_family(X,y,'standardization',4,'inverse')
    for fold,detail in zip(folds,records[0]['folds']):
        fi,va=fold['train_indices'],fold['validation_indices']
        assert not set(fi)&set(va)
        np.testing.assert_allclose(fold['preprocessing']['offset'],X[fi].mean(0))
        weights=detail['solver_diagnostics'][0]['weights']
        assert weights=={str(k):v for k,v in v8.pair_weights(y[fi],'inverse').items()}
    # Outer evaluation values do not change selected rules or fitted transforms.
    a=v8.fit_selected(X,y,np.zeros((2,2)),'standardization','inverse','balanced',records,0)
    b=v8.fit_selected(X,y,np.ones((2,2))*1e8,'standardization','inverse','balanced',records,0)
    assert a[1]==b[1]

def test_weighted_dual_kkt():
    rng=np.random.default_rng(10);X=rng.normal(size=(24,3));y=np.repeat([0,1],[16,8])
    model=v8.fit_model(X,y,1.,1.,'inverse')['models'][0]
    a=np.abs(model.dual_coef_[0]);signed=model.dual_coef_[0]
    weights=v8.pair_weights(y,'inverse');caps=np.array([weights[c] for c in y[model.support_]])
    assert np.all(a<=caps+1e-8);assert abs(signed.sum())<1e-8
    signed_y=2*y-1;margins=signed_y*model.decision_function(X)
    free=(a>1e-7)&(a<caps-1e-7)
    np.testing.assert_allclose(margins[model.support_[free]],1.,atol=2e-6)
    from sklearn.metrics.pairwise import rbf_kernel
    K=rbf_kernel(model.support_vectors_,gamma=.5)
    norm2=float(signed@K@signed)
    primal=.5*norm2+np.dot([weights[c] for c in y],np.maximum(0,1-margins))
    dual=float(a.sum()-.5*norm2)
    assert abs(primal-dual)<2e-6

def test_frozen_reference_and_iris_reconstruction():
    freeze_reference();X,y,inv=load('iris')
    assert inv['numeric_sha256']=='1b3e788db71aac2852e77ce253c00e9cedd173d8b5815b47aa530252a4c99e70'
    assert X.shape==(150,4)

def test_frozen_wine_prediction_replay():
    rows=pd.read_csv('results/v7_cross_dataset/v7-cross-final96-20260909/per_run.csv')
    row=rows[(rows.dataset=='wine')&(rows.model=='v7')&(rows.seed==8000)].iloc[0]
    X,y,spec=load('wine');tr,te=v7.split_indices(y,8000)
    pred,_,_,params=v8.disabled_fit_predict(X[tr],y[tr],X[te],spec,8000)
    assert pred.tolist()==json.loads(row.prediction_vector)
    saved=json.loads(row.selected_hyperparameters)
    assert params['C']==saved['C'] and params['alpha_rule']==saved['alpha_rule']

def test_unweighted_selection_bank_matches_frozen(monkeypatch):
    rng=np.random.default_rng(100);X=rng.normal(size=(45,3));y=np.repeat([1,2,3],15)
    for module in (v7,v8):
        monkeypatch.setattr(module,'ALPHA_RULES',('paper','med:0'))
        monkeypatch.setattr(module,'SVC_C_GRID',(.1,1.))
    pred,_,_,params=v7._v7_fit_predict(X,y,X[:5],{'transform':'standardization'},7)
    records,_,elapsed=v8.select_family(X,y,'standardization',7,'none')
    pp,qq,_,_=v8.fit_selected(X,y,X[:5],'standardization','none','error',records,elapsed)
    np.testing.assert_array_equal(pred,pp)
    assert (params['alpha_rule'],params['C'])==(qq['alpha_rule'],qq['C'])

def test_checkpoint_reuse_and_incompatible_resume(tmp_path):
    from ddr_mksvm.experiments.class_sensitive_runner import run_job,json_write
    prior=tmp_path/'prior';(prior/'checkpoints').mkdir(parents=True)
    cache=tmp_path/'current';cache.mkdir()
    record={'fingerprint':'old','variants':['v7','v8_b'],'app_id':'old-app',
            'records':[{'model':'v7'},{'model':'v8_b'}],
            'traces':{'v7':{},'v8_b':{}},'actual_job_wall_s':12.}
    json_write(prior/'checkpoints'/'synthetic-1.json',record)
    result=run_job('synthetic',1,['v7'],cache,'new',prior)
    assert result['new_compute_wall_s']==0 and result['records']==[{'model':'v7'}]
    assert result['reused_from']['fingerprint']=='old'
    assert run_job('synthetic',1,['v7'],cache,'new')==result
    with pytest.raises(AssertionError):run_job('synthetic',1,['v7'],cache,'incompatible')
