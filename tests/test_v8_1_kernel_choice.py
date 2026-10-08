import json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from ddr_mksvm import v8_1_kernel_choice as new
from ddr_mksvm import v8_class_sensitive as frozen
from archive.research_scripts.v8_1_provenance import verify_baselines,digest


def test_frozen_baselines_and_prospective_docs():
    assert verify_baselines()>=760
    design=json.loads(Path('results/v8_1/analysis/design_freeze.json').read_text())
    for path,expected in design['document_hashes'].items():assert digest(path)==expected


def test_quadratic_psd_normalization_and_solver_equivalence():
    rng=np.random.default_rng(73);X=rng.normal(size=(30,3));y=np.repeat([0,1],[20,10]);Z=rng.normal(size=(12,3))
    q=new.quadratic_scale(X);K=(1+X@X.T)**2/q
    assert abs(np.diag(K).mean()-1)<1e-14
    assert np.linalg.eigvalsh(K).min()>-1e-10
    model=new.fit_quadratic(X,y,1.)
    direct=SVC(C=1.,kernel='precomputed',class_weight=frozen.pair_weights(y,'sqrt'),tol=1e-9).fit(K,y)
    pred,margins,_=frozen.predict_model(model,Z)
    test=(1+Z@X.T)**2/q
    np.testing.assert_array_equal(pred,direct.predict(test))
    np.testing.assert_allclose(margins[:,0],direct.decision_function(test),atol=2e-6)


def test_selector_ties_and_disabled_exact():
    r=[dict(losses={'balanced':.2},order=[0,0,0]),dict(losses={'balanced':.1},order=[1,0,0])]
    q=[dict(losses={'balanced':.1},order=0)]
    assert new.choose(r,q)==('rbf',r[1])
    q[0]['losses']['balanced']=.01
    assert new.choose(r,q)==('quadratic',q[0])
    assert new.choose(r,q,include_quadratic=False)==('rbf',r[1])


def test_bank_transform_and_kernel_scale_are_fold_train_only(monkeypatch):
    monkeypatch.setattr(new,'SVC_C_GRID',(1.,))
    rng=np.random.default_rng(32);X=rng.normal(size=(30,3));y=np.repeat([0,1],[18,12])
    bank=new.quadratic_bank(X,y,'standardization',11000)
    for f,d in zip(bank['folds'],bank['records'][0]['folds']):
        fi=np.asarray(f['train_indices']);va=f['validation_indices']
        assert not set(fi)&set(va)
        np.testing.assert_allclose(f['preprocessing']['offset'],X[fi].mean(0))
        A=(X[fi]-X[fi].mean(0))/X[fi].std(0)
        assert abs(new.quadratic_scale(A)-d['quadratic_scale'])<1e-12


def test_margin_target_reparameterization_algebra():
    r,t=1.4,.3;y=np.asarray([-1,1]);g=np.asarray([-.4,.7]);slack=np.asarray([.8,.8])
    targets=np.where(y==1,r+t,r-t);original=y*(r*g+t)-targets+r*slack
    np.testing.assert_allclose(original,r*(y*g-1+slack))
