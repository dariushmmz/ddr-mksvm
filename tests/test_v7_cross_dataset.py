import ast, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from ddr_mksvm.v7_cross_dataset import fit_transform,load_dataset,split_indices,_v7_fit_predict

def test_train_only_preprocessing_does_not_use_test_values():
    a=np.array([[0.,2.],[2.,6.]]);b=np.array([[100.,-100.]])
    at,bt,d=fit_transform(a,b,"minmax")
    assert np.allclose(at,[[0,0],[1,1]])
    assert np.allclose(bt,[[50,-25.5]])

def test_sorted_stratified_split_is_deterministic():
    y=np.repeat([0,1],20);a,b=split_indices(y,123);c,d=split_indices(y,123)
    assert np.array_equal(a,c) and np.array_equal(b,d)
    assert np.all(a[:-1]<=a[1:]) and np.all(b[:-1]<=b[1:])

def test_frozen_iris_artifact_hashes_unchanged():
    root=Path("results/iris_experiments/iris-v7-rkhs-l2-final96-20260908")
    expected={"config.json":"62700911bd3a0bd131b622965bed6f9940cf0b5e3a07c4871f3a3baf0223cb89",
              "per_run.csv":"8d0c82a3940b9bd5b8d6aea18baaff5953f21000c6da70cb738d6865996b33bb",
              "splits.json":"80c40f57fa33cb1dd5e625f6852e8b707fd2aef49d728126b1506cb9c775d650"}
    for name,digest in expected.items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest

def test_frozen_v7_seed4000_prediction_replays():
    vals=pd.read_csv("dataset/iris_multiclass.csv").to_numpy(float);X,y=vals[:,:4],vals[:,4].astype(int)
    splits={x["seed"]:x for x in json.loads(Path("results/iris_experiments/iris-v7-rkhs-l2-final96-20260908/splits.json").read_text())["splits"]}
    rows=pd.read_csv("results/iris_experiments/iris-v7-rkhs-l2-final96-20260908/per_run.csv")
    row=rows[(rows.seed==4000)&(rows.architecture=="rkhs_l2_rbf_svc_cv_ovo")].iloc[0];s=splits[4000]
    pred,_,_,_= _v7_fit_predict(X[s["train_indices"]],y[s["train_indices"]],X[s["test_indices"]],{"transform":"none"},4000)
    assert pred.tolist()==ast.literal_eval(row.prediction_vector)
