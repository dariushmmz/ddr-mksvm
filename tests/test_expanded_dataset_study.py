import hashlib
import json

import numpy as np
import pandas as pd

from ddr_mksvm import expanded_dataset_study as study
import run_expanded_dataset_study as runner


V7_HASH = "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d"
V8_HASH = "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448"


def test_frozen_model_hashes_unchanged():
    for path, expected in (("ddr_mksvm/v7_cross_dataset.py", V7_HASH),
                           ("ddr_mksvm/v8_class_sensitive.py", V8_HASH)):
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == expected


def test_breast_semantic_repair_and_inventory():
    X, y, spec = study.load_dataset("breast_cancer_recurrence")
    assert X.shape == (286, 9)
    assert np.bincount(y).tolist() == [201, 85]
    assert "14-Oct" not in set(X["tumor_size"])
    assert "10-14" in set(X["tumor_size"])
    assert "5-9" in set(X["tumor_size"])
    assert "3-5" in set(X["inv_nodes"])
    assert int((X == "__MISSING__").sum().sum()) == 9
    assert spec["source_profile_only"] is True


def test_encoder_is_training_only_and_unknown_safe():
    train = pd.DataFrame({"a": ["x", "y", "x"], "b": ["p", "p", "q"]})
    test = pd.DataFrame({"a": ["z"], "b": ["q"]})
    A, B, meta = study._encode_scale(train, test)
    assert A.shape == (3, 4)
    assert B.shape == (1, 4)
    assert "z" not in meta["categories"][0]
    assert len(meta["preprocessing_hash"]) == 64
    assert np.isfinite(A).all() and np.isfinite(B).all()


def test_weighting_and_grids_are_frozen():
    assert study.MODELS == ("legacy", "v7", "v8_c")
    assert study.DATASETS["breast_cancer_recurrence"]["kernel"] == "hom_linear"
    assert study.DATASETS["dermatology"]["kernel"] == "inhom_quadratic"
    weights = study.v8.pair_weights(np.array([0, 0, 0, 1]), "sqrt")
    assert abs((3 * weights[0] + weights[1]) / 4 - 1) < 1e-12
    assert list(study.ALPHA_RULES) == list(study.frozen.ALPHA_RULES)
    assert list(study.SVC_C_GRID) == list(study.frozen.SVC_C_GRID)


def test_metric_augmentation_and_paired_direction():
    tr = np.array([0, 0, 0, 1])
    te = np.array([0, 1])
    pred = np.array([0, 0])
    base = study.frozen.metric_record("x", "legacy", 1, te, pred, 1.0,
                                      {"preprocessing": {}}, np.array([0, 1]), np.array([2, 3]))
    row = study._augment_record(base, tr, te, pred)
    assert row["rare_class"] == 1
    assert row["rare_class_recall"] == 0
    assert row["majority_only"] is True
    assert len(row["true_label_hash"]) == 64


def test_preprocessing_definition_hash_is_stable():
    value = study.preprocessing_definition_hash()
    assert len(value) == 64
    assert value == study._canonical_hash(study.PREPROCESSING_DEFINITION)


def test_promotion_csv_empty_cells_become_json_null(tmp_path, monkeypatch):
    source = tmp_path / "gate.csv"
    source.write_text("dataset,comparison,passed,endpoint,runtime_ratio\nx,c,True,test_error,\n", encoding="utf-8")
    args = type("Args", (), {"promotion_source": str(source), "stage": "gate24"})()
    monkeypatch.setattr(runner.study, "load_dataset", lambda name: (None, None, {"name": name}))
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda package: "test")
    monkeypatch.setattr(runner, "digest", lambda path: "0" * 64)
    config, _ = runner._config(args, ["x"], ["legacy", "v7"], [1])
    assert config["promotion_source"]["rows"][0]["runtime_ratio"] is None
    json.dumps(config, allow_nan=False)


def test_model_subset_policy_rejects_v8c_without_reference():
    X, y, spec = study.load_dataset("breast_cancer_recurrence")
    try:
        study.evaluate_seed("breast_cancer_recurrence", X, y, spec, 1, models=("v8_c",))
    except ValueError as exc:
        assert "requires" in str(exc)
    else:
        raise AssertionError("v8_c without v7 must be rejected before fitting")


def test_confirmation_rejects_comparison_not_promoted():
    promotion = {"rows": [
        {"dataset": "x", "comparison": "v7_vs_legacy", "passed": True},
        {"dataset": "x", "comparison": "v8c_vs_v7", "passed": False},
    ]}
    runner._validate_stage_promotion("confirmation", ["x"], ("legacy", "v7"), promotion)
    try:
        runner._validate_stage_promotion("confirmation", ["x"],
                                         ("legacy", "v7", "v8_c"), promotion)
    except ValueError as exc:
        assert "v8c_vs_v7" in str(exc)
    else:
        raise AssertionError("unpromoted V8-C comparison must not enter confirmation")


def test_paired_class_recall_direction_and_matching():
    rows = []
    for seed, legacy, v7 in ((1, {"0": .8, "1": .2}, {"0": .9, "1": .3}),
                             (2, {"0": .7, "1": .4}, {"0": .6, "1": .4})):
        for model, recalls in (("legacy", legacy), ("v7", v7)):
            rows.append({"dataset": "x", "model": model, "seed": seed,
                         "class_recall": json.dumps(recalls), "true_label_hash": str(seed)})
    result = study.paired_class_recall(pd.DataFrame(rows)).set_index("label")
    assert np.isclose(result.loc["1", "mean_delta"], .05)
    assert result.loc["1", "wins"] == 1
    assert result.loc["1", "ties"] == 1
