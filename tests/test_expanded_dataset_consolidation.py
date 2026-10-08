import hashlib

from consolidate_expanded_dataset_study import (
    confirmation_frames,
    model_summary,
    paired_statistics,
    stage_registry,
)


def test_all_usable_datasets_have_confirmation_evidence():
    legacy_v7, v8, _ = confirmation_frames()
    expected = {
        "parkinson", "blood_transfusion", "mammographicmass_binary",
        "breast_cancer_diagnostic", "breast_cancer_recurrence", "wine",
        "heart_disease", "dermatology", "iris",
    }
    assert set(legacy_v7.dataset) | set(v8.dataset) == expected
    assert set(legacy_v7.groupby(["dataset", "model"]).size()) == {96}
    assert set(v8.groupby(["dataset", "model"]).size()) == {96}


def test_comparison_registries_are_not_pooled():
    legacy_v7, v8, _ = confirmation_frames()
    left = model_summary(legacy_v7, "legacy_vs_v7")
    right = model_summary(v8, "v7_vs_v8_c")
    assert len(left) == 14
    assert len(right) == 14
    assert set(left.comparison) == {"legacy_vs_v7"}
    assert set(right.comparison) == {"v7_vs_v8_c"}


def test_key_confirmation_effects_and_stage_pruning():
    legacy_v7, v8, _ = confirmation_frames()
    p1, _ = paired_statistics(legacy_v7, "legacy", "v7")
    p2, _ = paired_statistics(v8, "v7", "v8_c")
    parkinson = p1[(p1.dataset == "parkinson") & (p1.metric == "test_error")].iloc[0]
    blood = p2[(p2.dataset == "blood_transfusion") &
               (p2.metric == "balanced_accuracy")].iloc[0]
    assert abs(parkinson.mean_delta + 0.07164115646258502) < 1e-15
    assert abs(blood.mean_delta - 0.0582048677229135) < 1e-15
    registry = stage_registry()
    pruned = registry[(registry.comparison == "legacy_vs_v7") &
                      (registry.decision == "pruned_negative")]
    assert set(pruned.dataset) == {"blood_transfusion", "mammographicmass_binary"}


def test_frozen_model_hashes_remain_unchanged():
    expected = {
        "ddr_mksvm/v7_cross_dataset.py": "39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d",
        "ddr_mksvm/v8_class_sensitive.py": "fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448",
    }
    for path, value in expected.items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == value
