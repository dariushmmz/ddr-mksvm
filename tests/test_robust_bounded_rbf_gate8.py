import numpy as np
import pandas as pd

from archive.research_scripts.run_robust_bounded_rbf_gate8 import (
    DATASETS,
    GATE_GUARDS,
    MODELS,
    SEEDS,
    TASKS,
    gate_decision,
    primary_pairs,
)


def synthetic_records():
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for model in MODELS:
                weighted_robust = model == "weighted_robust_rbf_q1"
                weighted_deterministic = model == "weighted_deterministic_rbf_q1"
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "model": model,
                    "test_error": 0.19 if weighted_robust else 0.20,
                    "balanced_accuracy": 0.71 if weighted_robust else 0.70,
                    "macro_f1": 0.71 if weighted_robust else 0.70,
                    "safety_recall": 0.61 if weighted_robust else 0.60,
                    "runtime_s": 1.5 if weighted_robust else 1.0,
                    "prediction_hash": f"{dataset}-{seed}-{model}",
                    "majority_only_prediction": False,
                    "all_solver_calls_accepted": True,
                    "all_calls_max_constraint_violation": 1e-9,
                    "objective_value": 1.0,
                    "tie_count": 0,
                })
    return pd.DataFrame(rows)


def test_gate_registry_is_exact_and_blocks_later_seeds():
    assert SEEDS == tuple(range(15300, 15308))
    assert len(TASKS) == GATE_GUARDS["expected_tasks"] == 128
    assert {task[0] for task in TASKS} == set(DATASETS)
    assert not any(task[1] >= 15400 or task[1] < 15300 for task in TASKS)


def test_primary_pairs_are_weighted_robust_minus_weighted_deterministic():
    pairs = primary_pairs(synthetic_records())
    assert len(pairs) == 32
    np.testing.assert_allclose(pairs.delta_error, -0.01)
    np.testing.assert_allclose(pairs.delta_balanced_accuracy, 0.01)
    np.testing.assert_allclose(pairs.delta_macro_f1, 0.01)
    np.testing.assert_allclose(pairs.delta_safety_recall, 0.01)


def test_prospective_gate_passes_good_fixture_and_rejects_collapse():
    records = synthetic_records()
    pairs = primary_pairs(records)
    guards, _, _ = gate_decision(records, [], pairs)
    assert all(guards.values())
    target = (records.model == "weighted_robust_rbf_q1")
    records.loc[target, "majority_only_prediction"] = True
    guards, _, _ = gate_decision(records, [], primary_pairs(records))
    assert not guards["candidate_no_majority_only"]


def test_feasibility_and_failure_guards_are_fail_closed():
    records = synthetic_records()
    records.loc[records.index[0], "all_calls_max_constraint_violation"] = 2e-7
    guards, _, _ = gate_decision(records, [], primary_pairs(records))
    assert not guards["feasibility"]
    guards, _, _ = gate_decision(records.iloc[:-1], [{"failure": True}], primary_pairs(records.iloc[:-1]))
    assert not guards["all_tasks_completed"]
