import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "archive/research_scripts/analyze_r9_gate8_failure.py"
OUTPUT = ROOT / "results/robust/bounded_rbf_redesign/r10_failure_analysis"


def test_r10_analyzer_contains_no_optimizer_or_fit_call():
    source = ANALYZER.read_text(encoding="utf-8")
    assert "linprog(" not in source
    assert "fit_binary(" not in source
    assert "fit_ova(" not in source
    assert "solver_calls\": 0" in source
    assert "new_model_fits\": 0" in source


def test_r10_reconstruction_accounts_for_all_available_primary_pairs():
    provenance = json.loads((OUTPUT / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["paired_seed_count"] == 31
    assert provenance["missing_pair_count"] == 1
    assert provenance["solver_calls"] == provenance["new_model_fits"] == 0
    missing = pd.read_csv(OUTPUT / "missing_pairs.csv")
    assert missing[["dataset", "seed"]].to_dict("records") == [
        {"dataset": "blood_transfusion", "seed": 15305}
    ]


def test_r10_outputs_preserve_per_seed_and_class_diagnostics():
    mechanisms = pd.read_csv(OUTPUT / "paired_seed_mechanisms.csv")
    objectives = pd.read_csv(OUTPUT / "paired_objective_deltas.csv")
    margins = pd.read_csv(OUTPUT / "paired_training_class_deltas.csv")
    flips = pd.read_csv(OUTPUT / "prediction_flips.csv")
    assert len(mechanisms) == 31
    assert set(mechanisms.dataset) == {
        "blood_transfusion", "parkinson", "mammographicmass_binary", "iris"
    }
    assert len(objectives) == 7 + 8 + 8 + 3 * 8
    assert not margins.empty
    assert len(flips[flips.scope == "overall"]) == 31
