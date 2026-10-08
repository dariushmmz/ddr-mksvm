"""Write the machine-readable inventory used by the expanded dataset study."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import pandas as pd


DECISIONS = {
    "blood_transfusion.csv": ("accepted_existing", "numeric dataset; existing frozen evidence"),
    "breast_cancer_diagnostic.csv": ("accepted_existing", "numeric dataset; existing frozen evidence"),
    "breast_cancer.csv": ("accepted_new", "categorical recurrence cohort; schema repair and train-only one-hot required"),
    "dermatology.csv": ("accepted_extension", "frozen complete-case cohort; add missing V8-C coverage"),
    "heart_disease.csv": ("accepted_existing", "frozen complete-case local five-class protocol"),
    "iris_multiclass.csv": ("accepted_existing", "numeric dataset; existing frozen evidence"),
    "mammographicmass_binary.csv": ("accepted_existing", "fixed local complete-case benchmark; existing evidence"),
    "parkinson.csv": ("accepted_existing", "numeric dataset; existing frozen evidence"),
    "wine.csv": ("accepted_existing", "numeric dataset; existing frozen evidence"),
    "uci_fetch_report.csv": ("rejected_metadata", "acquisition report, not observations and labels"),
}

SEMANTIC_TYPES = {
    "blood_transfusion.csv": (4, 0),
    "breast_cancer_diagnostic.csv": (30, 0),
    "breast_cancer.csv": (0, 9),
    "dermatology.csv": (1, 33),
    "heart_disease.csv": (5, 8),
    "iris_multiclass.csv": (4, 0),
    "mammographicmass_binary.csv": (1, 4),
    "parkinson.csv": (22, 0),
    "wine.csv": (13, 0),
    "uci_fetch_report.csv": (0, 0),
}


def write_inventory(output="results/expanded_dataset_study/inventory"):
    rows = []
    hashes = {}
    for path in sorted(Path("dataset").glob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.name] = digest
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = frame.isin(["", "?"]).sum()
        decision, reason = DECISIONS[path.name]
        if path.name == "uci_fetch_report.csv":
            class_distribution = {}
            task = "metadata"
        else:
            class_distribution = frame.iloc[:, -1].value_counts().sort_index().to_dict()
            task = "binary" if len(class_distribution) == 2 else "multiclass"
        numeric, categorical = SEMANTIC_TYPES[path.name]
        rows.append({
            "file": path.name, "sha256": digest, "samples": len(frame),
            "features": len(frame.columns) - 1 if task != "metadata" else None,
            "task": task, "classes": len(class_distribution) if class_distribution else None,
            "class_distribution": json.dumps(class_distribution, sort_keys=True),
            "missing_values": int(missing.sum()),
            "missing_by_column": json.dumps({key: int(value) for key, value in missing.items() if value}, sort_keys=True),
            "semantic_numeric_columns": numeric, "semantic_categorical_or_ordinal_columns": categorical,
            "decision": decision, "reason": reason,
        })
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "dataset_inventory.csv", index=False)
    (output / "dataset_inventory.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output / "file_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/expanded_dataset_study/inventory")
    write_inventory(parser.parse_args().output)
