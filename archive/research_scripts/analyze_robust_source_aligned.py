"""Rebuild and print the R8 exposed-seed qualification summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

from archive.research_scripts.run_robust_source_aligned import aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = aggregate(run_dir)
    columns = [
        "dataset",
        "seed",
        "model",
        "test_error",
        "balanced_accuracy",
        "macro_f1",
        "minority_recall",
        "selected_nu",
        "objective_value",
        "all_calls_max_constraint_violation",
        "runtime_s",
        "prediction_hash",
    ]
    print(output["records"][columns].sort_values(["dataset", "seed", "model"]).to_string(index=False))
    print(output["qualification"]["decision"])


if __name__ == "__main__":
    main()
