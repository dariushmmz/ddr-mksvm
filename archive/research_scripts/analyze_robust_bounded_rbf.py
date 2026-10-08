"""Read and validate a completed R9 exposed qualification directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def analyze(run_dir: Path) -> dict:
    qualification = json.loads((run_dir / "qualification.json").read_text(encoding="utf-8"))
    records = pd.read_csv(run_dir / "per_run.csv")
    calls = pd.read_csv(run_dir / "solver_calls.csv")
    output = {
        **qualification,
        "records": int(len(records)),
        "solver_calls": int(len(calls)),
        "status_failures": int((calls.solver_status_code != 0).sum()),
        "nonfinite_objectives": int((~pd.Series(calls.objective_value).notna()).sum()),
    }
    if output["records"] != qualification["completed_models"]:
        raise RuntimeError("R9 aggregate count mismatch")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    print(json.dumps(analyze(Path(parser.parse_args().run_dir)), indent=2))
