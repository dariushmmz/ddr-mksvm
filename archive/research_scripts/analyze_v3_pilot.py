"""Deterministic statistical summary for a downloaded V3 Modal pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATASETS = ("parkinson", "blood_transfusion", "mammographicmass_binary", "iris")
COMPARISONS = (
    ("legacy", "current_v2"),
    ("current_v2", "residual_v3"),
    ("legacy", "residual_v3"),
)


def _mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    sem = stats.sem(values)
    if sem == 0:
        return float(values.mean()), float(values.mean())
    half = stats.t.ppf((1 + confidence) / 2, values.size - 1) * sem
    return float(values.mean() - half), float(values.mean() + half)


def _summary(group):
    errors = group["test_error"].dropna().to_numpy(float)
    ci_low, ci_high = _mean_ci(errors)
    return pd.Series({
        "attempted": len(group),
        "completed": len(errors),
        "failures": int((group["status"] != "ok").sum()),
        "mean_error": np.mean(errors) if len(errors) else np.nan,
        "std_error": np.std(errors, ddof=1) if len(errors) > 1 else np.nan,
        "median_error": np.median(errors) if len(errors) else np.nan,
        "error_ci95_low": ci_low,
        "error_ci95_high": ci_high,
        "mean_runtime_s": group["runtime_s"].mean(),
        "total_runtime_s": group["runtime_s"].sum(),
        "planned_solver_calls": group["planned_solver_calls"].sum(),
        "completed_solver_calls": group["completed_solver_calls"].sum(min_count=1),
    })


def _paired_table(per_run):
    rows = []
    ok = per_run.loc[per_run["status"] == "ok"]
    for dataset_scope in ("all", *DATASETS):
        scoped = ok if dataset_scope == "all" else ok.loc[ok["dataset"] == dataset_scope]
        for reference, candidate in COMPARISONS:
            left = scoped.loc[scoped["model"] == reference, ["dataset", "seed", "test_error", "runtime_s"]]
            right = scoped.loc[scoped["model"] == candidate, ["dataset", "seed", "test_error", "runtime_s"]]
            pairs = left.merge(right, on=["dataset", "seed"], suffixes=("_reference", "_candidate"))
            delta = pairs["test_error_candidate"] - pairs["test_error_reference"]
            ci_low, ci_high = _mean_ci(delta)
            rows.append({
                "scope": dataset_scope,
                "reference": reference,
                "candidate": candidate,
                "n_pairs": len(pairs),
                "mean_error_delta": delta.mean(),
                "std_error_delta": delta.std(ddof=1),
                "median_error_delta": delta.median(),
                "error_delta_ci95_low": ci_low,
                "error_delta_ci95_high": ci_high,
                "relative_mean_error_delta": (
                    delta.mean() / pairs["test_error_reference"].mean()
                    if len(pairs) and pairs["test_error_reference"].mean() != 0 else np.nan
                ),
                "candidate_wins": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "candidate_losses": int((delta > 0).sum()),
                "mean_runtime_ratio": (
                    (pairs["runtime_s_candidate"] / pairs["runtime_s_reference"]).mean()
                    if len(pairs) else np.nan
                ),
                "runtime_ratio_of_totals": (
                    pairs["runtime_s_candidate"].sum() / pairs["runtime_s_reference"].sum()
                    if len(pairs) and pairs["runtime_s_reference"].sum() != 0 else np.nan
                ),
            })
    return pd.DataFrame(rows)


def analyze(input_root: Path, output_dir: Path):
    frames = []
    for dataset in DATASETS:
        path = input_root / dataset / "per_run.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing completed pilot artifact: {path}")
        frame = pd.read_csv(path)
        if set(frame["dataset"]) != {dataset}:
            raise ValueError(f"unexpected dataset values in {path}")
        frames.append(frame)
    per_run = pd.concat(frames, ignore_index=True).sort_values(["dataset", "seed", "model"])
    expected = len(DATASETS) * 4 * 3
    if len(per_run) != expected or per_run[["dataset", "seed", "model"]].duplicated().any():
        raise ValueError(f"pilot schema is not the expected {expected} unique matched runs")

    dataset_summary = per_run.groupby(["dataset", "model"], sort=False).apply(
        _summary, include_groups=False).reset_index()
    model_summary = per_run.groupby("model", sort=False).apply(
        _summary, include_groups=False).reset_index()
    paired = _paired_table(per_run)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output_dir / "per_run_combined.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)

    totals = {
        "runs": int(len(per_run)),
        "failures": int((per_run["status"] != "ok").sum()),
        "planned_solver_calls": int(per_run["planned_solver_calls"].sum()),
        "completed_solver_calls": int(per_run["completed_solver_calls"].sum()),
        "aggregate_worker_runtime_s": float(per_run["runtime_s"].sum()),
        "aggregate_worker_runtime_hours": float(per_run["runtime_s"].sum() / 3600),
    }
    (output_dir / "compute_summary.json").write_text(
        json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    print(model_summary.to_string(index=False))
    print("\nPaired comparisons (candidate - reference):")
    print(paired.loc[paired["scope"] == "all"].to_string(index=False))
    print("\nCompute:", json.dumps(totals, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/modal/pilot-v3-4seed"))
    parser.add_argument("--output", type=Path, default=Path("results/v3_pilot_analysis"))
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
