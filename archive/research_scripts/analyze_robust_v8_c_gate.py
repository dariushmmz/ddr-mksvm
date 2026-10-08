"""Paired gate statistics for a complete or explicitly partial Robust V8-C run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = {
    "test_error": -1.0,
    "balanced_accuracy": 1.0,
    "macro_f1": 1.0,
    "runtime_s": -1.0,
}


def paired_stats(values: np.ndarray, favorable_sign: float) -> dict:
    values = np.asarray(values, float)
    n = len(values)
    mean = float(np.mean(values)) if n else math.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        low, high = mean - half, mean + half
    else:
        low = high = math.nan
    favorable = favorable_sign * values
    wins, ties, losses = (int(np.sum(favorable > 1e-12)),
                           int(np.sum(np.abs(favorable) <= 1e-12)),
                           int(np.sum(favorable < -1e-12)))
    nonzero = favorable[np.abs(favorable) > 1e-12]
    if len(nonzero):
        wilcoxon = stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
        ranks = stats.rankdata(np.abs(nonzero))
        rank_biserial = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
        p_value = float(wilcoxon.pvalue)
    else:
        rank_biserial, p_value = 0.0, 1.0
    dz = mean / sd if n > 1 and sd > 0 else math.nan
    return {"n": n, "mean_difference": mean, "ci95_low": low, "ci95_high": high,
            "wins": wins, "ties": ties, "losses": losses,
            "wilcoxon_p": p_value, "cohen_dz": dz,
            "rank_biserial_favorable": rank_biserial}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir")
    parser.add_argument("--expected-datasets", required=True)
    parser.add_argument("--expected-seeds", type=int, default=24)
    args = parser.parse_args()
    root = Path(args.analysis_dir)
    frame = pd.read_csv(root / "per_run.csv")
    diagnostics = pd.read_csv(root / "solver_diagnostics.csv")
    expected = [value for value in args.expected_datasets.split(",") if value]
    completion_rows, paired_rows, class_rows = [], [], []
    for dataset in expected:
        group = frame[frame.dataset == dataset]
        base = group[group.model == "v8_c"].set_index("seed")
        robust = group[group.model == "robust_v8_c_p2"].set_index("seed")
        matched = sorted(set(base.index) & set(robust.index))
        completion_rows.append({"dataset": dataset, "v8_c_seeds": base.index.nunique(),
                                "robust_seeds": robust.index.nunique(), "matched_seeds": len(matched),
                                "expected_seeds": args.expected_seeds,
                                "complete": len(matched) == args.expected_seeds})
        if not matched:
            continue
        for metric, favorable_sign in METRICS.items():
            difference = robust.loc[matched, metric].to_numpy(float) - base.loc[matched, metric].to_numpy(float)
            paired_rows.append({"dataset": dataset, "metric": metric,
                                "v8_c_mean": float(base.loc[matched, metric].mean()),
                                "robust_mean": float(robust.loc[matched, metric].mean()),
                                **paired_stats(difference, favorable_sign)})
        labels = sorted(set().union(*(json.loads(value).keys() for value in base.loc[matched, "per_class_recall"])))
        for label in labels:
            before = np.asarray([json.loads(base.loc[seed, "per_class_recall"])[label] for seed in matched])
            after = np.asarray([json.loads(robust.loc[seed, "per_class_recall"])[label] for seed in matched])
            class_rows.append({"dataset": dataset, "class": label,
                               "v8_c_mean": float(before.mean()), "robust_mean": float(after.mean()),
                               **paired_stats(after - before, 1.0)})
    completion = pd.DataFrame(completion_rows)
    paired = pd.DataFrame(paired_rows)
    per_class = pd.DataFrame(class_rows)
    robust_diag = diagnostics[diagnostics.model == "robust_v8_c_p2"].copy()
    if robust_diag.empty:
        statuses = feasibility = pd.DataFrame()
    else:
        statuses = robust_diag.groupby(["dataset", "phase", "solver_status"], dropna=False).size().rename("count").reset_index()
        totals = robust_diag.groupby(["dataset", "phase"]).size().rename("total").reset_index()
        statuses = statuses.merge(totals, on=["dataset", "phase"])
        statuses["frequency"] = statuses["count"] / statuses["total"]
        feasibility = robust_diag.groupby(["dataset", "phase"]).agg(
            solves=("solver_status", "size"),
            max_constraint_violation=("max_constraint_violation", "max"),
            p95_constraint_violation=("max_constraint_violation", lambda x: float(np.quantile(x, .95))),
            mean_constraint_violation=("max_constraint_violation", "mean"),
            min_margin_residual=("min_margin_residual", "min"),
            min_cone_residual=("cone_residual", "min"),
            max_delta=("max_delta", "max"),
            mean_solver_iterations=("solver_iterations", "mean"),
            max_solver_iterations=("solver_iterations", "max"),
        ).reset_index()
    completion.to_csv(root / "gate_completion.csv", index=False)
    paired.to_csv(root / "gate_paired_statistics.csv", index=False)
    per_class.to_csv(root / "gate_per_class_statistics.csv", index=False)
    statuses.to_csv(root / "gate_solver_status.csv", index=False)
    feasibility.to_csv(root / "gate_feasibility.csv", index=False)
    verdict = {"expected_datasets": expected, "expected_seeds": args.expected_seeds,
               "all_matched_complete": bool(completion.complete.all()),
               "freeze_justified": False,
               "reason": "Incomplete gate; repeated CLARABEL failure on Blood robust tasks."
                         if not completion.complete.all() else "Statistical gate criteria require explicit review."}
    (root / "gate_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(completion.to_string(index=False))
    if not paired.empty:
        print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
