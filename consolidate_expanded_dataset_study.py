"""Consolidate immutable Legacy/V7/V8-C campaigns into paper-ready tables.

This module is analysis-only: it performs no model fitting and never mutates
the source experiment directories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon


ROOT = Path("results/expanded_dataset_study/analysis/cross_dataset")
V7_FINAL = Path("results/v7_cross_dataset/v7-cross-final96-20260909")
IRIS_FINAL = Path("results/iris_experiments/iris-v7-rkhs-l2-final96-20260908")
V8_FINAL = Path("results/v8/final96/v8-final96-20260909")
EXPANDED_FINAL = Path(
    "results/expanded_dataset_study/confirmation/expanded-confirmation-16300-16395-v1"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal_frame(path: Path, source_run: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "per_run.csv")
    frame["source_run"] = source_run
    return frame


def _iris_frame() -> pd.DataFrame:
    raw = pd.read_csv(IRIS_FINAL / "per_run.csv")
    raw = raw[raw.architecture.isin(
        ["matlab_source_legacy_highs", "rkhs_l2_rbf_svc_cv_ovo"]
    )].copy()
    raw["dataset"] = "iris"
    raw["model"] = raw.architecture.map({
        "matlab_source_legacy_highs": "legacy",
        "rkhs_l2_rbf_svc_cv_ovo": "v7",
    })
    split_data = json.loads((IRIS_FINAL / "splits.json").read_text(encoding="utf-8"))["splits"]
    counts = {int(row["seed"]): row["test_class_counts"] for row in split_data}
    indices = {int(row["seed"]): row["test_indices"] for row in split_data}
    iris_y = pd.read_csv("dataset/iris_multiclass.csv").iloc[:, -1].to_numpy(int)
    raw["labels"] = json.dumps([1, 2, 3])
    raw["class_recall"] = raw.apply(
        lambda row: json.dumps({str(i): float(row[f"recall_class_{i}"]) for i in (1, 2, 3)}), axis=1
    )
    raw["class_support"] = raw.seed.map(lambda seed: json.dumps(counts[int(seed)]))
    def iris_confusion(row):
        truth = iris_y[np.asarray(indices[int(row.seed)], int)]
        prediction = np.asarray(json.loads(row.prediction_vector), int)
        matrix = np.zeros((3, 3), dtype=int)
        for actual, predicted in zip(truth, prediction):
            matrix[int(actual) - 1, int(predicted) - 1] += 1
        return json.dumps(matrix.tolist())
    raw["confusion_matrix"] = raw.apply(iris_confusion, axis=1)
    raw["selected_hyperparameters"] = raw.hyperparameters
    raw["source_run"] = str(IRIS_FINAL).replace("\\", "/")
    return raw


def confirmation_frames():
    v7 = _normal_frame(V7_FINAL, str(V7_FINAL).replace("\\", "/"))
    # The expanded confirmation is the prospectively newest Dermatology
    # estimate; the original final96 remains a preserved replication.
    dermatology_replication = v7[v7.dataset == "dermatology"].copy()
    v7 = v7[v7.dataset != "dermatology"]
    expanded = _normal_frame(EXPANDED_FINAL, str(EXPANDED_FINAL).replace("\\", "/"))
    legacy_v7 = pd.concat([v7, _iris_frame(), expanded], ignore_index=True, sort=False)
    v8 = _normal_frame(V8_FINAL, str(V8_FINAL).replace("\\", "/"))
    return legacy_v7, v8, dermatology_replication


def _paired_row(dataset, reference, candidate, metric, a, b, source_run):
    delta = np.asarray(b, float) - np.asarray(a, float)
    n = len(delta)
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    favorable = -delta if metric in ("test_error", "runtime_s") else delta
    nonzero = delta[np.abs(delta) > 1e-14]
    return {
        "dataset": dataset, "reference": reference, "candidate": candidate,
        "metric": metric, "seeds": n, "mean_reference": float(np.mean(a)),
        "mean_candidate": float(np.mean(b)), "mean_delta": float(delta.mean()),
        "ci_low": float(delta.mean() - half), "ci_high": float(delta.mean() + half),
        "wins": int((favorable > 1e-14).sum()),
        "ties": int((np.abs(favorable) <= 1e-14).sum()),
        "losses": int((favorable < -1e-14).sum()),
        "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
        "standardized_paired_effect": float(delta.mean() / sd) if sd else np.nan,
        "runtime_ratio": np.nan, "source_run": source_run,
    }


def paired_statistics(frame: pd.DataFrame, reference: str, candidate: str):
    rows, class_rows = [], []
    for dataset, group in frame.groupby("dataset"):
        a = group[group.model == reference].set_index("seed").sort_index()
        b = group[group.model == candidate].set_index("seed").sort_index()
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        if len(common) != 96:
            raise AssertionError(f"{dataset}: expected 96 matched seeds, found {len(common)}")
        for field in ("train_indices_hash", "test_indices_hash"):
            if field in a and field in b and not (a[field] == b[field]).all():
                raise AssertionError(f"{dataset}: unmatched {field}")
        source_run = str(a.iloc[0].source_run)
        ratio = float(b.runtime_s.sum() / a.runtime_s.sum())
        for metric in ("test_error", "balanced_accuracy", "macro_f1", "runtime_s"):
            row = _paired_row(dataset, reference, candidate, metric,
                              a[metric].to_numpy(float), b[metric].to_numpy(float), source_run)
            row["runtime_ratio"] = ratio
            rows.append(row)
        labels = sorted({key for value in list(a.class_recall) + list(b.class_recall)
                         for key in json.loads(value)}, key=lambda x: float(x))
        support = {label: sum(json.loads(value).get(label, 0) for value in a.class_support)
                   for label in labels}
        rare = min(labels, key=lambda label: (support[label], label))
        for label in labels:
            ar = np.asarray([json.loads(value)[label] for value in a.class_recall], float)
            br = np.asarray([json.loads(value)[label] for value in b.class_recall], float)
            row = _paired_row(dataset, reference, candidate, "class_recall", ar, br, source_run)
            row.update({"label": label, "support": support[label], "is_rare_class": label == rare})
            row.pop("runtime_ratio")
            class_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(class_rows)


def model_summary(frame: pd.DataFrame, comparison: str):
    rows = []
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        labels = sorted({key for value in group.class_recall for key in json.loads(value)},
                        key=lambda x: float(x))
        supports = {label: sum(json.loads(value).get(label, 0) for value in group.class_support)
                    for label in labels}
        recalls = {label: float(np.average(
            [json.loads(value)[label] for value in group.class_recall],
            weights=[json.loads(value).get(label, 0) for value in group.class_support]
        )) for label in labels}
        matrices = [value for value in group.confusion_matrix.astype(str) if value and value != "nan"]
        aggregate_matrix = ""
        if matrices:
            aggregate_matrix = json.dumps(sum((np.asarray(json.loads(v), int) for v in matrices)).tolist())
        rare = min(labels, key=lambda label: (supports[label], label))
        rows.append({
            "dataset": dataset, "comparison": comparison, "model": model, "seeds": len(group),
            "mean_test_error": group.test_error.mean(),
            "mean_balanced_accuracy": group.balanced_accuracy.mean(),
            "mean_macro_f1": group.macro_f1.mean(),
            "mean_runtime_s": group.runtime_s.mean(), "total_runtime_s": group.runtime_s.sum(),
            "rare_class": rare, "rare_class_recall": recalls[rare],
            "class_recall": json.dumps(recalls, sort_keys=True),
            "class_support": json.dumps(supports, sort_keys=True),
            "aggregate_confusion_matrix": aggregate_matrix,
            "source_run": group.iloc[0].source_run,
        })
    return pd.DataFrame(rows)


def hyperparameter_frequency(frame: pd.DataFrame, comparison: str):
    rows = []
    for (dataset, model, value), group in frame.groupby(
        ["dataset", "model", "selected_hyperparameters"], dropna=False
    ):
        rows.append({"dataset": dataset, "comparison": comparison, "model": model,
                     "selected_hyperparameters": value, "count": len(group),
                     "source_run": group.iloc[0].source_run})
    return pd.DataFrame(rows)


def stage_registry():
    return pd.DataFrame([
        # Legacy versus V7 historical campaign.
        ("parkinson", "legacy_vs_v7", "smoke/gate8/gate24/confirmation96", "confirmed", "7000;7000-7007;7000-7023;8000-8095"),
        ("blood_transfusion", "legacy_vs_v7", "smoke/gate8", "pruned_negative", "7000;7000-7007"),
        ("mammographicmass_binary", "legacy_vs_v7", "smoke/gate8", "pruned_negative", "7000;7000-7007"),
        ("breast_cancer_diagnostic", "legacy_vs_v7", "smoke/gate8/gate24/confirmation96", "confirmed", "7000;7000-7007;7000-7023;8000-8095"),
        ("wine", "legacy_vs_v7", "smoke/gate8/gate24/confirmation96", "confirmed", "7000;7000-7007;7000-7023;8000-8095"),
        ("heart_disease", "legacy_vs_v7", "smoke/gate8/gate24/confirmation96", "confirmed", "7000;7000-7007;7000-7023;8000-8095"),
        ("dermatology", "legacy_vs_v7", "smoke/gate8/gate24/confirmation96 + expanded replication", "confirmed_twice", "7000;7000-7007;7000-7023;8000-8095 + 16000/16100-16107/16200-16223/16300-16395"),
        ("iris", "legacy_vs_v7", "smoke/gate8/gate24/confirmation96", "confirmed", "0;0-7;0-23;4000-4095"),
        ("breast_cancer_recurrence", "legacy_vs_v7", "smoke/gate8/gate24/confirmation96", "confirmed", "16000;16100-16107;16200-16223;16300-16395"),
        # V8-C campaign. Seven datasets have completed final96; expanded controls stopped prospectively.
        ("blood_transfusion", "v7_vs_v8_c", "smoke/gate8/gate24/confirmation96", "confirmed", "9000;9000-9007;9000-9023;10000-10095"),
        ("mammographicmass_binary", "v7_vs_v8_c", "smoke/gate8/gate24/confirmation96", "confirmed_null", "9000;9000-9007;9000-9023;10000-10095"),
        ("heart_disease", "v7_vs_v8_c", "smoke/gate8/gate24/confirmation96", "confirmed_tradeoff", "9000;9000-9007;9000-9023;10000-10095"),
        ("parkinson", "v7_vs_v8_c", "smoke/gate8/gate24/confirmation96", "confirmed_control", "9000;9000-9007;9000-9023;10000-10095"),
        ("wine", "v7_vs_v8_c", "gate8/gate24/confirmation96", "confirmed_control", "9000-9007;9000-9023;10000-10095"),
        ("iris", "v7_vs_v8_c", "gate24/confirmation96", "confirmed_control", "9000-9023;10000-10095"),
        ("breast_cancer_diagnostic", "v7_vs_v8_c", "gate24/confirmation96", "confirmed_control", "9000-9023;10000-10095"),
        ("dermatology", "v7_vs_v8_c", "expanded smoke/gate8/gate24", "not_promoted", "16000;16100-16107;16200-16223"),
        ("breast_cancer_recurrence", "v7_vs_v8_c", "expanded smoke/gate8/gate24", "not_promoted", "16000;16100-16107;16200-16223"),
    ], columns=["dataset", "comparison", "completed_stages", "decision", "seed_registry"])


def solver_status_summary():
    rows = []
    for root, comparison in ((V7_FINAL, "legacy_vs_v7"),
                             (EXPANDED_FINAL, "legacy_vs_v7")):
        diagnostics = pd.read_csv(root / "solver_diagnostics.csv", low_memory=False)
        grouped = diagnostics.groupby(
            ["dataset", "model", "solver", "solver_status"], dropna=False
        ).size().reset_index(name="records")
        for row in grouped.to_dict(orient="records"):
            row.update({"comparison": comparison, "record_scope": "all logged inner/outer calls",
                        "failures": 0, "source_run": str(root).replace("\\", "/")})
            rows.append(row)
    iris = pd.read_csv(IRIS_FINAL / "per_run.csv")
    iris = iris[iris.architecture.isin(
        ["matlab_source_legacy_highs", "rkhs_l2_rbf_svc_cv_ovo"]
    )].copy()
    iris["model"] = iris.architecture.map({
        "matlab_source_legacy_highs": "legacy", "rkhs_l2_rbf_svc_cv_ovo": "v7"
    })
    for row in iris.groupby(["model", "solver_backend", "solver_status"]).size().reset_index(name="records").to_dict(orient="records"):
        rows.append({"dataset": "iris", "model": row["model"],
                     "solver": row["solver_backend"], "solver_status": row["solver_status"],
                     "records": row["records"], "comparison": "legacy_vs_v7",
                     "record_scope": "selected outer model records", "failures": 0,
                     "source_run": str(IRIS_FINAL).replace("\\", "/")})
    trace_counts = {}
    for path in (V8_FINAL / "checkpoints").glob("*.json"):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        dataset = checkpoint["records"][0]["dataset"]
        for model, trace in checkpoint["traces"].items():
            diagnostics = trace["solver_diagnostics"]
            for diagnostic in diagnostics:
                solver = diagnostic.get("solver", "unknown")
                status = diagnostic.get("solver_status", diagnostic.get("status"))
                key = (dataset, model, solver, str(status))
                trace_counts[key] = trace_counts.get(key, 0) + 1
    for (dataset, model, solver, status), count in sorted(trace_counts.items()):
        rows.append({"dataset": dataset, "model": model, "solver": solver,
                     "solver_status": status, "records": count,
                     "comparison": "v7_vs_v8_c", "record_scope": "checkpoint trace diagnostics",
                     "failures": 0, "source_run": str(V8_FINAL).replace("\\", "/")})
    return pd.DataFrame(rows)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    legacy_v7, v8, derm_replication = confirmation_frames()
    p1, c1 = paired_statistics(legacy_v7, "legacy", "v7")
    p2, c2 = paired_statistics(v8, "v7", "v8_c")
    p1["source_report_wilcoxon_p"] = np.nan
    p1["source_report_artifact"] = ""
    old = pd.read_csv(V7_FINAL / "paired_comparisons.csv")
    for row in old.itertuples():
        mask = (p1.dataset == row.dataset) & (p1.metric == "test_error")
        p1.loc[mask, ["source_report_wilcoxon_p", "source_report_artifact"]] = [
            row.wilcoxon_p, str(V7_FINAL / "paired_comparisons.csv").replace("\\", "/")]
    iris_source = pd.read_csv(IRIS_FINAL / "paired_comparisons.csv")
    iris_row = iris_source[(iris_source.reference == "matlab_source_legacy_highs") &
                           (iris_source.candidate == "rkhs_l2_rbf_svc_cv_ovo")].iloc[0]
    p1.loc[(p1.dataset == "iris") & (p1.metric == "test_error"),
           ["source_report_wilcoxon_p", "source_report_artifact"]] = [
        iris_row.wilcoxon_p, str(IRIS_FINAL / "paired_comparisons.csv").replace("\\", "/")]
    expanded_source = pd.read_csv(EXPANDED_FINAL / "analysis" / "paired_metrics.csv")
    for row in expanded_source.itertuples():
        mask = (p1.dataset == row.dataset) & (p1.metric == row.metric)
        p1.loc[mask, ["source_report_wilcoxon_p", "source_report_artifact"]] = [
            row.wilcoxon_p, str(EXPANDED_FINAL / "analysis" / "paired_metrics.csv").replace("\\", "/")]
    v8_source = pd.read_csv(V8_FINAL / "paired_comparisons.csv")
    p2["source_report_wilcoxon_p"] = np.nan
    p2["source_report_artifact"] = ""
    for row in v8_source.itertuples():
        mask = (p2.dataset == row.dataset) & (p2.metric == row.metric)
        p2.loc[mask, ["source_report_wilcoxon_p", "source_report_artifact"]] = [
            row.wilcoxon_p, str(V8_FINAL / "paired_comparisons.csv").replace("\\", "/")]
    summaries = pd.concat([
        model_summary(legacy_v7, "legacy_vs_v7"),
        model_summary(v8, "v7_vs_v8_c"),
    ], ignore_index=True)
    summaries.to_csv(ROOT / "confirmed_model_summary.csv", index=False)
    pd.concat([p1, p2], ignore_index=True).to_csv(ROOT / "confirmed_paired_statistics.csv", index=False)
    pd.concat([c1, c2], ignore_index=True).to_csv(ROOT / "confirmed_paired_class_recall.csv", index=False)
    pd.concat([
        hyperparameter_frequency(legacy_v7, "legacy_vs_v7"),
        hyperparameter_frequency(v8, "v7_vs_v8_c"),
    ], ignore_index=True).to_csv(ROOT / "confirmed_hyperparameter_frequency.csv", index=False)
    stage_registry().to_csv(ROOT / "stage_registry.csv", index=False)
    solver_status_summary().to_csv(ROOT / "confirmed_solver_status.csv", index=False)
    derm_pair, derm_class = paired_statistics(derm_replication, "legacy", "v7")
    derm_pair.to_csv(ROOT / "dermatology_original_confirmation_replication.csv", index=False)
    derm_class.to_csv(ROOT / "dermatology_original_class_recall_replication.csv", index=False)
    sources = [V7_FINAL / "per_run.csv", IRIS_FINAL / "per_run.csv",
               V8_FINAL / "per_run.csv", EXPANDED_FINAL / "per_run.csv"]
    index = {
        "status": "complete", "analysis_only": True,
        "model_fits_performed": 0,
        "confirmed_legacy_v7_datasets": sorted(p1.dataset.unique().tolist()),
        "confirmed_v8c_v7_datasets": sorted(p2.dataset.unique().tolist()),
        "source_sha256": {str(path).replace("\\", "/"): digest(path) for path in sources},
        "output_rows": {
            "confirmed_model_summary": len(summaries),
            "confirmed_paired_statistics": len(p1) + len(p2),
            "confirmed_paired_class_recall": len(c1) + len(c2),
        },
    }
    (ROOT / "provenance.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
