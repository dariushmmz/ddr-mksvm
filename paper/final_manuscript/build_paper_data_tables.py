"""Build manuscript-supporting tables from frozen result artifacts only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "PAPER_DATA_TABLES.md"
STATS = ROOT / "results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv"
MODELS = ROOT / "results/expanded_dataset_study/analysis/cross_dataset/confirmed_model_summary.csv"
R9 = ROOT / "results/robust/bounded_rbf_redesign/gates/r9-gate8-15300-15307-v1/paired_statistics.csv"
R10OBJ = ROOT / "results/robust/bounded_rbf_redesign/r10_failure_analysis/paired_objective_deltas.csv"
R10FLIP = ROOT / "results/robust/bounded_rbf_redesign/r10_failure_analysis/dataset_flip_summary.csv"

NAMES = {
    "blood_transfusion": "Blood Transfusion",
    "breast_cancer_diagnostic": "Breast Cancer Diagnostic",
    "breast_cancer_recurrence": "Breast Cancer recurrence",
    "dermatology": "Dermatology",
    "heart_disease": "Heart Disease (5 class)",
    "iris": "Iris",
    "mammographicmass_binary": "Mammographic Mass",
    "parkinson": "Parkinson",
    "wine": "Wine",
}


def pct(x: float, digits: int = 3) -> str:
    return f"{100*x:.{digits}f}"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def pvalue(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.4f}"


def line(values: list[object]) -> str:
    return "| " + " | ".join(str(v) for v in values) + " |"


def main() -> None:
    stats = pd.read_csv(STATS)
    models = pd.read_csv(MODELS)
    out: list[str] = []
    out += [
        "# Paper data tables",
        "",
        "Generated from frozen machine-readable artifacts by `build_paper_data_tables.py`. Values are not copied from rounded prose reports. Percentage-point deltas are candidate minus reference.",
        "",
        "## A0. Architecture terminology",
        "",
        line(["Internal label", "Scientific manuscript name"]),
        line(["---", "---"]),
        line(["Legacy", "Source-Profile q1 Kernel SVM"]),
        line(["V7", "RKHS-RBF-OVO SVM"]),
        line(["V8-C", "CS-RKHS-RBF-OVO SVM"]),
        line(["Robust V8-C", "RKHS-Norm Robust Class-Sensitive SVM"]),
        line(["R8", "Source-Aligned Class-Sensitive Robust q1 LP"]),
        line(["R8.1", "Equilibrated Source-Aligned Robust q1 LP"]),
        line(["R9", "Bounded-RBF Class-Sensitive Robust q1 SVM"]),
        "",
        "## A1. Final local dataset inventory",
        "",
        line(["Dataset", "N", "Features", "Classes / distribution", "Final role"]),
        line(["---", "---:", "---:", "---", "---"]),
        line(["Blood Transfusion", 748, 4, "570/178", "RKHS Gate8 negative; class-sensitive confirmed"]),
        line(["Breast Cancer Diagnostic", 569, 30, "212/357", "both deterministic comparisons confirmed"]),
        line(["Breast Cancer recurrence", 286, 9, "201/85", "RKHS confirmed; class-sensitive stopped at Gate24"]),
        line(["Dermatology", "358 complete", 34, "111/60/71/48/48/20", "RKHS confirmed; class-sensitive stopped at Gate24"]),
        line(["Heart Disease", "297 complete", 13, "160/54/35/35/13", "RKHS confirmed; class-sensitive tradeoff"]),
        line(["Iris", 150, 4, "50/50/50", "both deterministic comparisons confirmed"]),
        line(["Mammographic Mass", 830, 5, "427/403", "RKHS Gate8 negative; class-sensitive null confirmation"]),
        line(["Parkinson", 195, 22, "48/147", "primary RKHS confirmation; class-sensitive control"]),
        line(["Wine", 178, 13, "59/71/48", "both deterministic comparisons confirmed"]),
        "",
        "The tenth local CSV, `uci_fetch_report.csv`, is acquisition metadata rather than a labeled observation table and is excluded from modeling.",
        "",
        "## A2. Source-Profile q1 versus RKHS-RBF-OVO: untouched 96-seed confirmations",
        "",
        line(["Dataset", "Source-profile error %", "RKHS-RBF-OVO error %", "Delta pp [95% CI]", "W/T/L", "Wilcoxon p", "Effect", "Runtime ratio"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    block = stats[(stats.reference == "legacy") & (stats.candidate == "v7") & (stats.metric == "test_error")]
    for _, r in block.sort_values("mean_delta").iterrows():
        out.append(
            line(
                [
                    NAMES[r.dataset],
                    pct(r.mean_reference),
                    pct(r.mean_candidate),
                    f"{pct(r.mean_delta)} [{pct(r.ci_low)}, {pct(r.ci_high)}]",
                    f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}",
                    pvalue(r.wilcoxon_p),
                    num(r.standardized_paired_effect),
                    f"{r.runtime_ratio:.3f}x",
                ]
            )
        )

    out += [
        "",
        "## A3. Class-sensitive versus unweighted RKHS-RBF-OVO: untouched 96-seed confirmations",
        "",
        line(["Dataset", "Delta error pp", "Delta balanced accuracy pp", "Delta macro-F1 pp", "Error W/T/L", "Runtime ratio", "Interpretation"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    v8 = stats[(stats.reference == "v7") & (stats.candidate == "v8_c")]
    for d in sorted(v8.dataset.unique()):
        sub = v8[v8.dataset == d].set_index("metric")
        e, b, f = sub.loc["test_error"], sub.loc["balanced_accuracy"], sub.loc["macro_f1"]
        interpretation = {
            "blood_transfusion": "strong balance/minority gain; neutral error",
            "heart_disease": "balance/F1 gain with error and runtime cost",
            "mammographicmass_binary": "null control",
        }.get(d, "no confirmed advantage")
        out.append(
            line(
                [
                    NAMES[d],
                    pct(e.mean_delta),
                    pct(b.mean_delta),
                    pct(f.mean_delta),
                    f"{int(e.wins)}/{int(e.ties)}/{int(e.losses)}",
                    f"{e.runtime_ratio:.3f}x",
                    interpretation,
                ]
            )
        )

    out += [
        "",
        "## A4. Bounded-RBF robust q1 fresh Gate8 primary paired comparison",
        "",
        "Weighted robust RBF q1 minus weighted deterministic RBF q1; screening evidence only.",
        "",
        line(["Dataset", "Pairs", "Delta error pp", "Delta BA pp", "Delta macro-F1 pp", "Delta safety/rare recall pp", "Error W/T/L"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    r9 = pd.read_csv(R9)
    for d in ["blood_transfusion", "parkinson", "mammographicmass_binary", "iris"]:
        s = r9[r9.dataset == d].set_index("metric")
        e = s.loc["delta_error"]
        out.append(
            line(
                [
                    NAMES[d],
                    int(e.n),
                    pct(e.mean_delta),
                    pct(s.loc["delta_balanced_accuracy", "mean_delta"]),
                    pct(s.loc["delta_macro_f1", "mean_delta"]),
                    pct(s.loc["delta_safety_recall", "mean_delta"]),
                    f"{int(e.wins)}/{int(e.ties)}/{int(e.losses)}",
                ]
            )
        )

    out += [
        "",
        "## A5. Robust-failure mechanistic diagnostics",
        "",
        line(["Dataset", "Mean L1 ref -> robust", "Support ref -> robust", "Slack sum ref -> robust", "Flips", "Corrected", "Introduced"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    obj = pd.read_csv(R10OBJ)
    flips = pd.read_csv(R10FLIP).set_index("dataset")
    for d in ["blood_transfusion", "parkinson", "mammographicmass_binary", "iris"]:
        s = obj[obj.dataset == d]
        f = flips.loc[d]
        out.append(
            line(
                [
                    NAMES[d],
                    f"{s.u_l1_reference.mean():.2f} -> {s.u_l1_candidate.mean():.2f}",
                    f"{s.support_count_gt_1em08_reference.mean():.2f} -> {s.support_count_gt_1em08_candidate.mean():.2f}",
                    f"{s.xi_sum_reference.mean():.2f} -> {s.xi_sum_candidate.mean():.2f}",
                    int(f.prediction_flips),
                    int(f.deterministic_errors_corrected),
                    int(f.new_errors_introduced),
                ]
            )
        )

    out += [
        "",
        "## A6. Provenance",
        "",
        f"- Deterministic paired statistics: `{STATS.relative_to(ROOT).as_posix()}`",
        f"- Deterministic model summaries: `{MODELS.relative_to(ROOT).as_posix()}`",
        f"- Bounded-RBF Gate8 paired statistics (historical artifact path): `{R9.relative_to(ROOT).as_posix()}`",
        f"- Robust-failure objective diagnostics (historical artifact path): `{R10OBJ.relative_to(ROOT).as_posix()}`",
        f"- Robust-failure flip counts (historical artifact path): `{R10FLIP.relative_to(ROOT).as_posix()}`",
        "",
    ]
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
