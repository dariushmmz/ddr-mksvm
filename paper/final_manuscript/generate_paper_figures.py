"""Generate the final manuscript figures from frozen machine-readable artifacts.

This script intentionally performs no model fitting.  It reads only completed
result tables and emits both 300 dpi PNG and vector SVG versions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#1f77b4",
    "orange": "#e69f00",
    "green": "#009e73",
    "red": "#d55e00",
    "purple": "#7b61a8",
    "gray": "#666666",
    "light": "#d9e7f5",
}

LABELS = {
    "breast_cancer_diagnostic": "Breast cancer\ndiagnostic",
    "breast_cancer_recurrence": "Breast cancer\nrecurrence",
    "blood_transfusion": "Blood\ntransfusion",
    "mammographicmass_binary": "Mammographic\nmass",
    "parkinson": "Parkinson",
    "wine": "Wine",
    "heart_disease": "Heart\n(5 class)",
    "dermatology": "Dermatology",
    "iris": "Iris",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_v7_error() -> None:
    df = pd.read_csv(
        ROOT
        / "results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv"
    )
    df = df[
        (df.reference == "legacy")
        & (df.candidate == "v7")
        & (df.metric == "test_error")
    ].copy()
    order = [
        "parkinson",
        "breast_cancer_recurrence",
        "iris",
        "heart_disease",
        "wine",
        "dermatology",
        "breast_cancer_diagnostic",
    ]
    df = df.set_index("dataset").loc[order].reset_index()
    y = np.arange(len(df))
    x = 100 * df.mean_delta.to_numpy()
    lo = x - 100 * df.ci_low.to_numpy()
    hi = 100 * df.ci_high.to_numpy() - x
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([lo, hi]),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["gray"],
        capsize=3,
        linewidth=1.2,
    )
    ax.set_yticks(y, [LABELS[d] for d in df.dataset])
    ax.invert_yaxis()
    ax.set_xlabel("Paired error change, RKHS-RBF-OVO − Source-Profile q1 (percentage points)")
    ax.set_title("RKHS-RBF-OVO error changes on untouched 96-seed confirmations")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)
    ax.text(
        0.01,
        -0.17,
        "Negative values favor RKHS-RBF-OVO; bars are paired 95% t confidence intervals.",
        transform=ax.transAxes,
        fontsize=8,
    )
    save(fig, "fig1_rkhs_error_deltas")


def figure_v8_tradeoff() -> None:
    df = pd.read_csv(
        ROOT
        / "results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv"
    )
    err = df[
        (df.reference == "v7")
        & (df.candidate == "v8_c")
        & (df.metric == "test_error")
    ].set_index("dataset")
    bal = df[
        (df.reference == "v7")
        & (df.candidate == "v8_c")
        & (df.metric == "balanced_accuracy")
    ].set_index("dataset")
    common = sorted(set(err.index) & set(bal.index))
    label_offsets = {
        "mammographicmass_binary": (12, 39),
        "iris": (-9, -19),
        "breast_cancer_diagnostic": (66, 20),
        "wine": (10, -17),
        "parkinson": (8, 7),
        "blood_transfusion": (8, 7),
        "heart_disease": (8, 7),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    for d in common:
        x = 100 * err.loc[d, "mean_delta"]
        y = 100 * bal.loc[d, "mean_delta"]
        color = COLORS["red"] if d in {"blood_transfusion", "heart_disease"} else COLORS["blue"]
        ax.scatter(x, y, s=42, color=color, zorder=3)
        ax.annotate(
            LABELS[d].replace("\n", " "),
            (x, y),
            xytext=label_offsets[d],
            textcoords="offset points",
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.55},
        )
    ax.set_xlabel("Paired error change, class-sensitive − unweighted RKHS (percentage points)")
    ax.set_ylabel("Paired balanced-accuracy change (percentage points; higher is better)")
    ax.set_title("Class-sensitive RKHS-RBF-OVO balance–error tradeoff")
    ax.set_ylim(-0.58, 6.12)
    ax.grid(color="#e6e6e6", linewidth=0.7)
    save(fig, "fig2_class_sensitive_tradeoff")


def figure_runtime() -> None:
    df = pd.read_csv(
        ROOT
        / "results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv"
    )
    v7 = df[
        (df.reference == "legacy")
        & (df.candidate == "v7")
        & (df.metric == "runtime_s")
    ].copy()
    v8 = df[
        (df.reference == "v7")
        & (df.candidate == "v8_c")
        & (df.metric == "runtime_s")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.3), sharex=False)
    for ax, block, title, color in [
        (axes[0], v7, "RKHS-RBF-OVO / Source-Profile q1", COLORS["blue"]),
        (axes[1], v8, "Class-sensitive / unweighted RKHS", COLORS["orange"]),
    ]:
        block = block.sort_values("runtime_ratio")
        y = np.arange(len(block))
        ax.barh(y, block.runtime_ratio, color=color, alpha=0.85)
        ax.axvline(1, color="black", linewidth=0.8)
        ax.set_yticks(y, [LABELS[d].replace("\n", " ") for d in block.dataset])
        ax.set_xscale("log")
        ax.set_xlabel("Runtime ratio (log scale)")
        ax.set_title(title)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)
    axes[0].set_xticks([0.1, 0.2, 0.5, 1.0], ["0.1×", "0.2×", "0.5×", "1×"])
    axes[0].set_xlim(0.08, 1.12)
    axes[1].set_xticks([0.5, 1.0, 2.0, 5.0], ["0.5×", "1×", "2×", "5×"])
    axes[1].set_xlim(0.45, 5.7)
    fig.suptitle("Runtime ratios on 96-seed confirmation runs", y=1.02, fontsize=10)
    save(fig, "fig3_runtime_ratios")


def figure_r9_gate8() -> None:
    df = pd.read_csv(
        ROOT
        / "results/robust/bounded_rbf_redesign/gates/r9-gate8-15300-15307-v1/paired_statistics.csv"
    )
    metrics = [
        ("delta_error", "Error", -1),
        ("delta_balanced_accuracy", "Balanced accuracy", 1),
        ("delta_macro_f1", "Macro F1", 1),
        ("delta_safety_recall", "Safety/rare recall", 1),
    ]
    datasets = ["blood_transfusion", "parkinson", "mammographicmass_binary", "iris"]
    x = np.arange(len(datasets))
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    palette = [COLORS["red"], COLORS["blue"], COLORS["green"], COLORS["purple"]]
    for j, (metric, name, direction) in enumerate(metrics):
        vals = []
        for d in datasets:
            row = df[(df.dataset == d) & (df.metric == metric)]
            vals.append(100 * float(row.mean_delta.iloc[0]))
        ax.bar(x + (j - 1.5) * width, vals, width, label=name, color=palette[j], alpha=0.88)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [LABELS[d].replace("\n", " ") for d in datasets])
    ax.set_ylabel("Robust − deterministic (percentage points)")
    ax.set_title("Fresh Gate8: bounded-RBF robust q1 versus weighted deterministic q1")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
    ax.text(
        0.01,
        -0.20,
        "For error, positive is worse; for the other metrics, negative is worse. Blood has 7 matched pairs; others have 8.",
        transform=ax.transAxes,
        fontsize=8,
    )
    save(fig, "fig4_bounded_rbf_gate8_deltas")


def figure_r10_mechanism() -> None:
    obj = pd.read_csv(
        ROOT
        / "results/robust/bounded_rbf_redesign/r10_failure_analysis/paired_objective_deltas.csv"
    )
    flip = pd.read_csv(
        ROOT
        / "results/robust/bounded_rbf_redesign/r10_failure_analysis/dataset_flip_summary.csv"
    ).set_index("dataset")
    datasets = ["blood_transfusion", "parkinson", "mammographicmass_binary", "iris"]
    metrics = [
        ("u_l1", "Coefficient L1"),
        ("support_count_gt_1em08", "Support count"),
        ("xi_sum", "Slack sum"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    x = np.arange(len(datasets))
    width = 0.24
    colors = [COLORS["blue"], COLORS["green"], COLORS["red"]]
    for j, (stem, name) in enumerate(metrics):
        ratios = []
        for d in datasets:
            part = obj[obj.dataset == d]
            ref = part[f"{stem}_reference"].mean()
            cand = part[f"{stem}_candidate"].mean()
            ratios.append(cand / ref)
        axes[0].bar(x + (j - 1) * width, ratios, width, label=name, color=colors[j], alpha=0.88)
    axes[0].axhline(1, color="black", linewidth=0.8)
    axes[0].set_xticks(x, [LABELS[d].replace("\n", " ") for d in datasets], rotation=18, ha="right")
    axes[0].set_ylabel("Robust / deterministic ratio")
    axes[0].set_title("Model-complexity and slack shifts")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#e6e6e6", linewidth=0.7)

    corrected = [flip.loc[d, "deterministic_errors_corrected"] for d in datasets]
    introduced = [flip.loc[d, "new_errors_introduced"] for d in datasets]
    axes[1].bar(x - width / 2, corrected, width, label="Errors corrected", color=COLORS["green"])
    axes[1].bar(x + width / 2, introduced, width, label="New errors introduced", color=COLORS["red"])
    axes[1].set_xticks(x, [LABELS[d].replace("\n", " ") for d in datasets], rotation=18, ha="right")
    axes[1].set_ylabel("Pooled test observations")
    axes[1].set_title("Consequences of prediction flips")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#e6e6e6", linewidth=0.7)
    save(fig, "fig5_robust_failure_mechanism")


def main() -> None:
    style()
    figure_v7_error()
    figure_v8_tradeoff()
    figure_runtime()
    figure_r9_gate8()
    figure_r10_mechanism()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
