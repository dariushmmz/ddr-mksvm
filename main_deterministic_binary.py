"""
Python translation of main_deterministic_binary.m

Driver script for the deterministic (non-robust) classifier.
Runs n_runs independent random train/test splits and reports
mean/std of overall and per-class testing error.

FULLY PARAMETERIZED: supports all 10 binary datasets from the paper.
Usage:
    python main_deterministic_binary.py
    python main_deterministic_binary.py --dataset mammographicmass_binary
    python main_deterministic_binary.py --dataset arrhythmia
    ... etc.

Results saved to results/<dataset-name>/deterministic/ directory.
"""

# IMPORTANT: force single-threaded BLAS/OpenMP BEFORE numpy is imported
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.io import savemat

from unit_of_work_deterministic_binary import unit_of_work_deterministic_binary, DATASET_CONFIG


def main():
    # ------------------------------------------------------------
    # Parse command-line arguments
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Deterministic SVM binary classification for any dataset from the paper."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mammographicmass_binary",
        help=(
            "Dataset name. Must be one of the keys in DATASET_CONFIG. "
            "Available: " + ", ".join(sorted(DATASET_CONFIG.keys()))
        ),
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=96,
        help="Number of independent random splits (default: 96, paper protocol).",
    )
    parser.add_argument(
        "--seeded",
        action="store_true",
        help="Use run-index seeding so run i uses the same split as run i of any "
             "other --seeded script on this dataset (e.g. main_ddr_binary.py "
             "--ablation legacy --seeded) -- for a fair, paired comparison instead "
             "of independently-sampled splits. Default off, matching the original protocol.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset

    # Validate dataset name
    if dataset_name not in DATASET_CONFIG:
        available = ", ".join(sorted(DATASET_CONFIG.keys()))
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    config = DATASET_CONFIG[dataset_name]
    csv_filename = "dataset/"+config["csv_name"]

    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"CSV file: {csv_filename}")
    print(f"Data transform: {config['data_transform']}")
    print(f"Kernel: {config['kernel']}")
    print(f"n_runs: {args.n_runs}")
    print("=" * 60)

    # ------------------------------------------------------------
    # Results directory
    # ------------------------------------------------------------
    results_dir = os.path.join("results", dataset_name, "deterministic")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}/")

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(
            f"Dataset CSV not found: '{csv_filename}'. "
            f"Please place it in the current directory."
        )

    DATA = pd.read_csv(csv_filename)
    print(f"Loaded {len(DATA)} rows, {len(DATA.columns)} columns.")

    # ------------------------------------------------------------
    # Run n_runs independent random splits
    # ------------------------------------------------------------
    n_runs = args.n_runs

    t0 = time.time()
    if args.seeded:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(unit_of_work_deterministic_binary)(DATA, dataset_name, seed=i)
            for i in range(n_runs)
        )
    else:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(unit_of_work_deterministic_binary)(DATA, dataset_name)
            for _ in range(n_runs)
        )
    total_time = time.time() - t0

    print(f"\nElapsed time: {total_time:.2f} s")

    testing_error = np.array([r[0] for r in results])
    testing_error_classA = np.array([r[1] for r in results])
    testing_error_classB = np.array([r[2] for r in results])

    mean_all = testing_error.mean()
    std_all = testing_error.std(ddof=0)
    mean_classA = testing_error_classA.mean()
    std_classA = testing_error_classA.std(ddof=0)
    mean_classB = testing_error_classB.mean()
    std_classB = testing_error_classB.std(ddof=0)

    print("\n=========== DETERMINISTIC RESULTS ===========")
    print("mean testing error")
    print(mean_all)
    print("std testing error")
    print(std_all)

    print("\nmean testing error class A")
    print(mean_classA)
    print("std testing error class A")
    print(std_classA)

    print("\nmean testing error class B")
    print(mean_classB)
    print("std testing error class B")
    print(std_classB)

    # ------------------------------------------------------------
    # Figure 1 - Boxplot of error distributions across runs
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [testing_error, testing_error_classA, testing_error_classB],
        tick_labels=["Overall", "Class A", "Class B"]
    )
    plt.ylabel("Classification Error")
    plt.title(f"Deterministic SVM Error Distribution ({n_runs} runs) — {dataset_name}")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Figure_Deterministic_Boxplot.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Figure 2 - Mean +/- std bar chart
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    labels = ["Overall", "Class A", "Class B"]
    means = [mean_all, mean_classA, mean_classB]
    stds = [std_all, std_classA, std_classB]
    x_pos = np.arange(len(labels))

    plt.bar(x_pos, means, yerr=stds, capsize=6,
            color=["#4C72B0", "#55A868", "#C44E52"])
    plt.xticks(x_pos, labels)
    plt.ylabel("Classification Error")
    plt.title(f"Deterministic SVM Mean Error +/- Std ({n_runs} runs) — {dataset_name}")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Figure_Deterministic_MeanStd.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    savemat(os.path.join(results_dir, "deterministic_results.mat"), {
        "dataset_name": dataset_name,
        "testing_error": testing_error,
        "testing_error_classA": testing_error_classA,
        "testing_error_classB": testing_error_classB,
        "mean_all": mean_all,
        "std_all": std_all,
        "mean_classA": mean_classA,
        "std_classA": std_classA,
        "mean_classB": mean_classB,
        "std_classB": std_classB,
        "total_time": total_time,
    })

    # Also save as CSV for easy inspection
    results_df = pd.DataFrame({
        "run": np.arange(1, n_runs + 1),
        "testing_error": testing_error,
        "testing_error_classA": testing_error_classA,
        "testing_error_classB": testing_error_classB,
    })
    results_df.to_csv(os.path.join(results_dir, "deterministic_results.csv"), index=False)

    print("\nResults saved successfully:")
    print(f"  {os.path.join(results_dir, 'Figure_Deterministic_Boxplot.png')}")
    print(f"  {os.path.join(results_dir, 'Figure_Deterministic_MeanStd.png')}")
    print(f"  {os.path.join(results_dir, 'deterministic_results.mat')}")
    print(f"  {os.path.join(results_dir, 'deterministic_results.csv')}")


if __name__ == "__main__":
    main()