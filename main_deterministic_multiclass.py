"""
main_deterministic_multiclass.py

Driver script for the deterministic multiclass classifier.
Runs n_runs independent random train/test splits and reports
mean/std of testing error.

FULLY PARAMETERIZED: supports all 4 multiclass datasets from the paper.
Usage:
    python main_deterministic_multiclass.py
    python main_deterministic_multiclass.py --dataset iris
    python main_deterministic_multiclass.py --dataset wine
    python main_deterministic_multiclass.py --dataset heart_disease
    python main_deterministic_multiclass.py --dataset dermatology

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

from unit_of_work_deterministic_multiclass import unit_of_work_deterministic_multiclass, DATASET_CONFIG


def main():
    # ------------------------------------------------------------
    # Parse command-line arguments
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Deterministic multiclass SVM for any dataset from the paper."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="iris",
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
        help="Run-index seeding for paired comparisons against main_ddr_multiclass.py "
             "--ablation legacy --seeded. Default off, matching the original protocol.",
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
    print(f"Classes: {config['classes']}")
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
            delayed(unit_of_work_deterministic_multiclass)(DATA, dataset_name, seed=i)
            for i in range(n_runs)
        )
    else:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(unit_of_work_deterministic_multiclass)(DATA, dataset_name)
            for _ in range(n_runs)
        )
    total_time = time.time() - t0

    print(f"\nElapsed time: {total_time:.2f} s")

    testing_error = np.array(results)

    mean_error = testing_error.mean()
    std_error = testing_error.std(ddof=0)

    print("\n=========== DETERMINISTIC MULTICLASS RESULTS ===========")
    print(f"mean testing error: {mean_error:.6f}")
    print(f"std testing error:  {std_error:.6f}")

    # ------------------------------------------------------------
    # Figure - Boxplot of error distribution across runs
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.boxplot([testing_error], tick_labels=["Overall"])
    plt.ylabel("Classification Error")
    plt.title(f"Deterministic Multiclass SVM Error ({n_runs} runs) — {dataset_name}")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Figure_Deterministic_Boxplot.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Figure - Histogram of error distribution
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(testing_error, bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(mean_error, color="red", linestyle="--", linewidth=2, label=f"Mean={mean_error:.4f}")
    plt.xlabel("Classification Error")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution ({n_runs} runs) — {dataset_name}")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Figure_Deterministic_Histogram.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    savemat(os.path.join(results_dir, "deterministic_results.mat"), {
        "dataset_name": dataset_name,
        "testing_error": testing_error,
        "mean_error": mean_error,
        "std_error": std_error,
        "total_time": total_time,
    })

    results_df = pd.DataFrame({
        "run": np.arange(1, n_runs + 1),
        "testing_error": testing_error,
    })
    results_df.to_csv(os.path.join(results_dir, "deterministic_results.csv"), index=False)

    print("\nResults saved successfully:")
    print(f"  {os.path.join(results_dir, 'Figure_Deterministic_Boxplot.png')}")
    print(f"  {os.path.join(results_dir, 'Figure_Deterministic_Histogram.png')}")
    print(f"  {os.path.join(results_dir, 'deterministic_results.mat')}")
    print(f"  {os.path.join(results_dir, 'deterministic_results.csv')}")


if __name__ == "__main__":
    main()