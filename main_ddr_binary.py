"""
main_ddr_binary.py  (v2)

Mirrors main_deterministic_binary.py's CLI convention exactly
(--dataset, --n_runs, "dataset/"+csv_name path, results/<dataset>/...
output dir) so it slots into the same Colab workflow shown in
running_deterministic_version.py -- e.g.:

    python main_ddr_binary.py --dataset mammographicmass_binary --ablation full
    python main_ddr_binary.py --dataset parkinson --ablation all --n_runs 96

Ablations (Section 6.6 of DDR-MKSVM_spec.md):
    legacy    dnn_on=False, mkl_on=False, dro_on=False  (must match
                                                          the deterministic
                                                          baseline exactly)
    dro_only  dnn_on=False, mkl_on=False, dro_on=True
    dnn_dro   dnn_on=True,  mkl_on=False, dro_on=True
    full      dnn_on=True,  mkl_on=True,  dro_on=True   (dataset kernel
                                                          mixed with an
                                                          extra RBF kernel)

Reports both empirical std and the theoretical binomial std side by
side, per the variance-reporting gap flagged in the earlier
reproduction report.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import savemat

from ddr_mksvm.checkpointing import (
    RunCheckpointStore,
    file_sha256,
    run_and_checkpoint,
)
from unit_of_work_ddr_binary import unit_of_work_ddr_binary
from unit_of_work_deterministic_binary import DATASET_CONFIG


ABLATIONS = {
    "legacy":   dict(dnn_on=False, mkl_on=False, dro_on=False),
    "dro_only": dict(dnn_on=False, mkl_on=False, dro_on=True, epsilon=0.001),
    "dnn_dro":  dict(dnn_on=True,  mkl_on=False, dro_on=True, epsilon=0.001),
    "full":     dict(dnn_on=True,  mkl_on=True,  dro_on=True, epsilon=0.001,
                      extra_kernel_specs=[dict(kind="rbf", alpha=1.0)]),
}


def theoretical_binomial_std(p_hat, n_test):
    return float(np.sqrt(p_hat * (1 - p_hat) / n_test))


def run_one_ablation(DATA, dataset_name, ablation_name, n_runs, n_jobs=-1, seeded=False,
                     checkpoint_dir=None, resume=False, checkpoint_metadata=None):
    flags = ABLATIONS[ablation_name]
    t0 = time.time()
    resumed_runs = 0
    if checkpoint_dir is not None:
        metadata = dict(checkpoint_metadata or {})
        metadata.update({
            "kind": "binary",
            "dataset": dataset_name,
            "ablation": ablation_name,
            "flags": flags,
        })
        store = RunCheckpointStore(
            checkpoint_dir, metadata, n_runs=n_runs, seeded=seeded,
            value_count=3, resume=resume,
        )
        completed = store.load_completed()
        resumed_runs = len(completed)
        pending = [i for i in range(n_runs) if i not in completed]

        if resumed_runs:
            print(f"[{dataset_name} / {ablation_name}] Resuming with "
                  f"{resumed_runs}/{n_runs} completed runs.")
        if pending:
            work_kwargs = {"dataset_name": dataset_name, **flags}
            new_results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(run_and_checkpoint)(
                    unit_of_work_ddr_binary, DATA, work_kwargs, i, store.seeds[i],
                    store.run_path(i),
                )
                for i in pending
            )
            completed.update({i: values for i, values in new_results})
        results = [completed[i] for i in range(n_runs)]
    else:
        if seeded:
            # PAIRED mode: run i always sees the exact same train/test split
            # and network initialization in every ablation.
            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(unit_of_work_ddr_binary)(
                    DATA, dataset_name=dataset_name, seed=i, **flags
                )
                for i in range(n_runs)
            )
        else:
            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(unit_of_work_ddr_binary)(DATA, dataset_name=dataset_name, **flags)
                for _ in range(n_runs)
            )
    elapsed = time.time() - t0

    testing_error = np.array([r[0] for r in results])
    testing_error_A = np.array([r[1] for r in results])
    testing_error_B = np.array([r[2] for r in results])

    mean_all = float(testing_error.mean())
    std_empirical = float(testing_error.std(ddof=0))
    n_test = int(round(len(DATA) * 0.25))
    std_theoretical = theoretical_binomial_std(mean_all, n_test)

    resume_note = f", {resumed_runs} loaded" if resumed_runs else ""
    print(f"\n[{dataset_name} / {ablation_name}]  "
          f"({elapsed:.1f}s, {n_runs} runs{resume_note})")
    print(f"  mean testing error = {mean_all:.4f}")
    print(f"  empirical std      = {std_empirical:.4f}")
    print(f"  theoretical std    = {std_theoretical:.4f}  (binomial, n_test={n_test})")
    if std_empirical < 0.3 * std_theoretical:
        print("  WARNING: empirical std is far below the theoretical binomial floor -- "
              "investigate before trusting this run.")

    return dict(mean=mean_all, std_empirical=std_empirical, std_theoretical=std_theoretical,
                testing_error=testing_error, testing_error_A=testing_error_A,
                testing_error_B=testing_error_B, elapsed=elapsed)


def save_aggregate_results(results_dir, dataset_name, all_results):
    """Refresh user-facing outputs after every completed ablation."""
    savemat(os.path.join(results_dir, "ddr_mksvm_results.mat"), {
        f"{name}_testing_error": res["testing_error"]
        for name, res in all_results.items()
    })
    summary = pd.DataFrame([
        dict(dataset=dataset_name, ablation=name, mean=res["mean"],
             std_empirical=res["std_empirical"],
             std_theoretical=res["std_theoretical"], elapsed_s=res["elapsed"])
        for name, res in all_results.items()
    ])
    summary.to_csv(os.path.join(results_dir, "ddr_mksvm_summary.csv"), index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="DDR-MKSVM binary classification for any dataset from the paper.")
    parser.add_argument("--dataset", type=str, default="mammographicmass_binary",
                         help="Available: " + ", ".join(sorted(DATASET_CONFIG.keys())))
    parser.add_argument("--ablation", type=str, default="full",
                         choices=list(ABLATIONS.keys()) + ["all"],
                         help="Which ablation(s) to run (default: full).")
    parser.add_argument("--n_runs", type=int, default=96,
                         help="Number of independent random splits (default: 96, paper protocol).")
    parser.add_argument("--seeded", action="store_true",
                         help="Use run-index seeding (run i always gets the same split as run i "
                              "of any other --seeded script/ablation on this dataset) for a fair, "
                              "paired comparison instead of independently-sampled splits. Default "
                              "off, matching the paper's uncontrolled-random-split protocol.")
    parser.add_argument("--resume", action="store_true",
                         help="Continue from per-run checkpoints for the same dataset, ablation, "
                              "n_runs, seed mode, and input CSV. Without this flag, checkpoints "
                              "for each selected ablation are restarted.")
    args = parser.parse_args()

    if args.dataset not in DATASET_CONFIG:
        available = ", ".join(sorted(DATASET_CONFIG.keys()))
        raise ValueError(f"Unknown dataset: '{args.dataset}'. Available datasets: {available}")

    config = DATASET_CONFIG[args.dataset]
    csv_filename = "dataset/" + config["csv_name"]

    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"CSV file: {csv_filename}")
    print(f"Data transform: {config['data_transform']}")
    print(f"Dataset kernel: {config['kernel']}")
    print(f"Ablation(s): {args.ablation}")
    print(f"n_runs: {args.n_runs}")
    print(f"Resume: {args.resume}")
    print("=" * 60)

    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Dataset CSV not found: '{csv_filename}'. Please place it in the current directory.")

    DATA = pd.read_csv(csv_filename)
    print(f"Loaded {len(DATA)} rows, {len(DATA.columns)} columns.")
    dataset_sha256 = file_sha256(csv_filename)

    results_dir = os.path.join("results", args.dataset, "ddr_mksvm")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}/")

    ablation_names = list(ABLATIONS.keys()) if args.ablation == "all" else [args.ablation]

    all_results = {}
    try:
        for name in ablation_names:
            checkpoint_dir = os.path.join(results_dir, "checkpoints", "binary", name)
            all_results[name] = run_one_ablation(
                DATA, args.dataset, name, args.n_runs, seeded=args.seeded,
                checkpoint_dir=checkpoint_dir, resume=args.resume,
                checkpoint_metadata={"dataset_sha256": dataset_sha256},
            )
            # With --ablation all, completed ablations are now visible even if
            # a later ablation is interrupted.
            summary = save_aggregate_results(results_dir, args.dataset, all_results)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Completed runs remain in {results_dir}/checkpoints/.")
        print("Re-run the same command with --resume to continue.")
        raise SystemExit(130)

    print(f"\nResults saved successfully to {results_dir}/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
