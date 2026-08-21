"""
main_ddr_multiclass.py

Mirrors main_deterministic_multiclass.py's CLI convention (--dataset,
--n_runs, "dataset/"+csv_name path, results/<dataset>/... output dir):

    python main_ddr_multiclass.py --dataset iris --ablation full
    python main_ddr_multiclass.py --dataset heart_disease --ablation all
    python main_ddr_multiclass.py --dataset dermatology --ablation dro_only

Same 4-ablation suite as main_ddr_binary.py (legacy / dro_only / dnn_dro / full).
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
from ddr_mksvm.runtime import resolve_n_jobs
from unit_of_work_ddr_multiclass import unit_of_work_ddr_multiclass
from unit_of_work_deterministic_multiclass import DATASET_CONFIG


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
    n_jobs = resolve_n_jobs(n_jobs, n_runs)
    flags = ABLATIONS[ablation_name]
    t0 = time.time()
    resumed_runs = 0
    if checkpoint_dir is not None:
        metadata = dict(checkpoint_metadata or {})
        metadata.update({
            "kind": "multiclass",
            "dataset": dataset_name,
            "ablation": ablation_name,
            "flags": flags,
        })
        store = RunCheckpointStore(
            checkpoint_dir, metadata, n_runs=n_runs, seeded=seeded,
            value_count=1, resume=resume,
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
                    unit_of_work_ddr_multiclass, DATA, work_kwargs, i, store.seeds[i],
                    store.run_path(i),
                )
                for i in pending
            )
            completed.update({i: values for i, values in new_results})
        results = [float(completed[i][0]) for i in range(n_runs)]
    else:
        if seeded:
            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(unit_of_work_ddr_multiclass)(
                    DATA, dataset_name=dataset_name, seed=i, **flags
                )
                for i in range(n_runs)
            )
        else:
            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(unit_of_work_ddr_multiclass)(
                    DATA, dataset_name=dataset_name, **flags
                )
                for _ in range(n_runs)
            )
    elapsed = time.time() - t0

    testing_error = np.array(results)
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

    return dict(mean=mean_all, std_empirical=std_empirical, std_theoretical=std_theoretical,
                testing_error=testing_error, elapsed=elapsed)


def save_aggregate_results(results_dir, dataset_name, all_results):
    """Refresh user-facing outputs after every completed ablation."""
    savemat(os.path.join(results_dir, "ddr_mksvm_multiclass_results.mat"), {
        f"{name}_testing_error": res["testing_error"]
        for name, res in all_results.items()
    })
    summary = pd.DataFrame([
        dict(dataset=dataset_name, ablation=name, mean=res["mean"],
             std_empirical=res["std_empirical"],
             std_theoretical=res["std_theoretical"], elapsed_s=res["elapsed"])
        for name, res in all_results.items()
    ])
    summary.to_csv(
        os.path.join(results_dir, "ddr_mksvm_multiclass_summary.csv"), index=False
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="DDR-MKSVM multiclass classification for any dataset from the paper.")
    parser.add_argument("--dataset", type=str, default="iris",
                         help="Available: " + ", ".join(sorted(DATASET_CONFIG.keys())))
    parser.add_argument("--ablation", type=str, default="full",
                         choices=list(ABLATIONS.keys()) + ["all"])
    parser.add_argument("--n_runs", type=int, default=96)
    parser.add_argument("--n_jobs", type=int, default=None,
                         help="Concurrent training workers. Default: container/CPU-quota-aware "
                              "CPU count. Use a smaller value if memory is limited; -1 also uses "
                              "the quota-aware count.")
    parser.add_argument("--seeded", action="store_true",
                         help="Run-index seeding for paired comparisons -- see main_ddr_binary.py.")
    parser.add_argument("--resume", action="store_true",
                         help="Continue from per-run checkpoints for the same dataset, ablation, "
                              "n_runs, seed mode, and input CSV. Without this flag, checkpoints "
                              "for each selected ablation are restarted.")
    args = parser.parse_args()
    n_jobs = resolve_n_jobs(args.n_jobs, args.n_runs)

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
    print(f"Classes: {config['classes']}")
    print(f"Ablation(s): {args.ablation}")
    print(f"n_runs: {args.n_runs}")
    print(f"n_jobs: {n_jobs}")
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
            checkpoint_dir = os.path.join(results_dir, "checkpoints", "multiclass", name)
            all_results[name] = run_one_ablation(
                DATA, args.dataset, name, args.n_runs, n_jobs=n_jobs, seeded=args.seeded,
                checkpoint_dir=checkpoint_dir, resume=args.resume,
                checkpoint_metadata={"dataset_sha256": dataset_sha256},
            )
            summary = save_aggregate_results(results_dir, args.dataset, all_results)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Completed runs remain in {results_dir}/checkpoints/.")
        print("Re-run the same command with --resume to continue.")
        raise SystemExit(130)

    print(f"\nResults saved successfully to {results_dir}/")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
