"""Local command-line entry points for supported reproducible workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .paths import PROJECT_ROOT


def _run(script: str, arguments: list[str]) -> int:
    command = [sys.executable, str(PROJECT_ROOT / script), *arguments]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def _run_module(module: str, arguments: list[str]) -> int:
    command = [sys.executable, "-m", module, *arguments]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def _csv(values: list[str]) -> str:
    return ",".join(values)


def _common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--run-name", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddr-mksvm")
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="rebuild the local dataset inventory")
    inventory.add_argument("--output", default="results/expanded_dataset_study/inventory")

    smoke = sub.add_parser("smoke", help="run one local matched-seed smoke experiment")
    smoke.add_argument("--dataset", default="breast_cancer_recurrence")
    smoke.add_argument("--seed", type=int, default=17000)
    smoke.add_argument("--n-jobs", type=int, choices=range(1, 5), default=1)
    smoke.add_argument("--run-name", default="local-smoke")
    smoke.add_argument("--output-root", default="results/local_smoke")

    v7 = sub.add_parser("v7", help="run Source-Profile q1 versus RKHS-RBF-OVO")
    _common_run_arguments(v7)
    v7.add_argument("--output-root", default="results/v7_cross_dataset")

    v8 = sub.add_parser("v8c", help="run RKHS-RBF-OVO versus its class-sensitive extension")
    _common_run_arguments(v8)
    v8.add_argument("--variants", nargs="+", default=["v7", "v8_c"])
    v8.add_argument("--output-root", default="results/v8")
    v8.add_argument("--reuse-root", default="")

    expanded = sub.add_parser("expanded", help="run a staged expanded-dataset experiment")
    _common_run_arguments(expanded)
    expanded.add_argument("--models", nargs="+", default=["legacy", "v7", "v8_c"])
    expanded.add_argument("--stage", choices=["smoke", "gate8", "gate24", "confirmation"], required=True)
    expanded.add_argument("--output-root", default="results/expanded_dataset_study")
    expanded.add_argument("--promotion-source", default="")

    analyze = sub.add_parser("analyze", help="audit and analyze one expanded-study run")
    analyze.add_argument("run_directory")

    sub.add_parser("consolidate", help="rebuild final cross-dataset summaries from frozen artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        return _run("inventory_expanded_datasets.py", ["--output", args.output])
    if args.command == "smoke":
        return _run("run_expanded_dataset_study.py", [
            "--datasets", args.dataset,
            "--models", "legacy,v7,v8_c",
            "--stage", "smoke",
            "--n-seeds", "1",
            "--seed-start", str(args.seed),
            "--n-jobs", str(args.n_jobs),
            "--run-name", args.run_name,
            "--output-root", args.output_root,
        ])
    if args.command == "v7":
        return _run("run_v7_cross_dataset.py", [
            "--datasets", _csv(args.datasets), "--seed-start", str(args.seed_start),
            "--n-seeds", str(args.n_seeds), "--n-jobs", str(args.n_jobs),
            "--run-name", args.run_name, "--output-root", args.output_root,
        ])
    if args.command == "v8c":
        command = [
            "--datasets", _csv(args.datasets), "--variants", _csv(args.variants),
            "--seed-start", str(args.seed_start), "--n-seeds", str(args.n_seeds),
            "--n-jobs", str(args.n_jobs), "--run-name", args.run_name,
            "--output-root", args.output_root,
        ]
        if args.reuse_root:
            command.extend(["--reuse-root", args.reuse_root])
        return _run_module("ddr_mksvm.experiments.class_sensitive_runner", command)
    if args.command == "expanded":
        command = [
            "--datasets", _csv(args.datasets), "--models", _csv(args.models),
            "--stage", args.stage, "--seed-start", str(args.seed_start),
            "--n-seeds", str(args.n_seeds), "--n-jobs", str(args.n_jobs),
            "--run-name", args.run_name, "--output-root", args.output_root,
        ]
        if args.promotion_source:
            command.extend(["--promotion-source", args.promotion_source])
        return _run("run_expanded_dataset_study.py", command)
    if args.command == "analyze":
        return _run("analyze_expanded_dataset_study.py", [args.run_directory])
    if args.command == "consolidate":
        return _run("consolidate_expanded_dataset_study.py", [])
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
