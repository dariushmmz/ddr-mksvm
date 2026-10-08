"""Rebuild aggregate robust-parity artifacts from complete or partial checkpoints."""
import argparse
from pathlib import Path
from archive.research_scripts.run_robust_matlab_parity import aggregate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir"); parser.add_argument("--output-dir")
    args = parser.parse_args(); root=Path(args.run_dir); destination=Path(args.output_dir) if args.output_dir else root/"analysis"
    output = aggregate(root, allow_partial=True, output_dir=destination)
    print(output["summary"].to_string(index=False) if not output["summary"].empty else "No completed checkpoints")
