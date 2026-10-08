"""Compatibility launcher for ``python -m ddr_mksvm.cli``."""

from ddr_mksvm.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
