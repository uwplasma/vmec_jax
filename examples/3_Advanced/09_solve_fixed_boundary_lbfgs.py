#!/usr/bin/env python
"""Wrapper: Step-7 fixed-boundary solve (L-BFGS) (categorized as 3_Advanced)."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "09_solve_fixed_boundary_lbfgs.py"), run_name="__main__")


if __name__ == "__main__":
    main()

