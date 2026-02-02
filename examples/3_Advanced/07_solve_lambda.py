#!/usr/bin/env python
"""Wrapper: Step-5 lambda-only solve (categorized as 3_Advanced)."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "07_solve_lambda.py"), run_name="__main__")


if __name__ == "__main__":
    main()

