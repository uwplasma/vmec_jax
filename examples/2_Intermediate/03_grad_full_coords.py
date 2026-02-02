#!/usr/bin/env python
"""Wrapper: Step-1 autodiff through full coords kernel (categorized as 2_Intermediate)."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "03_grad_full_coords.py"), run_name="__main__")


if __name__ == "__main__":
    main()

