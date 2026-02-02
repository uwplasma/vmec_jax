#!/usr/bin/env python
"""Wrapper: Step-1 initial guess + coords kernel (categorized as 1_Simple)."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "02_init_guess_and_coords.py"), run_name="__main__")


if __name__ == "__main__":
    main()

