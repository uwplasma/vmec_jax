#!/usr/bin/env python
"""Wrapper: minimal autodiff demo (categorized as 1_Simple)."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "01_grad_demo.py"), run_name="__main__")


if __name__ == "__main__":
    main()

