#!/usr/bin/env python
"""Wrapper: Step-3 profiles + volume integrals (categorized as 2_Intermediate)."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "05_profiles_and_volume.py"), run_name="__main__")


if __name__ == "__main__":
    main()

