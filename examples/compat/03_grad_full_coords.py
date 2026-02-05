#!/usr/bin/env python
"""Compatibility wrapper for the tutorial examples.

Canonical script: `examples/tutorial/03_grad_full_coords.py`.
"""

from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tutorial" / Path(__file__).name
runpy.run_path(str(TARGET), run_name="__main__")
