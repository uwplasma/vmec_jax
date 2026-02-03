#!/usr/bin/env python
"""Compatibility wrapper for the tutorial examples.

Canonical script: `examples/tutorial/08_solve_fixed_boundary.py`.
"""

from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("tutorial") / "08_solve_fixed_boundary.py"), run_name="__main__")

