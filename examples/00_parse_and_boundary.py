#!/usr/bin/env python
"""Compatibility wrapper for the tutorial examples.

Canonical script: `examples/tutorial/00_parse_and_boundary.py`.
"""

from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("tutorial") / "00_parse_and_boundary.py"), run_name="__main__")

