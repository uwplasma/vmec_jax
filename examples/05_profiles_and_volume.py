#!/usr/bin/env python
"""Compatibility wrapper for the tutorial examples.

Canonical script: `examples/tutorial/05_profiles_and_volume.py`.
"""

from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("tutorial") / "05_profiles_and_volume.py"), run_name="__main__")

