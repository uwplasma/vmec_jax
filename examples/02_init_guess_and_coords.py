"""Compatibility wrapper for the categorized examples.

The canonical version of this example lives in `examples/1_Simple/`.
"""

from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("1_Simple") / "02_init_guess_and_coords.py"), run_name="__main__")
