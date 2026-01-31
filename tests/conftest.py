"""Pytest configuration.

Allows running tests directly from the repo without requiring an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="session")
def load_case_lsp_low_res():
    """Load the bundled Landreman/Sengupta/Plunk low-res input used in examples."""
    from vmec_jax._compat import has_jax, enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import initial_guess_from_boundary

    if has_jax():
        enable_x64(True)

    inpath = _ROOT / "examples" / "input.LandremanSenguptaPlunk_section5p3_low_res"
    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    return cfg, indata, static, bdy, st0
