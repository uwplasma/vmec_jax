"""Pytest configuration.

Allows running tests directly from the repo without requiring an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import os


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def require_slow() -> None:
    """Skip tests marked as slow unless RUN_SLOW=1 is set."""
    if os.environ.get("RUN_SLOW", "") != "1":
        pytest.skip("Set RUN_SLOW=1 to run slow gradient/implicit tests")


@pytest.fixture(scope="session")
def load_case_li383_low_res():
    """Load the bundled li383 low-res symmetric (lasym=False) input used in examples."""
    from vmec_jax._compat import has_jax, enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import initial_guess_from_boundary

    if has_jax():
        enable_x64(True)

    inpath = _ROOT / "examples" / "data" / "input.li383_low_res"
    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    return cfg, indata, static, bdy, st0


@pytest.fixture(scope="session")
def load_case_circular_tokamak():
    """Load the bundled circular tokamak (axisymmetric, lasym=False) input used in examples."""
    from vmec_jax._compat import has_jax, enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import initial_guess_from_boundary

    if has_jax():
        enable_x64(True)

    inpath = _ROOT / "examples" / "data" / "input.circular_tokamak"
    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    return cfg, indata, static, bdy, st0
