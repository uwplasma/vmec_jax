from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.plotting import bmag_from_wout, closed_theta_grid, surface_rz_from_wout, zeta_grid
from vmec_jax.wout import read_wout


def test_plotting_surface_helpers_shapes():
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    wout_path = root / "examples/data/wout_n3are_R7.75B5.7_lowres.nc"
    wout = read_wout(wout_path)

    theta = closed_theta_grid(64)
    zeta = zeta_grid(32)
    s_index = int(wout.ns) - 1

    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index, nyq=False)
    B = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index)

    assert R.shape == (theta.size, zeta.size)
    assert Z.shape == (theta.size, zeta.size)
    assert B.shape == (theta.size, zeta.size)

    # Closed theta grid should make first/last points coincide.
    assert np.allclose(R[0], R[-1], rtol=1e-6, atol=1e-6)
    assert np.allclose(Z[0], Z[-1], rtol=1e-6, atol=1e-6)
