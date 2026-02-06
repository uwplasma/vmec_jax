from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.plotting import (
    bmag_from_wout,
    bmag_from_wout_physical,
    closed_theta_grid,
    surface_rz_from_wout,
    surface_rz_from_wout_physical,
    surface_rz_from_state_physical,
    zeta_grid,
    zeta_grid_field_period,
)
from vmec_jax.modes import vmec_mode_table
from vmec_jax.wout import read_wout
from vmec_jax.wout import state_from_wout


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


def test_plotting_physical_matches_field_period():
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    wout_path = root / "examples/data/wout_n3are_R7.75B5.7_lowres.nc"
    wout = read_wout(wout_path)

    theta = closed_theta_grid(64)
    phi = zeta_grid_field_period(32, nfp=int(wout.nfp))
    zeta = phi * float(wout.nfp)

    R_phys, Z_phys = surface_rz_from_wout_physical(wout, theta=theta, phi=phi, s_index=int(wout.ns) - 1, nyq=False)
    R_zeta, Z_zeta = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=int(wout.ns) - 1, nyq=False)
    np.testing.assert_allclose(R_phys, R_zeta, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(Z_phys, Z_zeta, rtol=1e-6, atol=1e-6)

    B_phys = bmag_from_wout_physical(wout, theta=theta, phi=phi, s_index=int(wout.ns) - 1)
    B_zeta = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=int(wout.ns) - 1)
    np.testing.assert_allclose(B_phys, B_zeta, rtol=1e-6, atol=1e-6)


def test_plotting_state_surface_matches_wout():
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    wout_path = root / "examples/data/wout_n3are_R7.75B5.7_lowres.nc"
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    theta = closed_theta_grid(64)
    phi = zeta_grid_field_period(32, nfp=int(wout.nfp))
    modes = vmec_mode_table(wout.mpol, wout.ntor)

    R_state, Z_state = surface_rz_from_state_physical(
        state,
        modes,
        theta=theta,
        phi=phi,
        s_index=int(wout.ns) - 1,
        nfp=int(wout.nfp),
    )
    R_wout, Z_wout = surface_rz_from_wout_physical(wout, theta=theta, phi=phi, s_index=int(wout.ns) - 1, nyq=False)

    np.testing.assert_allclose(R_state, R_wout, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(Z_state, Z_wout, rtol=1e-6, atol=1e-6)
