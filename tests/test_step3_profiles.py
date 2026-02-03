from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.geom import eval_geom
from vmec_jax.integrals import cumtrapz_s, dvds_from_sqrtg
from vmec_jax.namelist import InData
from vmec_jax.profiles import MU0, eval_profiles


def test_power_series_profiles_against_manual():
    # Construct a minimal indata with known coefficients.
    indata = InData(
        scalars={
            "PMASS_TYPE": "power_series",
            "PIOTA_TYPE": "power_series",
            "PCURR_TYPE": "power_series",
            "AM": [2.0, -1.0, 0.5],  # p(x) = 2 - x + 0.5 x^2
            "AI": [0.4, 0.2],        # iota(x) = 0.4 + 0.2 x
            "AC": [1.0, 2.0],        # I'(x) = 1 + 2 x  => I(x)= x + x^2
            "PRES_SCALE": 3.0,
            "BLOAT": 1.5,
            "SPRES_PED": 0.6,
            "LRFP": False,
            "NCURR": 0,
        },
        indexed={},
    )

    s = np.linspace(0.0, 1.0, 11)
    prof = eval_profiles(indata, s)

    x = np.minimum(np.abs(s * 1.5), 1.0)
    p = 3.0 * (2.0 + (-1.0) * x + 0.5 * x**2)
    x_ped = min(abs(0.6 * 1.5), 1.0)
    p_ped = 3.0 * (2.0 + (-1.0) * x_ped + 0.5 * x_ped**2)
    p = np.where(s > 0.6, p_ped, p)
    iota = 0.4 + 0.2 * x
    current = x + x**2

    np.testing.assert_allclose(np.asarray(prof["pressure_pa"]), p, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(prof["pressure"]), MU0 * p, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(prof["iota"]), iota, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(prof["current"]), current, rtol=0, atol=1e-12)


def test_volume_total_matches_vmec2000_wout_reference(load_case_lsp_low_res):
    netCDF4 = pytest.importorskip("netCDF4")

    cfg, _indata, static, _bdy, st0 = load_case_lsp_low_res

    g = eval_geom(st0, static)
    dvds = dvds_from_sqrtg(g.sqrtg, static.grid.theta, static.grid.zeta, cfg.nfp)
    V = cumtrapz_s(dvds, static.s)

    V_total = float(np.asarray(V[-1])) * float(cfg.nfp)  # full torus

    wout = Path(__file__).resolve().parents[1] / "examples" / "data" / "wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc"
    with netCDF4.Dataset(wout) as ds:
        V_ref = float(ds.variables["volume_p"][:])

    # Coarse grids + finite-difference Rs give a modest error; start loose and tighten later.
    assert np.isfinite(V_total)
    assert np.isfinite(V_ref)
    assert np.isclose(V_total, V_ref, rtol=0.05, atol=0.0)
