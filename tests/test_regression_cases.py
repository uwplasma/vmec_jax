from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataclasses import replace

from vmec_jax.config import load_config
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.integrals import volume_from_sqrtg_vmec
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


_CASES = [
    (
        "circular_tokamak",
        "examples/data/input.circular_tokamak",
        "examples/data/wout_circular_tokamak_reference.nc",
    ),
    (
        "li383_low_res",
        "examples/data/input.li383_low_res",
        "examples/data/wout_li383_low_res_reference.nc",
    ),
]


def _hi_res_cfg(cfg, *, mpol: int, ntor: int):
    # Use a higher-resolution angular grid for more accurate quadrature when
    # reconstructing wout Nyquist fields on a real-space grid.
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 16)
    ntheta = 2 * (ntheta // 2)  # VMEC uses even ntheta
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 16)
    if nzeta <= 0:
        nzeta = 1
    return replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))


def _half_mesh_coeffs(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


@pytest.mark.parametrize("case_name,input_rel,wout_rel", _CASES)
def test_volume_from_wout_nyquist_matches_volume_p(case_name: str, input_rel: str, wout_rel: str):
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / input_rel
    wout_path = root / wout_rel
    assert input_path.exists()
    assert wout_path.exists()

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    cfg_hi = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg_hi)

    # Nyquist basis for sqrtg.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)
    sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))

    _dvds, V = volume_from_sqrtg_vmec(
        sqrtg_ref,
        static.s,
        static.grid.theta,
        static.grid.zeta,
        signgs=wout.signgs,
    )
    V_total = float(np.asarray(V[-1]))

    assert np.isfinite(V_total)
    assert np.isfinite(wout.volume_p)
    assert np.isclose(V_total, float(wout.volume_p), rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("case_name,input_rel,wout_rel", _CASES)
def test_step4_wb_against_wout_reference(case_name: str, input_rel: str, wout_rel: str):
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / input_rel
    wout_path = root / wout_rel
    assert input_path.exists()
    assert wout_path.exists()

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    cfg_hi = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg_hi)
    st = state_from_wout(wout)
    # wout Nyquist fields (gmnc, bsup*) are stored on the *radial half mesh*.
    # Match that convention by averaging R/Z coefficients onto the half mesh
    # before building metric components.
    st_half = replace(
        st,
        Rcos=_half_mesh_coeffs(np.asarray(st.Rcos)),
        Rsin=_half_mesh_coeffs(np.asarray(st.Rsin)),
        Zcos=_half_mesh_coeffs(np.asarray(st.Zcos)),
        Zsin=_half_mesh_coeffs(np.asarray(st.Zsin)),
        # wout's stored lambda is post-processed; it is not needed for metric-only
        # computations in this test.
        Lcos=np.asarray(st.Lcos),
        Lsin=np.asarray(st.Lsin),
    )
    g = eval_geom(st_half, static)

    # Nyquist basis for reference sqrtg and bsup*.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)
    sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))

    # Use wout-provided contravariant B^u, B^v (Nyquist) for robust energy
    # regressions. wout's stored lambda can be post-processed for backward
    # compatibility and is not always consistent with bsup* across cases.
    bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))

    gtt = np.asarray(g.g_tt)
    gtp = np.asarray(g.g_tp)
    gpp = np.asarray(g.g_pp)
    B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2

    ds = float(static.s[1] - static.s[0]) if static.s.shape[0] > 1 else 1.0
    dtheta = 2.0 * np.pi / int(static.cfg.ntheta)
    dzeta = 2.0 * np.pi / int(static.cfg.nzeta)
    jac = wout.signgs * sqrtg_ref
    # VMEC's wb is stored normalized by (2π)^2 and uses half-mesh quantities
    # with a rectangle rule in s (js=2..ns). Our uniform grid matches that when
    # we exclude js=0.
    wb_calc = float(np.sum(0.5 * B2[1:] * jac[1:]) * ds * dtheta * dzeta) / (2.0 * np.pi) ** 2

    assert np.isfinite(wb_calc)
    assert np.isclose(wb_calc, float(wout.wb), rtol=5e-4, atol=0.0)


@pytest.mark.parametrize("case_name,input_rel,wout_rel", _CASES)
def test_wp_from_wout_vp_pres_matches_wout_wp(case_name: str, input_rel: str, wout_rel: str):
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    wout_path = root / wout_rel
    assert wout_path.exists()

    wout = read_wout(wout_path)
    if wout.ns < 2:
        return
    hs = 1.0 / float(wout.ns - 1)

    wp_calc = hs * float(np.sum(np.asarray(wout.vp)[1:] * np.asarray(wout.pres)[1:]))
    assert np.isfinite(wp_calc)
    assert np.isfinite(wout.wp)
    assert np.isclose(wp_calc, float(wout.wp), rtol=1e-12, atol=0.0)
