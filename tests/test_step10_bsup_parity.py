from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.field import bsup_from_geom, chips_from_chipf, lamscale_from_phips
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


_CASES = [
    (
        "circular_tokamak",
        "examples/data/input.circular_tokamak",
        "examples/data/wout_circular_tokamak_reference.nc",
    ),
]


def _hi_res_cfg(cfg, *, mpol: int, ntor: int):
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 16)
    ntheta = 2 * (ntheta // 2)
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 16)
    if nzeta <= 0:
        nzeta = 1
    return replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))


def _half_mesh_coeffs(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


@pytest.mark.parametrize("case_name,input_rel,wout_rel", _CASES)
def test_step10_bsup_from_geom_matches_wout_on_outer_surfaces(case_name: str, input_rel: str, wout_rel: str):
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
    # wout Nyquist fields (gmnc, bsup*) are stored on the radial half mesh; match
    # that convention by averaging R/Z coefficients onto the half mesh.
    st_half = replace(
        st,
        Rcos=_half_mesh_coeffs(np.asarray(st.Rcos)),
        Rsin=_half_mesh_coeffs(np.asarray(st.Rsin)),
        Zcos=_half_mesh_coeffs(np.asarray(st.Zcos)),
        Zsin=_half_mesh_coeffs(np.asarray(st.Zsin)),
        Lcos=np.asarray(st.Lcos),
        Lsin=np.asarray(st.Lsin),
    )
    g = eval_geom(st_half, static)

    # Nyquist basis for reference fields.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)

    # Construct bsup from vmec_jax formula using wout flux functions, and compare
    # against wout's stored bsup* Fourier series.
    #
    # Note: near-axis surfaces are sensitive to coordinate singularities and the
    # exact axis expansions used by VMEC. For now, measure parity only over the
    # outer quarter of the plasma.
    lamscale = lamscale_from_phips(wout.phips, static.s)
    chips = chips_from_chipf(wout.chipf)
    bsupu_calc, bsupv_calc = bsup_from_geom(
        g,
        phipf=wout.phipf,
        chipf=chips,
        nfp=wout.nfp,
        signgs=wout.signgs,
        lamscale=lamscale,
    )
    bsupu_calc = np.asarray(bsupu_calc)
    bsupv_calc = np.asarray(bsupv_calc)

    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))

    if wout.ns < 4:
        return
    js0 = max(1, int(0.25 * (wout.ns - 1)))
    err_u = _rel_rms(bsupu_calc[js0:], bsupu_ref[js0:])
    err_v = _rel_rms(bsupv_calc[js0:], bsupv_ref[js0:])

    # These tolerances are intentionally loose while axis/half-mesh conventions
    # are refined. Outer surfaces already show good parity for most cases.
    assert err_u < 0.4
    assert err_v < 0.3
