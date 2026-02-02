from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.field import bsub_from_bsup
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


_CASES = [
    (
        "lsp_low_res",
        "examples/input.LandremanSenguptaPlunk_section5p3_low_res",
        "examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc",
    ),
    (
        "circular_tokamak",
        "examples/input.circular_tokamak",
        "examples/wout_circular_tokamak_reference.nc",
    ),
    (
        "up_down_asymmetric_tokamak",
        "examples/input.up_down_asymmetric_tokamak",
        "examples/wout_up_down_asymmetric_tokamak_reference.nc",
    ),
    (
        "li383_low_res",
        "examples/input.li383_low_res",
        "examples/wout_li383_low_res_reference.nc",
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
def test_step10_bsubu_bsubv_from_metric_matches_wout(case_name: str, input_rel: str, wout_rel: str):
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

    bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    bsubu_calc, bsubv_calc = bsub_from_bsup(g, bsupu, bsupv)
    bsubu_calc = np.asarray(bsubu_calc)
    bsubv_calc = np.asarray(bsubv_calc)

    # wout provides covariant components on Nyquist modes.
    bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    # Exclude axis surface due to coordinate singularity.
    err_u = _rel_rms(bsubu_calc[1:], bsubu_ref[1:])
    err_v = _rel_rms(bsubv_calc[1:], bsubv_ref[1:])

    # These parity checks depend on half-mesh conventions and low-res geometry.
    # Keep tolerances loose for now; tighten as half-mesh handling is improved.
    assert err_v < 1.5e-2
    assert err_u < 9e-2
