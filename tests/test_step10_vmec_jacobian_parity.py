from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.grids import AngleGrid
from vmec_jax.integrals import dvds_from_sqrtg_zeta
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.vmec_jacobian import jacobian_half_mesh_from_parity
from vmec_jax.vmec_parity import internal_odd_from_physical, split_rzl_even_odd_m
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
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 32)
    ntheta = 2 * (ntheta // 2)
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 32)
    if nzeta <= 0:
        nzeta = 1
    return replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


@pytest.mark.parametrize("case_name,input_rel,wout_rel", _CASES)
def test_step10_vmec_halfmesh_jacobian_matches_wout_gmnc(case_name: str, input_rel: str, wout_rel: str):
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
    # Use our standard helical basis for evaluating even/odd-m pieces.
    parity = split_rzl_even_odd_m(st, static.basis, static.modes.m)

    R_odd_int = internal_odd_from_physical(parity.R_odd, static.s)
    Z_odd_int = internal_odd_from_physical(parity.Z_odd, static.s)
    Ru_odd_int = internal_odd_from_physical(parity.Rt_odd, static.s)
    Zu_odd_int = internal_odd_from_physical(parity.Zt_odd, static.s)

    jh = jacobian_half_mesh_from_parity(
        pr1_even=parity.R_even,
        pr1_odd=R_odd_int,
        pz1_even=parity.Z_even,
        pz1_odd=Z_odd_int,
        pru_even=parity.Rt_even,
        pru_odd=Ru_odd_int,
        pzu_even=parity.Zt_even,
        pzu_odd=Zu_odd_int,
        s=static.s,
    )
    sqrtg_calc = np.asarray(jh.sqrtg)

    # Reference sqrt(g) from wout Nyquist gmnc/gmns.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=np.asarray(static.grid.theta), zeta=np.asarray(static.grid.zeta), nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)
    sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))

    # Exclude axis.
    err = _rel_rms(sqrtg_calc[1:], sqrtg_ref[1:])
    # This parity kernel matches VMEC's half-mesh staggering and odd-m handling.
    # Remaining discrepancy is dominated by VMEC's endpoint-weighted theta grids
    # (ntheta2/ntheta3) and nscale/mscale normalization.
    assert err < 0.08

    # Volume derivative `vp` is an angular average of sqrt(g), and should match
    # much more tightly even if pointwise sqrt(g) differs.
    dvds = np.asarray(dvds_from_sqrtg_zeta(sqrtg_calc, static.grid.theta, static.grid.zeta, signgs=int(wout.signgs)))
    vp_calc = dvds / (4.0 * np.pi**2)
    vp_ref = np.asarray(wout.vp)
    vp_err = _rel_rms(vp_calc[1:], vp_ref[1:])
    assert vp_err < 1e-2
