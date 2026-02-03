from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.field import bsup_from_sqrtg_lambda, chips_from_chipf, lamscale_from_phips
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.wout import read_wout, state_from_wout


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def test_step4_bsup_and_wb_against_wout_reference(load_case_lsp_low_res):
    pytest.importorskip("netCDF4")

    cfg, _indata, static, _bdy, _st0 = load_case_lsp_low_res

    wout_path = Path(__file__).resolve().parents[1] / "examples" / "wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc"
    wout = read_wout(wout_path)
    st = state_from_wout(wout)

    g = eval_geom(st, static)

    # Nyquist basis for reference sqrtg and bsup*
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)

    sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))
    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))

    # VMEC's lambda in wout is scaled; recover lamscale from phips.
    lamscale = float(np.asarray(lamscale_from_phips(wout.phips, static.s)))
    chips = np.asarray(chips_from_chipf(wout.chipf))

    # Lambda derivatives on our grid (scaled lambda); bsupu uses d/dzeta, so divide by NFP.
    lam_u = np.asarray(g.L_theta)
    lam_v = np.asarray(g.L_phi) / wout.nfp

    bsupu, bsupv = bsup_from_sqrtg_lambda(
        sqrtg=sqrtg_ref,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=wout.phipf,
        chipf=chips,
        signgs=wout.signgs,
        lamscale=lamscale,
    )
    bsupu = np.asarray(bsupu)
    bsupv = np.asarray(bsupv)

    # Exclude axis (sqrtg=0) and any near-singular points.
    mask = np.isfinite(sqrtg_ref) & (np.abs(sqrtg_ref) > 1e-14)
    mask[0, :, :] = False

    du = bsupu[mask] - bsupu_ref[mask]
    dv = bsupv[mask] - bsupv_ref[mask]

    # The Nyquist fields are not perfectly representable on the default grid; use RMS tolerances.
    rel_rms_u = _rms(du) / _rms(bsupu_ref[mask])
    rel_rms_v = _rms(dv) / _rms(bsupv_ref[mask])
    assert rel_rms_u < 0.15
    assert rel_rms_v < 0.05

    # Magnetic energy check using computed bsup and reference sqrtg (more stable than using our FD sqrtg).
    gtt = np.asarray(g.g_tt)
    gtp = np.asarray(g.g_tp)
    gpp = np.asarray(g.g_pp)
    B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2

    ds = float(static.s[1] - static.s[0])
    dtheta = 2.0 * np.pi / cfg.ntheta
    dphi = 2.0 * np.pi / (wout.nfp * cfg.nzeta)
    jac = wout.signgs * sqrtg_ref
    E_per_period = float(np.sum(0.5 * B2 * jac) * ds * dtheta * dphi)
    E_total = E_per_period * float(wout.nfp)
    wb_calc = E_total / (2.0 * np.pi) ** 2

    assert np.isfinite(wb_calc)
    assert np.isclose(wb_calc, wout.wb, rtol=2e-3, atol=0.0)
