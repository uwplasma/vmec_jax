from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.field import bsup_from_sqrtg_lambda, lamscale_from_phips
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.solve import solve_lambda_gd
from vmec_jax.wout import read_wout, state_from_wout


def test_step5_solve_lambda_decreases_wb_toward_wout(load_case_lsp_low_res):
    pytest.importorskip("netCDF4")

    cfg, _indata, static, _bdy, _st0 = load_case_lsp_low_res

    wout_path = Path(__file__).resolve().parents[1] / "examples" / "wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc"
    wout = read_wout(wout_path)
    st_ref = state_from_wout(wout)

    # Nyquist basis for reference sqrtg.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)
    sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))

    # VMEC's lambda in wout is scaled; recover lamscale from phips.
    lamscale = float(np.asarray(lamscale_from_phips(wout.phips, static.s)))

    # Build an initial state with identical geometry but lambda=0.
    st0 = st_ref.__class__(
        layout=st_ref.layout,
        Rcos=st_ref.Rcos,
        Rsin=st_ref.Rsin,
        Zcos=st_ref.Zcos,
        Zsin=st_ref.Zsin,
        Lcos=np.zeros_like(np.asarray(st_ref.Lcos)),
        Lsin=np.zeros_like(np.asarray(st_ref.Lsin)),
    )

    res = solve_lambda_gd(
        st0,
        static,
        phipf=wout.phipf,
        chipf=wout.chipf,
        signgs=wout.signgs,
        lamscale=lamscale,
        sqrtg=sqrtg_ref,
        max_iter=25,
        step_size=0.05,
    )

    assert res.wb_history.shape[0] >= 2
    assert np.isfinite(res.wb_history).all()
    assert float(res.wb_history[-1]) < float(res.wb_history[0])
    assert np.all(np.diff(res.wb_history) < 0.0)

    # Reference wb using wout lambda, wout sqrtg (Nyquist), and vmec_jax metric.
    g_ref = eval_geom(st_ref, static)
    lam_u = np.asarray(g_ref.L_theta)
    lam_v = np.asarray(g_ref.L_phi) / wout.nfp
    bsupu, bsupv = bsup_from_sqrtg_lambda(
        sqrtg=sqrtg_ref,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=wout.phipf,
        chipf=wout.chipf,
        signgs=wout.signgs,
        lamscale=lamscale,
    )
    bsupu = np.asarray(bsupu)
    bsupv = np.asarray(bsupv)
    gtt = np.asarray(g_ref.g_tt)
    gtp = np.asarray(g_ref.g_tp)
    gpp = np.asarray(g_ref.g_pp)
    B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2

    ds = float(static.s[1] - static.s[0])
    dtheta = 2.0 * np.pi / cfg.ntheta
    dphi = 2.0 * np.pi / (wout.nfp * cfg.nzeta)
    jac = wout.signgs * sqrtg_ref
    E_per_period = float(np.sum(0.5 * B2 * jac) * ds * dtheta * dphi)
    E_total = E_per_period * float(wout.nfp)
    wb_ref = E_total / (2.0 * np.pi) ** 2

    # Solver should move the objective closer to the reference equilibrium.
    err0 = abs(float(res.wb_history[0]) - wb_ref)
    err1 = abs(float(res.wb_history[-1]) - wb_ref)
    assert err1 < err0

