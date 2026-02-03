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
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.wout import read_wout, state_from_wout


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


def test_step10_vmec_bcovar_halfmesh_smoke_circular_tokamak():
    """Smoke test: bcovar half-mesh kernels run and reproduce vp reasonably."""
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.circular_tokamak"
    wout_path = root / "examples/data/wout_circular_tokamak_reference.nc"
    assert input_path.exists()
    assert wout_path.exists()

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)

    # Keep this test fast: moderate angular resolution.
    cfg_mid = replace(cfg, ntheta=max(int(cfg.ntheta), 96), nzeta=max(int(cfg.nzeta), 96))
    static = build_static(cfg_mid)

    st = state_from_wout(wout)
    bc = vmec_bcovar_half_mesh_from_wout(state=st, static=static, wout=wout)

    # Volume derivative vp(s) should match tightly (it is an angular average of sqrt(g)).
    dvds = np.asarray(
        dvds_from_sqrtg_zeta(np.asarray(bc.jac.sqrtg), static.grid.theta, static.grid.zeta, signgs=int(wout.signgs))
    )
    vp_calc = dvds / (4.0 * np.pi**2)
    vp_err = _rel_rms(vp_calc[1:], np.asarray(wout.vp)[1:])
    assert vp_err < 2e-2

    # Reference fields on Nyquist modes.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=np.asarray(static.grid.theta), zeta=np.asarray(static.grid.zeta), nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)

    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    if wout.ns >= 4:
        js0 = max(1, int(0.25 * (wout.ns - 1)))
        err_bsup_u = _rel_rms(np.asarray(bc.bsupu)[js0:], bsupu_ref[js0:])
        err_bsup_v = _rel_rms(np.asarray(bc.bsupv)[js0:], bsupv_ref[js0:])
        err_bsub_u = _rel_rms(np.asarray(bc.bsubu)[js0:], bsubu_ref[js0:])
        err_bsub_v = _rel_rms(np.asarray(bc.bsubv)[js0:], bsubv_ref[js0:])

        assert err_bsup_u < 0.6
        assert err_bsup_v < 0.6
        assert err_bsub_u < 0.3
        assert err_bsub_v < 0.3
