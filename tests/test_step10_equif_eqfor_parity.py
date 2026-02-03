from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout
from vmec_jax.vmec_lforbal import _pwint_from_trig, equif_from_bcovar
from vmec_jax.vmec_tomnsp import vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


@pytest.mark.parametrize(
    "input_rel,wout_rel",
    [
        ("examples/input.circular_tokamak", "examples/wout_circular_tokamak_reference.nc"),
        ("examples/input.li383_low_res", "examples/wout_li383_low_res_reference.nc"),
    ],
)
def test_step10_equif_matches_eqfor_normalization(input_rel: str, wout_rel: str):
    """Reproduce VMEC2000 `eqfor.f` normalization for the `equif` diagnostic.

    `vmec_jax.vmec_lforbal.equif_from_bcovar` computes the *raw* `equif` from
    `fbal.f`. VMEC's netcdf `wout` output stores a *normalized* version computed
    in `eqfor.f`. This test checks that we match the netcdf `equif`.
    """
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / input_rel
    wout_path = root / wout_rel
    assert input_path.exists()
    assert wout_path.exists()

    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)

    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg, grid=grid)

    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )

    st = state_from_wout(wout)
    k = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata)

    # Raw <force-balance> profile, as in `fbal.f`.
    eq_raw = np.asarray(equif_from_bcovar(bc=k.bc, trig=trig, wout=wout, s=static.s))

    # Build surface-averaged currents from bsubu/bsubv, matching the discrete
    # forward-difference scheme in `fbal.f`.
    s = np.asarray(static.s, dtype=float)
    ns = int(s.shape[0])
    if ns < 2:
        return
    hs = float(s[1] - s[0])
    ohs = 1.0 / hs if hs != 0.0 else 0.0
    signgs = float(wout.signgs)

    bsubu = np.asarray(k.bc.bsubu, dtype=float)
    bsubv = np.asarray(k.bc.bsubv, dtype=float)
    pwint = np.asarray(_pwint_from_trig(trig, nzeta=int(bsubu.shape[2]), dtype=bsubu.dtype), dtype=float)
    w_s = (np.arange(ns, dtype=int) + 1 >= 2).astype(float)[:, None, None]  # js>=2 in Fortran
    pwint3 = w_s * pwint[None, :, :]
    buco = np.sum(bsubu * pwint3, axis=(1, 2))
    bvco = np.sum(bsubv * pwint3, axis=(1, 2))

    buco_fwd = np.concatenate([buco[1:], np.zeros((1,), dtype=buco.dtype)], axis=0)
    bvco_fwd = np.concatenate([bvco[1:], np.zeros((1,), dtype=bvco.dtype)], axis=0)
    jcurv = signgs * ohs * (buco_fwd - buco)
    jcuru = -signgs * ohs * (bvco_fwd - bvco)

    # Full-mesh vpphi and half-mesh pressure gradient (fbal.f).
    vp = np.asarray(wout.vp, dtype=float)
    vpphi = 0.5 * (np.concatenate([vp[1:], vp[-1:]], axis=0) + vp)
    pres = np.asarray(wout.pres, dtype=float)
    pres_fwd = np.concatenate([pres[1:], pres[-1:]], axis=0)
    presgrad = (pres_fwd - pres) * ohs

    # `eqfor.f` uses the *unscaled* phipf/chipf in the normalization denominator.
    twopi = 2.0 * np.pi
    phipf_int = np.asarray(wout.phipf, dtype=float) / (twopi * signgs)
    chipf_int = np.asarray(wout.chipf, dtype=float) / (twopi * signgs)

    denom = np.abs(jcurv * chipf_int) + np.abs(jcuru * phipf_int) + np.abs(presgrad * vpphi)
    denom = np.where(denom != 0.0, denom, 1.0)
    eq_norm = eq_raw * vpphi / denom

    # Endpoint extrapolation used by `eqfor.f` (eqfor prints these; wout stores them).
    if ns >= 3:
        eq_norm[0] = 2.0 * eq_norm[1] - eq_norm[2]
        eq_norm[-1] = 2.0 * eq_norm[-2] - eq_norm[-3]

    np.testing.assert_allclose(eq_norm, np.asarray(wout.equif), atol=5e-10, rtol=1e-8)
