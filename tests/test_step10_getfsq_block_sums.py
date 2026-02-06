from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar, vmec_fsq_from_tomnsps, vmec_fsq_sums_from_tomnsps
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


@pytest.mark.parametrize(
    "case_name,input_rel,wout_rel",
    [
        ("li383_low_res", "examples/data/input.li383_low_res", "examples/data/wout_li383_low_res_reference.nc"),
    ],
)
def test_step10_getfsq_block_sums_reconstruct_scalars(case_name: str, input_rel: str, wout_rel: str):
    """Internal consistency: per-block sums-of-squares reconstruct fsq scalars exactly."""
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
    st = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )

    k = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata)
    rzl = vmec_residual_internal_from_kernels(k, cfg_ntheta=int(cfg.ntheta), cfg_nzeta=int(cfg.nzeta), wout=wout, trig=trig)
    frzl = TomnspsRZL(
        frcc=rzl.frcc,
        frss=rzl.frss,
        fzsc=rzl.fzsc,
        fzcs=rzl.fzcs,
        flsc=rzl.flsc,
        flcs=rzl.flcs,
        frsc=rzl.frsc,
        frcs=rzl.frcs,
        fzcc=rzl.fzcc,
        fzss=rzl.fzss,
        flcc=rzl.flcc,
        flss=rzl.flss,
    )

    norms = vmec_force_norms_from_bcovar(bc=k.bc, trig=trig, wout=wout, s=static.s)
    scal = vmec_fsq_from_tomnsps(frzl=frzl, norms=norms, lconm1=bool(getattr(cfg, "lconm1", True)))
    sums = vmec_fsq_sums_from_tomnsps(frzl=frzl, lconm1=bool(getattr(cfg, "lconm1", True)), apply_m1_constraints=True)

    assert np.isfinite(scal.fsqr)
    assert np.isfinite(scal.fsqz)
    assert np.isfinite(scal.fsql)

    assert np.isfinite(sums.gcr2)
    assert np.isfinite(sums.gcz2)
    assert np.isfinite(sums.gcl2)

    # Reconstruct scalars from sums with tight tolerance (float64 reductions).
    fsqr2 = norms.r1 * norms.fnorm * sums.gcr2
    fsqz2 = norms.r1 * norms.fnorm * sums.gcz2
    fsql2 = norms.fnormL * sums.gcl2

    assert abs(fsqr2 - scal.fsqr) <= 1e-12 * max(abs(scal.fsqr), 1.0)
    assert abs(fsqz2 - scal.fsqz) <= 1e-12 * max(abs(scal.fsqz), 1.0)
    assert abs(fsql2 - scal.fsql) <= 1e-12 * max(abs(scal.fsql), 1.0)
