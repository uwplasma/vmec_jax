from __future__ import annotations

# Step-10 parity regression: VMEC-style forces/tomnsps/getfsq.

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar, vmec_fsq_from_tomnsps
from vmec_jax.vmec_forces import (
    vmec_forces_rz_from_wout,
    vmec_residual_internal_from_kernels,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def test_step10_getfsq_parity_circular_tokamak():
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/input.circular_tokamak"
    wout_path = root / "examples/wout_circular_tokamak_reference.nc"
    assert input_path.exists()
    assert wout_path.exists()

    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    cfg_hi = replace(cfg, ntheta=max(int(cfg.ntheta), 128), nzeta=max(int(cfg.nzeta), 128))
    grid = vmec_angle_grid(ntheta=int(cfg_hi.ntheta), nzeta=int(cfg_hi.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg_hi, grid=grid)
    st = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=int(cfg_hi.ntheta),
        nzeta=int(cfg_hi.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )

    k = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata)
    rzl = vmec_residual_internal_from_kernels(k, cfg_ntheta=int(cfg_hi.ntheta), cfg_nzeta=int(cfg_hi.nzeta), wout=wout, trig=trig)
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
    scal = vmec_fsq_from_tomnsps(frzl=frzl, norms=norms)

    # Target parity condition: these should agree once the remaining VMEC
    # conventions (lambda forces, endpoint-weighted grids, axis regularization,
    # and tomnsps normalization) are ported.
    assert np.isfinite(scal.fsqr)
    assert np.isfinite(scal.fsqz)
    assert np.isfinite(scal.fsql)
    # Remaining difference is dominated by minor endpoint/roundoff details and
    # should be at the few-percent level or better for this baseline case.
    assert abs(scal.fsqr - wout.fsqr) / max(abs(wout.fsqr), 1e-300) < 0.02
    assert abs(scal.fsqz - wout.fsqz) / max(abs(wout.fsqz), 1e-300) < 0.02
    assert abs(scal.fsql - wout.fsql) / max(abs(wout.fsql), 1e-300) < 0.02
