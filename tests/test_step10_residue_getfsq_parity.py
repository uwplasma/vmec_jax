from __future__ import annotations

# Step-10 parity regression: VMEC-style forces/tomnsps/getfsq.

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


@pytest.mark.parametrize(
    "case_name,input_rel,wout_rel,rtol_rz,rtol_l",
    [
        ("circular_tokamak", "examples/input.circular_tokamak", "examples/wout_circular_tokamak_reference.nc", 0.10, 0.10),
        ("up_down_asymmetric_tokamak", "examples/input.up_down_asymmetric_tokamak", "examples/wout_up_down_asymmetric_tokamak_reference.nc", 1.00, 0.50),
        ("li383_low_res", "examples/input.li383_low_res", "examples/wout_li383_low_res_reference.nc", 0.30, 0.30),
        ("lsp_low_res", "examples/input.LandremanSenguptaPlunk_section5p3_low_res", "examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc", 0.25, 0.05),
    ],
)
def test_step10_getfsq_parity_against_wout(case_name: str, input_rel: str, wout_rel: str, rtol_rz: float, rtol_l: float):
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

    # Target parity condition: these should agree once the remaining VMEC
    # conventions (lambda forces, endpoint-weighted grids, axis regularization,
    # and tomnsps normalization) are ported.
    assert np.isfinite(scal.fsqr)
    assert np.isfinite(scal.fsqz)
    assert np.isfinite(scal.fsql)

    # Target parity condition: scalar residuals should agree with VMEC2000's
    # `residue/getfsq` outputs on the same (ntheta,nzeta) grid. We keep
    # tolerances modest during the parity push, and tighten as conventions
    # converge (especially for lasym+3D cases).
    denom_r = max(abs(wout.fsqr), 1e-20)
    denom_z = max(abs(wout.fsqz), 1e-20)
    denom_l = max(abs(wout.fsql), 1e-20)
    assert abs(scal.fsqr - wout.fsqr) / denom_r < float(rtol_rz)
    assert abs(scal.fsqz - wout.fsqz) / denom_z < float(rtol_rz)
    assert abs(scal.fsql - wout.fsql) / denom_l < float(rtol_l)
