from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import (
    rz_residual_coeffs_from_kernels,
    rz_residual_scalars_like_vmec,
    vmec_forces_rz_from_wout_reference_fields,
)
from vmec_jax.wout import read_wout, state_from_wout


@pytest.mark.xfail(reason="Step-10 parity WIP: full VMEC residue/getfsq matching not yet achieved.")
def test_step10_getfsq_parity_circular_tokamak():
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/input.circular_tokamak"
    wout_path = root / "examples/wout_circular_tokamak_reference.nc"
    assert input_path.exists()
    assert wout_path.exists()

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    static = build_static(replace(cfg, ntheta=max(int(cfg.ntheta), 128), nzeta=max(int(cfg.nzeta), 128)))
    st = state_from_wout(wout)

    k = vmec_forces_rz_from_wout_reference_fields(state=st, static=static, wout=wout)
    coeffs = rz_residual_coeffs_from_kernels(k, static=static)
    scal = rz_residual_scalars_like_vmec(coeffs, bc=k.bc, wout=wout, s=static.s)

    # Target parity condition: these should agree once the remaining VMEC
    # conventions (lambda forces, endpoint-weighted grids, axis regularization,
    # and tomnsps normalization) are ported.
    assert np.isfinite(scal.fsqr_like)
    assert np.isfinite(scal.fsqz_like)
    assert abs(scal.fsqr_like - wout.fsqr) / max(abs(wout.fsqr), 1e-300) < 0.2
    assert abs(scal.fsqz_like - wout.fsqz) / max(abs(wout.fsqz), 1e-300) < 0.2

