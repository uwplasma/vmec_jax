from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import rz_residual_coeffs_from_kernels, vmec_forces_rz_from_wout_reference_fields
from vmec_jax.wout import read_wout, state_from_wout


def test_step10_vmec_forces_kernel_smoke_axisymmetric():
    """Smoke test: VMEC force kernel runs end-to-end and returns finite outputs."""
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/input.circular_tokamak"
    wout_path = root / "examples/wout_circular_tokamak_reference.nc"
    assert input_path.exists()
    assert wout_path.exists()

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)

    # Keep this test fast.
    static = build_static(replace(cfg, ntheta=max(int(cfg.ntheta), 64), nzeta=max(int(cfg.nzeta), 64)))
    st = state_from_wout(wout)

    k = vmec_forces_rz_from_wout_reference_fields(state=st, static=static, wout=wout)
    assert np.all(np.isfinite(np.asarray(k.armn_e)))
    assert np.all(np.isfinite(np.asarray(k.brmn_e)))
    assert np.all(np.isfinite(np.asarray(k.azmn_e)))
    assert np.all(np.isfinite(np.asarray(k.bzmn_e)))

    # Axisymmetric case should not produce C kernels.
    assert np.allclose(np.asarray(k.crmn_e), 0.0)
    assert np.allclose(np.asarray(k.czmn_e), 0.0)

    coeffs = rz_residual_coeffs_from_kernels(k, static=static)
    assert np.all(np.isfinite(np.asarray(coeffs.gcr_cos)))
    assert np.all(np.isfinite(np.asarray(coeffs.gcz_cos)))

