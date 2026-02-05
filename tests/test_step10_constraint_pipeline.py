from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout_reference_fields
from vmec_jax.vmec_tomnsp import vmec_angle_grid
from vmec_jax.wout import read_wout, state_from_wout


def _load_case(input_rel: str, wout_rel: str):
    root = Path(__file__).resolve().parents[1]
    input_path = root / input_rel
    wout_path = root / wout_rel
    assert input_path.exists()
    assert wout_path.exists()
    cfg, _ = load_config(str(input_path))
    wout = read_wout(wout_path)
    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg, grid=grid)
    st = state_from_wout(wout)
    return static, st, wout


def test_constraint_pipeline_zero_tcon0_yields_zero_gcon():
    pytest.importorskip("netCDF4")

    static, st, wout = _load_case(
        "examples/data/input.circular_tokamak",
        "examples/data/wout_circular_tokamak_reference.nc",
    )

    k = vmec_forces_rz_from_wout_reference_fields(
        state=st,
        static=static,
        wout=wout,
        constraint_tcon0=0.0,
    )

    assert np.allclose(np.asarray(k.gcon), 0.0)
    assert np.allclose(np.asarray(k.arcon_e), 0.0)
    assert np.allclose(np.asarray(k.azcon_e), 0.0)


def test_constraint_pipeline_nonzero_tcon0_produces_gcon():
    pytest.importorskip("netCDF4")

    static, st, wout = _load_case(
        "examples/data/input.li383_low_res",
        "examples/data/wout_li383_low_res_reference.nc",
    )

    k = vmec_forces_rz_from_wout_reference_fields(
        state=st,
        static=static,
        wout=wout,
        constraint_tcon0=0.3,
    )

    gcon_norm = float(np.linalg.norm(np.asarray(k.gcon)))
    assert gcon_norm > 0.0
