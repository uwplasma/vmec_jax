from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.vmec_residue import (
    vmec_force_norms_from_bcovar,
    vmec_force_norms_from_bcovar_dynamic,
)
from vmec_jax.vmec_tomnsp import vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


_CASES = [
    ("circular_tokamak", "examples/data/input.circular_tokamak", "examples/data/wout_circular_tokamak_reference.nc"),
]


def _rel_err(a: float, b: float) -> float:
    den = max(abs(b), 1e-30)
    return abs(a - b) / den


@pytest.mark.parametrize("case_name,input_rel,wout_rel", _CASES)
def test_step10_force_norms_dynamic_matches_wout_scalars(case_name: str, input_rel: str, wout_rel: str):
    """Dynamic force norms match the wout-derived VMEC conventions on the internal grid."""
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / input_rel
    wout_path = root / wout_rel
    assert input_path.exists()
    assert wout_path.exists()

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    st = state_from_wout(wout)

    # VMEC internal (theta,zeta) grid / trig tables used for Step-10 parity.
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

    # Compute bcovar on the half mesh. Use wout Nyquist bsup to isolate the
    # normalization/integration conventions from small bsup reconstruction differences.
    bc = vmec_bcovar_half_mesh_from_wout(state=st, static=static, wout=wout, use_wout_bsup=True)

    norms_dyn = vmec_force_norms_from_bcovar_dynamic(bc=bc, trig=trig, s=static.s, signgs=int(wout.signgs))

    # vp, wb, wp should match wout scalars (to tight tolerance).
    vp_dyn = np.asarray(norms_dyn.vp)
    assert vp_dyn.shape == np.asarray(wout.vp).shape
    assert np.isfinite(vp_dyn).all()
    assert float(np.max(np.abs(vp_dyn - np.asarray(wout.vp)))) < 1e-10

    assert np.isfinite(float(norms_dyn.wb))
    assert np.isfinite(float(norms_dyn.wp))
    assert _rel_err(float(norms_dyn.wb), float(wout.wb)) < 5e-13
    assert _rel_err(float(norms_dyn.wp), float(wout.wp)) < 5e-13

    # VMEC's force-normalization uses a rectangle rule in s:
    #   volume = hs * sum(vp(2:ns)),
    # which is *not* necessarily identical to `wout.volume_p/(2π)^2`.
    hs = float(np.asarray(static.s)[1] - np.asarray(static.s)[0])
    volume_ref = hs * float(np.sum(np.asarray(wout.vp)[1:]))
    assert _rel_err(float(norms_dyn.volume), volume_ref) < 5e-13

    # fnorm/fnormL should match the wout-driven implementation.
    norms_wout = vmec_force_norms_from_bcovar(bc=bc, trig=trig, wout=wout, s=static.s)
    assert _rel_err(float(norms_dyn.fnorm), float(norms_wout.fnorm)) < 5e-13
    assert _rel_err(float(norms_dyn.fnormL), float(norms_wout.fnormL)) < 5e-13
