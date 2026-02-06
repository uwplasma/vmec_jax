from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.vmec_residue import vmec_wint_from_trig
from vmec_jax.vmec_tomnsp import vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout


def _rel_err(a: float, b: float) -> float:
    den = max(abs(b), 1e-30)
    return abs(a - b) / den


@pytest.mark.parametrize(
    "case_name,input_rel,wout_rel",
    [
        ("circular_tokamak", "examples/data/input.circular_tokamak", "examples/data/wout_circular_tokamak_reference.nc"),
        ("li383_low_res", "examples/data/input.li383_low_res", "examples/data/wout_li383_low_res_reference.nc"),
    ],
)
def test_step10_wb_wp_integrals_match_wout(case_name: str, input_rel: str, wout_rel: str):
    """Energy/pressure integrals match VMEC2000 `wout` scalars on the internal grid."""
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / input_rel
    wout_path = root / wout_rel
    assert input_path.exists()
    assert wout_path.exists()

    cfg, _indata = load_config(str(input_path))
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

    # Evaluate Nyquist fields on the VMEC internal grid.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid_nyq = AngleGrid(theta=np.asarray(static.grid.theta), zeta=np.asarray(static.grid.zeta), nfp=int(wout.nfp))
    basis_nyq = build_helical_basis(modes_nyq, grid_nyq)
    sqrtg = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))
    bmag = np.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))

    # VMEC: wb = hs*abs(sum_{js=2..ns} sum(pwint*sqrtg*(|B|^2/2))).
    ns = int(np.asarray(wout.ns))
    assert ns == int(np.asarray(static.s).shape[0])
    hs = float(np.asarray(static.s)[1] - np.asarray(static.s)[0]) if ns >= 2 else 0.0
    wint = np.asarray(vmec_wint_from_trig(trig, nzeta=int(sqrtg.shape[2])))
    wblocal = np.sum(wint[None, :, :] * sqrtg * (0.5 * bmag * bmag), axis=(1, 2))
    wb_calc = hs * abs(float(np.sum(wblocal[1:])))

    # VMEC: wp = hs*sum_{js=2..ns} vp(js)*pres(js).
    vp = np.asarray(wout.vp, dtype=float)
    pres = np.asarray(wout.pres, dtype=float)
    wp_calc = hs * float(np.sum(vp[1:] * pres[1:]))

    assert np.isfinite(wb_calc)
    assert np.isfinite(wp_calc)
    assert _rel_err(wb_calc, float(wout.wb)) < 5e-13
    assert _rel_err(wp_calc, float(wout.wp)) < 5e-13
