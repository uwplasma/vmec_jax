from __future__ import annotations

from pathlib import Path

import numpy as np

from vmec_jax.namelist import read_indata
from vmec_jax.modes import vmec_mode_table
from vmec_jax.grids import make_angle_grid
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.fourier import build_helical_basis, eval_fourier


def test_circular_boundary(tmp_path: Path):
    # R = R0 + a cos(theta), Z = a sin(theta)
    R0 = 6.0
    a = 2.0
    txt = f"""
&INDATA
  NFP = 1
  MPOL = 6
  NTOR = 0
  NTHETA = 32
  NZETA = 1
  RBC(0,0) = {R0}
  RBC(0,1) = {a}
  ZBS(0,1) = {a}
/
"""
    p = tmp_path / "input.test"
    p.write_text(txt)

    indata = read_indata(p)
    modes = vmec_mode_table(mpol=6, ntor=0)
    grid = make_angle_grid(ntheta=32, nzeta=1, nfp=1)
    bdy = boundary_from_indata(indata, modes)
    basis = build_helical_basis(modes, grid)

    R = np.asarray(eval_fourier(bdy.R_cos, bdy.R_sin, basis))
    Z = np.asarray(eval_fourier(bdy.Z_cos, bdy.Z_sin, basis))

    theta = grid.theta
    assert np.allclose(R[:, 0], R0 + a * np.cos(theta), atol=1e-12)
    assert np.allclose(Z[:, 0], a * np.sin(theta), atol=1e-12)
