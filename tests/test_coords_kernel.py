import numpy as np

from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.config import VMECConfig
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.namelist import InData
from vmec_jax.static import build_static
from vmec_jax.coords import eval_coords
from vmec_jax.fourier import eval_fourier


def test_boundary_matches_state_surface():
    cfg = VMECConfig(ns=7, mpol=3, ntor=0, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=12, nzeta=3)
    static = build_static(cfg)
    K = int(static.modes.K)

    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)

    k00 = int(np.where((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0][0])
    k10 = int(np.where((np.asarray(static.modes.m) == 1) & (np.asarray(static.modes.n) == 0))[0][0])
    Rcos[k00] = 3.0
    Rcos[k10] = 1.0
    Zsin[k10] = 0.6

    bdy = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [3.0], "ZAXIS_CS": [0.0]}, indexed={})
    state0 = initial_guess_from_boundary(static, bdy, indata, vmec_project=False)

    coords = eval_coords(state0, static.basis)
    R = np.asarray(coords.R)
    Z = np.asarray(coords.Z)

    Rb = np.asarray(eval_fourier(bdy.R_cos, bdy.R_sin, static.basis))
    Zb = np.asarray(eval_fourier(bdy.Z_cos, bdy.Z_sin, static.basis))

    # Should be identical (up to floating point) since s=1 uses boundary coefficients.
    assert np.max(np.abs(R[-1] - Rb)) < 1e-12
    assert np.max(np.abs(Z[-1] - Zb)) < 1e-12
