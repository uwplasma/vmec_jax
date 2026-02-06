import numpy as np
import pytest

from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.config import VMECConfig
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.namelist import InData
from vmec_jax.static import build_static


def _k_index(modes, m, n):
    for k, (mm, nn) in enumerate(zip(modes.m, modes.n)):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def test_initial_guess_scaling_and_axis_blend():
    cfg = VMECConfig(mpol=3, ntor=2, ns=5, nfp=1, lasym=False, lconm1=True, ntheta=8, nzeta=6)
    static = build_static(cfg)
    K = static.modes.K

    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)

    k00 = _k_index(static.modes, 0, 0)
    k10 = _k_index(static.modes, 1, 0)
    k20 = _k_index(static.modes, 2, 0)

    Rcos[k00] = 10.0
    Rcos[k10] = 2.0
    Rcos[k20] = 3.0
    Rsin[k00] = 4.0

    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [5.0], "ZAXIS_CS": [0.0]}, indexed={})

    st0 = initial_guess_from_boundary(static, boundary, indata)
    s = np.asarray(static.s)
    rho = np.sqrt(s)

    # m>0 scaling uses rho**m
    assert st0.Rcos[1, k10] == pytest.approx(rho[1] * 2.0)
    assert st0.Rcos[1, k20] == pytest.approx((rho[1] ** 2) * 3.0)

    # m=0 Rcos blends between axis and boundary
    assert st0.Rcos[0, k00] == pytest.approx(5.0)
    assert st0.Rcos[-1, k00] == pytest.approx(10.0)
    assert st0.Rcos[2, k00] == pytest.approx(7.5)

    # m=0 non-Rcos components scale with s for regularity
    assert st0.Rsin[0, k00] == pytest.approx(0.0)
    assert st0.Rsin[-1, k00] == pytest.approx(4.0)
