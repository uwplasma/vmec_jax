from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.config import VMECConfig
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.namelist import InData
from vmec_jax.static import build_static
from vmec_jax.solve import solve_fixed_boundary_gd


def _k_index(modes, m, n):
    for k, (mm, nn) in enumerate(zip(modes.m, modes.n)):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def test_step6_fixed_boundary_solve_decreases_energy():
    pytest.importorskip("jax")

    cfg = VMECConfig(ns=7, mpol=3, ntor=0, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=12, nzeta=3)
    static = build_static(cfg)
    K = int(static.modes.K)
    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)
    k00 = _k_index(static.modes, 0, 0)
    k10 = _k_index(static.modes, 1, 0)
    Rcos[k00] = 3.0
    Rcos[k10] = 1.0
    Zsin[k10] = 0.6
    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [3.0], "ZAXIS_CS": [0.0], "PHIEDGE": 1.0, "GAMMA": 0.0}, indexed={})
    st0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    pressure = np.zeros_like(np.asarray(static.s))
    gamma = indata.get_float("GAMMA", 0.0)

    res = solve_fixed_boundary_gd(
        st0,
        static,
        phipf=flux.phipf,
        chipf=flux.chipf,
        signgs=signgs,
        lamscale=flux.lamscale,
        pressure=pressure,
        gamma=gamma,
        max_iter=2,
        step_size=5e-3,
        jacobian_penalty=1e3,
    )

    assert res.w_history.shape[0] >= 2
    assert np.isfinite(res.w_history).all()
    assert float(res.w_history[-1]) < float(res.w_history[0])
    assert np.all(np.diff(res.w_history) < 0.0)

    # Fixed-boundary constraint: edge coefficients are preserved exactly.
    for name in ("Rcos", "Rsin", "Zcos", "Zsin"):
        a0 = np.asarray(getattr(st0, name))[-1, :]
        a1 = np.asarray(getattr(res.state, name))[-1, :]
        assert np.max(np.abs(a1 - a0)) < 1e-14

    # Axis regularity: all m>0 coefficients are forced to 0 on s=0.
    m = np.asarray(static.modes.m)
    mask = m > 0
    for name in ("Rcos", "Rsin", "Zcos", "Zsin"):
        a = np.asarray(getattr(res.state, name))[0, mask]
        assert np.max(np.abs(a)) < 1e-14
