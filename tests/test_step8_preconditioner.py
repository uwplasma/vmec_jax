from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class _TinyCase:
    indata: InData
    static: any
    st0: any


def _tiny_axisym_case() -> _TinyCase:
    cfg = VMECConfig(ns=9, mpol=4, ntor=0, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=16, nzeta=4)
    static = build_static(cfg)
    K = int(static.modes.K)

    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)

    k00 = _k_index(static.modes, 0, 0)
    k10 = _k_index(static.modes, 1, 0)

    # Simple shaped boundary: R = R0 + a*cos(theta), Z = b*sin(theta)
    Rcos[k00] = 3.0
    Rcos[k10] = 1.0
    Zsin[k10] = 0.6

    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [3.0], "ZAXIS_CS": [0.0], "PHIEDGE": 1.0, "GAMMA": 0.0}, indexed={})
    st0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)
    return _TinyCase(indata=indata, static=static, st0=st0)


def test_step8_mode_diag_preconditioner_runs_and_decreases_energy():
    pytest.importorskip("jax")

    case = _tiny_axisym_case()
    indata, static, st0 = case.indata, case.static, case.st0

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
        preconditioner="mode_diag",
        precond_exponent=1.0,
    )

    assert res.w_history.shape[0] >= 2
    assert np.isfinite(res.w_history).all()
    assert float(res.w_history[-1]) < float(res.w_history[0])
    assert np.all(np.diff(res.w_history) < 0.0)


def test_step8_radial_tridi_preconditioner_runs_and_decreases_energy():
    pytest.importorskip("jax")

    case = _tiny_axisym_case()
    indata, static, st0 = case.indata, case.static, case.st0

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
        preconditioner="radial_tridi",
        precond_exponent=1.0,
        precond_radial_alpha=0.5,
    )

    assert res.w_history.shape[0] >= 2
    assert np.isfinite(res.w_history).all()
    assert float(res.w_history[-1]) < float(res.w_history[0])
    assert np.all(np.diff(res.w_history) < 0.0)
