from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.profiles import eval_profiles
from vmec_jax.solve import solve_fixed_boundary_gd


def test_step8_mode_diag_preconditioner_runs_and_decreases_energy(load_case_li383_low_res):
    pytest.importorskip("jax")

    _cfg, indata, static, _bdy, st0 = load_case_li383_low_res

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    prof = eval_profiles(indata, static.s)
    pressure = prof.get("pressure", np.zeros_like(np.asarray(static.s)))
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
        max_iter=4,
        step_size=5e-3,
        jacobian_penalty=1e3,
        preconditioner="mode_diag",
        precond_exponent=1.0,
    )

    assert res.w_history.shape[0] >= 2
    assert np.isfinite(res.w_history).all()
    assert float(res.w_history[-1]) < float(res.w_history[0])
    assert np.all(np.diff(res.w_history) < 0.0)


def test_step8_radial_tridi_preconditioner_runs_and_decreases_energy(load_case_li383_low_res):
    pytest.importorskip("jax")

    _cfg, indata, static, _bdy, st0 = load_case_li383_low_res

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    prof = eval_profiles(indata, static.s)
    pressure = prof.get("pressure", np.zeros_like(np.asarray(static.s)))
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
        max_iter=4,
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
