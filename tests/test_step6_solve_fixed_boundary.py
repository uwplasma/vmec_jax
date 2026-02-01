from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.profiles import eval_profiles
from vmec_jax.solve import solve_fixed_boundary_gd


def test_step6_fixed_boundary_solve_decreases_energy(load_case_lsp_low_res):
    pytest.importorskip("jax")

    _cfg, indata, static, _bdy, st0 = load_case_lsp_low_res

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
        max_iter=5,
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

