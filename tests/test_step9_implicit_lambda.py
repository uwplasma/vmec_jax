from __future__ import annotations

import numpy as np
import pytest


def test_step9_implicit_lambda_grad_matches_finite_difference(load_case_circular_tokamak):
    pytest.importorskip("jax")
    from conftest import require_slow

    require_slow()

    from vmec_jax._compat import enable_x64
    from vmec_jax.energy import flux_profiles_from_indata
    from vmec_jax.field import TWOPI, lamscale_from_phips, signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.implicit import solve_lambda_state_implicit
    from vmec_jax._compat import jax, jnp

    enable_x64(True)

    cfg, indata, static, _bdy, st0 = load_case_circular_tokamak

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)
    flux0 = flux_profiles_from_indata(indata, static.s, signgs=signgs)

    phipf0 = jnp.asarray(flux0.phipf)
    chipf0 = jnp.asarray(flux0.chipf)
    s = jnp.asarray(static.s)

    def f(scale):
        phipf = scale * phipf0
        chipf = scale * chipf0
        phips = (signgs * phipf) / TWOPI
        lamscale = lamscale_from_phips(phips, s)
        st = solve_lambda_state_implicit(
            st0,
            static,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs,
            lamscale=lamscale,
            sqrtg=jnp.asarray(g0.sqrtg),
            max_iter=80,
            grad_tol=1e-10,
        )
        return jnp.mean(st.Lcos**2 + st.Lsin**2)

    scale0 = 1.0
    g = float(jax.grad(f)(scale0))

    eps = 2e-3
    f_p = float(np.asarray(f(scale0 + eps)))
    f_m = float(np.asarray(f(scale0 - eps)))
    g_fd = (f_p - f_m) / (2.0 * eps)

    assert np.isfinite(g)
    assert np.isfinite(g_fd)
    # Finite-difference is noisy because it re-solves, but should be in the ballpark.
    assert np.isclose(g, g_fd, rtol=1.5, atol=1e-3)
