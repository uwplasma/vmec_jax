from __future__ import annotations

import numpy as np
import pytest


def test_step9_implicit_fixed_boundary_grad_matches_finite_difference():
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    from dataclasses import replace

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.config import load_config
    from vmec_jax.field import lamscale_from_phips
    from vmec_jax.geom import eval_geom
    from vmec_jax.implicit import ImplicitFixedBoundaryOptions, solve_fixed_boundary_state_implicit
    from vmec_jax.integrals import volume_from_sqrtg
    from vmec_jax.static import build_static
    from vmec_jax.wout import read_wout, state_from_wout

    enable_x64(True)

    cfg, _indata = load_config("examples/input.circular_tokamak")
    wout = read_wout("examples/wout_circular_tokamak_reference.nc")

    # Use a slightly higher angular resolution for smoother volume sensitivity.
    ntheta = max(int(cfg.ntheta), 32)
    ntheta = 2 * (ntheta // 2)
    cfg = replace(cfg, ntheta=ntheta, nzeta=max(int(cfg.nzeta), 1))
    static = build_static(cfg)

    st0 = state_from_wout(wout)
    signgs = int(wout.signgs)

    phipf0 = jnp.asarray(wout.phipf)
    chipf0 = jnp.asarray(wout.chipf)
    pressure0 = jnp.asarray(wout.presf)
    lamscale0 = lamscale_from_phips(jnp.asarray(wout.phips), jnp.asarray(static.s))

    def V_equilibrium(alpha):
        st = solve_fixed_boundary_state_implicit(
            st0,
            static,
            phipf=phipf0,
            chipf=alpha * chipf0,
            signgs=signgs,
            lamscale=lamscale0,
            pressure=pressure0,
            gamma=float(wout.gamma),
            jacobian_penalty=1e3,
            solver="lbfgs",
            max_iter=18,
            step_size=1.0,
            history_size=8,
            grad_tol=1e-10,
            preconditioner="mode_diag+radial_tridi",
            precond_exponent=1.0,
            precond_radial_alpha=0.5,
            implicit=ImplicitFixedBoundaryOptions(cg_max_iter=50, cg_tol=1e-10, damping=1e-5),
        )
        g = eval_geom(st, static)
        _dvds, V = volume_from_sqrtg(g.sqrtg, static.s, static.grid.theta, static.grid.zeta, nfp=int(cfg.nfp))
        return V[-1] * float(cfg.nfp)

    a0 = 1.0
    g_imp = float(np.asarray(jax.grad(V_equilibrium)(a0)))

    eps = 2e-2
    Vp = float(np.asarray(V_equilibrium(a0 + eps)))
    Vm = float(np.asarray(V_equilibrium(a0 - eps)))
    g_fd = (Vp - Vm) / (2.0 * eps)

    assert np.isfinite(g_imp)
    assert np.isfinite(g_fd)
    assert np.isclose(g_imp, g_fd, rtol=1e-1, atol=5e-4)
