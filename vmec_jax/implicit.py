"""Implicit differentiation utilities (step-9).

This module provides *custom VJP* wrappers for equilibrium sub-solves so that
outer objectives can differentiate through equilibrium states without
backpropagating through many optimization iterations.

Initial scope (step-9)
======================

Implicit differentiation for the **lambda-only** fixed-geometry solve.

Future work
===========

Extend to the full fixed-boundary solve over (R,Z,lambda) using the same
implicit-function machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np

from ._compat import has_jax, jax, jnp
from .field import TWOPI, b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda
from .fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from .geom import eval_geom
from .solve import _enforce_lambda_gauge, _mask_grad_for_constraints, _mode00_index, solve_fixed_boundary_gd, solve_fixed_boundary_lbfgs, solve_lambda_gd
from .state import VMECState, pack_state, unpack_state


@dataclass(frozen=True)
class ImplicitLambdaOptions:
    """Controls for the implicit backward pass."""

    cg_max_iter: int = 80
    cg_tol: float = 1e-10
    damping: float = 1e-6


@dataclass(frozen=True)
class ImplicitFixedBoundaryOptions:
    """Controls for the implicit backward pass (fixed-boundary solve)."""

    cg_max_iter: int = 80
    cg_tol: float = 1e-10
    damping: float = 1e-6


def _stop_gradient_tree(x):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, x)


def _flatten_L(Lcos, Lsin) -> Any:
    return jnp.concatenate([jnp.ravel(Lcos), jnp.ravel(Lsin)], axis=0)


def _unflatten_L(x, *, shape: Tuple[int, int]):
    ns, K = shape
    n = ns * K
    x = jnp.asarray(x)
    Lcos = jnp.reshape(x[:n], (ns, K))
    Lsin = jnp.reshape(x[n:], (ns, K))
    return Lcos, Lsin


def _cg_solve(
    matvec: Callable[[Any], Any],
    b: Any,
    *,
    x0: Any | None = None,
    tol: float,
    max_iter: int,
) -> Any:
    """Conjugate gradients for SPD systems (JAX-friendly)."""
    b = jnp.asarray(b)
    x = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    r = b - matvec(x)
    p = r
    rs = jnp.dot(r, r)

    tol2 = jnp.asarray(float(tol) ** 2, dtype=b.dtype)
    max_iter = int(max_iter)

    def cond_fun(carry):
        i, x, r, p, rs = carry
        return jnp.logical_and(i < max_iter, rs > tol2)

    def body_fun(carry):
        i, x, r, p, rs = carry
        Ap = matvec(p)
        alpha = rs / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / rs
        p_new = r_new + beta * p
        return i + 1, x_new, r_new, p_new, rs_new

    _i, x, _r, _p, _rs = jax.lax.while_loop(cond_fun, body_fun, (0, x, r, p, rs))
    return x


def solve_lambda_state_implicit(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    sqrtg: Any | None = None,
    max_iter: int = 60,
    step_size: float = 0.05,
    grad_tol: float = 1e-10,
    bt_factor: float = 0.5,
    max_backtracks: int = 12,
    implicit: ImplicitLambdaOptions | None = None,
) -> VMECState:
    """Solve lambda with a custom VJP that uses implicit differentiation.

    Notes
    -----
    - This wrapper intentionally treats ``state0`` and ``static`` as *constants*
      for differentiation purposes. Gradients are provided w.r.t. ``phipf``,
      ``chipf``, and ``lamscale``.
    - The backward pass solves a damped linear system involving the Hessian of
      ``wb(L)`` w.r.t. lambda coefficients using conjugate gradients and Hessian-
      vector products computed via ``jax.jvp``.
    """
    if not has_jax():
        raise ImportError("solve_lambda_state_implicit requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitLambdaOptions()

    state0_c = _stop_gradient_tree(state0)

    idx00 = _mode00_index(static.modes)
    nfp = int(static.cfg.nfp)

    # Metric depends only on R/Z, so compute it once from the fixed geometry.
    g0 = eval_geom(state0_c, static)
    gtt = jnp.asarray(g0.g_tt)
    gtp = jnp.asarray(g0.g_tp)
    gpp = jnp.asarray(g0.g_pp)
    sqrtg_use = jnp.asarray(g0.sqrtg if sqrtg is None else sqrtg)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta = theta[1] - theta[0]
    dzeta = zeta[1] - zeta[0]
    weight = ds * dtheta * dzeta

    signgs_i = int(signgs)

    def _wb_from_L(Lcos, Lsin, phipf, chipf, lamscale):
        lam_u = eval_fourier_dtheta(Lcos, Lsin, static.basis)
        lam_v = eval_fourier_dzeta_phys(Lcos, Lsin, static.basis) / nfp
        bsupu, bsupv = bsup_from_sqrtg_lambda(
            sqrtg=sqrtg_use,
            lam_u=lam_u,
            lam_v=lam_v,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs_i,
            lamscale=lamscale,
        )
        B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
        jac = signgs_i * sqrtg_use
        E_total = jnp.sum(0.5 * B2 * jac) * weight
        return E_total / (TWOPI * TWOPI)

    def _solve(phipf, chipf, lamscale):
        # Forward solve uses the existing robust optimizer (not differentiated through).
        res = solve_lambda_gd(
            state0_c,
            static,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs_i,
            lamscale=lamscale,
            sqrtg=sqrtg_use,
            max_iter=int(max_iter),
            step_size=float(step_size),
            grad_tol=float(grad_tol),
            bt_factor=float(bt_factor),
            max_backtracks=int(max_backtracks),
        )
        return res.state

    @jax.custom_vjp
    def _solve_cust(phipf, chipf, lamscale):
        return _solve(phipf, chipf, lamscale)

    def fwd(phipf, chipf, lamscale):
        st = _solve(phipf, chipf, lamscale)
        return st, (jnp.asarray(st.Lcos), jnp.asarray(st.Lsin), jnp.asarray(phipf), jnp.asarray(chipf), jnp.asarray(lamscale))

    def bwd(residual, ct_state):
        Lcos_star, Lsin_star, phipf_star, chipf_star, lamscale_star = residual
        ns, K = Lcos_star.shape

        # Gauge mask: exclude (m,n)=(0,0) everywhere.
        mask = jnp.ones((ns, K), dtype=Lcos_star.dtype)
        if idx00 is not None:
            mask = mask.at[:, int(idx00)].set(0.0)

        def grad_L_flat(Lcos, Lsin, phipf, chipf, lamscale):
            gcos, gsin = jax.grad(_wb_from_L, argnums=(0, 1))(Lcos, Lsin, phipf, chipf, lamscale)
            gcos = gcos * mask
            gsin = gsin * mask
            return _flatten_L(gcos, gsin)

        x_star = _flatten_L(Lcos_star, Lsin_star)

        def Hvp(u_flat):
            ucos, usin = _unflatten_L(u_flat, shape=(ns, K))
            ucos = ucos * mask
            usin = usin * mask

            def grad_pair(Lcos, Lsin):
                gcos, gsin = jax.grad(_wb_from_L, argnums=(0, 1))(Lcos, Lsin, phipf_star, chipf_star, lamscale_star)
                gcos = gcos * mask
                gsin = gsin * mask
                return gcos, gsin

            (_gcos, _gsin), (tcos, tsin) = jax.jvp(grad_pair, (Lcos_star, Lsin_star), (ucos, usin))
            t = _flatten_L(tcos * mask, tsin * mask)
            return t + jnp.asarray(float(implicit.damping), dtype=t.dtype) * u_flat

        # Right-hand side is cotangent w.r.t. the output state's lambda coefficients.
        ct_Lcos = jnp.asarray(ct_state.Lcos) * mask
        ct_Lsin = jnp.asarray(ct_state.Lsin) * mask
        b = _flatten_L(ct_Lcos, ct_Lsin)

        v = _cg_solve(Hvp, b, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))

        # Gradient w.r.t. parameters: dL/dp = - v^T (∂/∂p grad_L).
        def F_params(phipf, chipf, lamscale):
            return grad_L_flat(Lcos_star, Lsin_star, phipf, chipf, lamscale)

        (_out, vjp_fun) = jax.vjp(F_params, phipf_star, chipf_star, lamscale_star)
        dphipf, dchipf, dlamscale = vjp_fun(v)
        return (-dphipf, -dchipf, -dlamscale)

    _solve_cust.defvjp(fwd, bwd)

    return _solve_cust(jnp.asarray(phipf), jnp.asarray(chipf), jnp.asarray(lamscale))


def solve_fixed_boundary_state_implicit(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    pressure,
    gamma: float = 0.0,
    jacobian_penalty: float = 1e3,
    solver: str = "lbfgs",
    max_iter: int = 25,
    step_size: float = 5e-3,
    history_size: int = 10,
    grad_tol: float = 1e-10,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    implicit: ImplicitFixedBoundaryOptions | None = None,
) -> VMECState:
    """Fixed-boundary solve with a custom VJP using implicit differentiation.

    This is a step-9 building block: it returns an equilibrium state while
    exposing *implicit* gradients w.r.t. the 1D profiles/fluxes.

    Differentiable inputs (by design)
    ---------------------------------
    - ``phipf(s)``, ``chipf(s)``, ``pressure(s)``, and ``lamscale``.

    Notes
    -----
    - ``state0`` and ``static`` are treated as constants for differentiation.
    - The backward pass solves a damped linear system involving the Hessian of
      the total objective w.r.t. the (masked) Fourier coefficients using CG and
      Hessian-vector products computed via ``jax.jvp``.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_state_implicit requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitFixedBoundaryOptions()

    solver = str(solver).strip().lower()
    if solver not in ("gd", "lbfgs"):
        raise ValueError(f"solver must be 'gd' or 'lbfgs', got {solver!r}")

    state0_c = _stop_gradient_tree(state0)
    idx00 = _mode00_index(static.modes)

    signgs_i = int(signgs)
    nfp = int(static.cfg.nfp)
    gamma = float(gamma)
    jacobian_penalty = float(jacobian_penalty)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta = theta[1] - theta[0]
    dzeta = zeta[1] - zeta[0]
    weight = ds * dtheta * dzeta

    def _objective(state: VMECState, phipf, chipf, pressure, lamscale):
        g = eval_geom(state, static)
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs_i, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs_i * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        w = wb + wp / (gamma - 1.0)
        # Softly enforce a consistent Jacobian sign away from the axis.
        jac2 = jac.at[0, :, :].set(0.0)
        neg = jnp.minimum(jac2, 0.0)
        penalty = jacobian_penalty * jnp.mean(neg * neg)
        return w + penalty

    def _grad_flat(state: VMECState, phipf, chipf, pressure, lamscale):
        g = jax.grad(_objective)(state, phipf, chipf, pressure, lamscale)
        g = _mask_grad_for_constraints(g, static, idx00=idx00)
        return pack_state(g)

    def _solve(phipf, chipf, pressure, lamscale):
        if solver == "gd":
            res = solve_fixed_boundary_gd(
                state0_c,
                static,
                phipf=phipf,
                chipf=chipf,
                signgs=signgs_i,
                lamscale=lamscale,
                pressure=pressure,
                gamma=gamma,
                jacobian_penalty=jacobian_penalty,
                max_iter=int(max_iter),
                step_size=float(step_size),
                grad_tol=float(grad_tol),
                max_backtracks=int(max_backtracks),
                bt_factor=float(bt_factor),
                preconditioner=str(preconditioner),
                precond_exponent=float(precond_exponent),
                precond_radial_alpha=float(precond_radial_alpha),
            )
        else:
            res = solve_fixed_boundary_lbfgs(
                state0_c,
                static,
                phipf=phipf,
                chipf=chipf,
                signgs=signgs_i,
                lamscale=lamscale,
                pressure=pressure,
                gamma=gamma,
                history_size=int(history_size),
                max_iter=int(max_iter),
                step_size=float(step_size),
                grad_tol=float(grad_tol),
                max_backtracks=int(max_backtracks),
                bt_factor=float(bt_factor),
                preconditioner=str(preconditioner),
                precond_exponent=float(precond_exponent),
                precond_radial_alpha=float(precond_radial_alpha),
            )
        return res.state

    @jax.custom_vjp
    def _solve_cust(phipf, chipf, pressure, lamscale):
        return _solve(phipf, chipf, pressure, lamscale)

    def fwd(phipf, chipf, pressure, lamscale):
        st = _solve(phipf, chipf, pressure, lamscale)
        return st, (
            _stop_gradient_tree(st),
            jnp.asarray(phipf),
            jnp.asarray(chipf),
            jnp.asarray(pressure),
            jnp.asarray(lamscale),
        )

    def bwd(residual, ct_state):
        st_star, phipf_star, chipf_star, pressure_star, lamscale_star = residual
        layout = st_star.layout

        ct_state = _mask_grad_for_constraints(ct_state, static, idx00=idx00)
        b = pack_state(ct_state)

        def Hvp(u_flat):
            u_state = unpack_state(u_flat, layout)
            u_state = _mask_grad_for_constraints(u_state, static, idx00=idx00)
            _, hvp = jax.jvp(
                lambda st: _grad_flat(st, phipf_star, chipf_star, pressure_star, lamscale_star),
                (st_star,),
                (u_state,),
            )
            return hvp + jnp.asarray(float(implicit.damping), dtype=hvp.dtype) * u_flat

        v = _cg_solve(Hvp, b, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))

        def F_params(phipf, chipf, pressure, lamscale):
            return _grad_flat(st_star, phipf, chipf, pressure, lamscale)

        (_out, vjp_fun) = jax.vjp(F_params, phipf_star, chipf_star, pressure_star, lamscale_star)
        dphipf, dchipf, dpressure, dlamscale = vjp_fun(v)
        return (-dphipf, -dchipf, -dpressure, -dlamscale)

    _solve_cust.defvjp(fwd, bwd)

    return _solve_cust(jnp.asarray(phipf), jnp.asarray(chipf), jnp.asarray(pressure), jnp.asarray(lamscale))
