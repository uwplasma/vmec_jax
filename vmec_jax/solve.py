"""Step-5 solvers (fixed-boundary, early stages).

The first solver milestone is a robust "inner solve" for the VMEC ``lambda`` field
with R/Z held fixed. This is useful for:

- validating the magnetic energy objective against VMEC2000 `wout` files,
- building toward a full fixed-boundary equilibrium solve in later steps.

Notes
-----
This module intentionally avoids optional dependencies (e.g. jaxopt). The current
implementation uses gradient descent with a simple backtracking line search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._compat import has_jax, jax, jnp, jit
from .field import TWOPI, b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda
from .fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from .geom import eval_geom
from .grids import angle_steps
from .state import VMECState, pack_state, unpack_state


@dataclass(frozen=True)
class SolveLambdaResult:
    state: VMECState
    n_iter: int
    wb_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SolveFixedBoundaryResult:
    state: VMECState
    n_iter: int
    w_history: np.ndarray
    wb_history: np.ndarray
    wp_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SolveVmecResidualResult:
    state: VMECState
    n_iter: int
    w_history: np.ndarray
    fsqr2_history: np.ndarray
    fsqz2_history: np.ndarray
    fsql2_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


def _mode00_index(modes) -> Optional[int]:
    m = np.asarray(modes.m)
    n = np.asarray(modes.n)
    idx = np.where((m == 0) & (n == 0))[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def _enforce_lambda_gauge(Lcos, Lsin, *, idx00: Optional[int]):
    """Fix the (m,n)=(0,0) gauge mode to 0 (it is a nullspace)."""
    if idx00 is None:
        return Lcos, Lsin
    if hasattr(Lcos, "at"):
        # JAX arrays support .at[] updates.
        Lcos = Lcos.at[:, idx00].set(0.0)
        Lsin = Lsin.at[:, idx00].set(0.0)
        return Lcos, Lsin
    # numpy fallback (not performance critical here)
    Lcos = np.asarray(Lcos).copy()
    Lsin = np.asarray(Lsin).copy()
    Lcos[:, idx00] = 0.0
    Lsin[:, idx00] = 0.0
    return Lcos, Lsin


def _axis_m0_mask(static, *, dtype):
    m = jnp.asarray(static.modes.m)
    return (m == 0).astype(dtype)


def _enforce_fixed_boundary_and_axis(
    state: VMECState,
    static,
    *,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    enforce_axis: bool = True,
    enforce_edge: bool = True,
    enforce_lambda_axis: bool = True,
    idx00: Optional[int],
) -> VMECState:
    """Apply minimal VMEC regularity + fixed-boundary constraints.

    - Fix R/Z at the outer surface (s=1) to preserve the prescribed boundary.
    - Enforce axis regularity by zeroing all m>0 Fourier coefficients at s=0.
    - Enforce lambda gauge (m,n)=(0,0) = 0 everywhere.
    """
    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    Lcos = jnp.asarray(state.Lcos)
    Lsin = jnp.asarray(state.Lsin)

    if enforce_edge:
        Rcos = Rcos.at[-1, :].set(jnp.asarray(edge_Rcos))
        Rsin = Rsin.at[-1, :].set(jnp.asarray(edge_Rsin))
        Zcos = Zcos.at[-1, :].set(jnp.asarray(edge_Zcos))
        Zsin = Zsin.at[-1, :].set(jnp.asarray(edge_Zsin))

    if enforce_axis:
        mask_m0 = _axis_m0_mask(static, dtype=Rcos.dtype)
        Rcos = Rcos.at[0, :].set(Rcos[0, :] * mask_m0)
        Rsin = Rsin.at[0, :].set(Rsin[0, :] * mask_m0)
        Zcos = Zcos.at[0, :].set(Zcos[0, :] * mask_m0)
        Zsin = Zsin.at[0, :].set(Zsin[0, :] * mask_m0)

    if enforce_lambda_axis:
        Lcos = Lcos.at[0, :].set(0.0)
        Lsin = Lsin.at[0, :].set(0.0)

    Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)

    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


def _grad_rms_state(grad: VMECState) -> float:
    g = np.asarray(grad.Rcos) ** 2
    g = g + np.asarray(grad.Rsin) ** 2
    g = g + np.asarray(grad.Zcos) ** 2
    g = g + np.asarray(grad.Zsin) ** 2
    g = g + np.asarray(grad.Lcos) ** 2
    g = g + np.asarray(grad.Lsin) ** 2
    return float(np.sqrt(np.mean(g)))


def _update_state_gd(state: VMECState, grad: VMECState, *, step: float, scale_rz: float, scale_l: float) -> VMECState:
    step = jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype)
    scale_rz = jnp.asarray(scale_rz, dtype=step.dtype)
    scale_l = jnp.asarray(scale_l, dtype=step.dtype)
    return VMECState(
        layout=state.layout,
        Rcos=jnp.asarray(state.Rcos) - step * scale_rz * jnp.asarray(grad.Rcos),
        Rsin=jnp.asarray(state.Rsin) - step * scale_rz * jnp.asarray(grad.Rsin),
        Zcos=jnp.asarray(state.Zcos) - step * scale_rz * jnp.asarray(grad.Zcos),
        Zsin=jnp.asarray(state.Zsin) - step * scale_rz * jnp.asarray(grad.Zsin),
        Lcos=jnp.asarray(state.Lcos) - step * scale_l * jnp.asarray(grad.Lcos),
        Lsin=jnp.asarray(state.Lsin) - step * scale_l * jnp.asarray(grad.Lsin),
    )


def _mask_grad_for_constraints(
    grad: VMECState,
    static,
    *,
    idx00: Optional[int],
    mask_lambda_axis: bool = True,
) -> VMECState:
    """Project gradients onto the feasible set implied by our constraints."""
    gRcos = jnp.asarray(grad.Rcos)
    gRsin = jnp.asarray(grad.Rsin)
    gZcos = jnp.asarray(grad.Zcos)
    gZsin = jnp.asarray(grad.Zsin)
    gLcos = jnp.asarray(grad.Lcos)
    gLsin = jnp.asarray(grad.Lsin)

    # Fixed-boundary: don't update the edge surface for R/Z.
    gRcos = gRcos.at[-1, :].set(0.0)
    gRsin = gRsin.at[-1, :].set(0.0)
    gZcos = gZcos.at[-1, :].set(0.0)
    gZsin = gZsin.at[-1, :].set(0.0)

    # Axis regularity: don't update m>0 coefficients at s=0 for R/Z.
    m = jnp.asarray(static.modes.m)
    mask_m0 = (m == 0).astype(gRcos.dtype)
    gRcos = gRcos.at[0, :].set(gRcos[0, :] * mask_m0)
    gRsin = gRsin.at[0, :].set(gRsin[0, :] * mask_m0)
    gZcos = gZcos.at[0, :].set(gZcos[0, :] * mask_m0)
    gZsin = gZsin.at[0, :].set(gZsin[0, :] * mask_m0)

    # Lambda: optionally fix the axis row (older step-5 behavior).
    if bool(mask_lambda_axis):
        gLcos = gLcos.at[0, :].set(0.0)
        gLsin = gLsin.at[0, :].set(0.0)

    # Lambda gauge: (m,n)=(0,0) stays 0 everywhere.
    if idx00 is not None:
        gLcos = gLcos.at[:, idx00].set(0.0)
        gLsin = gLsin.at[:, idx00].set(0.0)

    return VMECState(
        layout=grad.layout,
        Rcos=gRcos,
        Rsin=gRsin,
        Zcos=gZcos,
        Zsin=gZsin,
        Lcos=gLcos,
        Lsin=gLsin,
    )


def _apply_preconditioner(
    grad: VMECState,
    static,
    *,
    kind: str,
    exponent: float = 1.0,
    radial_alpha: float = 0.0,
) -> VMECState:
    """Apply a simple diagonal preconditioner in (m,n) Fourier space.

    Parameters
    ----------
    kind:
        - ``"none"``: no preconditioning
        - ``"mode_diag"``: scale each (m,n) mode by ~(m^2 + (n*NFP)^2)^(-exponent)
        - ``"radial_tridi"``: apply a simple Dirichlet tri-diagonal smoother in s
        - ``"mode_diag+radial_tridi"``: apply both (order: mode, then radial)
    """
    kind = str(kind).strip().lower()
    if kind == "none":
        return grad

    kinds = [k.strip() for k in kind.replace("+", ",").split(",") if k.strip()]
    if not kinds:
        return grad

    exponent = float(exponent)
    if ("mode_diag" in kinds) and exponent <= 0.0:
        raise ValueError("preconditioner exponent must be > 0 for mode_diag")
    radial_alpha = float(radial_alpha)
    if ("radial_tridi" in kinds) and radial_alpha <= 0.0:
        raise ValueError("radial_alpha must be > 0 for radial_tridi")

    def _apply_mode_diag(g: VMECState) -> VMECState:
        m = jnp.asarray(static.modes.m)
        n = jnp.asarray(static.modes.n)
        nfp = float(static.cfg.nfp)
        k2 = m.astype(jnp.float64) ** 2 + (n.astype(jnp.float64) * nfp) ** 2
        # (1 + k2)^(-exponent) avoids singularity at (m,n)=(0,0).
        w = (1.0 + k2) ** (-exponent)
        w = w.astype(jnp.asarray(g.Rcos).dtype)

        def _scale(a):
            a = jnp.asarray(a)
            return a * w[None, :]

        return VMECState(
            layout=g.layout,
            Rcos=_scale(g.Rcos),
            Rsin=_scale(g.Rsin),
            Zcos=_scale(g.Zcos),
            Zsin=_scale(g.Zsin),
            Lcos=_scale(g.Lcos),
            Lsin=_scale(g.Lsin),
        )

    def _tridi_smooth_dirichlet(rhs, *, alpha: float):
        """Solve a simple tri-diagonal smoothing system along s for each mode.

        This applies a Dirichlet-boundary operator in s:

            (-α) x_{i-1} + (1+2α) x_i + (-α) x_{i+1} = rhs_i

        on interior points i=1..ns-2, treating x_0 and x_{ns-1} as fixed to rhs
        at those endpoints. This preserves any constraint-masked gradients at
        the endpoints while still coupling interior surfaces.
        """
        rhs = jnp.asarray(rhs)
        if rhs.ndim == 2:
            rhs2 = rhs
            orig_shape = None
        elif rhs.ndim == 3:
            ns = int(rhs.shape[0])
            rhs2 = rhs.reshape(ns, -1)
            orig_shape = rhs.shape
        else:
            raise ValueError(f"expected (ns,K) or (ns,M,N), got {rhs.shape}")
        ns = int(rhs2.shape[0])
        if ns < 3:
            return rhs
        alpha = jnp.asarray(alpha, dtype=rhs.dtype)
        a = -alpha
        b = 1.0 + 2.0 * alpha
        c = -alpha

        x0 = rhs2[0]
        xN = rhs2[-1]
        d = rhs2[1:-1]
        d = d.at[0].add(alpha * x0)
        d = d.at[-1].add(alpha * xN)

        n = int(d.shape[0])
        if n == 1:
            x_int = d / b
        else:
            # Forward sweep (Thomas algorithm), vectorized over modes K.
            cp0 = c / b
            dp0 = d[0] / b

            def fwd(carry, di):
                cp_prev, dp_prev = carry
                denom = b - a * cp_prev
                cp = c / denom
                dp = (di - a * dp_prev) / denom
                return (cp, dp), (cp, dp)

            (cp_last, dp_last), (cp_rest, dp_rest) = jax.lax.scan(fwd, (cp0, dp0), d[1:])
            cp = jnp.concatenate([jnp.asarray([cp0]), cp_rest], axis=0)
            dp = jnp.concatenate([dp0[None, :], dp_rest], axis=0)
            # Back substitution.
            x_last = dp_last

            def bwd(x_next, items):
                cpi, dpi = items
                xi = dpi - cpi * x_next
                return xi, xi

            _x0, x_rev = jax.lax.scan(bwd, x_last, (cp[:-1], dp[:-1]), reverse=True)
            x_int = jnp.concatenate([x_rev, x_last[None, :]], axis=0)

        return jnp.concatenate([x0[None, :], x_int, xN[None, :]], axis=0)

    def _apply_radial_tridi(g: VMECState) -> VMECState:
        return VMECState(
            layout=g.layout,
            Rcos=_tridi_smooth_dirichlet(g.Rcos, alpha=radial_alpha),
            Rsin=_tridi_smooth_dirichlet(g.Rsin, alpha=radial_alpha),
            Zcos=_tridi_smooth_dirichlet(g.Zcos, alpha=radial_alpha),
            Zsin=_tridi_smooth_dirichlet(g.Zsin, alpha=radial_alpha),
            Lcos=_tridi_smooth_dirichlet(g.Lcos, alpha=radial_alpha),
            Lsin=_tridi_smooth_dirichlet(g.Lsin, alpha=radial_alpha),
        )

    g = grad
    for k in kinds:
        if k == "mode_diag":
            g = _apply_mode_diag(g)
        elif k == "radial_tridi":
            g = _apply_radial_tridi(g)
        else:
            raise ValueError(f"Unknown preconditioner kind={k!r}")
    return g


def solve_lambda_gd(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    sqrtg: Any | None = None,
    max_iter: int = 50,
    step_size: float = 0.05,
    grad_tol: float = 1e-10,
    max_backtracks: int = 16,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveLambdaResult:
    """Solve for VMEC lambda (scaled coefficients) with fixed R/Z.

    Parameters
    ----------
    state0:
        Initial state. Only the lambda coefficients are updated.
    static:
        VMECStatic from :func:`vmec_jax.static.build_static`.
    phipf, chipf:
        1D flux functions (ns,) matching VMEC's `wout` meaning.
    signgs:
        Orientation (+1 or -1).
    lamscale:
        VMEC lambda scaling factor (see :func:`vmec_jax.field.lamscale_from_phips`).
    sqrtg:
        Optional signed Jacobian on the 3D grid. If provided (e.g. reconstructed from
        `wout` Nyquist coefficients), it is used for the objective and field formulas.
        Otherwise we use :func:`vmec_jax.geom.eval_geom`'s sqrtg.
    """
    if not has_jax():
        raise ImportError("solve_lambda_gd requires JAX (jax + jaxlib)")

    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")

    idx00 = _mode00_index(static.modes)
    preconditioner = str(preconditioner).strip().lower()
    if preconditioner not in ("none", "mode_diag"):
        raise ValueError(f"Unknown preconditioner kind={preconditioner!r}")
    precond_exponent = float(precond_exponent)
    if preconditioner != "none" and precond_exponent <= 0.0:
        raise ValueError("precond_exponent must be > 0 when using a preconditioner")

    # Metric depends only on R/Z, so compute it once.
    g0 = eval_geom(state0, static)
    gtt = jnp.asarray(g0.g_tt)
    gtp = jnp.asarray(g0.g_tp)
    gpp = jnp.asarray(g0.g_pp)

    sqrtg_use = jnp.asarray(g0.sqrtg if sqrtg is None else sqrtg)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    def _wb_from_L(Lcos, Lsin):
        lam_u = eval_fourier_dtheta(Lcos, Lsin, static.basis)
        lam_v = eval_fourier_dzeta_phys(Lcos, Lsin, static.basis) / nfp
        bsupu, bsupv = bsup_from_sqrtg_lambda(
            sqrtg=sqrtg_use,
            lam_u=lam_u,
            lam_v=lam_v,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs,
            lamscale=lamscale,
        )
        B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
        jac = signgs * sqrtg_use
        E_total = jnp.sum(0.5 * B2 * jac) * weight
        return E_total / (TWOPI * TWOPI)

    wb_and_grad = jax.value_and_grad(_wb_from_L, argnums=(0, 1))
    wb_only = _wb_from_L
    if jit_grad:
        wb_and_grad = jit(wb_and_grad)
        wb_only = jit(wb_only)

    Lcos = jnp.asarray(state0.Lcos)
    Lsin = jnp.asarray(state0.Lsin)
    Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)

    wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)
    wb_history = [float(np.asarray(wb0))]
    grad_rms_history = []
    step_history = []

    for it in range(max_iter):
        # Optional mode-diagonal preconditioning for the lambda subproblem.
        if preconditioner == "mode_diag":
            m = jnp.asarray(static.modes.m)
            n = jnp.asarray(static.modes.n)
            k2 = m.astype(jnp.float64) ** 2 + (n.astype(jnp.float64) * float(static.cfg.nfp)) ** 2
            w = (1.0 + k2) ** (-precond_exponent)
            w = w.astype(jnp.asarray(Lcos).dtype)
            gcos_p = gcos * w[None, :]
            gsin_p = gsin * w[None, :]
        else:
            gcos_p = gcos
            gsin_p = gsin

        grad_rms = float(np.sqrt(np.mean(np.asarray(gcos_p) ** 2 + np.asarray(gsin_p) ** 2)))
        grad_rms_history.append(grad_rms)

        if verbose:
            print(f"[solve_lambda_gd] iter={it:03d} wb={wb_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < grad_tol:
            break

        step = float(step_size)
        accepted = False

        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            Lcos_t = Lcos - step * gcos_p
            Lsin_t = Lsin - step * gsin_p
            Lcos_t, Lsin_t = _enforce_lambda_gauge(Lcos_t, Lsin_t, idx00=idx00)
            wb_t = wb_only(Lcos_t, Lsin_t)
            if float(np.asarray(wb_t)) < wb_history[-1]:
                accepted = True
                Lcos, Lsin, wb0 = Lcos_t, Lsin_t, wb_t
                break

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_lambda_gd] line search failed to improve objective; stopping")
            break

        wb_history.append(float(np.asarray(wb0)))
        wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)

    st = VMECState(
        layout=state0.layout,
        Rcos=state0.Rcos,
        Rsin=state0.Rsin,
        Zcos=state0.Zcos,
        Zsin=state0.Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )
    diag: Dict[str, Any] = {"idx00": idx00}
    return SolveLambdaResult(
        state=st,
        n_iter=len(wb_history) - 1,
        wb_history=np.asarray(wb_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_gd(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    pressure: Any | None = None,
    gamma: float = 0.0,
    jacobian_penalty: float = 1e3,
    max_iter: int = 25,
    step_size: float = 5e-3,
    scale_rz: float = 1.0,
    scale_l: float = 1.0,
    grad_tol: float = 1e-10,
    max_backtracks: int = 16,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveFixedBoundaryResult:
    """Minimize a VMEC-style energy objective over (R,Z,lambda) coefficients.

    This is the first "full" fixed-boundary solver step:
    - R/Z are evolved on interior surfaces only; the outer surface is held fixed.
    - Lambda gauge mode (0,0) is fixed to 0.

    The objective is::

        W = wb + wp/(gamma - 1)

    where ``wb`` is VMEC's normalized magnetic energy and
    ``wp = ∫ p dV /(2π)^2``.
    A soft penalty enforces a consistent Jacobian sign away from the axis.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_gd requires JAX (jax + jaxlib)")

    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")

    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("gamma=1 makes wp/(gamma-1) singular")

    idx00 = _mode00_index(static.modes)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    if pressure is None:
        pressure = jnp.zeros_like(s)
    pressure = jnp.asarray(pressure)
    if pressure.shape != s.shape:
        raise ValueError(f"pressure must have shape {s.shape}, got {pressure.shape}")

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    def _wb_wp_from_geom(g) -> Tuple[Any, Any]:
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        return wb, wp

    def _w_total_from_wb_wp(wb, wp) -> Any:
        return wb + wp / (gamma - 1.0)

    def _objective(state: VMECState) -> Any:
        # Softly enforce a consistent Jacobian sign away from the axis (s=0).
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        w = _w_total_from_wb_wp(wb, wp)
        jac = signgs * g.sqrtg
        jac = jac.at[0, :, :].set(0.0)
        neg = jnp.minimum(jac, 0.0)
        penalty = float(jacobian_penalty) * jnp.mean(neg * neg)
        return w + penalty

    def _w_terms(state: VMECState) -> Tuple[Any, Any, Any]:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        return wb, wp, _w_total_from_wb_wp(wb, wp)

    obj_and_grad = jax.value_and_grad(_objective)
    w_terms = _w_terms
    if jit_grad:
        obj_and_grad = jit(obj_and_grad)
        w_terms = jit(w_terms)

    # Start from a constraint-satisfying state.
    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=False,
        idx00=idx00,
    )

    wb0, wp0, w0 = w_terms(state)
    wb0 = float(np.asarray(wb0))
    wp0 = float(np.asarray(wp0))
    w0 = float(np.asarray(w0))
    wb_history = [wb0]
    wp_history = [wp0]
    grad_rms_history = []
    step_history = []

    obj0, grad0 = obj_and_grad(state)
    obj0 = float(np.asarray(obj0))
    w_history = [obj0]

    for it in range(max_iter):
        grad0m = _mask_grad_for_constraints(grad0, static, idx00=idx00)
        grad_raw = grad0m
        grad0m = _apply_preconditioner(
            grad0m,
            static,
            kind=preconditioner,
            exponent=precond_exponent,
            radial_alpha=precond_radial_alpha,
        )
        grad_rms = _grad_rms_state(grad0m)
        grad_rms_history.append(grad_rms)

        if verbose:
            print(f"[solve_fixed_boundary_gd] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < grad_tol:
            break

        step = float(step_size)
        accepted = False

        def _try_line_search(grad_step):
            step_local = float(step_size)
            for bt in range(max_backtracks + 1):
                if bt > 0:
                    step_local *= bt_factor
                trial = _update_state_gd(state, grad_step, step=step_local, scale_rz=scale_rz, scale_l=scale_l)
                trial = _enforce_fixed_boundary_and_axis(
                    trial,
                    static,
                    edge_Rcos=edge_Rcos,
                    edge_Rsin=edge_Rsin,
                    edge_Zcos=edge_Zcos,
                    edge_Zsin=edge_Zsin,
                    idx00=idx00,
                )
                obj_t = _objective(trial)
                obj_t = float(np.asarray(obj_t))
                if np.isfinite(obj_t) and obj_t < w_history[-1]:
                    return True, trial, obj_t, step_local
            return False, None, None, step_local

        accepted, trial, obj_t, step = _try_line_search(grad0m)
        if not accepted and preconditioner != "none":
            accepted, trial, obj_t, step = _try_line_search(grad_raw)
            if accepted and verbose:
                print("[solve_fixed_boundary_gd] fallback to unpreconditioned gradient")

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_fixed_boundary_gd] line search failed to improve objective; stopping")
            break

        state = trial
        obj0 = obj_t

        wb_t, wp_t, _w_t = w_terms(state)
        w_history.append(obj0)
        wb_history.append(float(np.asarray(wb_t)))
        wp_history.append(float(np.asarray(wp_t)))

        obj0, grad0 = obj_and_grad(state)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "gamma": gamma,
        "jacobian_penalty": float(jacobian_penalty),
        "scale_rz": float(scale_rz),
        "scale_l": float(scale_l),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
    }
    return SolveFixedBoundaryResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        wb_history=np.asarray(wb_history, dtype=float),
        wp_history=np.asarray(wp_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_lbfgs(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    pressure: Any | None = None,
    gamma: float = 0.0,
    history_size: int = 10,
    max_iter: int = 40,
    step_size: float = 1.0,
    grad_tol: float = 1e-10,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveFixedBoundaryResult:
    """Fixed-boundary solve using L-BFGS (no external deps).

    This solver minimizes::

        W = wb + wp/(gamma - 1)

    with:

    - fixed R/Z edge coefficients (prescribed boundary),
    - simple axis regularity,
    - lambda gauge (0,0)=0.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_lbfgs requires JAX (jax + jaxlib)")

    history_size = int(history_size)
    if history_size < 1:
        raise ValueError("history_size must be >= 1")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")

    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("gamma=1 makes wp/(gamma-1) singular")

    idx00 = _mode00_index(static.modes)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    if pressure is None:
        pressure = jnp.zeros_like(s)
    pressure = jnp.asarray(pressure)
    if pressure.shape != s.shape:
        raise ValueError(f"pressure must have shape {s.shape}, got {pressure.shape}")

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    def _wb_wp_from_geom(g) -> Tuple[Any, Any]:
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        return wb, wp

    def _w_total_from_wb_wp(wb, wp) -> Any:
        return wb + wp / (gamma - 1.0)

    def _w_only(state: VMECState) -> Any:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        return _w_total_from_wb_wp(wb, wp)

    def _w_terms_and_jacmin(state: VMECState) -> Tuple[Any, Any, Any, Any]:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        w = _w_total_from_wb_wp(wb, wp)
        jac = signgs * g.sqrtg
        if jac.shape[0] <= 1:
            jac_min = jnp.min(jac)
        else:
            jac_min = jnp.min(jac[1:, :, :])
        return wb, wp, w, jac_min

    w_and_grad = jax.value_and_grad(_w_only)
    w_terms = _w_terms_and_jacmin
    if jit_grad:
        w_and_grad = jit(w_and_grad)
        w_terms = jit(w_terms)

    def _lbfgs_direction(g_flat, s_hist, y_hist):
        if not s_hist:
            return -g_flat
        q = g_flat
        alpha = []
        rho = []
        for s_i, y_i in zip(reversed(s_hist), reversed(y_hist)):
            ys = jnp.dot(y_i, s_i)
            rho_i = jnp.where(ys != 0, 1.0 / ys, 0.0)
            a_i = rho_i * jnp.dot(s_i, q)
            q = q - a_i * y_i
            alpha.append(a_i)
            rho.append(rho_i)

        # Initial inverse-Hessian scaling (common L-BFGS choice)
        s0 = s_hist[-1]
        y0 = y_hist[-1]
        ys0 = jnp.dot(y0, s0)
        yy0 = jnp.dot(y0, y0)
        gamma0 = jnp.where(yy0 != 0, ys0 / yy0, 1.0)
        r = gamma0 * q

        for s_i, y_i, a_i, rho_i in zip(s_hist, y_hist, reversed(alpha), reversed(rho)):
            beta = rho_i * jnp.dot(y_i, r)
            r = r + s_i * (a_i - beta)

        return -r

    # Start from a constraint-satisfying state.
    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=False,
        idx00=idx00,
    )

    wb0, wp0, w0, jacmin0 = w_terms(state)
    w0 = float(np.asarray(w0))
    wb0 = float(np.asarray(wb0))
    wp0 = float(np.asarray(wp0))
    jacmin0 = float(np.asarray(jacmin0))
    if not np.isfinite(w0) or jacmin0 <= 0.0:
        raise ValueError("Initial state has invalid Jacobian sign or non-finite energy")

    w_history = [w0]
    wb_history = [wb0]
    wp_history = [wp0]
    grad_rms_history = []
    step_history = []

    w_val, grad = w_and_grad(state)
    grad = _mask_grad_for_constraints(grad, static, idx00=idx00, mask_lambda_axis=False)
    grad = _apply_preconditioner(
        grad,
        static,
        kind=preconditioner,
        exponent=precond_exponent,
        radial_alpha=precond_radial_alpha,
    )

    x = pack_state(state)
    g_flat = pack_state(grad)

    s_hist: list[Any] = []
    y_hist: list[Any] = []

    step0 = float(step_size)

    for it in range(max_iter):
        grad_rms = _grad_rms_state(grad)
        grad_rms_history.append(grad_rms)

        if verbose:
            print(f"[solve_fixed_boundary_lbfgs] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < grad_tol:
            break

        p_flat = _lbfgs_direction(g_flat, s_hist, y_hist)
        # Ensure descent direction; otherwise fall back to steepest descent.
        gtp = float(np.asarray(jnp.dot(g_flat, p_flat)))
        if not np.isfinite(gtp) or gtp >= 0.0:
            p_flat = -g_flat

        accepted = False
        step = step0

        x_old = x
        g_old = g_flat

        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            x_try = x_old + jnp.asarray(step, dtype=x_old.dtype) * p_flat
            st_try = unpack_state(x_try, state.layout)
            st_try = _enforce_fixed_boundary_and_axis(
                st_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                enforce_lambda_axis=False,
                idx00=idx00,
            )

            wb_t, wp_t, w_t, jacmin_t = w_terms(st_try)
            w_tf = float(np.asarray(w_t))
            jacmin_tf = float(np.asarray(jacmin_t))
            if np.isfinite(w_tf) and jacmin_tf > 0.0 and w_tf < w_history[-1]:
                state = st_try
                x = pack_state(state)
                accepted = True
                break

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_fixed_boundary_lbfgs] line search failed; stopping")
            break

        # New value/grad at accepted state.
        wb_t, wp_t, w_t, _jacmin_t = w_terms(state)
        w_history.append(float(np.asarray(w_t)))
        wb_history.append(float(np.asarray(wb_t)))
        wp_history.append(float(np.asarray(wp_t)))

        w_val, grad_new = w_and_grad(state)
        grad_new = _mask_grad_for_constraints(grad_new, static, idx00=idx00)
        grad_new = _apply_preconditioner(
            grad_new,
            static,
            kind=preconditioner,
            exponent=precond_exponent,
            radial_alpha=precond_radial_alpha,
        )
        g_flat_new = pack_state(grad_new)

        s_k = x - x_old
        y_k = g_flat_new - g_old
        ys = float(np.asarray(jnp.dot(y_k, s_k)))
        if np.isfinite(ys) and ys > 1e-14:
            s_hist.append(s_k)
            y_hist.append(y_k)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)

        grad = grad_new
        g_flat = g_flat_new
        step0 = float(step)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "gamma": gamma,
        "history_size": int(history_size),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
    }
    return SolveFixedBoundaryResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        wb_history=np.asarray(wb_history, dtype=float),
        wp_history=np.asarray(wp_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


@dataclass(frozen=True)
class _WoutLikeVmecForces:
    """Minimal `wout`-like container for VMEC force/residual kernels."""

    nfp: int
    mpol: int
    ntor: int
    lasym: bool
    signgs: int

    phipf: Any  # (ns,)
    phips: Any  # (ns,)
    chipf: Any  # (ns,)  (VMEC `wout` half-mesh averaged convention)
    pres: Any  # (ns,)  (half mesh, VMEC internal units mu0*Pa)


def solve_fixed_boundary_lbfgs_vmec_residual(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    w_rz: float = 1.0,
    w_l: float = 1.0,
    include_constraint_force: bool = True,
    objective_scale: float | None = None,
    apply_m1_constraints: bool = True,
    history_size: int = 10,
    max_iter: int = 40,
    step_size: float = 1.0,
    scale_rz: float = 1.0,
    scale_l: float = 1.0,
    grad_tol: float = 1e-10,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveVmecResidualResult:
    """Fixed-boundary solve by minimizing a VMEC-style force-residual objective.

    The objective follows the Step-10 parity pipeline
    ``bcovar -> forces -> tomnsps -> sum-of-squares of Fourier residual blocks``,
    using VMEC's ``getfsq`` conventions (post-``tomnsps`` ``scalxc`` scaling,
    optional converged-iteration m=1 constraints, and R/Z edge exclusion).

    For parity, build ``static`` with ``vmec_angle_grid(...)`` (see
    ``vmec_jax.vmec_tomnsp``). This solver does not include VMEC's
    iteration-dependent switching logic (e.g. ``lforbal`` triggering); it
    provides a differentiable objective suitable for regression and initial
    end-to-end parity.

    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_lbfgs_vmec_residual requires JAX (jax + jaxlib)")

    w_rz = float(w_rz)
    w_l = float(w_l)
    if w_rz < 0.0 or w_l < 0.0:
        raise ValueError("w_rz and w_l must be nonnegative")
    if objective_scale is not None and float(objective_scale) <= 0.0:
        raise ValueError("objective_scale must be positive when provided")
    scale_rz = float(scale_rz)
    scale_l = float(scale_l)
    if scale_rz <= 0.0 or scale_l <= 0.0:
        raise ValueError("scale_rz and scale_l must be positive")

    history_size = int(history_size)
    if history_size < 1:
        raise ValueError("history_size must be >= 1")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")

    idx00 = _mode00_index(static.modes)
    signgs = int(signgs)

    from .energy import flux_profiles_from_indata
    from .field import half_mesh_avg_from_full_mesh
    from .profiles import eval_profiles
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_force_norms_from_bcovar_dynamic,
        vmec_gcx2_from_tomnsps,
        vmec_zero_m1_zforce,
    )
    from .vmec_tomnsp import vmec_trig_tables

    s = jnp.asarray(static.s)

    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = half_mesh_avg_from_full_mesh(jnp.asarray(flux.chipf))

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    prof = eval_profiles(indata, s)
    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s)))

    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
    )

    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout_like.nfp),
        mmax=int(wout_like.mpol) - 1,
        nmax=int(wout_like.ntor),
        lasym=bool(wout_like.lasym),
        dtype=jnp.asarray(state0.Rcos).dtype,
    )

    objective_scale_f = float(objective_scale) if objective_scale is not None else None

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 0.0))

    def _fsq2_terms_and_jacmin(state: VMECState, zero_m1_zforce: Any):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        rzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=False,
        )
        rzl = vmec_zero_m1_zforce(frzl=rzl, enabled=zero_m1_zforce)
        gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
            frzl=rzl,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            apply_m1_constraints=bool(apply_m1_constraints),
            include_edge=False,
            apply_scalxc=True,
            s=s,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        fsqr2 = norms.r1 * norms.fnorm * gcr2
        fsqz2 = norms.r1 * norms.fnorm * gcz2
        fsql2 = norms.fnormL * gcl2

        w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
        if objective_scale_f is not None:
            w = jnp.asarray(objective_scale_f, dtype=jnp.asarray(w).dtype) * w

        jac = signgs * jnp.asarray(k.bc.jac.sqrtg)
        jac_min = jnp.min(jac) if jac.shape[0] <= 1 else jnp.min(jac[1:, :, :])
        return fsqr2, fsqz2, fsql2, w, jac_min

    def _w_only(state: VMECState, zero_m1_zforce: Any):
        return _fsq2_terms_and_jacmin(state, zero_m1_zforce)[3]

    w_and_grad = jax.value_and_grad(_w_only)
    w_terms = _fsq2_terms_and_jacmin
    if jit_grad:
        w_and_grad = jit(w_and_grad)
        w_terms = jit(w_terms)

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        idx00=idx00,
    )

    zero_m1 = jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
    fsqr2_0, fsqz2_0, fsql2_0, w0, jacmin0 = w_terms(state, zero_m1)
    w0 = float(np.asarray(w0))
    jacmin0 = float(np.asarray(jacmin0))
    if not np.isfinite(w0):
        raise ValueError("Initial state has non-finite residual objective")
    if jacmin0 <= 0.0 and verbose:
        print("[solve_fixed_boundary_lbfgs_vmec_residual] warning: initial Jacobian has non-positive entries")

    if objective_scale_f is None:
        # Auto-scale the objective to be O(1) on the initial iterate.
        objective_scale_f = 1.0 / max(abs(w0), 1.0)
        # Rebuild the objective closures with the now-fixed scale.
        def _fsq2_terms_and_jacmin(state: VMECState, zero_m1_zforce: Any):  # type: ignore[no-redef]
            k = vmec_forces_rz_from_wout(
                state=state,
                static=static,
                wout=wout_like,
                indata=None,
                constraint_tcon0=constraint_tcon0,
                use_vmec_synthesis=True,
                trig=trig,
            )
            rzl = vmec_residual_internal_from_kernels(
                k,
                cfg_ntheta=int(static.cfg.ntheta),
                cfg_nzeta=int(static.cfg.nzeta),
                wout=wout_like,
                trig=trig,
                apply_lforbal=False,
            )
            rzl = vmec_zero_m1_zforce(frzl=rzl, enabled=zero_m1_zforce)
            gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
                frzl=rzl,
                lconm1=bool(getattr(static.cfg, "lconm1", True)),
                apply_m1_constraints=bool(apply_m1_constraints),
                include_edge=False,
                apply_scalxc=True,
                s=s,
            )
            norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
            fsqr2 = norms.r1 * norms.fnorm * gcr2
            fsqz2 = norms.r1 * norms.fnorm * gcz2
            fsql2 = norms.fnormL * gcl2

            w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
            w = jnp.asarray(objective_scale_f, dtype=jnp.asarray(w).dtype) * w

            jac = signgs * jnp.asarray(k.bc.jac.sqrtg)
            jac_min = jnp.min(jac) if jac.shape[0] <= 1 else jnp.min(jac[1:, :, :])
            return fsqr2, fsqz2, fsql2, w, jac_min

        def _w_only(state: VMECState, zero_m1_zforce: Any):  # type: ignore[no-redef]
            return _fsq2_terms_and_jacmin(state, zero_m1_zforce)[3]

        w_and_grad = jax.value_and_grad(_w_only)
        w_terms = _fsq2_terms_and_jacmin
        if jit_grad:
            w_and_grad = jit(w_and_grad)
            w_terms = jit(w_terms)

        fsqr2_0, fsqz2_0, fsql2_0, w0, jacmin0 = w_terms(state, zero_m1)
        w0 = float(np.asarray(w0))

    w_history = [w0]
    fsqr2_history = [float(np.asarray(fsqr2_0))]
    fsqz2_history = [float(np.asarray(fsqz2_0))]
    fsql2_history = [float(np.asarray(fsql2_0))]
    grad_rms_history = []
    step_history = []

    w_val, grad = w_and_grad(state, zero_m1)
    grad = _mask_grad_for_constraints(grad, static, idx00=idx00, mask_lambda_axis=False)
    grad = _apply_preconditioner(
        grad,
        static,
        kind=preconditioner,
        exponent=precond_exponent,
        radial_alpha=precond_radial_alpha,
    )
    sr = jnp.asarray(scale_rz, dtype=jnp.asarray(grad.Rcos).dtype)
    sl = jnp.asarray(scale_l, dtype=jnp.asarray(grad.Lcos).dtype)
    grad = VMECState(
        layout=grad.layout,
        Rcos=jnp.asarray(grad.Rcos) * sr,
        Rsin=jnp.asarray(grad.Rsin) * sr,
        Zcos=jnp.asarray(grad.Zcos) * sr,
        Zsin=jnp.asarray(grad.Zsin) * sr,
        Lcos=jnp.asarray(grad.Lcos) * sl,
        Lsin=jnp.asarray(grad.Lsin) * sl,
    )

    x = pack_state(state)
    g_flat = pack_state(grad)

    s_hist: list[Any] = []
    y_hist: list[Any] = []

    step0 = float(step_size)

    def _lbfgs_direction(g_flat, s_hist, y_hist):
        if not s_hist:
            return -g_flat
        q = g_flat
        alpha = []
        rho = []
        for s_i, y_i in zip(reversed(s_hist), reversed(y_hist)):
            ys = jnp.dot(y_i, s_i)
            rho_i = jnp.where(ys != 0, 1.0 / ys, 0.0)
            a_i = rho_i * jnp.dot(s_i, q)
            q = q - a_i * y_i
            alpha.append(a_i)
            rho.append(rho_i)

        s0 = s_hist[-1]
        y0 = y_hist[-1]
        ys0 = jnp.dot(y0, s0)
        yy0 = jnp.dot(y0, y0)
        gamma0 = jnp.where(yy0 != 0, ys0 / yy0, 1.0)
        r = gamma0 * q

        for s_i, y_i, a_i, rho_i in zip(s_hist, y_hist, reversed(alpha), reversed(rho)):
            beta = rho_i * jnp.dot(y_i, r)
            r = r + s_i * (a_i - beta)

        return -r

    for it in range(max_iter):
        grad_rms = _grad_rms_state(grad)
        grad_rms_history.append(grad_rms)

        if verbose:
            print(f"[solve_fixed_boundary_lbfgs_vmec_residual] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < grad_tol:
            break

        p_flat = _lbfgs_direction(g_flat, s_hist, y_hist)
        gtp = float(np.asarray(jnp.dot(g_flat, p_flat)))
        if not np.isfinite(gtp) or gtp >= 0.0:
            p_flat = -g_flat

        accepted = False
        step = step0
        best_w = np.inf
        best_state = None
        best_step = None
        best_fsqr2 = None
        best_fsqz2 = None
        best_fsql2 = None

        x_old = x
        g_old = g_flat

        zero_m1 = jnp.asarray(1.0 if (it < 2) or (fsqz2_history[-1] < 1e-6) else 0.0, dtype=jnp.asarray(state.Rcos).dtype)
        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            x_try = x_old + jnp.asarray(step, dtype=x_old.dtype) * p_flat
            st_try = unpack_state(x_try, state.layout)
            st_try = _enforce_fixed_boundary_and_axis(
                st_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                idx00=idx00,
            )

            fsqr2_t, fsqz2_t, fsql2_t, w_t, jacmin_t = w_terms(st_try, zero_m1)
            w_tf = float(np.asarray(w_t))
            jacmin_tf = float(np.asarray(jacmin_t))
            if np.isfinite(w_tf) and w_tf < best_w:
                best_w = w_tf
                best_state = st_try
                best_step = step
                best_fsqr2 = float(np.asarray(fsqr2_t))
                best_fsqz2 = float(np.asarray(fsqz2_t))
                best_fsql2 = float(np.asarray(fsql2_t))
            if np.isfinite(w_tf) and jacmin_tf > 0.0 and w_tf < w_history[-1]:
                state = st_try
                x = pack_state(state)
                accepted = True
                fsqr2_accept = float(np.asarray(fsqr2_t))
                fsqz2_accept = float(np.asarray(fsqz2_t))
                fsql2_accept = float(np.asarray(fsql2_t))
                break

        step_history.append(step)

        if not accepted:
            if best_state is not None and np.isfinite(best_w):
                if verbose:
                    print(
                        "[solve_fixed_boundary_lbfgs_vmec_residual] line search failed; "
                        "accepting best finite step"
                    )
                state = best_state
                x = pack_state(state)
                w_t = best_w
                fsqr2_accept = best_fsqr2 if best_fsqr2 is not None else float(np.asarray(fsqr2_t))
                fsqz2_accept = best_fsqz2 if best_fsqz2 is not None else float(np.asarray(fsqz2_t))
                fsql2_accept = best_fsql2 if best_fsql2 is not None else float(np.asarray(fsql2_t))
                step_history[-1] = best_step
            else:
                if verbose:
                    print("[solve_fixed_boundary_lbfgs_vmec_residual] line search failed; stopping")
                break

        w_history.append(float(np.asarray(w_t)))
        fsqr2_history.append(fsqr2_accept)
        fsqz2_history.append(fsqz2_accept)
        fsql2_history.append(fsql2_accept)

        w_val, grad_new = w_and_grad(state, zero_m1)
        grad_new = _mask_grad_for_constraints(grad_new, static, idx00=idx00, mask_lambda_axis=False)
        grad_new = _apply_preconditioner(
            grad_new,
            static,
            kind=preconditioner,
            exponent=precond_exponent,
            radial_alpha=precond_radial_alpha,
        )
        g_flat_new = pack_state(grad_new)

        s_k = x - x_old
        y_k = g_flat_new - g_old
        ys = float(np.asarray(jnp.dot(y_k, s_k)))
        if np.isfinite(ys) and ys > 1e-14:
            s_hist.append(s_k)
            y_hist.append(y_k)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)

        grad = grad_new
        g_flat = g_flat_new
        step0 = float(step)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "w_rz": float(w_rz),
        "w_l": float(w_l),
        "objective_scale": float(objective_scale_f),
        "include_constraint_force": bool(include_constraint_force),
        "scale_rz": float(scale_rz),
        "scale_l": float(scale_l),
        "apply_m1_constraints": bool(apply_m1_constraints),
        "history_size": int(history_size),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
    }
    return SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_gn_vmec_residual(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    w_rz: float = 1.0,
    w_l: float = 1.0,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    objective_scale: float | None = None,
    damping: float = 1e-3,
    damping_increase: float = 10.0,
    damping_decrease: float = 0.5,
    max_damping: float = 1e6,
    max_retries: int = 6,
    zero_m1_iters: int = 50,
    zero_m1_fsqz_thresh: float = 1e-6,
    max_iter: int = 20,
    cg_tol: float = 1e-6,
    cg_maxiter: int = 80,
    step_size: float = 1.0,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_kernels: bool = True,
    verbose: bool = True,
) -> SolveVmecResidualResult:
    """Fixed-boundary solve using a Gauss-Newton (normal-equations) step on VMEC residuals.

    This treats the VMEC residual blocks returned by `tomnsps` as a least-squares
    problem and solves (approximately) for a step `dx` using conjugate gradients:

        (Jᵀ J + damping * I) dx = -Jᵀ r

    where `r(state)` is the stacked residual vector and `J` is its Jacobian.

    The residual vector uses the same conventions as `vmec_jax.vmec_residue`
    (post-`tomnsps` `scalxc` scaling, optional m=1 constraints, and R/Z edge
    exclusion) so the objective is consistent with Step-10 scalar definitions.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_gn_vmec_residual requires JAX (jax + jaxlib)")
    if damping < 0.0:
        raise ValueError("damping must be nonnegative")
    damping_increase = float(damping_increase)
    damping_decrease = float(damping_decrease)
    max_damping = float(max_damping)
    max_retries = int(max_retries)
    if damping_increase <= 1.0:
        raise ValueError("damping_increase must be > 1")
    if not (0.0 < damping_decrease <= 1.0):
        raise ValueError("damping_decrease must be in (0, 1]")
    if max_damping <= 0.0:
        raise ValueError("max_damping must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    zero_m1_iters = int(zero_m1_iters)
    zero_m1_fsqz_thresh = float(zero_m1_fsqz_thresh)
    if zero_m1_iters < 0:
        raise ValueError("zero_m1_iters must be >= 0")
    if zero_m1_fsqz_thresh < 0.0:
        raise ValueError("zero_m1_fsqz_thresh must be >= 0")
    w_rz = float(w_rz)
    w_l = float(w_l)
    if w_rz < 0.0 or w_l < 0.0:
        raise ValueError("w_rz and w_l must be nonnegative")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if cg_maxiter < 1:
        raise ValueError("cg_maxiter must be >= 1")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")
    if objective_scale is not None and float(objective_scale) <= 0.0:
        raise ValueError("objective_scale must be positive when provided")

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 0.0))

    signgs = int(signgs)
    idx00 = _mode00_index(static.modes)

    from .energy import flux_profiles_from_indata
    from .field import half_mesh_avg_from_full_mesh
    from .profiles import eval_profiles
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_zero_m1_zforce,
    )
    from .vmec_tomnsp import TomnspsRZL, vmec_trig_tables

    try:
        from jax.scipy.sparse.linalg import cg  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("solve_fixed_boundary_gn_vmec_residual requires jax.scipy.sparse.linalg.cg") from e

    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = half_mesh_avg_from_full_mesh(jnp.asarray(flux.chipf))

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    prof = eval_profiles(indata, s)
    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s)))

    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
    )

    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout_like.nfp),
        mmax=int(wout_like.mpol) - 1,
        nmax=int(wout_like.ntor),
        lasym=bool(wout_like.lasym),
        dtype=jnp.asarray(state0.Rcos).dtype,
    )

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    def _project_step(d: VMECState) -> VMECState:
        return _mask_grad_for_constraints(d, static, idx00=idx00, mask_lambda_axis=False)

    def _enforce_state(st: VMECState) -> VMECState:
        return _enforce_fixed_boundary_and_axis(
            st,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_lambda_axis=False,
            idx00=idx00,
        )

    def _zero_edge_rz(a):
        a = None if a is None else jnp.asarray(a)
        if a is None:
            return None
        if a.shape[0] < 2:
            return a
        return a.at[-1].set(jnp.zeros_like(a[-1]))

    def _residual_blocks(state: VMECState, zero_m1_zforce: Any):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        rzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=False,
        )
        frzl = rzl
        if bool(apply_m1_constraints):
            frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1_zforce)

        # VMEC convention: after tomnsps, scale Fourier-space forces by `scalxc`
        # before forming sums-of-squares/scalars (funct3d.f).
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)

        # VMEC convention: R/Z sums exclude the edge surface; enforce that by
        # zeroing R/Z blocks at js=ns (lambda blocks are left untouched).
        frzl = TomnspsRZL(
            frcc=_zero_edge_rz(frzl.frcc),
            frss=_zero_edge_rz(frzl.frss),
            fzsc=_zero_edge_rz(frzl.fzsc),
            fzcs=_zero_edge_rz(frzl.fzcs),
            flsc=frzl.flsc,
            flcs=frzl.flcs,
            frsc=_zero_edge_rz(getattr(frzl, "frsc", None)),
            frcs=_zero_edge_rz(getattr(frzl, "frcs", None)),
            fzcc=_zero_edge_rz(getattr(frzl, "fzcc", None)),
            fzss=_zero_edge_rz(getattr(frzl, "fzss", None)),
            flcc=getattr(frzl, "flcc", None),
            flss=getattr(frzl, "flss", None),
        )

        gcr2 = jnp.sum(jnp.asarray(frzl.frcc) ** 2)
        gcz2 = jnp.sum(jnp.asarray(frzl.fzsc) ** 2)
        gcl2 = jnp.sum(jnp.asarray(frzl.flsc) ** 2)
        if frzl.frss is not None:
            gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frss) ** 2)
        if frzl.fzcs is not None:
            gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcs) ** 2)
        if frzl.flcs is not None:
            gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcs) ** 2)

        if getattr(frzl, "frsc", None) is not None:
            gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frsc) ** 2)
        if getattr(frzl, "fzcc", None) is not None:
            gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcc) ** 2)
        if getattr(frzl, "flcc", None) is not None:
            gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcc) ** 2)

        if getattr(frzl, "frcs", None) is not None:
            gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frcs) ** 2)
        if getattr(frzl, "fzss", None) is not None:
            gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzss) ** 2)
        if getattr(frzl, "flss", None) is not None:
            gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flss) ** 2)

        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        fsqr2 = norms.r1 * norms.fnorm * gcr2
        fsqz2 = norms.r1 * norms.fnorm * gcz2
        fsql2 = norms.fnormL * gcl2
        return frzl, fsqr2, fsqz2, fsql2, norms

    def _residual_vec(state: VMECState, zero_m1_zforce: Any) -> Any:
        frzl, *_vals = _residual_blocks(state, zero_m1_zforce)
        norms = _vals[-1]
        scale_rz = jnp.sqrt(jnp.asarray(w_rz)) * jnp.sqrt(norms.r1 * norms.fnorm)
        scale_l = jnp.sqrt(jnp.asarray(w_l)) * jnp.sqrt(norms.fnormL)
        scale_rz = jnp.asarray(scale_rz, dtype=jnp.asarray(frzl.frcc).dtype)
        scale_l = jnp.asarray(scale_l, dtype=jnp.asarray(frzl.frcc).dtype)

        parts = [scale_rz * frzl.frcc, scale_rz * frzl.fzsc, scale_l * frzl.flsc]
        if frzl.frss is not None:
            parts.append(scale_rz * frzl.frss)
        if frzl.fzcs is not None:
            parts.append(scale_rz * frzl.fzcs)
        if frzl.flcs is not None:
            parts.append(scale_l * frzl.flcs)
        for name in ["frsc", "fzcc", "flcc", "frcs", "fzss", "flss"]:
            a = getattr(frzl, name, None)
            if a is not None:
                if name.startswith("fl"):
                    parts.append(scale_l * a)
                else:
                    parts.append(scale_rz * a)
        return jnp.concatenate([jnp.ravel(jnp.asarray(p)) for p in parts], axis=0)

    def _obj_terms(state: VMECState, zero_m1_zforce: Any):
        _frzl, fsqr2, fsqz2, fsql2, _norms = _residual_blocks(state, zero_m1_zforce)
        w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
        return fsqr2, fsqz2, fsql2, w

    if bool(jit_kernels):
        _residual_vec_jit = jit(_residual_vec)
        _obj_terms_jit = jit(_obj_terms)
    else:
        _residual_vec_jit = _residual_vec
        _obj_terms_jit = _obj_terms

    state = _enforce_state(state0)
    zero_m1 = jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
    fsqr2_0, fsqz2_0, fsql2_0, w0 = _obj_terms_jit(state, zero_m1)
    w0_f = float(np.asarray(w0))
    if not np.isfinite(w0_f):
        raise ValueError("Initial state has non-finite residual objective")

    scale_f = float(objective_scale) if objective_scale is not None else (1.0 / max(abs(w0_f), 1.0))

    w_history = [float(scale_f * w0_f)]
    fsqr2_history = [float(np.asarray(fsqr2_0))]
    fsqz2_history = [float(np.asarray(fsqz2_0))]
    fsql2_history = [float(np.asarray(fsql2_0))]
    grad_rms_history = []
    step_history = []

    damping_it = float(damping)
    for it in range(int(max_iter)):
        zero_m1 = jnp.asarray(
            1.0 if (it < zero_m1_iters) or (fsqz2_history[-1] < zero_m1_fsqz_thresh) else 0.0,
            dtype=jnp.asarray(state.Rcos).dtype,
        )
        r, pullback = jax.vjp(_residual_vec_jit, state, zero_m1)
        # Gradient of 0.5*||r||^2 is J^T r.
        g_state = pullback(r)[0]
        g_state = _project_step(g_state)
        grad_rms_history.append(_grad_rms_state(g_state))

        b_flat = -pack_state(g_state)

        accepted = False
        step = float(step_size)
        w_curr = w_history[-1]
        retry = 0
        while True:
            dmp = float(damping_it)

            def _matvec(v_flat):
                v_state = unpack_state(v_flat, state.layout)
                v_state = _project_step(v_state)
                zero_tangent = jnp.zeros_like(zero_m1)
                jv = jax.jvp(_residual_vec_jit, (state, zero_m1), (v_state, zero_tangent))[1]
                jt_jv = pullback(jv)[0]
                jt_jv = _project_step(jt_jv)
                if dmp != 0.0:
                    jt_jv = VMECState(
                        layout=jt_jv.layout,
                        Rcos=jt_jv.Rcos + dmp * v_state.Rcos,
                        Rsin=jt_jv.Rsin + dmp * v_state.Rsin,
                        Zcos=jt_jv.Zcos + dmp * v_state.Zcos,
                        Zsin=jt_jv.Zsin + dmp * v_state.Zsin,
                        Lcos=jt_jv.Lcos + dmp * v_state.Lcos,
                        Lsin=jt_jv.Lsin + dmp * v_state.Lsin,
                    )
                return pack_state(jt_jv)

            dx_flat, _info = cg(_matvec, b_flat, tol=float(cg_tol), maxiter=int(cg_maxiter))
            dx_state = unpack_state(dx_flat, state.layout)
            dx_state = _project_step(dx_state)

            step = float(step_size)
            for bt in range(int(max_backtracks) + 1):
                if bt > 0:
                    step *= float(bt_factor)
                st_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype) * jnp.asarray(dx_state.Rcos),
                    Rsin=jnp.asarray(state.Rsin) + jnp.asarray(step, dtype=jnp.asarray(state.Rsin).dtype) * jnp.asarray(dx_state.Rsin),
                    Zcos=jnp.asarray(state.Zcos) + jnp.asarray(step, dtype=jnp.asarray(state.Zcos).dtype) * jnp.asarray(dx_state.Zcos),
                    Zsin=jnp.asarray(state.Zsin) + jnp.asarray(step, dtype=jnp.asarray(state.Zsin).dtype) * jnp.asarray(dx_state.Zsin),
                    Lcos=jnp.asarray(state.Lcos) + jnp.asarray(step, dtype=jnp.asarray(state.Lcos).dtype) * jnp.asarray(dx_state.Lcos),
                    Lsin=jnp.asarray(state.Lsin) + jnp.asarray(step, dtype=jnp.asarray(state.Lsin).dtype) * jnp.asarray(dx_state.Lsin),
                )
                st_try = _enforce_state(st_try)
                fsqr2_t, fsqz2_t, fsql2_t, w_t = _obj_terms_jit(st_try, zero_m1)
                w_tf = float(np.asarray(w_t))
                w_scaled = float(scale_f * w_tf)
                if np.isfinite(w_scaled) and w_scaled < w_curr:
                    state = st_try
                    accepted = True
                    w_history.append(w_scaled)
                    fsqr2_history.append(float(np.asarray(fsqr2_t)))
                    fsqz2_history.append(float(np.asarray(fsqz2_t)))
                    fsql2_history.append(float(np.asarray(fsql2_t)))
                    break

            if accepted:
                # Levenberg-Marquardt style: relax damping after success.
                damping_it = max(damping_it * damping_decrease, 0.0)
                break

            if retry >= max_retries or damping_it >= max_damping:
                break
            # Increase damping and try again from the same state.
            damping_it = min(max_damping, damping_it * damping_increase)
            retry += 1

        if not accepted:
            # Robust fallback: take a small steepest-descent step on 0.5*||r||^2
            # using the already-computed gradient g_state = J^T r.
            dx_state = unpack_state(b_flat, state.layout)  # b_flat = -grad_flat
            dx_state = _project_step(dx_state)
            step = float(step_size)
            for bt in range(int(max_backtracks) + 1):
                if bt > 0:
                    step *= float(bt_factor)
                st_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype) * jnp.asarray(dx_state.Rcos),
                    Rsin=jnp.asarray(state.Rsin) + jnp.asarray(step, dtype=jnp.asarray(state.Rsin).dtype) * jnp.asarray(dx_state.Rsin),
                    Zcos=jnp.asarray(state.Zcos) + jnp.asarray(step, dtype=jnp.asarray(state.Zcos).dtype) * jnp.asarray(dx_state.Zcos),
                    Zsin=jnp.asarray(state.Zsin) + jnp.asarray(step, dtype=jnp.asarray(state.Zsin).dtype) * jnp.asarray(dx_state.Zsin),
                    Lcos=jnp.asarray(state.Lcos) + jnp.asarray(step, dtype=jnp.asarray(state.Lcos).dtype) * jnp.asarray(dx_state.Lcos),
                    Lsin=jnp.asarray(state.Lsin) + jnp.asarray(step, dtype=jnp.asarray(state.Lsin).dtype) * jnp.asarray(dx_state.Lsin),
                )
                st_try = _enforce_state(st_try)
                fsqr2_t, fsqz2_t, fsql2_t, w_t = _obj_terms_jit(st_try, zero_m1)
                w_tf = float(np.asarray(w_t))
                w_scaled = float(scale_f * w_tf)
                if np.isfinite(w_scaled) and w_scaled < w_curr:
                    state = st_try
                    accepted = True
                    w_history.append(w_scaled)
                    fsqr2_history.append(float(np.asarray(fsqr2_t)))
                    fsqz2_history.append(float(np.asarray(fsqz2_t)))
                    fsql2_history.append(float(np.asarray(fsql2_t)))
                    break

        step_history.append(step)
        if verbose:
            print(
                f"[solve_fixed_boundary_gn_vmec_residual] iter={it:03d} w={w_history[-1]:.8e} "
                f"step={step:.3e} accepted={accepted} damping={damping_it:.3e} retries={retry}"
            )

        if not accepted:
            break

    diag = {
        "idx00": idx00,
        "signgs": signgs,
        "w_rz": float(w_rz),
        "w_l": float(w_l),
        "objective_scale": float(scale_f),
        "apply_m1_constraints": bool(apply_m1_constraints),
        "damping": float(damping),
        "cg_tol": float(cg_tol),
        "cg_maxiter": int(cg_maxiter),
    }
    return SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_vmecpp_iter(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    max_iter: int = 50,
    step_size: float = 1.0,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    precond_radial_alpha: float = 0.5,
    precond_lambda_alpha: float = 0.5,
    mode_diag_exponent: float = 1.0,
    auto_flip_force: bool = True,
    vmecpp_strict_update: bool = True,
    vmecpp_reference_mode: bool = False,
    use_vmecpp_restart_triggers: bool | None = None,
    use_direct_fallback: bool | None = None,
    verbose: bool = True,
) -> SolveVmecResidualResult:
    """VMEC++-style fixed-point update loop using preconditioned force residuals."""
    if not has_jax():
        raise ImportError("solve_fixed_boundary_vmecpp_iter requires JAX (jax + jaxlib)")

    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    step_size = float(step_size)
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    signgs = int(signgs)
    idx00 = _mode00_index(static.modes)
    vmecpp_reference_mode = bool(vmecpp_reference_mode)
    if use_vmecpp_restart_triggers is None:
        use_vmecpp_restart_triggers = vmecpp_reference_mode
    if use_direct_fallback is None:
        use_direct_fallback = False
    use_vmecpp_restart_triggers = bool(use_vmecpp_restart_triggers)
    use_direct_fallback = bool(use_direct_fallback)

    from .energy import flux_profiles_from_indata
    from .field import half_mesh_avg_from_full_mesh
    from .energy import magnetic_wb_from_state
    from .profiles import eval_profiles
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_gcx2_from_tomnsps,
        vmec_rz_norm_from_state,
        vmec_wint_from_trig,
        vmec_zero_m1_zforce,
    )
    from .vmec_jacobian import vmec_half_mesh_jacobian_from_state
    from .vmec_tomnsp import TomnspsRZL, vmec_trig_tables

    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = half_mesh_avg_from_full_mesh(jnp.asarray(flux.chipf))

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    prof = eval_profiles(indata, s)
    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s)))

    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
    )

    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout_like.nfp),
        mmax=int(wout_like.mpol) - 1,
        nmax=int(wout_like.ntor),
        lasym=bool(wout_like.lasym),
        dtype=jnp.asarray(state0.Rcos).dtype,
    )

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 0.0))

    def _zero_edge_rz(a):
        a = None if a is None else jnp.asarray(a)
        if a is None:
            return None
        if a.shape[0] < 2:
            return a
        return a.at[-1].set(jnp.zeros_like(a[-1]))

    def _apply_radial_tridi(a, alpha: float):
        if alpha <= 0.0:
            return a
        return _tridi_smooth_dirichlet_vmecpp(jnp.asarray(a), alpha=alpha)

    def _tridi_smooth_dirichlet_vmecpp(rhs, *, alpha: float):
        """Dirichlet tridiagonal smoother along s for VMEC++ fixed-point updates."""
        rhs = jnp.asarray(rhs)
        if rhs.ndim == 2:
            rhs2 = rhs
            orig_shape = None
        elif rhs.ndim == 3:
            ns = int(rhs.shape[0])
            rhs2 = rhs.reshape(ns, -1)
            orig_shape = rhs.shape
        else:
            raise ValueError(f"expected (ns,K) or (ns,M,N), got {rhs.shape}")
        ns = int(rhs2.shape[0])
        if ns < 3:
            return rhs
        alpha = jnp.asarray(alpha, dtype=rhs2.dtype)
        a = -alpha
        b = 1.0 + 2.0 * alpha
        c = -alpha

        x0 = rhs2[0]
        xN = rhs2[-1]
        d = rhs2[1:-1]
        d = d.at[0].add(alpha * x0)
        d = d.at[-1].add(alpha * xN)

        n = int(d.shape[0])
        if n == 1:
            x_int = d / b
        else:
            cp0 = c / b
            dp0 = d[0] / b

            def fwd(carry, di):
                cp_prev, dp_prev = carry
                denom = b - a * cp_prev
                cp = c / denom
                dp = (di - a * dp_prev) / denom
                return (cp, dp), (cp, dp)

            (cp_last, dp_last), (cp, dp) = jax.lax.scan(fwd, (cp0, dp0), d[1:])

            def bwd(carry, cp_dp):
                x_next = carry
                cp_i, dp_i = cp_dp
                x_i = dp_i - cp_i * x_next
                return x_i, x_i

            _, x_rev = jax.lax.scan(bwd, dp_last, (cp, dp), reverse=True)
            x_int = jnp.concatenate([x_rev, dp_last[None, :]], axis=0)

        out = jnp.concatenate([x0[None, :], x_int, xN[None, :]], axis=0)
        if orig_shape is not None:
            out = out.reshape(orig_shape)
        return out

    def _metric_surface_precond_from_bcovar(bc):
        """Approximate VMEC++ radial preconditioner scaling from bcovar metrics."""
        guu = jnp.asarray(bc.guu)
        r12 = jnp.asarray(bc.jac.r12)
        bsubu = jnp.asarray(bc.bsubu)
        bsubv = jnp.asarray(bc.bsubv)
        nzeta = int(guu.shape[2])
        w_ang = vmec_wint_from_trig(trig, nzeta=nzeta).astype(guu.dtype)
        w3 = w_ang[None, :, :]

        # R/Z preconditioner proxy: VMEC force-norm denominator integrand.
        rz_denom = jnp.sum((guu * (r12 * r12)) * w3, axis=(1, 2))
        rz_scale = jnp.where(rz_denom > 0.0, 1.0 / jnp.sqrt(rz_denom), 1.0)

        # Lambda preconditioner proxy: VMEC lambda norm denominator integrand.
        l_denom = jnp.sum(((bsubu * bsubu) + (bsubv * bsubv)) * w3, axis=(1, 2))
        l_scale = jnp.where(l_denom > 0.0, 1.0 / jnp.sqrt(l_denom), 1.0)

        # Keep updates bounded and avoid axis/boundary blowups.
        rz_scale = jnp.clip(rz_scale, 1e-4, 1e2)
        l_scale = jnp.clip(l_scale, 1e-4, 1e2)
        return rz_scale, l_scale

    def _compute_forces(state: VMECState, *, include_edge: bool, zero_m1: Any):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        frzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=False,
        )
        if bool(apply_m1_constraints):
            frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1)
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)

        frzl = TomnspsRZL(
            frcc=_zero_edge_rz(frzl.frcc),
            frss=_zero_edge_rz(frzl.frss),
            fzsc=_zero_edge_rz(frzl.fzsc),
            fzcs=_zero_edge_rz(frzl.fzcs),
            flsc=frzl.flsc,
            flcs=frzl.flcs,
            frsc=_zero_edge_rz(getattr(frzl, "frsc", None)),
            frcs=_zero_edge_rz(getattr(frzl, "frcs", None)),
            fzcc=_zero_edge_rz(getattr(frzl, "fzcc", None)),
            fzss=_zero_edge_rz(getattr(frzl, "fzss", None)),
            flcc=getattr(frzl, "flcc", None),
            flss=getattr(frzl, "flss", None),
        )

        gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
            frzl=frzl,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=bool(include_edge),
            apply_scalxc=False,
            s=s,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        fsqr = norms.r1 * norms.fnorm * gcr2
        fsqz = norms.r1 * norms.fnorm * gcz2
        fsql = norms.fnormL * gcl2
        rz_scale, l_scale = _metric_surface_precond_from_bcovar(k.bc)
        return frzl, fsqr, fsqz, fsql, rz_scale, l_scale

    mpol = int(static.cfg.mpol)
    ntor = int(static.cfg.ntor)
    nrange = ntor + 1
    nfp = float(static.cfg.nfp)
    ncoeff = int(jnp.asarray(state0.Rcos).shape[1])

    idx_pos = -np.ones((mpol, nrange), dtype=np.int32)
    idx_neg = -np.ones((mpol, nrange), dtype=np.int32)
    for k, (m_k, n_k) in enumerate(zip(np.asarray(static.modes.m), np.asarray(static.modes.n))):
        m_i = int(m_k)
        n_i = int(n_k)
        if n_i >= 0:
            idx_pos[m_i, n_i] = int(k)
        else:
            idx_neg[m_i, -n_i] = int(k)

    m_idx_list = []
    n_idx_list = []
    kp_idx_list = []
    kn_idx_list = []
    for m_i in range(mpol):
        for n_i in range(nrange):
            kp = int(idx_pos[m_i, n_i])
            if kp < 0:
                continue
            m_idx_list.append(m_i)
            n_idx_list.append(n_i)
            kp_idx_list.append(kp)
            kn_idx_list.append(int(idx_neg[m_i, n_i]))

    m_idx = jnp.asarray(np.asarray(m_idx_list, dtype=np.int32))
    n_idx = jnp.asarray(np.asarray(n_idx_list, dtype=np.int32))
    kp_idx = jnp.asarray(np.asarray(kp_idx_list, dtype=np.int32))
    kn_idx_np = np.asarray(kn_idx_list, dtype=np.int32)
    kn_idx = jnp.asarray(kn_idx_np)
    has_kn_np = kn_idx_np >= 0
    has_kn = jnp.asarray(has_kn_np)
    has_kn_any = bool(np.any(has_kn_np))

    def _mn_cos_to_signed(cc, ss):
        cc = jnp.asarray(cc)
        ss = jnp.asarray(ss) if ss is not None else jnp.zeros_like(cc)
        if cc.ndim != 3:
            raise ValueError(f"expected (ns, mpol, nrange), got {cc.shape}")
        cc_mn = cc[:, m_idx, n_idx]
        ss_mn = ss[:, m_idx, n_idx]
        is_axis_m = (m_idx == 0)[None, :]
        is_n0 = (n_idx == 0)[None, :]
        pos = jnp.where(is_axis_m | is_n0, cc_mn, 0.5 * (cc_mn + ss_mn))
        out = jnp.zeros((cc.shape[0], ncoeff), dtype=cc.dtype)
        out = out.at[:, kp_idx].set(pos)
        if has_kn_any:
            neg = 0.5 * (cc_mn + (-ss_mn))
            out = out.at[:, kn_idx[has_kn]].set(neg[:, has_kn])
        return out

    def _mn_sin_to_signed(sc, cs):
        sc = jnp.asarray(sc)
        cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
        if sc.ndim != 3:
            raise ValueError(f"expected (ns, mpol, nrange), got {sc.shape}")
        sc_mn = sc[:, m_idx, n_idx]
        cs_mn = cs[:, m_idx, n_idx]
        is_n0 = (n_idx == 0)[None, :]
        pos = jnp.where(is_n0, sc_mn, 0.5 * (sc_mn - cs_mn))
        out = jnp.zeros((sc.shape[0], ncoeff), dtype=sc.dtype)
        out = out.at[:, kp_idx].set(pos)
        if has_kn_any:
            neg = 0.5 * (sc_mn + cs_mn)
            out = out.at[:, kn_idx[has_kn]].set(neg[:, has_kn])
        return out

    def _mode_diag_weights_mn(dtype):
        m = jnp.arange(mpol, dtype=jnp.float64)
        n = jnp.arange(nrange, dtype=jnp.float64) * nfp
        k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
        w = (1.0 + k2) ** (-float(mode_diag_exponent))
        return w.astype(dtype)

    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=False,
        idx00=idx00,
    )

    ftol = float(indata.get_float("FTOL", 1e-10))

    w_history = []
    fsqr2_history = []
    fsqz2_history = []
    fsql2_history = []
    fsqr1_history = []
    fsqz1_history = []
    fsql1_history = []
    fsq1_history = []
    step_status_history: list[str] = []
    restart_reason_history: list[str] = []
    pre_restart_reason_history: list[str] = []
    time_step_history: list[float] = []
    res0_history: list[float] = []
    fsq_prev_history: list[float] = []
    bad_growth_streak_history: list[int] = []
    iter1_history: list[int] = []
    include_edge_history: list[int] = []
    zero_m1_history: list[int] = []
    dt_eff_history: list[float] = []
    update_rms_history: list[float] = []
    w_curr_history: list[float] = []
    w_try_history: list[float] = []
    w_try_ratio_history: list[float] = []
    restart_path_history: list[str] = []
    min_tau_history: list[float] = []
    max_tau_history: list[float] = []
    bad_jacobian_history: list[int] = []
    grad_rms_history = []
    step_history = []

    # VMEC++-style time-stepping (conjugate-gradient-like) state.
    time_step = float(step_size)
    k_ndamp = 10
    inv_tau = [0.15 / time_step] * k_ndamp
    fsq_prev = 1.0
    vRcc = jnp.zeros((int(state.Rcos.shape[0]), mpol, nrange), dtype=jnp.asarray(state.Rcos).dtype)
    vRss = jnp.zeros_like(vRcc)
    vZsc = jnp.zeros_like(vRcc)
    vZcs = jnp.zeros_like(vRcc)
    vLsc = jnp.zeros_like(vRcc)
    vLcs = jnp.zeros_like(vRcc)
    flip_sign = 1.0
    max_coeff_delta_rms = 1e-5
    max_update_rms = 5e-3
    if bool(vmecpp_reference_mode):
        max_coeff_delta_rms = 5e-6
        max_update_rms = 1e-3
    ijacob = 0
    bad_resets = 0
    iter1 = 1
    res0 = -1.0
    k_preconditioner_update_interval = 25
    state_checkpoint = state
    bad_growth_streak = 0
    # VMEC++ RestartIteration factors (vmec.cc):
    # - BAD_JACOBIAN: delt0r *= 0.9
    # - BAD_PROGRESS: delt0r /= 1.03
    restart_badjac_factor = 0.9
    restart_badprog_factor = 1.03
    huge_force_restart_count = 0
    huge_force_restart_budget = 2

    def _edge_force_trigger(it: int, w_hist: list[float], fsqr_hist: list[float], fsqz_hist: list[float]) -> bool:
        """Heuristic for VMEC++-style edge-force inclusion.

        VMEC++ includes edge-force contributions early, and can keep them on when
        the residual drops rapidly between iterations. This helps avoid getting
        trapped in an over-aggressive interior-only update path.
        """
        if it == 0:
            return True
        # Keep edge terms active through the initial transient.
        if it < 8:
            return True
        if len(w_hist) >= 3:
            w0 = max(float(w_hist[-3]), 1e-30)
            w1 = max(float(w_hist[-2]), 1e-30)
            w2 = max(float(w_hist[-1]), 1e-30)
            # Re-enable only for sustained fast drops to avoid on/off chatter.
            if (w1 / w0 < 0.7) and (w2 / w1 < 0.7):
                return True
        if len(fsqr_hist) > 0:
            if float(fsqr_hist[-1]) + float(fsqz_hist[-1]) < 1e-6:
                return True
        return False

    def _safe_dt_from_force(*, dt_nominal: float, frcc, frss, fzsc, fzcs, flsc, flcs) -> float:
        """Limit dt so coefficient updates stay bounded during early iterations."""
        frcc = jnp.asarray(frcc)
        frss = jnp.asarray(frss) if frss is not None else jnp.zeros_like(frcc)
        fzsc = jnp.asarray(fzsc)
        fzcs = jnp.asarray(fzcs) if fzcs is not None else jnp.zeros_like(fzsc)
        flsc = jnp.asarray(flsc)
        flcs = jnp.asarray(flcs) if flcs is not None else jnp.zeros_like(flsc)
        rms = jnp.sqrt(jnp.mean(frcc * frcc + frss * frss + fzsc * fzsc + fzcs * fzcs + flsc * flsc + flcs * flcs))
        rms_f = float(np.asarray(rms))
        if not np.isfinite(rms_f) or rms_f <= 0.0:
            return max(float(dt_nominal), 1e-12)
        # With this integrator, first-step coefficient update is O(dt^2 * force).
        dt_lim = np.sqrt(max_coeff_delta_rms / max(rms_f, 1e-30))
        dt_eff = min(float(dt_nominal), float(dt_lim))
        return max(dt_eff, 1e-12)

    for it in range(max_iter):
        iter2 = it + 1
        iter_since_restart = iter2 - iter1
        zero_m1 = jnp.asarray(
            1.0 if (iter_since_restart < 2) or (len(fsqz2_history) and fsqz2_history[-1] < 1e-6) else 0.0,
            dtype=jnp.asarray(state.Rcos).dtype,
        )
        include_edge = bool(it < 50) and _edge_force_trigger(it, w_history, fsqr2_history, fsqz2_history)
        include_edge_history.append(int(bool(include_edge)))
        zero_m1_history.append(int(float(np.asarray(zero_m1)) > 0.5))

        frzl, fsqr, fsqz, fsql, rz_scale, l_scale = _compute_forces(state, include_edge=include_edge, zero_m1=zero_m1)
        fsqr_f = float(np.asarray(fsqr))
        fsqz_f = float(np.asarray(fsqz))
        fsql_f = float(np.asarray(fsql))

        w_history.append(fsqr_f + fsqz_f + fsql_f)
        fsqr2_history.append(fsqr_f)
        fsqz2_history.append(fsqz_f)
        fsql2_history.append(fsql_f)

        if verbose:
            print(
                f"[solve_fixed_boundary_vmecpp_iter] iter={it:03d} fsqr={fsqr_f:.3e} fsqz={fsqz_f:.3e} fsql={fsql_f:.3e} include_edge={include_edge}"
            )
        # VMEC++ terminates on invariant residuals (fsqr/fsqz/fsql), not fsq1.
        if (fsqr_f <= ftol) and (fsqz_f <= ftol) and (fsql_f <= ftol):
            if verbose:
                print(
                    f"[solve_fixed_boundary_vmecpp_iter] converged: "
                    f"fsqr={fsqr_f:.3e} fsqz={fsqz_f:.3e} fsql={fsql_f:.3e} <= ftol={ftol:.3e}"
                )
            break

        # Precondition forces (radial smoother).
        frcc = _apply_radial_tridi(frzl.frcc * rz_scale[:, None, None], precond_radial_alpha)
        frss = _apply_radial_tridi(frzl.frss * rz_scale[:, None, None], precond_radial_alpha) if frzl.frss is not None else None
        fzsc = _apply_radial_tridi(frzl.fzsc * rz_scale[:, None, None], precond_radial_alpha)
        fzcs = _apply_radial_tridi(frzl.fzcs * rz_scale[:, None, None], precond_radial_alpha) if frzl.fzcs is not None else None
        flsc = _apply_radial_tridi(frzl.flsc * l_scale[:, None, None], precond_lambda_alpha)
        flcs = _apply_radial_tridi(frzl.flcs * l_scale[:, None, None], precond_lambda_alpha) if frzl.flcs is not None else None

        frzl_pre = TomnspsRZL(
            frcc=frcc,
            frss=frss,
            fzsc=fzsc,
            fzcs=fzcs,
            flsc=flsc,
            flcs=flcs,
            frsc=getattr(frzl, "frsc", None),
            frcs=getattr(frzl, "frcs", None),
            fzcc=getattr(frzl, "fzcc", None),
            fzss=getattr(frzl, "fzss", None),
            flcc=getattr(frzl, "flcc", None),
            flss=getattr(frzl, "flss", None),
        )

        # Mode-diagonal preconditioning in VMEC++ (m, n>=0) storage.
        w_mode_mn = _mode_diag_weights_mn(jnp.asarray(frcc).dtype)
        frcc_u = frcc * w_mode_mn[None, :, :]
        frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w_mode_mn[None, :, :]
        fzsc_u = fzsc * w_mode_mn[None, :, :]
        fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w_mode_mn[None, :, :]
        flsc_u = flsc * w_mode_mn[None, :, :]
        flcs_u = (flcs if flcs is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]

        if auto_flip_force and it == 0:
            e0 = float(np.asarray(magnetic_wb_from_state(state, static, indata=indata, signgs=signgs)).ravel()[0])
            dR_test = _mn_cos_to_signed(frcc_u, frss_u)
            dZ_test = _mn_sin_to_signed(fzsc_u, fzcs_u)
            dL_test = _mn_sin_to_signed(flsc_u, flcs_u)
            test_state = VMECState(
                layout=state.layout,
                Rcos=jnp.asarray(state.Rcos) + (-step_size) * dR_test,
                Rsin=state.Rsin,
                Zcos=state.Zcos,
                Zsin=jnp.asarray(state.Zsin) + (-step_size) * dZ_test,
                Lcos=state.Lcos,
                Lsin=jnp.asarray(state.Lsin) + (-step_size) * dL_test,
            )
            e1 = float(np.asarray(magnetic_wb_from_state(test_state, static, indata=indata, signgs=signgs)).ravel()[0])
            if not np.isfinite(e1) or (e1 > e0):
                flip_sign = -1.0
                if verbose:
                    print("[solve_fixed_boundary_vmecpp_iter] flipping force sign (energy increase)")

        # VMEC++-style damping for the fixed-point update.
        gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
            frzl=frzl_pre,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=True,
            apply_scalxc=False,
            s=s,
        )
        rz_norm = vmec_rz_norm_from_state(state=state, static=static)
        f_norm1 = jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=rz_norm.dtype))
        delta_s = jnp.asarray(s[1] - s[0], dtype=rz_norm.dtype)
        fsqr1 = gcr2_p * f_norm1
        fsqz1 = gcz2_p * f_norm1
        fsql1 = gcl2_p * delta_s
        fsqr1_f = float(np.asarray(fsqr1))
        fsqz1_f = float(np.asarray(fsqz1))
        fsql1_f = float(np.asarray(fsql1))
        fsq1 = fsqr1_f + fsqz1_f + fsql1_f
        fsq1_history.append(fsq1)
        fsqr1_history.append(fsqr1_f)
        fsqz1_history.append(fsqz1_f)
        fsql1_history.append(fsql1_f)

        # VMEC++ time-step control trackers.
        fsq = fsqr_f + fsqz_f + fsql_f
        fsq_res = fsq if bool(vmecpp_reference_mode) else fsq1
        if (iter2 == iter1) or (res0 < 0.0):
            res0 = fsq_res
        res0_old = res0
        res0 = min(res0, fsq_res)

        # VMEC++ stores a "good" checkpoint once residual has improved for many
        # iterations since the last restart marker.
        if (fsq1 <= res0_old) and ((iter2 - iter1) > 10):
            state_checkpoint = state

        # VMEC++ restart triggers (bad progress / bad Jacobian proxy).
        bad_jacobian = False
        if bool(vmecpp_reference_mode) and bool(getattr(static.cfg, "lthreed", True)):
            jac = vmec_half_mesh_jacobian_from_state(state=state, modes=static.modes, trig=trig, s=s)
            tau = np.asarray(jac.tau)
            if tau.size:
                min_tau = float(np.min(tau))
                max_tau = float(np.max(tau))
                bad_jacobian = (iter2 > 1) and (int(zero_m1) == 0) and ((min_tau * max_tau) < 0.0)
                min_tau_history.append(min_tau)
                max_tau_history.append(max_tau)
                bad_jacobian_history.append(int(bad_jacobian))
            else:
                min_tau_history.append(float("nan"))
                max_tau_history.append(float("nan"))
                bad_jacobian_history.append(0)
        else:
            min_tau_history.append(float("nan"))
            max_tau_history.append(float("nan"))
            bad_jacobian_history.append(0)
        pre_restart_reason = "none"
        if fsq_res > 100.0 * max(res0, 1e-30):
            bad_growth_streak += 1
        else:
            bad_growth_streak = 0

        huge_initial_forces = False
        if bool(vmecpp_reference_mode):
            # VMEC++ behavior (vmec.cc):
            # - BAD_JACOBIAN if fsq > 100 * res0 and iter2 > iter1
            # - BAD_PROGRESS if iter2 progression is large and fsqr+fsqz > 1e-2
            if bad_jacobian:
                pre_restart_reason = "bad_jacobian"
            elif (iter2 > iter1) and (fsq > 100.0 * max(res0, 1e-30)):
                pre_restart_reason = "bad_jacobian"
            elif (
                (iter2 - iter1) > (k_preconditioner_update_interval // 2)
                and (iter2 > 2 * k_preconditioner_update_interval)
                and ((fsqr_f + fsqz_f) > 1.0e-2)
            ):
                pre_restart_reason = "bad_progress"
        else:
            if (iter2 > (iter1 + 8)) and (bad_growth_streak >= 2):
                pre_restart_reason = "bad_jacobian"
            elif (
                (iter2 - iter1) > (k_preconditioner_update_interval // 2)
                and (iter2 > 2 * k_preconditioner_update_interval)
                and (fsq1 > 5.0 * max(res0, 1e-30))
                and (fsq1 > 0.95 * max(fsq_prev, 1e-30))
            ):
                pre_restart_reason = "bad_progress"

        if use_vmecpp_restart_triggers and pre_restart_reason != "none":
            state = state_checkpoint
            vRcc = jnp.zeros_like(vRcc)
            vRss = jnp.zeros_like(vRss)
            vZsc = jnp.zeros_like(vZsc)
            vZcs = jnp.zeros_like(vZcs)
            vLsc = jnp.zeros_like(vLsc)
            vLcs = jnp.zeros_like(vLcs)
            if pre_restart_reason == "bad_jacobian":
                time_step = max(restart_badjac_factor * time_step, 1e-12)
                ijacob += 1
                step_status = "restart_bad_jacobian"
            else:
                time_step = max(time_step / restart_badprog_factor, 1e-12)
                step_status = "restart_bad_progress"
            if bool(huge_initial_forces) and (pre_restart_reason == "bad_jacobian"):
                huge_force_restart_count += 1
            else:
                huge_force_restart_count = 0
            if ijacob in (25, 50):
                scale = 0.98 if ijacob < 50 else 0.96
                time_step = max(scale * float(step_size), 1e-12)
            bad_resets += 1
            iter1 = iter2
            bad_growth_streak = 0
            step_history.append(0.0)
            dt_eff_history.append(0.0)
            update_rms_history.append(0.0)
            w_curr_history.append(float(fsqr_f + fsqz_f + fsql_f))
            w_try_history.append(float("nan"))
            w_try_ratio_history.append(float("nan"))
            restart_path_history.append("pre_restart_trigger")
            step_status_history.append(step_status)
            restart_reason_history.append(pre_restart_reason)
            pre_restart_reason_history.append(pre_restart_reason)
            time_step_history.append(float(time_step))
            res0_history.append(float(res0))
            fsq_prev_history.append(float(fsq_prev))
            bad_growth_streak_history.append(int(bad_growth_streak))
            iter1_history.append(int(iter1))
            grad_rms_history.append(float(np.sqrt(max(fsqr_f + fsqz_f + fsql_f, 0.0))))
            if verbose:
                print(
                    f"[solve_fixed_boundary_vmecpp_iter] iter={it:03d} "
                    f"dt_eff=0.000e+00 update_rms=0.000e+00 "
                    f"fsqr1={fsqr1_f:.3e} fsqz1={fsqz1_f:.3e} fsql1={fsql1_f:.3e} "
                    f"step_status={step_status}"
                )
            continue

        if iter2 == iter1:
            inv_tau = [0.15 / time_step] * k_ndamp
        else:
            invtau_num = 0.0 if fsq1 == 0.0 else min(abs(np.log(fsq1 / fsq_prev)), 0.15)
            inv_tau = inv_tau[1:] + [invtau_num / time_step]
        fsq_prev = fsq1

        otav = float(np.sum(inv_tau)) / float(k_ndamp)
        dtau = time_step * otav / 2.0
        b1 = 1.0 - dtau
        fac = 1.0 / (1.0 + dtau)

        if bool(vmecpp_strict_update):
            # VMEC++ update semantics: one preconditioned momentum update per
            # iteration in (m, n>=0) storage, no line-search accept/reject.
            w_curr = fsqr_f + fsqz_f + fsql_f
            state_backup = state
            dt_eff = _safe_dt_from_force(
                dt_nominal=time_step,
                frcc=frcc_u,
                frss=frss_u,
                fzsc=fzsc_u,
                fzcs=fzcs_u,
                flsc=flsc_u,
                flcs=flcs_u,
            )

            # VMEC++ semantics: v <- fac*(b1*v + dt*F), x <- x + dt*v.
            # Do not drop the dt factor in the force term; otherwise updates
            # scale like O(dt) instead of O(dt^2) and can immediately blow up.
            force_scale = float(dt_eff)

            vRcc = fac * (b1 * vRcc + force_scale * (flip_sign * jnp.asarray(frcc_u)))
            vRss = fac * (b1 * vRss + force_scale * (flip_sign * jnp.asarray(frss_u)))
            vZsc = fac * (b1 * vZsc + force_scale * (flip_sign * jnp.asarray(fzsc_u)))
            vZcs = fac * (b1 * vZcs + force_scale * (flip_sign * jnp.asarray(fzcs_u)))
            vLsc = fac * (b1 * vLsc + force_scale * (flip_sign * jnp.asarray(flsc_u)))
            vLcs = fac * (b1 * vLcs + force_scale * (flip_sign * jnp.asarray(flcs_u)))

            update_rms = float(
                np.asarray(
                    jnp.sqrt(
                        jnp.mean(
                            (dt_eff * vRcc) ** 2
                            + (dt_eff * vRss) ** 2
                            + (dt_eff * vZsc) ** 2
                            + (dt_eff * vZcs) ** 2
                            + (dt_eff * vLsc) ** 2
                            + (dt_eff * vLcs) ** 2
                        )
                    )
                )
            )
            if np.isfinite(update_rms) and (update_rms > max_update_rms):
                scl = max_update_rms / max(update_rms, 1e-30)
                vRcc = vRcc * scl
                vRss = vRss * scl
                vZsc = vZsc * scl
                vZcs = vZcs * scl
                vLsc = vLsc * scl
                vLcs = vLcs * scl
                update_rms = float(
                    np.asarray(
                        jnp.sqrt(
                            jnp.mean(
                                (dt_eff * vRcc) ** 2
                                + (dt_eff * vRss) ** 2
                                + (dt_eff * vZsc) ** 2
                                + (dt_eff * vZcs) ** 2
                                + (dt_eff * vLsc) ** 2
                                + (dt_eff * vLcs) ** 2
                            )
                        )
                    )
                )

            dR = dt_eff * _mn_cos_to_signed(vRcc, vRss)
            dZ = dt_eff * _mn_sin_to_signed(vZsc, vZcs)
            dL = dt_eff * _mn_sin_to_signed(vLsc, vLcs)
            state_try = VMECState(
                layout=state.layout,
                Rcos=jnp.asarray(state.Rcos) + dR,
                Rsin=state.Rsin,
                Zcos=state.Zcos,
                Zsin=jnp.asarray(state.Zsin) + dZ,
                Lcos=state.Lcos,
                Lsin=jnp.asarray(state.Lsin) + dL,
            )
            state_try = _enforce_fixed_boundary_and_axis(
                state_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                enforce_lambda_axis=False,
                idx00=idx00,
            )
            _, fsqr_t, fsqz_t, fsql_t, _, _ = _compute_forces(
                state_try,
                include_edge=include_edge,
                zero_m1=zero_m1,
            )
            w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
            w_try_ratio = w_try / max(w_curr, 1e-30) if np.isfinite(w_try) else float("inf")

            # Catastrophic guard only; VMEC++ handles difficult steps through
            # restart/timestep control, not per-step rejection.
            if np.isfinite(w_try) and (w_try <= 1.0e3 * max(w_curr, 1e-30)):
                state = state_try
                step_status = "momentum"
                restart_reason = "none"
                huge_force_restart_count = 0
                restart_path = "momentum_accept"
            else:
                if use_direct_fallback:
                    # Try a small direct-force step (no momentum memory) before
                    # a full restart. This is an experimental parity path.
                    dt_direct = max(0.1 * dt_eff, 1e-12)
                    force_rms = float(
                        np.asarray(
                            jnp.sqrt(
                                jnp.mean(
                                    frcc_u * frcc_u
                                    + frss_u * frss_u
                                    + fzsc_u * fzsc_u
                                    + fzcs_u * fzcs_u
                                    + flsc_u * flsc_u
                                    + flcs_u * flcs_u
                                )
                            )
                        )
                    )
                    if np.isfinite(force_rms) and force_rms > 0.0:
                        dt_cap = max_update_rms / max(force_rms, 1e-30)
                        dt_direct = max(min(dt_direct, float(dt_cap)), 1e-12)
                    dR_dir = dt_direct * _mn_cos_to_signed(flip_sign * frcc_u, flip_sign * frss_u)
                    dZ_dir = dt_direct * _mn_sin_to_signed(flip_sign * fzsc_u, flip_sign * fzcs_u)
                    dL_dir = dt_direct * _mn_sin_to_signed(flip_sign * flsc_u, flip_sign * flcs_u)
                    state_dir = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + dR_dir,
                        Rsin=state.Rsin,
                        Zcos=state.Zcos,
                        Zsin=jnp.asarray(state.Zsin) + dZ_dir,
                        Lcos=state.Lcos,
                        Lsin=jnp.asarray(state.Lsin) + dL_dir,
                    )
                    state_dir = _enforce_fixed_boundary_and_axis(
                        state_dir,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_lambda_axis=False,
                        idx00=idx00,
                    )
                    _, fsqr_d, fsqz_d, fsql_d, _, _ = _compute_forces(
                        state_dir,
                        include_edge=include_edge,
                        zero_m1=zero_m1,
                    )
                    w_dir = float(np.asarray(fsqr_d + fsqz_d + fsql_d))
                    if np.isfinite(w_dir) and (w_dir <= 1.5 * max(w_curr, 1e-30)):
                        state = state_dir
                        vRcc = jnp.zeros_like(vRcc)
                        vRss = jnp.zeros_like(vRss)
                        vZsc = jnp.zeros_like(vZsc)
                        vZcs = jnp.zeros_like(vZcs)
                        vLsc = jnp.zeros_like(vLsc)
                        vLcs = jnp.zeros_like(vLcs)
                        step_status = "fallback_direct"
                        restart_reason = "none"
                        huge_force_restart_count = 0
                        restart_path = "fallback_direct"
                        update_rms = float(
                            np.asarray(
                                jnp.sqrt(
                                    jnp.mean(
                                        (dt_direct * frcc_u) ** 2
                                        + (dt_direct * frss_u) ** 2
                                        + (dt_direct * fzsc_u) ** 2
                                        + (dt_direct * fzcs_u) ** 2
                                        + (dt_direct * flsc_u) ** 2
                                        + (dt_direct * flcs_u) ** 2
                                    )
                                )
                            )
                        )
                    else:
                        # VMEC++ RestartIteration-style rollback + zero velocity.
                        state = state_backup
                        vRcc = jnp.zeros_like(vRcc)
                        vRss = jnp.zeros_like(vRss)
                        vZsc = jnp.zeros_like(vZsc)
                        vZcs = jnp.zeros_like(vZcs)
                        vLsc = jnp.zeros_like(vLsc)
                        vLcs = jnp.zeros_like(vLcs)
                        # Tighten displacement caps when restarting from
                        # catastrophic growth; otherwise dt_eff can remain
                        # stuck at the same limit.
                        max_coeff_delta_rms = max(0.5 * max_coeff_delta_rms, 1e-12)
                        max_update_rms = max(0.8 * max_update_rms, 1e-6)
                        if not np.isfinite(w_try):
                            time_step = max(restart_badjac_factor * time_step, 1e-12)
                            ijacob += 1
                            restart_reason = "bad_jacobian"
                            step_status = "restart_bad_jacobian"
                            restart_path = "catastrophic_nonfinite"
                        else:
                            time_step = max(time_step / restart_badprog_factor, 1e-12)
                            restart_reason = "bad_progress"
                            step_status = "restart_bad_progress"
                            restart_path = "catastrophic_growth"
                        # VMEC++ adjusts delt0r at reset milestones.
                        if ijacob in (25, 50):
                            scale = 0.98 if ijacob < 50 else 0.96
                            time_step = max(scale * float(step_size), 1e-12)
                        bad_resets += 1
                        iter1 = iter2
                        update_rms = 0.0
                else:
                    # VMEC++ RestartIteration-style rollback + zero velocity.
                    state = state_backup
                    vRcc = jnp.zeros_like(vRcc)
                    vRss = jnp.zeros_like(vRss)
                    vZsc = jnp.zeros_like(vZsc)
                    vZcs = jnp.zeros_like(vZcs)
                    vLsc = jnp.zeros_like(vLsc)
                    vLcs = jnp.zeros_like(vLcs)
                    # Tighten displacement caps when restarting from catastrophic
                    # growth; otherwise dt_eff can remain stuck at the same limit.
                    max_coeff_delta_rms = max(0.5 * max_coeff_delta_rms, 1e-12)
                    max_update_rms = max(0.8 * max_update_rms, 1e-6)
                    if not np.isfinite(w_try):
                        time_step = max(restart_badjac_factor * time_step, 1e-12)
                        ijacob += 1
                        restart_reason = "bad_jacobian"
                        step_status = "restart_bad_jacobian"
                        restart_path = "catastrophic_nonfinite"
                    else:
                        time_step = max(time_step / restart_badprog_factor, 1e-12)
                        restart_reason = "bad_progress"
                        step_status = "restart_bad_progress"
                        restart_path = "catastrophic_growth"
                    # VMEC++ adjusts delt0r at reset milestones.
                    if ijacob in (25, 50):
                        scale = 0.98 if ijacob < 50 else 0.96
                        time_step = max(scale * float(step_size), 1e-12)
                    bad_resets += 1
                    iter1 = iter2
                    update_rms = 0.0
            step_history.append(float(dt_eff))
            w_curr_history.append(float(w_curr))
            w_try_history.append(float(w_try))
            w_try_ratio_history.append(float(w_try_ratio))
            restart_path_history.append(str(restart_path))
        else:
            accepted = False
            step_status = "rejected"
            step_factor = 1.0
            vRcc_best, vRss_best = vRcc, vRss
            vZsc_best, vZcs_best = vZsc, vZcs
            vLsc_best, vLcs_best = vLsc, vLcs
            state_best = state
            dt_eff = float(time_step)
            update_rms = 0.0
            w_curr = fsqr_f + fsqz_f + fsql_f

            for _bt in range(6):
                dt_try = time_step * step_factor
                vRcc_try = fac * (b1 * vRcc + dt_try * (flip_sign * jnp.asarray(frcc_u)))
                vRss_try = fac * (b1 * vRss + dt_try * (flip_sign * jnp.asarray(frss_u)))
                vZsc_try = fac * (b1 * vZsc + dt_try * (flip_sign * jnp.asarray(fzsc_u)))
                vZcs_try = fac * (b1 * vZcs + dt_try * (flip_sign * jnp.asarray(fzcs_u)))
                vLsc_try = fac * (b1 * vLsc + dt_try * (flip_sign * jnp.asarray(flsc_u)))
                vLcs_try = fac * (b1 * vLcs + dt_try * (flip_sign * jnp.asarray(flcs_u)))

                dR_try = dt_try * _mn_cos_to_signed(vRcc_try, vRss_try)
                dZ_try = dt_try * _mn_sin_to_signed(vZsc_try, vZcs_try)
                dL_try = dt_try * _mn_sin_to_signed(vLsc_try, vLcs_try)

                state_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + dR_try,
                    Rsin=state.Rsin,
                    Zcos=state.Zcos,
                    Zsin=jnp.asarray(state.Zsin) + dZ_try,
                    Lcos=state.Lcos,
                    Lsin=jnp.asarray(state.Lsin) + dL_try,
                )
                state_try = _enforce_fixed_boundary_and_axis(
                    state_try,
                    static,
                    edge_Rcos=edge_Rcos,
                    edge_Rsin=edge_Rsin,
                    edge_Zcos=edge_Zcos,
                    edge_Zsin=edge_Zsin,
                    enforce_lambda_axis=False,
                    idx00=idx00,
                )
                _, fsqr_t, fsqz_t, fsql_t, _, _ = _compute_forces(
                    state_try,
                    include_edge=include_edge,
                    zero_m1=zero_m1,
                )
                w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                if np.isfinite(w_try) and (w_try <= 1.05 * w_curr):
                    accepted = True
                    step_status = "momentum"
                    state_best = state_try
                    vRcc_best, vRss_best = vRcc_try, vRss_try
                    vZsc_best, vZcs_best = vZsc_try, vZcs_try
                    vLsc_best, vLcs_best = vLsc_try, vLcs_try
                    dt_eff = float(dt_try)
                    update_rms = float(
                        np.asarray(
                            jnp.sqrt(
                                jnp.mean(
                                    (dt_try * vRcc_try) ** 2
                                    + (dt_try * vRss_try) ** 2
                                    + (dt_try * vZsc_try) ** 2
                                    + (dt_try * vZcs_try) ** 2
                                    + (dt_try * vLsc_try) ** 2
                                    + (dt_try * vLcs_try) ** 2
                                )
                            )
                        )
                    )
                    break
                step_factor *= 0.5

            state = state_best
            vRcc, vRss = vRcc_best, vRss_best
            vZsc, vZcs = vZsc_best, vZcs_best
            vLsc, vLcs = vLsc_best, vLcs_best
            if not accepted:
                # No acceptable update was found; damp velocity to avoid runaway.
                vRcc = 0.5 * vRcc
                vRss = 0.5 * vRss
                vZsc = 0.5 * vZsc
                vZcs = 0.5 * vZcs
                vLsc = 0.5 * vLsc
                vLcs = 0.5 * vLcs
                dt_eff = float(step_size * step_factor)
                update_rms = 0.0
                step_status = "rejected"
            step_history.append(dt_eff)
            restart_reason = "none"
            w_curr_history.append(float(w_curr))
            w_try_history.append(float("nan"))
            w_try_ratio_history.append(float("nan"))
            restart_path_history.append("non_strict")
        dt_eff_history.append(float(dt_eff))
        update_rms_history.append(float(update_rms))
        if verbose:
            print(
                f"[solve_fixed_boundary_vmecpp_iter] iter={it:03d} "
                f"dt_eff={dt_eff:.3e} update_rms={update_rms:.3e} "
                f"fsqr1={fsqr1_f:.3e} fsqz1={fsqz1_f:.3e} fsql1={fsql1_f:.3e} "
                f"step_status={step_status}"
            )
        step_status_history.append(step_status)
        restart_reason_history.append(restart_reason)
        pre_restart_reason_history.append(pre_restart_reason)
        time_step_history.append(float(time_step))
        res0_history.append(float(res0))
        fsq_prev_history.append(float(fsq_prev))
        bad_growth_streak_history.append(int(bad_growth_streak))
        iter1_history.append(int(iter1))
        grad_rms_history.append(float(np.sqrt(max(fsqr_f + fsqz_f + fsql_f, 0.0))))

    diag: Dict[str, Any] = {
        "ftol": ftol,
        "step_size": float(step_size),
        "precond_radial_alpha": float(precond_radial_alpha),
        "precond_lambda_alpha": float(precond_lambda_alpha),
        "vmecpp_strict_update": bool(vmecpp_strict_update),
        "vmecpp_reference_mode": bool(vmecpp_reference_mode),
        "use_vmecpp_restart_triggers": bool(use_vmecpp_restart_triggers),
        "use_direct_fallback": bool(use_direct_fallback),
        "max_update_rms": float(max_update_rms),
        "ijacob": int(ijacob),
        "bad_resets": int(bad_resets),
        "iter1_final": int(iter1),
        "res0": float(res0),
        "step_status_history": np.asarray(step_status_history, dtype=object),
        "restart_reason_history": np.asarray(restart_reason_history, dtype=object),
        "pre_restart_reason_history": np.asarray(pre_restart_reason_history, dtype=object),
        "time_step_history": np.asarray(time_step_history, dtype=float),
        "res0_history": np.asarray(res0_history, dtype=float),
        "fsq_prev_history": np.asarray(fsq_prev_history, dtype=float),
        "bad_growth_streak_history": np.asarray(bad_growth_streak_history, dtype=int),
        "iter1_history": np.asarray(iter1_history, dtype=int),
        "include_edge_history": np.asarray(include_edge_history, dtype=int),
        "zero_m1_history": np.asarray(zero_m1_history, dtype=int),
        "dt_eff_history": np.asarray(dt_eff_history, dtype=float),
        "update_rms_history": np.asarray(update_rms_history, dtype=float),
        "w_curr_history": np.asarray(w_curr_history, dtype=float),
        "w_try_history": np.asarray(w_try_history, dtype=float),
        "w_try_ratio_history": np.asarray(w_try_ratio_history, dtype=float),
        "restart_path_history": np.asarray(restart_path_history, dtype=object),
        "min_tau_history": np.asarray(min_tau_history, dtype=float),
        "max_tau_history": np.asarray(max_tau_history, dtype=float),
        "bad_jacobian_history": np.asarray(bad_jacobian_history, dtype=int),
        "fsq1_history": np.asarray(fsq1_history, dtype=float),
        "fsqr1_history": np.asarray(fsqr1_history, dtype=float),
        "fsqz1_history": np.asarray(fsqz1_history, dtype=float),
        "fsql1_history": np.asarray(fsql1_history, dtype=float),
    }
    return SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )
