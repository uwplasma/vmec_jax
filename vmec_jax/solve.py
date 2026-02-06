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
        if rhs.ndim != 2:
            raise ValueError(f"expected (ns,K), got {rhs.shape}")
        ns = int(rhs.shape[0])
        if ns < 3:
            return rhs
        alpha = jnp.asarray(alpha, dtype=rhs.dtype)
        a = -alpha
        b = 1.0 + 2.0 * alpha
        c = -alpha

        x0 = rhs[0]
        xN = rhs[-1]
        d = rhs[1:-1]
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

    The objective is computed using the Step-10 parity pipeline:
      `bcovar` -> `forces` -> `tomnsps` -> sum-of-squares of Fourier residual blocks,
    using VMEC's `getfsq` conventions (notably: post-`tomnsps` `scalxc` scaling,
    optional converged-iteration m=1 constraints, and R/Z edge exclusion).

    Notes
    -----
    - For parity, build `static` with `vmec_angle_grid(...)` (see `vmec_jax.vmec_tomnsp`).
    - This solver does not include VMEC's iteration-dependent switching logic
      (e.g. `lforbal` triggering); it provides a differentiable objective suitable
      for regression and initial end-to-end parity.
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

    for it in range(int(max_iter)):
        zero_m1 = jnp.asarray(1.0 if (it < 2) or (fsqz2_history[-1] < 1e-6) else 0.0, dtype=jnp.asarray(state.Rcos).dtype)
        r, pullback = jax.vjp(_residual_vec_jit, state, zero_m1)
        # Gradient of 0.5*||r||^2 is J^T r.
        g_state = pullback(r)[0]
        g_state = _project_step(g_state)
        grad_rms_history.append(_grad_rms_state(g_state))

        b_flat = -pack_state(g_state)

        def _matvec(v_flat):
            v_state = unpack_state(v_flat, state.layout)
            v_state = _project_step(v_state)
            zero_tangent = jnp.zeros_like(zero_m1)
            jv = jax.jvp(_residual_vec_jit, (state, zero_m1), (v_state, zero_tangent))[1]
            jt_jv = pullback(jv)[0]
            jt_jv = _project_step(jt_jv)
            if damping != 0.0:
                jt_jv = VMECState(
                    layout=jt_jv.layout,
                    Rcos=jt_jv.Rcos + float(damping) * v_state.Rcos,
                    Rsin=jt_jv.Rsin + float(damping) * v_state.Rsin,
                    Zcos=jt_jv.Zcos + float(damping) * v_state.Zcos,
                    Zsin=jt_jv.Zsin + float(damping) * v_state.Zsin,
                    Lcos=jt_jv.Lcos + float(damping) * v_state.Lcos,
                    Lsin=jt_jv.Lsin + float(damping) * v_state.Lsin,
                )
            return pack_state(jt_jv)

        dx_flat, _info = cg(_matvec, b_flat, tol=float(cg_tol), maxiter=int(cg_maxiter))
        dx_state = unpack_state(dx_flat, state.layout)
        dx_state = _project_step(dx_state)

        accepted = False
        step = float(step_size)
        w_curr = w_history[-1]
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
            print(f"[solve_fixed_boundary_gn_vmec_residual] iter={it:03d} w={w_history[-1]:.8e} step={step:.3e} accepted={accepted}")

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
