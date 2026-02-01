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
from typing import Any, Dict, Optional

import numpy as np

from ._compat import has_jax, jax, jnp, jit
from .field import TWOPI, bsup_from_sqrtg_lambda
from .fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from .geom import eval_geom
from .state import VMECState


@dataclass(frozen=True)
class SolveLambdaResult:
    state: VMECState
    n_iter: int
    wb_history: np.ndarray
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
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    verbose: bool = False,
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
    dtheta = theta[1] - theta[0]
    dzeta = zeta[1] - zeta[0]
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

    wb_and_grad = jit(jax.value_and_grad(_wb_from_L, argnums=(0, 1)))
    wb_only = jit(_wb_from_L)

    Lcos = jnp.asarray(state0.Lcos)
    Lsin = jnp.asarray(state0.Lsin)
    Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)

    wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)
    wb_history = [float(np.asarray(wb0))]
    grad_rms_history = []
    step_history = []

    for it in range(max_iter):
        grad_rms = float(np.sqrt(np.mean(np.asarray(gcos) ** 2 + np.asarray(gsin) ** 2)))
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
            Lcos_t = Lcos - step * gcos
            Lsin_t = Lsin - step * gsin
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
