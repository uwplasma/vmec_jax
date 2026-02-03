"""Force/residual diagnostics (step-10).

VMEC reports scalar residual measures (e.g. ``fsqr/fsqz/fsql``) that track
how close a state is to satisfying the equilibrium Euler-Lagrange equations.

In vmec_jax we do not yet reproduce VMEC's full real-space force kernels, but we
*can* provide a useful, differentiable proxy based on the gradient of the total
objective. These diagnostics are meant to:

- support regression tests (consistency across refactors),
- provide solver stopping criteria beyond just energy decrease,
- and act as a stepping stone toward full VMEC ``residue/getfsq`` parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from ._compat import has_jax, jax, jnp
from .energy import FluxProfiles
from .field import TWOPI, b2_from_bsup, bsub_from_bsup, bsup_from_geom
from .geom import eval_geom
from .grids import angle_steps
from .solve import _mask_grad_for_constraints, _mode00_index
from .state import VMECState


@dataclass(frozen=True)
class ForceResiduals:
    """Scalar residual diagnostics derived from objective gradients."""

    fsqr_like: float
    fsqz_like: float
    fsql_like: float
    fsq_like: float
    grad_rms: float
    grad_rms_rz: float
    grad_rms_l: float
    diagnostics: Dict[str, Any]


def _rms(a: Any) -> float:
    x = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _sum_squares_state(grad: VMECState) -> Tuple[float, float, float]:
    """Return (sum_R, sum_Z, sum_L) of squared gradient coefficients."""
    gR = float(np.sum(np.asarray(grad.Rcos) ** 2) + np.sum(np.asarray(grad.Rsin) ** 2))
    gZ = float(np.sum(np.asarray(grad.Zcos) ** 2) + np.sum(np.asarray(grad.Zsin) ** 2))
    gL = float(np.sum(np.asarray(grad.Lcos) ** 2) + np.sum(np.asarray(grad.Lsin) ** 2))
    return gR, gZ, gL


def _objective_total(
    state: VMECState,
    static,
    *,
    flux: FluxProfiles,
    pressure,
    gamma: float,
    jacobian_penalty: float,
) -> Any:
    g = eval_geom(state, static)
    bsupu, bsupv = bsup_from_geom(
        g,
        phipf=flux.phipf,
        chipf=flux.chipf,
        nfp=int(static.cfg.nfp),
        signgs=int(flux.signgs),
        lamscale=flux.lamscale,
    )
    B2 = b2_from_bsup(g, bsupu, bsupv)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    ds = jnp.asarray(1.0, dtype=s.dtype) if s.shape[0] < 2 else (s[1] - s[0])
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    jac = int(flux.signgs) * g.sqrtg
    wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
    wp = (jnp.sum(jnp.asarray(pressure)[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
    w = wb + wp / (gamma - 1.0)

    jac2 = jac.at[0, :, :].set(0.0)
    neg = jnp.minimum(jac2, 0.0)
    penalty = float(jacobian_penalty) * jnp.mean(neg * neg)
    return w + penalty


def force_residuals_from_state(
    state: VMECState,
    static,
    *,
    flux: FluxProfiles,
    pressure,
    gamma: float,
    jacobian_penalty: float = 1e3,
    eps: float = 1e-30,
) -> ForceResiduals:
    """Compute force-like scalar residuals from the total-objective gradient.

    The outputs are *not* yet equal to VMEC's ``fsqr/fsqz/fsql``. They are
    normalized in a VMEC-inspired way and intended for diagnostics/regressions.
    """
    if not has_jax():
        raise ImportError("force_residuals_from_state requires JAX (jax + jaxlib)")

    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("gamma=1 makes wp/(gamma-1) singular")

    idx00 = _mode00_index(static.modes)

    pressure = jnp.asarray(pressure)
    if pressure.shape != jnp.asarray(static.s).shape:
        raise ValueError(f"pressure must have shape {np.asarray(static.s).shape}, got {pressure.shape}")

    obj = lambda st: _objective_total(st, static, flux=flux, pressure=pressure, gamma=gamma, jacobian_penalty=jacobian_penalty)
    val, grad = jax.value_and_grad(obj)(state)
    grad = _mask_grad_for_constraints(grad, static, idx00=idx00)

    gR2, gZ2, gL2 = _sum_squares_state(grad)
    grad_rms = float(np.sqrt((gR2 + gZ2 + gL2) / max(1, (np.asarray(grad.Rcos).size * 6))))
    grad_rms_rz = float(np.sqrt((gR2 + gZ2) / max(1, (np.asarray(grad.Rcos).size * 4))))
    grad_rms_l = float(np.sqrt(gL2 / max(1, (np.asarray(grad.Lcos).size * 2))))

    # VMEC-inspired normalization constants (approximate):
    # - r0scale==1 in VMEC's fixaray scaling, so r1 = 1/(2*r0scale)^2 = 1/4.
    r1 = 0.25

    # Build a geometry-based scale for R/Z forces (fnorm).
    g = eval_geom(state, static)
    jac = int(flux.signgs) * np.asarray(g.sqrtg)
    jac[0, :, :] = 0.0

    # Approximate VMEC's sum(guu*wint) with a volume-weighted integral of g_tt.
    denom_rz = float(np.sum(np.asarray(g.g_tt) * jac)) + eps

    # Approximate VMEC's r2 = max(wb,wp)/volume (use |wb| if pressure=0).
    bsupu, bsupv = bsup_from_geom(
        g,
        phipf=flux.phipf,
        chipf=flux.chipf,
        nfp=int(static.cfg.nfp),
        signgs=int(flux.signgs),
        lamscale=flux.lamscale,
    )
    B2 = np.asarray(b2_from_bsup(g, bsupu, bsupv))
    s = np.asarray(static.s)
    ds = float(s[1] - s[0]) if s.size > 1 else 1.0
    dtheta = 2.0 * np.pi / int(static.cfg.ntheta)
    dzeta = 2.0 * np.pi / int(static.cfg.nzeta)
    dV = float(np.sum(jac) * ds * dtheta * dzeta)
    wb = float(np.sum(0.5 * B2 * jac) * ds * dtheta * dzeta) / (TWOPI * TWOPI)
    wp = float(np.sum(np.asarray(pressure)[:, None, None] * jac) * ds * dtheta * dzeta) / (TWOPI * TWOPI)
    r2 = max(abs(wb), abs(wp)) / max(abs(dV), eps)
    fnorm = 1.0 / (denom_rz * (r2 * r2) + eps)

    # Lambda normalization based on covariant field magnitude (approximate fnormL).
    bsubu, bsubv = bsub_from_bsup(g, bsupu, bsupv)
    denom_l = float(np.sum((np.asarray(bsubu) ** 2 + np.asarray(bsubv) ** 2) * jac) * (float(flux.lamscale) ** 2)) + eps
    fnormL = 1.0 / denom_l

    fsqr_like = r1 * fnorm * gR2
    fsqz_like = r1 * fnorm * gZ2
    fsql_like = fnormL * gL2
    fsq_like = fsqr_like + fsqz_like + fsql_like

    diag = {
        "objective": float(np.asarray(val)),
        "wb": float(wb),
        "wp": float(wp),
        "volume": float(dV),
        "fnorm": float(fnorm),
        "fnormL": float(fnormL),
        "r2": float(r2),
        "idx00": idx00,
    }
    return ForceResiduals(
        fsqr_like=float(fsqr_like),
        fsqz_like=float(fsqz_like),
        fsql_like=float(fsql_like),
        fsq_like=float(fsq_like),
        grad_rms=float(grad_rms),
        grad_rms_rz=float(grad_rms_rz),
        grad_rms_l=float(grad_rms_l),
        diagnostics=diag,
    )
