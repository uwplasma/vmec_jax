"""VMEC-style half-mesh Jacobian construction.

This module ports the core logic of VMEC2000's ``jacobian.f`` / ``jacobian_par``
into a small, dependency-light implementation.

Motivation
----------
VMEC uses an internal representation in which *odd-m* Fourier content is stored
in a ``1/sqrt(s)`` form for axis regularity. In real space, many quantities are
represented as:

    X(s,θ,ζ) = X_even(s,θ,ζ) + sqrt(s) * X_odd(s,θ,ζ)

VMEC then constructs several derivatives and the Jacobian on the **radial half
mesh** with explicit correction terms arising from ``d/ds sqrt(s)``.

The direct Cartesian cross-product Jacobian in :mod:`vmec_jax.geom` is fine for
early steps, but does not match VMEC's discrete half-mesh convention used for
Nyquist ``wout`` fields like ``gmnc/gmns``. This module exists specifically for
Step-10 parity work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .vmec_realspace import vmec_realspace_synthesis, vmec_realspace_synthesis_dtheta
from .vmec_tomnsp import VmecTrigTables


@dataclass(frozen=True)
class VmecHalfMeshJacobian:
    """Half-mesh Jacobian outputs (VMEC conventions)."""

    # R on half mesh.
    r12: Any  # (ns, ntheta, nzeta)
    # Rs and Zs on half mesh.
    rs: Any  # (ns, ntheta, nzeta)
    zs: Any  # (ns, ntheta, nzeta)
    # Ru and Zu on half mesh.
    ru12: Any  # (ns, ntheta, nzeta)
    zu12: Any  # (ns, ntheta, nzeta)
    # tau = sqrt(g)/R on half mesh (VMEC name).
    tau: Any  # (ns, ntheta, nzeta)
    # sqrt(g) on half mesh.
    sqrtg: Any  # (ns, ntheta, nzeta)


def _safe_divide(x, y, *, eps: float = 1e-14):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    mask = jnp.abs(y) > eps
    y_safe = jnp.where(mask, y, jnp.ones_like(y))
    return mask.astype(x.dtype) * (x / y_safe)


def _pshalf_from_s(s: Any) -> Any:
    """Compute VMEC-like sqrt(s) on the half mesh."""
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    sh = 0.5 * (s[1:] + s[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, 0.0))


def jacobian_half_mesh_from_parity(
    *,
    pr1_even,
    pr1_odd,
    pz1_even,
    pz1_odd,
    pru_even,
    pru_odd,
    pzu_even,
    pzu_odd,
    s,
) -> VmecHalfMeshJacobian:
    """Compute half-mesh Jacobian quantities using VMEC's discrete formula.

    Parameters
    ----------
    pr1_even, pr1_odd, ... :
        Real-space fields representing the internal VMEC decomposition:

            X = X_even + sqrt(s)*X_odd

        Each array has shape ``(ns, ntheta, nzeta)``.
    s:
        Radial grid (ns,), assumed uniform.
    """
    pr1_even = jnp.asarray(pr1_even)
    pr1_odd = jnp.asarray(pr1_odd)
    pz1_even = jnp.asarray(pz1_even)
    pz1_odd = jnp.asarray(pz1_odd)
    pru_even = jnp.asarray(pru_even)
    pru_odd = jnp.asarray(pru_odd)
    pzu_even = jnp.asarray(pzu_even)
    pzu_odd = jnp.asarray(pzu_odd)
    s = jnp.asarray(s)

    ns = int(s.shape[0])
    if ns < 2:
        z = jnp.zeros_like(pr1_even)
        return VmecHalfMeshJacobian(r12=pr1_even, rs=z, zs=z, ru12=z, zu12=z, tau=z, sqrtg=z)

    hs = s[1] - s[0]
    ohs = _safe_divide(1.0, hs)
    # This is exactly VMEC's `p25 = (0.5)^2`.
    dshalfds = 0.25

    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    pshalf = _pshalf_from_s(s)[:, None, None]

    # Allocate outputs on the half-mesh indexing convention: index js corresponds
    # to the interval (js-1, js) for js>=1, with js=0 copied from js=1.
    shape = pr1_even.shape
    ru12 = jnp.zeros(shape, dtype=pr1_even.dtype)
    zu12 = jnp.zeros(shape, dtype=pr1_even.dtype)
    rs = jnp.zeros(shape, dtype=pr1_even.dtype)
    zs = jnp.zeros(shape, dtype=pr1_even.dtype)
    r12 = jnp.zeros(shape, dtype=pr1_even.dtype)
    tau = jnp.zeros(shape, dtype=pr1_even.dtype)

    # Slices for js>=1.
    sl = slice(1, ns)
    sm1 = slice(0, ns - 1)

    ru12 = ru12.at[sl].set(
        0.5
        * (
            pru_even[sl]
            + pru_even[sm1]
            + pshalf[sl] * (pru_odd[sl] + pru_odd[sm1])
        )
    )
    zs = zs.at[sl].set(
        ohs
        * (
            (pz1_even[sl] - pz1_even[sm1])
            + pshalf[sl] * (pz1_odd[sl] - pz1_odd[sm1])
        )
    )
    tau = tau.at[sl].set(
        ru12[sl] * zs[sl]
        + dshalfds
        * (
            pru_odd[sl] * pz1_odd[sl]
            + pru_odd[sm1] * pz1_odd[sm1]
            + _safe_divide(
                pru_even[sl] * pz1_odd[sl] + pru_even[sm1] * pz1_odd[sm1],
                pshalf[sl],
            )
        )
    )

    zu12 = zu12.at[sl].set(
        0.5
        * (
            pzu_even[sl]
            + pzu_even[sm1]
            + pshalf[sl] * (pzu_odd[sl] + pzu_odd[sm1])
        )
    )
    rs = rs.at[sl].set(
        ohs
        * (
            (pr1_even[sl] - pr1_even[sm1])
            + pshalf[sl] * (pr1_odd[sl] - pr1_odd[sm1])
        )
    )
    r12 = r12.at[sl].set(
        0.5
        * (
            pr1_even[sl]
            + pr1_even[sm1]
            + pshalf[sl] * (pr1_odd[sl] + pr1_odd[sm1])
        )
    )
    tau = tau.at[sl].set(
        tau[sl]
        - rs[sl] * zu12[sl]
        - dshalfds
        * (
            pzu_odd[sl] * pr1_odd[sl]
            + pzu_odd[sm1] * pr1_odd[sm1]
            + _safe_divide(
                pzu_even[sl] * pr1_odd[sl] + pzu_even[sm1] * pr1_odd[sm1],
                pshalf[sl],
            )
        )
    )

    # VMEC copies js=1 to js=0 for tau/r12 in the serial routine.
    ru12 = ru12.at[0].set(ru12[1])
    zu12 = zu12.at[0].set(zu12[1])
    rs = rs.at[0].set(rs[1])
    zs = zs.at[0].set(zs[1])
    r12 = r12.at[0].set(r12[1])
    tau = tau.at[0].set(tau[1])

    sqrtg = r12 * tau
    # Avoid NaNs on axis.
    sqrtg = jnp.where(psqrts == 0, 0.0, sqrtg)
    return VmecHalfMeshJacobian(r12=r12, rs=rs, zs=zs, ru12=ru12, zu12=zu12, tau=tau, sqrtg=sqrtg)


def vmec_half_mesh_jacobian_from_state(
    *,
    state,
    modes,
    trig: VmecTrigTables,
    s,
) -> VmecHalfMeshJacobian:
    """Compute VMEC half-mesh Jacobian directly from Fourier coefficients."""
    m = jnp.asarray(modes.m)
    mask_even = (m % 2) == 0
    mask_odd = jnp.logical_not(mask_even)

    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)

    Rcos_even = jnp.where(mask_even[None, :], Rcos, 0.0)
    Rsin_even = jnp.where(mask_even[None, :], Rsin, 0.0)
    Rcos_odd = jnp.where(mask_odd[None, :], Rcos, 0.0)
    Rsin_odd = jnp.where(mask_odd[None, :], Rsin, 0.0)

    Zcos_even = jnp.where(mask_even[None, :], Zcos, 0.0)
    Zsin_even = jnp.where(mask_even[None, :], Zsin, 0.0)
    Zcos_odd = jnp.where(mask_odd[None, :], Zcos, 0.0)
    Zsin_odd = jnp.where(mask_odd[None, :], Zsin, 0.0)

    pr1_even = vmec_realspace_synthesis(coeff_cos=Rcos_even, coeff_sin=Rsin_even, modes=modes, trig=trig)
    pr1_odd = vmec_realspace_synthesis(coeff_cos=Rcos_odd, coeff_sin=Rsin_odd, modes=modes, trig=trig)

    pz1_even = vmec_realspace_synthesis(coeff_cos=Zcos_even, coeff_sin=Zsin_even, modes=modes, trig=trig)
    pz1_odd = vmec_realspace_synthesis(coeff_cos=Zcos_odd, coeff_sin=Zsin_odd, modes=modes, trig=trig)

    pru_even = vmec_realspace_synthesis_dtheta(coeff_cos=Rcos_even, coeff_sin=Rsin_even, modes=modes, trig=trig)
    pru_odd = vmec_realspace_synthesis_dtheta(coeff_cos=Rcos_odd, coeff_sin=Rsin_odd, modes=modes, trig=trig)

    pzu_even = vmec_realspace_synthesis_dtheta(coeff_cos=Zcos_even, coeff_sin=Zsin_even, modes=modes, trig=trig)
    pzu_odd = vmec_realspace_synthesis_dtheta(coeff_cos=Zcos_odd, coeff_sin=Zsin_odd, modes=modes, trig=trig)

    return jacobian_half_mesh_from_parity(
        pr1_even=pr1_even,
        pr1_odd=pr1_odd,
        pz1_even=pz1_even,
        pz1_odd=pz1_odd,
        pru_even=pru_even,
        pru_odd=pru_odd,
        pzu_even=pzu_even,
        pzu_odd=pzu_odd,
        s=s,
    )
