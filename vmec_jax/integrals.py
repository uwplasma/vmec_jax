"""Angle and radial integrals (step-3).

This module computes simple derived quantities from geometric outputs, notably
the volume profile from the Jacobian ``sqrtg``.

Conventions
-----------
The geometry kernel in :mod:`vmec_jax.geom` computes ``sqrtg`` as the Jacobian
for coordinates (s, theta, phi_phys), where ``phi_phys`` is the *physical*
toroidal angle.

Most of vmec_jax works on one field period using the VMEC internal coordinate
``zeta`` in [0, 2π), related by:

    phi_phys = zeta / NFP

Therefore, when integrating over the stored ``zeta`` grid, include the factor
``dphi = dzeta / NFP``.

VMEC `wout` convention note
---------------------------
VMEC stores several 3D fields in `wout_*.nc` on the **radial half mesh** and
with angles (theta, zeta) over one field period. In VMEC's internal
bookkeeping, many 1D radial integrals are computed using a simple *rectangle
rule*:

    V = Σ_{js=2..ns} (dV/ds)(js) * hs

This differs from a trapezoidal rule and matters at low radial resolution.
For regression tests against `wout`, use the helpers in this module that
implement the same conventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._compat import jnp


def dvds_from_sqrtg(sqrtg, theta, zeta, nfp: int):
    """Compute dV/ds from ``sqrtg`` by integrating over angles.

    Parameters
    ----------
    sqrtg:
        Jacobian array of shape (ns, ntheta, nzeta).
    theta, zeta:
        1D angle grids used to compute sqrtg. ``zeta`` spans one field period.
    nfp:
        Number of field periods.

    Returns
    -------
    dvds:
        1D array (ns,) of volume derivative **per field period**.
    """
    sqrtg = jnp.asarray(sqrtg)
    theta = jnp.asarray(theta)
    zeta = jnp.asarray(zeta)
    nfp = int(nfp)
    if nfp <= 0:
        raise ValueError(f"nfp must be positive, got {nfp}")

    ntheta = int(theta.size)
    nzeta = int(zeta.size)
    if ntheta <= 0 or nzeta <= 0:
        raise ValueError("theta and zeta must be non-empty")

    dtheta = 2.0 * np.pi / ntheta
    dzeta = 2.0 * np.pi / nzeta
    dphi = dzeta / nfp

    # Use abs(sqrtg) so volume is positive regardless of coordinate handedness.
    return jnp.sum(jnp.abs(sqrtg), axis=(1, 2)) * dtheta * dphi


def dvds_from_sqrtg_zeta(sqrtg, theta, zeta, *, signgs: int = 1):
    """Compute dV/ds by integrating over (theta, zeta).

    This is useful for matching VMEC's `wout` convention where many quantities
    are stored on a single field period in ``zeta`` and the sign is tracked
    separately via ``signgs``.

    Parameters
    ----------
    sqrtg:
        Jacobian array of shape (ns, ntheta, nzeta).
    theta, zeta:
        1D angle grids spanning [0, 2π) (one field period in ``zeta``).
    signgs:
        +1 or -1 such that ``signgs * sqrtg`` is the signed Jacobian.

    Returns
    -------
    dvds:
        1D array (ns,) of dV/ds for the full torus (i.e. already integrated
        over one field period in ``zeta``; no extra NFP factor is needed).
    """
    sqrtg = jnp.asarray(sqrtg)
    theta = jnp.asarray(theta)
    zeta = jnp.asarray(zeta)
    signgs = int(signgs)

    ntheta = int(theta.size)
    nzeta = int(zeta.size)
    if ntheta <= 0 or nzeta <= 0:
        raise ValueError("theta and zeta must be non-empty")

    dtheta = 2.0 * np.pi / ntheta
    dzeta = 2.0 * np.pi / nzeta
    return jnp.sum(signgs * sqrtg, axis=(1, 2)) * dtheta * dzeta


def cumtrapz_s(y, s):
    """Cumulative trapezoidal integral in s with V(0)=0.

    Parameters
    ----------
    y:
        1D array (ns,) to integrate.
    s:
        1D array (ns,) of coordinates.

    Returns
    -------
    Y:
        1D array (ns,) where ``Y[i] = ∫_0^{s[i]} y(s') ds'`` (trapezoidal).
    """
    y = jnp.asarray(y)
    s = jnp.asarray(s)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError(f"y and s must be 1D, got shapes y={y.shape}, s={s.shape}")
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"y and s must have same length, got y={y.shape[0]} s={s.shape[0]}")
    if y.shape[0] == 0:
        return y

    if y.shape[0] == 1:
        return jnp.zeros_like(y)

    ds = s[1:] - s[:-1]
    seg = 0.5 * (y[1:] + y[:-1]) * ds
    out0 = jnp.zeros((1,), dtype=y.dtype)
    return jnp.concatenate([out0, jnp.cumsum(seg)], axis=0)


def cumrect_s_halfmesh(y, s):
    """Cumulative rectangle-rule integral for half-mesh data.

    Interprets ``y[j]`` as living on the interval (s[j-1], s[j]):

        Y[i] = Σ_{j=1..i} y[j] * (s[j] - s[j-1]),  with Y[0] = 0
    """
    y = jnp.asarray(y)
    s = jnp.asarray(s)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError(f"y and s must be 1D, got shapes y={y.shape}, s={s.shape}")
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"y and s must have same length, got y={y.shape[0]} s={s.shape[0]}")
    if y.shape[0] == 0:
        return y
    if y.shape[0] == 1:
        return jnp.zeros_like(y)

    ds = s[1:] - s[:-1]
    seg = y[1:] * ds
    out0 = jnp.zeros((1,), dtype=y.dtype)
    return jnp.concatenate([out0, jnp.cumsum(seg)], axis=0)


def volume_from_sqrtg(sqrtg, s, theta, zeta, nfp: int):
    """Convenience wrapper: compute (dvds, V) from sqrtg."""
    dvds = dvds_from_sqrtg(sqrtg, theta, zeta, nfp)
    V = cumtrapz_s(dvds, s)
    return dvds, V


def volume_from_sqrtg_vmec(sqrtg, s, theta, zeta, *, signgs: int = 1):
    """Compute (dvds, V) using VMEC-like (wout) conventions."""
    dvds = dvds_from_sqrtg_zeta(sqrtg, theta, zeta, signgs=signgs)
    V = cumrect_s_halfmesh(dvds, s)
    return dvds, V
