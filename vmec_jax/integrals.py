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


def volume_from_sqrtg(sqrtg, s, theta, zeta, nfp: int):
    """Convenience wrapper: compute (dvds, V) from sqrtg."""
    dvds = dvds_from_sqrtg(sqrtg, theta, zeta, nfp)
    V = cumtrapz_s(dvds, s)
    return dvds, V

