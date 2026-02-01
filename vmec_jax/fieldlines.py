"""Field-line tracing utilities.

This module provides a minimal field-line tracer on a single flux surface.

The tracer is intended for visualization (ParaView) and regression diagnostics,
not for production orbit following.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class FieldLine:
    """A traced field line on a flux surface."""

    phi: np.ndarray  # (N,)
    theta: np.ndarray  # (N,)
    x: np.ndarray  # (N,)
    y: np.ndarray  # (N,)
    z: np.ndarray  # (N,)
    Bmag: np.ndarray  # (N,)


def _wrap_angle(a: float) -> float:
    return float(np.mod(a, TWOPI))


def _bilinear_periodic(f: np.ndarray, u: float, v: float) -> float:
    """Bilinear interpolation on a periodic uniform grid for f[ntheta, nzeta]."""
    f = np.asarray(f)
    if f.ndim != 2:
        raise ValueError(f"f must be 2D, got shape {f.shape}")
    ntheta, nzeta = f.shape
    if ntheta <= 0 or nzeta <= 0:
        raise ValueError("empty grid")

    u = _wrap_angle(u)
    v = _wrap_angle(v)
    tu = u / TWOPI * ntheta
    tv = v / TWOPI * nzeta

    i0 = int(np.floor(tu)) % ntheta
    j0 = int(np.floor(tv)) % nzeta
    a = float(tu - np.floor(tu))
    b = float(tv - np.floor(tv))
    i1 = (i0 + 1) % ntheta
    j1 = (j0 + 1) % nzeta

    f00 = float(f[i0, j0])
    f10 = float(f[i1, j0])
    f01 = float(f[i0, j1])
    f11 = float(f[i1, j1])
    return (1 - a) * (1 - b) * f00 + a * (1 - b) * f10 + (1 - a) * b * f01 + a * b * f11


def trace_fieldline_on_surface(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    Bmag: np.ndarray,
    nfp: int,
    theta0: float,
    phi0: float,
    n_steps: int,
    dphi: float,
) -> FieldLine:
    """Trace a field line on a single flux surface using RK4 in physical phi.

    Parameters
    ----------
    R, Z:
        Surface geometry arrays of shape ``(ntheta, nzeta)`` evaluated on a
        uniform grid in ``(theta, zeta)`` over one field period.
    bsupu, bsupv:
        Contravariant components on the same grid, representing
        ``B^theta`` and ``B^phi`` (physical-toroidal contravariant component).
    Bmag:
        Magnitude |B| on the same grid (used only for output point data).
    nfp:
        Number of field periods (zeta = nfp * phi_phys).
    """
    nfp = int(nfp)
    if nfp <= 0:
        raise ValueError(f"nfp must be positive, got {nfp}")
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    dphi = float(dphi)
    if dphi == 0.0:
        raise ValueError("dphi must be nonzero")

    R = np.asarray(R)
    Z = np.asarray(Z)
    bsupu = np.asarray(bsupu)
    bsupv = np.asarray(bsupv)
    Bmag = np.asarray(Bmag)
    if R.shape != Z.shape or R.shape != bsupu.shape or R.shape != bsupv.shape or R.shape != Bmag.shape:
        raise ValueError("R,Z,bsupu,bsupv,Bmag must all have the same shape (ntheta,nzeta)")

    def rhs(theta: float, phi: float) -> float:
        zeta = nfp * phi
        bth = _bilinear_periodic(bsupu, theta, zeta)
        bph = _bilinear_periodic(bsupv, theta, zeta)
        if bph == 0.0:
            return 0.0
        return bth / bph

    phi = np.zeros((n_steps,), dtype=float)
    theta = np.zeros((n_steps,), dtype=float)
    phi[0] = float(phi0)
    theta[0] = float(theta0)

    for i in range(n_steps - 1):
        p = phi[i]
        t = theta[i]
        k1 = rhs(t, p)
        k2 = rhs(t + 0.5 * dphi * k1, p + 0.5 * dphi)
        k3 = rhs(t + 0.5 * dphi * k2, p + 0.5 * dphi)
        k4 = rhs(t + dphi * k3, p + dphi)
        theta[i + 1] = t + (dphi / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi[i + 1] = p + dphi

    x = np.zeros_like(phi)
    y = np.zeros_like(phi)
    z = np.zeros_like(phi)
    bmag = np.zeros_like(phi)

    for i in range(n_steps):
        t = theta[i]
        p = phi[i]
        zeta = nfp * p
        Rtp = _bilinear_periodic(R, t, zeta)
        Ztp = _bilinear_periodic(Z, t, zeta)
        ph = _wrap_angle(p)
        x[i] = Rtp * np.cos(ph)
        y[i] = Rtp * np.sin(ph)
        z[i] = Ztp
        bmag[i] = _bilinear_periodic(Bmag, t, zeta)

    return FieldLine(phi=phi, theta=theta, x=x, y=y, z=z, Bmag=bmag)
