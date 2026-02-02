"""Magnetic field representation utilities (step-4).

This module implements the contravariant magnetic-field components used by VMEC:

  - ``bsupu`` : B^u   (poloidal-angle contravariant component)
  - ``bsupv`` : B^v   (toroidal-angle contravariant component)

in terms of:
  - the Jacobian ``sqrtg`` (signed),
  - flux functions ``phipf(s) = dPhi/ds`` and ``chipf(s) = dChi/ds``,
  - and the VMEC ``lambda`` field (stored in a *scaled* form; see ``lamscale``).

Important convention
--------------------
VMEC's public `wout` files store the lambda coefficients in a scaled form, and VMEC
multiplies lambda-derivatives by a scalar ``lamscale`` before using them in B.
We follow that convention here so we can validate against the bundled `wout` reference.

The formulas implemented here match VMEC's `bcovar` + `add_fluxes` logic:

  bsupv = overg * (phipf + lamscale * lam_u)
  bsupu = overg * (chipf - lamscale * lam_v)

where ``overg = 1 / (signgs * sqrtg * 2π)`` for the coordinate conventions used in
this repo (see `vmec_jax.integrals` for the zeta/phi relationship).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._compat import jnp


TWOPI = 2.0 * np.pi


def signgs_from_sqrtg(sqrtg, *, axis_index: int = 1) -> int:
    """Infer VMEC's `signgs` (+1 or -1) from a signed Jacobian array.

    We intentionally compute this outside of any jitted objective; `signgs`
    should be treated as a fixed convention, not something to differentiate through.
    """
    a = np.asarray(sqrtg)
    if a.ndim < 1 or a.shape[0] <= axis_index:
        # Fall back to a safe default.
        return 1
    m = float(np.mean(a[axis_index:]))
    return 1 if m >= 0 else -1


def lamscale_from_phips(phips, s):
    """Compute VMEC's `lamscale` from `phips` and the radial grid `s`.

    VMEC computes (see `profil1d.f`):
        lamscale = sqrt(hs * sum_{js=2..ns} phips(js)^2)
    where hs is the uniform spacing in s.
    """
    phips = jnp.asarray(phips)
    s = jnp.asarray(s)
    if phips.ndim != 1 or s.ndim != 1:
        raise ValueError(f"phips and s must be 1D, got shapes phips={phips.shape}, s={s.shape}")
    if phips.shape[0] != s.shape[0]:
        raise ValueError(f"phips and s must have same length, got phips={phips.shape[0]} s={s.shape[0]}")
    if phips.shape[0] < 2:
        return jnp.asarray(1.0, dtype=phips.dtype)
    hs = s[1] - s[0]
    return jnp.sqrt(hs * jnp.sum(phips[1:] ** 2))


def _safe_divide(num, denom, *, eps: float = 1e-14):
    denom = jnp.asarray(denom)
    num = jnp.asarray(num)
    mask = jnp.abs(denom) > eps
    # Avoid 0/0 and NaNs in gradients near the magnetic axis by never dividing by
    # a small denominator. (Using `where(num/denom, 0)` can still produce NaN
    # gradients due to 0*NaN propagation in downstream reductions.)
    denom_safe = jnp.where(mask, denom, jnp.ones_like(denom))
    return mask.astype(num.dtype) * (num / denom_safe)


def bsup_from_sqrtg_lambda(
    *,
    sqrtg,
    lam_u,
    lam_v,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    eps: float = 1e-14,
):
    """Compute (bsupu, bsupv) from a Jacobian and lambda derivatives.

    Parameters
    ----------
    sqrtg:
        Signed Jacobian on the 3D grid (ns, ntheta, nzeta).
    lam_u:
        d(lambda_scaled)/d(theta), same shape as sqrtg.
    lam_v:
        d(lambda_scaled)/d(zeta), same shape as sqrtg.
    phipf, chipf:
        1D arrays (ns,) for toroidal/poloidal flux derivatives.
    signgs:
        +1 or -1, fixed for the run.
    lamscale:
        Scalar lambda scaling factor (VMEC convention).
    """
    sqrtg = jnp.asarray(sqrtg)
    lam_u = jnp.asarray(lam_u)
    lam_v = jnp.asarray(lam_v)
    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    signgs = int(signgs)
    lamscale = jnp.asarray(lamscale)

    if sqrtg.ndim != 3:
        raise ValueError(f"sqrtg must be (ns,ntheta,nzeta), got shape {sqrtg.shape}")

    denom = (signgs * sqrtg * TWOPI)
    num_u = chipf[:, None, None] - lamscale * lam_v
    num_v = phipf[:, None, None] + lamscale * lam_u
    bsupu = _safe_divide(num_u, denom, eps=eps)
    bsupv = _safe_divide(num_v, denom, eps=eps)
    return bsupu, bsupv


def bsup_from_geom(geom, *, phipf, chipf, nfp: int, signgs: int, lamscale, eps: float = 1e-14):
    """Compute (bsupu, bsupv) from a :class:`~vmec_jax.geom.Geom`.

    Notes
    -----
    The geometry kernel provides lambda derivatives w.r.t the physical toroidal
    angle ``phi_phys`` (full-torus angle). VMEC's bsupu formula uses ``lam_v``
    with respect to the internal field-period coordinate ``zeta``, so:

        lam_v = d(lambda)/dzeta = (1/NFP) * d(lambda)/dphi_phys
    """
    lam_u = geom.L_theta
    lam_v = geom.L_phi / int(nfp)
    return bsup_from_sqrtg_lambda(
        sqrtg=geom.sqrtg,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=phipf,
        chipf=chipf,
        signgs=signgs,
        lamscale=lamscale,
        eps=eps,
    )


def b2_from_bsup(geom, bsupu, bsupv):
    """Compute B^2 from contravariant components and the covariant metric."""
    bsupu = jnp.asarray(bsupu)
    bsupv = jnp.asarray(bsupv)
    return geom.g_tt * bsupu**2 + 2.0 * geom.g_tp * bsupu * bsupv + geom.g_pp * bsupv**2


def bsub_from_bsup(geom, bsupu, bsupv):
    """Compute covariant components (B_u, B_v) from contravariant (B^u, B^v).

    Notes
    -----
    With ``B^s=0`` in VMEC's representation, the angular covariant components are:

        B_u = g_uu B^u + g_uv B^v
        B_v = g_uv B^u + g_vv B^v

    In this codebase, u corresponds to ``theta`` and v corresponds to the physical
    toroidal angle ``phi_phys`` used in :mod:`vmec_jax.geom`.
    """
    bsupu = jnp.asarray(bsupu)
    bsupv = jnp.asarray(bsupv)
    bsubu = geom.g_tt * bsupu + geom.g_tp * bsupv
    bsubv = geom.g_tp * bsupu + geom.g_pp * bsupv
    return bsubu, bsubv


def b_cartesian_from_bsup(geom, bsupu, bsupv, *, zeta, nfp: int):
    """Compute Cartesian B=(Bx,By,Bz) from contravariant components.

    Parameters
    ----------
    geom:
        Geometry object from :func:`vmec_jax.geom.eval_geom`.
    bsupu, bsupv:
        Contravariant components returned by :func:`bsup_from_geom` (same shape as geom.sqrtg).
    zeta:
        1D toroidal grid used in geom (shape (nzeta,)).
    nfp:
        Number of field periods. Used to convert ``zeta`` to the physical toroidal angle
        ``phi_phys = zeta / nfp``.

    Returns
    -------
    B:
        Array of shape ``(ns, ntheta, nzeta, 3)`` containing (Bx,By,Bz).

    Notes
    -----
    In curvilinear coordinates, a vector can be written using covariant basis vectors:

        B = B^theta * e_theta + B^phi * e_phi

    where ``e_theta = ∂r/∂theta`` and ``e_phi = ∂r/∂phi_phys`` in Cartesian space.
    """
    bsupu = jnp.asarray(bsupu)
    bsupv = jnp.asarray(bsupv)
    zeta = jnp.asarray(zeta)
    nfp = int(nfp)
    if nfp <= 0:
        raise ValueError(f"nfp must be positive, got {nfp}")

    phi = zeta / nfp
    cosphi = jnp.cos(phi)[None, None, :]
    sinphi = jnp.sin(phi)[None, None, :]

    # Covariant basis vectors in Cartesian coordinates (same construction as geom.py).
    e_t = jnp.stack([geom.Rt * cosphi, geom.Rt * sinphi, geom.Zt], axis=-1)
    e_p = jnp.stack(
        [geom.Rp * cosphi - geom.R * sinphi, geom.Rp * sinphi + geom.R * cosphi, geom.Zp],
        axis=-1,
    )

    return bsupu[..., None] * e_t + bsupv[..., None] * e_p
