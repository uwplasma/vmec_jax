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
    return jnp.where(jnp.abs(denom) > eps, num / denom, jnp.zeros_like(num))


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

