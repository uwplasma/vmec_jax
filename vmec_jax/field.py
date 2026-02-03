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

from ._compat import jax, jnp


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


def chips_from_chipf(chipf):
    """Reconstruct VMEC's `chips(js)` (full-mesh poloidal flux function) from `chipf(js)`.

    VMEC outputs `chipf` in `wout_*.nc`, but the core `bcovar/add_fluxes` pipeline
    adds the **full-mesh** flux function `chip(js)=chips(js)` to `B^u` in real space.

    Internally (see `VMEC2000/Sources/General/add_fluxes.f90`, `lrfp=F` case),
    VMEC forms `chipf` from `chips` via a simple radial averaging scheme:

      chipf(1)     = 1.5*chips(2) - 0.5*chips(3)
      chipf(2:ns-1)= 0.5*(chips(2:ns-1) + chips(3:ns))
      chipf(ns)    = 1.5*chips(ns) - 0.5*chips(ns-1)

    This helper inverts that mapping deterministically (for ns>=3) using the
    forward recurrence implied by the interior relation and the axis closure:

      chips(2) = 0.5*(chipf(1) + chipf(2))
      chips(js+1) = 2*chipf(js) - chips(js)   for js=2..ns-1

    Notes
    -----
    - The returned `chips` has the same shape as `chipf` (ns,). `chips(1)` is
      set to 0 (VMEC does not use it near-axis).
    - This is intended for output-parity work using `wout` files; it does not
      attempt to handle the `lrfp=True` harmonic-mean variant.
    """
    chipf = jnp.asarray(chipf)
    if chipf.ndim != 1:
        raise ValueError(f"chipf must be 1D (ns,), got shape {chipf.shape}")
    ns = int(chipf.shape[0])
    if ns == 0:
        return chipf
    if ns == 1:
        return jnp.zeros_like(chipf)
    if ns == 2:
        chips = jnp.zeros_like(chipf)
        # With only two surfaces, VMEC's special-case handling effectively pins
        # chips(2) from the available value(s); use chipf(2) when present.
        chips = chips.at[1].set(chipf[1])
        return chips

    # ns >= 3
    if jax is None:  # pragma: no cover
        raise ImportError("chips_from_chipf requires JAX (jax + jaxlib)")

    chips = jnp.zeros_like(chipf)
    chips2 = 0.5 * (chipf[0] + chipf[1])
    chips = chips.at[1].set(chips2)

    def _body(js0, chips_acc):
        # Forward recurrence (1-based): chips(js+1) = 2*chipf(js) - chips(js) for js=2..ns-1.
        return chips_acc.at[js0 + 1].set(2.0 * chipf[js0] - chips_acc[js0])

    # 0-based js0=1..ns-2 corresponds to Fortran js=2..ns-1.
    chips = jax.lax.fori_loop(1, ns - 1, _body, chips)
    return chips


def bsup_from_geom(geom, *, phipf, chipf, nfp: int, signgs: int, lamscale, eps: float = 1e-14):
    """Compute (bsupu, bsupv) from a :class:`~vmec_jax.geom.Geom`.

    Notes
    -----
    The geometry kernel provides lambda derivatives w.r.t the physical toroidal
    angle ``phi_phys``.

    VMEC's internal trig derivative tables already include the NFP factor (see
    ``fixaray.f``: ``cosnvn/sinnvn`` scale by ``n*nfp``), so the stored
    derivatives correspond to ``d/dphi_phys``.
    """
    lam_u = geom.L_theta
    lam_v = geom.L_phi
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
