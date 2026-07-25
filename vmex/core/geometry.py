"""Flux-surface geometry: real-space R/Z/lambda channels and the half-mesh Jacobian.

VMEC2000 counterparts
---------------------
- ``Sources/General/totzsp_mod.f`` (``totzsps``/``totzspa``) — synthesis of the
  even-m / odd-m real-space geometry channels (``r1, ru, rv, z1, zu, zv, lu,
  lv`` with ``mparity`` planes) including the ``jmin1`` axis rules:
  :func:`real_space_geometry`, :func:`apply_lambda_axis_closure`.
- ``Sources/General/jacobian.f`` — half-mesh quantities ``r12, rs, zs, ru12,
  zu12``, the Jacobian factor ``tau = sqrt(g)/R`` and ``sqrt(g) = r12 * tau``,
  plus the Jacobian sign-change check (``taumax * taumin < 0`` -> ``irst = 2``):
  :func:`half_mesh_jacobian`.

Radial conventions (parity-critical)
------------------------------------
VMEC evolves odd-m Fourier coefficients in an internal ``1/sqrt(s)``
representation (``profil3d.f`` ``scalxc``), so every real-space quantity is
assembled as ``X = X_even + sqrt(s) * X_odd`` where ``X_odd`` is the *internal*
odd-m channel.  Radial derivatives and the Jacobian live on the **half mesh**
(``s_half(js) = (s(js) + s(js-1)) / 2``); half-mesh arrays keep the VMEC
convention of copying the first interior surface into the axis slot
(``X(js=1) = X(js=2)`` in Fortran indexing).

The numerics are ported verbatim from the parity-proven legacy kernels
``vmex.kernels.jacobian`` and the geometry stage of
``vmex.kernels.bcovar`` (``_compute_bcovar_parity_channels`` with
``use_vmec_synthesis=True``); equivalence is enforced in
``tests/test_geometry_fields_ab.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import jax.numpy as jnp

from .fourier import ModeTable, TrigTables
from .transforms import (
    _fourier_to_real_fft,
    fourier_to_real,
    register_pytree_dataclass as _register,
)

__all__ = [
    "RealSpaceGeometry",
    "HalfMeshJacobian",
    "apply_lambda_axis_closure",
    "real_space_geometry",
    "half_mesh_jacobian",
    "sqrt_s_half_mesh",
]

Array = Any

# VMEC ``p25``: d(sqrt(s))/ds * hs factor pair reduces to the constant 1/4 in
# the discrete half-mesh tau formula (jacobian.f, ``dshalfds = p25``).
_D_SQRT_S_HALF_DS = 0.25


def _safe_divide(x: Array, y: Array, *, eps: float = 1e-14) -> Array:
    """Zero-preserving division (legacy ``kernels.jacobian._safe_divide``)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    mask = jnp.abs(y) > eps
    y_safe = jnp.where(mask, y, jnp.ones_like(y))
    return mask.astype(x.dtype) * (x / y_safe)


def sqrt_s_half_mesh(s: Array) -> Array:
    """``sqrt(s)`` on the VMEC radial half mesh (VMEC: ``pshalf/shalf``).

    VMEC2000: ``profil3d.f`` — ``shalf(js) = sqrt(hs * (js - 1.5))``.  The axis
    slot repeats the first interior half-mesh value (VMEC's arrays are simply
    never read there, but legacy kernels fill it this way and downstream
    formulas rely on it).  Shape ``(ns,)`` in, ``(ns,)`` out.
    """
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    s_half = 0.5 * (s[1:] + s[:-1])
    s_half = jnp.concatenate([s_half[:1], s_half], axis=0)
    return jnp.sqrt(jnp.maximum(s_half, 0.0))


def apply_lambda_axis_closure(
    lambda_sin: Array,
    *,
    modes: ModeTable,
    ntor: int,
) -> Array:
    """Apply VMEC's symmetric-lambda axis closure to the sin coefficients.

    VMEC2000: ``totzsp_mod.f`` (``totzsps``) sets ``lmncs(js=1, n>0, m=0) =
    lmncs(js=2, n>0, m=0)`` for three-dimensional runs (``lthreed``).  In the
    signed-(m, n) packing used here this copies the first interior surface of
    the ``(m=0, n>0)`` sin coefficients into the axis row.  A no-op for 2D
    runs (``ntor = 0``) or ``ns < 2``.
    """
    lambda_sin = jnp.asarray(lambda_sin)
    ns = int(lambda_sin.shape[0])
    if int(ntor) <= 0 or ns < 2:
        return lambda_sin
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    copy_mask = (m == 0) & (n > 0)
    if not np.any(copy_mask):
        return lambda_sin
    copy = jnp.asarray(copy_mask)
    axis_row = jnp.where(copy, lambda_sin[1, :], lambda_sin[0, :])
    return jnp.concatenate([axis_row[None, :], lambda_sin[1:]], axis=0)


@dataclass(frozen=True)
class RealSpaceGeometry:
    """Even-m / odd-m real-space geometry channels on the internal grid.

    VMEC2000 names (``totzsp_mod.f`` outputs; ``_even``/``_odd`` are the
    ``mparity = 0/1`` planes, odd carrying the internal ``1/sqrt(s)``
    representation): ``R_even/R_odd = r1``, ``dR_dtheta_* = ru``,
    ``dR_dzeta_* = rv``, ``Z_* = z1``, ``dZ_dtheta_* = zu``, ``dZ_dzeta_* =
    zv``, ``dlambda_dtheta_* = lu`` and ``dlambda_dzeta_*`` (note VMEC stores
    ``lv = -d(lambda)/dzeta``; here the *plain* derivative is stored and the
    sign lives in :func:`vmex.core.fields.contravariant_b`).

    The odd planes carry the ``jmin1`` axis rules of ``totzsps``: on the axis
    row only the ``m = 1`` content survives, extrapolated from the first
    interior surface (``jmin1(m=1) = 1``, ``jmin1(m>=2) = 2``).

    All arrays have shape ``(ns, ntheta3, nzeta)``.
    """

    R_even: Array
    R_odd: Array
    Z_even: Array
    Z_odd: Array
    dR_dtheta_even: Array
    dR_dtheta_odd: Array
    dZ_dtheta_even: Array
    dZ_dtheta_odd: Array
    dR_dzeta_even: Array
    dR_dzeta_odd: Array
    dZ_dzeta_even: Array
    dZ_dzeta_odd: Array
    dlambda_dtheta_even: Array
    dlambda_dtheta_odd: Array
    dlambda_dzeta_even: Array
    dlambda_dzeta_odd: Array

    def theta_derivatives_full(self, s: Array) -> tuple[Array, Array]:
        """Return full-mesh ``(ru0, zu0) = X_even + sqrt(s) * X_odd``.

        VMEC2000: ``ru0/zu0`` formed in ``funct3d.f`` and consumed by the
        constraint scaling (``bcovar.f`` ``arnorm/aznorm``) and the constraint
        force (``alias.f`` ``ztemp``).
        """
        sqrt_s = jnp.sqrt(jnp.maximum(jnp.asarray(s), 0.0))[:, None, None]
        ru0 = self.dR_dtheta_even + sqrt_s * self.dR_dtheta_odd
        zu0 = self.dZ_dtheta_even + sqrt_s * self.dZ_dtheta_odd
        return ru0, zu0


@dataclass(frozen=True)
class HalfMeshJacobian:
    """Half-mesh Jacobian quantities (VMEC2000: ``Sources/General/jacobian.f``).

    Attributes (VMEC names in parentheses; all ``(ns, ntheta3, nzeta)`` except
    the flag):

    - ``r12`` (``r12``): R interpolated to the half mesh.
    - ``dR_ds`` (``rs``), ``dZ_ds`` (``zs``): radial finite differences on the
      half mesh, including the odd-m ``sqrt(s)`` chain-rule terms.
    - ``ru12`` (``ru12``), ``zu12`` (``zu12``): poloidal derivatives averaged
      to the half mesh.
    - ``tau`` (``tau``): ``sqrt(g)/R = ru12*zs - rs*zu12 + (odd-m d(sqrt s)/ds
      corrections)`` — eq. (10) of Hirshman & Whitson with the discrete
      half-mesh corrections of ``jacobian.f``.
    - ``sqrt_g`` (``gsqrt``): the coordinate Jacobian ``r12 * tau``.
    - ``jacobian_sign_changed``: scalar bool, True when ``tau`` changes sign
      over the interior surfaces (``taumax*taumin < 0`` -> VMEC
      ``irst = 2`` / ``bad_jacobian_flag``).  Computed without host branching.
    """

    r12: Array
    dR_ds: Array
    dZ_ds: Array
    ru12: Array
    zu12: Array
    tau: Array
    sqrt_g: Array
    jacobian_sign_changed: Array


for _cls in (RealSpaceGeometry, HalfMeshJacobian):
    _register(_cls)


def real_space_geometry(
    *,
    R_cos: Array,
    R_sin: Array,
    Z_cos: Array,
    Z_sin: Array,
    lambda_cos: Array,
    lambda_sin: Array,
    modes: ModeTable,
    trig: TrigTables,
    s: Array,
    use_fft: bool = False,
) -> RealSpaceGeometry:
    """Synthesize the VMEC even-m/odd-m geometry channels from coefficients.

    VMEC2000: ``totzsp_mod.f`` — ``totzsps`` (+ ``totzspa`` content in the
    signed-(m, n) packing) producing ``r1, ru, rv, z1, zu, zv, lu, lv`` split
    by poloidal-mode parity, with the ``profil3d.f`` ``scalxc`` odd-m
    ``1/sqrt(s)`` scaling and the ``jmin1`` axis rules applied.

    Parameters
    ----------
    R_cos, R_sin, Z_cos, Z_sin, lambda_cos, lambda_sin:
        Spectral coefficients, shape ``(ns, mnmax)``, in VMEC *internal*
        normalization (divided by ``mscale*nscale``) and in the *physical*
        basis (the ``residue.f90`` m=1 constraint already undone; see
        ``vmex.kernels.parity.vmec_m1_internal_to_physical_signed``).
        ``lambda_sin`` should already carry the 3D axis closure
        (:func:`apply_lambda_axis_closure`).
    s:
        Full-mesh radial grid, shape ``(ns,)`` (uniform, ``s in [0, 1]``).
    use_fft:
        Use separable toroidal FFT synthesis; False retains the real dense
        contraction used by the high-column implicit Jacobian.

    Returns
    -------
    :class:`RealSpaceGeometry` with all channels on the internal
    ``(ns, ntheta3, nzeta)`` grid.
    """
    coeff_cos = jnp.stack(
        [jnp.asarray(R_cos), jnp.asarray(Z_cos), jnp.asarray(lambda_cos)], axis=0
    )
    coeff_sin = jnp.stack(
        [jnp.asarray(R_sin), jnp.asarray(Z_sin), jnp.asarray(lambda_sin)], axis=0
    )

    m = np.asarray(modes.m, dtype=int)
    dtype = coeff_cos.dtype
    # Poloidal-parity masks.  The odd plane is split into m=1 and m>=3 so the
    # jmin1 axis rules can be applied per subset (jmin1(1)=1, jmin1(m>=2)=2).
    mask_even = jnp.asarray((m % 2) == 0, dtype=dtype)
    mask_m1 = jnp.asarray(m == 1, dtype=dtype)
    mask_odd_rest = jnp.asarray((m % 2 == 1) & (m != 1), dtype=dtype)
    mask_stack = jnp.stack([mask_even, mask_m1, mask_odd_rest], axis=0)

    # (mask, field, ns, mnmax) batch; scalxc is exactly 1 for even m, so a
    # single odd_m_sqrt_s=True synthesis serves all three parity subsets.
    batched_cos = coeff_cos[None, ...] * mask_stack[:, None, None, :]
    batched_sin = coeff_sin[None, ...] * mask_stack[:, None, None, :]
    synthesize = _fourier_to_real_fft if use_fft else fourier_to_real
    value, dtheta, dzeta = synthesize(
        batched_cos,
        batched_sin,
        modes=modes,
        trig=trig,
        derivatives=("value", "dtheta", "dzeta"),
        internal_coeffs=True,
        odd_m_sqrt_s=True,
        s=s,
    )

    def odd_internal(plane: Array, field: int) -> Array:
        """Combine m=1 and m>=3 odd channels with the jmin1 axis rules.

        VMEC2000: ``totzsp_mod.f`` — the axis row keeps only the m=1 content,
        origin-extrapolated from the first interior surface (``jmin1(1)=1``);
        odd modes with m >= 3 vanish on axis (``jmin1(m>=2)=2``).
        """
        m1 = plane[1, field]
        combined = m1 + plane[2, field]
        if combined.shape[0] < 2:
            return combined
        return jnp.concatenate([m1[1][None, ...], combined[1:]], axis=0)

    R_FIELD, Z_FIELD, L_FIELD = 0, 1, 2
    return RealSpaceGeometry(
        R_even=value[0, R_FIELD],
        R_odd=odd_internal(value, R_FIELD),
        Z_even=value[0, Z_FIELD],
        Z_odd=odd_internal(value, Z_FIELD),
        dR_dtheta_even=dtheta[0, R_FIELD],
        dR_dtheta_odd=odd_internal(dtheta, R_FIELD),
        dZ_dtheta_even=dtheta[0, Z_FIELD],
        dZ_dtheta_odd=odd_internal(dtheta, Z_FIELD),
        dR_dzeta_even=dzeta[0, R_FIELD],
        dR_dzeta_odd=odd_internal(dzeta, R_FIELD),
        dZ_dzeta_even=dzeta[0, Z_FIELD],
        dZ_dzeta_odd=odd_internal(dzeta, Z_FIELD),
        dlambda_dtheta_even=dtheta[0, L_FIELD],
        dlambda_dtheta_odd=odd_internal(dtheta, L_FIELD),
        dlambda_dzeta_even=dzeta[0, L_FIELD],
        dlambda_dzeta_odd=odd_internal(dzeta, L_FIELD),
    )


def half_mesh_jacobian(geometry: RealSpaceGeometry, *, s: Array) -> HalfMeshJacobian:
    """Half-mesh Jacobian from the even/odd geometry channels.

    VMEC2000: ``Sources/General/jacobian.f`` — with ``X = X_even + sqrt(s) *
    X_odd`` on the full mesh, builds (half-mesh index ``js``; Fortran
    ``js = 2..ns``):

    - ``r12  = 0.5*(R_e(js) + R_e(js-1) + shalf(js)*(R_o(js) + R_o(js-1)))``
    - ``rs   = ohs*(R_e(js) - R_e(js-1) + shalf(js)*(R_o(js) - R_o(js-1)))``
      (``zs`` analogous, ``ru12/zu12`` are the poloidal-derivative averages)
    - ``tau  = ru12*zs - rs*zu12 + 0.25*[ (ru_o*z1_o + ...) +
      (ru_e*z1_o + ...)/shalf ]`` — the ``d(sqrt s)/ds`` correction terms
    - ``sqrt_g = r12 * tau``  (VMEC: ``gsqrt``), zeroed on the axis row.

    The axis row of each half-mesh array copies the first interior surface
    (VMEC serial convention).  ``jacobian_sign_changed`` reproduces the
    ``jacobian.f`` check ``taumax*taumin < 0`` (over ``js = 2..ns``) that sets
    ``irst = 2`` (``bad_jacobian_flag``); it is returned as a traced boolean —
    reacting to it is the solver's job (no host branching here).
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    R_e, R_o = geometry.R_even, geometry.R_odd
    Z_e, Z_o = geometry.Z_even, geometry.Z_odd
    Ru_e, Ru_o = geometry.dR_dtheta_even, geometry.dR_dtheta_odd
    Zu_e, Zu_o = geometry.dZ_dtheta_even, geometry.dZ_dtheta_odd

    if ns < 2:
        zero = jnp.zeros_like(R_e)
        return HalfMeshJacobian(
            r12=R_e,
            dR_ds=zero,
            dZ_ds=zero,
            ru12=zero,
            zu12=zero,
            tau=zero,
            sqrt_g=zero,
            jacobian_sign_changed=jnp.asarray(False),
        )

    hs = s[1] - s[0]
    ohs = _safe_divide(1.0, hs)
    sqrt_s = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    sqrt_s_half = sqrt_s_half_mesh(s)[:, None, None]

    inner = slice(1, ns)
    prev = slice(0, ns - 1)
    sh = sqrt_s_half[inner]

    ru12_inner = 0.5 * (Ru_e[inner] + Ru_e[prev] + sh * (Ru_o[inner] + Ru_o[prev]))
    zu12_inner = 0.5 * (Zu_e[inner] + Zu_e[prev] + sh * (Zu_o[inner] + Zu_o[prev]))
    rs_inner = ohs * ((R_e[inner] - R_e[prev]) + sh * (R_o[inner] - R_o[prev]))
    zs_inner = ohs * ((Z_e[inner] - Z_e[prev]) + sh * (Z_o[inner] - Z_o[prev]))
    r12_inner = 0.5 * (R_e[inner] + R_e[prev] + sh * (R_o[inner] + R_o[prev]))

    # tau = ru12*zs - rs*zu12 + dshalfds * (odd-m correction terms)
    tau_inner = ru12_inner * zs_inner + _D_SQRT_S_HALF_DS * (
        Ru_o[inner] * Z_o[inner]
        + Ru_o[prev] * Z_o[prev]
        + _safe_divide(Ru_e[inner] * Z_o[inner] + Ru_e[prev] * Z_o[prev], sh)
    )
    tau_inner = tau_inner - rs_inner * zu12_inner - _D_SQRT_S_HALF_DS * (
        Zu_o[inner] * R_o[inner]
        + Zu_o[prev] * R_o[prev]
        + _safe_divide(Zu_e[inner] * R_o[inner] + Zu_e[prev] * R_o[prev], sh)
    )

    def with_axis_copy(body: Array) -> Array:
        return jnp.concatenate([body[:1], body], axis=0)

    r12 = with_axis_copy(r12_inner)
    tau = with_axis_copy(tau_inner)
    sqrt_g = r12 * tau
    # The axis row (s = 0) carries no volume; VMEC leaves gsqrt(js=1) unused.
    sqrt_g = jnp.where(sqrt_s == 0, 0.0, sqrt_g)

    # jacobian.f: taumax = MAXVAL(tau(2:)), taumin = MINVAL(tau(2:));
    # a sign change flags a self-intersecting coordinate mapping (irst = 2).
    jacobian_sign_changed = (jnp.max(tau_inner) * jnp.min(tau_inner)) < 0.0

    return HalfMeshJacobian(
        r12=r12,
        dR_ds=with_axis_copy(rs_inner),
        dZ_ds=with_axis_copy(zs_inner),
        ru12=with_axis_copy(ru12_inner),
        zu12=with_axis_copy(zu12_inner),
        tau=tau,
        sqrt_g=sqrt_g,
        jacobian_sign_changed=jacobian_sign_changed,
    )
