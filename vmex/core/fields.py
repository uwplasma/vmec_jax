"""Magnetic fields, metric elements, energies, force norms and constraint scaling.

VMEC2000 counterparts
---------------------
All functions in this module port pieces of ``Sources/General/bcovar.f`` (and
the helpers it calls):

- :func:`metric_elements`        — half-mesh ``guu/guv/gvv`` (bcovar.f metric block).
- :func:`lambda_scale`           — ``lamscale`` (profil1d.f).
- :func:`magnetic_fields`        — contravariant ``B^u/B^v`` (bcovar.f +
  add_fluxes.f), covariant ``B_u/B_v``, differential volume ``vp``, the mass ->
  pressure closure ``pres = mass / vp**gamma`` and the total pressure ``bsq``.
- :func:`energies_and_force_norms` — energies ``wb/wp``, plasma ``volume`` and
  the residual normalizations ``fnorm/fnormL`` + ``r1`` (bcovar.f, getfsq).
- :func:`preconditioned_force_norm` — ``fnorm1`` (bcovar.f).
- :func:`surface_currents`       — ``buco/bvco`` (fbal.f) and the derived
  scalars ``ctor``, ``rbtor``, ``rbtor0`` (bcovar.f).
- :func:`constraint_scaling`     — the spectral-condensation strength
  ``tcon(js)`` (bcovar.f + the diagonal elements of precondn.f).

All functions are pure and jit-friendly (no host round-trips, no value-based
Python branching).  The VMEC2000 recompute cadence (``ns4 = 25`` iterations for
the preconditioner/norms/tcon refresh) is deliberately *not* implemented here —
caching is the solver's job.

The numerics are ported verbatim from the parity-proven legacy kernels
``vmex.kernels.bcovar``, ``vmex.kernels.residue``,
``vmex.kernels.lforbal`` and ``vmex.kernels.constraints``; equivalence
is enforced in ``tests/test_geometry_fields_ab.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import jax.numpy as jnp

from .fourier import ModeTable, TrigTables
from .geometry import HalfMeshJacobian, RealSpaceGeometry, sqrt_s_half_mesh
from .transforms import register_pytree_dataclass as _register

__all__ = [
    "MetricElements",
    "MagneticFields",
    "EnergiesAndForceNorms",
    "CurrentDiagnostics",
    "metric_elements",
    "lambda_scale",
    "magnetic_fields",
    "energies_and_force_norms",
    "preconditioned_force_norm",
    "surface_currents",
    "constraint_scaling",
]

Array = Any

TWO_PI = 2.0 * np.pi


def _r0scale(trig: TrigTables) -> float:
    """VMEC ``r0scale = mscale(0)*nscale(0)`` (fixaray.f); identically 1."""
    return float(np.asarray(trig.mscale)[0] * np.asarray(trig.nscale)[0])


def _angle_weights(trig: TrigTables, dtype: Any) -> Array:
    """Angular integration weights ``wint`` as a jnp array (profil3d.f)."""
    return jnp.asarray(trig.wint, dtype=dtype)


def _axis_mask(ns: int, dtype: Any) -> Array:
    """Radial mask zeroing the axis surface (profil3d.f: ``pwint(:, 1) = 0``)."""
    return (jnp.arange(ns, dtype=jnp.int32) > 0).astype(dtype)


@dataclass(frozen=True)
class MetricElements:
    """Half-mesh metric elements (VMEC: ``guu, guv, gvv`` in ``bcovar.f``).

    ``guu = R_u^2 + Z_u^2``, ``guv = R_u R_v + Z_u Z_v``,
    ``gvv = R_v^2 + Z_v^2 + R^2``, each assembled from the even/odd-m channels
    and interpolated to the radial half mesh; ``guv/gvv`` are zeroed on the
    axis row.  Shapes ``(ns, ntheta3, nzeta)``.
    """

    guu: Array
    guv: Array
    gvv: Array


@dataclass(frozen=True)
class MagneticFields:
    """Half-mesh magnetic field state (VMEC2000: ``bcovar.f``).

    - ``bsupu/bsupv`` (VMEC ``bsupu, bsupv``): contravariant components
      ``B^u = phipog*(chip + lamscale*d(lambda)/dzeta)`` and
      ``B^v = phipog*(phip + lamscale*d(lambda)/dtheta)`` with
      ``phipog = 1/sqrt(g)`` on the half mesh (zeroed on axis).
    - ``bsubu/bsubv`` (``bsubu, bsubv``): covariant components
      ``B_u = guu B^u + guv B^v``, ``B_v = guv B^u + gvv B^v``.
    - ``total_pressure`` (``bsq``): ``|B|^2/2 + p`` with
      ``|B|^2 = B^u B_u + B^v B_v``.
    - ``pressure`` (``pres``, shape ``(ns,)``): kinetic pressure on the half
      mesh, ``mass/vp**gamma`` when a mass profile is given (adiabatic
      closure), zero on axis.
    - ``vp`` (``vp``): differential volume ``dV/ds / (2 pi)^2 =
      signgs * <sqrt(g)>`` per half-mesh surface, shape ``(ns,)``.
    - ``lamscale``: the lambda normalization (profil1d.f).
    - ``chips`` (``chips``): the effective half-mesh ``d(chi)/ds`` actually
      used in ``B^u`` — equal to the input except in the ``ncurr = 1``
      current-constrained mode (add_fluxes.f), shape ``(ns,)``.
    """

    bsupu: Array
    bsupv: Array
    bsubu: Array
    bsubv: Array
    total_pressure: Array
    pressure: Array
    vp: Array
    lamscale: Array
    chips: Array


@dataclass(frozen=True)
class EnergiesAndForceNorms:
    """Energies and force-residual normalizations (VMEC2000: ``bcovar.f``).

    - ``wb`` (``wb``): magnetic energy ``|hs * sum_{js>=2} <signgs*sqrt(g)*
      |B|^2/2>|`` (the wout ``wb`` normalization; multiply by ``(2 pi)^2`` for
      Joules/mu0).
    - ``wp`` (``wp``): kinetic energy ``hs * sum_{js>=2} vp*pres``.
    - ``volume``: plasma volume / ``(2 pi)^2`` = ``hs * sum(vp(2:ns))``.
    - ``vp``: per-surface differential volume (same as
      :class:`MagneticFields`).
    - ``energy_density`` (``r2``): ``max(wb, wp)/volume``.
    - ``fnorm`` (``fnorm``): ``1 / (sum(guu*r12^2*wint) * r2^2)`` — normalizes
      the physical R/Z force residuals ``fsqr/fsqz`` (getfsq).
    - ``fnormL`` (``fnormL``): ``1 / (sum((B_u^2 + B_v^2)*wint) * lamscale^2)``
      — normalizes the lambda residual ``fsql``.
    - ``r1``: the ``1/(2*r0scale)^2 = 1/4`` prefactor applied with ``fnorm``
      in ``getfsq``.
    """

    wb: Array
    wp: Array
    volume: Array
    vp: Array
    energy_density: Array
    fnorm: Array
    fnormL: Array
    r1: Array


@dataclass(frozen=True)
class CurrentDiagnostics:
    """Flux-surface current averages and derived scalars (VMEC2000 names).

    - ``buco`` (``buco``): ``<B_u>`` per half-mesh surface — the enclosed
      toroidal current profile (fbal.f), shape ``(ns,)``.
    - ``bvco`` (``bvco``): ``<B_v>`` — the poloidal current profile.
    - ``ctor`` (``ctor``): net toroidal plasma current (mu0*I),
      ``signgs * 2 pi * (1.5*buco(ns) - 0.5*buco(ns-1))`` (bcovar.f).
    - ``rbtor`` (``rbtor``): ``R*Btor`` at the edge,
      ``1.5*bvco(ns) - 0.5*bvco(ns-1)``.
    - ``rbtor0`` (``rbtor0``): ``R*Btor`` on axis,
      ``1.5*bvco(2) - 0.5*bvco(3)``.
    """

    buco: Array
    bvco: Array
    ctor: Array
    rbtor: Array
    rbtor0: Array


for _cls in (MetricElements, MagneticFields, EnergiesAndForceNorms, CurrentDiagnostics):
    _register(_cls)


# ---------------------------------------------------------------------------
# Metric elements (bcovar.f)
# ---------------------------------------------------------------------------


def _half_mesh_from_even_odd(even: Array, odd: Array, *, s: Array) -> Array:
    """Half-mesh average of ``X = X_even + sqrt(s)*X_odd`` (bcovar.f)."""
    s = jnp.asarray(s)
    if int(s.shape[0]) < 2:
        return even
    sh = sqrt_s_half_mesh(s)[:, None, None]
    inner = 0.5 * (even[1:] + even[:-1] + sh[1:] * (odd[1:] + odd[:-1]))
    return jnp.concatenate([inner[:1], inner], axis=0)


def _with_axis_zero(a: Array) -> Array:
    a = jnp.asarray(a)
    if a.shape[0] == 0:
        return a
    return jnp.concatenate([jnp.zeros_like(a[:1]), a[1:]], axis=0)


def _prepend_axis_zero(body: Array, like: Array) -> Array:
    return jnp.concatenate([jnp.zeros_like(jnp.asarray(like)[:1]), body], axis=0)


def metric_elements(geometry: RealSpaceGeometry, *, s: Array) -> MetricElements:
    """Half-mesh metric elements ``guu, guv, gvv``.

    VMEC2000: ``bcovar.f`` — with ``X = X_even + sqrt(s)*X_odd`` the squared /
    cross products decompose into even parts (``a0^2 + b0^2 + s*(a1^2 +
    b1^2)``) and odd parts (``2*(a0*a1 + b0*b1)``) which are then averaged to
    the half mesh with the ``sqrt(s_half)`` weight on the odd piece:

    - ``guu = R_u^2 + Z_u^2``
    - ``guv = R_u R_v + Z_u Z_v``
    - ``gvv = R_v^2 + Z_v^2 + R^2``  (the ``R^2`` term is the cylindrical
      toroidal metric; VMEC works at unit ``d(zeta_physical)/d(zeta)``)

    ``guv`` and ``gvv`` are zeroed on the axis row (VMEC never reads them
    there and the legacy kernels enforce it).
    """
    s = jnp.asarray(s)
    ss = s[:, None, None]
    g = geometry

    def squares_even_odd(a0: Array, a1: Array, b0: Array, b1: Array) -> tuple[Array, Array]:
        even = a0 * a0 + b0 * b0 + ss * (a1 * a1 + b1 * b1)
        odd = 2.0 * (a0 * a1 + b0 * b1)
        return even, odd

    def cross_even_odd(a0: Array, a1: Array, b0: Array, b1: Array) -> tuple[Array, Array]:
        even = a0 * b0 + ss * (a1 * b1)
        odd = a0 * b1 + a1 * b0
        return even, odd

    guu_e, guu_o = squares_even_odd(
        g.dR_dtheta_even, g.dR_dtheta_odd, g.dZ_dtheta_even, g.dZ_dtheta_odd
    )
    guv_e, guv_o = cross_even_odd(
        g.dR_dtheta_even, g.dR_dtheta_odd, g.dR_dzeta_even, g.dR_dzeta_odd
    )
    guv_e2, guv_o2 = cross_even_odd(
        g.dZ_dtheta_even, g.dZ_dtheta_odd, g.dZ_dzeta_even, g.dZ_dzeta_odd
    )
    guv_e = guv_e + guv_e2
    guv_o = guv_o + guv_o2
    gvv_e, gvv_o = squares_even_odd(
        g.dR_dzeta_even, g.dR_dzeta_odd, g.dZ_dzeta_even, g.dZ_dzeta_odd
    )
    # R^2 contribution to gvv.
    r2_even = g.R_even * g.R_even + ss * (g.R_odd * g.R_odd)
    r2_odd = 2.0 * g.R_even * g.R_odd

    guu = _half_mesh_from_even_odd(guu_e, guu_o, s=s)
    guv = _with_axis_zero(_half_mesh_from_even_odd(guv_e, guv_o, s=s))
    gvv = _with_axis_zero(_half_mesh_from_even_odd(gvv_e + r2_even, gvv_o + r2_odd, s=s))
    return MetricElements(guu=guu, guv=guv, gvv=gvv)


# ---------------------------------------------------------------------------
# Contravariant/covariant field, pressure (bcovar.f + add_fluxes.f)
# ---------------------------------------------------------------------------


def lambda_scale(phips: Array, s: Array) -> Array:
    """VMEC ``lamscale = sqrt(hs * sum_{js=2..ns} phips(js)^2)`` (profil1d.f).

    ``phips`` is the half-mesh ``d(phi)/ds / (2 pi)`` profile (VMEC internal
    units).  Lambda is evolved as ``lamscale * lambda`` internally; the factor
    reappears in ``B^u/B^v`` and in ``fnormL``.
    """
    phips = jnp.asarray(phips)
    s = jnp.asarray(s)
    if phips.shape[0] < 2:
        return jnp.asarray(1.0, dtype=phips.dtype)
    hs = s[1] - s[0]
    return jnp.sqrt(hs * jnp.sum(phips[1:] ** 2))


def magnetic_fields(
    *,
    geometry: RealSpaceGeometry,
    jacobian: HalfMeshJacobian,
    metrics: MetricElements,
    trig: TrigTables,
    s: Array,
    phips: Array,
    phipf: Array,
    chips: Array,
    signgs: int,
    gamma: float = 0.0,
    pressure: Array | None = None,
    mass: Array | None = None,
    ncurr: int = 0,
    enclosed_current: Array | None = None,
) -> MagneticFields:
    """Contravariant/covariant B, differential volume and total pressure.

    VMEC2000: ``bcovar.f`` (field/pressure blocks) + ``add_fluxes.f``:

    - ``lu = lamscale * d(lambda)/dtheta + phip``,
      ``lv = -lamscale * d(lambda)/dzeta`` on the full mesh (even/odd planes);
    - half-mesh staggering: ``B^v = phipog * <lu>``, ``B^u = phipog * <lv> +
      chip * phipog`` with ``phipog = 1/sqrt(g)`` (``overg``), both zero on
      the axis row;
    - ``ncurr = 1`` (add_fluxes.f): the half-mesh ``chips`` is solved from the
      prescribed enclosed toroidal current profile ``icurv``:
      ``chips = (icurv - <guu B^u_lam + guv B^v>) / <guu/sqrt(g)>``;
    - ``B_u = guu B^u + guv B^v``, ``B_v = guv B^u + gvv B^v``;
    - ``vp = signgs * <sqrt(g)>`` per surface (``dV/ds/(2 pi)^2``);
    - adiabatic pressure closure ``pres = mass / vp**gamma`` (zero where
      ``vp = 0``, i.e. on axis) when ``mass`` is given, else the supplied
      half-mesh ``pressure`` profile is used;
    - ``bsq = |B|^2/2 + pres`` (VMEC ``bsq``, the total pressure).

    Parameters
    ----------
    phips, phipf, chips:
        VMEC-internal flux derivative profiles, shape ``(ns,)``: half-mesh
        ``phips``, full-mesh ``phip`` (``= signgs*phipf_wout/(2 pi)``), and
        half-mesh ``chips`` (``d(chi)/ds``).
    signgs:
        Jacobian orientation sign (+1/-1).
    gamma:
        Adiabatic index (INDATA ``GAMMA``); used only with ``mass``.
    pressure, mass:
        Half-mesh profiles ``(ns,)`` in VMEC internal units (mu0*Pa).  Exactly
        one should be provided; ``mass`` takes precedence.
    ncurr, enclosed_current:
        ``ncurr = 1`` activates the current-constrained mode with the
        prescribed ``icurv`` profile (``enclosed_current``, defaults to zero).
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    sqrt_g = jacobian.sqrt_g
    dtype = sqrt_g.dtype

    lamscale = lambda_scale(jnp.asarray(phips, dtype=dtype), s)
    phipf = jnp.asarray(phipf, dtype=dtype)
    chips = jnp.asarray(chips, dtype=dtype)

    # phipog = 1/sqrt(g) (VMEC ``overg``), zero where sqrt(g) is zero (axis).
    safe_g = jnp.where(sqrt_g != 0, sqrt_g, jnp.asarray(1.0, dtype=dtype))
    phipog = jnp.where(sqrt_g != 0, 1.0 / safe_g, 0.0)

    # Full-mesh lambda-force fields (VMEC lu/lv; note lv = -d(lambda)/dzeta).
    lu_even = lamscale * geometry.dlambda_dtheta_even + phipf[:, None, None]
    lu_odd = lamscale * geometry.dlambda_dtheta_odd
    lv_even = -lamscale * geometry.dlambda_dzeta_even
    lv_odd = -lamscale * geometry.dlambda_dzeta_odd

    sh = sqrt_s_half_mesh(s)[:, None, None]
    zero = jnp.zeros_like(sqrt_g)
    if ns >= 2:
        bsupv_even = _prepend_axis_zero(0.5 * phipog[1:] * (lu_even[1:] + lu_even[:-1]), sqrt_g)
        bsupv_odd = _prepend_axis_zero(0.5 * phipog[1:] * (lu_odd[1:] + lu_odd[:-1]), sqrt_g)
        bsupu_even = _prepend_axis_zero(0.5 * phipog[1:] * (lv_even[1:] + lv_even[:-1]), sqrt_g)
        bsupu_odd = _prepend_axis_zero(0.5 * phipog[1:] * (lv_odd[1:] + lv_odd[:-1]), sqrt_g)
        bsupv = bsupv_even + sh * bsupv_odd
        bsupu_lambda = bsupu_even + sh * bsupu_odd
    else:
        bsupv = bsupu_lambda = zero

    # Angular weights with the axis surface masked (profil3d.f pwint).
    wint = _angle_weights(trig, dtype)
    pwint = wint[None, :, :] * _axis_mask(ns, dtype)[:, None, None]

    guu, guv = metrics.guu, metrics.guv
    if int(ncurr) == 1 and ns >= 2:
        # add_fluxes.f: solve <B_u> = icurv for the half-mesh chips profile.
        icurv = (
            jnp.zeros((ns,), dtype=dtype)
            if enclosed_current is None
            else jnp.asarray(enclosed_current, dtype=dtype)
        )
        top = icurv - jnp.sum(pwint * ((guu * bsupu_lambda) + (guv * bsupv)), axis=(1, 2))
        bot = jnp.sum(pwint * (phipog * guu), axis=(1, 2))
        safe_bot = jnp.where(bot != 0.0, bot, jnp.asarray(1.0, dtype=dtype))
        chips = jnp.concatenate(
            [jnp.zeros_like(chips[:1]), jnp.where(bot != 0.0, top / safe_bot, chips)[1:]],
            axis=0,
        )

    bsupu = _with_axis_zero(bsupu_lambda + chips[:, None, None] * phipog)
    bsupv = _with_axis_zero(bsupv)

    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + metrics.gvv * bsupv

    # Differential volume vp = signgs * <sqrt(g)> (dV/ds / (2 pi)^2).
    signgs_f = jnp.asarray(float(int(signgs)), dtype=dtype)
    vp = jnp.sum(signgs_f * pwint * sqrt_g, axis=(1, 2))

    # Pressure: adiabatic mass closure (bcovar.f: pres = mass/vp**gamma).
    if mass is not None:
        mass = jnp.asarray(mass, dtype=dtype)
        safe_vp = jnp.where(vp != 0.0, vp, jnp.asarray(1.0, dtype=dtype))
        pres = jnp.where(vp != 0.0, mass / (safe_vp ** float(gamma)), jnp.asarray(0.0, dtype=dtype))
    elif pressure is not None:
        pres = jnp.asarray(pressure, dtype=dtype)
    else:
        pres = jnp.zeros((ns,), dtype=dtype)

    b_squared = bsupu * bsubu + bsupv * bsubv
    total_pressure = 0.5 * b_squared + pres[:, None, None]

    return MagneticFields(
        bsupu=bsupu,
        bsupv=bsupv,
        bsubu=bsubu,
        bsubv=bsubv,
        total_pressure=total_pressure,
        pressure=pres,
        vp=vp,
        lamscale=lamscale,
        chips=chips,
    )


# ---------------------------------------------------------------------------
# Energies and force norms (bcovar.f)
# ---------------------------------------------------------------------------


def energies_and_force_norms(
    *,
    jacobian: HalfMeshJacobian,
    metrics: MetricElements,
    fields: MagneticFields,
    trig: TrigTables,
    s: Array,
    signgs: int,
) -> EnergiesAndForceNorms:
    """Energies ``wb/wp`` and residual normalizations ``fnorm/fnormL``.

    VMEC2000: ``bcovar.f`` —

    - ``wb = |hs * sum_{js=2..ns} <signgs*sqrt(g) * |B|^2/2>|``
    - ``wp = hs * sum_{js=2..ns} vp*pres``
    - ``volume = hs * sum(vp(2:ns))``, ``r2 = max(wb, wp)/volume``
    - ``fnorm  = 1 / (sum(guu * r12^2 * wint) * r2^2)``  (R/Z force norm)
    - ``fnormL = 1 / (sum((B_u^2 + B_v^2) * wint) * lamscale^2)``  (lambda)
    - ``r1 = 1/(2*r0scale)^2`` — the constant companion factor in ``getfsq``
      (``fsqr = r1 * fnorm * |F_R|^2`` etc.).

    For exact legacy parity the surface pressure entering ``wp`` is recovered
    from ``bsq - |B|^2/2`` (numerically identical to using ``fields.pressure``
    directly up to roundoff relative to ``wb``).
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    sqrt_g = jacobian.sqrt_g
    dtype = sqrt_g.dtype
    hs = jnp.asarray(s[1] - s[0], dtype=dtype)

    w_ang = _angle_weights(trig, dtype)
    mask_js = _axis_mask(ns, dtype)[:, None, None]

    jac = jnp.asarray(float(int(signgs)), dtype=dtype) * sqrt_g
    jac = jac * mask_js
    vp = jnp.sum(w_ang[None, :, :] * jac, axis=(1, 2))
    volume = hs * jnp.sum(vp[1:])

    b_squared = (fields.bsupu * fields.bsubu) + (fields.bsupv * fields.bsubv)
    wb_local = jnp.sum(w_ang[None, :, :] * jac * (0.5 * b_squared), axis=(1, 2))
    wb = jnp.abs(hs * jnp.sum(wb_local[1:]))

    pres_1d = (fields.total_pressure - (0.5 * b_squared))[:, 0, 0]
    wp = hs * jnp.sum(vp[1:] * pres_1d[1:])

    r2 = jnp.where(
        volume != 0.0,
        jnp.maximum(wb, wp) / volume,
        jnp.asarray(float("inf"), dtype=dtype),
    )

    r12 = jacobian.r12
    guu_r12sq = metrics.guu * (r12 * r12)
    mask_1d = _axis_mask(ns, dtype)
    rz_denom_surface = jnp.sum(guu_r12sq * w_ang[None, :, :], axis=(1, 2))
    denom_f = jnp.sum(rz_denom_surface * mask_1d)
    fnorm = jnp.where(
        denom_f != 0.0,
        1.0 / (denom_f * (r2 * r2)),
        jnp.asarray(float("inf"), dtype=dtype),
    )

    bsub_sq = (fields.bsubu * fields.bsubu) + (fields.bsubv * fields.bsubv)
    l_denom_surface = jnp.sum(bsub_sq * w_ang[None, :, :], axis=(1, 2))
    denom_L = jnp.sum(l_denom_surface * mask_1d)
    lamscale = jnp.asarray(fields.lamscale, dtype=dtype)
    fnormL = jnp.where(
        denom_L != 0.0,
        1.0 / (denom_L * (lamscale * lamscale)),
        jnp.asarray(float("inf"), dtype=dtype),
    )

    r1 = jnp.asarray(1.0 / (2.0 * _r0scale(trig)) ** 2, dtype=dtype)
    return EnergiesAndForceNorms(
        wb=wb,
        wp=wp,
        volume=volume,
        vp=vp,
        energy_density=r2,
        fnorm=fnorm,
        fnormL=fnormL,
        r1=r1,
    )


def preconditioned_force_norm(
    *,
    R_cos: Array,
    Z_sin: Array,
    modes: ModeTable,
    R_sin: Array | None = None,
    Z_cos: Array | None = None,
) -> Array:
    """Preconditioned R/Z force normalization ``fnorm1``.

    VMEC2000: ``bcovar.f`` — ``fnorm1 = 1/sum(xc**2)`` over the internal R/Z
    spectral state.  In the signed-(m, n) packing this is

        ``fnorm1 = 1 / sum_{js>=2} w_k * (Rcos_k^2 + Zsin_k^2 [+ Rsin_k^2 +
        Zcos_k^2 for lasym])``

    with ``w_k = 2`` for helical modes (``m > 0`` and ``n != 0``; the internal
    ``cc/ss`` blocks each carry half the signed content) and ``w_k = 1``
    otherwise, and the ``(m, n) = (0, 0)`` mode excluded from the ``Rcos``
    block (VMEC skips the ``R00`` profile).  The axis surface is excluded
    (``bcovar_par`` accumulates over ``l = 2..ns``).  Ported from the
    parity-proven ``ModeTransform.rz_norm`` in the legacy solver.

    Pass the *internal* (evolved) coefficients — the m=1 constrained
    representation — exactly as VMEC does with ``xc``.
    """
    R_cos = jnp.asarray(R_cos)
    Z_sin = jnp.asarray(Z_sin)
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    weight = np.where((m > 0) & (n != 0), 2.0, 1.0)
    include_R = weight * ((m > 0) | (n != 0))
    w_R = jnp.asarray(include_R, dtype=R_cos.dtype)[None, :]
    w = jnp.asarray(weight, dtype=R_cos.dtype)[None, :]

    interior = slice(1, None)
    rz_norm = jnp.sum(w[:, :] * Z_sin[interior] * Z_sin[interior]) + jnp.sum(
        w_R * R_cos[interior] * R_cos[interior]
    )
    if R_sin is not None:
        R_sin = jnp.asarray(R_sin)
        rz_norm = rz_norm + jnp.sum(w * R_sin[interior] * R_sin[interior])
    if Z_cos is not None:
        Z_cos = jnp.asarray(Z_cos)
        rz_norm = rz_norm + jnp.sum(w * Z_cos[interior] * Z_cos[interior])
    return jnp.where(
        rz_norm != 0.0,
        1.0 / rz_norm,
        jnp.asarray(float("inf"), dtype=R_cos.dtype),
    )


# ---------------------------------------------------------------------------
# Surface currents (fbal.f / bcovar.f)
# ---------------------------------------------------------------------------


def surface_currents(
    *,
    bsubu: Array,
    bsubv: Array,
    trig: TrigTables,
    s: Array,
    signgs: int,
) -> CurrentDiagnostics:
    """Flux-surface current averages ``buco/bvco`` and ``ctor/rbtor/rbtor0``.

    VMEC2000: ``fbal.f`` (``buco = <B_u>``, ``bvco = <B_v>`` with the
    axis-masked angular weights) and ``bcovar.f`` for the edge/axis
    extrapolations:

    - ``ctor  = signgs * 2 pi * (1.5*buco(ns) - 0.5*buco(ns-1))`` — the net
      toroidal plasma current in mu0*Ampere units;
    - ``rbtor = 1.5*bvco(ns) - 0.5*bvco(ns-1)`` (edge ``R*Btor``, VMEC
      ``fpsi(ns)`` extrapolation);
    - ``rbtor0 = 1.5*bvco(2) - 0.5*bvco(3)`` (axis ``R*Btor``).
    """
    bsubu = jnp.asarray(bsubu)
    bsubv = jnp.asarray(bsubv)
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    dtype = bsubu.dtype
    if ns < 2:
        z = jnp.zeros((ns,), dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        return CurrentDiagnostics(buco=z, bvco=z, ctor=zero, rbtor=zero, rbtor0=zero)

    wint = _angle_weights(trig, dtype)
    pwint = wint[None, :, :] * _axis_mask(ns, dtype)[:, None, None]
    buco = jnp.sum(bsubu * pwint, axis=(1, 2))
    bvco = jnp.sum(bsubv * pwint, axis=(1, 2))

    signgs_f = jnp.asarray(float(int(signgs)), dtype=dtype)
    two_pi = jnp.asarray(TWO_PI, dtype=dtype)
    ctor = signgs_f * two_pi * (1.5 * buco[-1] - 0.5 * buco[-2])
    rbtor = 1.5 * bvco[-1] - 0.5 * bvco[-2]
    rbtor0 = 1.5 * bvco[1] - 0.5 * bvco[2] if ns >= 3 else bvco[-1]
    return CurrentDiagnostics(buco=buco, bvco=bvco, ctor=ctor, rbtor=rbtor, rbtor0=rbtor0)


# ---------------------------------------------------------------------------
# Constraint scaling tcon (bcovar.f + precondn.f diagonal)
# ---------------------------------------------------------------------------


def constraint_scaling(
    *,
    tcon0: Array,
    geometry: RealSpaceGeometry,
    jacobian: HalfMeshJacobian,
    total_pressure: Array,
    trig: TrigTables,
    s: Array,
) -> Array:
    """Spectral-condensation constraint strength ``tcon(js)``.

    VMEC2000: ``bcovar.f`` (with the ``m = 0`` diagonal elements of
    ``precondn.f``) — per plan Appendix D::

        tcon(js) = min(|ard(js,1)|/arnorm(js), |azd(js,1)|/aznorm(js))
                   * tcon_multiplier * (32*hs)^2    for js = 2..ns-1,
        tcon(ns) = 0.5 * tcon(ns-1),

    where (precondn.f, with ``ptau = pfactor*r12^2*bsq*wint/sqrt(g)`` and
    ``pfactor = -4*r0scale^2``):

    - ``ax(js,1) = sum(ptau * (zu12*ohs)^2)``, ``ard(js,1) = ax(js,1) +
      ax(js+1,1)`` (the Z-derivative pair feeds the *R* diagonal, and vice
      versa for ``azd``);
    - ``arnorm(js) = <ru0^2>``, ``aznorm(js) = <zu0^2>`` with
      ``ru0 = ru_even + sqrt(s)*ru_odd``;
    - ``tcon_multiplier = min(|tcon0|, 1)*(1 + ns*(1/60 + ns/(200*120)))/(4*r0scale^2)^2``
      (the resolution-dependent ramp of ``bcovar.f``).

    The axis slot holds the clamped ``tcon0`` (VMEC initializes ``tcon(:) =
    tcon0`` before overwriting the interior; the value is never used by the
    constraint operator).  The ``ns4 = 25``-iteration refresh cadence of
    VMEC2000 lives in the solver, not here — this is the pure recompute.
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    bsq = jnp.asarray(total_pressure)
    dtype = bsq.dtype
    if ns < 2:
        return jnp.zeros((ns,), dtype=dtype)

    hs = jnp.asarray(s[1] - s[0], dtype=dtype)
    ohs = jnp.where(hs != 0, 1.0 / hs, jnp.asarray(0.0, dtype=dtype))
    r0scale = _r0scale(trig)
    pfactor = -4.0 * r0scale**2

    wint3 = _angle_weights(trig, dtype)[None, :, :]

    sqrt_g = jacobian.sqrt_g
    r12 = jacobian.r12
    safe_g = jnp.where(sqrt_g != 0, sqrt_g, jnp.ones_like(sqrt_g))
    ptau = (pfactor * (r12 * r12) * bsq * wint3) / safe_g

    ax_r = jnp.sum(ptau * ((jacobian.zu12 * ohs) ** 2), axis=(1, 2))
    ax_z = jnp.sum(ptau * ((jacobian.ru12 * ohs) ** 2), axis=(1, 2))
    ax_r = ax_r.at[0].set(0.0)
    ax_z = ax_z.at[0].set(0.0)
    ard1 = ax_r + jnp.concatenate([ax_r[1:], jnp.zeros((1,), dtype=dtype)], axis=0)
    azd1 = ax_z + jnp.concatenate([ax_z[1:], jnp.zeros((1,), dtype=dtype)], axis=0)

    ru0, zu0 = geometry.theta_derivatives_full(s)
    arnorm = jnp.sum((ru0 * ru0) * wint3, axis=(1, 2))
    aznorm = jnp.sum((zu0 * zu0) * wint3, axis=(1, 2))
    arnorm = jnp.where(arnorm != 0, arnorm, jnp.ones_like(arnorm))
    aznorm = jnp.where(aznorm != 0, aznorm, jnp.ones_like(aznorm))

    tcon0_clamped = jnp.minimum(jnp.abs(jnp.asarray(tcon0, dtype=dtype)), 1.0)
    ns_f = float(ns)
    tcon_multiplier = tcon0_clamped * (1.0 + ns_f * (1.0 / 60.0 + ns_f / (200.0 * 120.0)))
    tcon_multiplier = tcon_multiplier / ((4.0 * r0scale**2) ** 2)

    tcon = jnp.zeros((ns,), dtype=dtype)
    tcon = tcon.at[0].set(tcon0_clamped)
    if ns >= 3:
        js = jnp.arange(ns, dtype=jnp.int32) + 1  # 1-based, as in the Fortran
        interior = (js >= 2) & (js <= (ns - 1))
        ratio_r = jnp.abs(ard1) / arnorm
        ratio_z = jnp.abs(azd1) / aznorm
        core = jnp.minimum(ratio_r, ratio_z) * (
            jnp.asarray(tcon_multiplier, dtype=hs.dtype) * (32.0 * hs) ** 2
        )
        tcon = jnp.where(interior.astype(core.dtype), core, tcon)
        tcon = tcon.at[-1].set(0.5 * tcon[-2])
    return tcon
