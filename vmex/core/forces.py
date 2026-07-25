"""Real-space MHD force kernels and the spectral-condensation constraint force.

VMEC2000 counterparts
---------------------
- ``Sources/General/forces.f``  — the cylindrical R/Z force kernels
  ``armn/brmn/crmn`` and ``azmn/bzmn/czmn`` (even/odd-m planes) built from the
  half-mesh field/geometry quantities of ``bcovar.f``:
  :func:`mhd_force_kernels`.
- ``Sources/General/bcovar.f`` (lambda-force block) — the full-mesh covariant
  lambda force kernels ``blmn/clmn`` from ``bsubu/bsubv``:
  :func:`lambda_force_kernels`.
- ``Sources/General/alias.f`` + ``funct3d.f`` (constraint block) — the
  spectral-condensation constraint force ``gcon`` from
  ``ztemp = (rcon - rcon0)*ru0 + (zcon - zcon0)*zu0``:
  :func:`constraint_force` / :func:`alias_constraint_force`, with the
  ``faccon(m)`` weights of ``fixaray.f`` (:func:`faccon`).
- ``Sources/General/symforce.f`` — per-kernel symmetric/antisymmetric split
  for ``lasym`` runs (delegating to
  :func:`vmex.core.transforms.symforce_split`): :func:`symmetrize_forces`.

Equations (Hirshman & Whitson 1983; VMEC2000 forces.f)
------------------------------------------------------
With ``lu = bsq*R12`` and ``lv = bsq*tau`` on the half mesh (``bsq = |B|^2/2 +
p``) and the "GIJ" products ``guu = B^u B^u sqrt(g)``, ``guv = B^u B^v
sqrt(g)``, ``gvv = B^v B^v sqrt(g)``, the spectral force components are

    F_R = armn - d(brmn)/dtheta + d(crmn)/dzeta
    F_Z = azmn - d(bzmn)/dtheta + d(czmn)/dzeta
    F_lambda = -d(blmn)/dtheta + d(clmn)/dzeta

where the ``A`` kernels carry the radial derivative of the MHD energy
(``ohs``-differenced half-mesh terms) plus the ``R``-curvature term
``-gvv*R``, the ``B`` kernels the poloidal metric terms, and the ``C`` kernels
the toroidal metric terms.  Odd-m planes carry the internal ``sqrt(s)``
representation and the ``d(sqrt s)/ds = 1/(2 sqrt s)`` chain-rule terms
(``dshalfds = 0.25`` discrete factor).

The numerics are ported verbatim from the parity-proven legacy kernels
``vmex.kernels.forces`` (``_assemble_vmec_rz_radial_forces``,
``_constraint_kernels_from_state``), ``vmex.kernels.bcovar``
(``_compute_bcovar_lambda_force_assembly``) and
``vmex.kernels.constraints`` (``alias_gcon``/``faccon_from_signgs``);
equivalence is enforced in ``tests/test_forces_residuals_ab.py``.
All functions are pure ``jax.numpy`` (jit-friendly, no host round-trips).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

import jax.numpy as jnp
from jax import lax

from .fields import MagneticFields, MetricElements
from .fourier import ModeTable, TrigTables
from .geometry import HalfMeshJacobian, RealSpaceGeometry, sqrt_s_half_mesh
from .transforms import (
    SpectralForce,
    _fourier_to_real_fft,
    fourier_to_real,
    register_pytree_dataclass as _register,
    symforce_split,
    tomnspa,
    tomnsps,
)

__all__ = [
    "RealSpaceForces",
    "faccon",
    "lambda_force_kernels",
    "mhd_force_kernels",
    "alias_constraint_force",
    "constraint_force",
    "mhd_forces",
    "symmetrize_forces",
    "spectral_mhd_forces",
    "apply_m1_force_balance",
]

Array = Any

#: VMEC ``dshalfds = p25``: the discrete ``d(sqrt s)/ds * hs`` factor pair
#: (forces.f / jacobian.f).
_D_SQRT_S_HALF_DS = 0.25

#: VMEC ``pdamp`` (bcovar.f, v8.49): blending weight preserving the well
#: conditioning of the full-mesh ``bsubv`` reconstruction.
_LAMBDA_FORCE_PDAMP = 0.05


def _einsum(expr: str, *operands: Array) -> Array:
    """Einsum with HIGHEST precision (parity with the legacy kernels)."""
    return jnp.einsum(expr, *operands, precision=lax.Precision.HIGHEST)


# ---------------------------------------------------------------------------
# Radial staggering primitives (forces.f forward operations along s)
# ---------------------------------------------------------------------------


def _with_axis_zero(a: Array) -> Array:
    a = jnp.asarray(a)
    if a.shape[0] == 0:
        return a
    return jnp.concatenate([jnp.zeros_like(a[:1]), a[1:]], axis=0)


def _forward_average_half(a: Array) -> Array:
    """``a(js) <- 0.5*(a(js) + a(js+1))`` with ``a(ns+1) = 0`` (forces.f)."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    body = 0.5 * (a[:-1] + a[1:])
    tail = 0.5 * a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _forward_average_half_or_zero(a: Array) -> Array:
    """As :func:`_forward_average_half`, but zero for degenerate grids."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return jnp.zeros_like(a)
    return _forward_average_half(a)


def _forward_sum_half(a: Array) -> Array:
    """``a(js) <- a(js) + a(js+1)`` with ``a(ns+1) = 0`` (forces.f)."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    body = a[:-1] + a[1:]
    tail = a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _forward_difference_half(a: Array) -> Array:
    """``a(js) <- a(js+1) - a(js)`` with ``a(ns+1) = 0`` (forces.f)."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    body = a[1:] - a[:-1]
    tail = -a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _forward_difference_half_with_average(a: Array, b: Array) -> Array:
    """``a(js) <- a(js+1) - a(js) + 0.5*(b(js) + b(js+1))`` (forces.f armn)."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.shape[0] < 2:
        return a
    body = a[1:] - a[:-1] + 0.5 * (b[:-1] + b[1:])
    tail = -a[-1:] + 0.5 * b[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _scale_lambda_full_mesh(a: Array, lamscale: Array) -> Array:
    """``a(2:ns) <- -lamscale * a(2:ns)`` (bcovar.f blmn/clmn scaling).

    The axis row is left untouched (VMEC never reads it: ``jlam(m) = 2``).
    """
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    return jnp.concatenate([a[:1], -lamscale * a[1:]], axis=0)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RealSpaceForces:
    """Real-space MHD force kernels on the internal grid, split by m-parity.

    VMEC2000 names (``forces.f``/``bcovar.f``; ``_even/_odd`` are the
    ``mparity = 0/1`` planes, odd carrying the internal ``sqrt(s)``
    representation): ``force_R = armn``, ``force_R_du = brmn``, ``force_R_dv =
    crmn``, ``force_Z = azmn``, ``force_Z_du = bzmn``, ``force_Z_dv = czmn``,
    ``force_lambda_du = blmn``, ``force_lambda_dv = clmn``, ``constraint_R =
    arcon``, ``constraint_Z = azcon``.  Field names match the keyword
    arguments of :func:`vmex.core.transforms.tomnsps` so the projection
    call is direct.

    ``gcon`` is the spectral-condensation constraint force field (alias.f);
    ``rcon0/zcon0`` are the constraint baselines actually used (``funct3d.f``
    — the free-boundary ``x0.9`` per-iteration ramp is the solver's job).
    All arrays have shape ``(ns, ntheta3, nzeta)``.
    """

    force_R_even: Array
    force_R_odd: Array
    force_R_du_even: Array
    force_R_du_odd: Array
    force_R_dv_even: Array
    force_R_dv_odd: Array
    force_Z_even: Array
    force_Z_odd: Array
    force_Z_du_even: Array
    force_Z_du_odd: Array
    force_Z_dv_even: Array
    force_Z_dv_odd: Array
    force_lambda_du_even: Array
    force_lambda_du_odd: Array
    force_lambda_dv_even: Array
    force_lambda_dv_odd: Array
    constraint_R_even: Array
    constraint_R_odd: Array
    constraint_Z_even: Array
    constraint_Z_odd: Array
    gcon: Array | None = None
    rcon0: Array | None = None
    zcon0: Array | None = None


_register(RealSpaceForces)


# ---------------------------------------------------------------------------
# faccon (fixaray.f)
# ---------------------------------------------------------------------------


def faccon(mpol: int, signgs: int) -> np.ndarray:
    """Constraint-force mode weights ``faccon(m)`` (VMEC2000 ``fixaray.f``).

    ``faccon(0) = faccon(mpol-1) = 0`` and, for ``m = 1 .. mpol-2``,

        ``faccon(m) = -0.25 * signgs / xmpq(m+1, 1)**2``

    with ``xmpq(m, 1) = m*(m-1)``, i.e. the constraint spectrum is restricted
    to ``m in [1, mpol-2]`` (Appendix D).  Static (NumPy) output.
    """
    mpol = int(mpol)
    if mpol <= 0:
        raise ValueError("mpol must be positive")
    m = np.arange(mpol, dtype=float)
    fac = np.zeros((mpol,), dtype=float)
    if mpol >= 3:
        denominator = ((m[1:-1] + 1.0) * m[1:-1]) ** 2  # xmpq(m+1,1)^2
        fac[1:-1] = (-0.25 * float(int(signgs))) / denominator
    return fac


# ---------------------------------------------------------------------------
# Lambda force kernels blmn/clmn (bcovar.f)
# ---------------------------------------------------------------------------


def lambda_force_kernels(
    *,
    geometry: RealSpaceGeometry,
    jacobian: HalfMeshJacobian,
    metrics: MetricElements,
    fields: MagneticFields,
    s: Array,
    phipf: Array,
) -> tuple[Array, Array, Array, Array]:
    """Full-mesh covariant lambda force kernels (VMEC2000 ``bcovar.f``).

    The lambda force needs ``bsubu/bsubv`` on the radial *full* mesh.
    ``bsubu`` is the plain forward average of the half-mesh covariant
    component; ``bsubv`` is reconstructed directly on the full mesh from the
    lambda derivatives (``lvv = gvv/sqrt(g)``, ``bsubv = <lvv>*lu_even +
    <lvv*shalf>*lu_odd + <guv*B^u>``) and then blended with the plain average
    using the v8.49 damping ``bdamp = 2*0.05*(1 - s)``:

        ``bsubv_full = bdamp*bsubv_reconstructed + (1 - bdamp)*<bsubv>``.

    Both are scaled by ``-lamscale`` on ``js >= 2`` (the axis row is inactive,
    ``jlam = 2``) and the odd planes carry the extra ``sqrt(s)`` factor:

        ``clmn = -lamscale * bsubu_full``  (multiplies d(basis)/dzeta),
        ``blmn = -lamscale * bsubv_full``  (multiplies d(basis)/dtheta).

    Parameters mirror the pipeline objects; ``phipf`` is the full-mesh
    VMEC-internal ``phip`` profile (the same one passed to
    :func:`vmex.core.fields.magnetic_fields`).

    Returns ``(blmn_even, blmn_odd, clmn_even, clmn_odd)`` — the
    ``force_lambda_du_even/odd`` and ``force_lambda_dv_even/odd`` kernels.
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    sqrt_g = jacobian.sqrt_g
    dtype = sqrt_g.dtype
    lamscale = jnp.asarray(fields.lamscale, dtype=dtype)
    phipf = jnp.asarray(phipf, dtype=dtype)

    # Full-mesh lambda derivative channels (lu = lamscale*dl/du + phip).
    lu_even = lamscale * geometry.dlambda_dtheta_even + phipf[:, None, None]
    lu_odd = lamscale * geometry.dlambda_dtheta_odd

    safe_g = jnp.where(sqrt_g != 0, sqrt_g, jnp.asarray(1.0, dtype=dtype))
    phipog = jnp.where(sqrt_g != 0, 1.0 / safe_g, 0.0)
    phipog = _with_axis_zero(phipog)

    pshalf = sqrt_s_half_mesh(s)[:, None, None]
    lvv = phipog * metrics.gvv
    lvv_sh = lvv * pshalf
    # guv*B^u — the cross-metric contribution to bsubv (half mesh).
    bsubv_cross = metrics.guv * fields.bsupu

    if ns >= 2:
        bsubv_base = jnp.concatenate(
            [
                0.5 * (lvv[:-1] + lvv[1:]) * lu_even[:-1],
                0.5 * lvv[-1:] * lu_even[-1:],
            ],
            axis=0,
        )
        bsubv_extra = jnp.concatenate(
            [
                0.5 * ((lvv_sh[:-1] + lvv_sh[1:]) * lu_odd[:-1] + bsubv_cross[:-1] + bsubv_cross[1:]),
                0.5 * (lvv_sh[-1:] * lu_odd[-1:] + bsubv_cross[-1:]),
            ],
            axis=0,
        )
        bsubv_full = bsubv_base + bsubv_extra
    else:
        bsubv_full = jnp.zeros_like(fields.bsubv)

    bsubu_full = _forward_average_half_or_zero(fields.bsubu)

    bdamp = (2.0 * _LAMBDA_FORCE_PDAMP * (1.0 - s)).astype(dtype)[:, None, None]
    if ns >= 2:
        bsubv_full = bdamp * bsubv_full + (1.0 - bdamp) * _forward_average_half_or_zero(fields.bsubv)

    sqrt_s = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    clmn_even = _scale_lambda_full_mesh(bsubu_full, lamscale)
    blmn_even = _scale_lambda_full_mesh(bsubv_full, lamscale)
    return blmn_even, sqrt_s * blmn_even, clmn_even, sqrt_s * clmn_even


# ---------------------------------------------------------------------------
# R/Z force kernels (forces.f)
# ---------------------------------------------------------------------------


def mhd_force_kernels(
    *,
    geometry: RealSpaceGeometry,
    jacobian: HalfMeshJacobian,
    fields: MagneticFields,
    s: Array,
    lthreed: bool,
) -> tuple[Array, ...]:
    """The twelve cylindrical R/Z force kernels (VMEC2000 ``forces.f``).

    Inputs are the half-mesh quantities that ``bcovar.f`` leaves behind for
    ``forces``: ``lu = bsq*R12``, ``lv = bsq*tau`` and the "GIJ" B-products
    ``guu/guv/gvv = B^i B^j sqrt(g)`` (all with the axis row zeroed), plus the
    full-mesh even/odd geometry channels.  The radial staggering (forward
    differences/averages/sums along ``js`` with the ``dshalfds = 0.25`` odd-m
    chain-rule terms) is ported verbatim from ``forces.f``.

    For 2D runs (``lthreed = False``) the ``crmn/czmn`` kernels are never
    consumed by the zeta stage of ``tomnsps``; the returned placeholders
    reproduce VMEC's storage-overlay values (``crmn = lv*shalf``, ``czmn =
    lu``, ``crmn_odd = -lamscale*dlambda_dzeta_odd``, ``czmn_odd = lu_odd``)
    so kernel-level parity holds bit-for-bit.

    Returns the tuple ``(force_R_even, force_R_odd, force_R_du_even,
    force_R_du_odd, force_R_dv_even, force_R_dv_odd, force_Z_even,
    force_Z_odd, force_Z_du_even, force_Z_du_odd, force_Z_dv_even,
    force_Z_dv_odd)`` (VMEC ``armn/brmn/crmn/azmn/bzmn/czmn``).
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    dtype = jacobian.sqrt_g.dtype
    ohs = jnp.asarray(1.0 / (s[1] - s[0]), dtype=dtype) if ns >= 2 else jnp.asarray(0.0, dtype=dtype)
    dshalfds = jnp.asarray(_D_SQRT_S_HALF_DS, dtype=dtype)
    pshalf = sqrt_s_half_mesh(s)[:, None, None]
    sqrt_s = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]

    bsq = fields.total_pressure
    sqrt_g = jacobian.sqrt_g
    # bcovar.f "STORE LU * LV COMBINATIONS USED IN FORCES".
    lu_e = _with_axis_zero(bsq * jacobian.r12)
    lv_e = _with_axis_zero(bsq * jacobian.tau)
    guu = _with_axis_zero((fields.bsupu * fields.bsupu) * sqrt_g)
    guv = _with_axis_zero((fields.bsupu * fields.bsupv) * sqrt_g)
    gvv = _with_axis_zero((fields.bsupv * fields.bsupv) * sqrt_g)

    g = geometry
    pr1_0, pr1_1 = g.R_even, g.R_odd
    pz1_1 = g.Z_odd
    pru_0, pru_1 = g.dR_dtheta_even, g.dR_dtheta_odd
    pzu_0, pzu_1 = g.dZ_dtheta_even, g.dZ_dtheta_odd
    prv_0, prv_1 = g.dR_dzeta_even, g.dR_dzeta_odd
    pzv_0, pzv_1 = g.dZ_dzeta_even, g.dZ_dzeta_odd

    guus = guu * pshalf
    guvs = guv * pshalf
    gvvs = gvv * pshalf

    armn_e = ohs * jacobian.zu12 * lu_e
    azmn_e = -ohs * jacobian.ru12 * lu_e
    brmn_e = jacobian.dZ_ds * lu_e
    bzmn_e = -jacobian.dR_ds * lu_e
    bsqr = dshalfds * lu_e / jnp.where(pshalf != 0, pshalf, 1.0)

    armn_o = armn_e * pshalf
    azmn_o = azmn_e * pshalf
    brmn_o = brmn_e * pshalf
    bzmn_o = bzmn_e * pshalf

    guu_i = _forward_average_half(guu)
    gvv_i = _forward_average_half(gvv)
    guus_i = _forward_average_half(guus)
    gvvs_i = _forward_average_half(gvvs)
    guv_i = _forward_average_half(guv)
    guvs_i = _forward_average_half(guvs)
    bsqr_s = _forward_sum_half(bsqr)

    # Even-m planes: radial staggering + metric terms.
    armn_e = _forward_difference_half_with_average(armn_e, lv_e)
    azmn_e = _forward_difference_half(azmn_e)
    brmn_e = _forward_average_half(brmn_e)
    bzmn_e = _forward_average_half(bzmn_e)

    armn_e = armn_e - (gvvs_i * pr1_1 + gvv_i * pr1_0)
    brmn_e = brmn_e + bsqr_s * pz1_1 - (guus_i * pru_1 + guu_i * pru_0)
    bzmn_e = bzmn_e - (bsqr_s * pr1_1 + guus_i * pzu_1 + guu_i * pzu_0)

    # Odd-m planes: forward difference/average + d(sqrt s)/ds terms.
    lv_es = lv_e * pshalf
    lu_o = dshalfds * lu_e
    if ns >= 2:
        armn_o = jnp.concatenate(
            [
                armn_o[1:] - armn_o[:-1] - pzu_0[:-1] * bsqr_s[:-1] + 0.5 * (lv_es[:-1] + lv_es[1:]),
                -armn_o[-1:] - pzu_0[-1:] * bsqr_s[-1:] + 0.5 * lv_es[-1:],
            ],
            axis=0,
        )
        azmn_o = jnp.concatenate(
            [
                azmn_o[1:] - azmn_o[:-1] + pru_0[:-1] * bsqr_s[:-1],
                -azmn_o[-1:] + pru_0[-1:] * bsqr_s[-1:],
            ],
            axis=0,
        )
        brmn_o = _forward_average_half(brmn_o)
        bzmn_o = _forward_average_half(bzmn_o)
        lu_o = _forward_sum_half(lu_o)
    else:
        armn_o = -armn_o - pzu_0 * bsqr_s + 0.5 * lv_es
        azmn_o = -azmn_o + pru_0 * bsqr_s
        brmn_o = 0.5 * brmn_o
        bzmn_o = 0.5 * bzmn_o

    ss = (sqrt_s * sqrt_s).astype(guu_i.dtype)
    guu_s = guu_i * ss
    gvv_s = gvv_i * ss
    armn_o = armn_o - (pzu_1 * lu_o + gvv_s * pr1_1 + gvvs_i * pr1_0)
    azmn_o = azmn_o + pru_1 * lu_o
    brmn_o = brmn_o + pz1_1 * lu_o - (guu_s * pru_1 + guus_i * pru_0)
    bzmn_o = bzmn_o - (pr1_1 * lu_o + guu_s * pzu_1 + guus_i * pzu_0)

    if bool(lthreed):
        brmn_e = brmn_e - (guv_i * prv_0 + guvs_i * prv_1)
        bzmn_e = bzmn_e - (guv_i * pzv_0 + guvs_i * pzv_1)
        crmn_e = guv_i * pru_0 + gvv_i * prv_0 + gvvs_i * prv_1 + guvs_i * pru_1
        czmn_e = guv_i * pzu_0 + gvv_i * pzv_0 + gvvs_i * pzv_1 + guvs_i * pzu_1
        guv_s = guv_i * ss
        brmn_o = brmn_o - (guvs_i * prv_0 + guv_s * prv_1)
        bzmn_o = bzmn_o - (guvs_i * pzv_0 + guv_s * pzv_1)
        crmn_o = guvs_i * pru_0 + gvvs_i * prv_0 + gvv_s * prv_1 + guv_s * pru_1
        czmn_o = guvs_i * pzu_0 + gvvs_i * pzv_0 + gvv_s * pzv_1 + guv_s * pzu_1
    else:
        # 2D storage-overlay placeholders (never consumed by tomnsps).
        lamscale = jnp.asarray(fields.lamscale, dtype=dtype)
        crmn_e = lv_es
        czmn_e = lu_e
        crmn_o = -lamscale * g.dlambda_dzeta_odd
        czmn_o = lu_o

    return (
        armn_e, armn_o, brmn_e, brmn_o, crmn_e, crmn_o,
        azmn_e, azmn_o, bzmn_e, bzmn_o, czmn_e, czmn_o,
    )


# ---------------------------------------------------------------------------
# Spectral-condensation constraint force (alias.f)
# ---------------------------------------------------------------------------


def alias_constraint_force(
    ztemp: Array,
    *,
    trig: TrigTables,
    mpol: int,
    ntor: int,
    signgs: int,
    tcon: Array,
    lasym: bool,
) -> Array:
    """De-aliased constraint force ``gcon`` from ``ztemp`` (VMEC2000 ``alias.f``).

    Forward-transforms ``ztemp`` on the reduced theta interval, keeps only the
    ``m in [1, mpol-2]`` band (via the :func:`faccon` weights, which vanish at
    ``m = 0`` and ``m = mpol-1``), scales by ``tcon(js)``, and inverse
    transforms.  For ``lasym=True`` the ``alias_par`` reflection algebra is
    used: symmetric/antisymmetric zeta blocks are built from ``ztemp`` and its
    stellarator reflection, and the second theta half-interval is filled by
    ``gcon(2*pi - theta, -zeta) = -gcon_s + gcon_a``.

    ``ztemp`` has shape ``(ns, ntheta3, nzeta)``; the returned ``gcon``
    matches it.
    """
    ztemp = jnp.asarray(ztemp)
    ns, ntheta3, nzeta = ztemp.shape
    if int(ntheta3) != int(trig.ntheta3):
        raise ValueError("ztemp theta size must match trig.ntheta3")
    if int(nzeta) != int(trig.cosnv.shape[0]):
        raise ValueError("ztemp zeta size must match trig tables")

    mpol = int(mpol)
    ntor = int(ntor)
    fac = jnp.asarray(faccon(mpol, signgs), dtype=ztemp.dtype)
    tcon = jnp.asarray(tcon, dtype=ztemp.dtype)

    n_theta2 = int(trig.ntheta2)
    z_half = ztemp[:, :n_theta2, :]

    cosmui = jnp.asarray(trig.cosmui[:n_theta2, :mpol])
    sinmui = jnp.asarray(trig.sinmui[:n_theta2, :mpol])
    cosmu = jnp.asarray(trig.cosmu[:n_theta2, :mpol])
    sinmu = jnp.asarray(trig.sinmu[:n_theta2, :mpol])
    cosnv = jnp.asarray(trig.cosnv[:, : ntor + 1])
    sinnv = jnp.asarray(trig.sinnv[:, : ntor + 1])

    w_cos = _einsum("sik,im->smk", z_half, cosmui)
    w_sin = _einsum("sik,im->smk", z_half, sinmui)

    cosmu_fac = cosmu * fac[None, :]
    sinmu_fac = sinmu * fac[None, :]

    if not bool(lasym):
        gcs = tcon[:, None, None] * _einsum("smk,kn->smn", w_cos, sinnv)
        gsc = tcon[:, None, None] * _einsum("smk,kn->smn", w_sin, cosnv)
        work_cs = _einsum("smn,kn->smk", gcs, sinnv)
        work_sc = _einsum("smn,kn->smk", gsc, cosnv)
        gcon_half = _einsum("smk,im->sik", work_cs, cosmu_fac) + _einsum("smk,im->sik", work_sc, sinmu_fac)
        gcon = jnp.zeros((ns, ntheta3, nzeta), dtype=ztemp.dtype)
        return gcon.at[:, :n_theta2, :].set(gcon_half)

    # lasym path (alias_par): reflected ztemp restricted to theta in [0, pi].
    n_theta1 = int(trig.ntheta1)
    if ntheta3 != n_theta1:
        raise ValueError("lasym=True requires ntheta3 == ntheta1")
    i = np.arange(n_theta2, dtype=int)
    i_reflected = np.where(i == 0, 0, n_theta1 - i)
    k_reflected = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    z_reflected = ztemp[:, i_reflected, :][:, :, k_reflected]

    w_cos_r = _einsum("sik,im->smk", z_reflected, cosmui)
    w_sin_r = _einsum("sik,im->smk", z_reflected, sinmui)

    half = 0.5 * tcon[:, None, None]
    gcs = half * _einsum("smk,kn->smn", w_cos - w_cos_r, sinnv)
    gsc = half * _einsum("smk,kn->smn", w_sin - w_sin_r, cosnv)
    gss = half * _einsum("smk,kn->smn", w_sin + w_sin_r, sinnv)
    gcc = half * _einsum("smk,kn->smn", w_cos + w_cos_r, cosnv)

    work_cs = _einsum("smn,kn->smk", gcs, sinnv)
    work_sc = _einsum("smn,kn->smk", gsc, cosnv)
    work_cc = _einsum("smn,kn->smk", gcc, cosnv)
    work_ss = _einsum("smn,kn->smk", gss, sinnv)

    gcon_sym_half = _einsum("smk,im->sik", work_cs, cosmu_fac) + _einsum("smk,im->sik", work_sc, sinmu_fac)
    gcon_asym_half = _einsum("smk,im->sik", work_cc, cosmu_fac) + _einsum("smk,im->sik", work_ss, sinmu_fac)

    gcon = jnp.zeros((ns, ntheta3, nzeta), dtype=ztemp.dtype)
    gcon = gcon.at[:, :n_theta2, :].set(gcon_sym_half + gcon_asym_half)
    n_second = n_theta1 - n_theta2
    if n_second > 0:
        i2 = np.arange(n_theta2, n_theta1, dtype=int)
        i2_reflected = (n_theta1 - i2).astype(int)
        gcon_sym_ref = gcon_sym_half[:, i2_reflected, :][:, :, k_reflected]
        gcon_asym_ref = gcon_asym_half[:, i2_reflected, :][:, :, k_reflected]
        gcon = gcon.at[:, n_theta2:, :].set(-gcon_sym_ref + gcon_asym_ref)
    return gcon


def _internal_odd_channel(
    odd_m1: Array,
    odd_rest: Array,
    s: Array,
    *,
    eps: float = 1e-14,
) -> Array:
    """Internal odd channel from split physical odd-m fields (totzsps rules).

    VMEC2000: ``vmec_params.f`` ``jmin1`` — the internal odd field is the
    physical one divided by ``sqrt(s)``, with the axis row extrapolated from
    the first interior surface for ``m = 1`` (``jmin1(1) = 1``) and zeroed for
    ``m >= 3`` (``jmin1(m>=2) = 2``).
    """
    s = jnp.asarray(s)
    sqrt_s = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    mask = (sqrt_s > eps).astype(jnp.asarray(odd_m1).dtype)
    safe = jnp.where(sqrt_s > eps, sqrt_s, 1.0)

    m1_internal = jnp.asarray(odd_m1) * mask / safe
    rest_internal = jnp.asarray(odd_rest) * mask / safe
    if m1_internal.shape[0] >= 2:
        m1_internal = m1_internal.at[0].set(m1_internal[1])
        rest_internal = rest_internal.at[0].set(jnp.zeros_like(rest_internal[0]))
    return m1_internal + rest_internal


def constraint_force(
    *,
    R_cos: Array,
    R_sin: Array,
    Z_cos: Array,
    Z_sin: Array,
    geometry: RealSpaceGeometry,
    modes: ModeTable,
    trig: TrigTables,
    s: Array,
    tcon: Array,
    signgs: int,
    rcon0: Array | None = None,
    zcon0: Array | None = None,
    use_fft: bool = False,
) -> tuple[Array, Array, Array, Array, Array]:
    """Spectral-condensation constraint force pipeline (funct3d.f + alias.f).

    1. Synthesize ``rcon/zcon``: the real-space image of the ``xmpq(m,1) =
       m*(m-1)``-weighted R/Z coefficients (physical-signed internal
       normalization, same input convention as
       :func:`vmex.core.geometry.real_space_geometry`), assembled as
       ``X_even + sqrt(s) * X_odd_internal`` with the ``jmin1`` axis rules.
    2. Baselines: ``rcon0/zcon0`` default to the edge profile scaled by ``s``
       (``funct3d.f`` fixed-boundary initialization ``rcon0 = s * rcon(ns)``);
       pass persisted arrays to keep VMEC's caching/free-boundary 0.9-ramp
       semantics (that ramp lives in the solver).
    3. ``ztemp = (rcon - rcon0)*ru0 + (zcon - zcon0)*zu0`` with the physical
       full-mesh theta derivatives ``ru0/zu0``.
    4. ``gcon`` via :func:`alias_constraint_force` with ``tcon(js)`` scaling.

    Returns ``(gcon, rcon, zcon, rcon0, zcon0)``.
    """
    s = jnp.asarray(s)
    R_cos = jnp.asarray(R_cos)
    R_sin = jnp.asarray(R_sin)
    Z_cos = jnp.asarray(Z_cos)
    Z_sin = jnp.asarray(Z_sin)
    dtype = R_cos.dtype

    m = np.asarray(modes.m, dtype=int)
    xmpq1 = jnp.asarray((m * (m - 1)).astype(float), dtype=dtype)
    mask_even = jnp.asarray((m % 2) == 0, dtype=dtype)
    mask_m1 = jnp.asarray(m == 1, dtype=dtype)
    mask_odd_rest = jnp.asarray((m % 2 == 1) & (m != 1), dtype=dtype)

    coeff_cos = jnp.stack(
        [
            R_cos * xmpq1 * mask_even,
            Z_cos * xmpq1 * mask_even,
            R_cos * xmpq1 * mask_m1,
            R_cos * xmpq1 * mask_odd_rest,
            Z_cos * xmpq1 * mask_m1,
            Z_cos * xmpq1 * mask_odd_rest,
        ],
        axis=0,
    )
    coeff_sin = jnp.stack(
        [
            R_sin * xmpq1 * mask_even,
            Z_sin * xmpq1 * mask_even,
            R_sin * xmpq1 * mask_m1,
            R_sin * xmpq1 * mask_odd_rest,
            Z_sin * xmpq1 * mask_m1,
            Z_sin * xmpq1 * mask_odd_rest,
        ],
        axis=0,
    )
    synthesize = _fourier_to_real_fft if use_fft else fourier_to_real
    (value,) = synthesize(
        coeff_cos,
        coeff_sin,
        modes=modes,
        trig=trig,
        derivatives=("value",),
        internal_coeffs=True,
        odd_m_sqrt_s=False,
    )
    rcon_even, zcon_even, rcon_m1, rcon_rest, zcon_m1, zcon_rest = value

    sqrt_s = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    rcon = rcon_even + sqrt_s * _internal_odd_channel(rcon_m1, rcon_rest, s)
    zcon = zcon_even + sqrt_s * _internal_odd_channel(zcon_m1, zcon_rest, s)

    if rcon0 is None:
        rcon0 = (s[:, None, None] * rcon[-1][None, :, :]).astype(rcon.dtype)
    else:
        rcon0 = jnp.asarray(rcon0, dtype=rcon.dtype)
    if zcon0 is None:
        zcon0 = (s[:, None, None] * zcon[-1][None, :, :]).astype(zcon.dtype)
    else:
        zcon0 = jnp.asarray(zcon0, dtype=zcon.dtype)

    ru0, zu0 = geometry.theta_derivatives_full(s)
    ztemp = (rcon - rcon0) * ru0 + (zcon - zcon0) * zu0

    mpol = int(np.max(m)) + 1
    ntor = int(np.max(np.abs(np.asarray(modes.n, dtype=int)))) if modes.mnmax else 0
    gcon = alias_constraint_force(
        ztemp,
        trig=trig,
        mpol=mpol,
        ntor=ntor,
        signgs=signgs,
        tcon=tcon,
        lasym=bool(trig.lasym),
    )
    return gcon, rcon, zcon, rcon0, zcon0


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------


def mhd_forces(
    *,
    geometry: RealSpaceGeometry,
    jacobian: HalfMeshJacobian,
    metrics: MetricElements,
    fields: MagneticFields,
    R_cos: Array,
    R_sin: Array,
    Z_cos: Array,
    Z_sin: Array,
    modes: ModeTable,
    trig: TrigTables,
    s: Array,
    phipf: Array,
    tcon: Array,
    signgs: int,
    rcon0: Array | None = None,
    zcon0: Array | None = None,
    use_fft: bool = False,
) -> RealSpaceForces:
    """Assemble all real-space force kernels (funct3d.f force stage).

    Combines :func:`mhd_force_kernels`, :func:`lambda_force_kernels` and
    :func:`constraint_force`, applying the ``forces.f`` constraint add-back

        ``brmn += (rcon - rcon0)*gcon``,  ``bzmn += (zcon - zcon0)*gcon``

    (odd planes with the extra ``sqrt(s)``), and forming the constraint
    projection kernels ``arcon = ru0*gcon`` / ``azcon = zu0*gcon`` consumed by
    ``tomnsps`` with the ``xmpq`` weights.

    The R/Z/lambda coefficients are the physical-signed VMEC-internal ones
    used for :func:`vmex.core.geometry.real_space_geometry` (the m=1
    constraint of ``residue.f90`` already undone — see
    :func:`vmex.core.residuals.m1_constrained_to_physical`).  Passing
    ``tcon = 0`` disables the constraint force without retracing.
    """
    lthreed = bool(np.any(np.asarray(modes.n) != 0))
    (
        armn_e, armn_o, brmn_e, brmn_o, crmn_e, crmn_o,
        azmn_e, azmn_o, bzmn_e, bzmn_o, czmn_e, czmn_o,
    ) = mhd_force_kernels(geometry=geometry, jacobian=jacobian, fields=fields, s=s, lthreed=lthreed)
    blmn_e, blmn_o, clmn_e, clmn_o = lambda_force_kernels(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields, s=s, phipf=phipf
    )
    gcon, rcon, zcon, rcon0, zcon0 = constraint_force(
        R_cos=R_cos,
        R_sin=R_sin,
        Z_cos=Z_cos,
        Z_sin=Z_sin,
        geometry=geometry,
        modes=modes,
        trig=trig,
        s=s,
        tcon=tcon,
        signgs=signgs,
        rcon0=rcon0,
        zcon0=zcon0,
        use_fft=use_fft,
    )

    s = jnp.asarray(s)
    sqrt_s = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    rcon_force = (rcon - rcon0) * gcon
    zcon_force = (zcon - zcon0) * gcon
    ru0, zu0 = geometry.theta_derivatives_full(s)
    arcon_e = ru0 * gcon
    azcon_e = zu0 * gcon

    return RealSpaceForces(
        force_R_even=armn_e,
        force_R_odd=armn_o,
        force_R_du_even=brmn_e + rcon_force,
        force_R_du_odd=brmn_o + rcon_force * sqrt_s,
        force_R_dv_even=crmn_e,
        force_R_dv_odd=crmn_o,
        force_Z_even=azmn_e,
        force_Z_odd=azmn_o,
        force_Z_du_even=bzmn_e + zcon_force,
        force_Z_du_odd=bzmn_o + zcon_force * sqrt_s,
        force_Z_dv_even=czmn_e,
        force_Z_dv_odd=czmn_o,
        force_lambda_du_even=blmn_e,
        force_lambda_du_odd=blmn_o,
        force_lambda_dv_even=clmn_e,
        force_lambda_dv_odd=clmn_o,
        constraint_R_even=arcon_e,
        constraint_R_odd=arcon_e * sqrt_s,
        constraint_Z_even=azcon_e,
        constraint_Z_odd=azcon_e * sqrt_s,
        gcon=gcon,
        rcon0=rcon0,
        zcon0=zcon0,
    )


#: symforce.f dominant-symmetry table: kernels whose symmetric part pairs with
#: the reflection-even combination (True) vs the reversed ones (False).
_SYMFORCE_REFLECT_EVEN: dict[str, bool] = {
    "force_R": True,  # ars
    "force_R_du": False,  # brs
    "force_R_dv": False,  # crs
    "force_Z": False,  # azs
    "force_Z_du": True,  # bzs
    "force_Z_dv": True,  # czs
    "force_lambda_du": True,  # bls
    "force_lambda_dv": True,  # cls
    "constraint_R": True,  # rcs
    "constraint_Z": False,  # zcs
}


def symmetrize_forces(
    forces: RealSpaceForces,
    *,
    trig: TrigTables,
) -> tuple[RealSpaceForces, RealSpaceForces]:
    """Symmetric/antisymmetric decomposition of every kernel (``symforce.f``).

    For ``lasym=True`` runs each kernel is split under the stellarator
    reflection ``(theta, zeta) -> (2*pi - theta, -zeta)`` before the reduced
    interval projections; the dominant symmetry differs per kernel
    (``_SYMFORCE_REFLECT_EVEN``, matching the ``symforce.f`` kinds).  Returns
    ``(forces_symmetric, forces_antisymmetric)``; the ``gcon/rcon0/zcon0``
    diagnostics are carried on the symmetric output only.
    """
    sym_fields: dict[str, Any] = {}
    asym_fields: dict[str, Any] = {}
    for channel, reflect_even in _SYMFORCE_REFLECT_EVEN.items():
        for parity in ("even", "odd"):
            name = f"{channel}_{parity}"
            sym, asym = symforce_split(getattr(forces, name), trig=trig, reflect_even=reflect_even)
            sym_fields[name] = sym
            asym_fields[name] = asym
    return (
        replace(forces, **sym_fields),
        RealSpaceForces(**asym_fields),
    )


def spectral_mhd_forces(
    forces: RealSpaceForces,
    *,
    mpol: int,
    ntor: int,
    trig: TrigTables,
    include_edge: bool = False,
) -> SpectralForce:
    """Project the real-space force kernels onto the Fourier basis.

    VMEC2000: ``funct3d.f`` — ``tomnsps`` for stellarator-symmetric runs;
    for ``lasym=True`` the kernels are first decomposed with
    :func:`symmetrize_forces` (``symforce.f``) and the antisymmetric parts go
    through ``tomnspa``.  Returns a single
    :class:`vmex.core.transforms.SpectralForce` carrying both block
    families (antisymmetric blocks are ``None`` for symmetric runs).
    """

    def kernel_kwargs(f: RealSpaceForces) -> dict[str, Array]:
        return {
            f"{name}_{parity}": getattr(f, f"{name}_{parity}")
            for name in _SYMFORCE_REFLECT_EVEN
            for parity in ("even", "odd")
        }

    if not bool(trig.lasym):
        return tomnsps(
            **kernel_kwargs(forces), mpol=mpol, ntor=ntor, trig=trig, include_edge=include_edge
        )

    forces_sym, forces_asym = symmetrize_forces(forces, trig=trig)
    out_sym = tomnsps(
        **kernel_kwargs(forces_sym), mpol=mpol, ntor=ntor, trig=trig, include_edge=include_edge
    )
    out_asym = tomnspa(
        **kernel_kwargs(forces_asym), mpol=mpol, ntor=ntor, trig=trig, include_edge=include_edge
    )
    return replace(
        out_sym,
        force_R_sc=out_asym.force_R_sc,
        force_R_cs=out_asym.force_R_cs,
        force_Z_cc=out_asym.force_Z_cc,
        force_Z_ss=out_asym.force_Z_ss,
        force_lambda_cc=out_asym.force_lambda_cc,
        force_lambda_ss=out_asym.force_lambda_ss,
    )


def apply_m1_force_balance(
    force: SpectralForce,
    *,
    equif: Array,
    factor_R: Array,
    factor_Z: Array,
) -> SpectralForce:
    """Apply VMEC2000's ``LFORBAL`` ``(m=1,n=0)`` force replacement.

    ``tomnsp_mod.f`` replaces only the stellarator-symmetric ``frcc`` and
    ``fzsc`` blocks, including in ``LASYM`` runs.  With ``factor_R/Z`` equal
    to the pre-halving ``rzu_fac/rru_fac`` values from ``bcovar.f``, the
    interior full-mesh rows are

    ``work = frcc/factor_R - fzsc/factor_Z``

    ``frcc = (factor_R/2)*(equif + work)``

    ``fzsc = (factor_Z/2)*(equif - work)``.

    Axis and edge rows, all other modes, and all asymmetric blocks are
    unchanged.
    """
    frcc = force.force_R_cc
    fzsc = force.force_Z_sc
    if frcc is None or fzsc is None:
        return force
    frcc = jnp.asarray(frcc)
    fzsc = jnp.asarray(fzsc)
    ns, mpol, _ = frcc.shape
    if int(ns) < 3 or int(mpol) < 2:
        return force

    factor_R = jnp.asarray(factor_R, dtype=frcc.dtype)
    factor_Z = jnp.asarray(factor_Z, dtype=fzsc.dtype)
    equif = jnp.asarray(equif, dtype=frcc.dtype)
    old_R = frcc[1:-1, 1, 0]
    old_Z = fzsc[1:-1, 1, 0]
    work = old_R / factor_R[1:-1] - old_Z / factor_Z[1:-1]
    new_R = 0.5 * factor_R[1:-1] * (equif[1:-1] + work)
    new_Z = 0.5 * factor_Z[1:-1] * (equif[1:-1] - work)
    frcc = frcc.at[1:-1, 1, 0].set(new_R)
    fzsc = fzsc.at[1:-1, 1, 0].set(new_Z)
    return replace(force, force_R_cc=frcc, force_Z_sc=fzsc)
