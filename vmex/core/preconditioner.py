"""1D (radial) preconditioner for the VMEC R/Z/lambda force equations.

VMEC2000 counterparts
---------------------
- ``Sources/General/precondn.f``  — :func:`precondn`: radial tridiagonal matrix
  *elements* for the R and Z force equations from ``ptau``-type flux-surface
  integrals.  VMEC calls it twice per refresh: once with the Z geometry
  derivatives (producing ``arm/ard/brm/brd/crd`` for the R force) and once
  with the R geometry derivatives (producing ``azm/azd/bzm/bzd`` for the Z
  force).
- ``Sources/General/lamcal.f90``  — :func:`lamcal`: the *diagonal* lambda-force
  preconditioner ``faclam``.
- ``Sources/General/scalfor.f``   — :func:`scalfor_matrices` (assembly of the
  tridiagonal system ``ax/bx/dx`` with the ``m**2`` and ``(n*nfp)**2`` mode
  weights, ``edge_pedestal`` and the ZC(0,0)(ns) stabilization) and
  :func:`scalfor` (application: solve the system in place of the force).
- ``Sources/General/tridslv`` (``serial_tridslv`` in ``scalfor.f``) —
  :func:`tridiagonal_solve`: Thomas algorithm, vectorized over all (m,n)
  columns and stacked right-hand-side fields simultaneously.

Equations (VMEC2000 notation)
-----------------------------
``precondn`` builds, for the linearized radial force operator of the energy
functional, the flux-surface integrals over the half mesh (``js = 2..ns``)::

    ptau = pfactor * r12**2 * bsq * wint / gsqrt          (pfactor = -4)

    ax = < ptau * (xu12/hs ± t23)**2 >    (poloidal-derivative couplings)
    bx = < ptau * t1 * t2 >               (radial-derivative couplings)
    cx = < 0.25 * pfactor * (bsupv**2) * gsqrt >   (toroidal couplings)

which are then accumulated onto the full mesh as off-diagonal (``axm/bxm``,
between surfaces) and diagonal (``axd/bxd``, per surface) elements, each with
an even-m column and an odd-m column (odd-m carries the ``sm/sp`` internal
``sqrt(s)`` scalings from ``profil1d``).

``scalfor`` forms, per mode (m,n), the radial tridiagonal system::

    ax(js) = -(axm(js+1) + bxm(js+1) * m**2)          couples X(js+1)
    bx(js) = -(axm(js)   + bxm(js)   * m**2)          couples X(js-1)
    dx(js) = -(axd(js) + bxd(js)*m**2 + cx(js)*(n*nfp)**2)

with the even/odd column selected by ``mod(m,2)``, and solves
``[bx, dx, ax] . X = force`` by the Thomas algorithm (``tridslv``).

``lamcal`` builds the diagonal lambda scaling::

    blam = < guu / gsqrt >,  clam = < gvv / gsqrt >,  dlam = < guv / gsqrt >
    faclam = p_factor * sqrt(s)**power /
             (blam*(n*nfp)**2 + sign(dlam,blam)*2*m*(n*nfp) + clam*m**2)

with ``power = min(m**2/16**2, 8)`` (the ``sqrt(s)`` damping only bites for
``m > 16``) and a special ``(m,n)=(0,0)`` slot that preconditions the
chip/iota channel (VMEC stores ``chip``-like data in ``lmns(0,0)``).

Index conventions (parity-critical)
-----------------------------------
- Radial arrays are 0-based here: full-mesh index ``js = 0..ns-1`` maps to
  VMEC's ``1..ns``; half-mesh arrays are stored *without* VMEC's dummy axis
  row, so row ``j`` of a half-mesh input is VMEC's ``js = j+2`` half point
  (between full points ``j`` and ``j+1``).  Exception: :func:`lamcal` takes
  the half-mesh metrics *with* the dummy axis row (``ns`` rows), exactly as
  VMEC's ``guu/gvv/guv(1:ns)`` storage, and internally copies row 1 into
  row 0 (``blam(1) = blam(2)`` in ``lamcal.f90``).
- ``jmax``: number of evolved radial rows.  Fixed boundary: ``jmax = ns-1``
  (VMEC ``jmax = ns1`` when ``ivac < 1``); the edge row keeps the raw force.
  Free boundary: ``jmax = ns`` and the ``edge_pedestal``/ZC(0,0)
  stabilization of ``scalfor.f`` activates (it only applies when
  ``jmax >= ns``).
- ``jmin`` (start row of the tridiagonal solve, VMEC ``jmin2``): ``0`` for
  ``m = 0`` and ``1`` for ``m >= 1``; row 0 of the ``m >= 1`` solution is
  zeroed.  For ``m = 1`` the sub-diagonal coupling of row 1 to the axis row
  is folded into the diagonal (``dx(2,m=1) += bx(2,m=1)``, reflecting VMEC's
  ``X(1,m=1) = X(2,m=1)`` axis convention).
- lambda start index (VMEC ``jlam(m) = 2``): row 0 of ``faclam`` is zero for
  every mode except ``(0,0)``.

All functions are pure ``jax.numpy`` (jit-friendly, no host round-trips);
resolution parameters (``ns/mpol/ntor/nfp/...``) are Python ints/bools and
must be static under ``jax.jit``.  The numerics are ported verbatim from the
parity-proven legacy module :mod:`vmex.preconditioner_1d_jax`; equivalence
is enforced in ``tests/test_preconditioner_ab.py``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

import jax.numpy as jnp

from solvax import (
    tridiagonal_solve as _sx_tridiagonal_solve,
    tridiagonal_solve_checked as _sx_tridiagonal_solve_checked,
)

__all__ = [
    "RadialPreconditionerCoefficients",
    "TridiagonalMatrices",
    "angular_integration_weights",
    "precondn",
    "lamcal",
    "scalfor_matrices",
    "scalfor",
    "tridiagonal_solve",
]

Array = Any

#: VMEC2000 ``scalfor.f``: relative stiffening of the edge diagonal.
EDGE_PEDESTAL = 0.05

#: VMEC2000 ``scalfor.f`` (SPH): ZC(0,0)(ns) stabilization base factor.
ZC00_EDGE_FACTOR = 0.25


# ---------------------------------------------------------------------------
# Static helpers (trace-time constants)
# ---------------------------------------------------------------------------


def angular_integration_weights(*, ntheta: int, nzeta: int, lasym: bool) -> np.ndarray:
    """Poloidal integration weights for the flux-surface averages, ``(ntheta_eff,)``.

    VMEC2000: ``wint`` from ``profil3d.f`` (constant in zeta), i.e. ``dnorm3``
    with half weights at the ``theta = 0, pi`` endpoints of the reduced grid
    for stellarator-symmetric runs; uniform ``1/(nzeta*ntheta1)`` on the full
    grid when ``lasym``.  Matches ``TrigTables.wint[:, 0]`` of
    :mod:`vmex.core.fourier`.  Static (NumPy) by construction.
    """
    ntheta_even = 2 * (int(ntheta) // 2)
    ntheta_reduced = ntheta_even // 2 + 1
    ntheta_eff = ntheta_even if lasym else ntheta_reduced
    if ntheta_eff <= 0:
        return np.zeros((0,), dtype=np.float64)
    if lasym:
        dnorm3 = 1.0 / (float(nzeta) * float(ntheta_even))
    else:
        dnorm3 = 1.0 / (float(nzeta) * float(ntheta_reduced - 1))
    weights = np.full((ntheta_eff,), dnorm3, dtype=np.float64)
    if not lasym:
        weights[0] *= 0.5
        weights[-1] *= 0.5
    return weights


def _sqrt_s_profiles(ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Full/half-mesh ``sqrt(s)`` profiles (VMEC ``profil1d``: ``sqrts``, ``shalf``).

    ``sqrt_s_full[j] = sqrt(j/(ns-1))`` (``ns`` values) and
    ``sqrt_s_half[j] = sqrt((j+0.5)/(ns-1))`` (``ns-1`` values, no axis dummy).
    """
    ns = int(ns)
    if ns <= 1:
        return np.zeros((max(ns, 0),), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    full = np.sqrt(np.linspace(0.0, 1.0, ns, dtype=np.float64))
    half = np.sqrt((np.arange(ns - 1, dtype=np.float64) + 0.5) / float(ns - 1))
    return full, half


def _odd_m_half_mesh_scalings(ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Odd-m half-mesh interpolation ratios (VMEC ``profil1d``: ``sm``, ``sp``).

    ``sm[j] = shalf[j]/sqrts[j+1]`` (half point over its *outer* full point)
    and ``sp[j] = shalf[j]/sqrts[j]`` (over its *inner* full point), with the
    axis value ``sp[0] = sm[0]``.  Shapes ``(ns-1,)``.
    """
    sqrt_s_full, sqrt_s_half = _sqrt_s_profiles(ns)
    if int(ns) < 2:
        z = np.zeros((0,), dtype=np.float64)
        return z, z
    outer = np.where(sqrt_s_full[1:] != 0.0, sqrt_s_full[1:], 1.0)
    inner = np.where(sqrt_s_full[:-1] != 0.0, sqrt_s_full[:-1], 1.0)
    sm = sqrt_s_half / outer
    sp = sqrt_s_half / inner
    sp[0] = sm[0]
    return sm, sp


def _weight_bcast(angular_weight: Array) -> Array:
    """Broadcast ``(ntheta,)`` or ``(ntheta, nzeta)`` weights over ``(ns, ntheta, nzeta)``."""
    w = jnp.asarray(angular_weight)
    if w.ndim == 1:
        return w[None, :, None]
    if w.ndim == 2:
        return w[None, :, :]
    raise ValueError(f"angular_weight must be 1-D or 2-D, got shape {w.shape}")


# ---------------------------------------------------------------------------
# precondn (precondn.f)
# ---------------------------------------------------------------------------


class RadialPreconditionerCoefficients(NamedTuple):
    """Radial matrix elements from :func:`precondn` (VMEC2000 ``precondn.f``).

    Columns index the m-parity: ``[:, 0]`` even m, ``[:, 1]`` odd m (odd-m
    carries the internal ``sqrt(s)`` scalings ``sm/sp``).  At the R-force call
    site these are VMEC's ``arm/ard/brm/brd/crd``; at the Z-force call site
    ``azm/azd/bzm/bzd`` (``cx`` is shared).
    """

    axm: Array  #: (ns-1, 2) off-diagonal poloidal coupling (half-mesh rows).
    axd: Array  #: (ns, 2)   diagonal poloidal coupling (full-mesh accumulated).
    bxm: Array  #: (ns-1, 2) off-diagonal radial coupling (half-mesh rows).
    bxd: Array  #: (ns, 2)   diagonal radial coupling (full-mesh accumulated).
    cx: Array  #: (ns,)     toroidal coupling (full-mesh accumulated).


def precondn(
    *,
    dxds_half: Array,
    dxdu_half: Array,
    dxdu_even_full: Array,
    dxdu_odd_full: Array,
    x_odd_full: Array,
    r12_half: Array,
    bsq_half: Array,
    bsupv_half: Array,
    sqrt_g_half: Array,
    angular_weight: Array,
    delta_s: Array,
    ns: int,
) -> RadialPreconditionerCoefficients:
    """Radial tridiagonal matrix elements for one force family (``precondn.f``).

    For the **R force** pass the **Z** geometry derivatives (VMEC:
    ``precondn(..., zs, zu12, zu(even), zu(odd), z1(odd), ...)`` producing
    ``arm/ard/brm/brd/crd``); for the **Z force** pass the **R** derivatives
    (producing ``azm/azd/bzm/bzd``).

    Parameters
    ----------
    dxds_half:
        ``(ns-1, ntheta, nzeta)`` radial derivative of Z (or R) on the half
        mesh, no axis dummy row (VMEC ``zs``/``rs``).
    dxdu_half:
        ``(ns-1, ntheta, nzeta)`` poloidal derivative on the half mesh
        (VMEC ``zu12``/``ru12``).
    dxdu_even_full, dxdu_odd_full:
        ``(ns, ntheta, nzeta)`` even-m / odd-m (internal) poloidal-derivative
        channels on the full mesh (VMEC ``zu(:,0)``/``zu(:,1)``).
    x_odd_full:
        ``(ns, ntheta, nzeta)`` odd-m channel of Z (or R) itself
        (VMEC ``z1(:,1)``).
    r12_half, bsq_half, bsupv_half, sqrt_g_half:
        ``(ns-1, ntheta, nzeta)`` half-mesh ``R``, total pressure
        ``|B|**2/2 + p`` (VMEC ``bsq``), contravariant ``B^v`` and Jacobian
        ``gsqrt``, no axis dummy row.
    angular_weight:
        ``(ntheta,)`` or ``(ntheta, nzeta)`` integration weights
        (:func:`angular_integration_weights`; VMEC ``wint``).
    delta_s:
        Radial mesh spacing ``hs = 1/(ns-1)`` (VMEC ``hs``; ``ohs = 1/hs``).
    ns:
        Number of full-mesh surfaces (static).
    """
    dxds = jnp.asarray(dxds_half)
    dxdu12 = jnp.asarray(dxdu_half)
    dxdu_even = jnp.asarray(dxdu_even_full)
    dxdu_odd = jnp.asarray(dxdu_odd_full)
    x_odd = jnp.asarray(x_odd_full)
    r12 = jnp.asarray(r12_half)
    bsq = jnp.asarray(bsq_half)
    bsupv = jnp.asarray(bsupv_half)
    sqrt_g = jnp.asarray(sqrt_g_half)

    ns = int(ns)
    ns_half = int(dxds.shape[0])
    dtype = dxds.dtype
    if ns_half <= 0:
        z2 = jnp.zeros((0, 2), dtype=dtype)
        z1 = jnp.zeros((0,), dtype=dtype)
        return RadialPreconditionerCoefficients(z2, z2, z2, z2, z1)

    pfactor = jnp.asarray(-4.0, dtype=dtype)  # VMEC: -4*r0scale**2, r0scale=1
    delta_s = jnp.asarray(delta_s, dtype=dtype)
    sqrt_g_safe = jnp.where(sqrt_g != 0.0, sqrt_g, 1.0)

    _, sqrt_s_half = _sqrt_s_profiles(ns)
    sm, sp = _odd_m_half_mesh_scalings(ns)
    sqrt_s_half = jnp.asarray(np.where(sqrt_s_half != 0.0, sqrt_s_half, 1.0), dtype=dtype)
    sm = jnp.asarray(sm, dtype=dtype)
    sp = jnp.asarray(sp, dtype=dtype)

    sh = sqrt_s_half[:, None, None]
    w3 = _weight_bcast(angular_weight)

    # VMEC precondn: ptau = pfactor*r12**2*bsq*wint/gsqrt on the half mesh.
    ptau = pfactor * r12 * r12 * bsq / sqrt_g_safe * w3

    # Poloidal-derivative couplings (ax integrals).
    t1 = dxdu12 / delta_s  # ohs*xu12
    dxdu_even_outer = dxdu_even[1 : ns_half + 1]
    dxdu_even_inner = dxdu_even[:ns_half]
    dxdu_odd_outer = dxdu_odd[1 : ns_half + 1]
    dxdu_odd_inner = dxdu_odd[:ns_half]
    t2 = 0.25 * (dxdu_even_outer / sh + dxdu_odd_outer) / sh
    t3 = 0.25 * (dxdu_even_inner / sh + dxdu_odd_inner) / sh
    ax0 = jnp.sum(ptau * (t1 * t1), axis=(1, 2))
    ax1 = jnp.sum(ptau * (t1 + t2) * (-t1 + t3), axis=(1, 2))
    ax2 = jnp.sum(ptau * (t1 + t2) * (t1 + t2), axis=(1, 2))
    ax3 = jnp.sum(ptau * (-t1 + t3) * (-t1 + t3), axis=(1, 2))

    # Radial-derivative couplings (bx integrals).
    x_odd_outer = x_odd[1 : ns_half + 1]
    x_odd_inner = x_odd[:ns_half]
    t1b = 0.5 * (dxds + 0.5 / sh * x_odd_outer)
    t2b = 0.5 * (dxds + 0.5 / sh * x_odd_inner)
    bx0 = jnp.sum(ptau * t1b * t2b, axis=(1, 2))
    bx1 = jnp.sum(ptau * t1b * t1b, axis=(1, 2))
    bx2 = jnp.sum(ptau * t2b * t2b, axis=(1, 2))

    # Toroidal coupling (cx integral): <0.25*pfactor*(B^v)**2*gsqrt>.
    cx_half = jnp.sum(0.25 * pfactor * (bsupv * bsupv) * sqrt_g * w3, axis=(1, 2))

    # Off-diagonal (between-surface) elements, half-mesh rows.
    axm = jnp.stack([-ax0, ax1 * sm * sp], axis=1)
    bxm = jnp.stack([bx0, bx0 * sm * sp], axis=1)

    # Diagonal accumulation onto the full mesh (VMEC precondn loop over jf):
    # inner contribution from half point jf-1, outer from half point jf.
    zero = jnp.zeros((1,), dtype=dtype)

    def _inner(values: Array) -> Array:
        return jnp.concatenate([zero, values], axis=0)[:ns]

    def _outer(values: Array) -> Array:
        return jnp.concatenate([values, zero], axis=0)[:ns]

    axd = jnp.stack(
        [
            _inner(ax0) + _outer(ax0),
            _inner(ax2 * (sm * sm)) + _outer(ax3 * (sp * sp)),
        ],
        axis=1,
    )
    bxd = jnp.stack(
        [
            _inner(bx1) + _outer(bx2),
            _inner(bx1 * (sm * sm)) + _outer(bx2 * (sp * sp)),
        ],
        axis=1,
    )
    cx = _inner(cx_half) + _outer(cx_half)

    return RadialPreconditionerCoefficients(axm=axm, axd=axd, bxm=bxm, bxd=bxd, cx=cx)


# ---------------------------------------------------------------------------
# lamcal (lamcal.f90)
# ---------------------------------------------------------------------------


def lamcal(
    *,
    guu_half: Array,
    guv_half: Array,
    gvv_half: Array,
    sqrt_g_half: Array,
    lamscale: Array,
    angular_weight: Array,
    mpol: int,
    ntor: int,
    nfp: int,
    lthreed: bool,
    damping_factor: float = 2.0,
    r0scale: float = 1.0,
) -> Array:
    """Diagonal lambda-force preconditioner ``faclam`` (VMEC2000 ``lamcal.f90``).

    Parameters
    ----------
    guu_half, guv_half, gvv_half, sqrt_g_half:
        ``(ns, ntheta, nzeta)`` half-mesh metric elements and Jacobian
        *including* the dummy axis row (VMEC storage); the axis row is
        replaced internally (``blam(1) = blam(2)``).  ``guv`` is ignored
        unless ``lthreed``.
    lamscale:
        VMEC lambda scale factor (``bcovar``: ``sqrt(sum phip**2 * wint)``).
    angular_weight:
        As in :func:`precondn` (VMEC ``wint``).
    mpol, ntor, nfp, lthreed:
        Mode-space resolution (static).
    damping_factor, r0scale:
        VMEC uses ``2.0`` and ``mscale(0)*nscale(0) = 1``.

    Returns
    -------
    ``(ns, mpol, ntor+1)`` array ``faclam``: the multiplicative preconditioner
    of the lambda force, with

    - ``sqrt(s)**min(m**2/256, 8)`` damping (only active for ``m > 16``),
    - the ``(0,0)`` slot set to ``damping/(4*r0scale**2) / blam`` — VMEC's
      chip/iota channel (the ``lamscale**2`` in ``p_factor`` cancels there),
    - the axis row zeroed except ``(0,0)`` (VMEC ``jlam(m) = 2``).
    """
    guu = jnp.asarray(guu_half)
    guv = jnp.asarray(guv_half)
    gvv = jnp.asarray(gvv_half)
    sqrt_g = jnp.asarray(sqrt_g_half)
    dtype = guu.dtype

    ns = int(guu.shape[0])
    mpol = int(mpol)
    nrange = int(ntor) + 1
    if ns < 2:
        return jnp.zeros((ns, mpol, nrange), dtype=dtype)

    w3 = _weight_bcast(angular_weight)
    sqrt_g_safe = jnp.where(sqrt_g != 0.0, sqrt_g, 1.0)

    # Surface averages <g../gsqrt> on the half mesh (lamcal: blam/clam/dlam).
    blam = jnp.sum(guu / sqrt_g_safe * w3, axis=(1, 2))
    clam = jnp.sum(gvv / sqrt_g_safe * w3, axis=(1, 2))
    if bool(lthreed):
        dlam = jnp.sum(guv / sqrt_g_safe * w3, axis=(1, 2))
    else:
        dlam = jnp.zeros_like(blam)

    # blam(1) = blam(2), then full-mesh average blam(js) = (blam(js)+blam(js+1))/2
    # with blam(ns+1) = 0 (lamcal.f90).
    blam = blam.at[0].set(blam[1])
    clam = clam.at[0].set(clam[1])
    dlam = dlam.at[0].set(dlam[1])
    zero = jnp.zeros((1,), dtype=dtype)
    blam_full = blam.at[1:].set(0.5 * (blam[1:] + jnp.concatenate([blam[2:], zero])))
    clam_full = clam.at[1:].set(0.5 * (clam[1:] + jnp.concatenate([clam[2:], zero])))
    dlam_full = dlam.at[1:].set(0.5 * (dlam[1:] + jnp.concatenate([dlam[2:], zero])))

    lamscale = jnp.asarray(lamscale, dtype=dtype)
    p_factor = jnp.asarray(float(damping_factor), dtype=dtype) / (
        4.0 * (float(r0scale) ** 2) * lamscale * lamscale
    )

    sqrt_s_full, _ = _sqrt_s_profiles(ns)
    sqrt_s_full = sqrt_s_full.copy()
    sqrt_s_full[-1] = 1.0
    sqrt_s_full_j = jnp.asarray(sqrt_s_full, dtype=dtype)

    m = jnp.arange(mpol, dtype=dtype)
    n = jnp.arange(nrange, dtype=dtype)
    tnn = (n * float(nfp)) ** 2
    tmm = m * m
    tmn = 2.0 * m[:, None] * n[None, :] * float(nfp)
    power = jnp.minimum(tmm / (16.0 * 16.0), 8.0)  # sqrt(s) damping for m > 16

    b_mode = blam_full[:, None, None]
    c_mode = clam_full[:, None, None]
    d_mode = dlam_full[:, None, None]
    denominator = (
        tnn[None, None, :] * b_mode
        + tmn[None, :, :] * jnp.copysign(d_mode, b_mode)
        + tmm[None, :, None] * c_mode
    )
    denominator = jnp.where(denominator == 0.0, -1.0e-10, denominator)

    sqrt_s_power = sqrt_s_full_j[:, None, None] ** power[None, :, None]
    faclam = p_factor * sqrt_s_power / denominator

    # (0,0): chip/iota channel — the lamscale**2 cancels out of p_factor.
    blam_safe = jnp.where(blam_full != 0.0, blam_full, jnp.asarray(-1.0e-10, dtype=dtype))
    faclam = faclam.at[:, 0, 0].set(p_factor * (lamscale * lamscale) / blam_safe)

    # VMEC jlam(m) = 2: the axis row is inactive except for the (0,0) slot.
    axis_mask = jnp.zeros((mpol, nrange), dtype=dtype).at[0, 0].set(1.0)
    faclam = faclam.at[0].set(faclam[0] * axis_mask)
    return faclam


# ---------------------------------------------------------------------------
# scalfor (scalfor.f) + tridslv
# ---------------------------------------------------------------------------


class TridiagonalMatrices(NamedTuple):
    """Assembled per-mode radial tridiagonal system (VMEC2000 ``scalfor.f``).

    ``ax[js]`` couples row ``js`` to ``js+1`` (superdiagonal), ``bx[js]`` to
    ``js-1`` (subdiagonal), ``dx`` is the diagonal.  Shapes
    ``(jmax, mpol, ntor+1)``.
    """

    ax: Array
    bx: Array
    dx: Array


def scalfor_matrices(
    coefficients: RadialPreconditionerCoefficients,
    *,
    delta_s: Array,
    mpol: int,
    ntor: int,
    nfp: int,
    ns: int,
    jmax: int | None = None,
    stabilize_edge_zc00: bool = False,
) -> TridiagonalMatrices:
    """Assemble the per-mode tridiagonal system (matrix half of ``scalfor.f``).

    Parameters
    ----------
    coefficients:
        Output of :func:`precondn` — VMEC's ``(arm, ard, brm, brd, crd)`` for
        the R force or ``(azm, azd, bzm, bzd, crd)`` for the Z force.
    delta_s:
        Radial spacing ``hs`` (only used by the edge stabilization factor).
    mpol, ntor, nfp, ns:
        Static resolution parameters.
    jmax:
        Number of evolved rows; defaults to ``ns-1`` (fixed boundary,
        VMEC ``jmax = ns1``).  Pass ``ns`` for free-boundary-style solves —
        only then do the ``EDGE_PEDESTAL`` and ZC(0,0) edge terms activate.
    stabilize_edge_zc00:
        Apply the ZC(0,0)(ns) ``fac = 0.25`` stabilization
        (``mult_fac = min(fac, fac*hs*15)``); VMEC applies it to the **Z**
        force system only (``iflag = 1`` call site, ``.not. lfreeb``).

    Notes
    -----
    Assembly per mode (VMEC 1-based ``js``, even/odd column ``mod(m,2)``)::

        ax(js) = -(axm(js+1,mp) + bxm(js+1,mp)*m**2)
        bx(js) = -(axm(js,mp)   + bxm(js,mp)*m**2)
        dx(js) = -(axd(js,mp) + bxd(js,mp)*m**2 + cx(js)*(n*nfp)**2)

    with the axis row zeroed for ``m >= 1``, the m=1 axis fold
    ``dx(2,m=1) += bx(2,m=1)``, and (edge rows only, ``jmax >= ns``) the
    diagonal stiffened by ``1+EDGE_PEDESTAL`` (m <= 1) or
    ``1+2*EDGE_PEDESTAL`` (m >= 2).
    """
    axm = jnp.asarray(coefficients.axm)
    axd = jnp.asarray(coefficients.axd)
    bxm = jnp.asarray(coefficients.bxm)
    bxd = jnp.asarray(coefficients.bxd)
    cx = jnp.asarray(coefficients.cx)
    dtype = axd.dtype

    ns = int(ns)
    jmax = max(ns - 1, 1) if jmax is None else int(max(1, min(int(jmax), ns)))
    mpol = int(mpol)
    nrange = int(ntor) + 1

    m = jnp.arange(mpol, dtype=dtype)
    n = jnp.arange(nrange, dtype=dtype)
    m2 = (m * m)[None, :, None]
    n2 = ((n * float(nfp)) ** 2)[None, None, :]
    m_parity = (np.arange(mpol) % 2).astype(np.int32)  # VMEC mp = mod(m,2)+1

    # Half-mesh off-diagonal coefficients padded/truncated to jmax rows.
    ns_half = int(axm.shape[0])
    pad_rows = max(jmax - ns_half, 0)
    if pad_rows > 0:
        pad = jnp.zeros((pad_rows, 2), dtype=dtype)
        axm_rows = jnp.concatenate([axm, pad], axis=0)
        bxm_rows = jnp.concatenate([bxm, pad], axis=0)
    else:
        axm_rows = axm[:jmax]
        bxm_rows = bxm[:jmax]
    axd_rows = axd[:jmax]
    bxd_rows = bxd[:jmax]
    cx_rows = cx[:jmax]

    axm_m = axm_rows[:, m_parity]
    bxm_m = bxm_rows[:, m_parity]
    axd_m = axd_rows[:, m_parity]
    bxd_m = bxd_rows[:, m_parity]

    ax = -(axm_m[:, :, None] + bxm_m[:, :, None] * m2)
    dx = -(axd_m[:, :, None] + bxd_m[:, :, None] * m2 + cx_rows[:, None, None] * n2)
    bx = jnp.zeros_like(ax)
    if jmax > 1:
        bx = bx.at[1:].set(-(axm_m[:-1, :, None] + bxm_m[:-1, :, None] * m2))

    # Axis row inactive for m >= 1 (jmin = 1).
    if mpol > 1:
        ax = ax.at[0, 1:, :].set(0.0)
        dx = dx.at[0, 1:, :].set(0.0)

    # m = 1 axis fold: X(js=0, m=1) = X(js=1, m=1) => dx(1) += bx(1).
    if jmax > 1 and mpol > 1:
        dx = dx.at[1, 1, :].add(bx[1, 1, :])

    # Edge stiffening — VMEC scalfor.f, only when the edge row is evolved.
    if jmax >= ns and ns > 0:
        edge_pedestal = jnp.asarray(EDGE_PEDESTAL, dtype=dtype)
        edge = ns - 1
        if mpol > 0:
            dx = dx.at[edge, 0:1, :].multiply(1.0 + edge_pedestal)
        if mpol > 1:
            dx = dx.at[edge, 1:2, :].multiply(1.0 + edge_pedestal)
        if mpol > 2:
            dx = dx.at[edge, 2:, :].multiply(1.0 + 2.0 * edge_pedestal)
        if stabilize_edge_zc00 and mpol > 0 and nrange > 0:
            # SPH: avoid the ZC(0,0)(ns) divergence.  Net effect relative to
            # the un-pedestaled diagonal is *(1 - mult_fac).
            fac = jnp.asarray(ZC00_EDGE_FACTOR, dtype=dtype)
            hs = jnp.asarray(delta_s, dtype=dtype)
            mult_fac = jnp.minimum(fac, fac * hs * jnp.asarray(15.0, dtype=dtype))
            dx = dx.at[edge, 0, 0].multiply((1.0 - mult_fac) / (1.0 + edge_pedestal))

    return TridiagonalMatrices(ax=ax, bx=bx, dx=dx)


def tridiagonal_solve(
    superdiagonal: Array,
    diagonal: Array,
    subdiagonal: Array,
    rhs: Array,
    *,
    method: str = "auto",
) -> Array:
    """Radial tridiagonal solve vectorized over all (m,n) columns and fields.

    VMEC2000: ``serial_tridslv`` (in ``scalfor.f``): solve, along the leading
    (radial) axis, ``sub[j]*x[j-1] + diag[j]*x[j] + super[j]*x[j+1] = rhs[j]``
    for every trailing-column element simultaneously.  ``sub[0]`` and
    ``super[-1]`` are ignored.  ``rhs`` may carry extra trailing axes (stacked
    force fields) beyond the shape of ``diagonal``; they are solved in one
    pass.

    Thin arg-order adapter over :func:`solvax.tridiagonal_solve` (roadmap R18b
    shared-solver consolidation).  SOLVAX uses the ``(lower, diag, upper)``
    band convention (``lower = sub``, ``upper = super``), so the sub-/super-
    diagonals are swapped here to preserve vmex's ``(super, diag, sub)``
    signature and the two ``scalfor`` call sites.  The numerics are unchanged:
    SOLVAX's Thomas backend is the verbatim port of the legacy parity-proven
    sweep (same ``eps = 1e-12``), so ``method`` still selects it bit-for-bit —

    - ``"thomas"``: two ``lax.scan`` Thomas sweeps (jit-friendly, no host
      round-trips); the CPU path, **bitwise** identical to the pinned A/B tests.
    - ``"lax"``: XLA's fused batched solver ``jax.lax.linalg.tridiagonal_solve``
      (cuSPARSE ``gtsv2`` on CUDA, LAPACK ``gtsv`` on CPU); the accelerator fast
      path (numerically equivalent, not bit-identical to Thomas).
    - ``"auto"`` (default): Thomas when lowering for CPU (bit parity), the fused
      solver elsewhere, chosen with ``jax.lax.platform_dependent``.  Systems
      with fewer than 3 radial rows always use Thomas.
    """
    # solvax: tridiagonal_solve(lower=sub, diag, upper=super); swap to keep our API.
    return _sx_tridiagonal_solve(subdiagonal, diagonal, superdiagonal, rhs, method=method)


def scalfor(
    force: Array,
    matrices: TridiagonalMatrices,
    *,
    jmax: int,
    tridiagonal_method: str = "auto",
    return_safe: bool = False,
) -> Array | tuple[Array, Array]:
    """Apply the assembled preconditioner to a spectral force (``scalfor.f``).

    Solves the radial tridiagonal systems of ``matrices`` against ``force``
    respecting the VMEC start-index conventions (``jmin2``): the ``m = 0``
    block is solved on rows ``0..jmax-1``, the ``m >= 1`` blocks on rows
    ``1..jmax-1`` with the axis row of the solution set to zero.  Rows
    ``>= jmax`` (the fixed edge in fixed-boundary mode) keep the input force.

    Parameters
    ----------
    force:
        ``(ns, mpol, ntor+1)`` spectral force (VMEC ``gcr``/``gcz``), or
        ``(ns, mpol, ntor+1, nfields)`` to precondition several parity
        channels (e.g. ``frcc/frss`` or ``fzsc/fzcs``) in one batched solve.
    matrices:
        Output of :func:`scalfor_matrices` for the matching force family.
    jmax:
        Same ``jmax`` used to assemble ``matrices`` (static).
    tridiagonal_method:
        Forwarded to SOLVAX's checked tridiagonal solve
        (``"auto"``/``"thomas"``/``"lax"``; static).  The default picks per
        lowering platform: the bit-parity Thomas scan on CPU, the fused
        backend on accelerators.
    return_safe:
        Also return a scalar status.  Production calls use SOLVAX's
        unregularized pivot and backward-residual checks.  A rejected column
        receives the identity fallback (its RHS is preserved) and makes this
        status false, preventing a singular preconditioner from injecting a
        NaN/Inf update while retaining a diagnosable failure signal.
    """
    ax, bx, dx = matrices
    f = jnp.asarray(force)
    stacked = f.ndim == 4
    if not stacked:
        f = f[..., None]
    jmax = int(jmax)
    mpol = int(f.shape[1])
    out = f
    safe = jnp.asarray(True)

    def checked_solve(superdiagonal, diagonal, subdiagonal, rhs):
        result = _sx_tridiagonal_solve_checked(
            subdiagonal,
            diagonal,
            superdiagonal,
            rhs,
            method=tridiagonal_method,
            pivot_rtol=1.0e-8,
            residual_rtol=1.0e-8,
            fallback="identity",
        )
        return result.solution, jnp.all(~result.diagnostics.fallback_used)

    if jmax > 0:
        solution_m0, safe_m0 = checked_solve(
            ax[:jmax, 0, :], dx[:jmax, 0, :], bx[:jmax, 0, :], f[:jmax, 0, :, :],
        )
        safe = safe & safe_m0
        out = out.at[:jmax, 0].set(solution_m0)
        if mpol > 1 and jmax > 1:
            solution_m, safe_m = checked_solve(
                ax[1:jmax, 1:, :], dx[1:jmax, 1:, :], bx[1:jmax, 1:, :], f[1:jmax, 1:, :, :],
            )
            safe = safe & safe_m
            out = out.at[1:jmax, 1:].set(solution_m)
            out = out.at[0, 1:].set(0.0)

    result = out[..., 0] if not stacked else out
    return (result, safe) if return_safe else result
