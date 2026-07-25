"""Force residual scalars, the m=1 polar constraint and preconditioned residuals.

VMEC2000 counterparts
---------------------
- ``Sources/General/residue.f90`` — the m=1 polar constraint:
  coefficient-space mappings :func:`m1_constrained_to_physical` /
  :func:`m1_physical_to_constrained` (``readin.f`` ``lconm1`` convention,
  ``rbss = (rbs + zbc)/2``), the force-side rotation
  ``(gcr, gcz)_{m=1} -> ((gcr+gcz)/sqrt2, (gcr-gcz)/sqrt2)``
  (:func:`m1_residue_rotation`, ``constrain_m1``), its conditional release
  (:func:`zero_m1_z_force` + :func:`m1_zero_condition`), and the ``scale_m1``
  preconditioner factors (:func:`scale_m1_preconditioner_rhs`).
- ``Sources/General/getfsq.f`` — the invariant residuals ``fsqr/fsqz``
  (``gnorm = r1*fnorm``, ``jsmax = ns-1 + medge``) plus ``fsql`` and the edge
  diagnostic ``fedge``: :func:`force_residuals` (+
  :func:`edge_force_condition` for the free-boundary ``medge = 1`` rule).
- ``Sources/TimeStep/funct3d.f`` — the post-``tomnsps`` ``gc = gc*scalxc``
  odd-m scaling (:func:`scalxc_scale_force`) and the preconditioned residuals
  ``fsqr1/fsqz1 = fnorm1*sum(gc**2)``, ``fsql1 = hs*sum(gcl_preconditioned**2)``
  (:func:`preconditioned_residuals`, with
  :func:`apply_radial_preconditioner` / :func:`apply_lambda_preconditioner`
  wrapping :mod:`vmex.core.preconditioner`).

Conventions (parity-critical)
-----------------------------
The m=1 constraint makes the poloidal angle unique near the axis: VMEC evolves
``rss_int = (rss + zcs)/2`` and ``zcs_int = (rss - zcs)/2`` (and ``rsc/zcc``
for ``lasym``) instead of the physical pairs.  Geometry synthesis therefore
needs :func:`m1_constrained_to_physical` first (this is the mapping assumed by
:func:`vmex.core.geometry.real_space_geometry`).  All release conditions
are pure functions of traced values — the solver supplies the previous
``fsqz``/iteration counters and receives traced booleans, so a single compiled
step serves both execution lanes.

The numerics are ported verbatim from the parity-proven legacy kernels
``vmex.kernels.residue``, ``vmex.kernels.parity`` and the solver
modules ``solvers/fixed_boundary/residual/payload_blocks.py`` /
``preconditioning/operators.py``; equivalence is enforced in
``tests/test_forces_residuals_ab.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, overload

import numpy as np

import jax.numpy as jnp

from .fourier import ModeTable
from .preconditioner import RadialPreconditionerCoefficients, TridiagonalMatrices, scalfor
from .transforms import (
    SpectralForce, odd_m_sqrt_s_scaling,
    register_pytree_dataclass as _register,
)

__all__ = [
    "M1_FSQZ_RELEASE_THRESHOLD",
    "M1_STARTUP_ITERATIONS",
    "EDGE_FORCE_FSQ_THRESHOLD",
    "EDGE_FORCE_ITERATION_WINDOW",
    "ForceResiduals",
    "PreconditionedResiduals",
    "m1_constrained_to_physical",
    "m1_physical_to_constrained",
    "m1_residue_rotation",
    "zero_m1_z_force",
    "m1_zero_condition",
    "edge_force_condition",
    "scalxc_scale_force",
    "force_residuals",
    "scale_m1_preconditioner_rhs",
    "apply_radial_preconditioner",
    "apply_lambda_preconditioner",
    "preconditioned_residuals",
]

Array = Any

#: residue.f90 ``FThreshold``: the constrained m=1 Z force is zeroed once
#: ``fsqz`` drops below this (or during the first iterations after a restart).
M1_FSQZ_RELEASE_THRESHOLD = 1.0e-6

#: v8.50: ``iter2 < 2`` also zeroes the constrained Z force (reset support).
M1_STARTUP_ITERATIONS = 2

#: residue.f90 (SPH041117): free-boundary edge rows join ``getfsq``
#: (``jedge = 1``) when ``fsqr + fsqz`` has dropped below this ...
EDGE_FORCE_FSQ_THRESHOLD = 1.0e-6

#: ... within this many iterations of the last restart (``delIter < 50``).
EDGE_FORCE_ITERATION_WINDOW = 50

_R_BLOCKS = ("force_R_cc", "force_R_ss", "force_R_sc", "force_R_cs")
_Z_BLOCKS = ("force_Z_sc", "force_Z_cs", "force_Z_cc", "force_Z_ss")
_LAMBDA_BLOCKS = (
    "force_lambda_sc",
    "force_lambda_cs",
    "force_lambda_cc",
    "force_lambda_ss",
)


@dataclass(frozen=True)
class ForceResiduals:
    """Invariant force residuals (VMEC2000 ``getfsq.f`` / ``residue.f90``).

    - ``fsqr/fsqz`` (``fsqr, fsqz``): ``r1*fnorm * sum(gc**2)`` over the
      evolved surfaces (``jsmax = ns-1 + medge``);
    - ``fsql`` (``fsql``): ``fnormL * sum(gcl**2)`` (all surfaces);
    - ``fedge`` (``fedge``): ``r1*fnorm`` times the edge-row R/Z sums — only
      meaningful when the projection kept the edge row (``include_edge``);
    - ``gcr2/gcz2/gcl2``: the raw sums of squares (solver diagnostics).
    """

    fsqr: Array
    fsqz: Array
    fsql: Array
    fedge: Array
    gcr2: Array
    gcz2: Array
    gcl2: Array


@dataclass(frozen=True)
class PreconditionedResiduals:
    """Preconditioned residuals (VMEC2000 ``residue.f90``: fsqr1/fsqz1/fsql1)."""

    fsqr1: Array
    fsqz1: Array
    fsql1: Array


for _cls in (ForceResiduals, PreconditionedResiduals):
    _register(_cls)


# ---------------------------------------------------------------------------
# m=1 polar constraint: coefficient-space mappings (residue.f90 / readin.f)
# ---------------------------------------------------------------------------


def _m1_partner_tables(modes: ModeTable) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Static per-mode tables for the m=1 constraint in signed-(m, n) packing.

    Returns ``(partner, has_partner, sign, m1_mask)``: ``partner[k]`` is the
    index of the mode ``(m_k, -n_k)`` (``k`` itself when absent or ``n = 0``),
    ``sign[k] = +1/-1`` for ``n_k >= 0`` / ``n_k < 0``.
    """
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    index = {(int(mk), int(nk)): k for k, (mk, nk) in enumerate(zip(m, n))}
    partner = np.arange(m.size, dtype=np.int32)
    has_partner = np.zeros((m.size,), dtype=bool)
    for k in range(m.size):
        j = index.get((int(m[k]), -int(n[k])))
        if j is not None and int(n[k]) != 0:
            partner[k] = j
            has_partner[k] = True
    sign = np.where(n < 0, -1.0, 1.0)
    return partner, has_partner, sign, (m == 1)


def _m1_rotate_threed(
    R_cos: Array,
    Z_sin: Array,
    *,
    modes: ModeTable,
    alpha: float,
) -> tuple[Array, Array]:
    """Rotate the (rss, zcs) m=1 pair in signed packing: ``(x, y) -> (alpha*(x+y), alpha*(x-y))``.

    ``alpha = 1`` undoes the internal constraint (``residue.f90``
    ``rss = rss_int + zcs_int``); ``alpha = 0.5`` applies it.  Only the
    ``m = 1, n != 0`` signed modes change (``rss/zcs`` vanish at ``n = 0``).
    """
    partner, has_partner, sign, m1_mask = _m1_partner_tables(modes)
    active = m1_mask & has_partner
    if not np.any(active):
        return R_cos, Z_sin
    R_cos = jnp.asarray(R_cos)
    Z_sin = jnp.asarray(Z_sin)
    dtype = R_cos.dtype
    sign_j = jnp.asarray(sign, dtype=dtype)[None, :]
    active_j = jnp.asarray(active)[None, :]

    R_partner = R_cos[:, partner]
    Z_partner = Z_sin[:, partner]
    # (m, n >= 0) block values seen from every signed m=1 mode.
    rss = sign_j * (R_cos - R_partner)
    zcs = sign_j * (Z_partner - Z_sin)
    rss_new = alpha * (rss + zcs)
    zcs_new = alpha * (rss - zcs)
    R_new = 0.5 * ((R_cos + R_partner) + sign_j * rss_new)
    Z_new = 0.5 * ((Z_sin + Z_partner) - sign_j * zcs_new)
    return (
        jnp.where(active_j, R_new, R_cos),
        jnp.where(active_j, Z_new, Z_sin),
    )


def _m1_rotate_asym(
    R_sin: Array,
    Z_cos: Array,
    *,
    modes: ModeTable,
    alpha: float,
) -> tuple[Array, Array]:
    """Rotate the (rsc, zcc) m=1 pair in signed packing (``lasym`` sector).

    Applies to *all* ``n`` (the ``rsc/zcc`` blocks exist at ``n = 0`` too):
    ``rsc -> alpha*(rsc + zcc)``, ``zcc -> alpha*(rsc - zcc)``.
    """
    partner, has_partner, sign, m1_mask = _m1_partner_tables(modes)
    if not np.any(m1_mask):
        return R_sin, Z_cos
    R_sin = jnp.asarray(R_sin)
    Z_cos = jnp.asarray(Z_cos)
    dtype = R_sin.dtype
    sign_j = jnp.asarray(sign, dtype=dtype)[None, :]
    w_partner = jnp.asarray(has_partner.astype(float), dtype=dtype)[None, :]
    m1_j = jnp.asarray(m1_mask)[None, :]
    # n = 0 modes carry the whole block (no 1/2 reconstruction split).
    half = jnp.asarray(np.where(has_partner, 0.5, 1.0), dtype=dtype)[None, :]

    R_partner = R_sin[:, partner]
    Z_partner = Z_cos[:, partner]
    rsc = R_sin + w_partner * R_partner
    zcc = Z_cos + w_partner * Z_partner
    # Untouched complementary blocks (rcs of R_sin, zss of Z_cos).
    rcs = w_partner * sign_j * (R_partner - R_sin)
    zss = w_partner * sign_j * (Z_cos - Z_partner)
    rsc_new = alpha * (rsc + zcc)
    zcc_new = alpha * (rsc - zcc)
    R_new = half * rsc_new - 0.5 * sign_j * rcs
    Z_new = half * zcc_new + 0.5 * sign_j * zss
    return (
        jnp.where(m1_j, R_new, R_sin),
        jnp.where(m1_j, Z_new, Z_cos),
    )


def m1_constrained_to_physical(
    R_cos: Array,
    Z_sin: Array,
    R_sin: Array | None = None,
    Z_cos: Array | None = None,
    *,
    modes: ModeTable,
    lthreed: bool,
    lasym: bool,
    lconm1: bool = True,
) -> tuple[Array, Array, Array | None, Array | None]:
    """Undo the m=1 internal constraint on signed spectral coefficients.

    VMEC2000: ``residue.f90`` — the evolved (internal) m=1 pairs are
    ``rss_int = (rss + zcs)/2`` and ``zcs_int = (rss - zcs)/2`` (``lconm1``,
    cf. ``readin.f`` ``rbss = (rbs + zbc)/2``), so synthesis needs

        ``rss = rss_int + zcs_int``,  ``zcs = rss_int - zcs_int``

    (and ``rsc/zcc`` likewise for ``lasym``).  This is the mapping the
    geometry/fields pipeline expects on its inputs.  A no-op unless
    ``lconm1`` and the run is 3D or asymmetric.

    Returns ``(R_cos, Z_sin, R_sin, Z_cos)``.
    """
    if not bool(lconm1) or (not bool(lthreed) and not bool(lasym)):
        return R_cos, Z_sin, R_sin, Z_cos
    if bool(lthreed):
        R_cos, Z_sin = _m1_rotate_threed(R_cos, Z_sin, modes=modes, alpha=1.0)
    if bool(lasym):
        R_sin, Z_cos = _m1_rotate_asym(R_sin, Z_cos, modes=modes, alpha=1.0)
    return R_cos, Z_sin, R_sin, Z_cos


def m1_physical_to_constrained(
    R_cos: Array,
    Z_sin: Array,
    R_sin: Array | None = None,
    Z_cos: Array | None = None,
    *,
    modes: ModeTable,
    lthreed: bool,
    lasym: bool,
    lconm1: bool = True,
) -> tuple[Array, Array, Array | None, Array | None]:
    """Apply the m=1 internal constraint (inverse of
    :func:`m1_constrained_to_physical`): ``rss_int = (rss + zcs)/2`` etc."""
    if not bool(lconm1) or (not bool(lthreed) and not bool(lasym)):
        return R_cos, Z_sin, R_sin, Z_cos
    if bool(lthreed):
        R_cos, Z_sin = _m1_rotate_threed(R_cos, Z_sin, modes=modes, alpha=0.5)
    if bool(lasym):
        R_sin, Z_cos = _m1_rotate_asym(R_sin, Z_cos, modes=modes, alpha=0.5)
    return R_cos, Z_sin, R_sin, Z_cos


# ---------------------------------------------------------------------------
# m=1 polar constraint: force-side rotation and release (residue.f90)
# ---------------------------------------------------------------------------


def _set_m1_row(block: Array | None, row: Array) -> Array | None:
    if block is None:
        return None
    return jnp.asarray(block).at[:, 1, :].set(row)


def m1_residue_rotation(force: SpectralForce, *, lconm1: bool = True) -> SpectralForce:
    """Rotate the m=1 force pairs into the constrained basis (``constrain_m1``).

    VMEC2000: ``residue.f90`` — before ``getfsq`` the m=1 rows of the paired
    blocks are rotated with ``osqrt2 = 1/sqrt(2)``:

        ``gcr <- (gcr + gcz)/sqrt(2)``,  ``gcz <- (gcr - gcz)/sqrt(2)``

    for the symmetric pair ``(frss, fzcs)`` (3D) and the asymmetric pair
    ``(frsc, fzcc)`` (``lasym``).  The conditional zeroing of the constrained
    component is separate — see :func:`zero_m1_z_force` /
    :func:`m1_zero_condition`.
    """
    if not bool(lconm1):
        return force
    updates: dict[str, Array] = {}
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    if force.force_R_ss is not None and force.force_Z_cs is not None:
        if int(jnp.asarray(force.force_R_ss).shape[1]) > 1:
            gcr = jnp.asarray(force.force_R_ss)[:, 1, :]
            gcz = jnp.asarray(force.force_Z_cs)[:, 1, :]
            updates["force_R_ss"] = _set_m1_row(force.force_R_ss, inv_sqrt2 * (gcr + gcz))
            updates["force_Z_cs"] = _set_m1_row(force.force_Z_cs, inv_sqrt2 * (gcr - gcz))
    if force.force_R_sc is not None and force.force_Z_cc is not None:
        if int(jnp.asarray(force.force_R_sc).shape[1]) > 1:
            gcr = jnp.asarray(force.force_R_sc)[:, 1, :]
            gcz = jnp.asarray(force.force_Z_cc)[:, 1, :]
            updates["force_R_sc"] = _set_m1_row(force.force_R_sc, inv_sqrt2 * (gcr + gcz))
            updates["force_Z_cc"] = _set_m1_row(force.force_Z_cc, inv_sqrt2 * (gcr - gcz))
    return replace(force, **updates) if updates else force


def zero_m1_z_force(force: SpectralForce, enabled: Array) -> SpectralForce:
    """Zero the constrained m=1 Z-force rows (``fzcs``/``fzcc``) when enabled.

    VMEC2000: ``residue.f90`` (``constrain_m1``: ``IF (fsqz < FThreshold .OR.
    iter2 < 2) gcz = 0``).  ``enabled`` is a traced scalar (bool or 0/1 float,
    typically :func:`m1_zero_condition`), so no retracing occurs when the
    condition flips.
    """
    updates: dict[str, Array] = {}
    for name in ("force_Z_cs", "force_Z_cc"):
        block = getattr(force, name)
        if block is None:
            continue
        block = jnp.asarray(block)
        if int(block.shape[1]) <= 1:
            continue
        mask = jnp.asarray(enabled).astype(block.dtype)
        updates[name] = block.at[:, 1, :].set(block[:, 1, :] * (1.0 - mask))
    return replace(force, **updates) if updates else force


def m1_zero_condition(*, fsqz_previous: Array, iterations_since_restart: Array) -> Array:
    """Traced release condition for :func:`zero_m1_z_force` (``residue.f90``).

    ``True`` (zero the constrained Z force) when ``iter2 < 2`` — counted since
    the last restart — or the previous ``fsqz`` is below
    ``M1_FSQZ_RELEASE_THRESHOLD = 1e-6``.  Pure function of traced values; the
    caller keeps the values and decides scheduling.
    """
    return (jnp.asarray(iterations_since_restart) < M1_STARTUP_ITERATIONS) | (
        jnp.asarray(fsqz_previous) < M1_FSQZ_RELEASE_THRESHOLD
    )


def edge_force_condition(
    *,
    fsq_rz_previous: Array,
    iterations_since_restart: Array,
    free_boundary: bool,
) -> Array:
    """Traced ``getfsq`` edge-inclusion rule (``residue.f90``, SPH041117).

    ``medge = 1`` (edge rows join ``fsqr/fsqz``) only for free-boundary runs
    with ``iter2 - iter1 < 50`` and ``fsqr + fsqz < 1e-6``.  Returned as a
    traced boolean built from the supplied values.
    """
    cond = (jnp.asarray(iterations_since_restart) < EDGE_FORCE_ITERATION_WINDOW) & (
        jnp.asarray(fsq_rz_previous) < EDGE_FORCE_FSQ_THRESHOLD
    )
    return cond if bool(free_boundary) else jnp.zeros_like(cond)


# ---------------------------------------------------------------------------
# scalxc force scaling (funct3d.f) and getfsq
# ---------------------------------------------------------------------------


def _map_blocks(force: SpectralForce, names: tuple[str, ...], fn) -> dict[str, Array]:
    out: dict[str, Array] = {}
    for name in names:
        block = getattr(force, name)
        if block is not None:
            out[name] = fn(jnp.asarray(block))
    return out


def scalxc_scale_force(force: SpectralForce, *, s: Array) -> SpectralForce:
    """Apply the odd-m ``1/sqrt(s)`` scaling to every force block.

    VMEC2000: ``funct3d.f`` — ``gc = gc * scalxc`` right after ``tomnsps``,
    making the scaling part of the definition of the reported residuals.
    Uses :func:`vmex.core.transforms.odd_m_sqrt_s_scaling` (``profil3d.f``
    ``scalxc``).
    """
    mpol = int(jnp.asarray(force.force_R_cc).shape[1])
    scalxc = odd_m_sqrt_s_scaling(jnp.asarray(s), mpol)[:, :, None]
    updates = _map_blocks(
        force, _R_BLOCKS + _Z_BLOCKS + _LAMBDA_BLOCKS, lambda b: b * scalxc.astype(b.dtype)
    )
    return replace(force, **updates)


def force_residuals(
    force: SpectralForce,
    *,
    fnorm: Array,
    fnormL: Array,
    r1: Array,
    include_edge: bool = False,
) -> ForceResiduals:
    """Invariant residuals ``fsqr/fsqz/fsql`` and ``fedge`` (``getfsq.f``).

    ``fsqr = r1*fnorm*sum(gcr**2)`` and ``fsqz = r1*fnorm*sum(gcz**2)`` over
    surfaces ``js <= jsmax = ns-1 + medge`` (``include_edge`` is VMEC's
    ``medge = 1``; see :func:`edge_force_condition`); ``fsql = fnormL *
    sum(gcl**2)`` over all surfaces.  ``fedge = r1*fnorm`` times the edge-row
    R/Z sums (``residue.f90``) — nonzero only when the projection kept the
    edge row (``tomnsps(include_edge=True)``).

    The norms come from
    :func:`vmex.core.fields.energies_and_force_norms` (``fnorm/fnormL/r1``).
    Pass the force *after* :func:`m1_residue_rotation` /
    :func:`zero_m1_z_force` / :func:`scalxc_scale_force`, matching the
    ``funct3d.f`` -> ``residue.f90`` ordering.
    """
    ns = int(jnp.asarray(force.force_R_cc).shape[0])
    jsmax = ns if (bool(include_edge) or ns <= 1) else ns - 1

    def sum_sq(names: tuple[str, ...], sl: slice) -> Array:
        total = jnp.asarray(0.0, dtype=jnp.asarray(force.force_R_cc).dtype)
        for name in names:
            block = getattr(force, name)
            if block is not None:
                block = jnp.asarray(block)
                total = total + jnp.sum(block[sl] * block[sl])
        return total

    gcr2 = sum_sq(_R_BLOCKS, slice(0, jsmax))
    gcz2 = sum_sq(_Z_BLOCKS, slice(0, jsmax))
    gcl2 = sum_sq(_LAMBDA_BLOCKS, slice(0, ns))
    edge2 = sum_sq(_R_BLOCKS, slice(ns - 1, ns)) + sum_sq(_Z_BLOCKS, slice(ns - 1, ns))

    gnorm = jnp.asarray(r1) * jnp.asarray(fnorm)
    return ForceResiduals(
        fsqr=gnorm * gcr2,
        fsqz=gnorm * gcz2,
        fsql=jnp.asarray(fnormL) * gcl2,
        fedge=gnorm * edge2,
        gcr2=gcr2,
        gcz2=gcz2,
        gcl2=gcl2,
    )


# ---------------------------------------------------------------------------
# Preconditioned residuals (residue.f90 + scalfor.f)
# ---------------------------------------------------------------------------


def scale_m1_preconditioner_rhs(
    force: SpectralForce,
    *,
    coefficients_R: RadialPreconditionerCoefficients,
    coefficients_Z: RadialPreconditionerCoefficients,
    lconm1: bool = True,
) -> SpectralForce:
    """m=1 force balance factors before the radial solve (``scale_m1``).

    VMEC2000: ``residue.f90`` (``scale_m1_par``) — the constrained m=1 rows of
    the R and Z systems are weighted with the odd-parity preconditioner
    diagonals so the pair converges at one rate:

        ``fac_R = (ard + brd)/(ard + brd + azd + bzd)``  (odd column),
        ``fac_Z = (azd + bzd)/(ard + brd + azd + bzd)``,

    applied to the ``frss/fzcs`` (3D) and ``frsc/fzcc`` (``lasym``) m=1 rows.
    Following the parity-proven legacy port, the factors scale the right-hand
    side before :func:`apply_radial_preconditioner`.
    """
    if not bool(lconm1):
        return force
    mpol = int(jnp.asarray(force.force_R_cc).shape[1])
    if mpol <= 1:
        return force
    sr = jnp.asarray(coefficients_R.axd)[:, 1] + jnp.asarray(coefficients_R.bxd)[:, 1]
    sz = jnp.asarray(coefficients_Z.axd)[:, 1] + jnp.asarray(coefficients_Z.bxd)[:, 1]
    denominator = sr + sz
    safe = jnp.where(denominator != 0.0, denominator, 1.0)
    fac_R = jnp.where(denominator != 0.0, sr / safe, 1.0)[:, None]
    fac_Z = jnp.where(denominator != 0.0, sz / safe, 1.0)[:, None]

    updates: dict[str, Array] = {}
    for name, fac in (
        ("force_R_ss", fac_R),
        ("force_Z_cs", fac_Z),
        ("force_R_sc", fac_R),
        ("force_Z_cc", fac_Z),
    ):
        block = getattr(force, name)
        if block is not None:
            block = jnp.asarray(block)
            updates[name] = block.at[:, 1, :].set(fac.astype(block.dtype) * block[:, 1, :])
    return replace(force, **updates) if updates else force


@overload
def apply_radial_preconditioner(
    force: SpectralForce,
    *,
    matrices_R: TridiagonalMatrices,
    matrices_Z: TridiagonalMatrices,
    jmax: int,
    return_safe: Literal[False] = False,
) -> SpectralForce: ...


@overload
def apply_radial_preconditioner(
    force: SpectralForce,
    *,
    matrices_R: TridiagonalMatrices,
    matrices_Z: TridiagonalMatrices,
    jmax: int,
    return_safe: Literal[True],
) -> tuple[SpectralForce, Array]: ...


def apply_radial_preconditioner(
    force: SpectralForce,
    *,
    matrices_R: TridiagonalMatrices,
    matrices_Z: TridiagonalMatrices,
    jmax: int,
    return_safe: bool = False,
) -> SpectralForce | tuple[SpectralForce, Array]:
    """Solve the R and Z radial tridiagonal systems against all force blocks.

    VMEC2000: ``scalfor.f`` applied to ``gcr`` (with the ``arm/ard/brm/brd/
    crd`` matrices) and ``gcz`` (``azm/azd/bzm/bzd``); each parity family is
    solved in one batched :func:`vmex.core.preconditioner.scalfor` call.
    Lambda blocks pass through (see :func:`apply_lambda_preconditioner`).
    """
    updates: dict[str, Array] = {}
    safe = jnp.asarray(True)
    for names, matrices in ((_R_BLOCKS, matrices_R), (_Z_BLOCKS, matrices_Z)):
        present = [name for name in names if getattr(force, name) is not None]
        if not present:
            continue
        stacked = jnp.stack([jnp.asarray(getattr(force, name)) for name in present], axis=-1)
        solved, block_safe = scalfor(
            stacked, matrices, jmax=jmax, return_safe=True
        )
        safe = safe & block_safe
        for idx, name in enumerate(present):
            updates[name] = solved[..., idx]
    result = replace(force, **updates)
    return (result, safe) if return_safe else result


def apply_lambda_preconditioner(force: SpectralForce, faclam: Array) -> SpectralForce:
    """Scale the lambda force blocks by ``faclam`` (``residue.f90``:
    ``gcl = faclam*gcl``; ``faclam`` from
    :func:`vmex.core.preconditioner.lamcal`)."""
    faclam = jnp.asarray(faclam)
    updates = _map_blocks(force, _LAMBDA_BLOCKS, lambda b: b * faclam.astype(b.dtype))
    return replace(force, **updates)


def preconditioned_residuals(
    force_preconditioned: SpectralForce,
    *,
    fnorm1: Array,
    delta_s: Array,
) -> PreconditionedResiduals:
    """Preconditioned residuals ``fsqr1/fsqz1/fsql1`` (``residue.f90``).

    With ``gc`` the output of :func:`apply_radial_preconditioner` +
    :func:`apply_lambda_preconditioner`:

        ``fsqr1 = fnorm1 * sum(gcr**2)``,  ``fsqz1 = fnorm1 * sum(gcz**2)``
        (all surfaces — the edge row is whatever the projection left there),
        ``fsql1 = hs * sum_{js>=2}(gcl**2)``.

    ``fnorm1`` is the spectral-state normalization from
    :func:`vmex.core.fields.preconditioned_force_norm` (``bcovar.f``);
    ``delta_s`` is the radial spacing ``hs``.
    """
    ns = int(jnp.asarray(force_preconditioned.force_R_cc).shape[0])

    def sum_sq(names: tuple[str, ...], sl: slice) -> Array:
        total = jnp.asarray(0.0, dtype=jnp.asarray(force_preconditioned.force_R_cc).dtype)
        for name in names:
            block = getattr(force_preconditioned, name)
            if block is not None:
                block = jnp.asarray(block)
                total = total + jnp.sum(block[sl] * block[sl])
        return total

    fnorm1 = jnp.asarray(fnorm1)
    return PreconditionedResiduals(
        fsqr1=fnorm1 * sum_sq(_R_BLOCKS, slice(0, ns)),
        fsqz1=fnorm1 * sum_sq(_Z_BLOCKS, slice(0, ns)),
        fsql1=jnp.asarray(delta_s) * sum_sq(_LAMBDA_BLOCKS, slice(1, ns)),
    )
