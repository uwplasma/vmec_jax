"""Traceable omnigenity/QI objective.

A quasi-isodynamic (QI) omnigenity residual evaluated as a pure, traceable
function of a converged ``(SpectralState, SolverRuntime)`` pair, so it
composes with the implicit-gradient least-squares lane
(``vmex.core.optimize.least_squares(..., jac="implicit")``) exactly like
:class:`~vmex.core.optimize.QuasisymmetryRatioResidual` — no wout tables,
no host booz_xform round-trip.

Two pieces:

1. :func:`boozer_spectrum_state` — a *traceable* Boozer ``|B|`` transform.
   vmex owns only the equilibrium side: snapping requested surfaces to the
   half mesh and building wout-convention single-surface spectral tables
   from the live state (:func:`vmex.core.boozer_tables.boozer_input_tables`).
   The Boozer transform itself — generating potential, Boozer angles, and
   the angle-transform quadrature — executes in ``booz_xform_jax``'s
   jittable kernel (``booz_xform_jax.jax_api.booz_xform_jax_impl``) for
   symmetric and ``lasym`` states alike, so vmex carries no second
   implementation of the transform and the whole chain stays end-to-end
   differentiable.  The kernel runs on a ``booz_xform_jax.BoozerPlan``
   (mode and trig tables of one static resolution) that vmex builds once
   per distinct resolution and reuses across surfaces, objective
   evaluations, and jit traces (:func:`_boozer_plan`).
   (:func:`boozer_bmnc_state` remains as a deprecated alias of the
   previous name.)

2. :func:`omnigenity_residual` / :class:`QIResidual` — a smooth, lightweight
   surrogate for poloidally-closed-contour (``M = 0, N = 1``) omnigenity.
   It distills conditions used in the constructed-QI target of
   **Goodman et al., "Constructing precisely quasi-isodynamic magnetic
   fields", J. Plasma Phys. 89, 905890504 (2023), arXiv:2211.09829** into a
   level-set form: on each surface, ``|B|`` is sampled along
   Boozer field lines ``theta_B = alpha + iota * phi_B`` over one field
   period and the residual stacks, per surface,

   - **bounce-distance uniformity** (``well_weight``): for every trapping
     level ``B*``, the bounce distance ``delta(alpha, B*)`` between the two
     monotone branches of the magnetic well (smooth occupancy integrals of
     the running-maximum branch envelopes) minus its field-line average —
     the Cary–Shasharina omnigenity condition (Cary & Shasharina, PRL 78,
     674 (1997)) that Goodman's "shuffle" step enforces;
   - **extremum alignment** (``extremum_weight``): the per-field-line
     ``B_min``/``B_max`` minus their field-line averages — poloidal closure
     of the extremal ``|B|`` contours (Goodman's "align the maxima" step;
     also the flat-``B_max`` condition of Dudt et al., J. Plasma Phys. 90,
     905900120 (2024), arXiv:2305.08026);
   - **single-well monotonicity** (``squash_weight``): the pointwise distance
     between ``|B|`` and its monotone branch envelopes — Goodman's "squash"
     distance, penalizing side extrema (multiple wells per period).

   Each piece is an exact zero of an exactly QI field, every operation is
   smooth or piecewise-smooth (sigmoid occupancies, running maxima), and the
   full pipeline is jit/grad/jvp-transparent for the implicit lane.  The
   converse does not hold at finite sampling: this inexpensive surrogate can
   be driven low by a field that still has a large full squash-and-shuffle
   distance.  Use :class:`vmex.core.qi.ConstructedQIResidual` for production
   QI optimization and resolved reporting.

Scope notes
-----------
- Symmetric and non-stellarator-symmetric (``lasym``) states are supported;
  the latter retain both cosine and sine Boozer ``|B|`` harmonics.
- Requested surfaces are snapped to the *nearest half-mesh surface* (the
  Boozer transform is a per-surface construction — same convention as
  :func:`vmex.core.optimize.boozer_modes_from_wout`), not interpolated.
- The wout-engine analogue for cross-checks is
  :func:`vmex.core.optimize.quasi_isodynamic_residual_from_wout`
  (host booz_xform_jax, finite-difference-only).
"""

from __future__ import annotations

import collections
import dataclasses
import warnings
from typing import Any, Iterable

import numpy as np

import jax
import jax.numpy as jnp

from .boozer_tables import boozer_input_tables, high_order_boozer_input_tables
from .solver import SolverRuntime, SpectralState
from .statephysics import _as_1d

__all__ = [
    "boozer_bmnc_high_order",
    "boozer_bmnc_state",
    "boozer_spectrum_high_order",
    "boozer_spectrum_state",
    "omnigenity_residual",
    "QIResidual",
]

Array = Any


# ---------------------------------------------------------------------------
# Traceable Boozer |B| spectrum from a core state
# ---------------------------------------------------------------------------


def _nearest_half_mesh_rows(ns: int, surfaces) -> tuple[np.ndarray, np.ndarray]:
    """Return half-mesh coordinates and one-based nearest row indices.

    Exact midpoint ties select the lower-flux surface, matching booz_xform's
    ``s_in`` lookup independently of round-off in the two grid constructions.
    """
    s_half = (np.arange(ns - 1) + 0.5) / (ns - 1)
    requested = np.atleast_1d(
        np.asarray(list(np.ravel(surfaces)), dtype=float)
    )
    if requested.size == 0:
        raise ValueError("surfaces must be non-empty")
    rows = []
    for surface in requested:
        distance = np.abs(s_half - surface)
        tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(float(surface)))
        nearest = np.flatnonzero(distance <= distance.min() + tolerance)[0]
        rows.append(int(nearest) + 1)
    return s_half, np.asarray(rows, dtype=int)


def _refine_booz_grids(constants, grids, oversample, nfp):
    """booz_xform_jax constants/grids on an ``oversample``-times finer grid.

    ``prepare_booz_xform_constants`` pins the angle-transform quadrature at
    ``2*(2*mboz+1)`` by ``2*(2*nboz+1)`` points, but the transform reads the
    counts back off ``constants`` for its Fourier normalization, so an integer
    refinement of the flattened ``(theta, zeta)`` grid is a drop-in reduction
    of the ``cos(m theta_B - n zeta_B)`` aliasing.
    """
    factor = int(oversample)
    if factor == 1:
        return constants, grids
    ntheta = factor * int(constants.ntheta)
    nzeta = factor * int(constants.nzeta) if int(constants.nzeta) > 1 else 1
    nu2_b = ntheta // 2 + 1
    # booz_xform's stellarator-symmetric grid spans theta in [0, pi] only
    # (nu2_b rows, boundary rows half-weighted and the normalization read
    # from nu2_b); the asymmetric grid spans the full circle (ntheta rows).
    nu3_b = ntheta if bool(constants.asym) else nu2_b
    theta = 2.0 * np.pi * np.arange(nu3_b) / ntheta
    zeta = 2.0 * np.pi * np.arange(nzeta) / (nzeta * int(nfp))
    return (
        dataclasses.replace(constants, ntheta=ntheta, nzeta=nzeta,
                            nu2_b=nu2_b),
        dataclasses.replace(grids,
                            theta_grid=jnp.asarray(np.repeat(theta, nzeta)),
                            zeta_grid=jnp.asarray(np.tile(zeta, nu3_b))),
    )


def _build_plan_tables(grids, xm, xn, xm_nyq, xn_nyq, *, constants, trig_f32):
    """The per-resolution trig/mode tables, op by op under the caller's context.

    The body is upstream's own ``_build_tables`` — the very function a
    plan-less kernel call runs inside the caller's trace — so vmex carries
    no trig table of its own.  Deliberately *not* a jit lane: staging the
    build as one fused program is ~5x cheaper to run (measured 419 ms ->
    83 ms cold, 4 ms warm at ``mboz = nboz = 16, oversample = 2``) but XLA
    then contracts the two products of ``cos m cos n + sin m sin n`` and the
    spectrum moves in the last bit (measured 1.0e-16 absolute on
    ``bmnc_b``).  Building it the way the plan-less kernel did keeps the
    adoption exactly bit-identical, which is worth more than a fraction of a
    second paid once per resolution per process.
    """
    from booz_xform_jax.jax_api import _build_tables

    return _build_tables(constants, grids, xm, xn, xm_nyq, xn_nyq,
                         trig_f32=trig_f32)


#: Bounded content-keyed registry of transform plans (the
#: ``implicit._canonical_config`` precedent): equal content is the same
#: ``BoozerPlan`` instance, so the tables are built once per distinct
#: resolution per process and every objective evaluation and jit trace
#: closes over the same arrays.  Eviction only costs a rebuild.  The bound
#: matches upstream's own plan cache; one plan at vmex's default
#: ``mboz = nboz = 16, oversample = 2`` holds ~44 MiB of tables, so a
#: session that really does mix eight resolutions pays for them.
_PLAN_CACHE: "collections.OrderedDict[tuple, Any]" = collections.OrderedDict()
_PLAN_CACHE_MAX = 8


def _boozer_plan(*, nfp, asym, xm, xn, xm_nyq, xn_nyq, mboz, nboz,
                 oversample=1, surface_chunk="auto"):
    """The reusable ``booz_xform_jax`` plan for one static resolution.

    A plan-less kernel call rebuilds the trig/mode tables inside every
    trace, and XLA does not fold them away: the plan is worth 1.2-2.2x on
    the jitted transform and 1.1-1.2x on its gradient
    (``benchmarks/boozer_plan_adoption_m4.json``).  Upstream's
    ``prepare_booz_xform_plan`` builds only on the transform's own
    ``2*(2*mboz+1)`` by ``2*(2*nboz+1)`` quadrature and exposes no grid
    hook, so vmex's ``oversample`` refinement runs upstream's table builder
    itself, on the refined constants and grids
    (:func:`_build_plan_tables`); ``test_vmex_plan_matches_upstreams_own_at
    _oversample_one`` pins that reconstruction against
    ``prepare_booz_xform_plan``.  The plan holds an explicit
    ``BoozerConfig``, so upstream's ``BOOZ_XFORM_JAX_TRIG_F32`` environment
    default never reaches this path.  Built under compile-time evaluation so
    the tables are concrete constants even when the first call happens
    inside a trace; the registry key carries the x64 flag and the default
    device because the tables' dtype and placement are fixed at build time.
    """
    from booz_xform_jax.jax_api import (
        BoozerConfig,
        BoozerPlan,
        prepare_booz_xform_constants,
    )

    xm, xn, xm_nyq, xn_nyq = (
        np.asarray(a, dtype=np.int32) for a in (xm, xn, xm_nyq, xn_nyq))
    key = (int(nfp), bool(asym), int(mboz), int(nboz), int(oversample),
           surface_chunk, bool(jax.config.jax_enable_x64),
           repr(jax.config.jax_default_device),
           xm.tobytes(), xn.tobytes(), xm_nyq.tobytes(), xn_nyq.tobytes())
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        _PLAN_CACHE.move_to_end(key)
        return hit
    config = BoozerConfig(mboz=int(mboz), nboz=int(nboz),
                          surface_chunk=surface_chunk)
    with jax.ensure_compile_time_eval():
        constants, grids = prepare_booz_xform_constants(
            nfp=int(nfp), mboz=int(mboz), nboz=int(nboz), asym=bool(asym),
            xm=xm, xn=xn, xm_nyq=xm_nyq, xn_nyq=xn_nyq)
        constants, grids = _refine_booz_grids(constants, grids, oversample, nfp)
        tables = _build_plan_tables(grids, xm, xn, xm_nyq, xn_nyq,
                                    constants=constants,
                                    trig_f32=config.trig_f32)
    plan = BoozerPlan(config=config, constants=constants, grids=grids,
                      tables=tables)
    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > _PLAN_CACHE_MAX:
        _PLAN_CACHE.popitem(last=False)
    return plan


def _run_plan(plan, families):
    """``booz_xform_jax_impl`` on a plan; ``families`` are the stacked tables.

    The kernel reads its mode-list arguments only when building tables
    inline; with a plan they are dead, so pass ``None`` rather than a copy
    (a future upstream that reads them again fails loudly here).
    """
    from booz_xform_jax.jax_api import booz_xform_jax_impl

    return booz_xform_jax_impl(
        **families, xm=None, xn=None, xm_nyq=None, xn_nyq=None,
        constants=plan.constants, grids=plan.grids, plan=plan)


def _boozer_kernel_state(state, rt, *, rows, s_half, mboz, nboz, oversample,
                         surface_chunk):
    """State tables through booz_xform_jax's validated full transform.

    One code path for both parities: symmetric states pass only the cosine
    families (the kernel returns an all-zero ``bmns_b`` block), ``lasym``
    states add the independent sine families.
    """
    lasym = bool(rt.setup.lasym)
    tables = [boozer_input_tables(state, rt, int(row)) for row in rows]
    stack = lambda name: jnp.stack([table[name] for table in tables])  # noqa: E731
    first = tables[0]
    xm, xn = np.asarray(first["xm"]), np.asarray(first["xn"])
    # An axisymmetric state (no toroidal input harmonics) has exactly no
    # n != 0 Boozer content -- the Boozer relabelling of an axisymmetric
    # field is axisymmetric -- so drop the toroidal band outright.  This
    # keeps the historical mode-list contract (tokamak decks return no
    # toroidal harmonics) and skips the dead quadrature.
    if not np.any(xn):
        nboz = 0
    plan = _boozer_plan(
        nfp=int(rt.resolution.nfp), asym=lasym, xm=xm, xn=xn, xm_nyq=xm,
        xn_nyq=xn, mboz=mboz, nboz=nboz, oversample=oversample,
        surface_chunk=surface_chunk)
    families = dict(
        rmnc=stack("rmnc"), zmns=stack("zmns"), lmns=stack("lmns"),
        bmnc=stack("bmnc"), bsubumnc=stack("bsubumnc"),
        bsubvmnc=stack("bsubvmnc"), iota=stack("iota"))
    if lasym:
        families.update(
            rmns=stack("rmns"), zmnc=stack("zmnc"), lmnc=stack("lmnc"),
            bmns=stack("bmns"), bsubumns=stack("bsubumns"),
            bsubvmns=stack("bsubvmns"))
    out = _run_plan(plan, families)
    setup = rt.setup
    return {
        "bmnc_b": out["bmnc_b"], "bmns_b": out["bmns_b"],
        "xm_b": np.asarray(plan.grids.xm_b, dtype=float),
        "xn_b": np.asarray(plan.grids.xn_b, dtype=float),
        "iota_b": stack("iota"), "G_b": stack("G"), "I_b": stack("I"),
        "nfp": int(rt.resolution.nfp),
        "s_b": jnp.asarray(s_half, dtype=jnp.asarray(setup.s_full).dtype)[rows - 1],
        "psi_b": jnp.asarray(setup.psi_half)[rows],
        "psi_edge": jnp.asarray(setup.psi_edge),
    }


def boozer_spectrum_high_order(
    state,
    *,
    surfaces,
    mboz: int = 16,
    nboz: int = 16,
    asym: bool = False,
    ntheta: int | None = None,
    nzeta: int | None = None,
    surface_chunk: int | str = "auto",
) -> dict[str, Array]:
    """Transform continuous native surfaces with BOOZ_XFORM_JAX in memory.

    ``surface_chunk`` as in :func:`boozer_spectrum_state`.
    """
    surface_values = np.atleast_1d(np.asarray(surfaces, dtype=float))
    if np.any((surface_values <= 0.0) | (surface_values > 1.0)):
        raise ValueError("surfaces must satisfy 0 < s <= 1")
    tables = [
        high_order_boozer_input_tables(
            state,
            np.sqrt(surface),
            ntheta=ntheta,
            nzeta=nzeta,
        )
        for surface in surface_values
    ]
    first = tables[0]
    stack = lambda name: jnp.stack([table[name] for table in tables])  # noqa: E731
    plan = _boozer_plan(
        nfp=int(state.nfp), asym=bool(asym), xm=first["xm"], xn=first["xn"],
        xm_nyq=first["xm_nyq"], xn_nyq=first["xn_nyq"], mboz=mboz, nboz=nboz,
        surface_chunk=surface_chunk)
    out = _run_plan(plan, {
        name: stack(name) for name in (
            "rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota",
            "rmns", "zmnc", "lmnc", "bmns", "bsubumns", "bsubvmns")})
    return {
        "bmnc_b": out["bmnc_b"],
        "bmns_b": out["bmns_b"],
        "xm_b": np.asarray(plan.grids.xm_b, dtype=float),
        "xn_b": np.asarray(plan.grids.xn_b, dtype=float),
        "iota_b": stack("iota"),
        "G_b": out["bvco_b"],
        "I_b": out["buco_b"],
        "nfp": int(state.nfp),
        "s_b": jnp.asarray(surface_values),
    }


def boozer_spectrum_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces,
    mboz: int = 16,
    nboz: int = 16,
    oversample: int = 2,
    surface_chunk: int | str = "auto",
) -> dict[str, Array]:
    """Boozer ``|B|`` spectrum of selected surfaces, fully traceable.

    The jnp analogue of :func:`vmex.core.optimize.boozer_modes_from_wout`
    evaluated without a wout file: vmex snaps the requested surfaces and
    builds the wout-convention single-surface spectral tables from the
    solver's internal state (:func:`vmex.core.boozer_tables
    .boozer_input_tables`), then ``booz_xform_jax``'s jittable kernel runs
    the Boozer construction itself — generating potential, Boozer angles,
    and the angle-transform quadrature — for symmetric and ``lasym`` states
    through the same code path, on a plan built once per resolution
    (:func:`_boozer_plan`).

    ``surfaces`` are normalized-flux values snapped to the nearest half-mesh
    surfaces (one Boozer construction per requested value, duplicates kept so
    outputs align with ``surfaces``).  ``oversample`` refines booz_xform's
    pinned ``(theta, zeta)`` quadrature grid by an integer factor before the
    angle transform, reducing the aliasing of ``cos(m theta_B - n zeta_B)``
    products.  ``surface_chunk`` bounds how many surfaces one compiled
    contraction batches (``booz_xform_jax.BoozerConfig``): the default
    ``"auto"`` batches every requested surface into a single ``vmap``, the
    fastest schedule measured on CPU at objective sizes — at 8 and 16
    surfaces (``mboz = nboz = 16``, ``oversample = 2``, ``input.nfp1_QI``) a
    fixed chunk of 1, 2 or 4 cost 1.09-1.55x on the value and 1.32-2.52x on
    the gradient; only at 3 surfaces did a small chunk help the value
    (0.41x at chunk 1), and it cost 3.3x on the gradient there.  An integer
    bounds the per-chunk intermediates instead (``lax.map`` over fixed
    blocks with an edge-replicated tail), for a device that cannot hold the
    whole batch.  Per-surface arithmetic is identical either way; the
    schedule moves results at rounding level only (measured 0.0 at chunk 2
    and 4, 5.6e-16 absolute at chunk 1).

    Returns ``{bmnc_b (nsurf, nmodes), bmns_b, xm_b, xn_b (physical),
    iota_b, G_b, I_b, nfp, s_b, psi_b, psi_edge}``; ``bmns_b`` is the
    all-zero cosine partner for symmetric states, and ``psi_b``/``psi_edge``
    are the signed toroidal flux divided by ``2*pi``.
    ``bmnc_b/xm_b/xn_b/iota_b/G_b/I_b/nfp`` are the inputs of
    :func:`omnigenity_residual` and of
    :func:`vmex.core.optimize.quasi_isodynamic_residual`.
    """
    setup = rt.setup
    if int(oversample) < 1:
        raise ValueError("oversample must be >= 1")
    s = jnp.asarray(setup.s_full)
    ns = int(s.shape[0])
    if ns < 3:
        raise ValueError(f"boozer_spectrum_state needs ns >= 3, got ns = {ns}")

    # -- surface selection: nearest half-mesh rows (static, shape-only) -----
    s_half_np, rows = _nearest_half_mesh_rows(ns, surfaces)
    return _boozer_kernel_state(
        state, rt, rows=rows, s_half=s_half_np, mboz=mboz, nboz=nboz,
        oversample=oversample, surface_chunk=surface_chunk)


def boozer_bmnc_state(*args, **kwargs) -> dict[str, Array]:
    """Deprecated alias of :func:`boozer_spectrum_state`.

    Same signature and return contract (including the ``bmns_b`` block).
    The name changed when the in-repo symmetric FFT transform was retired in
    favor of booz_xform_jax's kernel for both parities.
    """
    warnings.warn(
        "vmex.core.omnigenity.boozer_bmnc_state is deprecated; call "
        "boozer_spectrum_state (identical signature and return contract)",
        DeprecationWarning, stacklevel=2)
    return boozer_spectrum_state(*args, **kwargs)


def boozer_bmnc_high_order(*args, **kwargs) -> dict[str, Array]:
    """Deprecated alias of :func:`boozer_spectrum_high_order`."""
    warnings.warn(
        "vmex.core.omnigenity.boozer_bmnc_high_order is deprecated; call "
        "boozer_spectrum_high_order (identical signature and return "
        "contract)",
        DeprecationWarning, stacklevel=2)
    return boozer_spectrum_high_order(*args, **kwargs)


# ---------------------------------------------------------------------------
# Omnigenity residual on Boozer |B| harmonics
# ---------------------------------------------------------------------------


def omnigenity_residual(
    *,
    bmnc_b,
    bmns_b=None,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    weights: Iterable[float] | None = None,
    nphi: int = 97,
    nalpha: int = 25,
    n_levels: int = 16,
    softness: float = 2.0e-2,
    well_weight: float = 1.0,
    extremum_weight: float = 1.0,
    squash_weight: float = 1.0,
) -> dict[str, Array]:
    """Smooth constructed-QI-target omnigenity residual (module docstring).

    ``|B|`` is synthesized from Boozer harmonics along field lines
    ``theta_B = alpha + iota * phi_B`` on ``nalpha`` labels over one field
    period (``nphi`` periodic points), normalized per surface to ``[0, 1]``
    by the surface extrema.  Each field line is split at its minimum into
    two monotone branch envelopes (periodic running maxima); the residual
    stacks the three Goodman-construction distances:

    - ``well``: bounce distance ``delta(alpha, B*) = d_left + d_right``
      (sigmoid occupancy integrals of the branch envelopes at ``n_levels``
      trapping levels, in field-period fraction units) minus its
      ``alpha``-average — Cary–Shasharina bounce-distance omnigenity;
    - ``extremum``: per-line min/max of ``|B|`` minus their
      ``alpha``-averages — poloidally closed extremal contours;
    - ``squash``: pointwise ``envelope - |B|`` monotonicity defect —
      one magnetic well per field period.

    All three vanish on an exactly QI field.  ``softness`` is the sigmoid
    level width in normalized ``|B|`` units.  Returns ``residuals1d`` (flat
    least-squares vector), ``total = sum(residuals1d**2)`` and diagnostics.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    bmns_b = (jnp.zeros_like(bmnc_b) if bmns_b is None
              else jnp.asarray(bmns_b, dtype=bmnc_b.dtype))
    xm_b = jnp.asarray(np.asarray(xm_b, dtype=float))
    xn_b = jnp.asarray(np.asarray(xn_b, dtype=float))
    iota_b = jnp.atleast_1d(jnp.asarray(iota_b, dtype=jnp.float64))
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if bmns_b.shape != bmnc_b.shape:
        raise ValueError("bmns_b must have the same shape as bmnc_b")
    if nphi < 8 or nalpha < 2 or n_levels < 2:
        raise ValueError("omnigenity residual needs nphi >= 8, nalpha >= 2, n_levels >= 2")
    nsurf = int(bmnc_b.shape[0])
    dtype = bmnc_b.dtype
    w_arr = jnp.ones((nsurf,), dtype=dtype) if weights is None else _as_1d(weights)
    if int(w_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as surfaces")
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    eps = jnp.maximum(jnp.asarray(float(softness), dtype=dtype),
                      jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))

    # -- |B| along field lines over one field period (periodic phi grid) -----
    period = 2.0 * np.pi / float(nfp)
    phi = jnp.asarray(period * np.arange(nphi) / nphi, dtype=dtype)
    alpha = jnp.asarray(2.0 * np.pi * np.arange(nalpha) / nalpha, dtype=dtype)
    theta = alpha[None, :, None] + iota_b[:, None, None] * phi[None, None, :]
    angle = (theta[..., None] * xm_b - phi[None, None, :, None] * xn_b)
    b = (jnp.einsum("sapm,sm->sap", jnp.cos(angle), bmnc_b)
         + jnp.einsum("sapm,sm->sap", jnp.sin(angle), bmns_b))

    bmin = jnp.min(b, axis=(1, 2), keepdims=True)
    bmax = jnp.max(b, axis=(1, 2), keepdims=True)
    bhat = (b - bmin) / jnp.maximum(bmax - bmin, tiny)

    # -- monotone branch envelopes about the per-line minimum ----------------
    imin = jnp.argmin(bhat, axis=-1)                          # (nsurf, nalpha)
    nk = nphi // 2 + 1
    offs = jnp.arange(nk, dtype=jnp.int32)
    idx_l = jnp.mod(imin[:, :, None] - offs[None, None, :], nphi)
    idx_r = jnp.mod(imin[:, :, None] + offs[None, None, :], nphi)
    raw_l = jnp.take_along_axis(bhat, idx_l, axis=-1)         # (nsurf, nalpha, nk)
    raw_r = jnp.take_along_axis(bhat, idx_r, axis=-1)
    env_l = jax.lax.cummax(raw_l, axis=raw_l.ndim - 1)
    env_r = jax.lax.cummax(raw_r, axis=raw_r.ndim - 1)

    sqrt_w = jnp.sqrt(w_arr)[:, None, None]
    pieces: list[jnp.ndarray] = []

    # -- bounce-distance uniformity (Cary-Shasharina / Goodman "shuffle") ----
    levels = jnp.linspace(0.0, 1.0, int(n_levels) + 2, dtype=dtype)[1:-1]
    occ_l = jax.nn.sigmoid((levels[None, None, None, :] - env_l[..., None]) / eps)
    occ_r = jax.nn.sigmoid((levels[None, None, None, :] - env_r[..., None]) / eps)
    delta = (jnp.sum(occ_l, axis=2) + jnp.sum(occ_r, axis=2)) / float(nphi)
    well_res = (delta - jnp.mean(delta, axis=1, keepdims=True)) * sqrt_w * float(well_weight)
    pieces.append(jnp.ravel(well_res) / np.sqrt(float(nalpha * n_levels)))

    # -- extremum alignment (poloidally closed B_min / B_max contours) -------
    line_min = env_l[..., 0]                                  # = bhat at the minimum
    line_max = jnp.maximum(env_l[..., -1], env_r[..., -1])
    ext = jnp.stack([line_min, line_max], axis=-1)            # (nsurf, nalpha, 2)
    ext_res = (ext - jnp.mean(ext, axis=1, keepdims=True)) * sqrt_w * float(extremum_weight)
    pieces.append(jnp.ravel(ext_res) / np.sqrt(float(2 * nalpha)))

    # -- single-well monotonicity (Goodman "squash" distance) ----------------
    squash = jnp.concatenate([env_l - raw_l, (env_r - raw_r)[..., 1:]], axis=-1)
    squash_res = squash * sqrt_w * float(squash_weight)
    pieces.append(jnp.ravel(squash_res) / np.sqrt(float(nalpha * (2 * nk - 1))))

    residuals1d = jnp.concatenate(pieces)
    return {
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "bhat": bhat,
        "delta": delta,
        "line_min": line_min,
        "line_max": line_max,
        "levels": levels,
        "phi": phi,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Objective-term wrapper (QuasisymmetryRatioResidual-style interface)
# ---------------------------------------------------------------------------


class QIResidual:
    """Traceable smooth quasi-isodynamic surrogate (module docstring).

    Composition of :func:`boozer_spectrum_state` (traceable Boozer ``|B|``
    spectrum on the requested surfaces) and :func:`omnigenity_residual`
    (a smooth level-set surrogate for the Goodman construction).  The
    interface mirrors
    :class:`~vmex.core.optimize.QuasisymmetryRatioResidual`: the instance
    is a :func:`~vmex.core.optimize.least_squares` objective term for
    both gradient modes — ``jac=None`` calls :meth:`J` on the converged
    :class:`~vmex.core.optimize.Equilibrium`, ``jac="implicit"`` picks up
    the traceable :meth:`residuals_state` vector (full pointwise
    Gauss-Newton geometry, exact implicit gradients).  Use
    :class:`vmex.core.qi.ConstructedQIResidual` when the optimized quantity
    must be the full squash-and-shuffle distance.

    Example::

        qi = QIResidual(np.linspace(0.25, 0.9, 4))
        result = least_squares([(qi, 0.0, 10.0), ...], inp, jac="implicit")
    """

    name = "qi"

    def __init__(
        self,
        surfaces,
        *,
        weights: Iterable[float] | None = None,
        mboz: int = 16,
        nboz: int = 16,
        oversample: int = 2,
        nphi: int = 97,
        nalpha: int = 25,
        n_levels: int = 16,
        softness: float = 2.0e-2,
        well_weight: float = 1.0,
        extremum_weight: float = 1.0,
        squash_weight: float = 1.0,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.weights = None if weights is None else np.asarray(list(weights), dtype=float)
        if self.weights is not None and self.weights.shape[0] != self.surfaces.shape[0]:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz = int(mboz)
        self.nboz = int(nboz)
        self.oversample = int(oversample)
        self.nphi = int(nphi)
        self.nalpha = int(nalpha)
        self.n_levels = int(n_levels)
        self.softness = float(softness)
        self.well_weight = float(well_weight)
        self.extremum_weight = float(extremum_weight)
        self.squash_weight = float(squash_weight)

    # -- traceable (state, runtime) evaluation --------------------------------

    def compute_state(self, state: SpectralState, rt: SolverRuntime) -> dict[str, Array]:
        """Full diagnostics dict (Boozer spectrum + residual pieces)."""
        booz = boozer_spectrum_state(
            state, rt, surfaces=self.surfaces, mboz=self.mboz, nboz=self.nboz,
            oversample=self.oversample)
        out = omnigenity_residual(
            bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"),
            xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], nfp=booz["nfp"], weights=self.weights,
            nphi=self.nphi, nalpha=self.nalpha, n_levels=self.n_levels,
            softness=self.softness, well_weight=self.well_weight,
            extremum_weight=self.extremum_weight, squash_weight=self.squash_weight)
        out.update(booz)
        return out

    def residuals_state(self, state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
        """Traceable flat residual vector with ``sum(r**2) = total_state``."""
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable scalar omnigenity objective ``sum(residuals**2)``."""
        return self.compute_state(state, rt)["total"]

    # -- Equilibrium entry points (jac=None objective term) -------------------

    def J(self, eq) -> jnp.ndarray:
        """Objective-term entry point for ``least_squares`` (residual vector)."""
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = J

    def residuals(self, eq) -> jnp.ndarray:
        """Alias of :meth:`J` (simsopt-style vocabulary)."""
        return self.J(eq)

    def total(self, eq) -> Array:
        """Scalar omnigenity objective of a converged equilibrium."""
        return self.total_state(eq.state, eq.runtime)
