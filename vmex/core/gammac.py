"""Differentiable fast-ion confinement proxy ``Gamma_c``.

``Gamma_c`` measures how far the contours of the second adiabatic invariant
``J`` deviate from flux surfaces: trapped particles whose bounce-averaged
drift has a radial component ride superbanana orbits out of the device, and
Velasco et al. 2021, eqs. (20)-(21), relate the prompt-loss fraction of
energetic ions approximately linearly to ``Gamma_c`` (more linearly still
to the ``|gamma_c*|`` variant).  The proxy
of Nemov, Kasilov, Kernbichler, Leitold, Phys. Plasmas 15, 052501 (2008),
equation 61, as organized by Velasco et al., Nucl. Fusion 61, 116059 (2021),
equation 16, is

.. math::

   \\gamma_c = \\frac{2}{\\pi}\\arctan
     \\frac{v_r}{v_p},
   \\qquad
   \\Gamma_c = \\frac{\\pi}{4\\sqrt 2}\\left\\langle
     \\int_{1/B_{\\max}}^{1/B}\\!\\mathrm d\\lambda\\,
     \\frac{B}{\\sqrt{1-\\lambda B}}\\,\\gamma_c^2\\right\\rangle,

where :math:`v_r` and :math:`v_p` are the bounce-averaged radial and
poloidal-tangential projections of the magnetic drift
:math:`\\mathbf v_M \\propto (1-\\lambda B/2)\\,
\\mathbf B\\times\\nabla B/B^3`, bounce averages run between bounce points,
and :math:`\\langle\\cdot\\rangle` is the flux-surface average.  Expanded
over wells ``w`` along field lines this is evaluated as

.. math::

   \\Gamma_c = \\frac{\\pi}{8\\sqrt 2}\\,
     \\frac{\\int\\mathrm d\\lambda\\sum_w (v\\tau_b\\,\\gamma_c^2)_w}
          {\\int \\mathrm dl/B},

averaging numerator and denominator over the sampled field-line labels.

VMEX evaluates the drift ratio in Nemov's own form, using DESC's rewrite of
Nemov's equations 21-22 into single-valued maps of periodic quantities
(``desc.compute._fast_ion``, on the bounce kernel of Unalmis et al.,
J. Plasma Phys. 92(3), 2026, doi:10.1017/S0022377826101652):

.. math::

   \\tan\\frac{\\pi\\gamma_c}{2} =
   \\frac{\\int\\mathrm dl\\, w\\,
          (\\mathbf B\\times\\nabla B)\\cdot\\nabla\\psi/B^3}
        {\\big(|\\nabla\\rho|\\,\\Vert\\mathbf e_\\alpha\\Vert\\big)_{B_{\\min}}
         \\int \\mathrm dl\\,\\big[w\\,\\partial_\\rho B|_{\\vartheta,\\phi}
         + \\sqrt{1-\\lambda B}\\,K\\big]/B},

with :math:`w = (1-\\lambda B/2)/\\sqrt{1-\\lambda B}`, PEST angles
:math:`(\\vartheta,\\phi)`, :math:`\\rho=\\sqrt{s}`,
:math:`K = \\iota_\\rho\\,(\\nabla\\psi\\times\\mathbf b)\\cdot\\nabla\\phi
- 2\\,\\partial_\\rho B|_{\\vartheta,\\phi}
+ B\\,\\partial_\\rho B^\\phi|_{\\vartheta,\\phi}/B^\\phi`, and the
surface-tangency factor interpolated to the field minimum of each well.
Unlike the literal line-integral form of Velasco's equations 14-15, whose
poloidal drift carries a secular shear term that grows with the sampled
line length, this form is built from periodic quantities only and converges
as transits are added — the reason it is also DESC's default ``GammaC``
variant.  Velasco's :math:`\\gamma_c^*` differs from Nemov's only by the
tangency factor in the arctan argument (Velasco 2021, footnote to eq. 14).

Field lines are the PEST lines of :mod:`vmex.core.stability` /
:mod:`vmex.core.turbulence`; every geometric ingredient is an exact spectral
point evaluation with JAX AD, and the singular bounce integrals reuse the
drift kernels of :func:`vmex.core.bounce.bounce_action`.  Everything is jnp:
the proxy is traceable and differentiable end to end.  Stellarator-symmetric
states with ``iota != 0`` on the target surfaces only, inherited from the
field-line parameterization.

Two derivative semantics, named explicitly rather than switched by options:
:class:`GammaC` is the hard physical diagnostic — its *value* is the number
to report, but its discretized boundary derivative is exact-yet-divergent
under grid refinement (branch noise; the measured ladder is pinned in
``tests/test_gammac.py``) — and :class:`GammaCSmooth` is the optimization
surrogate: the same wells and quadrature with the two measured noise
structures (separatrix bounce-time curvature and superbanana drift-angle
corners) regularized on a fixed, annealable ``temperature`` scale, giving a
boundary derivative that is stable in sign and magnitude under the same
refinement ladder.  A matched-topology tracked variant
(``GammaCTracked``) can join them without changing either meaning.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from .bounce import bounce_action
from .solver import SolverRuntime, SpectralState
from .stability import (
    _ballooning_context, _pest_lambda, _surface_closures, _surface_tables,
    _theta_vmec_from_pest,
)

Array = Any

__all__ = [
    "GammaC",
    "GammaCSmooth",
    "gamma_c_from_fieldlines",
    "gamma_c_from_wout",
    "gamma_c_smooth_from_fieldlines",
    "gamma_c_smooth_state",
    "gamma_c_state",
]


def gamma_c_from_fieldlines(
    *,
    bmag,
    radial_drift,
    radial_gradient,
    drift_correction,
    tangency,
    dl_dx,
    length,
    pitch,
    pitch_weights,
    max_wells: int | None = None,
    quadrature_order: int = 32,
) -> dict[str, Array]:
    """Nemov ``Gamma_c`` of one flux surface from sampled field lines.

    ``bmag`` has shape ``(nline, nx)`` on a uniform bounded grid covering
    ``[0, length]`` in the line parameter ``x``, with arc-length element
    ``dl_dx``.  The remaining along-line arrays are the module-docstring
    ingredients: ``radial_drift`` is
    ``(B x grad B) . grad psi / B^3``, ``radial_gradient`` is
    ``dB/drho at fixed PEST angles / B``, ``drift_correction`` is ``K / B``,
    and ``tangency`` is ``|grad rho| ||e_alpha||``.  ``pitch`` and
    ``pitch_weights`` are quadrature nodes and weights of the lambda
    integral.  All complete wells contribute; ``excluded_fraction`` reports
    the measure lost to hard ``overflow`` exclusion so callers can reject
    an under-provisioned ``max_wells``.
    """
    return _wells_assembly(
        bmag=bmag, radial_drift=radial_drift,
        radial_gradient=radial_gradient, drift_correction=drift_correction,
        tangency=tangency, dl_dx=dl_dx, length=length, pitch=pitch,
        pitch_weights=pitch_weights, max_wells=max_wells,
        quadrature_order=quadrature_order, kernel_floor=None)


def gamma_c_smooth_from_fieldlines(
    *,
    bmag,
    radial_drift,
    radial_gradient,
    drift_correction,
    tangency,
    dl_dx,
    length,
    pitch,
    pitch_weights,
    max_wells: int | None = None,
    quadrature_order: int = 32,
    temperature: float = 0.15,
) -> dict[str, Array]:
    """Derivative-safe smooth surrogate of :func:`gamma_c_from_fieldlines`.

    Same ingredient contract, well detection, and sine-mapped bounce
    quadrature as the hard diagnostic — including its exact per-well
    cancellations, so the axisymmetric limit stays at quadrature noise —
    with the hard-branch noise sources of the discretized boundary
    derivative regularized on one scale ``delta`` (a quadrature parameter,
    held fixed under differentiation like the pitch grid):

    - **capped bounce kernels**: the inverse parallel speed becomes
      ``1/sqrt(max(1 - lambda B, 0) + delta)`` (the ``kernel_floor`` of
      :func:`vmex.core.bounce.bounce_action`).  Near a separatrix the exact
      bounce time diverges logarithmically in the reflecting level, so a
      pitch node landing at distance ``d`` from a barrier top contributes a
      ``1/d`` derivative term whose sign depends on which side the node
      fell — the sign-flipping refinement ladder measured for the hard
      gradient.  The floor caps that curvature at scale ``delta``, which
      the pitch grid resolves, so the discretized derivative converges;
    - the same floor also removes the hard **birth/death jump**: the exact
      bounce time of a well born at a quadratic ``|B|`` minimum starts at the
      finite oscillator period (amplitude-independent), so the hard value
      jumps when a boundary variation creates or destroys a well — the
      birth/death events of the well-connection graph of Ochs, J. Plasma
      Phys. (2025).  With the floor, a newborn well's bounce time is
      ``~ width/sqrt(delta) -> 0``, so wells fade in and out continuously.

    ``delta = temperature * (B_max - B_min)/B_max`` of the sampled lines —
    a fixed fraction of the trapped range, with softmax-weighted extrema so
    the scale is differentiable (see the note at its construction: freezing
    it under ``stop_gradient`` left the AD slope at 0.37x of the true FD
    slope of the objective, and the missing coherent term let branch noise
    set the through-the-solve sign).  Deliberately **not** tied to the grid
    spacings: a per-resolution ``delta`` makes every refinement rung
    differentiate a different functional, and the measured li383 ladder
    stayed sign-erratic that way; a fixed fraction converges (the
    regression test pins it).  The resolution must resolve the smoothing
    instead: ``num_pitch * temperature >~ 4`` keeps the pitch spacing
    inside ``delta`` (the default pairing 0.15 x 32 satisfies it).
    Annealing = re-optimizing at a lower ``temperature`` with a
    correspondingly finer budget; the surrogate converges to the hard
    ``Gamma_c`` in that joint limit.
    """
    bmag = jnp.asarray(bmag)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    # Softmax-weighted extrema, not hard min/max: differentiable (the FD
    # consistency of the objective includes d(delta)/dp — freezing it under
    # stop_gradient left AD at 0.37x of the true FD slope, measured), and
    # exactly stellarator-reflection symmetric where hard extrema send the
    # whole cotangent to one member of each mirror pair.  The weighted
    # average is a continuum functional of the sampled lines, so refinement
    # converges it (an unnormalized logsumexp would inflate the range by
    # softscale * log(nsamples) — resolution-dependent).
    softscale = 0.01 * jnp.mean(bmag)
    weight_max = jax.nn.softmax(jnp.ravel(bmag) / softscale)
    weight_min = jax.nn.softmax(-jnp.ravel(bmag) / softscale)
    b_max = jnp.vdot(weight_max, jnp.ravel(bmag))
    b_min = jnp.vdot(weight_min, jnp.ravel(bmag))
    delta = jnp.asarray(temperature, bmag.dtype) * (b_max - b_min) / b_max
    return _wells_assembly(
        bmag=bmag, radial_drift=radial_drift,
        radial_gradient=radial_gradient, drift_correction=drift_correction,
        tangency=tangency, dl_dx=dl_dx, length=length, pitch=pitch,
        pitch_weights=pitch_weights, max_wells=max_wells,
        quadrature_order=quadrature_order, kernel_floor=delta,
        corner_floor=float(temperature))


def _wells_assembly(
    *,
    bmag,
    radial_drift,
    radial_gradient,
    drift_correction,
    tangency,
    dl_dx,
    length,
    pitch,
    pitch_weights,
    max_wells,
    quadrature_order,
    kernel_floor,
    corner_floor=None,
) -> dict[str, Array]:
    """Shared well detection, drift kernels, and ``Gamma_c`` assembly.

    ``kernel_floor=None`` is the exact hard diagnostic; a positive floor
    (with its ``corner_floor`` companion) is the smooth surrogate.  One
    body, so the two lanes cannot drift apart in anything but the
    deliberate smoothing.
    """
    bmag = jnp.asarray(bmag)
    if bmag.ndim != 2:
        raise ValueError("bmag must have shape (nline, nx)")
    pitch = jnp.atleast_1d(jnp.asarray(pitch, dtype=bmag.dtype))
    weights = jnp.atleast_1d(jnp.asarray(pitch_weights, dtype=bmag.dtype))
    if pitch.shape != weights.shape:
        raise ValueError("pitch_weights must have the same length as pitch")
    dl_dx = jnp.broadcast_to(jnp.asarray(dl_dx, dtype=bmag.dtype), bmag.shape)

    drift_integrands = {"radial": radial_drift, "gradient": radial_gradient}
    parallel_integrands = {"correction": drift_correction}
    if corner_floor is not None:
        # L1 companions of the same kernels: the per-cell drift magnitude
        # scale for the superbanana-corner regularization below.
        drift_integrands["absradial"] = jnp.abs(jnp.asarray(radial_drift))
        drift_integrands["absgradient"] = jnp.abs(jnp.asarray(radial_gradient))
        parallel_integrands["abscorrection"] = jnp.abs(
            jnp.asarray(drift_correction))
    out = bounce_action(
        bmag, pitch, dl_dphi=dl_dx, length=length, periodic=False,
        max_wells=max_wells, quadrature_order=quadrature_order,
        kernel_floor=kernel_floor,
        drift_integrands=drift_integrands,
        parallel_integrands=parallel_integrands,
        argmin_integrands={"tangency": tangency})
    keep = out["well_mask"] & ~out["overflow"][..., None]

    def masked(name):
        return jnp.where(keep, out[name], 0.0)

    v_tau = masked("bounce_time")
    numerator = masked("drift_radial")
    denominator = masked("argmin_tangency") * (
        masked("drift_gradient") + masked("parallel_correction"))
    if corner_floor is None:
        safe = jnp.where(denominator != 0.0, denominator, 1.0)
        gamma = (2.0 / jnp.pi) * jnp.arctan(
            jnp.where(denominator != 0.0, numerator / safe, 0.0))
    else:
        # Superbanana corners: where the radial AND tangential bounce-drift
        # integrals vanish together, d(arctan(N/D))/dp ~ 1/|(N, D)|, and a
        # well-cell passing near that codim-2 origin dominates the gradient
        # with an arbitrary sign (measured: +-4 spikes in a delta sweep, on
        # a total of ~0.3).  gamma**2 is even, so the denominator magnitude
        # is floored at ``corner_floor`` times the cell's own L1 drift
        # scale — the same bounce integrals of the |integrands| — which is
        # what N and D would be without cancellation.  gamma is therefore
        # touched only where the bounce average cancels by more than
        # ~1/corner_floor, exactly the near-superbanana structure whose
        # drift angle the discretization cannot determine stably; every
        # drift-dominated cell keeps its hard value, and small wells are
        # judged against their own scale, not a global one.  Stop-gradient
        # as for every quadrature parameter.
        magnitude = masked("drift_absradial") + masked("argmin_tangency") * (
            masked("drift_absgradient") + masked("parallel_abscorrection"))
        sigma = corner_floor * magnitude
        gamma = (2.0 / jnp.pi) * jnp.arctan(
            numerator / jnp.sqrt(
                denominator**2 + sigma**2 + jnp.finfo(bmag.dtype).tiny))
    # Wells drifting across the ends of the bounded trace would blink in and
    # out; the taper fades them over the outer half-transit on each side, and
    # the same window weights the normalizing length so the ratio stays the
    # ergodic-limit estimate.
    x = jnp.linspace(0.0, length, int(bmag.shape[-1]), dtype=bmag.dtype)
    ramp = jnp.minimum(jnp.asarray(length, dtype=bmag.dtype) / 2.0, jnp.pi)

    def window(position):
        edge = jnp.minimum(position, length - position)
        return jnp.sin(
            0.5 * jnp.pi * jnp.clip(edge / ramp, 0.0, 1.0)) ** 2

    center = jnp.where(keep, 0.5 * (out["well_left"] + out["well_right"]), 0.0)
    velocity_sum = jnp.sum(
        window(center) * v_tau * gamma * gamma, axis=-1)     # (nline, npitch)
    line_length = jnp.trapezoid(window(x) * dl_dx / bmag, x, axis=-1)
    gamma_c = (jnp.pi / (8.0 * jnp.sqrt(2.0))
               * jnp.sum(weights * jnp.mean(velocity_sum, axis=0))
               / jnp.mean(line_length))
    out.update({
        "gamma_c": gamma_c,
        "gamma_per_well": jnp.where(keep, gamma, jnp.nan),
        "line_length": line_length,
        "excluded_fraction": jnp.mean(out["overflow"]),
        "overflow_fraction": jnp.mean(out["overflow"]),
    })
    if kernel_floor is not None:
        out["epsilon"] = kernel_floor
    return out



def _make_gamma_point_fn(m: Array, xn: Array, tabs: dict, iota: Array,
                         diota: Array, phipf_j: Array):
    """Point-evaluation closure for one flux surface (Nemov drift set).

    Built on :func:`vmex.core.stability._surface_closures`, so the sine-parity
    spectra of an asymmetric state are carried without a second
    implementation.  Returns at ``q = (t, theta, phi)`` (evaluated at
    ``t = s - s_j = 0``) the tuple

    ``(|B|, B^phi, B x grad|B| . grad s, d|B|/ds|PEST, dB^phi/ds|PEST,
    (grad s x b) . grad phi, |grad s|, ||e_theta|| / (1 + lambda_theta))``

    where the PEST-fixed radial derivatives follow the moving VMEC angle
    ``dtheta/ds = -lambda_s / (1 + lambda_theta)`` and the last entry is
    ``||e_alpha||`` at fixed ``(rho, phi)``.
    """
    pos_fn, lam_fn, b_vector, modb_fn, bsupphi_fn = _surface_closures(
        m, xn, tabs, iota, diota, phipf_j)

    def point(q: Array):
        J = jax.jacfwd(pos_fn)(q)
        dual = jnp.linalg.inv(J)                      # rows: grad s, grad th, grad ph
        lam_g = jax.grad(lam_fn)(q)
        B = b_vector(q)
        modB = jnp.linalg.norm(B)
        dB = jax.grad(modb_fn)(q)                     # (d/ds, d/dth, d/dph) at fixed VMEC angles
        dG = jax.grad(bsupphi_fn)(q)
        grad_modB = dB[0] * dual[0] + dB[1] * dual[1] + dB[2] * dual[2]
        one_lam = 1.0 + lam_g[1]
        dtheta_ds = -lam_g[0] / one_lam               # fixed PEST angle, moving VMEC angle
        grad_s = dual[0]
        b = B / modB
        return (
            modB,
            phipf_j * one_lam / jnp.linalg.det(J),
            jnp.cross(B, grad_modB) @ grad_s,
            dB[0] + dB[1] * dtheta_ds,
            dG[0] + dG[1] * dtheta_ds,
            jnp.cross(grad_s, b) @ dual[2],
            jnp.linalg.norm(grad_s),
            jnp.linalg.norm(J[:, 1]) / one_lam,
        )

    return point


def _surface_rows(surfaces, ns: int) -> tuple[int, ...]:
    """Map normalized-flux values to distinct interior full-mesh rows."""
    values = np.atleast_1d(np.asarray(surfaces, dtype=float))
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(
            (values <= 0.0) | (values >= 1.0)):
        raise ValueError("surfaces must be finite and strictly inside (0, 1)")
    rows = tuple(
        min(max(int(round(value * (ns - 1))), 2), ns - 2) for value in values)
    if len(set(rows)) != len(rows):
        raise ValueError(
            f"surfaces resolve to duplicate radial rows at ns = {ns}")
    return rows


#: Per-surface outputs of :func:`_gamma_c_rows`.  Everything else the bounce
#: kernel builds stays inside the jit and is eliminated as dead code.
_ROW_FIELDS = ("gamma_c", "excluded_fraction", "overflow_fraction",
               "line_length", "pitch")


def _sampling_grids(dtype, nalpha: int, num_transit: int,
                    points_per_transit: int, num_pitch: int):
    """Field-line-label, along-line, and pitch-level grids of one budget.

    The along-line grid is centred on the field-line label, not ``[0, L]``:
    the stellarator reflection maps ``(alpha, x) -> (-alpha, -x)`` and the
    alpha set is closed under negation, so a window symmetric in ``x`` makes
    the estimator exactly invariant under it -- and ``Gamma_c`` is even while
    the sine-parity spectra are odd, so ``d(Gamma_c)/d(sine)`` then vanishes
    on a symmetric state as it must.  A one-sided window does not: measured,
    it left a spurious symmetry-breaking gradient at 55% of the physical one.
    The level nodes are an open midpoint rule, uniform in the reflecting
    level ``1/lambda`` rather than ``lambda`` (the Unalmis et al.
    pitch-sampling guidance); the open ends avoid the incomputable bounce
    integral at the global maximum.
    """
    length = 2.0 * np.pi * int(num_transit)
    alpha = jnp.asarray(
        2.0 * np.pi * np.arange(int(nalpha)) / int(nalpha), dtype=dtype)
    x = jnp.linspace(
        -0.5 * length, 0.5 * length,
        int(points_per_transit) * int(num_transit) + 1, dtype=dtype)
    level_nodes = jnp.asarray(
        (np.arange(int(num_pitch)) + 0.5) / int(num_pitch), dtype=dtype)
    return length, alpha, x, level_nodes


def _pitch_grid(bmag, level_nodes, num_pitch: int):
    """``(pitch, pitch_weights)`` across the trapped range of ``bmag``.

    The pitch nodes are a quadrature grid, not an observable, and the
    ``|B|`` extrema over a reflection-closed set of lines are attained at
    mirror-image PAIRS of points.  Differentiating through jnp.min/max
    sends the whole cotangent to one member of each pair, which breaks a
    symmetry Gamma_c has exactly: on a stellarator-symmetric state
    d(Gamma_c)/d(sine spectra) must vanish, and it came out at 17% of
    the physical gradient.  Worse, that term does not converge -- over
    num_pitch = 12/24/48/96 the violation ran 1.7e-1, 1.8e-1, 6.0e-3,
    1.6e-1, so it is an erratic subgradient with no limit, not a
    discretization term.  Holding the grid fixed under differentiation
    restores the identity to 1e-10 and moves the physical gradient by
    2e-4.
    """
    b_min = jax.lax.stop_gradient(jnp.min(bmag))
    b_max = jax.lax.stop_gradient(jnp.max(bmag))
    level = b_min + (b_max - b_min) * level_nodes
    return 1.0 / level, (b_max - b_min) / int(num_pitch) / level**2


def _line_ingredients(ctx: dict, j: int, zeta0: Array, alpha: Array,
                      x: Array) -> dict[str, Array]:
    """Sampled Nemov drift-set arrays on one full-mesh row's field lines.

    Returns the keyword arguments of :func:`gamma_c_from_fieldlines` /
    :func:`gamma_c_smooth_from_fieldlines` (all ``(nline, nx)``) plus the
    row ``iota``, so the hard diagnostic and the smooth surrogate share one
    geometry evaluation.
    """
    hs, psi_edge = ctx["hs"], ctx["psi_edge"]
    iota = 0.5 * (ctx["iotas"][j] + ctx["iotas"][j + 1])
    diota = (ctx["iotas"][j + 1] - ctx["iotas"][j]) / hs
    tabs = _surface_tables(ctx, j)
    point = _make_gamma_point_fn(
        ctx["m"], ctx["xn"], tabs, iota, diota, ctx["phipf"][j])
    lmns0, lmnc0 = _pest_lambda(tabs)

    def line(a):
        theta_star = a + x
        phi = zeta0 + x / iota
        theta_v = _theta_vmec_from_pest(
            theta_star, phi, lmns0, ctx["m"], ctx["xn"], lmnc0)
        q = jnp.stack([jnp.zeros_like(theta_v), theta_v, phi], axis=-1)
        return jax.vmap(point)(q)

    (modB, b_sup_phi, bxgb_gs, modb_s_pest, bsupphi_s_pest,
     gs_cross_b_gphi, grad_s_norm, e_alpha_norm) = jax.vmap(line)(alpha)

    two_rho = 2.0 * jnp.sqrt(ctx["s"][j])
    modb_r = two_rho * modb_s_pest                # d|B|/drho at fixed PEST angles
    correction = (
        two_rho * diota * psi_edge * gs_cross_b_gphi
        - 2.0 * modb_r + modB * (two_rho * bsupphi_s_pest) / b_sup_phi)
    return dict(
        bmag=modB,
        radial_drift=psi_edge * bxgb_gs / modB**3,
        radial_gradient=modb_r / modB,
        drift_correction=correction / modB,
        tangency=grad_s_norm / two_rho * e_alpha_norm,
        dl_dx=modB / jnp.abs(iota * b_sup_phi),
        iota=iota,
    )


def _rows_from_context(
    ctx: dict,
    zeta0: Array,
    *,
    rows: tuple[int, ...],
    nalpha: int,
    num_transit: int,
    points_per_transit: int,
    num_pitch: int,
    quadrature_order: int,
    max_wells: int,
) -> dict[str, Array]:
    """``Gamma_c`` of the full-mesh rows ``rows`` of one spectral context.

    ``ctx`` is the shared field-line context of
    :func:`vmex.core.stability._ballooning_context` — either from a live
    ``(state, runtime)`` pair or normalized from a WOUT file — so the live
    optimization path and the read-only plotting path run the identical
    numerics.  Everything that sets an array shape is static in the jitted
    wrappers below.
    """
    length, alpha, x, level_nodes = _sampling_grids(
        ctx["s"].dtype, nalpha, num_transit, points_per_transit, num_pitch)
    results = []
    iotas = []
    for j in rows:
        ing = _line_ingredients(ctx, j, jnp.asarray(zeta0, ctx["s"].dtype),
                                alpha, x)
        iotas.append(ing.pop("iota"))
        pitch, weights = _pitch_grid(ing["bmag"], level_nodes, num_pitch)
        results.append(gamma_c_from_fieldlines(
            **ing, length=length, pitch=pitch, pitch_weights=weights,
            max_wells=max_wells, quadrature_order=quadrature_order))

    stacked: dict[str, Array] = {
        name: jnp.stack([out[name] for out in results]) for name in _ROW_FIELDS
    }
    # The field-line map phi = x / iota is meaningless through iota ~ 0;
    # poison the result instead of returning a plausible number.
    iota_row = jnp.stack(iotas)
    stacked["gamma_c"] = jnp.where(
        jnp.abs(iota_row) > 1.0e-6, stacked["gamma_c"], jnp.nan)
    stacked["s"] = ctx["s"][jnp.asarray(rows)]
    stacked["iota"] = iota_row
    return stacked


_ROW_STATICS = ("rows", "nalpha", "num_transit", "points_per_transit",
                "num_pitch", "quadrature_order", "max_wells")

#: Traced context entries consumed by :func:`_rows_from_context`; the
#: remaining `_ballooning_context` keys (normalizations, pressure) are unused
#: here, and ``lasym`` is Python control flow, so it rides as a static.
_CTX_KEYS = ("s", "hs", "psi_edge", "iotas", "phipf", "m", "xn",
             "rmnc", "zmns", "lmns", "rmns", "zmnc", "lmnc")


@functools.partial(jax.jit, static_argnames=_ROW_STATICS)
def _gamma_c_rows(
    state: SpectralState, rt: SolverRuntime, zeta0: Array, *,
    rows, nalpha, num_transit, points_per_transit, num_pitch,
    quadrature_order, max_wells,
) -> dict[str, Array]:
    """Live-state ``Gamma_c`` rows, as one XLA executable.

    Module-level rather than a ``jax.jit`` at the call site, so every boundary
    iterate of an optimization stage reuses one compilation instead of
    re-tracing the spectral field-line machinery.  Measured on a three-surface
    nfp = 2 case: 0.96 s eager against 0.03 s here, values bit-identical.
    """
    return _rows_from_context(
        _ballooning_context(state, rt), zeta0, rows=rows, nalpha=nalpha,
        num_transit=num_transit, points_per_transit=points_per_transit,
        num_pitch=num_pitch, quadrature_order=quadrature_order,
        max_wells=max_wells)


@functools.partial(jax.jit, static_argnames=_ROW_STATICS + ("lasym",))
def _gamma_c_rows_from_tables(
    tables: dict, zeta0: Array, *, lasym: bool,
    rows, nalpha, num_transit, points_per_transit, num_pitch,
    quadrature_order, max_wells,
) -> dict[str, Array]:
    """WOUT-context ``Gamma_c`` rows; one executable per (shapes, settings)."""
    return _rows_from_context(
        dict(tables, lasym=lasym), zeta0, rows=rows, nalpha=nalpha,
        num_transit=num_transit, points_per_transit=points_per_transit,
        num_pitch=num_pitch, quadrature_order=quadrature_order,
        max_wells=max_wells)


#: Per-surface outputs of the smooth surrogate rows.
_SMOOTH_ROW_FIELDS = _ROW_FIELDS + ("epsilon",)


def _smooth_rows_from_context(
    ctx: dict,
    zeta0: Array,
    *,
    rows: tuple[int, ...],
    nalpha: int,
    num_transit: int,
    points_per_transit: int,
    num_pitch: int,
    quadrature_order: int,
    max_wells: int,
    temperature: float,
) -> dict[str, Array]:
    """Smooth-surrogate rows on the same grids and geometry as the hard rows."""
    length, alpha, x, level_nodes = _sampling_grids(
        ctx["s"].dtype, nalpha, num_transit, points_per_transit, num_pitch)
    results = []
    iotas = []
    for j in rows:
        ing = _line_ingredients(ctx, j, jnp.asarray(zeta0, ctx["s"].dtype),
                                alpha, x)
        iotas.append(ing.pop("iota"))
        pitch, weights = _pitch_grid(ing["bmag"], level_nodes, num_pitch)
        results.append(gamma_c_smooth_from_fieldlines(
            **ing, length=length, pitch=pitch, pitch_weights=weights,
            max_wells=max_wells, quadrature_order=quadrature_order,
            temperature=temperature))
    stacked: dict[str, Array] = {
        name: jnp.stack([out[name] for out in results])
        for name in _SMOOTH_ROW_FIELDS
    }
    iota_row = jnp.stack(iotas)
    stacked["gamma_c"] = jnp.where(
        jnp.abs(iota_row) > 1.0e-6, stacked["gamma_c"], jnp.nan)
    stacked["s"] = ctx["s"][jnp.asarray(rows)]
    stacked["iota"] = iota_row
    return stacked


@functools.partial(
    jax.jit, static_argnames=_ROW_STATICS + ("temperature",))
def _gamma_c_smooth_rows(
    state: SpectralState, rt: SolverRuntime, zeta0: Array, *,
    rows, nalpha, num_transit, points_per_transit, num_pitch,
    quadrature_order, max_wells, temperature,
) -> dict[str, Array]:
    """Live-state smooth-surrogate rows, one executable per settings.

    ``temperature`` is static (an annealing stage changes it a handful of
    times per optimization, and a concrete value lets XLA fold the smoothing
    scales), everything shape-setting likewise.
    """
    return _smooth_rows_from_context(
        _ballooning_context(state, rt), zeta0, rows=rows, nalpha=nalpha,
        num_transit=num_transit, points_per_transit=points_per_transit,
        num_pitch=num_pitch, quadrature_order=quadrature_order,
        max_wells=max_wells, temperature=temperature)


def _validate_settings(nalpha, num_transit, points_per_transit, num_pitch):
    if nalpha < 2:
        raise ValueError("nalpha must be >= 2")
    if num_transit < 1 or points_per_transit < 16:
        raise ValueError("num_transit must be >= 1 and points_per_transit >= 16")
    if num_pitch < 2:
        raise ValueError("num_pitch must be >= 2")


def gamma_c_from_wout(
    wout,
    *,
    surfaces: Sequence[float] = (0.35, 0.6, 0.85),
    nalpha: int = 9,
    num_transit: int = 4,
    points_per_transit: int = 64,
    num_pitch: int = 32,
    quadrature_order: int = 32,
    max_wells: int | None = None,
    zeta0: float = 0.0,
) -> dict[str, Array]:
    """Evaluate ``Gamma_c`` from a VMEC-compatible WOUT, without a solve.

    ``wout`` is a :class:`~vmex.core.wout.WoutData` or a path accepted by
    :func:`vmex.read_wout`.  The WOUT tables are normalized to the live-state
    spectral context (the :mod:`vmex.core.turbulence` read-only route), so the
    numerics — field lines, drift kernels, pitch quadrature — are identical to
    :func:`gamma_c_state`; only the radial coefficient tables come from the
    file instead of the solver state.  Same arguments and return contract as
    :func:`gamma_c_state`.  Read-only: not the differentiable route.
    """
    from .turbulence import _wout_ballooning_context
    from .wout import read_wout

    if isinstance(wout, (str, Path)):
        wout = read_wout(str(wout))
    _validate_settings(nalpha, num_transit, points_per_transit, num_pitch)
    ctx = _wout_ballooning_context(wout)
    rows = _surface_rows(surfaces, int(ctx["ns"]))
    out = _gamma_c_rows_from_tables(
        {key: ctx[key] for key in _CTX_KEYS},
        jnp.asarray(float(zeta0)), lasym=bool(ctx["lasym"]), rows=rows,
        nalpha=int(nalpha), num_transit=int(num_transit),
        points_per_transit=int(points_per_transit), num_pitch=int(num_pitch),
        quadrature_order=int(quadrature_order),
        max_wells=16 * int(num_transit) if max_wells is None else int(max_wells))
    return {**out, "surface_rows": rows}


def gamma_c_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces: Sequence[float] = (0.35, 0.6, 0.85),
    nalpha: int = 9,
    num_transit: int = 4,
    points_per_transit: int = 64,
    num_pitch: int = 32,
    quadrature_order: int = 32,
    max_wells: int | None = None,
    zeta0: float = 0.0,
) -> dict[str, Array]:
    """Evaluate ``Gamma_c`` on flux surfaces of a converged equilibrium.

    ``surfaces`` are normalized-flux values mapped to the nearest interior
    full-mesh rows (the surface-selection convention of
    :func:`vmex.core.neoclassical.epsilon_effective_from_wout`).  Each
    surface samples ``nalpha`` field lines over ``num_transit`` poloidal turns
    at ``points_per_transit`` points per turn, centred on the field-line label
    so the stellarator reflection is an exact invariance of the estimator; the
    lambda integral uses ``num_pitch`` open-midpoint nodes across the trapped
    range ``[1/B_max, 1/B_min]`` of the sampled lines, held fixed under
    differentiation (see the note at the node construction).  ``max_wells``
    defaults to ``16 * num_transit`` slots per pitch level;
    ``overflow_fraction`` reports any spill.  Returns per-surface arrays led
    by ``gamma_c``.

    The absolute value carries roughly 10-20 % scatter at these resolutions --
    it is a comparative proxy, and the resolution belongs in any quoted
    number.  The numerics live in the jitted :func:`_gamma_c_rows`; only the
    validation and the row selection happen here, because those set shapes.
    """
    _validate_settings(nalpha, num_transit, points_per_transit, num_pitch)
    rows = _surface_rows(surfaces, int(np.shape(rt.setup.s_full)[0]))
    out = _gamma_c_rows(
        state, rt, jnp.asarray(float(zeta0)), rows=rows, nalpha=int(nalpha),
        num_transit=int(num_transit),
        points_per_transit=int(points_per_transit), num_pitch=int(num_pitch),
        quadrature_order=int(quadrature_order),
        max_wells=16 * int(num_transit) if max_wells is None else int(max_wells))
    return {**out, "surface_rows": rows}


def gamma_c_smooth_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces: Sequence[float] = (0.35, 0.6, 0.85),
    nalpha: int = 9,
    num_transit: int = 4,
    points_per_transit: int = 64,
    num_pitch: int = 32,
    quadrature_order: int = 32,
    max_wells: int | None = None,
    temperature: float = 0.15,
    zeta0: float = 0.0,
) -> dict[str, Array]:
    """Smooth-surrogate ``Gamma_c`` with a refinement-stable boundary gradient.

    Same sampling contract, arguments, and outputs as :func:`gamma_c_state`
    — the surfaces, field lines, pitch grid, wells, and Nemov drift
    ingredients are identical — with the hard-branch derivative-noise
    sources regularized on the resolution-tied scale of
    :func:`gamma_c_smooth_from_fieldlines` (returned per surface as
    ``epsilon``).  ``temperature`` scales that smoothing; refinement anneals
    it automatically.

    This is the **optimization** lane: its value converges to the hard
    ``Gamma_c`` as the grids refine, and its boundary derivative is stable
    in sign and magnitude under refinement where the hard diagnostic's is
    not (see :class:`GammaCSmooth` for the measured ladder).  Report the
    hard value; differentiate this one.
    """
    _validate_settings(nalpha, num_transit, points_per_transit, num_pitch)
    if not float(temperature) > 0.0:
        raise ValueError("temperature must be positive")
    rows = _surface_rows(surfaces, int(np.shape(rt.setup.s_full)[0]))
    out = _gamma_c_smooth_rows(
        state, rt, jnp.asarray(float(zeta0)), rows=rows, nalpha=int(nalpha),
        num_transit=int(num_transit),
        points_per_transit=int(points_per_transit), num_pitch=int(num_pitch),
        quadrature_order=int(quadrature_order),
        max_wells=16 * int(num_transit) if max_wells is None else int(max_wells),
        temperature=float(temperature))
    return {**out, "surface_rows": rows}


class GammaC:
    """Composable fast-ion confinement proxy, one ``Gamma_c`` row per surface.

    Nemov et al., Phys. Plasmas 15, 052501 (2008), eq. 61, organized as in
    Velasco et al., Nucl. Fusion 61, 116059 (2021), eq. 16 — see the module
    docstring for the formula and conventions.  ``residuals_state`` returns
    ``sqrt(weight) * Gamma_c`` rows for VMEX's least-squares interface, so
    the total cost is the weighted sum of ``Gamma_c**2``.  That square is
    the least-squares form, not a physical scaling: Velasco et al. 2021,
    eqs. (20)-(21), relate the prompt-loss fraction approximately linearly
    to ``Gamma_c`` (more linearly to the ``|gamma_c*|`` variant).
    Traceable in both gradient modes.

    **This is the hard physical diagnostic — its value is the number to
    report, but its boundary derivative is NOT a convergent quantity**: the
    discretized gradient is exact yet flips sign under grid refinement,
    because hard well detection and hard branch selection make the
    discretized ``Gamma_c`` piecewise with grid-dependent breakpoints (the
    measured ladder is in ``tests/test_gammac.py``).  Do not drive a
    boundary optimization with this class; use :class:`GammaCSmooth`, whose
    derivative is refinement-stable, and recompute this hard value to report
    the result.
    """

    name = "gamma_c"

    def __init__(
        self, surfaces=(0.35, 0.6, 0.85), *, weights: Iterable[float] | None = None,
        nalpha: int = 9, num_transit: int = 4, points_per_transit: int = 64,
        num_pitch: int = 32, quadrature_order: int = 32,
        max_wells: int | None = None, zeta0: float = 0.0,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        if (self.surfaces.size == 0 or np.any(~np.isfinite(self.surfaces))
                or np.any((self.surfaces <= 0.0) | (self.surfaces >= 1.0))
                or np.any(np.diff(self.surfaces) <= 0.0)):
            raise ValueError(
                "increasing surfaces strictly inside (0, 1) are required")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None:
            if self.weights.shape != self.surfaces.shape:
                raise ValueError("weights must have the same length as surfaces")
            if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0.0):
                raise ValueError("weights must be finite and non-negative")
        if min(int(nalpha), int(num_transit), int(points_per_transit),
               int(num_pitch), int(quadrature_order)) <= 0:
            raise ValueError("resolution arguments must be positive")
        self.nalpha, self.num_transit = int(nalpha), int(num_transit)
        self.points_per_transit = int(points_per_transit)
        self.num_pitch = int(num_pitch)
        self.quadrature_order = int(quadrature_order)
        self.max_wells = None if max_wells is None else int(max_wells)
        self.zeta0 = float(zeta0)

    def compute_state(self, state: SpectralState, rt: SolverRuntime) -> dict[str, Array]:
        """Return per-surface ``Gamma_c`` and sampling diagnostics."""
        return gamma_c_state(
            state, rt, surfaces=self.surfaces, nalpha=self.nalpha,
            num_transit=self.num_transit,
            points_per_transit=self.points_per_transit,
            num_pitch=self.num_pitch, quadrature_order=self.quadrature_order,
            max_wells=self.max_wells, zeta0=self.zeta0)

    def residuals_state(self, state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
        gamma_c = self.compute_state(state, rt)["gamma_c"]
        if self.weights is None:
            return gamma_c
        return jnp.sqrt(jnp.asarray(self.weights, dtype=gamma_c.dtype)) * gamma_c

    def total_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        rows = self.residuals_state(state, rt)
        return jnp.vdot(rows, rows)

    def J(self, eq) -> jnp.ndarray:
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = J
    residuals = J

    def total(self, eq) -> Array:
        return self.total_state(eq.state, eq.runtime)


class GammaCSmooth(GammaC):
    """Derivative-safe smooth surrogate of :class:`GammaC` for optimization.

    Same rows, weights, and least-squares contract as :class:`GammaC`, but
    ``compute_state`` evaluates :func:`gamma_c_smooth_state`: the same wells
    and bounce quadrature with the branch-noise sources of the boundary
    derivative regularized, so the derivative is stable in sign and
    magnitude under refinement of every sampling grid — the property the
    hard diagnostic's derivative measurably lacks.  The smoothing biases the
    value slightly (measured in ``tests/test_gammac.py``; it anneals away
    under refinement and preserves the tokamak << QA << unoptimized-3D
    ordering), so use this class inside the optimizer and always recompute
    the hard :class:`GammaC` value on the accepted result.

    The three derivative semantics of the plan are thus explicit in the
    class name: :class:`GammaC` is the hard value (no derivative promise
    across topology events), ``GammaCSmooth`` is the smooth annealed
    surrogate, and a matched-topology tracked variant can join them later
    without changing either meaning.  ``temperature`` multiplies the
    resolution-tied smoothing scales (see
    :func:`gamma_c_smooth_from_fieldlines`); annealing = optimizing stages
    at decreasing ``temperature``.
    """

    name = "gamma_c_smooth"

    def __init__(
        self, surfaces=(0.35, 0.6, 0.85), *, weights: Iterable[float] | None = None,
        nalpha: int = 9, num_transit: int = 4, points_per_transit: int = 64,
        num_pitch: int = 32, quadrature_order: int = 32,
        max_wells: int | None = None, temperature: float = 0.15,
        zeta0: float = 0.0,
    ):
        super().__init__(
            surfaces, weights=weights, nalpha=nalpha, num_transit=num_transit,
            points_per_transit=points_per_transit, num_pitch=num_pitch,
            quadrature_order=quadrature_order, max_wells=max_wells,
            zeta0=zeta0)
        if not float(temperature) > 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def compute_state(self, state: SpectralState, rt: SolverRuntime) -> dict[str, Array]:
        """Return per-surface smooth ``Gamma_c`` and its smoothing scale."""
        return gamma_c_smooth_state(
            state, rt, surfaces=self.surfaces, nalpha=self.nalpha,
            num_transit=self.num_transit,
            points_per_transit=self.points_per_transit,
            num_pitch=self.num_pitch, quadrature_order=self.quadrature_order,
            max_wells=self.max_wells, temperature=self.temperature,
            zeta0=self.zeta0)
