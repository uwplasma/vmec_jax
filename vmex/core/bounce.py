"""Differentiable second-adiabatic-invariant kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

import jax.numpy as jnp

Array = Any

__all__ = [
    "bounce_action",
    "bounce_action_from_boozer",
    "trace_boozer_field_lines",
]


def _interp_uniform(values, x, *, length, periodic):
    """Linear interpolation on the uniform grid used by :func:`bounce_action`."""
    values = jnp.asarray(values)
    n = int(values.shape[-1])
    flat = values.reshape((-1, n))
    query = x.reshape((flat.shape[0], -1))
    intervals = n if periodic else n - 1
    u = query * (intervals / length)
    if periodic:
        u = jnp.mod(u, intervals)
        lo = jnp.floor(u).astype(jnp.int32)
        hi = jnp.mod(lo + 1, n)
    else:
        u = jnp.clip(u, 0.0, intervals)
        lo = jnp.minimum(jnp.floor(u).astype(jnp.int32), n - 2)
        hi = lo + 1
    fraction = u - lo
    left = jnp.take_along_axis(flat, lo, axis=1)
    right = jnp.take_along_axis(flat, hi, axis=1)
    return (left + fraction * (right - left)).reshape(x.shape)


def bounce_action(
    bmag,
    pitch,
    *,
    dl_dphi=None,
    length: float = 2.0 * np.pi,
    periodic: bool = True,
    max_wells: int | None = None,
    quadrature_order: int = 64,
    topology_tolerance: float = 1.0e-6,
    drift_integrands: dict[str, Any] | None = None,
    parallel_integrands: dict[str, Any] | None = None,
    argmin_integrands: dict[str, Any] | None = None,
    kernel_floor: Any | None = None,
) -> dict[str, Array]:
    """Compute ``J = 2 int sqrt(1 - pitch * B) dl`` in every magnetic well.

    ``bmag`` has shape ``(..., nphi)`` on a uniform grid beginning at zero.
    A periodic grid omits its duplicate endpoint; a bounded trace includes it.
    ``pitch`` is a shared one-dimensional array in inverse-tesla units.
    ``dl_dphi`` supplies the line element and defaults to one.

    Crossings are retained in fixed-width arrays and paired without Python
    control flow. Gauss-Legendre quadrature uses a sine map whose Jacobian
    removes the square-root endpoint singularity. The result is exact under
    JIT, JVP, and VJP while the well topology is unchanged.

    Invalid slots contain NaN, never a plausible zero. Use ``well_mask`` for
    complete wells and ``usable_mask`` to exclude levels near a local minimum
    (``marginal``) or maximum (``merged``). ``truncated`` marks a bounded
    trace that cuts a well. At most eight wells are retained by default;
    ``overflow`` requests a larger explicit ``max_wells``.

    The three optional integrand mappings take names to along-line arrays
    broadcastable to ``bmag`` and add the per-well bounce kernels of Nemov
    et al. (2008): ``drift_<name>`` is
    ``int f (1 - pitch B / 2) / sqrt(1 - pitch B) dl``, ``parallel_<name>``
    is ``int f sqrt(1 - pitch B) dl``, and ``argmin_<name>`` is ``f`` at the
    field minimum of the well (a differentiable softmin average over the
    quadrature nodes). Any of them also adds ``bounce_time``,
    ``v tau = 2 int dl / sqrt(1 - pitch B)``. The sine-map Jacobian cancels
    the inverse-square-root bounce-point singularity.

    ``kernel_floor`` (``None`` for the exact kernels) adds a positive floor
    ``delta`` under the inverse-speed square root,
    ``1/sqrt(max(1 - pitch B, 0) + delta)``, entering ``bounce_time`` and
    the ``drift_*`` kernels.  Near a separatrix the exact bounce time
    diverges logarithmically in the reflecting level, so its derivative with
    respect to any parameter the barrier top depends on is ``1/distance`` on
    a pitch node that lands close — the branch-noise term of hard
    discretizations.  The floor caps that curvature at scale ``delta`` in
    ``1 - pitch B`` while leaving kernels at depth ``>> delta`` untouched.
    It also removes the birth/death jump of well topology events: a newborn
    well's floored bounce time starts at ``~ width/sqrt(delta) -> 0``
    instead of the finite oscillator period.
    """
    bmag = jnp.asarray(bmag)
    pitch = jnp.atleast_1d(jnp.asarray(pitch, dtype=bmag.dtype))
    if bmag.ndim < 1 or int(bmag.shape[-1]) < (3 if periodic else 4):
        raise ValueError("bmag needs at least three periodic or four bounded samples")
    if pitch.ndim != 1:
        raise ValueError("pitch must be scalar or one-dimensional")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be >= 2")

    n = int(bmag.shape[-1])
    n_intervals = n if periodic else n - 1
    if max_wells is None:
        max_wells = min(8, max(1, n_intervals // 2))
    if not 1 <= int(max_wells) <= n_intervals:
        raise ValueError("max_wells must lie between 1 and the interval count")
    max_wells = int(max_wells)

    lead_shape = bmag.shape[:-1]
    n_lines = int(np.prod(lead_shape)) if lead_shape else 1
    b = bmag.reshape((n_lines, n))
    dl = jnp.ones_like(b) if dl_dphi is None else jnp.broadcast_to(
        jnp.asarray(dl_dphi, dtype=b.dtype), bmag.shape).reshape((n_lines, n))
    length_arr = jnp.asarray(length, dtype=b.dtype)
    step = length_arr / n_intervals

    if periodic:
        left_b, right_b = b, jnp.roll(b, -1, axis=-1)
        left_x = step * jnp.arange(n, dtype=b.dtype)
    else:
        left_b, right_b = b[:, :-1], b[:, 1:]
        left_x = step * jnp.arange(n - 1, dtype=b.dtype)

    threshold = 1.0 / pitch
    bi = left_b[:, None, :]
    bj = right_b[:, None, :]
    level = threshold[None, :, None]
    down = (bi >= level) & (bj < level)
    up = (bi < level) & (bj >= level)
    denominator = bj - bi
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    root = left_x[None, None, :] + step * (level - bi) / safe_denominator
    infinity = jnp.asarray(jnp.inf, dtype=b.dtype)
    down_roots = jnp.sort(jnp.where(down, root, infinity), axis=-1)
    up_roots = jnp.sort(jnp.where(up, root, infinity), axis=-1)

    well_left = down_roots[..., :max_wells]
    offset = (up_roots[..., 0] < down_roots[..., 0]).astype(jnp.int32)
    up_index = jnp.arange(max_wells) + offset[..., None]
    in_range = up_index < n_intervals
    well_right = jnp.take_along_axis(
        up_roots, jnp.minimum(up_index, n_intervals - 1), axis=-1)
    well_right = jnp.where(in_range, well_right, infinity)
    if periodic:
        well_right = jnp.where(
            jnp.isfinite(well_right), well_right, up_roots[..., :1] + length_arr)
    well_mask = (
        jnp.isfinite(well_left) & jnp.isfinite(well_right)
        & (well_right > well_left)
    )

    left = jnp.where(well_mask, well_left, 0.0)
    right = jnp.where(well_mask, well_right, 0.0)
    midpoint = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    angle = jnp.asarray(0.5 * np.pi * nodes, dtype=b.dtype)
    weight = jnp.asarray(0.5 * np.pi * weights, dtype=b.dtype)
    x = midpoint[..., None] + half_width[..., None] * jnp.sin(angle)
    bq = _interp_uniform(b, x, length=length_arr, periodic=periodic)
    dlq = _interp_uniform(dl, x, length=length_arr, periodic=periodic)
    parallel_square = jnp.where(
        well_mask[..., None],
        1.0 - pitch[None, :, None, None] * bq,
        1.0,
    )
    parallel_speed = jnp.sqrt(jnp.maximum(parallel_square, 0.0))
    action = 2.0 * jnp.sum(
        weight * parallel_speed * dlq
        * half_width[..., None] * jnp.cos(angle),
        axis=-1,
    )

    extras: dict[str, Array] = {}
    if (drift_integrands is not None or parallel_integrands is not None
            or argmin_integrands is not None):
        def line_values(values):
            return _interp_uniform(
                jnp.broadcast_to(
                    jnp.asarray(values, dtype=b.dtype), bmag.shape,
                ).reshape((n_lines, n)),
                x, length=length_arr, periodic=periodic)

        if kernel_floor is None:
            # Inside a well the piecewise-linear interpolant stays below the
            # crossing level, so parallel_speed vanishes only at roundoff;
            # dropped nodes there mimic the exact zero of the mapped integrand.
            inverse_speed = jnp.where(
                parallel_speed > 0.0,
                1.0 / jnp.maximum(parallel_speed, jnp.finfo(b.dtype).tiny), 0.0)
        else:
            inverse_speed = 1.0 / jnp.sqrt(
                jnp.maximum(parallel_square, 0.0)
                + jnp.asarray(kernel_floor, dtype=b.dtype))
        measure = weight * dlq * half_width[..., None] * jnp.cos(angle)
        extras["bounce_time"] = 2.0 * jnp.sum(inverse_speed * measure, axis=-1)
        drift_weight = (
            1.0 - 0.5 * pitch[None, :, None, None] * bq) * inverse_speed
        for name, values in (drift_integrands or {}).items():
            extras[f"drift_{name}"] = jnp.sum(
                line_values(values) * drift_weight * measure, axis=-1)
        for name, values in (parallel_integrands or {}).items():
            extras[f"parallel_{name}"] = jnp.sum(
                line_values(values) * parallel_speed * measure, axis=-1)
        if argmin_integrands is not None:
            # A hard node argmin jumps between quadrature nodes as parameters
            # move; the softmin weight keeps the selection differentiable and
            # concentrated on the field minimum of the well.
            b_floor = jnp.min(bq, axis=-1, keepdims=True)
            spread = jnp.max(bq, axis=-1, keepdims=True) - b_floor
            temperature = 0.01 * spread + jnp.finfo(b.dtype).eps
            softmin = jnp.exp((b_floor - bq) / temperature)
            softmin = softmin / jnp.sum(softmin, axis=-1, keepdims=True)
            for name, values in argmin_integrands.items():
                extras[f"argmin_{name}"] = jnp.sum(
                    line_values(values) * softmin, axis=-1)

    scale = jnp.maximum(jnp.max(b, axis=-1) - jnp.min(b, axis=-1),
                        jnp.finfo(b.dtype).eps)
    previous = jnp.roll(b, 1, axis=-1)
    following = jnp.roll(b, -1, axis=-1)
    local_min = (b <= previous) & (b <= following)
    local_max = (b >= previous) & (b >= following)
    if not periodic:
        local_min = local_min.at[:, (0, -1)].set(False)
        local_max = local_max.at[:, (0, -1)].set(False)
    distance = jnp.abs(b[:, None, :] - level)
    close = distance <= float(topology_tolerance) * scale[:, None, None]
    marginal = jnp.any(close & local_min[:, None, :], axis=-1)
    merged = jnp.any(close & local_max[:, None, :], axis=-1)
    n_down = jnp.sum(down, axis=-1)
    n_up = jnp.sum(up, axis=-1)
    overflow = n_down > max_wells
    if periodic:
        truncated = jnp.zeros_like(overflow)
    else:
        boundary_cut = (
            (b[:, :1] < threshold[None, :])
            | (b[:, -1:] < threshold[None, :])
        )
        truncated = (n_down != n_up) | boundary_cut
    usable = well_mask & ~(marginal | merged | overflow)[..., None]
    action = jnp.where(well_mask, action, jnp.nan)

    out_shape = lead_shape + (int(pitch.shape[0]), max_wells)
    status_shape = lead_shape + (int(pitch.shape[0]),)
    return {
        **{name: jnp.where(well_mask, value, jnp.nan).reshape(out_shape)
           for name, value in extras.items()},
        "action": action.reshape(out_shape),
        "well_left": well_left.reshape(out_shape),
        "well_right": well_right.reshape(out_shape),
        "well_mask": well_mask.reshape(out_shape),
        "usable_mask": usable.reshape(out_shape),
        "absent": (~jnp.any(well_mask, axis=-1)).reshape(status_shape),
        "marginal": marginal.reshape(status_shape),
        "merged": merged.reshape(status_shape),
        "truncated": truncated.reshape(status_shape),
        "overflow": overflow.reshape(status_shape),
        "pitch": pitch,
    }


def trace_boozer_field_lines(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    G_b,
    I_b,
    nfp: int,
    alpha,
    points_per_period: int = 128,
    num_periods: int = 4,
    bmns_b=None,
) -> dict[str, Array]:
    """Synthesize ``B`` and ``dl/dzeta`` along Boozer field lines.

    Mode numbers follow the ``booz_xform`` convention: ``xn_b`` is physical.
    ``G_b`` and ``I_b`` are the Boozer covariant current functions, so
    ``dl/dzeta = abs(G + iota I) / B``. The endpoint is retained because the
    returned trace is bounded rather than assumed to close.
    """
    bmnc_b = jnp.asarray(bmnc_b)
    if bmnc_b.ndim != 2:
        raise ValueError("bmnc_b must have shape (nsurface, nmode)")
    if points_per_period < 8 or num_periods < 1:
        raise ValueError("points_per_period must be >= 8 and num_periods >= 1")
    dtype = bmnc_b.dtype
    xm = jnp.asarray(xm_b, dtype=dtype)
    xn = jnp.asarray(xn_b, dtype=dtype)
    iota = jnp.atleast_1d(jnp.asarray(iota_b, dtype=dtype))
    alpha = jnp.atleast_1d(jnp.asarray(alpha, dtype=dtype))
    sine = jnp.zeros_like(bmnc_b) if bmns_b is None else jnp.asarray(
        bmns_b, dtype=dtype)
    if sine.shape != bmnc_b.shape:
        raise ValueError("bmns_b must match bmnc_b")

    count = int(points_per_period) * int(num_periods) + 1
    length = 2.0 * np.pi * int(num_periods) / int(nfp)
    phi = jnp.linspace(0.0, length, count, dtype=dtype)
    theta = alpha[None, :, None] + iota[:, None, None] * phi[None, None, :]
    phase = theta[..., None] * xm - phi[None, None, :, None] * xn
    bmag = jnp.einsum("sapm,sm->sap", jnp.cos(phase), bmnc_b)
    bmag += jnp.einsum("sapm,sm->sap", jnp.sin(phase), sine)
    current_factor = jnp.abs(
        jnp.atleast_1d(jnp.asarray(G_b, dtype=dtype))
        + iota * jnp.atleast_1d(jnp.asarray(I_b, dtype=dtype))
    )
    return {
        "bmag": bmag,
        "dl_dphi": current_factor[:, None, None] / bmag,
        "phi": phi,
        "length": jnp.asarray(length, dtype=dtype),
        "alpha": alpha,
    }


def bounce_action_from_boozer(*, pitch, max_wells=None, quadrature_order=64, **kwargs):
    """Trace Boozer field lines and evaluate :func:`bounce_action`."""
    trace = trace_boozer_field_lines(**kwargs)
    result = bounce_action(
        trace["bmag"], pitch, dl_dphi=trace["dl_dphi"],
        length=trace["length"], periodic=False, max_wells=max_wells,
        quadrature_order=quadrature_order)
    result.update(trace)
    return result
