"""Traceable maximum-J residuals."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import jax.numpy as jnp

from .bounce import bounce_action_from_boozer
from .omnigenity import boozer_bmnc_state
from .statephysics import _as_1d

Array = Any

__all__ = ["MaximumJResidual", "maximum_j_residual_from_boozer"]


def maximum_j_residual_from_boozer(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    G_b,
    I_b,
    nfp: int,
    psi_b,
    psi_edge,
    pitch,
    bmns_b=None,
    weights: Iterable[float] | None = None,
    target: float = 0.0,
    nalpha: int = 17,
    points_per_period: int = 128,
    num_periods: int = 4,
    max_wells: int | None = None,
    quadrature_order: int = 64,
    match_tolerance: float | None = None,
) -> dict[str, Array]:
    """Penalize positive ``dJ/dpsi`` at common physical pitch.

    Wells on adjacent surfaces are paired only when they are reciprocal
    nearest neighbours in Boozer toroidal angle. Every line must retain the
    same nonzero number of complete, usable wells on both surfaces. Ambiguous
    topology, a nonmonotone signed flux coordinate, or an excessive matching
    displacement invalidates the pitch block with NaN.

    ``target`` is the upper bound on the dimensionless logarithmic slope
    ``abs(psi_edge) / J * dJ/dpsi``. The default enforces the maximum-J
    condition ``dJ/dpsi <= 0`` without changing its physical sign.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    psi = jnp.atleast_1d(jnp.asarray(psi_b, dtype=bmnc_b.dtype))
    if bmnc_b.ndim != 2 or int(bmnc_b.shape[0]) < 2:
        raise ValueError("bmnc_b needs at least two surfaces")
    nsurface = int(bmnc_b.shape[0])
    if psi.shape != (nsurface,):
        raise ValueError("psi_b must have the same length as the Boozer surfaces")
    if nalpha < 2:
        raise ValueError("nalpha must be >= 2")
    weight = (jnp.ones(nsurface, dtype=bmnc_b.dtype) if weights is None
              else _as_1d(weights))
    if int(weight.shape[0]) != nsurface:
        raise ValueError("weights must have the same length as surfaces")

    alpha = jnp.arange(nalpha, dtype=bmnc_b.dtype) * (2.0 * jnp.pi / nalpha)
    out = bounce_action_from_boozer(
        bmnc_b=bmnc_b, xm_b=xm_b, xn_b=xn_b, iota_b=iota_b,
        G_b=G_b, I_b=I_b, nfp=nfp, alpha=alpha,
        points_per_period=points_per_period, num_periods=num_periods,
        bmns_b=bmns_b, pitch=pitch, max_wells=max_wells,
        quadrature_order=quadrature_order)
    action, usable = out["action"], out["usable_mask"]
    center = 0.5 * (out["well_left"] + out["well_right"])
    lo_center, hi_center = center[:-1], center[1:]
    lo_usable, hi_usable = usable[:-1], usable[1:]
    distance = jnp.abs(lo_center[..., :, None] - hi_center[..., None, :])
    candidates = lo_usable[..., :, None] & hi_usable[..., None, :]
    infinity = jnp.asarray(jnp.inf, dtype=bmnc_b.dtype)
    candidate_distance = jnp.where(candidates, distance, infinity)
    hi_index = jnp.argmin(candidate_distance, axis=-1)
    lo_index = jnp.argmin(candidate_distance, axis=-2)
    matched_hi_usable = jnp.take_along_axis(hi_usable, hi_index, axis=-1)
    reciprocal = jnp.take_along_axis(lo_index, hi_index, axis=-1)
    reciprocal = reciprocal == jnp.arange(action.shape[-1])
    match_distance = jnp.take_along_axis(
        candidate_distance, hi_index[..., None], axis=-1)[..., 0]
    tolerance = (np.pi / float(nfp) if match_tolerance is None
                 else float(match_tolerance))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("match_tolerance must be positive and finite")
    matched = (
        lo_usable & matched_hi_usable & reciprocal
        & (match_distance <= tolerance)
    )

    hard_invalid = out["marginal"] | out["merged"] | out["overflow"]
    lo_count = jnp.sum(lo_usable, axis=-1)
    hi_count = jnp.sum(hi_usable, axis=-1)
    valid_line = (
        (lo_count > 0) & (lo_count == hi_count)
        & (jnp.sum(matched, axis=-1) == lo_count)
        & ~(hard_invalid[:-1] | hard_invalid[1:])
    )
    valid_pitch = jnp.all(valid_line, axis=1)

    dpsi = jnp.diff(psi)
    psi_edge = jnp.asarray(psi_edge, dtype=bmnc_b.dtype)
    eps = jnp.finfo(bmnc_b.dtype).eps
    outward = jnp.all(dpsi * psi_edge > 0.0)
    valid_flux = (
        outward & jnp.isfinite(psi_edge) & (jnp.abs(psi_edge) > eps)
        & jnp.isfinite(dpsi) & (jnp.abs(dpsi) > eps)
    )
    valid_pitch &= valid_flux[:, None]

    hi_action = jnp.take_along_axis(action[1:], hi_index, axis=-1)
    lo_action = jnp.where(matched, action[:-1], 1.0)
    hi_action = jnp.where(matched, hi_action, 1.0)
    dJ_dpsi = (hi_action - lo_action) / dpsi[:, None, None, None]
    mean_action = 0.5 * (jnp.abs(lo_action) + jnp.abs(hi_action))
    relative_slope = dJ_dpsi * jnp.abs(psi_edge) / jnp.maximum(
        mean_action, eps)
    violation = jnp.maximum(relative_slope - float(target), 0.0)
    count = jnp.sum(matched, axis=(1, 3))
    pair_weight = 0.5 * (weight[:-1] + weight[1:])
    scale = jnp.sqrt(
        pair_weight[:, None] / jnp.maximum(count, 1))[:, None, :, None]
    residual = jnp.where(matched, violation * scale, 0.0)
    residual = jnp.where(valid_pitch[:, None, :, None], residual, jnp.nan)
    residuals1d = jnp.ravel(residual)
    out.update({
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "dJ_dpsi": jnp.where(matched, dJ_dpsi, jnp.nan),
        "relative_slope": jnp.where(matched, relative_slope, jnp.nan),
        "matched_well_mask": matched,
        "matched_well_index": hi_index,
        "match_distance": jnp.where(matched, match_distance, jnp.nan),
        "valid_pitch_pair": valid_pitch,
        "psi_b": psi,
        "psi_edge": psi_edge,
    })
    return out


class MaximumJResidual:
    """Composable maximum-J objective at fixed physical pitch."""

    name = "maximum_j"

    def __init__(
        self, surfaces, pitch, *, weights=None, mboz=16, nboz=16,
        oversample=2, **residual_options,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.pitch = np.atleast_1d(np.asarray(pitch, dtype=float))
        if (self.surfaces.size < 2 or np.any(np.diff(self.surfaces) <= 0.0)
                or self.pitch.size == 0 or not np.all(np.isfinite(self.pitch))
                or np.any(self.pitch <= 0.0)):
            raise ValueError(
                "increasing surfaces and positive finite physical pitch are required")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.shape != self.surfaces.shape:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz, self.nboz = int(mboz), int(nboz)
        self.oversample = int(oversample)
        self.residual_options = residual_options

    def compute_state(self, state, rt):
        """Return maximum-J residuals, actions, matches, and flux slopes."""
        booz = boozer_bmnc_state(
            state, rt, surfaces=self.surfaces, mboz=self.mboz,
            nboz=self.nboz, oversample=self.oversample)
        out = maximum_j_residual_from_boozer(
            bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
            nfp=booz["nfp"], psi_b=booz["psi_b"],
            psi_edge=booz["psi_edge"], pitch=self.pitch,
            weights=self.weights, **self.residual_options)
        out.update(booz)
        return out

    def residuals_state(self, state, rt):
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state, rt):
        return self.compute_state(state, rt)["total"]

    def J(self, eq):
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = J
    residuals = J

    def total(self, eq):
        return self.total_state(eq.state, eq.runtime)
