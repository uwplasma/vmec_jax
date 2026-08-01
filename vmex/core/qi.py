"""Traceable quasi-isodynamic residual terms."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import jax.numpy as jnp

from .bounce import bounce_action_from_boozer
from .omnigenity import boozer_bmnc_state
from .optimize import quasi_isodynamic_residual
from .statephysics import _as_1d

Array = Any

__all__ = [
    "ConstructedQIResidual",
    "JInvariantQIResidual",
    "j_invariant_qi_residual_from_boozer",
]


def j_invariant_qi_residual_from_boozer(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    G_b,
    I_b,
    nfp: int,
    pitch,
    bmns_b=None,
    weights: Iterable[float] | None = None,
    nalpha: int = 17,
    points_per_period: int = 128,
    num_periods: int = 4,
    max_wells: int | None = None,
    quadrature_order: int = 64,
) -> dict[str, Array]:
    """Measure variation of bounce action at shared physical pitch.

    Complete wells from all field lines and periods are compared on each
    surface. Edge-cut wells are excluded, while marginal, merged, overflow,
    or absent pitch blocks return NaN instead of a plausible zero.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    if bmnc_b.ndim != 2:
        raise ValueError("bmnc_b must have shape (nsurface, nmode)")
    if nalpha < 2:
        raise ValueError("nalpha must be >= 2")
    nsurface = int(bmnc_b.shape[0])
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
    action, mask = out["action"], out["usable_mask"]
    line_count = jnp.sum(mask, axis=3)
    count = jnp.sum(mask, axis=(1, 3))
    total_action = jnp.sum(jnp.where(mask, action, 0.0), axis=(1, 3))
    mean = total_action / jnp.maximum(count, 1)
    scale = jnp.maximum(jnp.abs(mean), jnp.finfo(bmnc_b.dtype).eps)
    hard_invalid = out["marginal"] | out["merged"] | out["overflow"]
    valid_pitch = jnp.all(line_count > 0, axis=1) & ~jnp.any(
        hard_invalid, axis=1)
    safe_action = jnp.where(mask, action, mean[:, None, :, None])
    residual = jnp.where(
        mask,
        (safe_action - mean[:, None, :, None]) / scale[:, None, :, None]
        * jnp.sqrt(weight[:, None, None, None] / count[:, None, :, None]),
        0.0,
    )
    residual = jnp.where(valid_pitch[:, None, :, None], residual, jnp.nan)
    residuals1d = jnp.ravel(residual)
    out.update({
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "action_mean": mean,
        "valid_pitch": valid_pitch,
        "usable_count": count,
        "usable_count_per_line": line_count,
    })
    return out


class _QIResidualTerm:
    """Shared least-squares objective interface."""

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


class ConstructedQIResidual(_QIResidualTerm):
    """Goodman squash-and-shuffle residual with a traceable Boozer transform."""

    name = "constructed_qi"

    def __init__(
        self, surfaces, *, weights=None, mboz=16, nboz=16, oversample=2,
        **residual_options,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.shape != self.surfaces.shape:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz, self.nboz = int(mboz), int(nboz)
        self.oversample = int(oversample)
        self.residual_options = residual_options

    def compute_state(self, state, rt):
        """Return the constructed-QI residual and Boozer diagnostics."""
        booz = boozer_bmnc_state(
            state, rt, surfaces=self.surfaces, mboz=self.mboz,
            nboz=self.nboz, oversample=self.oversample)
        out = quasi_isodynamic_residual(
            bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], nfp=booz["nfp"], weights=self.weights,
            **self.residual_options)
        out.update(booz)
        return out


class JInvariantQIResidual(_QIResidualTerm):
    """Bounce-action invariance residual at caller-supplied physical pitch."""

    name = "j_invariant_qi"

    def __init__(
        self, surfaces, pitch, *, weights=None, mboz=16, nboz=16,
        oversample=2, nalpha=17, points_per_period=128, num_periods=4,
        max_wells=None, quadrature_order=64,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.pitch = np.atleast_1d(np.asarray(pitch, dtype=float))
        if (self.surfaces.size == 0 or self.pitch.size == 0
                or not np.all(np.isfinite(self.pitch)) or np.any(self.pitch <= 0)):
            raise ValueError("surfaces and positive finite physical pitch are required")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.shape != self.surfaces.shape:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz, self.nboz = int(mboz), int(nboz)
        self.oversample, self.nalpha = int(oversample), int(nalpha)
        self.points_per_period, self.num_periods = (
            int(points_per_period), int(num_periods))
        self.max_wells = None if max_wells is None else int(max_wells)
        self.quadrature_order = int(quadrature_order)

    def compute_state(self, state, rt):
        """Return action-invariance residuals and topology diagnostics."""
        booz = boozer_bmnc_state(
            state, rt, surfaces=self.surfaces, mboz=self.mboz,
            nboz=self.nboz, oversample=self.oversample)
        out = j_invariant_qi_residual_from_boozer(
            bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
            nfp=booz["nfp"], pitch=self.pitch, weights=self.weights,
            nalpha=self.nalpha, points_per_period=self.points_per_period,
            num_periods=self.num_periods, max_wells=self.max_wells,
            quadrature_order=self.quadrature_order)
        out.update(booz)
        return out
