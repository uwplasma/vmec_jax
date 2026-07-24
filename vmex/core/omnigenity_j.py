"""Optional J-based omnigenity / maximum-J objectives sharing one Boozer pass.

These classes are intentionally separate from :mod:`vmex.core.omnigenity` so
the existing Goodman-style ``QIResidual`` behavior stays unchanged.  The goal
here is a higher-fidelity, still-differentiable surrogate of the local
``qi_functions_mod.py`` objectives:

- ``QuasiIsodynamicResidual``: compare the second adiabatic invariant of the
  original well against a smooth Goodman-constructed comparison well.
- ``maxJResidual``: penalize positive normalized radial trends of the
  comparison-well invariant.

The implementation uses the traceable Boozer spectrum from
``boozer_bmnc_state(...)`` once, then builds both residual blocks from shared
``J`` diagnostics.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import jax
import jax.numpy as jnp

from .omnigenity import boozer_bmnc_state
from .statephysics import _as_1d

__all__ = [
    "j_invariant_qi_maxj_residual",
    "j_invariant_qi_maxj_residual_from_boozer",
    "JInvariantQIAndMaxJResidual",
    "JInvariantQIResidual",
    "JInvariantMaxJResidual",
]

Array = Any


def _soft_min_idx(values, beta: float = 50.0):
    values = jnp.asarray(values, dtype=jnp.float64)
    weights = jax.nn.softmax(-jnp.asarray(beta, dtype=values.dtype) * values)
    return jnp.sum(jnp.arange(values.shape[0], dtype=values.dtype) * weights)


def _apply_smooth_goodman_transform(b_line, phi_coords):
    """Smooth squash/stretch surrogate of the Goodman constructed-QI well."""

    b_line = jnp.asarray(b_line, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    n = int(b_line.shape[0])
    indices = jnp.arange(n, dtype=b_line.dtype)
    s_indmin = _soft_min_idx(b_line)
    mask_l = jax.nn.sigmoid(2.0 * (s_indmin - indices))
    mask_r = 1.0 - mask_l
    bl_sq = jnp.minimum.accumulate(b_line)
    br_seed = jnp.where(indices >= s_indmin, b_line, b_line[0])
    br_sq = jnp.maximum.accumulate(br_seed)
    pmax = jnp.asarray(50.0, dtype=b_line.dtype)
    pmin = jnp.asarray(15.0, dtype=b_line.dtype)
    b_min_val = jnp.interp(s_indmin, indices, b_line)
    phi_mid = jnp.interp(s_indmin, indices, phi_coords)
    phi_start = phi_coords[0]
    phi_end = phi_coords[-1]
    x1_l = (phi_coords - phi_start) / (phi_mid - phi_start + 1.0e-10)
    x1_r = (phi_coords - phi_mid) / (phi_end - phi_mid + 1.0e-10)
    shape_l = (jnp.cos(2.0 * jnp.pi * x1_l) + 1.0) / 2.0
    shape_r = (jnp.cos(2.0 * jnp.pi * x1_r) + 1.0) / 2.0
    f_l = jnp.where(
        x1_l < 0.5,
        (1.0 - bl_sq) * (shape_l**pmax),
        (-b_min_val) * (shape_l**pmin),
    )
    f_r = jnp.where(
        x1_r < 0.5,
        (-b_min_val) * (shape_r**pmin),
        (1.0 - br_sq[-1]) * (shape_r**pmax),
    )
    return mask_l * (bl_sq + f_l) + mask_r * (br_sq + f_r)


def _compute_j_pair(phi_coords, b_input, b_target, bj_levels, gi_value, *, nphi_int: int = 128):
    """Return ``(J_input, J_constructed)`` on the requested bounce levels."""

    b_input = jnp.asarray(b_input, dtype=jnp.float64)
    b_target = jnp.asarray(b_target, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    bj_levels = jnp.asarray(bj_levels, dtype=jnp.float64)
    gi_value = jnp.asarray(gi_value, dtype=jnp.float64)

    indices = jnp.arange(b_target.shape[0], dtype=jnp.int32)
    indmin = jnp.argmin(b_target)
    high = jnp.asarray(1.1 * jnp.max(b_target), dtype=b_target.dtype)
    b_l = jnp.where(indices <= indmin, b_target, high)
    b_r = jnp.where(indices >= indmin, b_target, high)
    left_phi = jnp.flip(phi_coords)
    left_b = jnp.flip(b_l)
    right_phi = phi_coords
    right_b = b_r

    # Smooth inverse-branch evaluation: estimate the crossing location of
    # ``B(phi) = Bj`` with a soft argmin over the branch residual instead of a
    # hard inverse interpolation on a geometry-dependent xp-grid.
    branch_eps = jnp.maximum(5.0e-3 * jnp.max(bj_levels), 1.0e-6)
    left_logits = -((left_b[None, :] - bj_levels[:, None]) / branch_eps) ** 2
    right_logits = -((right_b[None, :] - bj_levels[:, None]) / branch_eps) ** 2
    left_w = jax.nn.softmax(left_logits, axis=1)
    right_w = jax.nn.softmax(right_logits, axis=1)
    p1 = jnp.sum(left_w * left_phi[None, :], axis=1)
    p2 = jnp.sum(right_w * right_phi[None, :], axis=1)

    t = jnp.linspace(0.0, 1.0, int(nphi_int), dtype=b_target.dtype)
    phi_grid = p1[:, None] + t[None, :] * (p2 - p1)[:, None]
    bi_g = jnp.interp(phi_grid, phi_coords, b_input)
    bc_g = jnp.interp(phi_grid, phi_coords, b_target)
    metric_factor = gi_value / (bi_g + 1.0e-9)
    bj_v = bj_levels[:, None]

    res_i = 1.0 - bi_g / (bj_v + 1.0e-9)
    res_c = 1.0 - bc_g / (bj_v + 1.0e-9)
    vi_g = jnp.sign(res_i) * jnp.sqrt(jnp.abs(res_i) + 1.0e-9)
    vc_g = jnp.sign(res_c) * jnp.sqrt(jnp.abs(res_c) + 1.0e-9)

    ji = jnp.trapezoid(vi_g * metric_factor, x=phi_grid, axis=1)
    jc = jnp.trapezoid(vc_g * metric_factor, x=phi_grid, axis=1)
    return ji, jc


def _synthesize_boozer_field_lines(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    nphi: int,
    nalpha: int,
):
    """Synthesize ``|B|`` along Boozer field lines over one field period."""

    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    xm_b = jnp.asarray(np.asarray(xm_b, dtype=float))
    xn_b = jnp.asarray(np.asarray(xn_b, dtype=float))
    iota_b = jnp.atleast_1d(jnp.asarray(iota_b, dtype=jnp.float64))
    dtype = bmnc_b.dtype

    period = 2.0 * np.pi / float(nfp)
    phi = jnp.linspace(0.0, period, int(nphi), endpoint=True, dtype=dtype)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, int(nalpha), endpoint=False, dtype=dtype)
    theta = alpha[None, :, None] + iota_b[:, None, None] * phi[None, None, :]
    angle = theta[..., None] * xm_b - phi[None, None, :, None] * xn_b
    b = jnp.einsum("sapm,sm->sap", jnp.cos(angle), bmnc_b)
    return phi, alpha, b


def j_invariant_qi_maxj_residual_from_boozer(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    gi_b,
    s_b,
    nfp: int,
    weights: Iterable[float] | None = None,
    nphi: int = 101,
    nalpha: int = 51,
    n_bounce: int = 66,
    p_j: float = 1.0,
    p_lambda: float = 1.0,
    nphi_int: int = 128,
    target_maxj: float = -0.06,
    qi_weight: float = 1.0,
    maxj_weight: float = 1.0,
    include_qi: bool = True,
    include_maxj: bool = True,
) -> dict[str, Array]:
    """Shared-J QI/max-J residual blocks from precomputed Boozer spectra."""

    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    iota_b = jnp.asarray(iota_b, dtype=jnp.float64)
    gi_b = jnp.asarray(gi_b, dtype=jnp.float64)
    s_b = jnp.asarray(s_b, dtype=jnp.float64)
    nsurf = int(bmnc_b.shape[0])
    if nsurf == 0:
        raise ValueError("Boozer surfaces must be non-empty")
    if int(nphi) < 8 or int(nalpha) < 2 or int(n_bounce) < 2:
        raise ValueError("shared-J QI/max-J residual needs nphi >= 8, nalpha >= 2, n_bounce >= 2")

    w_arr = jnp.ones((nsurf,), dtype=jnp.float64) if weights is None else _as_1d(weights)
    if int(w_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as the Boozer surfaces")

    phi, alpha, b_lines = _synthesize_boozer_field_lines(
        bmnc_b=bmnc_b,
        xm_b=xm_b,
        xn_b=xn_b,
        iota_b=iota_b,
        nfp=int(nfp),
        nphi=int(nphi),
        nalpha=int(nalpha),
    )
    bj_norm = jnp.power(
        jnp.arange(int(n_bounce), dtype=jnp.float64) / jnp.maximum(int(n_bounce) - 1, 1),
        float(p_lambda),
    )

    def _per_surface(b_surface, gi_surface):
        def _per_line(b_line):
            bmin = jnp.min(b_line)
            bmax = jnp.max(b_line)
            scale = jnp.maximum(bmax - bmin, 1.0e-10)
            b_norm = (b_line - bmin) / scale
            b_target_norm = _apply_smooth_goodman_transform(b_norm, phi)
            b_target = b_target_norm * scale + bmin
            bj_phys = bj_norm * scale + bmin
            return _compute_j_pair(phi, b_line, b_target, bj_phys, gi_surface, nphi_int=int(nphi_int))

        ji_all, jc_all = jax.vmap(_per_line)(b_surface)
        return ji_all, jc_all

    ji_all, jc_all = jax.vmap(_per_surface, in_axes=(0, 0))(b_lines, gi_b)
    ji_pow = jnp.abs(ji_all) ** float(p_j)
    jc_pow = jnp.abs(jc_all) ** float(p_j)

    residual_blocks: list[jnp.ndarray] = []
    diagnostics: dict[str, Array] = {
        "phi": phi,
        "alpha": alpha,
        "surfaces": s_b,
        "ji": ji_all,
        "jc": jc_all,
        "ji_pow": ji_pow,
        "jc_pow": jc_pow,
    }

    if bool(include_qi):
        mean_jc = jnp.mean(jc_pow, axis=1, keepdims=True)
        mean_ji = jnp.mean(ji_pow, axis=1, keepdims=True)
        denom = jnp.mean(ji_pow, axis=(1, 2), keepdims=True) + jnp.mean(jc_pow, axis=(1, 2), keepdims=True)
        sqrt_w = jnp.sqrt(w_arr)[:, None, None]
        qi_res_i = (ji_pow - mean_jc) / (denom + 1.0e-10)
        qi_res_c = (jc_pow - mean_ji) / (denom + 1.0e-10)
        qi_block = float(qi_weight) * sqrt_w * jnp.concatenate([qi_res_i, qi_res_c], axis=1)
        qi_block = jnp.ravel(qi_block) / jnp.sqrt(jnp.asarray(2.0 * nalpha * n_bounce, dtype=jnp.float64))
        residual_blocks.append(qi_block)
        diagnostics["qi_surface"] = jnp.sqrt(jnp.mean((qi_res_i**2 + qi_res_c**2), axis=(1, 2)))
        diagnostics["qi_objective"] = jnp.sum(qi_block * qi_block)
    else:
        diagnostics["qi_surface"] = jnp.zeros((nsurf,), dtype=jnp.float64)
        diagnostics["qi_objective"] = jnp.asarray(0.0, dtype=jnp.float64)

    if bool(include_maxj):
        if nsurf < 2:
            maxj_block = jnp.zeros((0,), dtype=jnp.float64)
            maxj_surface = jnp.zeros((0,), dtype=jnp.float64)
        else:
            ds = s_b[1:] - s_b[:-1]
            ds = jnp.where(jnp.abs(ds) > 0.0, ds, 1.0e-10)
            jc_lo = jc_pow[:-1, :, 1:]
            jc_hi = jc_pow[1:, :, 1:]
            jc_lo_mean = jnp.mean(jc_lo, axis=1, keepdims=True)
            slope = (jc_hi - jc_lo_mean) / (ds[:, None, None] * (0.5 * (jc_hi + jc_lo_mean) + 1.0e-10))
            violation = jnp.maximum(0.0, slope - float(target_maxj))
            pair_w = jnp.sqrt(0.5 * (w_arr[:-1] + w_arr[1:]))[:, None, None]
            maxj_surface = jnp.sqrt(jnp.mean(violation**2, axis=(1, 2)))
            maxj_block = float(maxj_weight) * pair_w * violation
            maxj_block = jnp.ravel(maxj_block) / jnp.sqrt(jnp.asarray((n_bounce - 1) * nalpha, dtype=jnp.float64))
        residual_blocks.append(maxj_block)
        diagnostics["maxj_surface"] = maxj_surface
        diagnostics["maxj_objective"] = jnp.sum(maxj_block * maxj_block)
    else:
        diagnostics["maxj_surface"] = jnp.zeros((max(nsurf - 1, 0),), dtype=jnp.float64)
        diagnostics["maxj_objective"] = jnp.asarray(0.0, dtype=jnp.float64)

    if not residual_blocks:
        raise ValueError("At least one of include_qi/include_maxj must be True.")

    residuals1d = jnp.concatenate(residual_blocks)
    diagnostics["residual_block_sizes"] = jnp.asarray([block.size for block in residual_blocks], dtype=jnp.int32)
    return {
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        **diagnostics,
    }


def j_invariant_qi_maxj_residual(
    state,
    rt,
    *,
    surfaces,
    weights: Iterable[float] | None = None,
    mboz: int = 16,
    nboz: int = 16,
    oversample: int = 2,
    nphi: int = 101,
    nalpha: int = 51,
    n_bounce: int = 66,
    p_j: float = 1.0,
    p_lambda: float = 1.0,
    nphi_int: int = 128,
    target_maxj: float = -0.06,
    qi_weight: float = 1.0,
    maxj_weight: float = 1.0,
    include_qi: bool = True,
    include_maxj: bool = True,
) -> dict[str, Array]:
    """Shared-J QI/max-J residual blocks from one traceable Boozer evaluation."""

    booz = boozer_bmnc_state(
        state,
        rt,
        surfaces=surfaces,
        mboz=int(mboz),
        nboz=int(nboz),
        oversample=int(oversample),
    )
    return j_invariant_qi_maxj_residual_from_boozer(
        bmnc_b=booz["bmnc_b"],
        xm_b=booz["xm_b"],
        xn_b=booz["xn_b"],
        iota_b=booz["iota_b"],
        gi_b=booz["gi_b"],
        s_b=booz["s_b"],
        nfp=int(booz["nfp"]),
        weights=weights,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        p_j=p_j,
        p_lambda=p_lambda,
        nphi_int=nphi_int,
        target_maxj=target_maxj,
        qi_weight=qi_weight,
        maxj_weight=maxj_weight,
        include_qi=include_qi,
        include_maxj=include_maxj,
    )


class JInvariantQIAndMaxJResidual:
    """Combined J-based QI/max-J residual term sharing one ``J`` evaluation."""

    name = "j_invariant_qi_maxj"

    def __init__(
        self,
        surfaces,
        *,
        weights: Iterable[float] | None = None,
        mboz: int = 16,
        nboz: int = 16,
        oversample: int = 2,
        nphi: int = 101,
        nalpha: int = 51,
        n_bounce: int = 66,
        p_j: float = 1.0,
        p_lambda: float = 1.0,
        nphi_int: int = 128,
        target_maxj: float = -0.06,
        qi_weight: float = 1.0,
        maxj_weight: float = 1.0,
        include_qi: bool = True,
        include_maxj: bool = True,
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
        self.n_bounce = int(n_bounce)
        self.p_j = float(p_j)
        self.p_lambda = float(p_lambda)
        self.nphi_int = int(nphi_int)
        self.target_maxj = float(target_maxj)
        self.qi_weight = float(qi_weight)
        self.maxj_weight = float(maxj_weight)
        self.include_qi = bool(include_qi)
        self.include_maxj = bool(include_maxj)

    def compute_state(self, state, rt) -> dict[str, Array]:
        return j_invariant_qi_maxj_residual(
            state,
            rt,
            surfaces=self.surfaces,
            weights=self.weights,
            mboz=self.mboz,
            nboz=self.nboz,
            oversample=self.oversample,
            nphi=self.nphi,
            nalpha=self.nalpha,
            n_bounce=self.n_bounce,
            p_j=self.p_j,
            p_lambda=self.p_lambda,
            nphi_int=self.nphi_int,
            target_maxj=self.target_maxj,
            qi_weight=self.qi_weight,
            maxj_weight=self.maxj_weight,
            include_qi=self.include_qi,
            include_maxj=self.include_maxj,
        )

    def residuals_state(self, state, rt) -> jnp.ndarray:
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state, rt) -> Array:
        return self.compute_state(state, rt)["total"]

    def J(self, eq) -> jnp.ndarray:
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = J

    def residuals(self, eq) -> jnp.ndarray:
        return self.J(eq)

    def total(self, eq) -> Array:
        return self.total_state(eq.state, eq.runtime)


class JInvariantQIResidual(JInvariantQIAndMaxJResidual):
    """J-based omnigenity residual only."""

    name = "j_invariant_qi"

    def __init__(self, surfaces, **kwargs):
        kwargs.setdefault("include_qi", True)
        kwargs.setdefault("include_maxj", False)
        super().__init__(surfaces, **kwargs)


class JInvariantMaxJResidual(JInvariantQIAndMaxJResidual):
    """J-based maximum-J residual only."""

    name = "j_invariant_maxj"

    def __init__(self, surfaces, **kwargs):
        kwargs.setdefault("include_qi", False)
        kwargs.setdefault("include_maxj", True)
        super().__init__(surfaces, **kwargs)
