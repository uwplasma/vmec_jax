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


def _cummin(values):
    values = jnp.asarray(values, dtype=jnp.float64)
    return jax.lax.associative_scan(jnp.minimum, values)


def _cummax(values):
    values = jnp.asarray(values, dtype=jnp.float64)
    return jax.lax.associative_scan(jnp.maximum, values)


def _smooth_signed_sqrt(values, eps: float = 1.0e-9):
    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    abs_smooth = jnp.sqrt(values * values + eps_arr * eps_arr)
    return values / jnp.sqrt(abs_smooth + eps_arr)


def _smooth_positive_sqrt(values, eps: float = 1.0e-10):
    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    positive = jnp.maximum(values, 0.0)
    # Finite derivative at zero, while staying close to sqrt(max(values, 0)).
    return jnp.sqrt(positive + eps_arr) - jnp.sqrt(eps_arr)


def _apply_smooth_goodman_transform(b_line, phi_coords):
    """Smooth squash/stretch surrogate of the Goodman constructed-QI well."""

    b_line = jnp.asarray(b_line, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    n = int(b_line.shape[0])
    indices = jnp.arange(n, dtype=b_line.dtype)
    s_indmin = _soft_min_idx(b_line)
    split_beta = jnp.asarray(10.0, dtype=b_line.dtype)
    peak_beta = jnp.asarray(120.0, dtype=b_line.dtype)
    cap_beta = jnp.asarray(35.0, dtype=b_line.dtype)
    mask_l = jax.nn.sigmoid(split_beta * (s_indmin - indices))
    mask_r = 1.0 - mask_l

    # Left branch: mirror the reference logic by identifying the LHS peak,
    # flattening before it, then squashing toward the minimum.
    lhs_gate = jax.nn.sigmoid(split_beta * (s_indmin - indices))
    lhs_penalty = jnp.asarray(10.0, dtype=b_line.dtype) * (1.0 - lhs_gate)
    lhs_weights = jax.nn.softmax(peak_beta * (b_line - lhs_penalty))
    lhs_peak_val = jnp.sum(lhs_weights * b_line)
    lhs_peak_idx = jnp.sum(lhs_weights * indices)
    before_peak = jax.nn.sigmoid(cap_beta * (lhs_peak_idx - indices))
    bl_base = before_peak * lhs_peak_val + (1.0 - before_peak) * b_line
    bl_sq = _cummin(bl_base)

    # Right branch: build a smoother analogue of the reference logic:
    # isolate the RHS of the well, locate its peak, flatten only after that
    # peak, then squash from right-to-left with a reverse cumulative minimum.
    rhs_gate = jax.nn.sigmoid(split_beta * (indices - s_indmin))
    rhs_penalty = jnp.asarray(10.0, dtype=b_line.dtype) * (1.0 - rhs_gate)
    rhs_weights = jax.nn.softmax(peak_beta * (b_line - rhs_penalty))
    rhs_peak_val = jnp.sum(rhs_weights * b_line)
    rhs_peak_idx = jnp.sum(rhs_weights * indices)
    after_peak = jax.nn.sigmoid(cap_beta * (indices - rhs_peak_idx))
    br_base = after_peak * rhs_peak_val + (1.0 - after_peak) * b_line
    br_sq = jnp.flip(_cummin(jnp.flip(br_base)))

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
        (1.0 - rhs_peak_val) * (shape_r**pmax),
    )
    out = mask_l * (bl_sq + f_l) + mask_r * (br_sq + f_r)
    out = jnp.clip(out, 0.0, 1.0)
    x_edge = (phi_coords - phi_start) / (phi_end - phi_start + 1.0e-10)
    edge_beta = jnp.asarray(120.0, dtype=b_line.dtype)
    edge_width = jnp.asarray(0.015, dtype=b_line.dtype)
    left_edge = jax.nn.sigmoid(edge_beta * (edge_width - x_edge))
    right_edge = jax.nn.sigmoid(edge_beta * (x_edge - (1.0 - edge_width)))
    edge_blend = left_edge + right_edge - left_edge * right_edge
    out = (1.0 - edge_blend) * out + edge_blend
    return out


def _branch_crossings(phi_coords, b_line, bj_level):
    """Smooth inverse-branch evaluation for ``B(phi)=Bj`` crossings."""

    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    b_line = jnp.asarray(b_line, dtype=jnp.float64)
    bj_level = jnp.asarray(bj_level, dtype=jnp.float64)
    indices = jnp.arange(b_line.shape[0], dtype=jnp.float64)
    s_indmin = _soft_min_idx(b_line)
    branch_beta = jnp.asarray(10.0, dtype=b_line.dtype)
    left_mask = jax.nn.sigmoid(branch_beta * (s_indmin - indices))
    right_mask = 1.0 - left_mask

    def _invert_branch(phi_branch, b_branch, branch_mask):
        phi0 = phi_branch[:-1]
        phi1 = phi_branch[1:]
        b0 = b_branch[:-1]
        b1 = b_branch[1:]
        m0 = branch_mask[:-1]
        m1 = branch_mask[1:]
        db = b1 - b0
        scale = jnp.maximum(5.0e-3 * jnp.max(jnp.abs(b_branch)), 1.0e-6)

        # On both oriented branches, B should increase as we move away from the minimum.
        mono_gate = jax.nn.sigmoid(60.0 * db / scale)
        lo = jnp.minimum(b0, b1)
        hi = jnp.maximum(b0, b1)
        between_gate = jax.nn.sigmoid(40.0 * (bj_level - lo) / scale) * jax.nn.sigmoid(
            40.0 * (hi - bj_level) / scale
        )
        slope_gate = jnp.tanh(jnp.abs(db) / scale) ** 2
        branch_gate = m0 * m1
        valid = branch_gate * mono_gate * between_gate * slope_gate + 1.0e-14

        t_raw = (bj_level - b0) / (db + 1.0e-12)
        t = jnp.clip(t_raw, 0.0, 1.0)
        phi_seg = phi0 + t * (phi1 - phi0)
        return jnp.sum(valid * phi_seg) / jnp.sum(valid)

    phi_min = jnp.interp(s_indmin, indices, phi_coords)
    phi_lo = _invert_branch(jnp.flip(phi_coords), jnp.flip(b_line), jnp.flip(left_mask))
    phi_hi = _invert_branch(phi_coords, b_line, right_mask)
    # Match the normalized reference ``GetBranches(..., Bmax=1, Bmin=0)``.
    endpoint_beta = jnp.asarray(250.0, dtype=b_line.dtype)
    endpoint_width = jnp.asarray(0.01, dtype=b_line.dtype)
    low_blend = jax.nn.sigmoid(endpoint_beta * (endpoint_width - bj_level))
    high_blend = jax.nn.sigmoid(endpoint_beta * (bj_level - (1.0 - endpoint_width)))
    phi_lo = (1.0 - low_blend) * phi_lo + low_blend * phi_min
    phi_hi = (1.0 - low_blend) * phi_hi + low_blend * phi_min
    phi_lo = (1.0 - high_blend) * phi_lo + high_blend * phi_coords[0]
    phi_hi = (1.0 - high_blend) * phi_hi + high_blend * phi_coords[-1]
    return phi_lo, phi_hi


def _compute_j_pair(phi_coords, b_input, b_target, bj_levels, gi_value, *, nphi_int: int = 128):
    """Return ``(J_input, J_constructed)`` on the requested bounce levels."""

    b_input = jnp.asarray(b_input, dtype=jnp.float64)
    b_target = jnp.asarray(b_target, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    bj_levels = jnp.asarray(bj_levels, dtype=jnp.float64)
    gi_value = jnp.asarray(gi_value, dtype=jnp.float64)

    bmin = jnp.min(b_target)
    bmax = jnp.max(b_target)
    scale = jnp.maximum(bmax - bmin, 1.0e-12)
    b_target_norm = (b_target - bmin) / scale
    bj_norm = (bj_levels - bmin) / scale

    p1, p2 = jax.vmap(lambda bj: _branch_crossings(phi_coords, b_target_norm, bj))(bj_norm)

    t = jnp.linspace(0.0, 1.0, int(nphi_int), dtype=b_target.dtype)
    phi_grid = p1[:, None] + t[None, :] * (p2 - p1)[:, None]
    bi_g = jnp.interp(phi_grid, phi_coords, b_input)
    bc_g = jnp.interp(phi_grid, phi_coords, b_target)
    bi_safe = jnp.maximum(bi_g, 1.0e-9)
    bj_v = jnp.maximum(bj_levels[:, None], 1.0e-9)
    metric_factor = gi_value / bi_safe

    res_i = 1.0 - bi_g / bj_v
    res_c = 1.0 - bc_g / bj_v
    vi_g = _smooth_signed_sqrt(res_i)
    vc_g = _smooth_positive_sqrt(res_c)

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
    ji_pow = jnp.power(ji_all, float(p_j))
    jc_pow = jnp.power(jc_all, float(p_j))

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
        nalpha_f = jnp.asarray(float(nalpha), dtype=jnp.float64)
        sum_ji = jnp.sum(ji_pow, axis=1)
        sum_jc = jnp.sum(jc_pow, axis=1)
        sum_ji_sq = jnp.sum(ji_pow * ji_pow, axis=1)
        sum_jc_sq = jnp.sum(jc_pow * jc_pow, axis=1)
        qi_pair_sum = nalpha_f * (sum_ji_sq + sum_jc_sq) - 2.0 * sum_ji * sum_jc
        qi_num = jnp.sum(qi_pair_sum, axis=1)
        mean_denom = ((jnp.sum(sum_ji + sum_jc, axis=1) / (2.0 * float(n_bounce))) ** 2) + 1.0e-10
        qi_surface = jnp.sqrt(jnp.maximum(qi_num, 0.0) / mean_denom)
        qi_block = float(qi_weight) * jnp.sqrt(w_arr) * qi_surface
        residual_blocks.append(qi_block)
        diagnostics["qi_surface"] = qi_surface
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
            jc_lo = jc_pow[:-1, 1:, :]
            jc_hi = jc_pow[1:, 1:, :]
            ds3 = ds[:, None, None]

            def _surface_pair_slope(hi_surface, lo_surface, ds_surface):
                def _alpha_slope(hi_alpha):
                    slope_terms = (hi_alpha[:, None] - lo_surface) / (
                        ds_surface * (0.5 * (hi_alpha[:, None] + lo_surface) + 1.0e-10)
                    )
                    return jnp.mean(slope_terms, axis=1)

                return jax.vmap(_alpha_slope, in_axes=1, out_axes=1)(hi_surface)

            slope = jax.vmap(_surface_pair_slope, in_axes=(0, 0, 0))(jc_hi, jc_lo, ds)
            violation = jnp.maximum(0.0, slope - float(target_maxj))
            pair_w = jnp.sqrt(0.5 * (w_arr[:-1] + w_arr[1:]))[:, None, None]
            maxj_surface = jnp.sqrt(jnp.sum(violation**2, axis=(1, 2)))
            maxj_block = float(maxj_weight) * jnp.sqrt(0.5 * (w_arr[:-1] + w_arr[1:])) * maxj_surface
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
