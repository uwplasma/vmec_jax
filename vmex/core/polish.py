"""High-order/legacy transfer and the VMEX low-order polish preconditioner.

The high-order strong operator is matrix-free.  Its first preconditioner is
the exact nearest-neighbour raw-force block linearization already used by the
implicit VMEX tangent and adjoint paths.  This module only adapts coordinate
representations:

``high residual -> legacy packing -> stored block solve -> high correction``.

The maps preserve the regularized ``rho**abs(m)`` representation, fixed R/Z
edge, VMEX Fourier normalization and m=1 constraint, stellarator symmetry,
and lambda gauge.  No dense high-order Jacobian is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .residuals import m1_constrained_to_physical, m1_physical_to_constrained
from .solver import SpectralState
from .strong_force import HighOrderEquilibriumState
from .transforms import physical_to_internal_scale

Array = Any
_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


@dataclass(frozen=True)
class HighOrderCorrection:
    """Regularized spline coefficients for one geometry/lambda correction."""

    R_cos: Array
    R_sin: Array
    Z_cos: Array
    Z_sin: Array
    L_cos: Array
    L_sin: Array


jax.tree_util.register_dataclass(
    HighOrderCorrection,
    data_fields=list(_FIELDS),
    meta_fields=[],
)


@dataclass(frozen=True, eq=False)
class HighLowTransfer:
    """Linear maps between regularized splines and legacy VMEX packing.

    ``evaluation`` has shape ``(mnmax, ns, nbasis)`` and includes the
    mode-dependent ``rho**abs(m)`` factor.  ``geometry_fit`` and
    ``lambda_fit`` are per-mode left inverses.  Geometry fits have a zero
    terminal coefficient, so a correction cannot change the fixed boundary.
    """

    evaluation: Array
    geometry_fit: Array
    lambda_fit: Array
    mode_scale: Array
    phipf: Array
    lamscale: Array
    m: np.ndarray
    n: np.ndarray
    lthreed: bool
    lasym: bool
    lconm1: bool
    low_project: Callable[[SpectralState], SpectralState]

    @property
    def mnmax(self) -> int:
        return int(self.evaluation.shape[0])

    @property
    def ns(self) -> int:
        return int(self.evaluation.shape[1])

    @property
    def nbasis(self) -> int:
        return int(self.evaluation.shape[2])

    def zeros_high(self, dtype: Any | None = None) -> HighOrderCorrection:
        """Return a zero correction with this transfer layout."""

        dtype = jnp.asarray(self.evaluation).dtype if dtype is None else dtype
        value = jnp.zeros((self.mnmax, self.nbasis), dtype=dtype)
        return HighOrderCorrection(*(value,) * len(_FIELDS))

    def zeros_low(self, dtype: Any | None = None) -> SpectralState:
        """Return a zero legacy tangent with this transfer layout."""

        dtype = jnp.asarray(self.evaluation).dtype if dtype is None else dtype
        value = jnp.zeros((self.ns, self.mnmax), dtype=dtype)
        return SpectralState(*(value,) * len(_FIELDS))

    def project_high(self, correction: HighOrderCorrection) -> HighOrderCorrection:
        """Enforce fixed-edge, symmetry, structural-zero, and gauge constraints."""

        values = {name: jnp.asarray(getattr(correction, name)) for name in _FIELDS}
        expected = (self.mnmax, self.nbasis)
        for name, value in values.items():
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}; expected {expected}")

        for name in ("R_cos", "R_sin", "Z_cos", "Z_sin"):
            values[name] = values[name].at[:, -1].set(0.0)
        if not self.lasym:
            for name in ("R_sin", "Z_cos", "L_cos"):
                values[name] = jnp.zeros_like(values[name])
        gauge = jnp.asarray((self.m == 0) & (self.n == 0))[:, None]
        values["R_sin"] = jnp.where(gauge, 0.0, values["R_sin"])
        values["Z_sin"] = jnp.where(gauge, 0.0, values["Z_sin"])
        values["L_cos"] = jnp.where(gauge, 0.0, values["L_cos"])
        values["L_sin"] = jnp.where(gauge, 0.0, values["L_sin"])
        return HighOrderCorrection(**values)

    def _sample(self, coefficients: Array) -> Array:
        return jnp.einsum(
            "msk,mk->sm",
            jnp.asarray(self.evaluation),
            jnp.asarray(coefficients),
            precision=jax.lax.Precision.HIGHEST,
        )

    def restrict(self, correction: HighOrderCorrection) -> SpectralState:
        """Sample a high-order correction in internal constrained VMEX packing."""

        correction = self.project_high(correction)
        scale = jnp.asarray(self.mode_scale)
        R_cos = self._sample(correction.R_cos) * scale[None, :]
        R_sin = self._sample(correction.R_sin) * scale[None, :]
        Z_cos = self._sample(correction.Z_cos) * scale[None, :]
        Z_sin = self._sample(correction.Z_sin) * scale[None, :]
        R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
            R_cos,
            Z_sin,
            R_sin,
            Z_cos,
            modes=_mode_table(self.m, self.n),
            lthreed=self.lthreed,
            lasym=self.lasym,
            lconm1=self.lconm1,
        )
        lambda_scale = (
            scale[None, :] * jnp.asarray(self.phipf)[:, None] / jnp.asarray(self.lamscale)
        )
        low = SpectralState(
            R_cos=R_cos,
            R_sin=R_sin,
            Z_cos=Z_cos,
            Z_sin=Z_sin,
            L_cos=self._sample(correction.L_cos) * lambda_scale,
            L_sin=self._sample(correction.L_sin) * lambda_scale,
        )
        return self.low_project(low)

    def _fit(self, samples: Array, inverse: Array) -> Array:
        return jnp.einsum(
            "mks,sm->mk",
            jnp.asarray(inverse),
            jnp.asarray(samples),
            precision=jax.lax.Precision.HIGHEST,
        )

    def prolong(self, tangent: SpectralState) -> HighOrderCorrection:
        """Fit a projected legacy correction in regularized spline space."""

        tangent = self.low_project(tangent)
        R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
            tangent.R_cos,
            tangent.Z_sin,
            tangent.R_sin,
            tangent.Z_cos,
            modes=_mode_table(self.m, self.n),
            lthreed=self.lthreed,
            lasym=self.lasym,
            lconm1=self.lconm1,
        )
        inverse_scale = 1.0 / jnp.asarray(self.mode_scale)[None, :]
        safe_phip = jnp.where(jnp.asarray(self.phipf) != 0.0, self.phipf, 1.0)
        lambda_scale = (
            inverse_scale * jnp.asarray(self.lamscale) / jnp.asarray(safe_phip)[:, None]
        )
        correction = HighOrderCorrection(
            R_cos=self._fit(R_cos * inverse_scale, self.geometry_fit),
            R_sin=self._fit(R_sin * inverse_scale, self.geometry_fit),
            Z_cos=self._fit(Z_cos * inverse_scale, self.geometry_fit),
            Z_sin=self._fit(Z_sin * inverse_scale, self.geometry_fit),
            L_cos=self._fit(tangent.L_cos * lambda_scale, self.lambda_fit),
            L_sin=self._fit(tangent.L_sin * lambda_scale, self.lambda_fit),
        )
        return self.project_high(correction)

    def restrict_transpose(self, cotangent: SpectralState) -> HighOrderCorrection:
        """Apply the exact transpose of :meth:`restrict`."""

        dtype = jax.tree.leaves(cotangent)[0].dtype
        return jax.linear_transpose(self.restrict, self.zeros_high(dtype))(cotangent)[0]

    def prolong_transpose(self, cotangent: HighOrderCorrection) -> SpectralState:
        """Apply the exact transpose of :meth:`prolong`."""

        dtype = jax.tree.leaves(cotangent)[0].dtype
        return jax.linear_transpose(self.prolong, self.zeros_low(dtype))(cotangent)[0]


class PreconditionerQuality(NamedTuple):
    """True operator residual after one right-preconditioner application."""

    relative_residual: Array
    maximum: Array
    rms: Array


@dataclass(frozen=True)
class PreconditionerRefreshPolicy:
    """Thresholds for rebuilding a stored low-order factorization."""

    max_alpha_change: float = 0.25
    max_krylov_iterations: int = 80
    max_relative_residual: float = 0.5
    min_jacobian_margin_ratio: float = 0.7
    max_parameter_distance: float = 0.1

    def __post_init__(self) -> None:
        if self.max_alpha_change <= 0.0:
            raise ValueError("max_alpha_change must be positive")
        if self.max_krylov_iterations < 1:
            raise ValueError("max_krylov_iterations must be positive")
        if self.max_relative_residual <= 0.0:
            raise ValueError("max_relative_residual must be positive")
        if not 0.0 < self.min_jacobian_margin_ratio <= 1.0:
            raise ValueError("min_jacobian_margin_ratio must lie in (0, 1]")
        if self.max_parameter_distance <= 0.0:
            raise ValueError("max_parameter_distance must be positive")


@dataclass(frozen=True)
class PreconditionerSnapshot:
    """Cheap nonlinear-stage data used by the factor refresh policy."""

    alpha: float
    radial_degree: int
    radial_size: int
    krylov_iterations: int
    relative_residual: float
    jacobian_margin: float
    parameter_distance: float = 0.0
    transpose_converged: bool = True


class PreconditionerRefreshDecision(NamedTuple):
    """Host-side refresh decision with reviewer-visible reasons."""

    refresh: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class LowOrderPreconditioner:
    """Stored raw-force block inverse lifted to high-order coefficient space."""

    transfer: HighLowTransfer
    system: Any
    factor_build_seconds: float

    def apply(self, rhs: HighOrderCorrection) -> HighOrderCorrection:
        """Apply ``prolong * A_low^-1 * restrict``."""

        from .implicit import _raw_block_apply

        low_rhs = self.transfer.restrict(rhs)
        return self.transfer.prolong(_raw_block_apply(self.system, low_rhs))

    def apply_transpose(self, rhs: HighOrderCorrection) -> HighOrderCorrection:
        """Apply the algebraic transpose using the same stored factors."""

        from .implicit import _raw_block_apply

        low_rhs = self.transfer.prolong_transpose(rhs)
        low_solution = _raw_block_apply(self.system, low_rhs, transpose=True)
        return self.transfer.restrict_transpose(low_solution)


def _mode_table(m: np.ndarray, n: np.ndarray):
    """Construct only the mode metadata needed by the m=1 linear maps."""

    from .fourier import ModeTable

    return ModeTable(m=np.asarray(m), n=np.asarray(n))


def make_high_low_transfer(
    native: HighOrderEquilibriumState,
    runtime: Any,
    *,
    low_project: Callable[[SpectralState], SpectralState] | None = None,
) -> HighLowTransfer:
    """Build reusable high/low transfer matrices for one equilibrium layout."""

    modes = runtime.modes
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if not np.array_equal(m, np.asarray(native.m)) or not np.array_equal(
        n, np.asarray(native.n)
    ):
        raise ValueError("native and legacy mode tables must match")
    s = np.asarray(runtime.setup.s_full, dtype=float)
    rho = np.sqrt(np.maximum(s, 0.0))
    basis_values = np.asarray(native.radial_basis.basis_matrix(s), dtype=float)
    evaluation = rho[None, :, None] ** np.abs(m)[:, None, None] * basis_values[None]
    nbasis = int(native.radial_basis.size)
    geometry_fit = np.zeros((m.size, nbasis, s.size), dtype=float)
    lambda_fit = np.zeros_like(geometry_fit)
    for mode in range(m.size):
        geometry_fit[mode, :-1] = np.linalg.pinv(
            evaluation[mode, :, :-1], rcond=1.0e-12
        )
        lambda_fit[mode] = np.linalg.pinv(evaluation[mode], rcond=1.0e-12)
    project = (lambda value: value) if low_project is None else low_project
    return HighLowTransfer(
        evaluation=jnp.asarray(evaluation),
        geometry_fit=jnp.asarray(geometry_fit),
        lambda_fit=jnp.asarray(lambda_fit),
        mode_scale=jnp.asarray(physical_to_internal_scale(modes, runtime.trig)),
        phipf=jnp.asarray(runtime.setup.phipf),
        lamscale=jnp.asarray(runtime.setup.lamscale),
        m=m,
        n=n,
        lthreed=bool(runtime.setup.lthreed),
        lasym=bool(runtime.setup.lasym),
        lconm1=bool(runtime.setup.lconm1),
        low_project=project,
    )


def build_low_order_preconditioner(
    native: HighOrderEquilibriumState,
    params: Any,
    config: Any,
    legacy_state: SpectralState,
    dof_mask: SpectralState,
    *,
    probe_chunk_size: int = 8,
) -> LowOrderPreconditioner:
    """Assemble and factor the existing exact low-order raw-force operator."""

    from . import implicit

    runtime = implicit.runtime_from_params(params, config)
    project = implicit._dof_projector(config, dof_mask)
    transfer = make_high_low_transfer(native, runtime, low_project=project)
    started = perf_counter()
    system = implicit._raw_block_system(
        params,
        config,
        legacy_state,
        dof_mask,
        implicit._active_state_fields(config),
        int(probe_chunk_size),
    )
    elapsed = perf_counter() - started
    return LowOrderPreconditioner(transfer, system, elapsed)


def preconditioner_quality(
    operator: Callable[[HighOrderCorrection], HighOrderCorrection],
    preconditioner: Callable[[HighOrderCorrection], HighOrderCorrection],
    probes: HighOrderCorrection,
) -> PreconditionerQuality:
    """Measure true relative residuals for a batch of leading-axis probes."""

    responses = jax.vmap(lambda rhs: operator(preconditioner(rhs)))(probes)
    residuals = jax.tree.map(jnp.subtract, responses, probes)

    def norms(tree):
        leaves = jax.tree.leaves(tree)
        squared = sum(jnp.sum(jnp.abs(leaf) ** 2, axis=tuple(range(1, leaf.ndim))) for leaf in leaves)
        return jnp.sqrt(squared)

    dtype = jax.tree.leaves(probes)[0].dtype
    relative = norms(residuals) / jnp.maximum(norms(probes), jnp.finfo(dtype).tiny)
    return PreconditionerQuality(
        relative_residual=relative,
        maximum=jnp.max(relative),
        rms=jnp.sqrt(jnp.mean(relative * relative)),
    )


def preconditioner_refresh_decision(
    previous: PreconditionerSnapshot,
    current: PreconditionerSnapshot,
    policy: PreconditionerRefreshPolicy | None = None,
) -> PreconditionerRefreshDecision:
    """Return whether nonlinear progress has invalidated stored factors."""

    policy = PreconditionerRefreshPolicy() if policy is None else policy
    reasons: list[str] = []
    if abs(current.alpha - previous.alpha) > policy.max_alpha_change:
        reasons.append("continuation-step")
    if (
        current.radial_degree != previous.radial_degree
        or current.radial_size != previous.radial_size
    ):
        reasons.append("radial-grid")
    if current.krylov_iterations > policy.max_krylov_iterations:
        reasons.append("krylov-work")
    if current.relative_residual > policy.max_relative_residual:
        reasons.append("linear-quality")
    reference_margin = max(abs(previous.jacobian_margin), np.finfo(float).tiny)
    if current.jacobian_margin < policy.min_jacobian_margin_ratio * reference_margin:
        reasons.append("jacobian-margin")
    if current.parameter_distance > policy.max_parameter_distance:
        reasons.append("parameter-distance")
    if not current.transpose_converged:
        reasons.append("transpose-certificate")
    return PreconditionerRefreshDecision(bool(reasons), tuple(reasons))


__all__ = [
    "HighLowTransfer",
    "HighOrderCorrection",
    "LowOrderPreconditioner",
    "PreconditionerQuality",
    "PreconditionerRefreshDecision",
    "PreconditionerRefreshPolicy",
    "PreconditionerSnapshot",
    "build_low_order_preconditioner",
    "make_high_low_transfer",
    "preconditioner_quality",
    "preconditioner_refresh_decision",
]
