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

from dataclasses import dataclass, replace
from functools import partial
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

    def _restrict_unprojected(self, correction: HighOrderCorrection) -> SpectralState:
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
        return SpectralState(
            R_cos=R_cos,
            R_sin=R_sin,
            Z_cos=Z_cos,
            Z_sin=Z_sin,
            L_cos=self._sample(correction.L_cos) * lambda_scale,
            L_sin=self._sample(correction.L_sin) * lambda_scale,
        )

    def restrict(self, correction: HighOrderCorrection) -> SpectralState:
        """Sample a high-order correction in internal constrained VMEX packing."""

        return self.low_project(self._restrict_unprojected(correction))

    def _fit(self, samples: Array, inverse: Array) -> Array:
        return jnp.einsum(
            "mks,sm->mk",
            jnp.asarray(inverse),
            jnp.asarray(samples),
            precision=jax.lax.Precision.HIGHEST,
        )

    def _prolong_projected(self, tangent: SpectralState) -> HighOrderCorrection:
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

    def prolong(self, tangent: SpectralState) -> HighOrderCorrection:
        """Fit a projected legacy correction in regularized spline space."""

        return self._prolong_projected(self.low_project(tangent))

    def restrict_transpose(self, cotangent: SpectralState) -> HighOrderCorrection:
        """Apply the exact transpose of :meth:`restrict`."""

        dtype = jax.tree.leaves(cotangent)[0].dtype
        # The evolved-DOF projector is symmetric.  Applying it explicitly
        # avoids transposing the m=1 indexed-update implementation, which JAX
        # cannot lower when positive/negative mode index arrays may alias.
        projected = self.low_project(cotangent)
        return jax.linear_transpose(
            self._restrict_unprojected, self.zeros_high(dtype)
        )(projected)[0]

    def prolong_transpose(self, cotangent: HighOrderCorrection) -> SpectralState:
        """Apply the exact transpose of :meth:`prolong`."""

        dtype = jax.tree.leaves(cotangent)[0].dtype
        low_cotangent = jax.linear_transpose(
            self._prolong_projected, self.zeros_low(dtype)
        )(cotangent)[0]
        return self.low_project(low_cotangent)


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
class StrongRootGroup:
    """One small independent ``(m, |n|)`` native-spline coordinate block."""

    high_indices: np.ndarray
    basis: Array
    start: int
    stop: int
    m: int
    abs_n: int


def _flatten_high(correction: HighOrderCorrection) -> Array:
    return jnp.concatenate(
        tuple(jnp.ravel(jnp.asarray(getattr(correction, name))) for name in _FIELDS)
    )


def _unflatten_high(vector: Array, mnmax: int, nbasis: int) -> HighOrderCorrection:
    vector = jnp.asarray(vector)
    block = int(mnmax) * int(nbasis)
    values = [
        vector[index * block : (index + 1) * block].reshape((mnmax, nbasis))
        for index in range(len(_FIELDS))
    ]
    return HighOrderCorrection(*values)


@dataclass(frozen=True, eq=False)
class StrongRootLayout:
    """Independent native-spline coordinates for the square strong root.

    Each small ``(m, |n|)`` block is the numerical image of the tested
    ``prolong(restrict(.))`` map.  This removes fixed-edge, symmetry, gauge,
    inactive-axis, and coupled 3-D ``m=1,+/-n`` coordinates without building
    a global dense projector.  Packing and unpacking use orthonormal local SVD
    bases and are therefore exact transposes.
    """

    mnmax: int
    nbasis: int
    groups: tuple[StrongRootGroup, ...]

    @property
    def size(self) -> int:
        return 0 if not self.groups else int(self.groups[-1].stop)

    def pack(self, correction: HighOrderCorrection) -> Array:
        flat = _flatten_high(correction)
        return jnp.concatenate(
            tuple(
                jnp.asarray(group.basis).T
                @ flat[jnp.asarray(group.high_indices)]
                for group in self.groups
            )
        )

    def unpack(self, vector: Array) -> HighOrderCorrection:
        vector = jnp.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError(f"free vector has shape {vector.shape}; expected {(self.size,)}")
        flat = jnp.zeros((len(_FIELDS) * self.mnmax * self.nbasis,), dtype=vector.dtype)
        for group in self.groups:
            values = jnp.asarray(group.basis) @ vector[group.start : group.stop]
            flat = flat.at[jnp.asarray(group.high_indices)].add(values)
        return _unflatten_high(flat, self.mnmax, self.nbasis)


@dataclass(frozen=True, eq=False)
class StrongPhysicalChart:
    """Gauge-free coordinates and equations for the strong-force root.

    ``coordinate_basis`` spans the nullspace of the exactly linear tangential
    coordinate equation.  ``equation_basis`` spans the actual radial/helical
    force-output channels in the constrained layout.  The equation basis is
    assembled from small structural layout blocks; constructing this chart
    never probes or stores the physical-force Jacobian.
    """

    coordinate_basis: Array
    equation_basis: Array
    gauge_rank: int
    build_seconds: float

    @property
    def full_size(self) -> int:
        return int(self.coordinate_basis.shape[0])

    @property
    def size(self) -> int:
        return int(self.coordinate_basis.shape[1])

    def lift(self, vector: Array) -> Array:
        """Lift one gauge-free vector into the full constrained layout."""

        vector = jnp.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError(
                f"physical vector has shape {vector.shape}; expected {(self.size,)}"
            )
        return jnp.asarray(self.coordinate_basis) @ vector

    def project(self, residual: Array) -> Array:
        """Project a full residual away from coordinate-gauge equations."""

        residual = jnp.asarray(residual)
        if residual.shape != (self.full_size,):
            raise ValueError(
                f"full residual has shape {residual.shape}; "
                f"expected {(self.full_size,)}"
            )
        return jnp.asarray(self.equation_basis).T @ residual


@dataclass(frozen=True, eq=False)
class StrongRootRuntime:
    """Reusable grids, transforms, constraints, and scaling for a square root."""

    native: HighOrderEquilibriumState
    transfer: HighLowTransfer
    low_preconditioner: LowOrderPreconditioner
    layout: StrongRootLayout
    coordinate_scale: Array
    equation_scale: Array
    radial_nodes: Array
    theta: Array
    zeta: Array
    cosine_projection: Array
    sine_projection: Array
    radial_fit: Array
    normalization_denominator: Array
    gauge_length: Array
    strong_block_sign: Array
    strong_scale: Array
    operator_balance: Array
    force_floor: float


@dataclass(frozen=True, eq=False)
class StrongModeBlockPreconditioner:
    """Bounded Fourier-mode factors for a strong-root Jacobian pencil."""

    indices: tuple[Array, ...]
    low_blocks: tuple[Array, ...]
    strong_blocks: tuple[Array, ...]
    build_seconds: float

    def apply(
        self,
        rhs: Array,
        alpha: Array = 1.0,
        dtau: Array | float = jnp.inf,
    ) -> Array:
        """Apply regularized block solves without a dense global Jacobian."""

        return self._apply(rhs, alpha, dtau, transpose=False)

    def apply_transpose(
        self,
        rhs: Array,
        alpha: Array = 1.0,
        dtau: Array | float = jnp.inf,
    ) -> Array:
        """Apply the exact transpose factors used by implicit adjoints."""

        return self._apply(rhs, alpha, dtau, transpose=True)

    def _apply(
        self,
        rhs: Array,
        alpha: Array,
        dtau: Array | float,
        *,
        transpose: bool,
    ) -> Array:
        rhs = jnp.asarray(rhs)
        alpha = jnp.asarray(alpha, dtype=rhs.dtype)
        inverse_dtau = jnp.where(
            jnp.isfinite(jnp.asarray(dtau)),
            1.0 / jnp.asarray(dtau, dtype=rhs.dtype),
            jnp.asarray(0.0, dtype=rhs.dtype),
        )
        result = jnp.zeros_like(rhs)
        for indices, low, strong in zip(
            self.indices, self.low_blocks, self.strong_blocks, strict=True
        ):
            matrix = (1.0 - alpha) * low + alpha * strong
            if transpose:
                matrix = matrix.T
            scale = jnp.maximum(jnp.linalg.norm(matrix, ord=jnp.inf), 1.0)
            regularization = jnp.where(
                inverse_dtau > 0.0,
                32.0 * jnp.finfo(rhs.dtype).eps * scale,
                0.0,
            )
            shifted = matrix + (
                inverse_dtau + regularization
            ) * jnp.eye(matrix.shape[0], dtype=rhs.dtype)
            result = result.at[indices].set(jnp.linalg.solve(shifted, rhs[indices]))
        return result


@dataclass(frozen=True, eq=False)
class LowOrderPreconditioner:
    """Stored raw-force block inverse lifted to high-order coefficient space."""

    transfer: HighLowTransfer
    system: Any
    legacy_coordinates: SpectralState
    legacy_defect: SpectralState
    legacy_residual: Callable[[SpectralState], SpectralState]
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

    def residual(self, tangent: SpectralState) -> SpectralState:
        """Evaluate and row-scale the nonlinear legacy raw-force endpoint."""

        candidate = self.system.project(
            jax.tree.map(jnp.add, self.legacy_coordinates, tangent)
        )
        force = jax.tree.map(
            jnp.subtract,
            self.legacy_residual(candidate),
            self.legacy_defect,
        )
        scaled = self.system.pack(force) * jnp.asarray(self.system.row_scale)
        return self.system.project(self.system.unpack(scaled))

    def solve_scaled(self, rhs: SpectralState) -> SpectralState:
        """Invert a row-scaled legacy residual with the stored raw factors."""

        from .implicit import _raw_block_apply

        raw_packed = self.system.pack(rhs) / jnp.asarray(self.system.row_scale)
        return _raw_block_apply(self.system, self.system.unpack(raw_packed))

    def solve_scaled_transpose(self, rhs: SpectralState) -> SpectralState:
        """Invert the transpose of the row-scaled legacy residual.

        If the raw block operator is ``A`` and ``D`` is its stored row
        scaling, :meth:`residual` linearizes to ``D A``.  Its transpose
        inverse is therefore ``D^-1 A^-T``; the order differs from the
        forward :meth:`solve_scaled` path and is kept explicit here.
        """

        from .implicit import _raw_block_apply

        raw_solution = _raw_block_apply(self.system, rhs, transpose=True)
        scaled = self.system.pack(raw_solution) / jnp.asarray(self.system.row_scale)
        return self.system.project(self.system.unpack(scaled))


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


def make_strong_root_layout(
    dof_mask: SpectralState,
    native: HighOrderEquilibriumState,
    *,
    transfer: HighLowTransfer | None = None,
    lconm1: bool = True,
) -> StrongRootLayout:
    """Build independent local native-spline coordinates.

    ``lconm1`` is retained for source compatibility; the supplied transfer is
    the source of truth for that constraint and all other structural masks.
    """

    masks = {
        name: np.asarray(getattr(dof_mask, name), dtype=bool)
        for name in _FIELDS
    }
    expected_low_shape = masks["R_cos"].shape
    if (
        any(mask.shape != expected_low_shape for mask in masks.values())
        or expected_low_shape[1] != np.asarray(native.m).size
    ):
        raise ValueError("dof mask and native mode layout must match")
    del lconm1
    if transfer is None:
        raise ValueError("native strong-root layout requires a high/low transfer")
    _, mnmax = expected_low_shape
    nbasis = int(native.radial_basis.size)
    m = np.asarray(native.m, dtype=int)
    n = np.asarray(native.n, dtype=int)
    structurally_active = np.asarray(
        _flatten_high(transfer.project_high(
            HighOrderCorrection(*(
                jnp.ones((mnmax, nbasis), dtype=jnp.float64)
                for _ in _FIELDS
            ))
        )),
        dtype=bool,
    )
    # The transfer owns the production low projector, while ``dof_mask`` is
    # also an explicit validation input.  Retain only field/mode blocks with
    # at least one evolved legacy sample so a stale or zero mask cannot create
    # apparently free native coordinates.
    low_active = np.stack(
        tuple(np.any(masks[name], axis=0) for name in _FIELDS)
    )
    active = structurally_active & np.repeat(low_active.reshape(-1), nbasis)
    block = mnmax * nbasis
    groups: list[StrongRootGroup] = []
    start = 0
    for mode_key in sorted({(int(mm), abs(int(nn))) for mm, nn in zip(m, n)}):
        mode_indices = np.flatnonzero(
            (m == mode_key[0]) & (np.abs(n) == mode_key[1])
        )
        candidates = []
        for field in range(len(_FIELDS)):
            for mode in mode_indices:
                base = field * block + int(mode) * nbasis
                candidates.extend(base + np.arange(nbasis, dtype=np.int32))
        candidates = np.asarray(candidates, dtype=np.int32)
        candidates = candidates[active[candidates]]
        if candidates.size == 0:
            continue

        def local_project(values):
            flat = jnp.zeros((len(_FIELDS) * block,), dtype=values.dtype)
            flat = flat.at[jnp.asarray(candidates)].set(values)
            high = _unflatten_high(flat, mnmax, nbasis)
            feasible = transfer.prolong(transfer.restrict(high))
            return _flatten_high(feasible)[jnp.asarray(candidates)]

        identity = jnp.eye(candidates.size, dtype=jnp.float64)
        # vmap rows are input probes; transpose so columns are image vectors.
        image = np.asarray(jax.vmap(local_project)(identity)).T
        left, singular, _ = np.linalg.svd(image, full_matrices=False)
        if singular.size == 0:
            continue
        rank = int(np.sum(singular > 1.0e-10 * singular[0]))
        if rank == 0:
            continue
        stop = start + rank
        groups.append(StrongRootGroup(
            high_indices=candidates,
            basis=jnp.asarray(left[:, :rank]),
            start=start,
            stop=stop,
            m=mode_key[0],
            abs_n=mode_key[1],
        ))
        start = stop
    return StrongRootLayout(
        mnmax=mnmax,
        nbasis=nbasis,
        groups=tuple(groups),
    )


def apply_high_order_correction(
    native: HighOrderEquilibriumState,
    correction: HighOrderCorrection,
) -> HighOrderEquilibriumState:
    """Add a constrained geometry correction while leaving profiles fixed."""

    return replace(
        native,
        R_cos=native.R_cos + correction.R_cos,
        R_sin=native.R_sin + correction.R_sin,
        Z_cos=native.Z_cos + correction.Z_cos,
        Z_sin=native.Z_sin + correction.Z_sin,
        L_cos=native.L_cos + correction.L_cos,
        L_sin=native.L_sin + correction.L_sin,
        source=f"{native.source}; strong-root correction",
    )


def _coordinate_gauge_samples(
    state: HighOrderEquilibriumState,
    native: HighOrderEquilibriumState,
    runtime: StrongRootRuntime,
    points: Array,
) -> Array:
    """Evaluate the linear tangential-displacement coordinate equation."""

    from .strong_force import _RZL

    def coordinate_gauge(point):
        base_rz = jnp.asarray(_RZL(native, point)[:2])
        current_rz = jnp.asarray(_RZL(state, point)[:2])
        theta_direction = jnp.asarray([0.0, 1.0, 0.0], dtype=point.dtype)
        _, tangent = jax.jvp(
            lambda location: jnp.asarray(_RZL(runtime.native, location)[:2]),
            (point,),
            (theta_direction,),
        )
        tangent_norm = jnp.sqrt(
            jnp.vdot(tangent, tangent).real + float(runtime.force_floor) ** 2
        )
        return jnp.vdot(current_rz - base_rz, tangent).real / (
            tangent_norm * jnp.asarray(runtime.gauge_length)
        )

    return jax.vmap(coordinate_gauge)(points)


def _fit_regularized_channel(
    samples: Array,
    angular_projection: Array,
    radial: Array,
    runtime: StrongRootRuntime,
) -> Array:
    """Fourier project and remove analytic axis powers before radial fitting."""

    samples = jnp.asarray(samples).reshape((radial.size, -1))
    modes = jnp.einsum(
        "ra,ma->rm", samples, jnp.asarray(angular_projection)
    )
    powers = radial[:, None] ** jnp.asarray(np.abs(runtime.native.m))[None, :]
    safe_powers = jnp.maximum(powers, jnp.finfo(radial.dtype).tiny)
    return jnp.asarray(runtime.radial_fit) @ (modes / safe_powers)


def _coordinate_gauge_residual_unscaled(
    vector: Array,
    runtime: StrongRootRuntime,
) -> Array:
    """Project only the linear coordinate equation, without physical forces."""

    correction = runtime.layout.unpack(
        jnp.asarray(runtime.coordinate_scale) * jnp.asarray(vector)
    )
    state = apply_high_order_correction(runtime.native, correction)
    radial = jnp.asarray(runtime.radial_nodes)
    rr, tt, zz = jnp.meshgrid(
        radial,
        jnp.asarray(runtime.theta),
        jnp.asarray(runtime.zeta),
        indexing="ij",
    )
    points = jnp.stack((rr.reshape(-1), tt.reshape(-1), zz.reshape(-1)), axis=-1)
    gauge = _coordinate_gauge_samples(state, runtime.native, runtime, points)
    gauge_coefficients = _fit_regularized_channel(
        gauge,
        runtime.sine_projection,
        radial,
        runtime,
    )
    zero = jnp.zeros_like(jnp.asarray(runtime.native.R_cos))
    coefficients = HighOrderCorrection(
        R_cos=zero,
        R_sin=zero,
        Z_cos=zero,
        Z_sin=gauge_coefficients.T,
        L_cos=zero,
        L_sin=zero,
    )
    return jnp.asarray(runtime.equation_scale) * runtime.layout.pack(coefficients)


def _strong_residual_unscaled(
    vector: Array,
    runtime: StrongRootRuntime,
    native: HighOrderEquilibriumState | None = None,
    *,
    include_coordinate_gauge: bool = True,
) -> Array:
    """Project normalized physical force onto the reduced solve space."""

    from .strong_force import evaluate_strong_force

    native = runtime.native if native is None else native
    correction = runtime.layout.unpack(
        jnp.asarray(runtime.coordinate_scale) * jnp.asarray(vector)
    )
    state = apply_high_order_correction(native, correction)
    radial = jnp.asarray(runtime.radial_nodes)
    theta = jnp.asarray(runtime.theta)
    zeta = jnp.asarray(runtime.zeta)
    rr, tt, zz = jnp.meshgrid(radial, theta, zeta, indexing="ij")
    samples = evaluate_strong_force(state, rr, tt, zz)
    denominator = jnp.asarray(runtime.normalization_denominator)
    # DESC's two-component force objective uses the coordinate-volume factor
    # on both physical channels.  This preserves the off-axis zero set while
    # giving the projected equations their regular near-axis measure.  Apply
    # it before Fourier/radial fitting; post-fit row scaling is not equivalent.
    volume_weight = jnp.abs(samples.sqrt_g)
    radial_force = (
        2.0 * samples.signed_radial_force_density * volume_weight / denominator
    )
    helical_force = (
        2.0 * samples.signed_helical_force_density * volume_weight / denominator
    )
    radial_coefficients = _fit_regularized_channel(
        radial_force, runtime.cosine_projection, radial, runtime
    )
    helical_coefficients = _fit_regularized_channel(
        helical_force, runtime.sine_projection, radial, runtime
    )
    zero = jnp.zeros_like(jnp.asarray(runtime.native.R_cos))
    if include_coordinate_gauge:
        points = jnp.stack(
            (rr.reshape(-1), tt.reshape(-1), zz.reshape(-1)),
            axis=-1,
        )
        gauge = _coordinate_gauge_samples(state, native, runtime, points)
        gauge_coefficients = _fit_regularized_channel(
            gauge, runtime.sine_projection, radial, runtime
        ).T
    else:
        gauge_coefficients = zero
    force_coefficients = HighOrderCorrection(
        R_cos=radial_coefficients.T,
        R_sin=zero,
        Z_cos=zero,
        Z_sin=gauge_coefficients,
        L_cos=zero,
        L_sin=helical_coefficients.T,
    )
    signs = jnp.asarray(runtime.strong_block_sign)
    oriented = replace(
        force_coefficients,
        R_cos=force_coefficients.R_cos * signs[0],
        R_sin=force_coefficients.R_sin * signs[0],
        Z_cos=force_coefficients.Z_cos * signs[1],
        Z_sin=force_coefficients.Z_sin * signs[1],
        L_cos=force_coefficients.L_cos * signs[2],
        L_sin=force_coefficients.L_sin * signs[2],
    )
    return jnp.asarray(runtime.equation_scale) * runtime.layout.pack(oriented)


@partial(jax.jit, static_argnames=("runtime",))
def strong_root_residual(
    vector: Array,
    runtime: StrongRootRuntime,
    alpha: Array = 1.0,
) -> Array:
    """Square residual homotopy from legacy raw force to strong force."""

    vector = jnp.asarray(vector)
    high_tangent = runtime.layout.unpack(
        jnp.asarray(runtime.coordinate_scale) * vector
    )
    low_tangent = runtime.transfer.restrict(high_tangent)
    low_force = runtime.low_preconditioner.residual(low_tangent)
    low = jnp.asarray(runtime.equation_scale) * runtime.layout.pack(
        runtime.transfer.prolong(low_force)
    )
    strong = _strong_residual_unscaled(vector, runtime) / jnp.asarray(runtime.strong_scale)
    alpha = jnp.asarray(alpha, dtype=vector.dtype)
    return low + alpha * (strong - low)


@partial(jax.jit, static_argnames=("runtime",))
def strong_root_residual_at_native(
    vector: Array,
    native: HighOrderEquilibriumState,
    runtime: StrongRootRuntime,
) -> Array:
    """Evaluate the frozen-chart strong endpoint at a dynamic native state.

    The collocation grid, normalization, row scaling, transfer, and gauge
    length remain fixed in ``runtime``.  This is the local residual required
    by implicit tangents and adjoints; at a converged root, derivatives of
    any positive residual scaling do not change the implicit derivative.
    """

    vector = jnp.asarray(vector)
    strong = _strong_residual_unscaled(vector, runtime, native)
    return strong / jnp.asarray(runtime.strong_scale)


def _physical_equation_basis(layout: StrongRootLayout) -> np.ndarray:
    """Build an orthonormal basis for radial/helical force-output rows."""

    full_size = layout.size
    high_block = int(layout.mnmax) * int(layout.nbasis)
    radial_field = _FIELDS.index("R_cos")
    helical_field = _FIELDS.index("L_sin")
    columns: list[np.ndarray] = []
    for group in layout.groups:
        fields = np.asarray(group.high_indices, dtype=int) // high_block
        physical = (fields == radial_field) | (fields == helical_field)
        injection = np.asarray(group.basis, dtype=float).T[:, physical]
        if injection.size == 0:
            continue
        left, singular_values, _ = np.linalg.svd(
            injection,
            full_matrices=False,
        )
        if singular_values.size == 0 or singular_values[0] <= 0.0:
            continue
        rank = int(np.sum(singular_values > 1.0e-10 * singular_values[0]))
        for local in left[:, :rank].T:
            column = np.zeros((full_size,), dtype=float)
            column[group.start : group.stop] = local
            columns.append(column)
    if not columns:
        raise ValueError("strong root has no physical force-output equations")
    return np.column_stack(columns)


def make_strong_physical_chart(
    runtime: StrongRootRuntime,
    *,
    relative_tolerance: float = 1.0e-10,
) -> StrongPhysicalChart:
    """Eliminate the exactly linear coordinate gauge from a strong root.

    The one-time dense factorization is restricted to the coordinate-gauge
    operator.  The nonlinear physical force and all subsequent JVP/VJP calls
    remain matrix-free.  ``relative_tolerance`` defines the numerical rank of
    the gauge operator and must leave at least one physical coordinate.
    """

    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    started = perf_counter()
    size = runtime.layout.size
    zero = jnp.zeros((size,), dtype=jnp.asarray(runtime.native.R_cos).dtype)
    gauge_operator = jax.jacfwd(
        lambda value: _coordinate_gauge_residual_unscaled(value, runtime)
    )(zero)
    _, singular_values, right_transpose = np.linalg.svd(
        np.asarray(jax.device_get(gauge_operator)),
        full_matrices=True,
    )
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("coordinate-gauge operator has no independent equations")
    gauge_rank = int(
        np.sum(singular_values > relative_tolerance * singular_values[0])
    )
    if gauge_rank <= 0 or gauge_rank >= size:
        raise ValueError(
            "coordinate-gauge rank must be positive and smaller than the root"
        )
    equation_basis = _physical_equation_basis(runtime.layout)
    physical_size = size - gauge_rank
    if equation_basis.shape != (size, physical_size):
        raise ValueError(
            "physical force-output equation count does not match gauge-free "
            f"coordinates: {equation_basis.shape[1]} != {physical_size}"
        )
    return StrongPhysicalChart(
        coordinate_basis=jnp.asarray(right_transpose[gauge_rank:].T),
        equation_basis=jnp.asarray(equation_basis),
        gauge_rank=gauge_rank,
        build_seconds=perf_counter() - started,
    )


@partial(jax.jit, static_argnames=("runtime", "chart"))
def strong_physical_residual(
    vector: Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
    alpha: Array = 1.0,
) -> Array:
    """Evaluate the square strong root in exact gauge-free coordinates."""

    full = chart.lift(vector)
    low = chart.project(strong_root_residual(full, runtime, 0.0))
    strong = chart.project(
        _strong_residual_unscaled(
            full,
            runtime,
            include_coordinate_gauge=False,
        )
        / jnp.asarray(runtime.strong_scale)
    )
    alpha = jnp.asarray(alpha, dtype=jnp.asarray(vector).dtype)
    return low + alpha * (strong - low)


def _streaming_ruiz_scales(
    residual: Callable[[Array], Array],
    zero: Array,
    *,
    iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrate global row/column 2-norms without retaining a Jacobian."""

    _, jvp = jax.linearize(residual, zero)
    apply_jvp = jax.jit(jvp)

    size = int(np.asarray(zero).size)
    dtype = np.asarray(zero).dtype
    rows = np.ones((size,), dtype=float)
    columns = np.ones((size,), dtype=float)
    tiny = np.finfo(float).tiny
    limit = 1.0e12
    for _ in range(int(iterations)):
        row_squared = np.zeros((size,), dtype=float)
        column_norm = np.zeros((size,), dtype=float)
        for index in range(size):
            direction = np.zeros((size,), dtype=dtype)
            direction[index] = columns[index]
            response = rows * np.asarray(apply_jvp(jnp.asarray(direction)))
            row_squared += response**2
            column_norm[index] = np.linalg.norm(response)
        row_norm = np.sqrt(row_squared)
        row_floor = max(
            1.0e-14 * float(np.max(row_norm, initial=0.0)), tiny
        )
        column_floor = max(
            1.0e-14 * float(np.max(column_norm, initial=0.0)), tiny
        )
        rows *= 1.0 / np.sqrt(np.maximum(row_norm, row_floor))
        columns *= 1.0 / np.sqrt(np.maximum(column_norm, column_floor))
        rows = np.clip(rows, 1.0 / limit, limit)
        columns = np.clip(columns, 1.0 / limit, limit)
    return np.clip(rows, 1.0 / limit, limit), np.clip(columns, 1.0 / limit, limit)


def make_strong_root_runtime(
    native: HighOrderEquilibriumState,
    low_preconditioner: LowOrderPreconditioner,
    dof_mask: SpectralState,
    *,
    force_floor: float = 1.0e-30,
    balance_iterations: int = 4,
    orientation_eigenpairs: int = 6,
) -> StrongRootRuntime:
    """Build distinct collocation/projection data and balance the strong residual."""

    if force_floor <= 0.0:
        raise ValueError("force_floor must be positive")
    if balance_iterations < 1:
        raise ValueError("balance_iterations must be positive")
    if orientation_eigenpairs < 1:
        raise ValueError("orientation_eigenpairs must be positive")
    transfer = low_preconditioner.transfer
    layout = make_strong_root_layout(
        dof_mask,
        native,
        transfer=transfer,
        lconm1=transfer.lconm1,
    )
    if layout.size == 0:
        raise ValueError("strong-root layout contains no free physical displacement")
    # The force contains nonlinear products of first and second radial
    # derivatives, so sampling it at exactly ``nbasis`` collocation points can
    # alias unresolved radial content into the square residual.  Evaluate on
    # the basis' higher-order Gauss rule and project back to the same
    # ``nbasis`` coefficients.  The residual remains square after the
    # projection, while trial states that only improve the solve nodes can no
    # longer hide large between-node force.
    radial_s_nodes = np.asarray(
        native.radial_basis.quadrature_nodes, dtype=float
    )
    radial_nodes = np.sqrt(radial_s_nodes)
    radial_weights = np.asarray(native.radial_basis.quadrature_weights, dtype=float)
    radial_matrix = np.asarray(
        native.radial_basis.basis_matrix(radial_s_nodes), dtype=float
    )
    sqrt_weights = np.sqrt(radial_weights)
    weighted_matrix = sqrt_weights[:, None] * radial_matrix
    radial_fit = np.linalg.pinv(weighted_matrix, rcond=1.0e-12) * sqrt_weights[None, :]
    m = np.asarray(native.m, dtype=int)
    n = np.asarray(native.n, dtype=int)
    # The nonlinear force contains metric inverses and is not band-limited at
    # the retained geometry order.  The former ``2*mmax + 3`` grid resolves
    # the requested output modes but aliases their nonlinear source.  The
    # production m=5 rank gate gains one physical direction at 25 points and
    # is unchanged at 37, so retain that converged ``4*mmax + 5`` rule.
    ntheta = max(4 * int(np.max(np.abs(m), initial=0)) + 5, 4)
    nzeta = max(2 * int(np.max(np.abs(n), initial=0)) + 3, 1)
    theta_grid = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta_grid = 2.0 * np.pi * np.arange(nzeta) / nzeta
    theta, zeta = np.meshgrid(theta_grid, zeta_grid, indexing="ij")
    phase = m[:, None] * theta.reshape(1, -1) - n[:, None] * zeta.reshape(1, -1)
    angular_count = phase.shape[1]
    nonconstant = ((m != 0) | (n != 0)).astype(float)[:, None]
    normalization = (1.0 + nonconstant) / float(angular_count)
    cosine_projection = normalization * np.cos(phase)
    sine_projection = normalization * np.sin(phase)
    rr, tt, zz = jnp.meshgrid(
        jnp.asarray(radial_nodes),
        jnp.asarray(theta_grid),
        jnp.asarray(zeta_grid),
        indexing="ij",
    )
    from .strong_force import evaluate_strong_force

    base_samples = evaluate_strong_force(native, rr, tt, zz)
    base_lorentz = jnp.cross(base_samples.J, base_samples.B)
    base_grad_pressure = base_lorentz - base_samples.force
    floor_squared = float(force_floor) ** 2
    normalization_denominator = (
        jnp.sqrt(jnp.sum(base_lorentz * base_lorentz, axis=-1) + floor_squared)
        + jnp.sqrt(jnp.sum(base_grad_pressure * base_grad_pressure, axis=-1) + floor_squared)
        + float(force_floor)
    )
    from .strong_force import _RZL

    base_points = jnp.stack((rr.reshape(-1), tt.reshape(-1), zz.reshape(-1)), axis=-1)

    def base_tangent_norm(point):
        _, tangent = jax.jvp(
            lambda location: jnp.asarray(_RZL(native, location)[:2]),
            (point,),
            (jnp.asarray([0.0, 1.0, 0.0], dtype=point.dtype),),
        )
        return jnp.vdot(tangent, tangent).real

    gauge_length = jnp.sqrt(jnp.mean(jax.vmap(base_tangent_norm)(base_points)))
    provisional = StrongRootRuntime(
        native=native,
        transfer=transfer,
        low_preconditioner=low_preconditioner,
        layout=layout,
        coordinate_scale=jnp.ones((layout.size,)),
        equation_scale=jnp.ones((layout.size,)),
        radial_nodes=jnp.asarray(radial_nodes),
        theta=jnp.asarray(theta_grid),
        zeta=jnp.asarray(zeta_grid),
        cosine_projection=jnp.asarray(cosine_projection),
        sine_projection=jnp.asarray(sine_projection),
        radial_fit=jnp.asarray(radial_fit),
        normalization_denominator=normalization_denominator,
        gauge_length=gauge_length,
        strong_block_sign=jnp.ones((3,)),
        strong_scale=jnp.asarray(1.0),
        operator_balance=jnp.asarray(1.0),
        force_floor=float(force_floor),
    )
    # Stream exact global row/column 2-norms through one compiled forward JVP.
    # This captures cross-mode coupling with O(n) memory: no production-scale
    # dense Jacobian is retained, no direction is dropped, and no second
    # transpose program is compiled.  The positive row scale multiplies both
    # homotopy endpoints, leaving every root and branch fixed.
    base_vector = jnp.zeros(
        (layout.size,), dtype=jnp.asarray(native.R_cos).dtype
    )
    initial = _strong_residual_unscaled(base_vector, provisional)
    rms = jnp.linalg.norm(initial) / np.sqrt(float(layout.size))
    base_scale = jnp.maximum(rms, jnp.asarray(1.0e-12, dtype=rms.dtype))
    equation_scale, coordinate_scale = _streaming_ruiz_scales(
        lambda value: _strong_residual_unscaled(value, provisional),
        base_vector,
    )
    provisional = replace(
        provisional,
        coordinate_scale=jnp.asarray(coordinate_scale),
        equation_scale=jnp.asarray(equation_scale),
    )
    equilibrated_initial = _strong_residual_unscaled(base_vector, provisional)
    equilibrated_rms = jnp.linalg.norm(equilibrated_initial) / np.sqrt(
        float(layout.size)
    )
    scaled = replace(provisional, strong_scale=base_scale)
    zero = jnp.zeros((layout.size,), dtype=rms.dtype)

    @jax.jit
    def low_solve(value: Array) -> Array:
        high = layout.unpack(value / jnp.asarray(provisional.equation_scale))
        low = transfer.restrict(high)
        solved = low_preconditioner.solve_scaled(low)
        return layout.pack(transfer.prolong(solved)) / jnp.asarray(
            provisional.coordinate_scale
        )

    # A sign change cannot alter the strong root, but it changes the real
    # generalized spectrum of the low-to-strong pencil.  Select the three
    # block signs that maximize its leftmost Ritz value, thereby moving folds
    # caused only by equation orientation as close to alpha=1 as possible.
    from itertools import product

    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    component_runtimes = tuple(
        replace(
            scaled,
            strong_block_sign=jnp.eye(3, dtype=rms.dtype)[index],
        )
        for index in range(3)
    )

    @jax.jit
    def strong_components(value: Array) -> Array:
        return jnp.stack(tuple(
            jax.jvp(
                lambda vector: _strong_residual_unscaled(
                    vector, component_runtime
                ) / base_scale,
                (zero,),
                (value,),
            )[1]
            for component_runtime in component_runtimes
        ))

    strong_components(zero).block_until_ready()
    low_solve(zero).block_until_ready()
    eigenpairs = min(int(orientation_eigenpairs), max(1, layout.size - 2))
    dense_components = None
    if layout.size <= 64:
        dense_components = jax.jacfwd(strong_components)(zero)
    initial_arnoldi = np.linspace(-0.5, 0.7, layout.size, dtype=float)
    initial_arnoldi /= np.linalg.norm(initial_arnoldi)
    best_score = -np.inf
    best_block_sign = np.ones((3,), dtype=float)
    for signs in product((-1.0, 1.0), repeat=3):
        block_sign = jnp.asarray(signs, dtype=rms.dtype)

        def matvec(value: np.ndarray) -> np.ndarray:
            response = jnp.tensordot(
                block_sign,
                strong_components(jnp.asarray(value)),
                axes=1,
            )
            return np.asarray(jax.device_get(low_solve(response)))

        if dense_components is not None:
            oriented = jnp.tensordot(block_sign, dense_components, axes=1)
            matrix = jax.vmap(low_solve, in_axes=1, out_axes=1)(oriented)
            values = np.linalg.eigvals(np.asarray(jax.device_get(matrix)))
        else:
            operator = LinearOperator(
                (layout.size, layout.size), matvec=matvec, dtype=np.float64
            )
            try:
                values = eigs(
                    operator,
                    k=eigenpairs,
                    which="SR",
                    v0=initial_arnoldi,
                    maxiter=max(100, 2 * layout.size),
                    tol=1.0e-7,
                    return_eigenvectors=False,
                )
            except ArpackNoConvergence as error:
                values = error.eigenvalues
        if values.size:
            score = float(np.min(np.real(values)))
            if score > best_score:
                best_score = score
                best_block_sign = np.asarray(signs, dtype=float)
    scaled = replace(
        scaled,
        strong_block_sign=jnp.asarray(best_block_sign, dtype=rms.dtype),
    )

    def preconditioned_strong(value: Array) -> Array:
        _, response = jax.jvp(
            lambda vector: _strong_residual_unscaled(vector, scaled) / base_scale,
            (zero,),
            (value,),
        )
        return low_solve(response)

    direction = jnp.linspace(-0.5, 0.7, layout.size, dtype=rms.dtype)
    direction = direction / jnp.linalg.norm(direction)
    estimate = jnp.asarray(1.0, dtype=rms.dtype)
    for _ in range(int(balance_iterations)):
        response = preconditioned_strong(direction)
        response_norm = jnp.linalg.norm(response)
        estimate = jnp.maximum(estimate, response_norm)
        direction = response / jnp.maximum(
            response_norm, jnp.finfo(response_norm.dtype).tiny
        )
    effective_balance = base_scale * estimate / jnp.maximum(
        equilibrated_rms,
        jnp.finfo(equilibrated_rms.dtype).tiny,
    )
    return replace(
        scaled,
        strong_scale=base_scale * estimate,
        operator_balance=effective_balance,
    )


def strong_root_rank(
    runtime: StrongRootRuntime,
    vector: Array | None = None,
    *,
    relative_tolerance: float = 1.0e-9,
) -> tuple[int, Array]:
    """Assemble a small diagnostic Jacobian and return numerical rank/SVD."""

    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    point = jnp.zeros((runtime.layout.size,)) if vector is None else jnp.asarray(vector)
    jacobian = jax.jacfwd(lambda value: strong_root_residual(value, runtime))(point)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    threshold = float(relative_tolerance) * singular_values[0]
    return int(jnp.sum(singular_values > threshold)), singular_values


def build_strong_mode_block_preconditioner(
    runtime: StrongRootRuntime,
    vector: Array | None = None,
    *,
    poloidal_bandwidth: int = 3,
) -> StrongModeBlockPreconditioner:
    """Probe bounded same-mode blocks at one reusable linearization point."""

    if poloidal_bandwidth < 1:
        raise ValueError("poloidal_bandwidth must be positive")
    started = perf_counter()
    base = (
        jnp.zeros(
            (runtime.layout.size,),
            dtype=jnp.asarray(runtime.native.R_cos).dtype,
        )
        if vector is None
        else jnp.asarray(vector)
    )
    if base.shape != (runtime.layout.size,):
        raise ValueError(
            f"block linearization has shape {base.shape}; "
            f"expected {(runtime.layout.size,)}"
        )
    grouped: dict[tuple[int, int], list[int]] = {}
    for group in runtime.layout.groups:
        key = (
            int(group.abs_n),
            int(group.m) // int(poloidal_bandwidth),
        )
        grouped.setdefault(key, []).extend(range(group.start, group.stop))
    indices = tuple(
        jnp.asarray(grouped[key], dtype=jnp.int32)
        for key in sorted(grouped)
    )
    low_blocks: list[Array] = []
    strong_blocks: list[Array] = []
    for block_indices in indices:
        local_zero = jnp.zeros((block_indices.size,), dtype=base.dtype)

        def block_residual(local: Array, alpha: float) -> Array:
            candidate = base.at[block_indices].add(local)
            return strong_root_residual(candidate, runtime, alpha)[block_indices]

        low_blocks.append(
            jax.jacfwd(lambda local: block_residual(local, 0.0))(local_zero)
        )
        strong_blocks.append(
            jax.jacfwd(lambda local: block_residual(local, 1.0))(local_zero)
        )
    jax.block_until_ready((low_blocks, strong_blocks))
    return StrongModeBlockPreconditioner(
        indices,
        tuple(low_blocks),
        tuple(strong_blocks),
        perf_counter() - started,
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
    legacy_coordinates = project(legacy_state)
    raw_residual = implicit.residual_fn(
        config,
        jax.lax.stop_gradient(legacy_state),
        dof_mask,
        formulation="raw",
    )

    def legacy_residual(coordinates: SpectralState) -> SpectralState:
        return raw_residual(coordinates, params)

    legacy_defect = legacy_residual(legacy_coordinates)

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
    return LowOrderPreconditioner(
        transfer,
        system,
        legacy_coordinates,
        legacy_defect,
        legacy_residual,
        elapsed,
    )


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
    "StrongModeBlockPreconditioner",
    "StrongPhysicalChart",
    "StrongRootLayout",
    "StrongRootRuntime",
    "apply_high_order_correction",
    "build_low_order_preconditioner",
    "build_strong_mode_block_preconditioner",
    "make_high_low_transfer",
    "make_strong_physical_chart",
    "make_strong_root_layout",
    "make_strong_root_runtime",
    "preconditioner_quality",
    "preconditioner_refresh_decision",
    "strong_root_rank",
    "strong_physical_residual",
    "strong_root_residual",
    "strong_root_residual_at_native",
]
