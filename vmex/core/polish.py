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
class StrongRootLayout:
    """Independent R/Z/lambda coordinates for the square strong root.

    The R channel stores every active ``R_cos`` legacy degree of freedom.
    The Z channel stores active ``Z_sin`` entries, with each constrained 3D
    ``m=1,+/-n`` pair represented once as ``(z_+ + z_-)/sqrt(2)``.  Hence
    packing and unpacking are exact transposes and no coordinate duplicate is
    present in the square root system.  Lambda gauge/axis entries come from
    the existing evolved-DOF mask and are absent structurally.
    """

    ns: int
    mnmax: int
    r_indices: np.ndarray
    z_indices: np.ndarray
    z_weights: np.ndarray
    l_indices: np.ndarray

    @property
    def size(self) -> int:
        return int(self.r_indices.size + self.z_indices.shape[0] + self.l_indices.size)

    def pack(self, tangent: SpectralState) -> Array:
        """Project a legacy R/Z tangent onto independent reduced coordinates."""

        r = jnp.ravel(jnp.asarray(tangent.R_cos))[jnp.asarray(self.r_indices)]
        z_flat = jnp.ravel(jnp.asarray(tangent.Z_sin))
        z = jnp.sum(
            z_flat[jnp.asarray(self.z_indices)] * jnp.asarray(self.z_weights),
            axis=1,
        )
        lam = jnp.ravel(jnp.asarray(tangent.L_sin))[jnp.asarray(self.l_indices)]
        return jnp.concatenate((r, z, lam))

    def unpack(self, vector: Array) -> SpectralState:
        """Lift independent reduced coordinates to a projected legacy tangent."""

        vector = jnp.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError(f"free vector has shape {vector.shape}; expected {(self.size,)}")
        nr = int(self.r_indices.size)
        r = jnp.zeros((self.ns * self.mnmax,), dtype=vector.dtype)
        r = r.at[jnp.asarray(self.r_indices)].set(vector[:nr])
        z = jnp.zeros_like(r)
        nz = int(self.z_indices.shape[0])
        z_values = vector[nr : nr + nz, None] * jnp.asarray(
            self.z_weights, dtype=vector.dtype
        )
        z = z.at[jnp.asarray(self.z_indices).reshape(-1)].add(z_values.reshape(-1))
        lam = jnp.zeros_like(r)
        lam = lam.at[jnp.asarray(self.l_indices)].set(vector[nr + nz :])
        zero = jnp.zeros((self.ns, self.mnmax), dtype=vector.dtype)
        return SpectralState(
            R_cos=r.reshape((self.ns, self.mnmax)),
            R_sin=zero,
            Z_cos=zero,
            Z_sin=z.reshape((self.ns, self.mnmax)),
            L_cos=zero,
            L_sin=lam.reshape((self.ns, self.mnmax)),
        )


@dataclass(frozen=True, eq=False)
class StrongRootRuntime:
    """Reusable grids, transforms, constraints, and scaling for a square root."""

    native: HighOrderEquilibriumState
    transfer: HighLowTransfer
    low_preconditioner: LowOrderPreconditioner
    layout: StrongRootLayout
    radial_nodes: Array
    theta: Array
    zeta: Array
    cosine_projection: Array
    sine_projection: Array
    radial_fit: Array
    normalization_denominator: Array
    gauge_length: Array
    strong_row_sign: Array
    strong_scale: Array
    operator_balance: Array
    force_floor: float


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
    lconm1: bool = True,
) -> StrongRootLayout:
    """Eliminate inactive entries and constrained m=1 duplicate coordinates."""

    r_mask = np.asarray(dof_mask.R_cos, dtype=bool)
    z_mask = np.asarray(dof_mask.Z_sin, dtype=bool)
    l_mask = np.asarray(dof_mask.L_sin, dtype=bool)
    if (
        r_mask.shape != z_mask.shape
        or r_mask.shape != l_mask.shape
        or r_mask.shape[1] != np.asarray(native.m).size
    ):
        raise ValueError("dof mask and native mode layout must match")
    ns, mnmax = r_mask.shape
    r_indices = np.flatnonzero(r_mask.reshape(-1)).astype(np.int32)
    m = np.asarray(native.m, dtype=int)
    n = np.asarray(native.n, dtype=int)
    mode_index = {(int(mm), int(nn)): index for index, (mm, nn) in enumerate(zip(m, n))}
    groups: list[tuple[int, int]] = []
    weights: list[tuple[float, float]] = []
    paired: set[tuple[int, int]] = set()
    root_two = np.sqrt(2.0)
    for radial in range(ns):
        for mode in range(mnmax):
            if not z_mask[radial, mode] or (radial, mode) in paired:
                continue
            partner = mode_index.get((int(m[mode]), -int(n[mode])))
            if (
                lconm1
                and int(m[mode]) == 1
                and int(n[mode]) != 0
                and partner is not None
                and z_mask[radial, partner]
            ):
                first, second = sorted((mode, partner))
                groups.append((radial * mnmax + first, radial * mnmax + second))
                weights.append((1.0 / root_two, 1.0 / root_two))
                paired.add((radial, first))
                paired.add((radial, second))
            else:
                index = radial * mnmax + mode
                groups.append((index, index))
                weights.append((1.0, 0.0))
                paired.add((radial, mode))
    return StrongRootLayout(
        ns=ns,
        mnmax=mnmax,
        r_indices=r_indices,
        z_indices=np.asarray(groups, dtype=np.int32).reshape((-1, 2)),
        z_weights=np.asarray(weights, dtype=float).reshape((-1, 2)),
        l_indices=np.flatnonzero(l_mask.reshape(-1)).astype(np.int32),
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


def _strong_residual_unscaled(
    vector: Array,
    runtime: StrongRootRuntime,
    native: HighOrderEquilibriumState | None = None,
) -> Array:
    """Project normalized physical force onto the reduced solve space."""

    from .strong_force import _RZL, evaluate_strong_force

    native = runtime.native if native is None else native
    low_tangent = runtime.layout.unpack(vector)
    correction = runtime.transfer.prolong(low_tangent)
    state = apply_high_order_correction(native, correction)
    radial = jnp.asarray(runtime.radial_nodes)
    theta = jnp.asarray(runtime.theta)
    zeta = jnp.asarray(runtime.zeta)
    rr, tt, zz = jnp.meshgrid(radial, theta, zeta, indexing="ij")
    samples = evaluate_strong_force(state, rr, tt, zz)
    denominator = jnp.asarray(runtime.normalization_denominator)
    radial_force = 2.0 * samples.signed_radial_force_density / denominator
    helical_force = 2.0 * samples.signed_helical_force_density / denominator
    points = jnp.stack((rr.reshape(-1), tt.reshape(-1), zz.reshape(-1)), axis=-1)

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

    gauge = jax.vmap(coordinate_gauge)(points).reshape(rr.shape)
    radial_force = radial_force.reshape((radial.size, -1))
    helical_force = helical_force.reshape((radial.size, -1))
    gauge = gauge.reshape((radial.size, -1))
    radial_modes = jnp.einsum(
        "ra,ma->rm", radial_force, jnp.asarray(runtime.cosine_projection)
    )
    helical_modes = jnp.einsum(
        "ra,ma->rm", helical_force, jnp.asarray(runtime.sine_projection)
    )
    gauge_modes = jnp.einsum(
        "ra,ma->rm", gauge, jnp.asarray(runtime.sine_projection)
    )
    powers = radial[:, None] ** jnp.asarray(np.abs(runtime.native.m))[None, :]
    safe_powers = jnp.maximum(powers, jnp.finfo(radial.dtype).tiny)
    radial_q = radial_modes / safe_powers
    helical_q = helical_modes / safe_powers
    gauge_q = gauge_modes / safe_powers
    radial_coefficients = jnp.asarray(runtime.radial_fit) @ radial_q
    helical_coefficients = jnp.asarray(runtime.radial_fit) @ helical_q
    gauge_coefficients = jnp.asarray(runtime.radial_fit) @ gauge_q
    zero = jnp.zeros_like(jnp.asarray(runtime.native.R_cos))
    force_coefficients = HighOrderCorrection(
        R_cos=radial_coefficients.T,
        R_sin=zero,
        Z_cos=zero,
        Z_sin=gauge_coefficients.T,
        L_cos=zero,
        L_sin=helical_coefficients.T,
    )
    packed = runtime.layout.pack(runtime.transfer.restrict(force_coefficients))
    return packed * jnp.asarray(runtime.strong_row_sign)


@partial(jax.jit, static_argnames=("runtime",))
def strong_root_residual(
    vector: Array,
    runtime: StrongRootRuntime,
    alpha: Array = 1.0,
) -> Array:
    """Square residual homotopy from legacy raw force to strong force."""

    vector = jnp.asarray(vector)
    low_tangent = runtime.layout.unpack(vector)
    low = runtime.layout.pack(runtime.low_preconditioner.residual(low_tangent))
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
    radial_nodes = np.asarray(native.radial_basis.quadrature_nodes, dtype=float)
    radial_weights = np.asarray(native.radial_basis.quadrature_weights, dtype=float)
    radial_matrix = np.asarray(native.radial_basis.basis_matrix(radial_nodes), dtype=float)
    sqrt_weights = np.sqrt(radial_weights)
    weighted_matrix = sqrt_weights[:, None] * radial_matrix
    radial_fit = np.linalg.pinv(weighted_matrix, rcond=1.0e-12) * sqrt_weights[None, :]
    m = np.asarray(native.m, dtype=int)
    n = np.asarray(native.n, dtype=int)
    ntheta = max(2 * int(np.max(np.abs(m), initial=0)) + 3, 4)
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
        radial_nodes=jnp.asarray(radial_nodes),
        theta=jnp.asarray(theta_grid),
        zeta=jnp.asarray(zeta_grid),
        cosine_projection=jnp.asarray(cosine_projection),
        sine_projection=jnp.asarray(sine_projection),
        radial_fit=jnp.asarray(radial_fit),
        normalization_denominator=normalization_denominator,
        gauge_length=gauge_length,
        strong_row_sign=jnp.ones((layout.size,)),
        strong_scale=jnp.asarray(1.0),
        operator_balance=jnp.asarray(1.0),
        force_floor=float(force_floor),
    )
    initial = _strong_residual_unscaled(jnp.zeros((layout.size,)), provisional)
    rms = jnp.linalg.norm(initial) / np.sqrt(float(layout.size))
    base_scale = jnp.maximum(rms, jnp.asarray(1.0e-12, dtype=rms.dtype))
    scaled = replace(provisional, strong_scale=base_scale)
    zero = jnp.zeros((layout.size,), dtype=rms.dtype)

    block_edges = (
        0,
        int(layout.r_indices.size),
        int(layout.r_indices.size + layout.z_indices.shape[0]),
        layout.size,
    )

    @jax.jit
    def strong_linearized(value: Array) -> Array:
        _, response = jax.jvp(
            lambda vector: _strong_residual_unscaled(vector, scaled) / base_scale,
            (zero,),
            (value,),
        )
        return response

    @jax.jit
    def low_solve(value: Array) -> Array:
        tangent = layout.unpack(value)
        return layout.pack(low_preconditioner.solve_scaled(tangent))

    # A sign change cannot alter the strong root, but it changes the real
    # generalized spectrum of the low-to-strong pencil.  Select the three
    # block signs that maximize its leftmost Ritz value, thereby moving folds
    # caused only by equation orientation as close to alpha=1 as possible.
    from itertools import product

    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    strong_linearized(zero).block_until_ready()
    low_solve(zero).block_until_ready()
    eigenpairs = min(int(orientation_eigenpairs), max(1, layout.size - 2))
    dense_strong = None
    if layout.size <= 64:
        dense_strong = jax.jacfwd(strong_linearized)(zero)
    initial_arnoldi = np.linspace(-0.5, 0.7, layout.size, dtype=float)
    initial_arnoldi /= np.linalg.norm(initial_arnoldi)
    best_score = -np.inf
    best_row_sign = np.ones((layout.size,), dtype=float)
    for signs in product((-1.0, 1.0), repeat=3):
        row_sign = np.concatenate(
            [
                np.full((stop - start,), signs[index], dtype=float)
                for index, (start, stop) in enumerate(
                    zip(block_edges[:-1], block_edges[1:])
                )
            ]
        )

        def matvec(value: np.ndarray) -> np.ndarray:
            response = strong_linearized(jnp.asarray(value)) * jnp.asarray(row_sign)
            return np.asarray(jax.device_get(low_solve(response)))

        if dense_strong is not None:
            oriented = dense_strong * jnp.asarray(row_sign)[:, None]
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
                best_row_sign = row_sign
    strong_row_sign = jnp.asarray(best_row_sign, dtype=rms.dtype)
    scaled = replace(scaled, strong_row_sign=strong_row_sign)

    def preconditioned_strong(value: Array) -> Array:
        _, response = jax.jvp(
            lambda vector: _strong_residual_unscaled(vector, scaled) / base_scale,
            (zero,),
            (value,),
        )
        low_response = layout.unpack(response)
        return layout.pack(low_preconditioner.solve_scaled(low_response))

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
    return replace(
        scaled,
        strong_scale=base_scale * estimate,
        operator_balance=estimate,
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
    "StrongRootLayout",
    "StrongRootRuntime",
    "apply_high_order_correction",
    "build_low_order_preconditioner",
    "make_high_low_transfer",
    "make_strong_root_layout",
    "make_strong_root_runtime",
    "preconditioner_quality",
    "preconditioner_refresh_decision",
    "strong_root_rank",
    "strong_root_residual",
    "strong_root_residual_at_native",
]
