"""Projected free-boundary tangents and state adjoints.

This module linearizes the fully rebuilt VMEX--NESTOR residual at a converged
single-stage free-boundary equilibrium.  It provides the lower-level B2--B4
operations: the exact evolved-coordinate projector, scalar parameter tangents,
state pullbacks, scalar state-objective adjoints, and one-current convenience
wrappers.

The adaptive host solve remains outside differentiation.  All derivatives are
taken with respect to the converged projected residual equation.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from solvax import gmres as _solvax_gmres

from . import implicit as _fixed_implicit
from .freeboundary import FreeBoundaryResidualEvaluator
from .solver import SpectralState
from .transforms import register_pytree_dataclass as _register

__all__ = [
    "FreeBoundaryAdjointResult",
    "FreeBoundaryStatePullbackResult",
    "FreeBoundaryTangentConfig",
    "FreeBoundaryTangentResult",
    "free_boundary_dof_mask",
    "make_projected_free_boundary_residual",
    "one_current_adjoint",
    "one_current_tangent",
    "scalar_parameter_state_pullback",
    "scalar_parameter_tangent",
    "scalar_state_objective_adjoint",
]

Array = Any
_STATE_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
_RESOLUTION_CACHE_FIELDS = (
    "mpol",
    "ntor",
    "ntheta",
    "nzeta",
    "nfp",
    "lasym",
    "ns",
)

def _resolution_cache_key(resolution: Any) -> tuple[Any, ...]:
    """Return hashable primitive metadata for production and test runtimes."""
    return tuple(
        getattr(resolution, name, None)
        for name in _RESOLUTION_CACHE_FIELDS
    )


@dataclass(frozen=True)
class FreeBoundaryTangentConfig:
    """Static controls shared by forward tangents and state adjoints."""

    rtol: float = 1.0e-8
    atol: float = 0.0
    restart: int = 30
    max_restarts: int = 50
    base_residual_atol: float = 1.0e-5
    adjoint_backend: str = "forward_dense"
    adjoint_batch_size: int = 4

@dataclass(frozen=True)
class FreeBoundaryTangentResult:
    """One branch-local free-boundary equilibrium sensitivity.

    ``state_tangent`` is with respect to the scalar ``alpha`` supplied to
    :func:`scalar_parameter_tangent`.
    """

    state_tangent: SpectralState
    dof_mask: SpectralState
    base_residual_norm: Array
    rhs_norm: Array
    linear_residual_norm: Array
    relative_linear_residual: Array
    iterations: Array
    converged: Array
    krylov_converged: Array


_register(FreeBoundaryTangentResult)

@dataclass(frozen=True)
class FreeBoundaryStatePullbackResult:
    """Pullback of one state cotangent through a scalar-parameter root.

    ``parameter_cotangent`` is the implicit contribution

    ``-lambda.T @ F_alpha``

    for the supplied state cotangent.  ``adjoint_residual_norm`` is measured
    by a fresh application of the transposed, already-linearized state map
    after the linear solve returns; it is not copied from a solver estimate.
    """

    parameter_cotangent: Array
    adjoint: SpectralState
    dof_mask: SpectralState
    base_residual_norm: Array
    state_cotangent_norm: Array
    parameter_residual_derivative_norm: Array
    adjoint_residual_norm: Array
    relative_adjoint_residual: Array
    active_dimension: Array
    iterations: Array
    converged: Array
    linear_solver_converged: Array
    backend: str


_register(FreeBoundaryStatePullbackResult, meta=("backend",))

@dataclass(frozen=True)
class FreeBoundaryAdjointResult:
    """One scalar state objective and its implicit parameter derivative."""

    objective_value: Array
    derivative: Array
    state_pullback: FreeBoundaryStatePullbackResult


_register(FreeBoundaryAdjointResult)


def _tree_norm(tree) -> Array:
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x).real for x in leaves))


def _tree_dot(left, right) -> Array:
    return sum(jnp.vdot(a, b).real for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))


def _require_finite_residual_diagnostics(
    diagnostics: dict[str, Any],
    *,
    context: str,
) -> dict[str, float]:
    """Return scalar residual diagnostics or reject any NaN/Inf explicitly."""
    values = {name: float(value) for name, value in diagnostics.items()}
    invalid = [name for name, value in values.items() if not np.isfinite(value)]
    if invalid:
        details = ", ".join(f"{name}={values[name]!r}" for name in invalid)
        raise ValueError(f"{context} has non-finite residual diagnostic(s): {details}")
    return values


def _coerce_scalar_alpha0(alpha0: float | Array, state: SpectralState) -> Array:
    """Return one concrete finite scalar parameter in the state's dtype."""
    alpha = jnp.asarray(alpha0, dtype=jnp.asarray(state.R_cos).dtype)
    if alpha.ndim != 0:
        raise ValueError(f"alpha0 must be scalar, got shape {alpha.shape}")
    scalar = float(np.asarray(alpha))
    if not np.isfinite(scalar):
        raise ValueError(f"alpha0 must be finite, got {scalar!r}")
    return alpha

def _validated_config(
    config: FreeBoundaryTangentConfig | None,
) -> FreeBoundaryTangentConfig:
    """Validate the shared tangent/adjoint Krylov and root tolerances."""
    cfg = FreeBoundaryTangentConfig() if config is None else config
    if not np.isfinite(float(cfg.rtol)) or cfg.rtol <= 0.0:
        raise ValueError("rtol must be finite and > 0")
    if not np.isfinite(float(cfg.atol)) or cfg.atol < 0.0:
        raise ValueError("atol must be finite and >= 0")
    if cfg.restart < 1 or cfg.max_restarts < 1:
        raise ValueError("restart and max_restarts must be >= 1")
    if not np.isfinite(float(cfg.base_residual_atol)) or cfg.base_residual_atol <= 0.0:
        raise ValueError("base_residual_atol must be finite and > 0")
    if cfg.adjoint_backend not in {
        "forward_dense",
        "forward_dense_jax",
        "reverse_gmres",
    }:
        raise ValueError(
            "adjoint_backend must be 'forward_dense', 'forward_dense_jax', "
            "or 'reverse_gmres'"
        )
    if cfg.adjoint_batch_size < 1:
        raise ValueError("adjoint_batch_size must be >= 1")
    return cfg


def _projector_spec(evaluator: FreeBoundaryResidualEvaluator):
    rt = evaluator.runtime
    return types.SimpleNamespace(
        resolution=rt.resolution,
        lconm1=bool(rt.setup.lconm1),
    )


def _projector(
    evaluator: FreeBoundaryResidualEvaluator,
    dof_mask: SpectralState,
) -> Callable[[SpectralState], SpectralState]:
    # The m=1 constrained-pair projector is identical in fixed- and
    # free-boundary VMEC.  Reuse the parity-tested implementation; only the
    # elementwise mask differs because free boundary evolves the R/Z edge.
    return _fixed_implicit._dof_projector(_projector_spec(evaluator), dof_mask)


def free_boundary_dof_mask(
    evaluator: FreeBoundaryResidualEvaluator,
) -> SpectralState:
    """Return the exact stellarator-symmetric free-boundary DOF mask.

    VMEC's signed spectral packing makes the structural support explicit:

    - ``R_cos`` is active everywhere except the ``m > 0`` axis entries;
    - ``Z_sin`` is additionally zero for the identically vanishing ``(0, 0)``
      sine harmonic;
    - ``L_sin`` has no axis degrees of freedom and no ``(0, 0)`` harmonic;
    - ``R_sin``, ``Z_cos`` and ``L_cos`` vanish under stellarator symmetry.

    In particular, no edge entries are removed.  A one-sample structural
    row/column audit against the CTH free-boundary force gives exactly these
    counts (579, 564, 560 for ``R_cos/Z_sin/L_sin`` at ``ns=15, mnmax=41``).
    Released m=1 constrained pair combinations are non-elementwise and are
    handled by the symmetric projector, not by deleting either pair column.
    """
    rt = evaluator.runtime
    res = rt.resolution
    if bool(res.lasym):
        raise NotImplementedError("free_boundary_dof_mask currently supports stellarator symmetry only")

    m = np.asarray(rt.modes.m, dtype=int)
    n = np.asarray(rt.modes.n, dtype=int)
    shape = (int(res.ns), int(m.size))
    zero = np.zeros(shape, dtype=np.float64)
    r_cos = np.ones(shape, dtype=np.float64)
    z_sin = np.ones(shape, dtype=np.float64)
    l_sin = np.ones(shape, dtype=np.float64)

    m0n0 = (m == 0) & (n == 0)
    r_cos[0, m > 0] = 0.0
    z_sin[:, m0n0] = 0.0
    z_sin[0, m > 0] = 0.0
    l_sin[:, m0n0] = 0.0
    l_sin[0, :] = 0.0

    return SpectralState(
        R_cos=jnp.asarray(r_cos),
        R_sin=jnp.asarray(zero),
        Z_cos=jnp.asarray(zero),
        Z_sin=jnp.asarray(z_sin),
        L_cos=jnp.asarray(zero),
        L_sin=jnp.asarray(l_sin),
    )


def _validate_dof_mask(
    evaluator: FreeBoundaryResidualEvaluator,
    state: SpectralState,
    dof_mask: SpectralState,
) -> None:
    """Reject masks that invalidate the square projected linearization."""
    if jax.tree.structure(dof_mask) != jax.tree.structure(state):
        raise ValueError("dof_mask must have the SpectralState tree structure")
    expected = free_boundary_dof_mask(evaluator)
    for name in _STATE_FIELDS:
        value = np.asarray(getattr(dof_mask, name))
        reference = np.asarray(getattr(expected, name))
        if value.shape != np.shape(getattr(state, name)):
            raise ValueError(
                f"dof_mask.{name} shape {value.shape} does not match state shape {np.shape(getattr(state, name))}"
            )
        if not np.all((value == 0.0) | (value == 1.0)):
            raise ValueError(f"dof_mask.{name} must be binary")
        if not np.array_equal(value, reference):
            raise ValueError(f"dof_mask.{name} does not match the free-boundary structural mask")

    # This is redundant with the exact-mask comparison but documents and
    # guards the non-elementwise condition required for P(P(x)) == P(x).
    if bool(evaluator.runtime.setup.lconm1) and int(evaluator.runtime.resolution.ntor) > 0:
        pos, neg = _fixed_implicit._m1_pair_columns(_projector_spec(evaluator))
        zmask = np.asarray(dof_mask.Z_sin)
        if not np.array_equal(zmask[:, pos], zmask[:, neg]):
            raise ValueError("dof_mask m=1 pair columns must match")


@dataclass(frozen=True)
class _ProjectedResidualExecutable:
    """One resolution-static projected residual and its exact projector."""

    evaluate: Any
    projector: Callable[[SpectralState], SpectralState]
    run: Any


# Production evaluators created by ``make_free_boundary_residual_evaluator``
# share their heavy residual executable through the fused-vacuum cache.  Keep
# the light projected wrapper on that same identity boundary.  Values retain
# the callable as an identity guard, so a recycled Python ``id`` cannot alias
# an obsolete executable.
_PROJECTED_RESIDUAL_EXECUTABLE_CACHE: dict[
    tuple[Any, ...],
    _ProjectedResidualExecutable,
] = {}


def _projected_residual_executable(
    evaluator: FreeBoundaryResidualEvaluator,
    dof_mask: SpectralState,
) -> _ProjectedResidualExecutable | None:
    """Return a reusable projected residual when the evaluator exposes one.

    Synthetic/duck-typed evaluators used by diagnostics may implement only
    ``__call__``.  They retain the historical per-instance path; VMEX
    production evaluators expose ``_evaluate(state, runtime, field)`` and can
    keep every accepted-root array as a traced argument.
    """
    evaluate = getattr(evaluator, "_evaluate", None)
    if evaluate is None or not bool(
        getattr(evaluator, "_runtime_argument_reusable", False)
    ):
        return None
    runtime = evaluator.runtime
    mask_signature = tuple(
        (
            tuple(np.shape(getattr(dof_mask, name))),
            np.dtype(np.asarray(getattr(dof_mask, name)).dtype).str,
        )
        for name in _STATE_FIELDS
    )
    key = (
        id(evaluate),
        _resolution_cache_key(runtime.resolution),
        bool(runtime.setup.lconm1),
        mask_signature,
    )
    cached = _PROJECTED_RESIDUAL_EXECUTABLE_CACHE.get(key)
    if cached is not None and cached.evaluate is evaluate:
        return cached

    P = _projector(evaluator, dof_mask)

    def run(
        z: SpectralState,
        frozen: SpectralState,
        dynamic_runtime: Any,
        external_field: Any,
    ) -> SpectralState:
        frozen = jax.lax.stop_gradient(frozen)
        projected_delta = P(jax.tree.map(lambda a, b: a - b, z, frozen))
        physical = jax.tree.map(lambda a, b: a + b, frozen, projected_delta)
        return P(evaluate(physical, dynamic_runtime, external_field).residual)

    executable = _ProjectedResidualExecutable(
        evaluate=evaluate,
        projector=P,
        run=jax.jit(run),
    )
    _PROJECTED_RESIDUAL_EXECUTABLE_CACHE[key] = executable
    return executable


def make_projected_free_boundary_residual(
    evaluator: FreeBoundaryResidualEvaluator,
    frozen_state: SpectralState,
    dof_mask: SpectralState,
) -> tuple[
    Callable[[SpectralState, Any], SpectralState],
    SpectralState,
    Callable[[SpectralState], SpectralState],
]:
    """Return ``(F, z_star, P)`` for the evolved free-boundary subspace.

    ``F(z, field)`` reconstructs the physical state as

    ``x = frozen_state + P(z - frozen_state)``

    and returns ``P(coupled_residual(x, field))``.  At
    ``z_star = P(frozen_state)`` this is exactly the supplied equilibrium.
    The R/Z edge rows participate in both ``z`` and ``F`` whenever supported
    by the structural mask.
    """
    _validate_dof_mask(evaluator, frozen_state, dof_mask)
    frozen = jax.lax.stop_gradient(frozen_state)
    executable = _projected_residual_executable(evaluator, dof_mask)
    P = (
        executable.projector
        if executable is not None
        else _projector(evaluator, dof_mask)
    )
    z_star = P(frozen_state)

    if executable is not None:
        def residual(z: SpectralState, external_field: Any) -> SpectralState:
            return executable.run(
                z,
                frozen,
                evaluator.runtime,
                external_field,
            )
    else:
        def assemble(z: SpectralState) -> SpectralState:
            projected_delta = P(jax.tree.map(lambda a, b: a - b, z, frozen))
            return jax.tree.map(lambda a, b: a + b, frozen, projected_delta)

        def residual(z: SpectralState, external_field: Any) -> SpectralState:
            return P(evaluator(assemble(z), external_field).residual)

        residual = jax.jit(residual)

    return residual, z_star, P


def _linearized_projected_root(
    evaluator: FreeBoundaryResidualEvaluator,
    state: SpectralState,
    external_field: Any,
    dof_mask: SpectralState | None,
    config: FreeBoundaryTangentConfig | None,
):
    """Build and validate the one linearization shared by B3 and B4."""
    cfg = _validated_config(config)
    if dof_mask is None:
        dof_mask = free_boundary_dof_mask(evaluator)
    F, z_star, P = make_projected_free_boundary_residual(evaluator, state, dof_mask)
    base_residual, state_matvec = jax.linearize(
        lambda z: F(z, external_field),
        z_star,
    )
    base_norm = _tree_norm(base_residual)
    base_norm_value = _require_finite_residual_diagnostics(
        {"||F||": base_norm},
        context="free-boundary implicit sensitivity root",
    )["||F||"]
    if base_norm_value > float(cfg.base_residual_atol):
        raise ValueError(
            "free-boundary implicit sensitivity requires a converged residual "
            f"root: ||F||={base_norm_value:.3e} > "
            f"base_residual_atol={float(cfg.base_residual_atol):.3e}"
        )
    return cfg, dof_mask, F, z_star, P, base_norm, state_matvec


@dataclass(frozen=True)
class _ActiveSubspace:
    """Sparse orthonormal basis ``Q`` for the exact range of ``P``."""

    full_size: int
    single_indices: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray
    unravel: Callable[[Array], SpectralState]
    dtype: Any

    @property
    def size(self) -> int:
        return int(self.single_indices.size + self.pair_left.size)


_ACTIVE_SUBSPACE_CACHE: dict[tuple[Any, ...], _ActiveSubspace] = {}


def _active_subspace(
    evaluator: FreeBoundaryResidualEvaluator,
    dof_mask: SpectralState,
) -> _ActiveSubspace:
    """Construct ``Q`` without materializing its sparse full matrix.

    Elementwise active entries contribute unit columns.  Each released m=1
    Z-sine pair contributes one ``(e_pos + e_neg) / sqrt(2)`` column, matching
    the symmetric pair projector exactly and removing its duplicate null
    coordinate from the dense fallback.
    """
    flat_mask_jax, unravel = ravel_pytree(dof_mask)
    flat_mask = np.asarray(flat_mask_jax)
    rt = evaluator.runtime
    key = (
        _resolution_cache_key(rt.resolution),
        bool(rt.setup.lconm1),
        tuple(np.shape(getattr(dof_mask, name)) for name in _STATE_FIELDS),
        np.dtype(flat_mask.dtype).str,
        flat_mask.tobytes(),
    )
    cached = _ACTIVE_SUBSPACE_CACHE.get(key)
    if cached is not None:
        return cached
    active = flat_mask == 1.0

    pair_left: list[int] = []
    pair_right: list[int] = []
    if bool(rt.setup.lconm1) and int(rt.resolution.ntor) > 0:
        pos, neg = _fixed_implicit._m1_pair_columns(_projector_spec(evaluator))
        z_sin_offset = sum(
            int(np.size(getattr(dof_mask, name))) for name in _STATE_FIELDS[: _STATE_FIELDS.index("Z_sin")]
        )
        mnmax = int(np.shape(dof_mask.Z_sin)[1])
        for radial_index in range(int(rt.resolution.ns)):
            for pos_column, neg_column in zip(pos, neg):
                left = z_sin_offset + radial_index * mnmax + int(pos_column)
                right = z_sin_offset + radial_index * mnmax + int(neg_column)
                if active[left] and active[right]:
                    active[left] = False
                    active[right] = False
                    pair_left.append(left)
                    pair_right.append(right)

    space = _ActiveSubspace(
        full_size=int(flat_mask.size),
        single_indices=np.flatnonzero(active),
        pair_left=np.asarray(pair_left, dtype=np.int32),
        pair_right=np.asarray(pair_right, dtype=np.int32),
        unravel=unravel,
        dtype=flat_mask_jax.dtype,
    )
    _ACTIVE_SUBSPACE_CACHE[key] = space
    return space


def _expand_active(space: _ActiveSubspace, vector: Array) -> SpectralState:
    """Apply the sparse orthonormal basis ``Q`` to active coordinates."""
    value = jnp.asarray(vector, dtype=space.dtype)
    if value.ndim != 1 or int(value.shape[0]) != space.size:
        raise ValueError(f"active vector must have shape ({space.size},), got {value.shape}")
    n_single = int(space.single_indices.size)
    full = jnp.zeros((space.full_size,), dtype=value.dtype)
    full = full.at[space.single_indices].set(value[:n_single])
    if space.pair_left.size:
        paired = value[n_single:] / jnp.sqrt(jnp.asarray(2.0, value.dtype))
        full = full.at[space.pair_left].set(paired)
        full = full.at[space.pair_right].set(paired)
    return space.unravel(full)


def _compress_active(space: _ActiveSubspace, tree: SpectralState) -> Array:
    """Apply ``Q.T`` to a full spectral tree."""
    full = ravel_pytree(tree)[0]
    single = full[space.single_indices]
    if not space.pair_left.size:
        return single
    paired = (full[space.pair_left] + full[space.pair_right]) / jnp.sqrt(jnp.asarray(2.0, full.dtype))
    return jnp.concatenate((single, paired))


def _assemble_forward_active_jacobian(
    state_matvec: Callable[[SpectralState], SpectralState],
    space: _ActiveSubspace,
    *,
    batch_size: int,
) -> np.ndarray:
    """Assemble ``Q.T @ F_state @ Q`` using only forward JVP matvecs."""
    size = space.size

    def active_matvec(vector):
        return _compress_active(space, state_matvec(_expand_active(space, vector)))

    # Keep the batch static and pad the final block, avoiding a second large
    # compilation for a short tail.  The outer JIT packages the validated B3
    # JVP into one reusable executable; no reverse-mode residual is compiled.
    batched_matvec = jax.jit(jax.vmap(active_matvec))
    matrix = np.empty((size, size), dtype=np.dtype(space.dtype))
    for start in range(0, size, batch_size):
        count = min(batch_size, size - start)
        basis = np.zeros((batch_size, size), dtype=np.dtype(space.dtype))
        rows = np.arange(count)
        basis[rows, start + rows] = 1.0
        image = np.asarray(batched_matvec(jnp.asarray(basis)))[:count]
        matrix[:, start : start + count] = image.T
    return matrix


def _assemble_forward_active_jacobian_jax(
    state_matvec: Callable[[SpectralState], SpectralState],
    space: _ActiveSubspace,
    *,
    batch_size: int,
) -> Array:
    """Assemble ``Q.T @ F_state @ Q`` in shape-static device chunks.

    ``lax.map`` creates only one ``batch_size x active_size`` one-hot block at
    a time.  Its stacked output is the padded matrix itself, so this does not
    materialize a second ``active_size x active_size`` identity on the device.
    """
    if np.dtype(space.dtype) != np.dtype(np.float64):
        raise TypeError(
            "forward_dense_jax requires float64 active coordinates; "
            f"got {space.dtype}. Enable JAX x64 before state creation."
        )
    size = space.size
    if size < 1:
        raise ValueError("forward_dense_jax active dimension must be nonzero")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size,
        (int, np.integer),
    ):
        raise TypeError("forward_dense_jax batch_size must be an integer >= 1")
    if batch_size < 1:
        raise ValueError("forward_dense_jax batch_size must be >= 1")
    effective_batch_size = min(int(batch_size), size)
    chunk_count = (size + effective_batch_size - 1) // effective_batch_size
    padded_size = chunk_count * effective_batch_size

    def active_matvec(vector):
        return _compress_active(space, state_matvec(_expand_active(space, vector)))

    batched_matvec = jax.vmap(active_matvec)

    def assemble_chunks(chunk_indices):
        row_offsets = jnp.arange(effective_batch_size, dtype=jnp.int32)

        def assemble_chunk(chunk_index):
            column_indices = chunk_index * effective_batch_size + row_offsets
            basis = jax.nn.one_hot(column_indices, size, dtype=space.dtype)
            return batched_matvec(basis)

        return jax.lax.map(assemble_chunk, chunk_indices)

    chunk_indices = jnp.arange(chunk_count, dtype=jnp.int32)
    chunk_images = jax.jit(assemble_chunks)(chunk_indices)
    row_images = jnp.reshape(chunk_images, (padded_size, size))[:size]
    return jnp.swapaxes(row_images, 0, 1)


@functools.partial(jax.jit, static_argnames=("transpose",))
def _jax_dense_solve_impl(
    matrix: Array,
    rhs: Array,
    *,
    transpose: bool,
) -> tuple[Array, Array, Array]:
    """Device-native solve and diagnostics for one right-hand side."""
    operator = jnp.swapaxes(matrix, 0, 1) if transpose else matrix
    solution = jnp.linalg.solve(operator, rhs)
    residual = operator @ solution - rhs
    matrix_finite = jnp.all(jnp.isfinite(matrix))
    solution_finite = (
        matrix_finite
        & jnp.all(jnp.isfinite(rhs))
        & jnp.all(jnp.isfinite(solution))
        & jnp.all(jnp.isfinite(residual))
    )
    return solution, residual, solution_finite


def _jax_dense_solve(
    matrix: Array,
    rhs: Array,
    *,
    transpose: bool = False,
) -> tuple[Array, Array, Array]:
    """Validate static metadata and solve float64 on the selected JAX device.

    The returned tuple contains the solution, fresh residual, and one
    numeric-finite flag. Shape and dtype errors raise from static metadata.
    Numeric failures remain a device flag for the caller's compact
    convergence gate; no dense input or output is materialized on the host.
    """
    if type(transpose) is not bool:
        raise TypeError("transpose must be a bool")
    matrix = jnp.asarray(matrix)
    rhs = jnp.asarray(rhs)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
    ):
        raise ValueError(f"matrix must be nonempty and square, got shape {matrix.shape}")
    if rhs.ndim != 1 or rhs.shape[0] != matrix.shape[0]:
        raise ValueError(
            "rhs must have shape (n,) matching the matrix, "
            f"got matrix {matrix.shape} and rhs {rhs.shape}"
        )
    required_dtype = np.dtype(np.float64)
    if (
        np.dtype(matrix.dtype) != required_dtype
        or np.dtype(rhs.dtype) != required_dtype
    ):
        raise TypeError(
            "forward_dense_jax requires a float64 matrix and right-hand side; "
            f"got {matrix.dtype} and {rhs.dtype}. Enable JAX x64 before array creation."
        )
    return _jax_dense_solve_impl(matrix, rhs, transpose=transpose)

def scalar_parameter_tangent(
    evaluator: FreeBoundaryResidualEvaluator,
    state: SpectralState,
    field_from_alpha: Callable[[Array], Any],
    *,
    alpha0: float | Array = 0.0,
    dof_mask: SpectralState | None = None,
    config: FreeBoundaryTangentConfig | None = None,
) -> FreeBoundaryTangentResult:
    """Solve one branch-local scalar external-field equilibrium tangent.

    ``field_from_alpha(alpha)`` must be a JAX-traceable constructor returning
    any registered external-field pytree accepted by the residual evaluator.
    The returned tangent solves

    ``F_state @ dstate/dalpha = -F_alpha``

    at the finite scalar ``alpha0``.  This generic entry point supports mgrid
    currents, direct coil-current pytrees, and later coil-shape
    parameterizations without coupling this module to any one field class.
    """
    alpha = _coerce_scalar_alpha0(alpha0, state)
    external_field = field_from_alpha(alpha)
    (
        cfg,
        dof_mask,
        F,
        z_star,
        P,
        base_norm,
        state_matvec,
    ) = _linearized_projected_root(evaluator, state, external_field, dof_mask, config)

    _, parameter_derivative = jax.jvp(
        lambda value: F(z_star, field_from_alpha(value)),
        (alpha,),
        (jnp.ones_like(alpha),),
    )
    rhs = jax.tree.map(jnp.negative, parameter_derivative)
    rhs_flat, unravel = ravel_pytree(rhs)

    def matvec_flat(vector):
        return ravel_pytree(state_matvec(unravel(vector)))[0]

    solution = _solvax_gmres(
        matvec_flat,
        rhs_flat,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        restart=int(cfg.restart),
        max_restarts=int(cfg.max_restarts),
    )
    tangent = P(unravel(solution.x))
    linear_residual = jax.tree.map(
        lambda lhs, right: lhs - right,
        state_matvec(tangent),
        rhs,
    )
    rhs_norm = _tree_norm(rhs)
    linear_norm = _tree_norm(linear_residual)
    relative = linear_norm / jnp.maximum(rhs_norm, jnp.asarray(1.0e-30, dtype=rhs_norm.dtype))
    tolerance = jnp.maximum(
        jnp.asarray(cfg.atol, dtype=rhs_norm.dtype),
        jnp.asarray(cfg.rtol, dtype=rhs_norm.dtype) * rhs_norm,
    )
    converged = linear_norm <= tolerance
    return FreeBoundaryTangentResult(
        state_tangent=tangent,
        dof_mask=dof_mask,
        base_residual_norm=base_norm,
        rhs_norm=rhs_norm,
        linear_residual_norm=linear_norm,
        relative_linear_residual=relative,
        iterations=solution.iterations,
        converged=converged,
        krylov_converged=solution.converged,
    )

def _coerce_state_cotangent(
    state_cotangent: SpectralState,
    state: SpectralState,
) -> SpectralState:
    """Validate a concrete state cotangent and cast it to the state dtypes."""
    if jax.tree.structure(state_cotangent) != jax.tree.structure(state):
        raise ValueError("state_cotangent must have the SpectralState tree structure")
    coerced = []
    for name in _STATE_FIELDS:
        source = getattr(state_cotangent, name)
        state_leaf = jnp.asarray(getattr(state, name))
        source_host = np.asarray(source)
        shape = source_host.shape
        expected = state_leaf.shape
        if shape != expected:
            raise ValueError(f"state_cotangent.{name} shape {shape} does not match state shape {expected}")
        if np.iscomplexobj(source_host) and not np.issubdtype(state_leaf.dtype, np.complexfloating):
            raise ValueError(f"state_cotangent.{name} must be real-valued")
        if not np.all(np.isfinite(source_host)):
            raise ValueError(f"state_cotangent.{name} must be finite")
        leaf = jnp.asarray(source_host, dtype=state_leaf.dtype)
        if not np.all(np.isfinite(leaf)):
            raise ValueError(f"state_cotangent.{name} must remain finite in the state dtype")
        coerced.append(leaf)
    return SpectralState(*coerced)


@dataclass(frozen=True)
class _ProjectedStateAdjoint:
    """One solved projected state adjoint, before parameter contraction."""

    config: FreeBoundaryTangentConfig
    dof_mask: SpectralState
    residual: Callable[[SpectralState, Any], SpectralState]
    z_star: SpectralState
    adjoint: SpectralState
    base_residual_norm: Array
    state_cotangent_norm: Array
    adjoint_residual_norm: Array
    relative_adjoint_residual: Array
    active_dimension: Array
    iterations: Array
    converged: Array
    linear_solver_converged: Array


def _solve_projected_state_adjoint(
    evaluator: FreeBoundaryResidualEvaluator,
    state: SpectralState,
    external_field: Any,
    state_cotangent: SpectralState,
    dof_mask: SpectralState | None,
    config: FreeBoundaryTangentConfig | None,
) -> _ProjectedStateAdjoint:
    """Solve one projected state-adjoint system."""
    state_cotangent = _coerce_state_cotangent(state_cotangent, state)
    (
        cfg,
        dof_mask,
        F,
        z_star,
        P,
        base_norm,
        state_matvec,
    ) = _linearized_projected_root(evaluator, state, external_field, dof_mask, config)

    # x(z) = frozen + P(z - frozen), so the objective cotangent in the
    # projected coordinate is P @ J_state.  P is symmetric and idempotent.
    rhs = P(state_cotangent)
    rhs_norm = _tree_norm(rhs)

    if cfg.adjoint_backend == "reverse_gmres":
        rhs_flat, unravel = ravel_pytree(rhs)

        # Transpose only the already-linearized z map.  In particular this is
        # not jax.vjp(F), which would retrace/relinearize nonlinear NESTOR.
        state_transpose = jax.linear_transpose(state_matvec, z_star)

        def transpose_matvec_tree(cotangent):
            return P(state_transpose(P(cotangent))[0])

        def transpose_matvec_flat(vector):
            return ravel_pytree(transpose_matvec_tree(unravel(vector)))[0]

        solution = _solvax_gmres(
            transpose_matvec_flat,
            rhs_flat,
            rtol=float(cfg.rtol),
            atol=float(cfg.atol),
            restart=int(cfg.restart),
            max_restarts=int(cfg.max_restarts),
        )
        adjoint = P(unravel(solution.x))
        adjoint_residual = jax.tree.map(
            lambda lhs, right: lhs - right,
            transpose_matvec_tree(adjoint),
            rhs,
        )
        adjoint_norm = _tree_norm(adjoint_residual)
        iterations = solution.iterations
        linear_solver_converged = solution.converged
        active_dimension = _active_subspace(evaluator, dof_mask).size
    elif cfg.adjoint_backend == "forward_dense":
        # Reverse compilation of the monolithic NESTOR residual is currently
        # prohibitively expensive on CPU.  Assemble the exact active Jacobian
        # from the already-validated B3 forward JVP, then transpose that
        # nonsymmetric matrix explicitly.  Q removes all structural/gauge and
        # duplicate m=1 pair coordinates before the dense solve.
        import scipy.linalg

        space = _active_subspace(evaluator, dof_mask)
        active_matrix = _assemble_forward_active_jacobian(
            state_matvec,
            space,
            batch_size=int(cfg.adjoint_batch_size),
        )
        rhs_active = np.asarray(_compress_active(space, rhs))
        adjoint_active = scipy.linalg.solve(
            active_matrix.T,
            rhs_active,
            assume_a="gen",
            check_finite=True,
        )
        active_residual = active_matrix.T @ adjoint_active - rhs_active
        adjoint = _expand_active(space, jnp.asarray(adjoint_active))
        adjoint_residual = _expand_active(space, jnp.asarray(active_residual))
        adjoint_norm = _tree_norm(adjoint_residual)
        iterations = jnp.asarray(1, dtype=jnp.int32)
        linear_solver_converged = jnp.asarray(
            np.all(np.isfinite(adjoint_active)),
            dtype=jnp.bool_,
        )
        active_dimension = space.size
    else:
        # The same explicit nonsymmetric transpose solve as forward_dense,
        # with the active matrix, RHS, solution, and residual kept on the
        # selected JAX device.  The helper enforces float64 before dispatch.
        space = _active_subspace(evaluator, dof_mask)
        active_matrix = _assemble_forward_active_jacobian_jax(
            state_matvec,
            space,
            batch_size=int(cfg.adjoint_batch_size),
        )
        rhs_active = _compress_active(space, rhs)
        (
            adjoint_active,
            active_residual,
            linear_solver_converged,
        ) = _jax_dense_solve(active_matrix, rhs_active, transpose=True)
        del active_matrix
        adjoint = _expand_active(space, adjoint_active)
        adjoint_residual = _expand_active(space, active_residual)
        adjoint_norm = _tree_norm(adjoint_residual)
        iterations = jnp.asarray(1, dtype=jnp.int32)
        active_dimension = space.size

    relative = adjoint_norm / jnp.maximum(rhs_norm, jnp.asarray(1.0e-30, dtype=rhs_norm.dtype))
    tolerance = jnp.maximum(
        jnp.asarray(cfg.atol, dtype=rhs_norm.dtype),
        jnp.asarray(cfg.rtol, dtype=rhs_norm.dtype) * rhs_norm,
    )
    return _ProjectedStateAdjoint(
        config=cfg,
        dof_mask=dof_mask,
        residual=F,
        z_star=z_star,
        adjoint=adjoint,
        base_residual_norm=base_norm,
        state_cotangent_norm=rhs_norm,
        adjoint_residual_norm=adjoint_norm,
        relative_adjoint_residual=relative,
        active_dimension=jnp.asarray(active_dimension, dtype=jnp.int32),
        iterations=iterations,
        converged=adjoint_norm <= tolerance,
        linear_solver_converged=linear_solver_converged,
    )


def scalar_parameter_state_pullback(
    evaluator: FreeBoundaryResidualEvaluator,
    state: SpectralState,
    field_from_alpha: Callable[[Array], Any],
    state_cotangent: SpectralState,
    *,
    alpha0: float | Array = 0.0,
    dof_mask: SpectralState | None = None,
    config: FreeBoundaryTangentConfig | None = None,
) -> FreeBoundaryStatePullbackResult:
    """Pull one state cotangent back through a scalar-parameter root.

    This is the reusable B4 primitive.  It solves

    ``F_state.T @ lambda = P @ state_cotangent``

    at the finite scalar ``alpha0`` and returns ``-lambda.T @ F_alpha``.
    ``adjoint_backend="forward_dense"``
    (the CPU default) assembles ``Q.T @ F_state @ Q`` exclusively from the JVP
    map returned by :func:`jax.linearize` and explicitly solves its transpose;
    it never assumes the Jacobian is symmetric.  ``"forward_dense_jax"`` is an
    opt-in float64 equivalent that retains the dense operands on the selected
    JAX device.  ``"reverse_gmres"`` transposes only that already-linearized
    map.  On the CTH CPU validation case its cold transpose compilation did not
    finish within an eight-minute ceiling, so it remains an opt-in future
    optimization rather than the validated default.
    ``F_alpha`` is evaluated by one scalar forward JVP, avoiding a VJP of the
    complete external-field pytree.  Direct objective dependence on ``alpha``
    is intentionally outside this state-only pullback.
    """
    alpha = _coerce_scalar_alpha0(alpha0, state)
    state_cotangent = _coerce_state_cotangent(state_cotangent, state)
    external_field = field_from_alpha(alpha)
    solution = _solve_projected_state_adjoint(
        evaluator,
        state,
        external_field,
        state_cotangent,
        dof_mask,
        config,
    )

    # The only parameter operation in reverse is this scalar contraction.
    # The (potentially large) field pytree is traversed in forward mode.
    _, parameter_residual_derivative = jax.jvp(
        lambda value: solution.residual(solution.z_star, field_from_alpha(value)),
        (alpha,),
        (jnp.ones_like(alpha),),
    )
    parameter_cotangent = -_tree_dot(solution.adjoint, parameter_residual_derivative)

    return FreeBoundaryStatePullbackResult(
        parameter_cotangent=parameter_cotangent,
        adjoint=solution.adjoint,
        dof_mask=solution.dof_mask,
        base_residual_norm=solution.base_residual_norm,
        state_cotangent_norm=solution.state_cotangent_norm,
        parameter_residual_derivative_norm=_tree_norm(parameter_residual_derivative),
        adjoint_residual_norm=solution.adjoint_residual_norm,
        relative_adjoint_residual=solution.relative_adjoint_residual,
        active_dimension=solution.active_dimension,
        iterations=solution.iterations,
        converged=solution.converged,
        linear_solver_converged=solution.linear_solver_converged,
        backend=solution.config.adjoint_backend,
    )

def scalar_state_objective_adjoint(
    evaluator: FreeBoundaryResidualEvaluator,
    state: SpectralState,
    field_from_alpha: Callable[[Array], Any],
    objective_fn: Callable[[SpectralState], Array],
    *,
    alpha0: float | Array = 0.0,
    dof_mask: SpectralState | None = None,
    config: FreeBoundaryTangentConfig | None = None,
) -> FreeBoundaryAdjointResult:
    """Differentiate one scalar state-only objective by the B4 adjoint."""
    objective_value, state_cotangent = jax.value_and_grad(objective_fn)(state)
    if np.shape(objective_value) != ():
        raise ValueError("objective_fn must return one scalar")
    pullback = scalar_parameter_state_pullback(
        evaluator,
        state,
        field_from_alpha,
        state_cotangent,
        alpha0=alpha0,
        dof_mask=dof_mask,
        config=config,
    )
    return FreeBoundaryAdjointResult(
        objective_value=objective_value,
        derivative=pullback.parameter_cotangent,
        state_pullback=pullback,
    )


def one_current_tangent(
    evaluator: FreeBoundaryResidualEvaluator,
    state: SpectralState,
    external_field: Any,
    current_index: int,
    *,
    current_scale: float | Array = 1.0,
    dof_mask: SpectralState | None = None,
    config: FreeBoundaryTangentConfig | None = None,
) -> FreeBoundaryTangentResult:
    """Convenience tangent for one ``external_field.extcur`` entry.

    The scalar parameter is ``alpha`` with

    ``extcur[current_index] += alpha * current_scale``.

    ``current_scale`` is expressed in the units of the ``extcur`` leaf; those
    are physical amperes only for field objects whose table normalization has
    that convention.  Passing the base leaf value produces a fractional
    current derivative.
    """
    if not hasattr(external_field, "extcur"):
        raise TypeError("one_current_tangent requires an external_field.extcur leaf")
    extcur = jnp.asarray(external_field.extcur)
    if extcur.ndim != 1:
        raise ValueError("external_field.extcur must be one-dimensional")
    index = int(current_index)
    if not 0 <= index < int(extcur.shape[0]):
        raise IndexError(f"current_index {index} outside extcur length {extcur.shape[0]}")
    scale = jnp.asarray(current_scale, dtype=extcur.dtype)

    def field_from_alpha(alpha):
        return dataclasses.replace(
            external_field,
            extcur=extcur.at[index].add(alpha * scale),
        )

    return scalar_parameter_tangent(
        evaluator,
        state,
        field_from_alpha,
        dof_mask=dof_mask,
        config=config,
    )


def one_current_adjoint(
    evaluator: FreeBoundaryResidualEvaluator,
    state: SpectralState,
    external_field: Any,
    current_index: int,
    objective_fn: Callable[[SpectralState], Array],
    *,
    current_scale: float | Array = 1.0,
    dof_mask: SpectralState | None = None,
    config: FreeBoundaryTangentConfig | None = None,
) -> FreeBoundaryAdjointResult:
    """Convenience scalar adjoint for one ``external_field.extcur`` entry.

    The parameter convention is the same as :func:`one_current_tangent`:
    ``extcur[current_index] += alpha * current_scale``.
    """
    if not hasattr(external_field, "extcur"):
        raise TypeError("one_current_adjoint requires an external_field.extcur leaf")
    extcur = jnp.asarray(external_field.extcur)
    if extcur.ndim != 1:
        raise ValueError("external_field.extcur must be one-dimensional")
    index = int(current_index)
    if not 0 <= index < int(extcur.shape[0]):
        raise IndexError(f"current_index {index} outside extcur length {extcur.shape[0]}")
    scale = jnp.asarray(current_scale, dtype=extcur.dtype)

    def field_from_alpha(alpha):
        return dataclasses.replace(
            external_field,
            extcur=extcur.at[index].add(alpha * scale),
        )

    return scalar_state_objective_adjoint(
        evaluator,
        state,
        field_from_alpha,
        objective_fn,
        dof_mask=dof_mask,
        config=config,
    )
