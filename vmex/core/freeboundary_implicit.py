"""Projected free-boundary derivatives and scalar implicit solve wrapper.

This module linearizes the fully rebuilt VMEX--NESTOR residual at a converged
single-stage free-boundary equilibrium.  It provides the lower-level B2--B4
operations: the exact evolved-coordinate projector, scalar parameter tangents,
state pullbacks, scalar state-objective adjoints, and one-current convenience
wrappers.  It also provides a scalar custom VJP around the opaque adaptive host
solve; only the converged projected residual equation is differentiated.

The adaptive host solve remains outside differentiation.  All derivatives are
taken with respect to the converged projected residual equation.
"""

from __future__ import annotations

import dataclasses
import functools
import threading
import types
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from solvax import gmres as _solvax_gmres

from . import implicit as _fixed_implicit
from .device import AUTO, device_context
from .freeboundary import (
    FreeBoundaryResidualEvaluator,
    _validate_edge_force_tolerance,
    make_free_boundary_residual_evaluator,
    solve_free_boundary,
)
from .input import VmecInput
from .solver import SolveResult, SpectralState
from .solver import resolution_from_input as _resolution_from_input
from .transforms import register_pytree_dataclass as _register
from .wout import wout_from_state

__all__ = [
    "FreeBoundaryAdjointResult",
    "FreeBoundaryImplicitConfig",
    "FreeBoundaryStatePullbackResult",
    "FreeBoundaryTangentConfig",
    "FreeBoundaryTangentResult",
    "free_boundary_dof_mask",
    "free_boundary_implicit_result",
    "free_boundary_implicit_stats",
    "make_free_boundary_implicit_config",
    "make_projected_free_boundary_residual",
    "one_current_adjoint",
    "one_current_tangent",
    "reset_free_boundary_implicit_stats",
    "scalar_parameter_state_pullback",
    "scalar_parameter_tangent",
    "scalar_state_objective_adjoint",
    "solve_free_boundary_implicit",
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


# ---------------------------------------------------------------------------
# B5: opaque host free-boundary solve with an implicit scalar custom VJP
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FreeBoundaryImplicitConfig:
    """Identity-hashed context for one scalar free-boundary solve family.

    ``field_from_alpha`` is the JAX-traceable scalar field constructor used by
    B3/B4. Host solves begin from ``branch_inp``, whose LCFS and axis are
    rebound to ``anchor_state``. Each distinct target restarts from that common
    anchor and follows a deterministic, evenly spaced continuation path.

    With ``preserve_m1_constraint_slice=True``, the rebound input stays fixed
    and exact accepted states are carried between path points. Otherwise, the
    input is rebound at each accepted intermediate point. Every accepted point
    is checked against the config's common projected residual before it may be
    cached or used by the custom VJP.

    ``eq=False`` preserves identity hashing for JAX's static ``cfg`` argument
    and the weak callback caches below. ``anchor_residual_norm`` is the active
    ``||P(F)||`` used by B3/B4; the raw norm and projected maximum entry are
    retained as diagnostics.
    """

    inp: VmecInput
    field_from_alpha: Callable[[Array], Any]
    alpha_anchor: float
    resolution: Any
    ftol: float
    max_iterations: int
    continuation_step: float
    max_continuation_steps: int
    anchor_result: SolveResult
    anchor_state: SpectralState
    anchor_iterations: int
    anchor_residual_norm: float
    anchor_raw_residual_norm: float
    anchor_projected_residual_max_abs: float
    anchor_volume: float
    branch_inp: VmecInput
    evaluator: FreeBoundaryResidualEvaluator
    dof_mask: SpectralState
    linear_config: FreeBoundaryTangentConfig
    device: Any = AUTO
    preserve_m1_constraint_slice: bool = False
    include_edge_in_convergence: bool = False
    edge_force_tolerance: float | None = None

    def __post_init__(self) -> None:
        if type(self.preserve_m1_constraint_slice) is not bool:
            raise TypeError("preserve_m1_constraint_slice must be a bool")
        if type(self.include_edge_in_convergence) is not bool:
            raise TypeError("include_edge_in_convergence must be a bool")
        tolerance = _validate_edge_force_tolerance(
            self.edge_force_tolerance,
            include_edge_in_convergence=self.include_edge_in_convergence,
        )
        if self.include_edge_in_convergence and tolerance is None:
            tolerance = _validate_edge_force_tolerance(
                self.ftol,
                include_edge_in_convergence=True,
            )
        object.__setattr__(self, "edge_force_tolerance", tolerance)

    @property
    def dtype(self):
        return jnp.asarray(self.anchor_state.R_cos).dtype


_FREEB_IMPLICIT_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_FREEB_IMPLICIT_SOLVES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_FREEB_IMPLICIT_ROOTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_FREEB_IMPLICIT_STATS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_FREEB_IMPLICIT_META_LOCK = threading.RLock()


def _new_implicit_stats(*, anchor_solve: bool) -> dict[str, Any]:
    return {
        "anchor_host_solves": int(anchor_solve),
        "forward_callbacks": 0,
        "forward_host_solves": 0,
        "forward_iterations": 0,
        "forward_memo_hits": 0,
        "forward_failures": 0,
        "backward_callbacks": 0,
        "backward_linear_solves": 0,
        "backward_failures": 0,
        "last_forward_alpha": None,
        "last_forward_raw_residual_norm": None,
        "last_forward_projected_residual_norm": None,
        "last_forward_projected_residual_max_abs": None,
        "last_forward_constraint_slice_defect_norm": None,
        "last_forward_constraint_slice_atol": None,
        "last_forward_error": None,
        "last_backward": None,
    }


def _config_lock(cfg: FreeBoundaryImplicitConfig) -> threading.RLock:
    with _FREEB_IMPLICIT_META_LOCK:
        lock = _FREEB_IMPLICIT_LOCKS.get(cfg)
        if lock is None:
            lock = threading.RLock()
            _FREEB_IMPLICIT_LOCKS[cfg] = lock
        return lock


def _host_scalar(value, *, name: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar!r}")
    return scalar


def _alpha_key(value: float) -> bytes:
    """Return a bit-exact float64 memo key without tolerance aliases."""
    return np.asarray(np.float64(value)).tobytes()


def _set_last_forward_root_diagnostics(
    stats: dict[str, Any],
    diagnostics: dict[str, float] | None,
) -> None:
    """Bind root metrics to the same target as ``last_forward_alpha``."""
    names = (
        "raw_residual_norm",
        "projected_residual_norm",
        "projected_residual_max_abs",
        "constraint_slice_defect_norm",
        "constraint_slice_atol",
    )
    for name in names:
        stats[f"last_forward_{name}"] = (
            None if diagnostics is None else diagnostics.get(name)
        )


def _implicit_limits(
    inp: VmecInput,
    ftol: float | None,
    max_iterations: int | None,
) -> tuple[float, int]:
    """Resolve the final-grid defaults used by the implicit host solve."""
    resolved_ftol = (
        float(np.asarray(inp.ftol_array).reshape(-1)[-1])
        if ftol is None
        else float(ftol)
    )
    resolved_iterations = (
        int(np.asarray(inp.niter_array).reshape(-1)[-1])
        if max_iterations is None
        else int(max_iterations)
    )
    return resolved_ftol, resolved_iterations


def _implicit_resolution(inp: VmecInput, resolution):
    """Use the final NS_ARRAY grid unless a resolution is supplied."""
    if resolution is not None:
        return resolution
    final_ns = int(np.asarray(inp.ns_array).reshape(-1)[-1])
    return _resolution_from_input(inp, ns=final_ns)


def _branch_input_from_result(
    source: VmecInput,
    result: SolveResult,
) -> VmecInput:
    """Rebind a branch input to one converged LCFS and magnetic axis."""
    wout = wout_from_state(
        inp=source,
        state=result.state,
        fsqr=float(result.fsqr),
        fsqz=float(result.fsqz),
        fsql=float(result.fsql),
        niter=int(result.iterations),
        converged=bool(result.converged),
    )
    rbc = np.zeros_like(source.rbc)
    zbs = np.zeros_like(source.zbs)
    n_input = (np.asarray(wout.xn) / float(wout.nfp)).astype(int)
    for mode, (m_value, n_value) in enumerate(
        zip(np.asarray(wout.xm, dtype=int), n_input)
    ):
        if m_value < source.mpol and abs(n_value) <= source.ntor:
            rbc[n_value + source.ntor, m_value] = np.asarray(wout.rmnc)[-1, mode]
            zbs[n_value + source.ntor, m_value] = np.asarray(wout.zmns)[-1, mode]
    axis_size = int(source.ntor) + 1
    return dataclasses.replace(
        source,
        rbc=rbc,
        zbs=zbs,
        raxis_c=np.asarray(wout.raxis_cc)[:axis_size],
        zaxis_s=np.asarray(wout.zaxis_cs)[:axis_size],
    )


def _implicit_constraint_slice_atol(state: SpectralState) -> float:
    """Return a dtype-aware coefficient tolerance for the common chart."""
    state_scale = max(1.0, float(_tree_norm(state)))
    state_dtype = np.dtype(np.asarray(state.R_cos).dtype)
    epsilon = (
        float(np.finfo(state_dtype).eps)
        if np.issubdtype(state_dtype, np.floating)
        else float(np.finfo(np.float64).eps)
    )
    return max(1.0e-10, 100.0 * epsilon * state_scale)


def _implicit_projected_root_diagnostics(
    evaluator: FreeBoundaryResidualEvaluator,
    dof_mask: SpectralState,
    state: SpectralState,
    field: Any,
    *,
    base_residual_atol: float,
    context: str,
    reference_state: SpectralState | None = None,
) -> dict[str, float]:
    """Gate one state against the projected equation and common chart."""
    residual = evaluator(state, field).residual
    projector = _projector(evaluator, dof_mask)
    projected = projector(residual)
    raw_residual_norm = float(_tree_norm(residual))
    projected_residual_norm = float(_tree_norm(projected))
    projected_residual_max_abs = max(
            float(jnp.max(jnp.abs(leaf)))
            for leaf in jax.tree.leaves(projected)
        )
    named_diagnostics = {
        "raw ||F||": raw_residual_norm,
        "||P(F)||": projected_residual_norm,
        "max|P(F)|": projected_residual_max_abs,
    }
    constraint_slice_defect_norm = None
    constraint_slice_atol = None
    if reference_state is not None:
        delta = jax.tree.map(
            lambda actual, reference: actual - reference,
            state,
            reference_state,
        )
        projected_delta = projector(delta)
        slice_defect = jax.tree.map(
            lambda actual, active: actual - active,
            delta,
            projected_delta,
        )
        constraint_slice_defect_norm = float(_tree_norm(slice_defect))
        constraint_slice_atol = _implicit_constraint_slice_atol(state)
        named_diagnostics["constraint-slice defect"] = (
            constraint_slice_defect_norm
        )
    try:
        checked = _require_finite_residual_diagnostics(
            named_diagnostics,
            context=context,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if checked["||P(F)||"] > float(base_residual_atol):
        raise RuntimeError(
            f"{context} is not a root of the projected free-boundary residual: "
            f"||P(F)||={checked['||P(F)||']:.3e} > "
            f"base_residual_atol={float(base_residual_atol):.3e} "
            f"(raw ||F||={checked['raw ||F||']:.3e})"
        )
    if (
        constraint_slice_defect_norm is not None
        and constraint_slice_atol is not None
        and constraint_slice_defect_norm > constraint_slice_atol
    ):
        raise RuntimeError(
            f"{context} left the common implicit constraint slice: "
            f"||(I-P)(state-anchor)||={constraint_slice_defect_norm:.3e} > "
            f"constraint_slice_atol={constraint_slice_atol:.3e}"
        )
    diagnostics = {
        "raw_residual_norm": checked["raw ||F||"],
        "projected_residual_norm": checked["||P(F)||"],
        "projected_residual_max_abs": checked["max|P(F)|"],
    }
    if constraint_slice_defect_norm is not None:
        diagnostics["constraint_slice_defect_norm"] = (
            constraint_slice_defect_norm
        )
        diagnostics["constraint_slice_atol"] = constraint_slice_atol
    return diagnostics


def make_free_boundary_implicit_config(
    inp: VmecInput,
    field_from_alpha: Callable[[Array], Any],
    *,
    alpha_anchor: float | Array = 0.0,
    resolution=None,
    ftol: float | None = None,
    max_iterations: int | None = None,
    continuation_step: float = 5.0e-2,
    max_continuation_steps: int = 64,
    initial_state: SpectralState | None = None,
    device: Any = AUTO,
    preserve_m1_constraint_slice: bool = False,
    include_edge_in_convergence: bool = False,
    edge_force_tolerance: float | None = None,
    linear_config: FreeBoundaryTangentConfig | None = None,
) -> FreeBoundaryImplicitConfig:
    """Prepare a deterministic scalar free-boundary implicit solve family.

    The factory performs and validates one complete host solve at
    ``alpha_anchor``, then rebinds a common input to that solution's LCFS and
    axis. Every later target follows a target-specific continuation path and
    is accepted only when it solves the same projected residual used by the
    implicit tangent and adjoint. ``device`` controls both the nonlinear host
    solves and the projected-root/adjoint work inside the callbacks.
    """
    if not isinstance(inp, VmecInput):
        raise TypeError("inp must be a VmecInput")
    if not bool(inp.lfreeb):
        raise ValueError("free-boundary implicit config requires LFREEB=T")
    if bool(inp.lasym):
        raise NotImplementedError(
            "free-boundary implicit config currently supports "
            "stellarator symmetry only"
        )
    if not callable(field_from_alpha):
        raise TypeError("field_from_alpha must be callable")
    if type(preserve_m1_constraint_slice) is not bool:
        raise TypeError("preserve_m1_constraint_slice must be a bool")
    if type(include_edge_in_convergence) is not bool:
        raise TypeError("include_edge_in_convergence must be a bool")
    edge_tolerance_option = _validate_edge_force_tolerance(
        edge_force_tolerance,
        include_edge_in_convergence=include_edge_in_convergence,
    )
    anchor = _host_scalar(alpha_anchor, name="alpha_anchor")
    step = float(continuation_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("continuation_step must be finite and > 0")
    max_steps = int(max_continuation_steps)
    if max_steps < 1:
        raise ValueError("max_continuation_steps must be >= 1")
    resolved_resolution = _implicit_resolution(inp, resolution)
    resolved_ftol, resolved_iterations = _implicit_limits(
        inp,
        ftol,
        max_iterations,
    )
    if not np.isfinite(resolved_ftol) or resolved_ftol <= 0.0:
        raise ValueError("ftol must be finite and > 0")
    resolved_edge_tolerance = (
        resolved_ftol
        if include_edge_in_convergence and edge_tolerance_option is None
        else edge_tolerance_option
    )
    if resolved_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    linear = (
        FreeBoundaryTangentConfig()
        if linear_config is None
        else _validated_config(linear_config)
    )

    anchor_alpha = jnp.asarray(anchor, dtype=jnp.float64)
    try:
        anchor_field = field_from_alpha(anchor_alpha)
        anchor_result = solve_free_boundary(
            inp,
            external_field=anchor_field,
            resolution=resolved_resolution,
            ftol=resolved_ftol,
            max_iterations=resolved_iterations,
            error_on_no_convergence=False,
            initial_state=initial_state,
            device=device,
            preserve_m1_constraint_slice=preserve_m1_constraint_slice,
            include_edge_in_convergence=include_edge_in_convergence,
            edge_force_tolerance=resolved_edge_tolerance,
        )
    except Exception as exc:
        raise RuntimeError(
            f"anchor free-boundary solve failed at alpha={anchor:.17g}: {exc}"
        ) from exc
    if not bool(anchor_result.converged):
        edge = (
            f", fedge={float(anchor_result.fedge):.3e}, "
            f"edge_force_tolerance={float(resolved_edge_tolerance):.3e}"
            if include_edge_in_convergence
            else ""
        )
        raise RuntimeError(
            "anchor free-boundary solve did not converge: "
            f"alpha={anchor:.17g}, iterations={int(anchor_result.iterations)}, "
            f"fsq=({float(anchor_result.fsqr):.3e}, "
            f"{float(anchor_result.fsqz):.3e}, "
            f"{float(anchor_result.fsql):.3e}){edge}"
        )

    branch_inp = _branch_input_from_result(inp, anchor_result)
    with device_context(device, resolved_resolution):
        evaluator = make_free_boundary_residual_evaluator(
            branch_inp,
            resolution=resolved_resolution,
        )
        dof_mask = free_boundary_dof_mask(evaluator)
        anchor_diagnostics = _implicit_projected_root_diagnostics(
            evaluator,
            dof_mask,
            anchor_result.state,
            anchor_field,
            base_residual_atol=linear.base_residual_atol,
            context="anchor projected rebound free-boundary root",
            reference_state=anchor_result.state,
        )
        anchor_volume = float(
            _fixed_implicit.plasma_volume(
                anchor_result.state,
                evaluator.runtime,
            )
        )
    cfg = FreeBoundaryImplicitConfig(
        inp=inp,
        field_from_alpha=field_from_alpha,
        alpha_anchor=anchor,
        resolution=resolved_resolution,
        ftol=resolved_ftol,
        max_iterations=resolved_iterations,
        continuation_step=step,
        max_continuation_steps=max_steps,
        anchor_result=anchor_result,
        anchor_state=anchor_result.state,
        anchor_iterations=int(anchor_result.iterations),
        anchor_residual_norm=anchor_diagnostics["projected_residual_norm"],
        anchor_raw_residual_norm=anchor_diagnostics["raw_residual_norm"],
        anchor_projected_residual_max_abs=anchor_diagnostics[
            "projected_residual_max_abs"
        ],
        anchor_volume=anchor_volume,
        branch_inp=branch_inp,
        evaluator=evaluator,
        dof_mask=dof_mask,
        linear_config=linear,
        device=device,
        preserve_m1_constraint_slice=preserve_m1_constraint_slice,
        include_edge_in_convergence=include_edge_in_convergence,
        edge_force_tolerance=resolved_edge_tolerance,
    )
    with _FREEB_IMPLICIT_META_LOCK:
        _FREEB_IMPLICIT_LOCKS[cfg] = threading.RLock()
        _FREEB_IMPLICIT_SOLVES[cfg] = {
            _alpha_key(anchor): anchor_result,
        }
        _FREEB_IMPLICIT_ROOTS[cfg] = {
            _alpha_key(anchor): dict(anchor_diagnostics),
        }
        _FREEB_IMPLICIT_STATS[cfg] = _new_implicit_stats(anchor_solve=True)
    return cfg


def free_boundary_implicit_stats(
    cfg: FreeBoundaryImplicitConfig,
) -> dict[str, Any]:
    """Return callback, host-solve, memo, root, and adjoint diagnostics."""
    if not isinstance(cfg, FreeBoundaryImplicitConfig):
        raise TypeError("cfg must be a FreeBoundaryImplicitConfig")
    with _config_lock(cfg):
        stats = dict(_FREEB_IMPLICIT_STATS[cfg])
        last = stats.get("last_backward")
        if isinstance(last, dict):
            stats["last_backward"] = dict(last)
        stats["memo_entries"] = len(_FREEB_IMPLICIT_SOLVES[cfg])
        stats["anchor_iterations"] = cfg.anchor_iterations
        stats["anchor_residual_norm"] = cfg.anchor_residual_norm
        stats["anchor_raw_residual_norm"] = cfg.anchor_raw_residual_norm
        stats["anchor_projected_residual_max_abs"] = (
            cfg.anchor_projected_residual_max_abs
        )
        stats["anchor_volume"] = cfg.anchor_volume
        stats["preserve_m1_constraint_slice"] = (
            cfg.preserve_m1_constraint_slice
        )
        stats["include_edge_in_convergence"] = cfg.include_edge_in_convergence
        stats["edge_force_tolerance"] = cfg.edge_force_tolerance
        stats["device"] = cfg.device
        return stats


def reset_free_boundary_implicit_stats(
    cfg: FreeBoundaryImplicitConfig,
    *,
    clear_memo: bool = False,
) -> None:
    """Reset counters, optionally dropping trials but retaining the anchor."""
    if not isinstance(cfg, FreeBoundaryImplicitConfig):
        raise TypeError("cfg must be a FreeBoundaryImplicitConfig")
    with _config_lock(cfg):
        _FREEB_IMPLICIT_STATS[cfg] = _new_implicit_stats(anchor_solve=False)
        if clear_memo:
            _FREEB_IMPLICIT_SOLVES[cfg] = {
                _alpha_key(cfg.alpha_anchor): cfg.anchor_result,
            }
            _FREEB_IMPLICIT_ROOTS[cfg] = {
                _alpha_key(cfg.alpha_anchor): {
                    "raw_residual_norm": cfg.anchor_raw_residual_norm,
                    "projected_residual_norm": cfg.anchor_residual_norm,
                    "projected_residual_max_abs": (
                        cfg.anchor_projected_residual_max_abs
                    ),
                    "constraint_slice_defect_norm": 0.0,
                    "constraint_slice_atol": (
                        _implicit_constraint_slice_atol(cfg.anchor_state)
                    ),
                },
            }


def free_boundary_implicit_result(
    alpha: float | Array,
    cfg: FreeBoundaryImplicitConfig,
) -> SolveResult:
    """Return the full host ``SolveResult`` through the shared exact memo."""
    if not isinstance(cfg, FreeBoundaryImplicitConfig):
        raise TypeError("cfg must be a FreeBoundaryImplicitConfig")
    value = _host_scalar(alpha, name="alpha")
    return _solve_alpha_from_anchor(cfg, value)


def _solve_alpha_from_anchor(
    cfg: FreeBoundaryImplicitConfig,
    alpha: float,
) -> SolveResult:
    """Host solve with exact memo and target-specific anchor continuation."""
    key = _alpha_key(alpha)
    lock = _config_lock(cfg)
    with lock:
        cache = _FREEB_IMPLICIT_SOLVES[cfg]
        stats = _FREEB_IMPLICIT_STATS[cfg]
        stats["last_forward_alpha"] = alpha
        _set_last_forward_root_diagnostics(stats, None)
        hit = cache.get(key)
        if hit is not None:
            stats["forward_memo_hits"] += 1
            root_cache = _FREEB_IMPLICIT_ROOTS.get(cfg, {})
            _set_last_forward_root_diagnostics(stats, root_cache.get(key))
            stats["last_forward_error"] = None
            return hit

        delta = alpha - cfg.alpha_anchor
        count = max(1, int(np.ceil(abs(delta) / cfg.continuation_step)))
        if count > cfg.max_continuation_steps:
            stats["forward_failures"] += 1
            stats["last_forward_error"] = (
                f"target requires {count} continuation steps, "
                f"limit is {cfg.max_continuation_steps}"
            )
            raise RuntimeError(stats["last_forward_error"])

        source = cfg.branch_inp
        initial_state = cfg.anchor_state
        result = None
        for index in range(1, count + 1):
            point = (
                alpha
                if index == count
                else cfg.alpha_anchor
                + delta * (float(index) / float(count))
            )
            try:
                field = cfg.field_from_alpha(
                    jnp.asarray(point, dtype=cfg.dtype)
                )
                result = solve_free_boundary(
                    source,
                    external_field=field,
                    resolution=cfg.resolution,
                    ftol=cfg.ftol,
                    max_iterations=cfg.max_iterations,
                    error_on_no_convergence=False,
                    initial_state=(
                        initial_state
                        if cfg.preserve_m1_constraint_slice
                        else None
                    ),
                    device=cfg.device,
                    preserve_m1_constraint_slice=(
                        cfg.preserve_m1_constraint_slice
                    ),
                    include_edge_in_convergence=(
                        cfg.include_edge_in_convergence
                    ),
                    edge_force_tolerance=cfg.edge_force_tolerance,
                )
            except Exception as exc:
                stats["forward_failures"] += 1
                _set_last_forward_root_diagnostics(stats, None)
                stats["last_forward_error"] = (
                    "free-boundary continuation failed at "
                    f"alpha={point:.17g}: {exc}"
                )
                raise RuntimeError(stats["last_forward_error"]) from exc
            stats["forward_host_solves"] += 1
            stats["forward_iterations"] += int(result.iterations)
            if not bool(result.converged):
                stats["forward_failures"] += 1
                _set_last_forward_root_diagnostics(stats, None)
                edge = (
                    f", fedge={float(result.fedge):.3e}, "
                    f"edge_force_tolerance="
                    f"{float(cfg.edge_force_tolerance):.3e}"
                    if cfg.include_edge_in_convergence
                    else ""
                )
                stats["last_forward_error"] = (
                    "free-boundary continuation did not converge at "
                    f"alpha={point:.17g}: "
                    f"iterations={int(result.iterations)}, "
                    f"fsq=({float(result.fsqr):.3e}, "
                    f"{float(result.fsqz):.3e}, "
                    f"{float(result.fsql):.3e}){edge}"
                )
                raise RuntimeError(stats["last_forward_error"])

            try:
                with device_context(cfg.device, cfg.resolution):
                    diagnostics = _implicit_projected_root_diagnostics(
                        cfg.evaluator,
                        cfg.dof_mask,
                        result.state,
                        field,
                        base_residual_atol=(
                            cfg.linear_config.base_residual_atol
                        ),
                        context=(
                            "accepted scalar continuation point "
                            f"{index}/{count} at alpha={point:.17g}"
                        ),
                        reference_state=cfg.anchor_state,
                    )
            except Exception as exc:
                stats["forward_failures"] += 1
                _set_last_forward_root_diagnostics(stats, None)
                stats["last_forward_error"] = str(exc)
                raise RuntimeError(stats["last_forward_error"]) from exc
            _set_last_forward_root_diagnostics(stats, diagnostics)

            if index < count:
                if cfg.preserve_m1_constraint_slice:
                    initial_state = result.state
                else:
                    try:
                        source = _branch_input_from_result(source, result)
                    except Exception as exc:
                        stats["forward_failures"] += 1
                        _set_last_forward_root_diagnostics(stats, None)
                        stats["last_forward_error"] = (
                            "free-boundary scalar continuation could not "
                            f"rebind accepted point {index}/{count}, "
                            f"alpha={point:.17g}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        raise RuntimeError(stats["last_forward_error"]) from exc

        if result is None:  # pragma: no cover - count is always at least one.
            raise RuntimeError("free-boundary continuation produced no result")
        cache[key] = result
        root_cache = _FREEB_IMPLICIT_ROOTS.get(cfg)
        if root_cache is None:
            root_cache = {}
            _FREEB_IMPLICIT_ROOTS[cfg] = root_cache
        root_cache[key] = dict(diagnostics)
        stats["last_forward_error"] = None
        return result


def _state_as_numpy(
    state: SpectralState,
    cfg: FreeBoundaryImplicitConfig,
) -> SpectralState:
    dtype = np.dtype(cfg.dtype)
    return jax.tree.map(
        lambda value: np.asarray(value, dtype=dtype),
        state,
    )


def _host_free_boundary_state_callback(
    cfg: FreeBoundaryImplicitConfig,
    alpha_value,
) -> SpectralState:
    alpha = _host_scalar(alpha_value, name="alpha")
    with _config_lock(cfg):
        _FREEB_IMPLICIT_STATS[cfg]["forward_callbacks"] += 1
    result = _solve_alpha_from_anchor(cfg, alpha)
    return _state_as_numpy(result.state, cfg)


def _host_free_boundary_pullback_callback(
    cfg: FreeBoundaryImplicitConfig,
    alpha_value,
    state_value: SpectralState,
    state_cotangent_value: SpectralState,
):
    alpha = _host_scalar(alpha_value, name="alpha")
    with _config_lock(cfg):
        _FREEB_IMPLICIT_STATS[cfg]["backward_callbacks"] += 1

    try:
        with device_context(cfg.device, cfg.resolution):
            state = jax.tree.map(jnp.asarray, state_value)
            state_cotangent = jax.tree.map(
                jnp.asarray,
                state_cotangent_value,
            )
            result = scalar_parameter_state_pullback(
                cfg.evaluator,
                state,
                cfg.field_from_alpha,
                state_cotangent,
                alpha0=alpha,
                dof_mask=cfg.dof_mask,
                config=cfg.linear_config,
            )
        diagnostics = {
            "alpha": alpha,
            "base_residual_norm": float(result.base_residual_norm),
            "adjoint_residual_norm": float(result.adjoint_residual_norm),
            "relative_adjoint_residual": float(
                result.relative_adjoint_residual
            ),
            "active_dimension": int(result.active_dimension),
            "linear_solver_converged": bool(
                result.linear_solver_converged
            ),
            "converged": bool(result.converged),
            "backend": result.backend,
            "error": None,
        }
        derivative = float(result.parameter_cotangent)
        diagnostics["parameter_cotangent"] = derivative
        valid = (
            np.isfinite(derivative)
            and diagnostics["linear_solver_converged"]
            and diagnostics["converged"]
            and diagnostics["base_residual_norm"]
            <= cfg.linear_config.base_residual_atol
        )
        with _config_lock(cfg):
            stats = _FREEB_IMPLICIT_STATS[cfg]
            stats["backward_linear_solves"] += 1
            stats["last_backward"] = diagnostics
            if not valid:
                stats["backward_failures"] += 1
        if not valid:
            derivative = np.nan
    except Exception as exc:
        derivative = np.nan
        with _config_lock(cfg):
            stats = _FREEB_IMPLICIT_STATS[cfg]
            stats["backward_failures"] += 1
            stats["last_backward"] = {
                "alpha": alpha,
                "error": f"{type(exc).__name__}: {exc}",
                "linear_solver_converged": False,
                "converged": False,
            }
    return np.asarray(derivative, dtype=np.dtype(cfg.dtype))


def _free_boundary_state_struct(
    cfg: FreeBoundaryImplicitConfig,
) -> SpectralState:
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(np.shape(value), cfg.dtype),
        cfg.anchor_state,
    )


def _coerce_implicit_alpha(
    alpha: float | Array,
    cfg: FreeBoundaryImplicitConfig,
) -> Array:
    if not isinstance(cfg, FreeBoundaryImplicitConfig):
        raise TypeError("cfg must be a FreeBoundaryImplicitConfig")
    value = jnp.asarray(alpha, dtype=cfg.dtype)
    if value.ndim != 0:
        raise ValueError(f"alpha must be scalar, got shape {value.shape}")
    return value


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def solve_free_boundary_implicit(
    alpha: float | Array,
    cfg: FreeBoundaryImplicitConfig,
) -> SpectralState:
    """Run an opaque host NESTOR solve with a B4 implicit scalar pullback.

    Adaptive activation, cadence, continuation, and iteration history stay
    outside AD. The backward callback differentiates only the validated
    converged projected residual and launches no nonlinear equilibrium solve.
    Direct terms in ``alpha`` should be composed outside this wrapper.
    """
    value = _coerce_implicit_alpha(alpha, cfg)
    return jax.pure_callback(
        functools.partial(_host_free_boundary_state_callback, cfg),
        _free_boundary_state_struct(cfg),
        value,
    )


def _solve_free_boundary_implicit_fwd(alpha, cfg):
    value = _coerce_implicit_alpha(alpha, cfg)
    state = jax.pure_callback(
        functools.partial(_host_free_boundary_state_callback, cfg),
        _free_boundary_state_struct(cfg),
        value,
    )
    return state, (value, state)


def _solve_free_boundary_implicit_bwd(cfg, residual, state_cotangent):
    alpha, state = residual
    derivative = jax.pure_callback(
        functools.partial(_host_free_boundary_pullback_callback, cfg),
        jax.ShapeDtypeStruct((), cfg.dtype),
        alpha,
        state,
        state_cotangent,
    )
    return (derivative,)


solve_free_boundary_implicit.defvjp(
    _solve_free_boundary_implicit_fwd,
    _solve_free_boundary_implicit_bwd,
)
