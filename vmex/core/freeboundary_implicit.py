"""Implicit derivatives of a coupled free-boundary VMEX equilibrium.

The forward pass uses the ordinary host-driven free-boundary solver.  The
reverse pass differentiates the converged plasma--vacuum root: NESTOR is
re-evaluated on the current edge and its vacuum pressure enters the evolved
VMEC edge-force rows.  Solver iterations are therefore absent from the AD
tape; one matrix-free adjoint supplies derivatives with respect to plasma
profiles and explicit external-field parameters (including ESSOS coil shape
and current degrees of freedom or an :class:`~vmex.core.mgrid.MgridField`
current vector).
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.sparse.linalg import LinearOperator, gcrotmk

from . import implicit as im
from .freeboundary import (
    _presf_ns_scale,
    _solve_free_boundary_stage,
    _vacuum_executables,
    free_boundary_resolution,
)
from .errors import VmecError
from .input import VmecInput
from .solver import SpectralState, evaluate_forces

Array = Any


@dataclass(frozen=True, eq=False)
class FreeBoundaryImplicitConfig:
    """Static controls for :func:`solve_free_boundary_implicit`.

    ``field_from_parameters`` reconstructs the differentiable field from the
    second solve argument. ``implicit`` holds the shared Krylov tolerances and
    differentiable input-to-runtime map.
    """

    implicit: im.ImplicitConfig
    field_from_parameters: Callable[[Any], Any]
    vacuum_program: Any = None

    @property
    def resolution(self):
        return self.implicit.resolution


def make_free_boundary_config(
    inp: VmecInput,
    external_field: Any,
    *,
    ns: int | None = None,
    ftol: float | None = None,
    max_iterations: int | None = None,
    adjoint_tol: float = 1e-10,
    adjoint_maxiter: int = 300,
    adjoint_gcrot_m: int = 30,
    adjoint_gcrot_k: int = 5,
    field_from_parameters: Callable[[Any], Any] | None = None,
) -> FreeBoundaryImplicitConfig:
    """Build a coupled free-boundary derivative configuration.

    By default the second solve argument is an external-field pytree. For a
    smaller AD graph, pass ``field_from_parameters`` and then supply only the
    actual current/coil parameters to :func:`solve_free_boundary_implicit`.
    ``external_field`` here is the concrete reference used to fix resolution.
    """
    if not inp.lfreeb:
        raise ValueError("free-boundary implicit differentiation requires LFREEB=T")
    resolution = free_boundary_resolution(inp, external_field, ns=ns)
    cfg = im.make_config(
        inp, ns=resolution.ns, ftol=ftol, max_iterations=max_iterations,
        adjoint_tol=adjoint_tol, adjoint_maxiter=adjoint_maxiter,
        adjoint_gcrot_m=adjoint_gcrot_m, adjoint_gcrot_k=adjoint_gcrot_k,
    )
    if cfg.resolution != resolution:
        cfg = dataclasses.replace(cfg, resolution=resolution)
    config = FreeBoundaryImplicitConfig(
        cfg, (lambda value: value) if field_from_parameters is None
        else field_from_parameters,
    )
    return dataclasses.replace(config, vacuum_program=_vacuum_program(config))


def _vacuum_program(cfg: FreeBoundaryImplicitConfig):
    """Return the cached differentiable NESTOR program for ``cfg``."""
    icfg = cfg.implicit
    rt = im._template_runtime(icfg)
    # Some free-boundary decks intentionally leave the axis guess blank. The
    # executable only needs a non-degenerate static topology here; its actual
    # axis coordinates remain dynamic inputs to every NESTOR call.
    r00 = float(np.asarray(icfg.inp.rbc)[int(icfg.inp.ntor), 0])
    axis_r = jnp.full((icfg.resolution.nzeta,), r00)
    axis_z = jnp.zeros_like(axis_r)
    return _vacuum_executables(
        icfg.resolution, mf=int(icfg.inp.mpol) + 1,
        nf=int(icfg.inp.ntor), signgs=int(rt.setup.signgs),
        wint=np.asarray(rt.trig.wint), modes=rt.modes,
        axis_r0=axis_r, axis_z0=axis_z, use_fft=False,
        solve_on_plasma_device=True,
    )[1]


def _projected_residual(
    cfg: FreeBoundaryImplicitConfig,
    dof_mask: SpectralState,
) -> Callable:
    """Return the coupled plasma--vacuum root with dynamic linearization data."""
    icfg = cfg.implicit
    project = im._dof_projector(icfg, dof_mask)
    # The executable/topology was fixed concretely when the config was built;
    # all equilibrium and coil values below remain dynamic traced arrays.
    fused = cfg.vacuum_program
    pres_scale = jnp.asarray(
        _presf_ns_scale(icfg.inp, int(icfg.resolution.ns)), dtype=jnp.float64
    )

    @jax.jit
    def residual(z, params, field_parameters, frozen, rcon0, zcon0):
        # Unlike fixed boundary, every active edge coefficient comes from z;
        # the input boundary is only the forward solver's initial guess.
        dz = project(jax.tree.map(lambda a, b: a - b, z, frozen))
        state = jax.tree.map(jnp.add, frozen, dz)
        rt = dataclasses.replace(
            im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
            lfreeb=True, jmax=int(icfg.resolution.ns),
            presf_ns_scale=pres_scale,
        )
        external_field = cfg.field_from_parameters(field_parameters)
        rt = dataclasses.replace(
            rt, bsqvac_edge=fused.bsq(state, rt, external_field)
        )
        force, _, _ = evaluate_forces(state, rt)
        return project(force)

    return residual


_FREE_MASK_CACHE: dict[tuple, SpectralState] = {}
_FREE_HOT_CACHE: dict[FreeBoundaryImplicitConfig, SpectralState] = {}
_FREE_LAST_RESULT: dict[FreeBoundaryImplicitConfig, Any] = {}


def _mask_key(cfg: FreeBoundaryImplicitConfig) -> tuple:
    icfg = cfg.implicit
    return (icfg.resolution, bool(icfg.lconm1), int(icfg.inp.ncurr), "free")


def _host_solve_and_mask(
    cfg, params_np, field_parameters_np, *, error_on_no_convergence=True,
):
    """Opaque forward solve plus one structural free-boundary dof mask."""
    icfg = cfg.implicit
    params = jax.tree.map(jnp.asarray, params_np)
    field_parameters = jax.tree.map(jnp.asarray, field_parameters_np)
    field = cfg.field_from_parameters(field_parameters)
    inp = im.input_with_params(icfg.inp, params)
    seed = _FREE_HOT_CACHE.get(cfg)
    try:
        stage = _solve_free_boundary_stage(
            inp, external_field=field, resolution=icfg.resolution,
            ftol=icfg.ftol, max_iterations=icfg.max_iterations,
            error_on_no_convergence=error_on_no_convergence,
            initial_state=seed, use_fft=False,
        )
    except VmecError:
        if seed is None:
            raise
        stage = _solve_free_boundary_stage(
            inp, external_field=field, resolution=icfg.resolution,
            ftol=icfg.ftol, max_iterations=icfg.max_iterations,
            error_on_no_convergence=error_on_no_convergence, use_fft=False,
        )
    _FREE_HOT_CACHE[cfg] = stage.continuation_state
    _FREE_LAST_RESULT[cfg] = stage.result
    state = stage.result.state
    rcon0, zcon0 = stage.rcon0, stage.zcon0

    # Prime the static runtime/NESTOR closures before a transformed residual
    # sees them, then identify only structurally active state entries.
    rt = im.runtime_from_params(params, icfg)
    key = _mask_key(cfg)
    mask = _FREE_MASK_CACHE.get(key)
    if mask is None:
        # The active mode families are fixed by VMEC symmetry/constraints,
        # not by the dense NESTOR response. Freeze the converged edge pressure
        # while finding structural force support; tracing NESTOR here would
        # compile its LU pullback once merely to rediscover the same mask.
        rt_mask = dataclasses.replace(
            rt, rcon0=rcon0, zcon0=zcon0, lfreeb=True,
            jmax=int(icfg.resolution.ns),
            bsqvac_edge=jax.lax.stop_gradient(stage.vacuum.bsqvac),
            presf_ns_scale=jnp.asarray(
                _presf_ns_scale(inp, int(icfg.resolution.ns))
            ),
        )
        force = lambda x: evaluate_forces(x, rt_mask)[0]  # noqa: E731
        mask = im._dof_mask(
            state, rt_mask, icfg, evaluator=force, fixed_edge=False
        )
        _FREE_MASK_CACHE[key] = mask

    to_numpy = lambda tree: jax.tree.map(  # noqa: E731
        lambda value: np.asarray(value, dtype=np.float64), tree
    )
    return to_numpy(state), to_numpy(mask), to_numpy(rcon0), to_numpy(zcon0)


def _host_solve_and_mask_status(cfg, params_np, field_parameters_np):
    """Exception-free free-boundary callback for optimizer trial points."""
    try:
        state, mask, rcon0, zcon0 = _host_solve_and_mask(
            cfg, params_np, field_parameters_np, error_on_no_convergence=False,
        )
    except VmecError:
        icfg = cfg.implicit
        state = _FREE_HOT_CACHE.get(cfg)
        runtime = im._template_runtime(icfg)
        if state is None:
            state = im._initial_state(runtime.setup)
        mask = _FREE_MASK_CACHE.get(_mask_key(cfg))
        if mask is None:
            mask = jax.tree.map(jnp.zeros_like, state)
        to_numpy = lambda tree: jax.tree.map(  # noqa: E731
            lambda value: np.asarray(value, dtype=np.float64), tree
        )
        return (to_numpy(state), to_numpy(mask), to_numpy(runtime.rcon0),
                to_numpy(runtime.zcon0), np.int32(1), np.float64(np.inf),
                np.float64(np.inf))

    result = _FREE_LAST_RESULT[cfg]
    fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
    ratio = fsq / cfg.implicit.ftol
    status = 0 if bool(result.converged) or ratio <= cfg.implicit.max_fsq_ratio else 2
    return state, mask, rcon0, zcon0, np.int32(status), np.float64(fsq), np.float64(ratio)


def _baseline_struct(cfg: FreeBoundaryImplicitConfig):
    rt = im._template_runtime(cfg.implicit)
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, jnp.float64), rt.rcon0
    ), jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, jnp.float64), rt.zcon0
    )


def _callback(params, field_parameters, cfg):
    rcon_struct, zcon_struct = _baseline_struct(cfg)
    return jax.pure_callback(
        functools.partial(_host_solve_and_mask, cfg),
        (im._state_struct(cfg.implicit), im._state_struct(cfg.implicit),
         rcon_struct, zcon_struct),
        params, field_parameters,
    )


def _callback_status(params, field_parameters, cfg):
    """Return the free-boundary state, linearization data, and solve status."""
    rcon_struct, zcon_struct = _baseline_struct(cfg)
    scalar = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.pure_callback(
        functools.partial(_host_solve_and_mask_status, cfg),
        (im._state_struct(cfg.implicit), im._state_struct(cfg.implicit),
         rcon_struct, zcon_struct, jax.ShapeDtypeStruct((), jnp.int32),
         scalar, scalar),
        params, field_parameters,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def solve_free_boundary_implicit(
    params: im.ImplicitParams,
    field_parameters: Any,
    cfg: FreeBoundaryImplicitConfig,
) -> SpectralState:
    """Return a differentiable converged free-boundary spectral state."""
    state, _, _, _ = _callback(params, field_parameters, cfg)
    return state


def _solve_fwd(params, field_parameters, cfg):
    state, mask, rcon0, zcon0 = _callback(params, field_parameters, cfg)
    return state, (params, field_parameters, state, mask, rcon0, zcon0)


def _solve_bwd(cfg, saved, state_bar):
    params, field_parameters, state, mask, rcon0, zcon0 = saved
    frozen = jax.lax.stop_gradient(state)
    project = im._dof_projector(cfg.implicit, mask)
    residual = _projected_residual(cfg, mask)
    z_star = project(state)

    _, state_pullback = jax.vjp(
        lambda z: residual(
            z, params, field_parameters, frozen, rcon0, zcon0), z_star
    )
    def operator(cotangent):
        return state_pullback(cotangent)[0]

    rhs = project(state_bar)
    if any(isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(rhs)):
        # An outer jax.jit needs a staged Krylov loop. Ordinary SciPy/JAXopt
        # drivers call the concrete lane below, which compiles only one
        # transpose matvec and has a much smaller cold memory peak.
        lam, _ = im._adjoint_solve_gcrot(operator, rhs, cfg.implicit)
    else:
        lam = _host_adjoint(
            residual, z_star, params, field_parameters, frozen, rcon0, zcon0,
            rhs, cfg.implicit)

    _, parameter_pullback = jax.vjp(
        lambda p, field: residual(
            z_star, p, field, frozen, rcon0, zcon0),
        params, field_parameters,
    )
    params_bar, field_bar = parameter_pullback(
        jax.tree.map(jnp.negative, lam)
    )
    return params_bar, field_bar


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def solve_free_boundary_implicit_status(
    params: im.ImplicitParams,
    field_parameters: Any,
    cfg: FreeBoundaryImplicitConfig,
) -> tuple[SpectralState, Array, Array, Array]:
    """Differentiable state with an exception-free optimizer-trial status.

    Status 0 is derivative-certified, 1 denotes a failed solve, and 2 an
    under-converged solve. Only status 0 evaluates the implicit pullback.
    """
    state, _, _, _, status, fsq, ratio = _callback_status(
        params, field_parameters, cfg)
    return state, status, fsq, ratio


def _solve_status_fwd(params, field_parameters, cfg):
    state, mask, rcon0, zcon0, status, fsq, ratio = _callback_status(
        params, field_parameters, cfg)
    saved = (params, field_parameters, state, mask, rcon0, zcon0, status)
    return (state, status, fsq, ratio), saved


def _solve_status_bwd(cfg, saved, cotangents):
    params, field_parameters, state, mask, rcon0, zcon0, status = saved
    state_bar, _, _, _ = cotangents
    zeros = (jax.tree.map(jnp.zeros_like, params),
             jax.tree.map(jnp.zeros_like, field_parameters))

    def success(values):
        prm, field, solved, dof_mask, rcon, zcon, bar = values
        return _solve_bwd(
            cfg, (prm, field, solved, dof_mask, rcon, zcon), bar)

    if not isinstance(status, jax.core.Tracer):
        return success(
            (params, field_parameters, state, mask, rcon0, zcon0, state_bar)
        ) if int(status) == 0 else zeros
    return jax.lax.cond(
        status == 0, success, lambda _: zeros,
        (params, field_parameters, state, mask, rcon0, zcon0, state_bar),
    )


def _host_adjoint(
    residual, z_star, params, field_parameters, frozen, rcon0, zcon0, rhs, cfg,
):
    """Solve one adjoint while reusing a separately compiled JAX matvec.

    Staging GCROT together with the coupled NESTOR--VMEC transpose makes XLA
    inline that large operator into every Arnoldi loop and greatly increases
    cold compilation memory. SciPy keeps the small Krylov bookkeeping on the
    host and calls one compiled JAX operator; only vectors cross the boundary.
    """
    rhs_flat, unravel = ravel_pytree(rhs)

    @jax.jit
    def matvec(value, z, p, field, base, rcon, zcon):
        _, pullback = jax.vjp(
            lambda zz: residual(zz, p, field, base, rcon, zcon), z
        )
        return ravel_pytree(pullback(unravel(value))[0])[0]

    dynamic = (z_star, params, field_parameters, frozen, rcon0, zcon0)

    matvec(rhs_flat, *dynamic).block_until_ready()
    dtype = np.asarray(rhs_flat).dtype
    shape = rhs_flat.shape
    calls = 0

    def apply(value):
        nonlocal calls
        calls += 1
        return np.asarray(matvec(
            jnp.asarray(value, dtype=rhs_flat.dtype), *dynamic))

    matrix = LinearOperator((shape[0], shape[0]), matvec=apply, dtype=dtype)
    solution, _info = gcrotmk(
        matrix, np.asarray(rhs_flat), rtol=cfg.adjoint_tol, atol=0.0,
        m=min(cfg.adjoint_gcrot_m, shape[0]),
        k=min(cfg.adjoint_gcrot_k, shape[0]),
        maxiter=cfg.adjoint_maxiter,
    )
    residual_norm = float(np.linalg.norm(np.asarray(rhs_flat) - apply(solution)))
    tolerance = float(im._adjoint_acceptance(
        cfg, np.linalg.norm(np.asarray(rhs_flat))))
    if residual_norm > tolerance:
        im._raise_adjoint_unconverged(
            cfg, iterations=calls, residual_norm=residual_norm,
            tolerance=tolerance, method="host GCROT",
        )
    return unravel(jnp.asarray(solution, dtype=rhs_flat.dtype))


solve_free_boundary_implicit.defvjp(_solve_fwd, _solve_bwd)
solve_free_boundary_implicit_status.defvjp(_solve_status_fwd, _solve_status_bwd)


__all__ = [
    "FreeBoundaryImplicitConfig",
    "make_free_boundary_config",
    "solve_free_boundary_implicit",
    "solve_free_boundary_implicit_status",
]
