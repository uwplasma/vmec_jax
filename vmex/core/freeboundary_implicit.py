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
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import LinearOperator, gcrotmk
from solvax import SpluFactorization

from . import implicit as im
from .device import AUTO, resolve_implicit_device
from .freeboundary import (
    _presf_ns_scale,
    _presf_ns_scale_traceable,
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
    adjoint_solver: str = "coupled_gcrot"
    schur_probe_chunk_size: int = 1
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
    adjoint_solver: str = "coupled_gcrot",
    schur_probe_chunk_size: int = 1,
    field_from_parameters: Callable[[Any], Any] | None = None,
    device: Any = AUTO,
) -> FreeBoundaryImplicitConfig:
    """Build a coupled free-boundary derivative configuration.

    By default the second solve argument is an external-field pytree. For a
    smaller AD graph, pass ``field_from_parameters`` and then supply only the
    actual current/coil parameters to :func:`solve_free_boundary_implicit`.
    ``external_field`` here is the concrete reference used to fix resolution.
    ``device="auto"`` uses the CPU for the coupled implicit response on an
    accelerator host unless the process already pins JAX placement; pass an
    explicit device to override that measured lower-memory default.
    ``adjoint_solver="coupled_gcrot"`` is the certified default;
    ``"boundary_schur"`` selects the advanced radial-elimination path.
    """
    if not inp.lfreeb:
        raise ValueError("free-boundary implicit differentiation requires LFREEB=T")
    resolution = free_boundary_resolution(inp, external_field, ns=ns)
    solve_device = resolve_implicit_device(device, resolution)
    cfg = im.make_config(
        inp, ns=resolution.ns, ftol=ftol, max_iterations=max_iterations,
        adjoint_tol=adjoint_tol, adjoint_maxiter=adjoint_maxiter,
        adjoint_gcrot_m=adjoint_gcrot_m, adjoint_gcrot_k=adjoint_gcrot_k,
        device=solve_device,
    )
    if cfg.resolution != resolution:
        cfg = dataclasses.replace(cfg, resolution=resolution)
    if adjoint_solver not in {"boundary_schur", "coupled_gcrot"}:
        raise ValueError(
            "adjoint_solver must be 'boundary_schur' or 'coupled_gcrot'")
    if schur_probe_chunk_size < 1:
        raise ValueError("schur_probe_chunk_size must be positive")
    config = FreeBoundaryImplicitConfig(
        implicit=cfg,
        field_from_parameters=(lambda value: value) if field_from_parameters is None
        else field_from_parameters,
        adjoint_solver=adjoint_solver,
        schur_probe_chunk_size=int(schur_probe_chunk_size),
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
    axis_r = im._device_pin(
        icfg, jnp.full((icfg.resolution.nzeta,), r00))
    axis_z = im._device_pin(icfg, jnp.zeros_like(axis_r))
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
    *,
    formulation: str = "preconditioned",
    fixed_bsqvac: Array | None = None,
) -> Callable:
    """Return a projected coupled root in preconditioned or raw form.

    ``fixed_bsqvac`` freezes only NESTOR's edge pressure.  The resulting raw
    Jacobian is exactly block tridiagonal in radius and is the bulk operator
    used by the boundary-Schur adjoint.
    """
    if formulation not in {"preconditioned", "raw"}:
        raise ValueError(f"unknown formulation {formulation!r}")
    icfg = cfg.implicit
    project = im._dof_projector(icfg, dof_mask)
    # The executable/topology was fixed concretely when the config was built;
    # all equilibrium and coil values below remain dynamic traced arrays.
    fused = cfg.vacuum_program

    @jax.jit
    def residual(z, params, field_parameters, frozen, rcon0, zcon0):
        # Unlike fixed boundary, every active edge coefficient comes from z;
        # the input boundary is only the forward solver's initial guess.
        dz = project(jax.tree.map(lambda a, b: a - b, z, frozen))
        state = jax.tree.map(jnp.add, frozen, dz)
        rt = dataclasses.replace(
            im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
            lfreeb=True, jmax=int(icfg.resolution.ns),
            presf_ns_scale=_presf_ns_scale_traceable(
                params, icfg.inp, int(icfg.resolution.ns)),
        )
        if fixed_bsqvac is None:
            external_field = cfg.field_from_parameters(field_parameters)
            bsqvac = fused.bsq(state, rt, external_field)
        else:
            bsqvac = fixed_bsqvac
        rt = dataclasses.replace(rt, bsqvac_edge=bsqvac)
        if formulation == "preconditioned":
            force, _, _ = evaluate_forces(state, rt)
        else:
            force = im._raw_force_state(state, rt, include_edge=True)
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
    """Run the callback on the implicit config's explicitly selected device."""
    with im._device_context(cfg.implicit):
        return _host_solve_and_mask_impl(
            cfg, params_np, field_parameters_np,
            error_on_no_convergence=error_on_no_convergence,
        )


def _host_solve_and_mask_impl(
    cfg, params_np, field_parameters_np, *, error_on_no_convergence=True,
):
    """Opaque forward solve plus one structural free-boundary dof mask."""
    icfg = cfg.implicit
    params = im._device_pin(icfg, jax.tree.map(jnp.asarray, params_np))
    field_parameters = im._device_pin(
        icfg, jax.tree.map(jnp.asarray, field_parameters_np))
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
            # Host value on purpose: this runtime only discovers which dofs
            # have structural force support, a discrete answer that no
            # derivative is taken through.  The adjoint lanes above use the
            # traceable ratio instead.
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
        sharding=im._callback_sharding(cfg.implicit),
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
        sharding=im._callback_sharding(cfg.implicit),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def solve_free_boundary_implicit(
    params: im.ImplicitParams,
    field_parameters: Any,
    cfg: FreeBoundaryImplicitConfig,
) -> SpectralState:
    """Return a differentiable converged free-boundary spectral state."""
    icfg = cfg.implicit
    with im._device_context(icfg):
        params, field_parameters = im._device_pin(
            icfg, (params, field_parameters))
        state, _, _, _ = _callback(params, field_parameters, cfg)
    return state


def _solve_fwd(params, field_parameters, cfg):
    icfg = cfg.implicit
    with im._device_context(icfg):
        params, field_parameters = im._device_pin(
            icfg, (params, field_parameters))
        state, mask, rcon0, zcon0 = _callback(
            params, field_parameters, cfg)
        state, mask, rcon0, zcon0 = im._device_pin(
            icfg, (state, mask, rcon0, zcon0))
    return state, (params, field_parameters, state, mask, rcon0, zcon0)


def _solve_bwd(cfg, saved, state_bar):
    icfg = cfg.implicit
    with im._device_context(icfg):
        saved, state_bar = im._device_pin(icfg, (saved, state_bar))
        return _solve_bwd_impl(cfg, saved, state_bar)


def _solve_bwd_impl(cfg, saved, state_bar):
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
    traced = any(
        isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(rhs)
    )
    if traced:
        # An outer jax.jit needs a staged Krylov loop. Ordinary SciPy/JAXopt
        # drivers call the concrete lane below, which compiles only one
        # transpose matvec and has a much smaller cold memory peak.
        lam, _ = im._adjoint_solve_gcrot(operator, rhs, cfg.implicit)
    elif cfg.adjoint_solver == "boundary_schur":
        lam = _host_boundary_schur_adjoint(
            cfg, z_star, params, field_parameters, frozen, rcon0, zcon0,
            mask, rhs,
        )
        residual = _projected_residual(cfg, mask, formulation="raw")
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


def _host_boundary_schur_adjoint(
    cfg, z_star, params, field_parameters, frozen, rcon0, zcon0, mask, rhs,
):
    """Solve the coupled adjoint through an exact edge Schur complement.

    With NESTOR's converged edge pressure frozen, the raw VMEC Jacobian ``A``
    is block tridiagonal in radius.  The difference ``E = J - A`` has nonzero
    rows only at the free boundary.  Eliminating the bulk gives

    ``(I + U.T @ A.T^-1 @ E.T @ U) mu = U.T @ A.T^-1 @ rhs``,

    where ``U`` injects the evolved edge row. Direct three-surface assembly
    already retains every terminal VMEC stencil coupling in ``A``; only
    NESTOR's response to the moving edge remains in ``E``. One sparse bulk
    factorization and the edge solve recover the full adjoint. The final
    answer is certified against the original coupled transpose operator.
    """
    icfg = cfg.implicit
    project = im._dof_projector(icfg, mask)
    field = cfg.field_from_parameters(field_parameters)
    rt = dataclasses.replace(
        im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=int(icfg.resolution.ns),
        presf_ns_scale=_presf_ns_scale_traceable(
            params, icfg.inp, int(icfg.resolution.ns)),
    )
    bsqvac = jax.lax.stop_gradient(cfg.vacuum_program.bsq(frozen, rt, field))
    frozen_residual = _projected_residual(
        cfg, mask, formulation="raw", fixed_bsqvac=bsqvac)
    frozen_root = lambda z, p: frozen_residual(  # noqa: E731
        z, p, field_parameters, frozen, rcon0, zcon0)
    system = im._raw_block_system(
        params, icfg, frozen, mask, im._active_state_fields(icfg),
        probe_chunk_size=cfg.schur_probe_chunk_size, residual=frozen_root,
        z_star=z_star, runtime=dataclasses.replace(rt, bsqvac_edge=bsqvac),
        physical_state=frozen, include_edge=True, factor=False,
    )
    coupled_residual = _projected_residual(cfg, mask, formulation="raw")
    _, coupled_pullback = jax.vjp(
        lambda z: coupled_residual(
            z, params, field_parameters, frozen, rcon0, zcon0), z_star)
    if im._adjoint_debug_enabled():
        coupled_jvp = jax.jvp(
            lambda z: coupled_residual(
                z, params, field_parameters, frozen, rcon0, zcon0),
            (z_star,), (rhs,))[1]
        edge_response = jax.tree.map(
            jnp.subtract, coupled_jvp, system.band_operator(rhs))
        print("[vmex adjoint] Schur E row norms:",
              np.asarray(jnp.linalg.norm(system.pack(edge_response), axis=1)))

    boundary_rows = 1
    active_fields = im._active_state_fields(icfg)
    mn = int(mask.R_cos.shape[1])
    packed_mask = np.asarray(system.pack(mask)[-boundary_rows:]).reshape(-1)
    columns = []
    paired = {}
    if bool(icfg.lconm1) and int(icfg.resolution.ntor) > 0:
        positive, negative = im._m1_pair_columns(icfg)
        for pos, neg in zip(positive, negative):
            paired[("Z_sin", int(pos))] = (int(neg), 1.0)
            if bool(icfg.resolution.lasym):
                paired[("Z_cos", int(pos))] = (int(neg), -1.0)
    for radial_row in range(boundary_rows):
        row_start = radial_row * len(active_fields) * mn
        for field_index, name in enumerate(active_fields):
            for mode in range(mn):
                index = row_start + field_index * mn + mode
                if packed_mask[index] == 0.0:
                    continue
                pair = paired.get((name, mode))
                if any(name == pair_name and mode == pair_value[0]
                       for (pair_name, _), pair_value in paired.items()):
                    continue
                column = np.zeros_like(packed_mask)
                if pair is None:
                    column[index] = 1.0
                else:
                    other, sign = pair
                    column[index] = 1.0 / np.sqrt(2.0)
                    column[row_start + field_index * mn + other] = (
                        sign / np.sqrt(2.0))
                columns.append(column)
    edge_basis = jnp.asarray(np.stack(columns, axis=1), dtype=rhs.R_cos.dtype)

    def edge_pack(tree):
        values = system.pack(project(tree))[-boundary_rows:].reshape(-1)
        return edge_basis.T @ values

    def edge_unpack(vector):
        block_size = len(active_fields) * mn
        matrix = jnp.zeros(
            (int(icfg.resolution.ns), block_size), dtype=vector.dtype)
        matrix = matrix.at[-boundary_rows:].set(
            (edge_basis @ vector).reshape((boundary_rows, block_size)))
        return project(system.unpack(matrix))

    def edge_correction(cotangent):
        coupled = coupled_pullback(cotangent)[0]
        bulk = system.band_operator_t(cotangent)
        return jax.tree.map(jnp.subtract, coupled, bulk)

    # The raw radial system is strongly scaled near the magnetic axis. A
    # globally pivoted sparse LU is materially more accurate there than the
    # no-pivot block-Thomas elimination, while retaining O(ns) block storage.
    ns, block_size = np.asarray(system.diagonal).shape[:2]
    bulk_row_scale = np.asarray(system.row_scale)
    bulk_column_scale = np.asarray(system.column_scale)
    previous = np.maximum(np.arange(ns) - 1, 0)
    following = np.minimum(np.arange(ns) + 1, ns - 1)
    lower = (bulk_row_scale[:, :, None] * np.asarray(system.lower)
             * bulk_column_scale[previous, None, :])
    diagonal = (bulk_row_scale[:, :, None] * np.asarray(system.diagonal)
                * bulk_column_scale[:, None, :])
    upper = (bulk_row_scale[:, :, None] * np.asarray(system.upper)
             * bulk_column_scale[following, None, :])
    blocks, indices, indptr = [], [], [0]
    for radial_row in range(ns):
        if radial_row:
            blocks.append(lower[radial_row]); indices.append(radial_row - 1)
        blocks.append(diagonal[radial_row]); indices.append(radial_row)
        if radial_row + 1 < ns:
            blocks.append(upper[radial_row]); indices.append(radial_row + 1)
        indptr.append(len(blocks))
    sparse_bulk = bsr_matrix(
        (np.asarray(blocks), np.asarray(indices), np.asarray(indptr)),
        shape=(ns * block_size, ns * block_size)).tocsc()
    bulk_lu = SpluFactorization(sparse_bulk)

    def sparse_inverse_packed(packed, *, transpose):
        values = np.asarray(packed)
        batched = values.ndim == 3
        values = values if batched else values[None]
        scale_rhs = bulk_column_scale if transpose else bulk_row_scale
        scale_solution = bulk_row_scale if transpose else bulk_column_scale
        flat_rhs = np.moveaxis(values * scale_rhs[None], 0, -1).reshape(
            ns * block_size, -1)
        flat_solution = np.asarray(bulk_lu.solve(
            flat_rhs, trans="T" if transpose else "N"))
        solution = np.moveaxis(
            flat_solution.reshape(ns, block_size, -1), -1, 0)
        solution = solution * scale_solution[None]
        return solution if batched else solution[0]

    def sparse_inverse(tree, *, transpose):
        packed = sparse_inverse_packed(
            np.asarray(system.pack(system.project(tree))),
            transpose=transpose)
        return system.project(system.unpack(jnp.asarray(packed)))

    base = sparse_inverse(rhs, transpose=True)
    if im._adjoint_debug_enabled():
        bulk_defect = jax.tree.map(
            jnp.subtract, rhs, system.band_operator_t(base))
        print("[vmex adjoint] bulk inverse defect row norms:",
              np.asarray(jnp.linalg.norm(system.pack(bulk_defect), axis=1)))

    @jax.jit
    def correction_packed(edge_value):
        return system.pack(system.project(
            edge_correction(edge_unpack(edge_value))))

    @jax.jit
    def solved_to_edge(packed):
        return edge_pack(system.project(system.unpack(packed)))

    def schur_matvec(edge_value):
        packed = correction_packed(jnp.asarray(edge_value, edge_basis.dtype))
        solved = sparse_inverse_packed(np.asarray(packed), transpose=True)
        return np.asarray(edge_value) + np.asarray(
            solved_to_edge(jnp.asarray(solved)))

    edge_rhs = edge_pack(base)
    warm_value = schur_matvec(edge_rhs)
    edge_rhs_np = np.asarray(edge_rhs)
    if im._adjoint_debug_enabled():
        print(f"[vmex adjoint] Schur size={edge_rhs.size} "
              f"rhs={np.linalg.norm(edge_rhs_np):.3e} "
              f"action={np.linalg.norm(np.asarray(warm_value)):.3e} "
              f"finite={np.all(np.isfinite(np.asarray(warm_value)))}")
    calls = 0

    def apply(value):
        nonlocal calls
        calls += 1
        return schur_matvec(value)

    nedge = int(edge_rhs.size)
    matrix = LinearOperator((nedge, nedge), matvec=apply,
                            dtype=edge_rhs_np.dtype)

    if nedge <= 512:
        eye = jnp.eye(nedge, dtype=edge_rhs.dtype)
        packed_corrections = im.chunk_map(
            correction_packed, eye,
            chunk_size=cfg.schur_probe_chunk_size)
        solved = sparse_inverse_packed(
            np.asarray(packed_corrections), transpose=True)
        schur = np.asarray(
            eye + jax.vmap(solved_to_edge)(jnp.asarray(solved))).T
        calls += nedge
        tiny = np.finfo(schur.dtype).tiny
        row_scale = 1.0 / np.maximum(np.max(np.abs(schur), axis=1), tiny)
        row_scaled = row_scale[:, None] * schur
        column_scale = 1.0 / np.maximum(
            np.max(np.abs(row_scaled), axis=0), tiny)
        balanced = row_scaled * column_scale[None, :]
        condition = np.linalg.cond(balanced)
        def solve_reduced(value):
            scaled_rhs = row_scale * value
            if np.isfinite(condition) and condition < 1.0 / np.finfo(
                    schur.dtype).eps:
                balanced_solution = np.linalg.solve(balanced, scaled_rhs)
            else:
                # The edge system can inherit redundant m=1 directions. A
                # rank-revealing solve avoids amplifying them; the exact
                # coupled-residual certificate below remains authoritative.
                balanced_solution = np.linalg.lstsq(
                    balanced, scaled_rhs,
                    rcond=np.finfo(schur.dtype).eps * max(schur.shape))[0]
            return column_scale * balanced_solution

        edge_solution = solve_reduced(edge_rhs_np)
        # Dense iterative refinement is cheap at edge size and recovers the
        # residual digits lost to the raw near-axis scaling.
        for _ in range(3):
            edge_solution += solve_reduced(
                edge_rhs_np - schur @ edge_solution)
        if im._adjoint_debug_enabled():
            print(f"[vmex adjoint] balanced Schur condition={condition:.3e}")
    else:
        edge_solution, _info = gcrotmk(
            matrix, edge_rhs_np, rtol=icfg.adjoint_tol, atol=0.0,
            m=min(icfg.adjoint_gcrot_m, nedge),
            k=min(icfg.adjoint_gcrot_k, nedge), maxiter=icfg.adjoint_maxiter,
        )
    correction = sparse_inverse(
        edge_correction(edge_unpack(jnp.asarray(edge_solution))),
        transpose=True)
    solution = jax.tree.map(jnp.subtract, base, correction)
    defect = jax.tree.map(jnp.subtract, rhs, coupled_pullback(solution)[0])
    residual_norm = float(im._tree_norm(defect))
    rhs_norm = float(im._tree_norm(rhs))
    tolerance = float(im._adjoint_acceptance(icfg, rhs_norm))
    if im._adjoint_debug_enabled():
        print("[vmex adjoint] Schur defect row norms:",
              np.asarray(jnp.linalg.norm(system.pack(defect), axis=1)))
    if not np.isfinite(residual_norm) or residual_norm > tolerance:
        # The raw near-axis scaling can leave the reduced solve a few ulps
        # outside the strict certificate. Continue from it with the original
        # coupled operator; this changes no mathematics and usually needs only
        # a small correction rather than a cold whole-state Krylov search.
        return _host_adjoint(
            coupled_residual, z_star, params, field_parameters, frozen, rcon0,
            zcon0, rhs, icfg, x0=solution)
    return solution


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
    icfg = cfg.implicit
    with im._device_context(icfg):
        params, field_parameters = im._device_pin(
            icfg, (params, field_parameters))
        state, _, _, _, status, fsq, ratio = _callback_status(
            params, field_parameters, cfg)
    return state, status, fsq, ratio


def _solve_status_fwd(params, field_parameters, cfg):
    icfg = cfg.implicit
    with im._device_context(icfg):
        params, field_parameters = im._device_pin(
            icfg, (params, field_parameters))
        state, mask, rcon0, zcon0, status, fsq, ratio = _callback_status(
            params, field_parameters, cfg)
        state, mask, rcon0, zcon0 = im._device_pin(
            icfg, (state, mask, rcon0, zcon0))
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
    *, x0=None,
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
    x0_flat = None if x0 is None else np.asarray(ravel_pytree(x0)[0])
    solution, _info = gcrotmk(
        matrix, np.asarray(rhs_flat), rtol=cfg.adjoint_tol, atol=0.0,
        m=min(cfg.adjoint_gcrot_m, shape[0]),
        k=min(cfg.adjoint_gcrot_k, shape[0]),
        maxiter=cfg.adjoint_maxiter, x0=x0_flat,
    )
    residual_norm = float(np.linalg.norm(np.asarray(rhs_flat) - apply(solution)))
    tolerance = float(im._adjoint_acceptance(
        cfg, np.linalg.norm(np.asarray(rhs_flat))))
    if not np.isfinite(residual_norm) or residual_norm > tolerance:
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
