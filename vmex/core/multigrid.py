"""Coarse -> fine radial interpolation of the spectral state (multigrid).

VMEC2000: ``Sources/TimeStep/interp.f`` — when the ``NS_ARRAY`` ladder moves
to the next radial resolution, the converged spectral coefficients ``xc`` are
interpolated linearly in radius onto the new grid with a VMEC-specific
convention (VMEC++ performs the same linear interpolation of the spectral
coefficients over the ``sqrt(s)``-scaled internal representation):

1. scale the coefficients by ``scalxc`` (``profil3d.f``) so odd-m harmonics
   enter in VMEC's internal ``1/sqrt(s)`` representation — linear in
   ``sqrt(s)`` near the axis;
2. extrapolate odd-m modes to the axis on the *scaled* array,
   ``x(js=1) = 2*x(js=2) - x(js=3)`` (Fortran 1-based);
3. interpolate linearly between the bracketing coarse surfaces using
   ``interp.f``'s ``js1/js2/xint`` uniform-grid construction;
4. divide by ``scalxc`` on the fine grid to return unscaled (internal
   physical) coefficients — the state enters and exits WITHOUT the ``scalxc``
   factor, exactly like the solver's :class:`~vmex.core.solver.SpectralState`;
5. zero odd-m coefficients on the output axis row (edge convention
   ``sqrts(ns) = 1`` is built into ``scalxc``).

The interpolation acts on the m = 1-*constrained* internal coefficients that
:mod:`vmex.core.solver` evolves (``interp.f`` interpolates the internal
``xc``, which is in the constrained basis): every step above is a per-mode
linear map that mixes only coefficients with the same poloidal mode number
``m``, so it commutes with the signed-(m, n) packing and with the m = 1
constraint rotation, and no basis conversion is required.

Math ported from the parity-proven legacy port
``vmex/multigrid.py`` (``interp_vmec_radial_coeffs``).  Pure JAX,
jit-compatible (``ns_coarse``/``ns_fine`` are static shape information), no
host round-trips of traced values.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import replace
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from .device import AUTO, _placement_device, _put_numeric_leaves, device_context
from .errors import MORE_ITER_FLAG, SUCCESSFUL_TERM_FLAG
from .fourier import ModeTable, mode_table
from .input import VmecInput, _vmec_ns_prefix
from .preconditioner_2d import Prec2DConfig
from .solver import (
    SolveResult, SpectralState, _finalize, _prefetch_block_lane,
    _release_used_lane_executables, _result_from_carry, _solve_stage,
    _resolve_use_fft, hot_restart_state, prepare_runtime, resolution_from_input,
    runtime_with_baselines,
)
from .transforms import odd_m_sqrt_s_scaling

__all__ = [
    "interpolate_coefficients", "interpolate_state", "solve_multigrid",
    "solve_free_boundary_multigrid",
]

Array = Any


def _interp_tables(ns_coarse: int, ns_fine: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``interp.f``'s ``js1/js2/xint`` uniform-grid interpolation stencil.

    Static (host, numpy): both grid sizes are shape information, so the
    gather indices and weights are compile-time constants under ``jit``.
    """
    j = np.arange(ns_fine, dtype=np.int64)
    j1 = (j * (ns_coarse - 1)) // (ns_fine - 1)
    j2 = np.minimum(j1 + 1, ns_coarse - 1)
    xint = j.astype(np.float64) * float(ns_coarse - 1) / float(ns_fine - 1) - j1
    xint = np.clip(xint, 0.0, 1.0)
    return j1.astype(np.int32), j2.astype(np.int32), xint


def _scalxc_per_mode(ns: int, m: np.ndarray, dtype) -> Array:
    """``scalxc(js, k)`` per stored mode, shape ``(ns, mnmax)``.

    VMEC2000 ``profil3d.f``: ``1/max(sqrts(js), sqrts(2))`` for odd m, 1 for
    even m, on the uniform full mesh ``s = linspace(0, 1, ns)`` with
    ``sqrts(ns) = 1`` exactly (equals :class:`~vmex.core.setup.RunSetup`
    ``.scalxc`` gathered per mode).
    """
    s = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    mpol = int(np.max(m)) + 1
    table = odd_m_sqrt_s_scaling(s, mpol)              # (ns, mpol)
    return table[:, np.asarray(m, dtype=np.int64)]     # (ns, mnmax)


def interpolate_coefficients(x_coarse: Array, *, m: np.ndarray, ns_fine: int) -> Array:
    """Interpolate one ``(ns_coarse, mnmax)`` coefficient array to ``ns_fine``.

    VMEC2000 ``interp.f`` (see the module docstring for the convention).
    ``x_coarse`` is unscaled (no ``scalxc``); the result is unscaled on the
    fine grid.  ``m`` gives the poloidal mode number of each column (static
    numpy).  ``ns_coarse == ns_fine`` returns the input unchanged (the legacy
    short-circuit); the interior-surface values are reproduced exactly in
    that case by the general path too, since ``xint`` vanishes identically.
    """
    x_coarse = jnp.asarray(x_coarse)
    ns_coarse, mnmax = int(x_coarse.shape[0]), int(x_coarse.shape[1])
    ns_fine = int(ns_fine)
    m = np.asarray(m, dtype=np.int64)
    if m.shape != (mnmax,):
        raise ValueError(f"m has shape {m.shape}, expected ({mnmax},)")
    if ns_coarse <= 0 or ns_fine <= 0:
        return jnp.zeros((max(ns_fine, 0), mnmax), dtype=x_coarse.dtype)
    if ns_coarse == ns_fine:
        return x_coarse
    if ns_fine == 1:
        return x_coarse[:1]
    if ns_coarse == 1:
        return jnp.broadcast_to(x_coarse[:1], (ns_fine, mnmax))

    dtype = x_coarse.dtype
    is_odd = jnp.asarray((m % 2) == 1)

    # 1. enter the scaled (internal odd-m 1/sqrt(s)) representation.
    scal_coarse = _scalxc_per_mode(ns_coarse, m, dtype)
    x_scaled = x_coarse * scal_coarse

    # 2. odd-m axis extrapolation on the scaled array (interp.f):
    #    x(1) = 2*x(2) - x(3)  (Fortran 1-based).
    if ns_coarse >= 3:
        axis_row = jnp.where(is_odd, 2.0 * x_scaled[1] - x_scaled[2], x_scaled[0])
        x_scaled = x_scaled.at[0].set(axis_row)

    # 3. linear interpolation between bracketing coarse surfaces.
    j1, j2, xint = _interp_tables(ns_coarse, ns_fine)
    xint = jnp.asarray(xint, dtype=dtype)
    x_fine_scaled = (1.0 - xint)[:, None] * x_scaled[j1] + xint[:, None] * x_scaled[j2]

    # 4. leave the scaled representation on the fine grid.
    scal_fine = _scalxc_per_mode(ns_fine, m, dtype)
    x_fine = x_fine_scaled / scal_fine

    # 5. zero odd-m modes on the output axis row.
    axis_row = jnp.where(is_odd, jnp.asarray(0.0, dtype=dtype), x_fine[0])
    return x_fine.at[0].set(axis_row)


def interpolate_state(
    state_coarse: SpectralState,
    *,
    ns_fine: int,
    modes: ModeTable,
    ns_coarse: int | None = None,
) -> SpectralState:
    """Interpolate a coarse solver state onto a finer radial grid.

    VMEC2000 ``interp.f``: the multigrid coarse -> fine transfer of ``xc``
    between ``NS_ARRAY`` stages.  ``state_coarse`` is the
    :class:`~vmex.core.solver.SpectralState` of the converged coarse
    stage — signed-(m, n) internal packing, m = 1-constrained, odd-m WITHOUT
    the ``scalxc`` factor — and the result is in the same representation with
    ``ns_fine`` surfaces, ready for :func:`vmex.core.solver.evaluate_forces`
    on the fine :class:`~vmex.core.solver.SolverRuntime`.

    ``modes`` must be the ``mode_table(mpol, ntor)`` shared by both stages
    (multigrid only changes ``ns``).  ``ns_coarse`` is optional (checked
    against the array shapes when given).  Jit-compatible: ``ns_fine`` and
    ``modes`` are static, all array work is traced ``jax.numpy``.
    """
    ns_state = int(jnp.shape(state_coarse.R_cos)[0])
    if ns_coarse is not None and int(ns_coarse) != ns_state:
        raise ValueError(f"ns_coarse={ns_coarse} does not match state ns={ns_state}")
    m = np.asarray(modes.m, dtype=np.int64)
    interp = lambda x: interpolate_coefficients(x, m=m, ns_fine=int(ns_fine))  # noqa: E731
    return SpectralState(
        R_cos=interp(state_coarse.R_cos), R_sin=interp(state_coarse.R_sin),
        Z_cos=interp(state_coarse.Z_cos), Z_sin=interp(state_coarse.Z_sin),
        L_cos=interp(state_coarse.L_cos), L_sin=interp(state_coarse.L_sin),
    )


# ---------------------------------------------------------------------------
# NS_ARRAY ladder driver (runvmec.f)
# ---------------------------------------------------------------------------


#: Cheap proxy keys of rungs a prefetch thread was already launched for in
#: this process.  Repeated warm ladders (optimization/implicit trial solves)
#: then spawn no background threads at all — their executables are already
#: in place.  Entries are purged at the ``release_stage_cache`` point, where
#: the executables they vouch for are dropped.
_PREFETCH_ATTEMPTED: set = set()


def _launch_lane_prefetch(
    inp: VmecInput, *, ns_next: int, ftol_next: float, niter_next: int,
    lconm1: bool, time_step, tcon0, gamma, nstep, precon_type,
    prec2d_threshold, prec2d, use_fft, device,
) -> threading.Thread:
    """Start compiling the NEXT rung's block-lane executable in the background.

    The ladder is known up front, so rung k+1's runtime template is built and
    its CLI block lane AOT-compiled (``jit.lower(...).compile()``) on a
    daemon thread while rung k iterates — XLA compilation releases the GIL,
    so the compile overlaps rung k's device work.  This is cache warming
    only: the runtime built here is discarded (the main loop builds its own,
    structurally identical one), and any failure leaves the rung to compile
    on demand exactly as without prefetch.
    """
    def worker() -> None:
        try:
            resolution = resolution_from_input(inp, ns=ns_next)
            with device_context(device, resolution):
                rt = prepare_runtime(
                    inp, resolution, ftol=float(ftol_next),
                    max_iterations=int(niter_next), lconm1=lconm1,
                    time_step=time_step, tcon0=tcon0, gamma=gamma,
                    nstep=nstep, precon_type=precon_type,
                    prec2d_threshold=prec2d_threshold, prec2d=prec2d,
                    use_fft=use_fft,
                )
                _prefetch_block_lane(rt, use_fft=use_fft)
        except Exception:      # silent fallback: on-demand compilation
            pass

    thread = threading.Thread(
        target=worker, name="vmex-lane-prefetch", daemon=True)
    thread.start()
    return thread


def solve_multigrid(
    inp: VmecInput,
    ns_array=None,
    ftol_array=None,
    niter_array=None,
    *,
    mode: str = "cli",
    lconm1: bool = True,
    verbose: bool = False,
    emit=print,
    initial_state: SpectralState | None = None,
    time_step: float | None = None, tcon0: float | None = None,
    gamma: float | None = None, nstep: int | None = None,
    precon_type: str | None = None,
    prec2d_threshold: float | None = None,
    prec2d: Prec2DConfig | None = None,
    jacobian_retries: int = 2,
    device: Any = AUTO,
    raise_on_max_iterations: bool = True,
    use_fft: bool | None = None,
    release_stage_cache: bool = False,
    prefetch_compile: bool = False,
) -> SolveResult:
    """Fixed-boundary multigrid solve over the ``NS_ARRAY`` ladder.

    VMEC2000 ``Sources/TimeStep/runvmec.f``: for each grid ``igrid`` the
    radial resolution ``nsval = ns_array(igrid)`` is solved with tolerance
    ``ftol_array(igrid)`` and iteration cap ``niter_array(igrid)``; stages
    stop at the first decreasing entry during ``readin.f`` normalization
    (equal entries re-run), and each executed stage after the first starts from the
    ``interp.f`` coarse -> fine interpolation (:func:`interpolate_state`) of
    the previous stage's final state.  Although ``initialize_radial.f`` reads
    ``xstore``, ``allocate_ns.f`` first overwrites it from the old ``xc``;
    therefore the effective VMEC2000 continuation source is the final
    iterate, including when the preceding stage exhausts NITER.  The time
    step resets to the input
    ``DELT`` at every stage, and each stage prints its own ``NS = ...``
    banner (``verbose=True``, ``mode="cli"``).

    ``ns_array/ftol_array/niter_array`` default to the input's ladder; when
    given they are broadcast to a common stage count (shorter ``ftol/niter``
    arrays repeat their last entry).  ``initial_state`` seeds the *first*
    executed stage (hot restart; must match that stage's ``ns``).
    ``precon_type``, ``prec2d_threshold``, and ``prec2d`` override the input's
    optional 2D-preconditioner configuration at every stage.
    ``jacobian_retries`` applies the same bounded best-checkpoint/``DELT``
    recovery as :func:`vmex.core.solver.solve` independently at each stage;
    zero preserves VMEC2000's immediate fatal stop after 75 resets.

    Intermediate stages are allowed to exhaust their iteration cap
    (``more_iter_flag`` — VMEC2000 proceeds to the next grid); any other
    failure raises immediately, and the final stage must converge
    (:class:`~vmex.core.errors.VmecConvergenceError` otherwise), exactly
    like :func:`vmex.core.solver.solve`.  With
    ``raise_on_max_iterations=False`` a final stage that merely hits NITER
    returns its last state instead (``converged=False``, ``ier_flag =
    more_iter_flag``).  This exposes the state to callers; the CLI writes it
    only when ``LFULL3D1OUT=T``, matching the VMEC2000 driver policy.

    Executable reuse: stage runtimes are structural pytrees (solver.py), so
    one XLA executable is compiled per distinct stage structure ``(ns,
    ftol, niter, ...)`` per session, and repeated ladders (parameter scans,
    hot restarts) recompile nothing.  Full radial padding to
    ``max(ns_array)`` — ONE executable for all stages — would need masked
    radial reductions through geometry/fields/forces/preconditioner and is
    not attempted here.

    ``device`` places each stage's jitted lanes (see
    :func:`vmex.core.solver.solve`): an explicit ``"cpu"``/``"gpu"``/
    ``jax.Device`` is always honored; ``"auto"`` (default) applies the
    measured per-stage policy of :mod:`vmex.core.device`, while ``None``
    follows JAX placement.  Auto never overrides an active JAX device or
    platform selection.

    ``use_fft`` has the same measured-hardware default and implicit-AD role as in
    :func:`vmex.core.solver.solve`.

    ``release_stage_cache=True`` clears JAX's in-memory compilation/staging
    caches between distinct radial grids. This lowers peak RSS for a one-shot
    high-resolution ladder while leaving the persistent on-disk cache intact.
    It is false by default so repeated library solves retain warm executables;
    the standalone CLI enables it.

    ``prefetch_compile=True`` (False by default: a library call must not
    spawn background compile threads implicitly — concurrent prefetching
    workers can exhaust CI-runner memory; the standalone CLI enables it)
    overlaps compilation with iteration on cold runs: while rung k
    iterates, a background thread AOT-compiles rung k+1's block-lane
    executable (the ladder is known up front).  Cache
    warming only — results are bit-identical, and any prefetch failure falls
    back silently to on-demand compilation.  The prefetch thread is joined
    *before* the ``release_stage_cache`` point, and the prefetched
    executable is held as a standalone object, so releasing rung k's caches
    cannot evict rung k+1's just-built executable.  Only the ``mode="cli"``
    block lane is prefetched, and equal-structure reruns (same ``ns``,
    ``ftol``, ``niter``) skip the prefetch because the previous rung's
    executable is reused directly.

    Returns the final stage's :class:`~vmex.core.solver.SolveResult`.
    """
    ns_arr = _vmec_ns_prefix(inp.ns_array if ns_array is None else ns_array)
    if ns_arr.size == 0:
        raise ValueError("ns_array has no positive stages")
    n_stages = int(ns_arr.size)

    def _stage_values(values, default, dtype):
        arr = np.atleast_1d(np.asarray(
            default if values is None else values, dtype=dtype)).ravel()
        if arr.size == 0:
            raise ValueError("empty ftol/niter stage array")
        if arr.size < n_stages:  # repeat the last entry (VmecInput convention)
            arr = np.concatenate([arr, np.full(n_stages - arr.size, arr[-1], dtype=dtype)])
        return arr[:n_stages]

    ftol_arr = _stage_values(ftol_array, inp.ftol_array, np.float64)
    niter_arr = _stage_values(niter_array, inp.niter_array, np.int64)

    state: SpectralState | None = initial_state
    first_executed = True
    residual_continuation = None
    carry = rt = None
    prefetch_thread: threading.Thread | None = None
    for igrid in range(n_stages):
        nsval = int(ns_arr[igrid])
        resolution = resolution_from_input(inp, ns=nsval)
        stage_use_fft = _resolve_use_fft(use_fft, device, resolution)
        with device_context(device, resolution):
            rt = prepare_runtime(
                inp, resolution, ftol=float(ftol_arr[igrid]),
                max_iterations=int(niter_arr[igrid]), lconm1=lconm1,
                time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
                precon_type=precon_type, prec2d_threshold=prec2d_threshold,
                prec2d=prec2d, use_fft=stage_use_fft,
            )
        if state is not None and int(state.R_cos.shape[0]) != nsval:
            state = interpolate_state(state, ns_fine=nsval, modes=rt.modes)
        if state is not None:
            if first_executed:
                # user-provided hot-restart seed: adapt to this input's boundary
                state = hot_restart_state(rt, state)
            # funct3d.f: rcon0/zcon0 are set from the state at iter2 == iter1,
            # i.e. from THIS stage's starting state, not the interior guess.
            rt = runtime_with_baselines(rt, state, use_fft=stage_use_fft)
        # Overlapped compilation (see the docstring): start building rung
        # igrid+1's executable now, so it is (mostly) ready when this rung's
        # iterations finish.  Equal-structure next rungs reuse this rung's
        # executable directly, so nothing needs prefetching.
        if (
            prefetch_compile
            and mode == "cli"
            and igrid + 1 < n_stages
            and not jax.config.jax_disable_jit
            and (
                int(ns_arr[igrid + 1]) != nsval
                or float(ftol_arr[igrid + 1]) != float(ftol_arr[igrid])
                or int(niter_arr[igrid + 1]) != int(niter_arr[igrid])
            )
        ):
            ns_next = int(ns_arr[igrid + 1])
            use_fft_next = _resolve_use_fft(
                use_fft, device, resolution_from_input(inp, ns=ns_next))
            attempt_key = (
                ns_next, float(ftol_arr[igrid + 1]), int(niter_arr[igrid + 1]),
                bool(use_fft_next), int(inp.mpol), int(inp.ntor),
                int(inp.nfp), bool(inp.lasym), bool(lconm1), time_step, tcon0,
                gamma, nstep, precon_type, prec2d_threshold, prec2d is None,
                str(device),
            )
            if attempt_key not in _PREFETCH_ATTEMPTED:
                _PREFETCH_ATTEMPTED.add(attempt_key)
                prefetch_thread = _launch_lane_prefetch(
                    inp, ns_next=ns_next,
                    ftol_next=float(ftol_arr[igrid + 1]),
                    niter_next=int(niter_arr[igrid + 1]),
                    lconm1=lconm1, time_step=time_step, tcon0=tcon0,
                    gamma=gamma, nstep=nstep, precon_type=precon_type,
                    prec2d_threshold=prec2d_threshold, prec2d=prec2d,
                    use_fft=use_fft_next,
                    device=device,
                )
        with device_context(device, resolution):
            carry = _solve_stage(
                rt, state, mode=mode, verbose=verbose, emit=emit,
                # initialize_radial.f resets ijacob at every NS stage, so
                # both bad-Jacobian and LMOVE_AXIS first-force retries remain
                # available after interpolation and on hot starts.
                try_axis_reguess=True,
                jacobian_retries=jacobian_retries,
                residual_continuation=residual_continuation,
                use_fft=stage_use_fft,
                emit_legend=first_executed,
            )
        # Join the prefetch before any cache release below: a mid-compile
        # jax.clear_caches() on another thread is the one interaction the
        # overlap must never have.
        if prefetch_thread is not None:
            prefetch_thread.join()
            prefetch_thread = None
        first_executed = False
        ier = int(carry.ier)
        last_stage = not np.any(ns_arr[igrid + 1:] >= nsval)
        if ier not in (SUCCESSFUL_TERM_FLAG, MORE_ITER_FLAG) or (
                last_stage and ier != SUCCESSFUL_TERM_FLAG
                and not (ier == MORE_ITER_FLAG and not raise_on_max_iterations)):
            _finalize(carry, rt)  # raises the typed error for this stage
        # allocate_ns.f saves old xc and copies it into the newly allocated
        # xstore before initialize_radial.f scales/interpolates that array.
        state = carry.state
        # initialize_radial.f resets fsq/iter counters but not the three
        # invariant residual module variables.  The next grid's residue.f90
        # edge and m=1 startup gates therefore see this stage's final values.
        residual_continuation = (carry.fsqr, carry.fsqz, carry.fsql)
        # Release this stage's compiled executables/buffers before the next
        # (larger) radial grid compiles its own: peak RSS is otherwise the sum
        # over the whole ladder instead of the largest single rung.
        future = ns_arr[igrid + 1:]
        executable = future[future >= nsval]
        if (
            release_stage_cache
            and executable.size
            and int(executable[0]) != nsval
        ):
            jax.block_until_ready(carry)
            jax.clear_caches()
            # Companion release for the standalone prefetched executables:
            # consumed ones are dropped with the jit caches; the next rung's
            # just-prefetched executable (not yet used) survives.  The
            # attempt registry is purged with them so a later ladder in this
            # process re-prefetches what was just released.
            _release_used_lane_executables()
            _PREFETCH_ATTEMPTED.clear()
            gc.collect()

    if int(carry.ier) == MORE_ITER_FLAG and not raise_on_max_iterations:
        return _result_from_carry(carry, rt)
    return _finalize(carry, rt)


def solve_free_boundary_multigrid(
    inp: VmecInput,
    ns_array=None,
    ftol_array=None,
    niter_array=None,
    *,
    mgrid_path=None,
    external_field: Any = None,
    verbose: bool = False,
    emit=print,
    initial_state: SpectralState | None = None,
    device: Any = AUTO,
    raise_on_max_iterations: bool = True,
    time_step: float | None = None,
    tcon0: float | None = None,
    gamma: float | None = None,
    nstep: int | None = None,
    lconm1: bool = True,
    precon_type: str | None = None,
    prec2d_threshold: float | None = None,
    prec2d: Prec2DConfig | None = None,
    jacobian_retries: int = 2,
    use_fft: bool | None = None,
    release_stage_cache: bool = False,
    prefetch_compile: bool = False,
) -> SolveResult:
    """Free-boundary solve over the VMEC2000 ``NS_ARRAY`` ladder.

    The plasma continuation follows :func:`solve_multigrid`: each increasing
    grid starts from ``interp.f`` interpolation of the preceding stage's final
    state, equal grids rerun without interpolation, and the ladder stops at
    the first decreasing entry.  The external field is loaded once.  Resolution-
    specific NESTOR bases, Green-function programs, axis-current filament
    tables and traced vacuum loops are selected/rebuilt when the radial grid
    changes; equal-grid reruns reuse their dynamic vacuum cache.

    VMEC2000 carries ``ivac`` between grids.  Consequently, after vacuum has
    activated on a coarse stage, the next stage begins with vacuum active and
    uses the carried coarse-grid boundary pressure on iteration 1, then performs
    a full vacuum update on iteration 2 with new caches (no second turn-on
    banner or soft restart).  The prior stage's ``fsqr/fsqz/fsql`` values are
    retained too, matching ``initialize_radial.f`` and its first-pass
    ``residue.f90`` edge gate.  If an intermediate stage reaches ``NITER``
    before activation, the next stage continues in the pre-activation lane.

    ``initial_state`` hot-starts the first executed stage and is interpolated
    when its radial shape differs from that stage.  It follows reset-file
    semantics: the first stage repeats vacuum activation, while subsequent
    radial stages carry it.
    The fixed-boundary ladder's solver controls (``time_step``, ``tcon0``,
    ``gamma``, ``nstep``, ``lconm1``, device placement, and 2D-preconditioner
    configuration) are accepted and forwarded identically, including bounded
    ``jacobian_retries`` recovery (zero restores the VMEC2000 fatal policy).
    ``device="auto"`` (default) applies the measured policy independently at
    each grid and relocates carried plasma/vacuum arrays when the policy changes;
    ``None`` leaves placement to JAX.
    The final stage's publishable potential and surface fields are retained in
    ``result.vacuum``; internal NESTOR matrix caches are not exposed.

    ``prefetch_compile=True`` (False by default, exactly like
    :func:`solve_multigrid`: a library call must not spawn background
    compile threads implicitly) overlaps compilation with iteration on cold
    runs.  A free-boundary rung selects among several lanes (first force
    pass, pre-activation loop, fused NESTOR update, turn-on pass, batched
    steady-state vacuum loop) and only the activation *iteration* is
    runtime-dependent — every lane *structure* is known once the rung's
    runtimes exist.  Two overlaps therefore apply (see
    ``freeboundary._launch_free_lane_prefetch``): within each rung, the
    post-entry lanes compile in the background while the entry lanes
    compile/iterate on the main thread; across rungs, the next rung's whole
    lane set compiles while the current rung iterates.  Cache warming only —
    results are bit-identical, any prefetch failure falls back silently to
    on-demand compilation, and background threads are joined before the
    ``release_stage_cache`` point exactly like the fixed-boundary ladder.
    """
    if not bool(inp.lfreeb):
        raise ValueError("solve_free_boundary_multigrid requires an LFREEB=T input")

    ns_arr = _vmec_ns_prefix(inp.ns_array if ns_array is None else ns_array)
    if ns_arr.size == 0:
        raise ValueError("ns_array has no positive stages")
    n_stages = int(ns_arr.size)

    def _stage_values(values, default, dtype):
        arr = np.atleast_1d(np.asarray(
            default if values is None else values, dtype=dtype)).ravel()
        if arr.size == 0:
            raise ValueError("empty ftol/niter stage array")
        if arr.size < n_stages:
            arr = np.concatenate([
                arr, np.full(n_stages - arr.size, arr[-1], dtype=dtype),
            ])
        return arr[:n_stages]

    ftol_arr = _stage_values(ftol_array, inp.ftol_array, np.float64)
    niter_arr = _stage_values(niter_array, inp.niter_array, np.int64)

    # Lazy import avoids a module cycle: freeboundary uses SolverRuntime while
    # this module owns the shared interp.f transfer.
    from .freeboundary import (
        _FB_PREFETCH_ATTEMPTED, _external_field_from_input,
        _launch_free_lane_prefetch, _solve_free_boundary_stage,
        free_boundary_resolution,
    )

    if external_field is None:
        external_field = _external_field_from_input(inp, mgrid_path)

    state = initial_state
    interpolation_source = initial_state
    previous_ns = int(initial_state.R_cos.shape[0]) if initial_state is not None else None
    vacuum_continuation = None
    constraint_continuation = None
    residual_continuation = None
    stage_result = None
    prefetch_handle: tuple[threading.Thread, threading.Event] | None = None
    modes = mode_table(int(inp.mpol), int(inp.ntor))
    for igrid in range(n_stages):
        nsval = int(ns_arr[igrid])
        # resolution_from_input plus VMEC2000's mgrid angular-compatibility
        # policy: NZETA must divide the field table's planes per period
        # (automatic NZETA selects the smallest compatible divisor; an
        # explicit incompatible NZETA raises before iteration one).
        resolution = free_boundary_resolution(inp, external_field, ns=nsval)
        same_grid = previous_ns == nsval
        if state is not None and previous_ns != nsval:
            # initialize_radial.f interpolates pxstore only when ns increases;
            # an equal-grid entry returns before allocation/interpolation and
            # therefore reruns the current xc state unchanged.
            state = interpolate_state(
                interpolation_source, ns_fine=nsval, modes=modes)

        # Overlapped compilation across rungs (see the docstring): while this
        # rung iterates, a background thread AOT-compiles rung igrid+1's lane
        # set.  Equal-structure next rungs reuse this rung's executables
        # directly, so nothing needs prefetching.  ``vacuum_on`` reflects the
        # carried IVAC monotonicity: once on, every later rung enters with
        # vacuum active; before that, both entry modes are compiled.
        if (
            prefetch_compile
            and igrid + 1 < n_stages
            and not jax.config.jax_disable_jit
            and (
                int(ns_arr[igrid + 1]) != nsval
                or float(ftol_arr[igrid + 1]) != float(ftol_arr[igrid])
                or int(niter_arr[igrid + 1]) != int(niter_arr[igrid])
            )
        ):
            ns_next = int(ns_arr[igrid + 1])
            handle = _launch_free_lane_prefetch(
                inp, external_field=external_field, ns=ns_next,
                ftol=float(ftol_arr[igrid + 1]),
                max_iterations=int(niter_arr[igrid + 1]),
                time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
                lconm1=lconm1, precon_type=precon_type,
                prec2d_threshold=prec2d_threshold, prec2d=prec2d,
                use_fft=_resolve_use_fft(
                    use_fft, device,
                    free_boundary_resolution(inp, external_field, ns=ns_next)),
                device=device, lmove_axis=None,
                vacuum_on=(True if (vacuum_continuation is not None
                                    and vacuum_continuation.turned_on)
                           else None),
                entry_lanes=True,
            )
            if handle is not None:
                prefetch_handle = handle

        last_stage = not np.any(ns_arr[igrid + 1:] >= nsval)
        stage_device = device
        requested_target = _placement_device(device, resolution)
        # LASYM free-boundary trajectories can bifurcate from accelerator
        # roundoff during vacuum turn-on.  Seed an explicitly requested GPU
        # ladder on its coarsest CPU rung, then transfer the converged branch
        # to every finer GPU rung.  This preserves the requested final
        # placement and matches the VMEC2000 branch; AUTO already makes this
        # measured small-rung choice on its own.
        if (
            bool(inp.lasym)
            and igrid == 0
            and np.any(ns_arr[igrid + 1:] > nsval)
            and requested_target is not None
            and requested_target.platform == "gpu"
        ):
            stage_device = "cpu"
        target = _placement_device(stage_device, resolution)
        external_field = _put_numeric_leaves(external_field, target)
        state = _put_numeric_leaves(state, target)
        constraint_continuation = _put_numeric_leaves(
            constraint_continuation, target)
        if (vacuum_continuation is not None and target is not None
                and any(getattr(vacuum_continuation, name) is not None for name in (
                    "bsqvac", "rbsq", "mode_matrix", "mode_factor",
                    "mode_pivots", "bvec_nonsing", "potvac", "surface_fields",
                ))):
            vacuum_continuation = replace(
                vacuum_continuation,
                bsqvac=_put_numeric_leaves(vacuum_continuation.bsqvac, target),
                rbsq=_put_numeric_leaves(vacuum_continuation.rbsq, target),
                mode_matrix=_put_numeric_leaves(
                    vacuum_continuation.mode_matrix, target),
                mode_factor=_put_numeric_leaves(
                    vacuum_continuation.mode_factor, target),
                mode_pivots=_put_numeric_leaves(
                    vacuum_continuation.mode_pivots, target),
                bvec_nonsing=_put_numeric_leaves(
                    vacuum_continuation.bvec_nonsing, target),
                potvac=_put_numeric_leaves(vacuum_continuation.potvac, target),
                surface_fields=_put_numeric_leaves(
                    vacuum_continuation.surface_fields, target),
            )
        try:
            with device_context(stage_device, resolution):
                stage_result = _solve_free_boundary_stage(
                    inp, external_field=external_field, resolution=resolution,
                    ftol=float(ftol_arr[igrid]),
                    max_iterations=int(niter_arr[igrid]), verbose=verbose,
                    emit=emit,
                    error_on_no_convergence=bool(
                        last_stage and raise_on_max_iterations),
                    initial_state=state,
                    vacuum_continuation=vacuum_continuation,
                    time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
                    lconm1=lconm1,
                    precon_type=precon_type,
                    prec2d_threshold=prec2d_threshold, prec2d=prec2d,
                    jacobian_retries=jacobian_retries,
                    use_fft=_resolve_use_fft(use_fft, stage_device, resolution),
                    constraint_continuation=(
                        constraint_continuation if same_grid else None),
                    reuse_vacuum_cache=bool(same_grid),
                    residual_continuation=residual_continuation,
                    emit_legend=(igrid == 0),
                    prefetch_compile=prefetch_compile,
                    prefetch_device=stage_device,
                )
        except BaseException:
            # A failed stage propagates with no live background compiler:
            # stop the cross-rung prefetch at its next lane boundary and
            # join it before unwinding.
            if prefetch_handle is not None:
                prefetch_handle[1].set()
                prefetch_handle[0].join()
                prefetch_handle = None
            raise
        # Join the cross-rung prefetch before any cache release below: a
        # mid-compile jax.clear_caches() on another thread is the one
        # interaction the overlap must never have.  Unfinished lanes are NOT
        # cancelled — the next rung consumes them either way.
        if prefetch_handle is not None:
            prefetch_handle[0].join()
            prefetch_handle = None
        # allocate_ns.f overwrites xstore from old xc before interp.f, so the
        # effective VMEC2000 source is the stage's final state, not its
        # best-residual restart checkpoint.
        interpolation_source = stage_result.result.state
        state = stage_result.result.state
        previous_ns = nsval
        vacuum_continuation = stage_result.vacuum
        constraint_continuation = (stage_result.rcon0, stage_result.zcon0)
        residual_continuation = (
            stage_result.result.fsqr,
            stage_result.result.fsqz,
            stage_result.result.fsql,
        )
        # Match the fixed-boundary ladder: drop this rung's compiled
        # executables before the next (larger) grid compiles its own, so peak
        # memory tracks the largest single rung rather than the whole ladder.
        # The vacuum/state continuation above is materialised first.
        future = ns_arr[igrid + 1:]
        executable = future[future >= nsval]
        if (
            release_stage_cache
            and executable.size
            and int(executable[0]) != nsval
        ):
            jax.block_until_ready((state, vacuum_continuation))
            jax.clear_caches()
            # Companion release for the standalone prefetched lane
            # executables (see solve_multigrid): consumed ones are dropped,
            # the next rung's just-prefetched set survives, and the attempt
            # registry is purged so a later ladder re-prefetches.
            _release_used_lane_executables()
            _FB_PREFETCH_ATTEMPTED.clear()
            gc.collect()

    if stage_result is None:  # defensive; positive ns_arr guarantees a stage
        raise ValueError("ns_array has no executable stages")
    return stage_result.result
