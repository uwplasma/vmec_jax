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

from typing import Any

import numpy as np

import jax.numpy as jnp

from .device import device_context
from .errors import MORE_ITER_FLAG, SUCCESSFUL_TERM_FLAG
from .fourier import ModeTable
from .input import VmecInput
from .solver import (
    SolveResult, SpectralState, _finalize, _result_from_carry, _solve_stage,
    hot_restart_state, prepare_runtime, resolution_from_input,
    runtime_with_baselines,
)
from .transforms import odd_m_sqrt_s_scaling

__all__ = ["interpolate_coefficients", "interpolate_state", "solve_multigrid"]

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
    device: Any = None,
    raise_on_max_iterations: bool = True,
) -> SolveResult:
    """Fixed-boundary multigrid solve over the ``NS_ARRAY`` ladder.

    VMEC2000 ``Sources/TimeStep/runvmec.f``: for each grid ``igrid`` the
    radial resolution ``nsval = ns_array(igrid)`` is solved with tolerance
    ``ftol_array(igrid)`` and iteration cap ``niter_array(igrid)``; stages
    with ``nsval`` *below* the best resolution reached so far are skipped
    (``IF (nsval < ns_min) CYCLE`` — decreasing entries are ignored, equal
    entries re-run), and each executed stage after the first starts from the
    ``interp.f`` coarse -> fine interpolation (:func:`interpolate_state`) of
    the previous stage's final state.  The time step resets to the input
    ``DELT`` at every stage, and each stage prints its own ``NS = ...``
    banner (``verbose=True``, ``mode="cli"``).

    ``ns_array/ftol_array/niter_array`` default to the input's ladder; when
    given they are broadcast to a common stage count (shorter ``ftol/niter``
    arrays repeat their last entry).  ``initial_state`` seeds the *first*
    executed stage (hot restart; must match that stage's ``ns``).

    Intermediate stages are allowed to exhaust their iteration cap
    (``more_iter_flag`` — VMEC2000 proceeds to the next grid); any other
    failure raises immediately, and the final stage must converge
    (:class:`~vmex.core.errors.VmecConvergenceError` otherwise), exactly
    like :func:`vmex.core.solver.solve`.  With
    ``raise_on_max_iterations=False`` a final stage that merely hits NITER
    returns its last state instead (``converged=False``, ``ier_flag =
    more_iter_flag``) — VMEC2000's own behavior, which writes the
    NITER-exhausted state to the wout file.

    Executable reuse: stage runtimes are structural pytrees (solver.py,
    Phase 2 item (1)), so one XLA executable is compiled per distinct
    stage structure ``(ns, ftol, niter, ...)`` per session, and repeated
    ladders (parameter scans, hot restarts) recompile nothing.  Full radial
    padding to ``max(ns_array)`` — ONE executable for all stages — is the
    recorded follow-up (§7 item 1); it requires masked radial
    reductions through geometry/fields/forces/preconditioner and is not
    attempted here.

    ``device`` places each stage's jitted lanes (see
    :func:`vmex.core.solver.solve`): an explicit ``"cpu"``/``"gpu"``/
    ``jax.Device`` is always honored; ``None`` (default) applies the measured
    per-stage policy of :mod:`vmex.core.device` (small per-iteration work
    solves on CPU) unless the user pinned ``JAX_PLATFORMS``.

    Returns the final stage's :class:`~vmex.core.solver.SolveResult`.
    """
    ns_arr = np.atleast_1d(np.asarray(
        inp.ns_array if ns_array is None else ns_array, dtype=np.int64)).ravel()
    ns_arr = ns_arr[: int(np.argmax(ns_arr <= 0))] if np.any(ns_arr <= 0) else ns_arr
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
    ns_min = 0
    carry = rt = None
    for igrid in range(n_stages):
        nsval = int(ns_arr[igrid])
        if nsval < ns_min:      # runvmec.f: decreasing ns values are skipped
            continue
        ns_min = nsval
        resolution = resolution_from_input(inp, ns=nsval)
        rt = prepare_runtime(
            inp, resolution, ftol=float(ftol_arr[igrid]),
            max_iterations=int(niter_arr[igrid]), lconm1=lconm1,
            time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
        )
        if state is not None and int(state.R_cos.shape[0]) != nsval:
            state = interpolate_state(state, ns_fine=nsval, modes=rt.modes)
        if state is not None:
            if first_executed:
                # user-provided hot-restart seed: adapt to this input's boundary
                state = hot_restart_state(rt, state)
            # funct3d.f: rcon0/zcon0 are set from the state at iter2 == iter1,
            # i.e. from THIS stage's starting state, not the interior guess.
            rt = runtime_with_baselines(rt, state)
        with device_context(device, resolution):
            carry = _solve_stage(
                rt, state, mode=mode, verbose=verbose, emit=emit,
                # the eqsolve.f axis re-guess applies to the fresh interior guess
                try_axis_reguess=first_executed and state is None,
            )
        first_executed = False
        ier = int(carry.ier)
        last_stage = not np.any(ns_arr[igrid + 1:] >= nsval)
        if ier not in (SUCCESSFUL_TERM_FLAG, MORE_ITER_FLAG) or (
                last_stage and ier != SUCCESSFUL_TERM_FLAG
                and not (ier == MORE_ITER_FLAG and not raise_on_max_iterations)):
            _finalize(carry, rt)  # raises the typed error for this stage
        state = carry.state

    if int(carry.ier) == MORE_ITER_FLAG and not raise_on_max_iterations:
        return _result_from_carry(carry, rt)
    return _finalize(carry, rt)
