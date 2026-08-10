# mypy: disable-error-code="call-overload"
"""Single-grid fixed-boundary solve loop: funct3d evaluation + eqsolve iteration.

Wires the ported core modules (:mod:`geometry`, :mod:`fields`, :mod:`forces`,
:mod:`residuals`, :mod:`preconditioner`, :mod:`step`) into one force
evaluation (:func:`evaluate_forces`) and the damped Richardson iteration
(:func:`solve`) with VMEC2000 restart/escalation semantics.

VMEC2000 counterparts
---------------------
- ``Sources/General/funct3d.f`` — one force evaluation: odd-m internal
  synthesis (``totzsps``), constraint baselines ``rcon0/zcon0``, Jacobian
  (early exit on sign change), ``bcovar`` (with the ``ns4 = 25``-iteration
  refresh of the preconditioner / force norms / ``tcon``), constraint force
  (``alias``), MHD forces, ``tomnsps``, ``gc = gc*scalxc`` and ``residue``:
  :func:`evaluate_forces`.
- ``Sources/TimeStep/evolve.f`` — damping window + momentum update (via
  :mod:`vmex.core.step`) and ``TimeStepControl`` (store/restore of the
  best state, residual-growth back-off ``irst = 3``): the iteration body.
- ``Sources/TimeStep/eqsolve.f`` — the outer force-iteration loop: axis
  re-guess after a first bad Jacobian (``ijacob == 0``), time-step resets at
  ``ijacob = 25/50``, abort at ``ijacob >= 75`` (``jac75_flag``), convergence
  when ``fsqr, fsqz, fsql <= ftolv`` simultaneously: :func:`solve`.
- ``Sources/TimeStep/restart.f`` — state restore semantics (zero velocity,
  ``delt`` rescaling) via :mod:`vmex.core.step`.

Execution lanes (§5.3)
------------------------------
Both lanes share one traced single-iteration body:

- ``mode="cli"``: Python ``while`` over a jitted ``lax.scan`` block of
  ``block_size = 10`` iterations, with host residual checks and VMEC2000
  screen printing (``printout.f`` cadence) between blocks;
- ``mode="jit"``: a single ``lax.while_loop`` over the same body (fully
  traced iteration; the setup and error raising remain host code).

The per-iteration ``(fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, ...)`` trajectory
is recorded in a preallocated buffer carried through the loop, so the two
lanes produce identical histories.

Structural executable reuse
---------------------------
:class:`SolverRuntime` is a registered pytree passed as an *argument* to the
module-level jitted lanes :func:`_while_lane`/:func:`_block_lane`.
Array-valued run data
(:class:`~vmex.core.setup.RunSetup`, ``rcon0/zcon0``) are pytree data;
the hashable configuration (:class:`~vmex.core.fourier.Resolution`,
``gamma/tcon0/ftol/max_iterations/time_step0/nstep/jmax``) is pytree meta;
the NumPy mode/trig/weight/gather tables — consumed with ``np.*``/fancy
indexing at trace time — are derived from the meta resolution through the
cached :func:`_static_tables` and never enter the pytree.  Consequence: two
different runtimes with equal structure (e.g. different boundary values at
one resolution, hot restarts, multigrid re-runs) share one XLA executable
per lane.  :func:`solve` additionally accepts ``initial_state`` (hot
restart), and the stage machinery is factored into :func:`_solve_stage`/
:func:`_finalize` for reuse by :func:`vmex.core.multigrid.solve_multigrid`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass, replace
from typing import Any, Callable

import functools
import platform

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)  # float64 mandatory (§7.7)


def _harden_compilation_cache() -> None:
    """Idempotent persistent compile-cache setup at ``core.solver`` import.

    Launching Python from a cwd containing the repo checkout can resolve
    ``vmex`` as a *namespace* package, so ``vmex/__init__.py`` — which
    configures the persistent XLA compilation cache — never runs and every
    ``solve()`` pays a full recompile.  This module always executes on any
    core solve path, so the cache policy is re-applied here: warn on the
    shadowed import, then configure the ``_compat`` cache defaults *only*
    when neither the user (``JAX_COMPILATION_CACHE_DIR`` env / an explicit
    ``jax.config.update``) nor ``vmex/__init__`` already set a cache
    directory.
    """
    import os
    import sys
    import warnings

    top = sys.modules.get(__name__.partition(".")[0])
    if top is not None and getattr(top, "__file__", None) is None:
        warnings.warn(
            "vmex was imported as a namespace package (vmex.__file__ is "
            "missing) — its __init__.py never ran.  This usually means the "
            "current working directory shadows the installed package (e.g. "
            "running Python from the directory containing the vmex "
            "checkout).  Change directory or fix sys.path; package-level "
            "defaults such as the persistent compilation cache are otherwise "
            "skipped.",
            RuntimeWarning,
            stacklevel=3,
        )
    try:
        current = jax.config.jax_compilation_cache_dir
    except AttributeError:  # pragma: no cover - very old jax
        return
    if current:  # user env/jax.config or vmex/__init__ already set it
        return
    try:
        from .._compat import _configure_compilation_cache, _default_compilation_cache_dir
    except Exception:  # pragma: no cover - core used standalone
        return
    cache_dir = _default_compilation_cache_dir()
    if cache_dir is None:  # policy says no cache (e.g. CPU-only, not forced)
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:  # pragma: no cover - unwritable cache location
        return
    _configure_compilation_cache(jax, cache_dir)


_harden_compilation_cache()

import jax.numpy as jnp
from jax import lax

from .device import (
    AUTO,
    GPU_MAX_SPECTRAL_MODES,
    _placement_device,
    _put_numeric_leaves,
    device_context,
)
from .errors import (
    AXIS_REGUESS_FLAG, BAD_JACOBIAN_FLAG, JAC75_FLAG, MISC_ERROR_FLAG, MORE_ITER_FLAG,
    NONFINITE_FLAG, NORM_TERM_FLAG, SUCCESSFUL_TERM_FLAG,
    VmecConvergenceError, VmecJacobianError, VmecNumericalError,
    WERROR_MESSAGES,
)
from .fields import (
    constraint_scaling, energies_and_force_norms,
    force_balance_preconditioner_factor, magnetic_fields, metric_elements,
    preconditioned_force_norm, radial_force_balance_error,
)
from .forces import (
    apply_m1_force_balance, constraint_force, mhd_forces,
    spectral_mhd_forces,
)
from .fourier import ModeTable, Resolution, TrigTables, mode_table, trig_tables
from .geometry import apply_lambda_axis_closure, half_mesh_jacobian, real_space_geometry
from .input import VmecInput
from .preconditioner import (
    RadialPreconditionerCoefficients, ScalforPivotData, TridiagonalMatrices,
    angular_integration_weights, lamcal, precondn, scalfor_matrices,
    scalfor_pivot_data,
)
from .preconditioner_2d import Prec2DConfig, newton_direction
from .printing import (
    FORCE_ITERATIONS_BANNER, compile_notice, improved_axis_block,
    screen_header, screen_line, stage_banner,
)
from .residuals import (
    ForceResiduals, PreconditionedResiduals, apply_lambda_preconditioner,
    apply_radial_preconditioner, edge_force_condition, force_residuals,
    m1_constrained_to_physical, m1_residue_rotation, m1_zero_condition,
    preconditioned_residuals, scale_m1_preconditioner_rhs, scalxc_scale_force,
    zero_m1_z_force,
)
from .setup import RunSetup, guess_axis, interior_guess, run_setup
from .step import (
    DAMPING_CAP, GROWTH_BACKOFF_DIVISOR, GROWTH_LIMIT, GROWTH_MIN_ITERATIONS,
    JACOBIAN_RESET_FACTOR, NDAMP, RESTART_GROWTH, RESTART_JACOBIAN, STEP_OK,
    StepControl, damping_coefficients, momentum_update,
)
from .transforms import (
    SpectralForce, physical_to_internal_scale,
    register_pytree_dataclass as _register,
)

__all__ = [
    "NS4", "SpectralState", "PreconditionerCache", "FunctDiagnostics",
    "SolverRuntime", "SolveResult", "Prec2DConfig", "resolution_from_input",
    "prepare_runtime", "evaluate_forces", "solve",
    "hot_restart_state", "runtime_with_baselines",
]

Array = Any

#: bcovar.f ``ns4``: preconditioner / force-norm / tcon refresh cadence.
NS4 = 25

#: Number of iterations per jitted CLI-lane scan block (§5.3).
BLOCK_SIZE = 10

#: Trajectory buffer columns:
#: iter, fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, r00, z00, wmhd, delt.
_TRAJ_COLS = 11

_TWO_PI_SQ = (2.0 * np.pi) ** 2


def _select(mask: Array, new, old):
    """Elementwise pytree select ``mask ? new : old`` (scalar traced mask)."""
    return jax.tree.map(lambda a, b: jnp.where(mask, a, b), new, old)


# -- State containers ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralState:
    """Evolved spectral state in the signed-(m, n) internal packing.

    VMEC2000 ``xc``: internal normalization (``mscale*nscale`` divided out),
    m = 1-*constrained* basis (``residue.f90`` ``lconm1``), odd-m WITHOUT the
    ``scalxc`` factor (``totzsps`` applies it during synthesis).  For
    stellarator-symmetric runs ``R_sin/Z_cos/L_cos`` are identically zero.
    All arrays have shape ``(ns, mnmax)``.
    """

    R_cos: Array
    R_sin: Array
    Z_cos: Array
    Z_sin: Array
    L_cos: Array
    L_sin: Array


@dataclass(frozen=True)
class PreconditionerCache:
    """Quantities refreshed on the ``ns4 = 25`` cadence (VMEC2000 ``bcovar.f``).

    ``tcon`` (constraint scaling), the residual norms ``fnorm/fnormL/fnorm1``,
    the :func:`precondn` coefficients and assembled :func:`scalfor_matrices`
    for the R and Z force families (plus their band-frozen
    :func:`scalfor_pivot_data` checked-solve masks), the ``LFORBAL``
    ``rzu_fac/rru_fac`` factors, and the ``lamcal`` diagonal ``faclam``.
    """

    tcon: Array
    fnorm: Array
    fnormL: Array
    fnorm1: Array
    coefficients_R: RadialPreconditionerCoefficients
    coefficients_Z: RadialPreconditionerCoefficients
    force_balance_R: Array
    force_balance_Z: Array
    matrices_R: TridiagonalMatrices
    matrices_Z: TridiagonalMatrices
    faclam: Array
    pivot_R: ScalforPivotData
    pivot_Z: ScalforPivotData


@dataclass(frozen=True)
class ForcePipelineHealth:
    """Finite-stage flags for the ``forces -> residue`` pipeline.

    These booleans deliberately retain no equilibrium values.  They let the
    privacy-preserving diagnostic distinguish real-space force assembly,
    Fourier projection, residual scaling, and preconditioning failures.
    """

    real_space_finite: Array
    spectral_finite: Array
    scaled_finite: Array
    rhs_finite: Array
    radial_preconditioner_safe: Array
    radial_solve_finite: Array
    preconditioned_finite: Array


@dataclass(frozen=True)
class NumericalHealth:
    """Privacy-safe first-force-pass health flags (no input-derived values)."""

    geometry_finite: Array
    jacobian_nonzero: Array
    fields_finite: Array
    volume_valid: Array
    energy_scale_valid: Array
    force_norm_finite: Array
    lambda_scale_valid: Array
    lambda_norm_finite: Array
    pipeline: ForcePipelineHealth
    raw_r_sum_finite: Array
    raw_z_sum_finite: Array
    raw_lambda_sum_finite: Array
    raw_r_residual_finite: Array
    raw_z_residual_finite: Array
    raw_lambda_residual_finite: Array
    cache_finite: Array


@dataclass(frozen=True)
class FunctDiagnostics:
    """Per-evaluation diagnostics returned by :func:`evaluate_forces`.

    ``preconditioned`` are the residue.f90 ``fsqr1/fsqz1/fsql1``; ``wb/wp``
    the MHD energies (``bcovar.f``); ``r00/z00`` the axis position at
    ``theta = zeta = 0`` (``funct3d.f``); ``jacobian_sign_changed`` the
    ``jacobian.f`` ``irst = 2`` flag; ``cache`` the (possibly refreshed)
    :class:`PreconditionerCache` to carry into the next iteration.
    """

    preconditioned: PreconditionedResiduals
    wb: Array
    wp: Array
    r00: Array
    z00: Array
    jacobian_sign_changed: Array
    cache: PreconditionerCache
    health: NumericalHealth


@dataclass(frozen=True)
class _EvalResult:
    """Internal bundle: force + residuals + diagnostics for the loop body."""

    gc: SpectralState
    residuals: ForceResiduals
    pre: PreconditionedResiduals
    wb: Array
    wp: Array
    r00: Array
    z00: Array
    jacobian_sign_changed: Array
    cache: PreconditionerCache
    health: NumericalHealth


@dataclass(frozen=True)
class _LoopCarry:
    """Traced carry of the eqsolve iteration loop (one per solve).

    VMEC2000 names: ``time_step = delt``, ``inv_tau = otau`` (the ndamp
    damping window), ``fsq`` (previous ``fsqr1+fsqz1+fsql1``), ``res0/res1``
    (best preconditioned/raw totals, ``TimeStepControl``), ``iteration =
    iter2``, ``iter1`` (last restart), ``ijacob``; ``trajectory`` is the
    ``(max_iterations, _TRAJ_COLS)`` per-iteration history buffer.
    """

    state: SpectralState; xcdot: SpectralState; xstore: SpectralState
    cache: PreconditionerCache
    time_step: Array; inv_tau: Array; fsq: Array; res0: Array; res1: Array
    fsqr: Array; fsqz: Array; fsql: Array
    fsqr1: Array; fsqz1: Array; fsql1: Array
    wb: Array; wp: Array; r00: Array
    iteration: Array; iter1: Array; ijacob: Array
    done: Array; ier: Array
    trajectory: Array


for _cls in (
    SpectralState, PreconditionerCache, ForcePipelineHealth, NumericalHealth,
    FunctDiagnostics, _EvalResult, _LoopCarry,
):
    _register(_cls)


# -- Runtime (per-solve context: a jit-passable pytree) ---------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _static_tables(resolution: Resolution):
    """Cached trace-time-static tables derived from a :class:`Resolution`.

    Returns ``(modes, trig, weights, gather_m, gather_n, cos_w, sin_w)`` —
    all host NumPy objects consumed *at trace time* (fancy-index tables,
    trig-table matmuls inside :mod:`vmex.core.transforms`, angular
    weights).  The lru_cache guarantees that two runtimes built from the same
    ``Resolution`` share the *identical* table objects, so the runtime pytree
    treedefs compare equal and ``jax.jit`` reuses one executable across
    solves with different boundary/profile values.
    """
    modes = mode_table(resolution.mpol, resolution.ntor)
    trig = trig_tables(resolution)
    weights = angular_integration_weights(
        ntheta=resolution.ntheta, nzeta=resolution.nzeta, lasym=resolution.lasym
    )
    gather_m, gather_n, cos_w, sin_w = _force_gather_tables(modes)
    return modes, trig, weights, gather_m, gather_n, cos_w, sin_w


@dataclass(frozen=True, eq=False)
class SolverRuntime:
    """Per-solve context, registered as a JAX pytree.

    Passed *as an argument* to the module-level jitted lanes
    (:func:`_while_lane` / :func:`_block_lane`):

    - **data fields** (traced): the :class:`RunSetup` arrays (profiles,
      radial grids, boundary, initial state — everything that changes with
      boundary values) and the fixed-boundary constraint baselines
      ``rcon0/zcon0`` (``funct3d.f`` — constant per run because the edge
      spectral row never evolves, but boundary-value dependent);
    - **meta fields** (static, hashable): the :class:`Resolution` plus the
      scalar configuration (``gamma`` is consumed concretely by
      ``fields.magnetic_fields``; ``max_iterations`` sizes the trajectory
      buffer; the rest are loop-control constants).

    The NumPy mode/trig/weight/gather tables are *derived* from the meta
    ``resolution`` via the cached :func:`_static_tables` (exposed as
    properties), so they never enter the pytree: they are used with
    ``np.*``/fancy indexing at trace time and must stay concrete.  Two
    runtimes with equal structure (same resolution + scalars, same array
    shapes) therefore share one XLA executable per lane.
    """

    resolution: Resolution
    setup: RunSetup
    rcon0: Array; zcon0: Array
    gamma: float; tcon0: float; ftol: float
    max_iterations: int; time_step0: float; nstep: int
    jmax: int                           # evolved radial rows (fixed: ns-1)
    lforbal: bool = False               # tomnsp_mod.f m=1,n=0 force replacement
    lmove_axis: bool = True             # funct3d.f first-force irst=4 path

    # -- free-boundary seam (core/freeboundary.py; funct3d.f/forces.f) ------
    # lfreeb=True selects the vacuum-coupled lane: the edge row is evolved
    # (jmax = ns is passed by the free-boundary driver), tomnsps keeps the
    # edge row, and forces.f's `armn(ns) += zu0*rbsq` / `azmn(ns) -= ru0*rbsq`
    # edge terms are injected with rbsq = (bsqvac + presf_ns)*R(edge)*ohs.
    # `bsqvac_edge` is the NESTOR 0.5*|B|^2 on the (ntheta3, nzeta) boundary
    # grid, refreshed by the host driver between iterations (data field: no
    # retrace).  `presf_ns_scale` is the static funct3d.f edge-pressure
    # factor pmass(1)/pmass(hs*(ns-1.5)) applied to pres(ns).
    lfreeb: bool = False
    bsqvac_edge: Array | None = None
    presf_ns_scale: Array | None = None

    # -- 2D block preconditioner seam (core/preconditioner_2d.py; precon2d.f) -
    # None disables it (the default 1D-only path is then byte-identical); a
    # Prec2DConfig switches on the matrix-free Newton step (finest grid, once
    # fsqr+fsqz+fsql < threshold).  Static meta: the branch is only traced when
    # present, so a NONE run never compiles the GMRES/HVP graph.
    prec2d: Any = None

    # -- trace-time-static tables, derived from the meta resolution ---------
    @property
    def modes(self) -> ModeTable:
        return _static_tables(self.resolution)[0]

    @property
    def trig(self) -> TrigTables:
        return _static_tables(self.resolution)[1]

    @property
    def weights(self) -> np.ndarray:    # angular integration weights (wint)
        return _static_tables(self.resolution)[2]

    # force-block gather tables (static, from _force_gather_tables):
    # cos_w weights the (cc, ss) blocks; sin_w the (sc, cs) blocks.
    @property
    def gather_m(self) -> np.ndarray:
        return _static_tables(self.resolution)[3]

    @property
    def gather_n(self) -> np.ndarray:
        return _static_tables(self.resolution)[4]

    @property
    def cos_w(self) -> np.ndarray:
        return _static_tables(self.resolution)[5]

    @property
    def sin_w(self) -> np.ndarray:
        return _static_tables(self.resolution)[6]


_register(SolverRuntime, meta=(
    "resolution", "gamma", "tcon0", "ftol", "max_iterations", "time_step0",
    "nstep", "jmax", "lforbal", "lmove_axis",
    "lfreeb", "prec2d",
))


def _force_gather_tables(modes: ModeTable) -> tuple[np.ndarray, ...]:
    """Static tables mapping VMEC ``(m, n>=0)`` force blocks to signed modes.

    Inverse of the signed -> block relations used throughout VMEC (see the
    parity-proven ``vmex.kernels.parity`` maps): for the cos family
    (``cc/ss`` blocks) and the sin family (``sc/cs`` blocks),

    - ``n = 0``:            signed = A                      -> (1, 0)
    - ``n > 0, m > 0``:     cos: (cc + ss)/2 -> (1/2, 1/2); sin: (sc - cs)/2
    - ``n > 0, m = 0``:     cos: cc -> (1, 0);              sin: -cs -> (0, -1)
    - ``n < 0`` (m > 0):    cos: (cc - ss)/2;               sin: (sc + cs)/2
    """
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    cos_w = np.zeros((m.size, 2))
    sin_w = np.zeros((m.size, 2))
    for k in range(m.size):
        if n[k] == 0:
            cos_w[k] = (1.0, 0.0); sin_w[k] = (1.0, 0.0)
        elif n[k] > 0 and m[k] > 0:
            cos_w[k] = (0.5, 0.5); sin_w[k] = (0.5, -0.5)
        elif n[k] > 0:  # m == 0
            cos_w[k] = (1.0, 0.0); sin_w[k] = (0.0, -1.0)
        else:           # n < 0 (m > 0 only in the VMEC mode table)
            cos_w[k] = (0.5, -0.5); sin_w[k] = (0.5, 0.5)
    return m.astype(np.int32), np.abs(n).astype(np.int32), cos_w, sin_w


def _blocks_to_signed(
    block_a, block_b, rt: SolverRuntime, w: np.ndarray, ns: int
) -> Array:
    """Gather one ``(ns, mpol, ntor+1)`` block pair into signed coefficients."""
    dtype = rt.setup.s_full.dtype
    if block_a is None:
        return jnp.zeros((ns, rt.modes.mnmax), dtype=dtype)
    a = jnp.asarray(block_a)[:, rt.gather_m, rt.gather_n]
    out = jnp.asarray(w[:, 0], dtype=a.dtype)[None, :] * a
    if block_b is not None:
        b = jnp.asarray(block_b)[:, rt.gather_m, rt.gather_n]
        out = out + jnp.asarray(w[:, 1], dtype=b.dtype)[None, :] * b
    return out


def _force_to_state(force: SpectralForce, rt: SolverRuntime) -> SpectralState:
    """Preconditioned :class:`SpectralForce` -> signed increments (``gc``).

    VMEC evolves ``xc`` in the same block layout as ``gc``; the signed
    packing is an equivalent linear reparametrization, so the momentum step
    commutes with this conversion.
    """
    ns = int(jnp.asarray(force.force_R_cc).shape[0])
    return SpectralState(
        R_cos=_blocks_to_signed(force.force_R_cc, force.force_R_ss, rt, rt.cos_w, ns),
        R_sin=_blocks_to_signed(force.force_R_sc, force.force_R_cs, rt, rt.sin_w, ns),
        Z_cos=_blocks_to_signed(force.force_Z_cc, force.force_Z_ss, rt, rt.cos_w, ns),
        Z_sin=_blocks_to_signed(force.force_Z_sc, force.force_Z_cs, rt, rt.sin_w, ns),
        L_cos=_blocks_to_signed(force.force_lambda_cc, force.force_lambda_ss, rt, rt.cos_w, ns),
        L_sin=_blocks_to_signed(force.force_lambda_sc, force.force_lambda_cs, rt, rt.sin_w, ns),
    )


# -- Runtime construction --------------------------------------------------------------------------------------------


def resolution_from_input(inp: VmecInput, *, ns: int | None = None) -> Resolution:
    """Resolve the internal grid sizes from an input (``read_indata.f``).

    ``ntheta <= 0 -> 2*mpol + 6``; ``nzeta = 1`` for axisymmetric inputs with
    ``nzeta = 0``, else ``nzeta <= 0 -> 2*ntor + 4``.  ``ns`` defaults to the
    first ``ns_array`` stage (single-grid solve).
    """
    ntheta = int(inp.ntheta) if int(inp.ntheta) > 0 else 2 * int(inp.mpol) + 6
    nzeta = int(inp.nzeta)
    if int(inp.ntor) == 0 and nzeta == 0:
        nzeta = 1
    elif nzeta <= 0:
        nzeta = 2 * int(inp.ntor) + 4
    return Resolution(
        mpol=int(inp.mpol), ntor=int(inp.ntor), ntheta=ntheta, nzeta=nzeta,
        nfp=int(inp.nfp), lasym=bool(inp.lasym), ns=int(inp.ns_array[0]) if ns is None else int(ns),
    )


def _initial_state(setup: RunSetup) -> SpectralState:
    return SpectralState(
        R_cos=setup.R_cos, R_sin=setup.R_sin, Z_cos=setup.Z_cos,
        Z_sin=setup.Z_sin, L_cos=setup.lambda_cos, L_sin=setup.lambda_sin,
    )


def _physical_coefficients(state: SpectralState, *, modes, lthreed, lasym, lconm1):
    """Undo the m=1 constraint; return the geometry-synthesis coefficients."""
    R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos,
        modes=modes, lthreed=lthreed, lasym=lasym, lconm1=lconm1,
    )
    return R_cos, R_sin, Z_cos, Z_sin


def _geometry(
    state: SpectralState, rt: SolverRuntime, *, use_fft: bool = False
):
    """Constrained state -> physical coefficients + real-space geometry."""
    setup = rt.setup
    R_cos, R_sin, Z_cos, Z_sin = _physical_coefficients(
        state, modes=rt.modes, lthreed=setup.lthreed, lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    lambda_sin = apply_lambda_axis_closure(state.L_sin, modes=rt.modes, ntor=rt.resolution.ntor)
    geometry = real_space_geometry(
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        lambda_cos=state.L_cos, lambda_sin=lambda_sin,
        modes=rt.modes, trig=rt.trig, s=setup.s_full, use_fft=use_fft,
    )
    return (R_cos, R_sin, Z_cos, Z_sin), geometry


def _resolve_prec2d(
    source: VmecInput | RunSetup,
    prec2d: Prec2DConfig | None,
    precon_type: str | None,
    prec2d_threshold: float | None,
    *,
    finest: bool = True,
) -> Prec2DConfig | None:
    """Resolve the 2D-preconditioner config (``precon2d.f`` / ``evolve.f``).

    An explicit ``prec2d`` config wins (used verbatim); otherwise it is built
    from ``precon_type``/``prec2d_threshold`` (input defaults, overridable).
    ``NONE`` and VMEC2000's ``DEFAULT`` select the parity 1-D path.  ``GMRES``
    selects VMEX's documented matrix-free JAX/SOLVAX Newton--GMRES path.
    VMEC2000's distinct ``CG``, ``GMRESR`` and ``TFQMR`` algorithms are
    rejected rather than silently aliased to GMRES.
    """
    if prec2d is not None:
        return prec2d
    if isinstance(source, VmecInput):
        pt = source.precon_type if precon_type is None else precon_type
        thr = source.prec2d_threshold if prec2d_threshold is None else prec2d_threshold
    else:
        pt = "NONE" if precon_type is None else precon_type
        thr = 1e-30 if prec2d_threshold is None else prec2d_threshold
    pt_normalized = str(pt).strip().upper()
    if pt_normalized in {"NONE", "DEFAULT"}:
        return None
    if pt_normalized != "GMRES":
        raise NotImplementedError(
            "PRECON_TYPE must be NONE/DEFAULT or GMRES; VMEC2000's distinct "
            "CG, GMRESR, and TFQMR evolution algorithms are not implemented"
        )
    return Prec2DConfig(threshold=float(thr), finest=bool(finest))


def prepare_runtime(
    source: VmecInput | RunSetup,
    resolution: Resolution | None = None,
    *,
    ftol: float | None = None, max_iterations: int | None = None,
    time_step: float | None = None, tcon0: float | None = None,
    gamma: float | None = None, nstep: int | None = None,
    lconm1: bool = True, lforbal: bool | None = None,
    setup: RunSetup | None = None,
    precon_type: str | None = None, prec2d_threshold: float | None = None,
    prec2d: Prec2DConfig | None = None,
    use_fft: bool = False,
) -> SolverRuntime:
    """Build the static solver context from an input file or a RunSetup.

    Defaults come from the input (``delt``, ``tcon0``, ``gamma``, ``nstep``,
    ``ftol_array(1)``, ``niter_array(1)``); explicit keywords override.  The
    fixed-boundary constraint baselines ``rcon0/zcon0 = s * rcon(ns)``
    (``funct3d.f``) are computed once here from the initial state — the edge
    spectral row never evolves in fixed-boundary mode, so they are constants
    of the run.

    ``lforbal`` overrides the input's non-variational ``(m=1,n=0)`` force
    replacement when supplied (a prebuilt :class:`RunSetup` defaults it to
    false).  ``precon_type``/``prec2d_threshold`` (or an explicit ``prec2d``
    :class:`~vmex.core.preconditioner_2d.Prec2DConfig`) switch on the
    optional 2D block preconditioner (``precon2d.f``); ``None``/``"NONE"``
    (the default) leaves the VMEC2000 1-D radial/lambda path unchanged.
    """
    if isinstance(source, RunSetup):
        if resolution is None:
            raise ValueError("prepare_runtime(RunSetup) requires a Resolution")
        setup = source
        defaults = dict(ftol=1e-10, niter=100, delt=1.0, tcon0=1.0, gamma=0.0,
                        nstep=200, lforbal=False, lmove_axis=True)
    else:
        inp = source
        if resolution is None:
            resolution = resolution_from_input(inp)
        if setup is None:
            # VMEC2000 profil3d.f starts from the axis coefficients exactly as
            # supplied, including an all-zero/missing axis.  Its eqsolve.f
            # driver calls guess_axis once only after the first bad-Jacobian
            # or high-force (irst=4) pass.  The setup helper retains its
            # explicit legacy inference option for geometry-only callers, but
            # a production solve must not pre-guess here and then guess a
            # second time in _solve_stage.
            setup = run_setup(
                inp, resolution, lconm1=lconm1,
                infer_axis_if_missing=False, use_fft=use_fft,
            )
        defaults = dict(ftol=float(inp.ftol_array[0]), niter=int(inp.niter_array[0]),
                        delt=float(inp.delt), tcon0=float(inp.tcon0),
                        gamma=float(inp.gamma), nstep=int(inp.nstep),
                        lforbal=bool(inp.lforbal),
                        lmove_axis=bool(inp.lmove_axis))
    requested_iterations = int(
        defaults["niter"] if max_iterations is None else max_iterations
    )
    # readin.f can leave -1 in a partially assigned NITER_ARRAY.  eqsolve.f
    # still performs its first evolve/funct3d pass before ``iter2 >= niter``
    # terminates the stage, so the effective trajectory capacity is one.
    effective_iterations = max(1, requested_iterations)
    rt = SolverRuntime(
        resolution=resolution, setup=setup,
        gamma=float(defaults["gamma"] if gamma is None else gamma),
        tcon0=float(defaults["tcon0"] if tcon0 is None else tcon0),
        ftol=float(defaults["ftol"] if ftol is None else ftol),
        max_iterations=effective_iterations,
        time_step0=float(defaults["delt"] if time_step is None else time_step),
        nstep=int(defaults["nstep"] if nstep is None else nstep),
        jmax=int(resolution.ns) - 1,
        lforbal=bool(defaults["lforbal"] if lforbal is None else lforbal),
        lmove_axis=bool(defaults["lmove_axis"]),
        rcon0=jnp.zeros(()), zcon0=jnp.zeros(()),  # placeholder, replaced below
        prec2d=_resolve_prec2d(source, prec2d, precon_type, prec2d_threshold),
    )
    rcon0, zcon0 = _constraint_baselines(
        _initial_state(setup), rt, use_fft=use_fft
    )
    return replace(rt, rcon0=rcon0, zcon0=zcon0)


def hot_restart_state(rt: SolverRuntime, state: SpectralState) -> SpectralState:
    """Adapt a previous solve's state to this runtime's boundary (hot restart).

    In fixed-boundary mode the R/Z edge row never evolves, so restarting from
    ``state`` unchanged would silently re-solve the OLD boundary.  Replacing
    only the edge row injects a discontinuous shear between the last two
    surfaces (measured: initial ``fsqr ~ 0.5`` on cth, i.e. *worse* than the
    fresh interior guess).  Instead the boundary delta is spread smoothly
    into the volume with the ``profil3d.f`` interior-guess radial profile —
    ``sqrts**m`` for ``m > 0``, linear in ``s`` for ``m = 0`` (the axis is
    held fixed) — which lands the edge row exactly on the new boundary
    (``facj(ns) = 1``) and keeps the interior near equilibrium (measured:
    initial ``fsqr ~ 4e-6`` for a 1% ``RBC(0,1)`` perturbation on cth).
    Lambda is carried over unchanged.
    """
    setup = rt.setup
    s = jnp.asarray(setup.s_full)
    dtype = s.dtype
    m = np.asarray(rt.modes.m, dtype=int)
    rho = jnp.sqrt(jnp.maximum(s, 0.0)).at[-1].set(jnp.asarray(1.0, dtype=dtype))
    m_j = jnp.asarray(m)[None, :]
    facj = jnp.where(m_j > 0, rho[:, None] ** m_j,
                     s[:, None] * jnp.ones((1, m.size), dtype=dtype))

    def shift(old, new_boundary):
        old = jnp.asarray(old, dtype=dtype)
        delta = jnp.asarray(new_boundary, dtype=dtype)[-1] - old[-1]
        return old + facj * delta[None, :]

    return replace(
        state,
        R_cos=shift(state.R_cos, setup.R_cos),
        R_sin=shift(state.R_sin, setup.R_sin),
        Z_cos=shift(state.Z_cos, setup.Z_cos),
        Z_sin=shift(state.Z_sin, setup.Z_sin),
    )


def runtime_with_baselines(
    rt: SolverRuntime, state: SpectralState, *, use_fft: bool = False
) -> SolverRuntime:
    """Rebind ``rcon0/zcon0`` to a new starting state (``funct3d.f``).

    VMEC2000 sets the constraint baselines from the *current* state whenever
    ``iter2 == iter1`` — i.e. at the start of every grid.  A runtime from
    :func:`prepare_runtime` carries baselines for the ``profil3d.f`` interior
    guess; callers that start from a different state (hot restart via
    ``solve(initial_state=...)``, multigrid stages starting from the
    ``interp.f`` interpolant) must rebind them, or the constraint force —
    and hence the converged equilibrium — is subtly wrong (observed as a
    ~1e-8 relative ``wb`` shift on the nfp4_QH ladder).
    """
    rcon0, zcon0 = _constraint_baselines(state, rt, use_fft=use_fft)
    return replace(rt, rcon0=rcon0, zcon0=zcon0)


def _constraint_baselines(
    state: SpectralState, rt: SolverRuntime, *, use_fft: bool = False
):
    """One-time ``rcon0/zcon0 = s * rcon(ns)`` (funct3d.f, iter2 == iter1)."""
    (R_cos, R_sin, Z_cos, Z_sin), geometry = _geometry(
        state, rt, use_fft=use_fft
    )
    ns = int(rt.setup.s_full.shape[0])
    _, _, _, rcon0, zcon0 = constraint_force(
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        geometry=geometry, modes=rt.modes, trig=rt.trig, s=rt.setup.s_full,
        tcon=jnp.zeros((ns,), dtype=rt.setup.s_full.dtype),
        signgs=rt.setup.signgs,
        use_fft=use_fft,
    )
    return rcon0, zcon0


def _zero_cache(rt: SolverRuntime) -> PreconditionerCache:
    """Zero-filled cache (shapes only; iteration 1 always refreshes it)."""
    res = rt.resolution
    ns, mpol, nr = res.ns, res.mpol, res.ntor + 1
    dtype = rt.setup.s_full.dtype
    z = lambda shape: jnp.zeros(shape, dtype=dtype)  # noqa: E731
    coeffs = RadialPreconditionerCoefficients(
        axm=z((ns - 1, 2)), axd=z((ns, 2)), bxm=z((ns - 1, 2)), bxd=z((ns, 2)),
        cx=z((ns,)),
    )
    mats = TridiagonalMatrices(ax=z((rt.jmax, mpol, nr)), bx=z((rt.jmax, mpol, nr)),
                               dx=z((rt.jmax, mpol, nr)))
    zb = lambda shape: jnp.zeros(shape, dtype=bool)  # noqa: E731
    pivot = ScalforPivotData(m0_ok=zb((nr,)), m0_scale=z((nr,)),
                             m1_ok=zb((mpol - 1, nr)), m1_scale=z((mpol - 1, nr)))
    return PreconditionerCache(
        tcon=z((ns,)), fnorm=z(()), fnormL=z(()), fnorm1=z(()),
        coefficients_R=coeffs, coefficients_Z=coeffs,
        force_balance_R=z((ns,)), force_balance_Z=z((ns,)),
        matrices_R=mats, matrices_Z=mats, faclam=z((ns, mpol, nr)),
        pivot_R=pivot, pivot_Z=pivot,
    )


# -- One funct3d pass ------------------------------------------------------------------------------------------------


def _all_finite(value) -> Array:
    """Return one traced boolean for all numeric leaves in ``value``."""
    finite = jnp.asarray(True)

    def visit(item):
        nonlocal finite
        if item is None:
            return
        if is_dataclass(item):
            for field in dataclass_fields(item):
                visit(getattr(item, field.name))
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
            return
        finite = finite & jnp.all(jnp.isfinite(jnp.asarray(item)))

    visit(value)
    return finite


def _force_pipeline(
    *,
    geometry, jacobian, metrics, fields,
    R_cos: Array, R_sin: Array, Z_cos: Array, Z_sin: Array,
    cache: PreconditionerCache, rt: SolverRuntime,
    iteration: Array, fsqz_previous: Array,
    collect_health: bool = False,
    use_fft: bool = False,
) -> tuple[Any, Any, ForcePipelineHealth]:
    """MHD forces -> residue.f90 chain -> scalfor/faclam preconditioning.

    The ``funct3d.f`` -> ``forces.f`` -> ``tomnsps`` -> ``residue.f90`` segment
    that both :func:`_evaluate` (the iteration body) and
    :func:`_preconditioned_force_signed` (the 2D-preconditioner force map)
    consume, factored out so they cannot drift.  Returns ``(scaled,
    preconditioned)``: the ``scalxc``-scaled force (input to the invariant
    residuals ``getfsq``) and the 1D-preconditioned force (input to
    ``fsqr1/fsqz1`` and the update direction ``gc``).  All preconditioner
    matrices come from the frozen ``cache`` (``ns4`` cadence).  The third
    return value contains finite/not-finite stage flags only.
    """
    setup = rt.setup
    res = rt.resolution
    s = setup.s_full
    hs = setup.hs

    forces = mhd_forces(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields,
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        modes=rt.modes, trig=rt.trig, s=s, phipf=setup.phipf,
        tcon=cache.tcon, signgs=setup.signgs, rcon0=rt.rcon0, zcon0=rt.zcon0,
        use_fft=use_fft,
    )
    if rt.lfreeb:
        # forces.f (ivac >= 1): vacuum-pressure edge force.  funct3d.f builds
        # rbsq = (bsqvac + presf_ns) * (r1(ns,0) + r1(ns,1)) * ohs with
        # presf_ns = [pmass(1)/pmass(hs*(ns-1.5))] * pres(ns), then forces.f
        # adds zu0*rbsq to the even AND odd armn edge rows (and -ru0*rbsq to
        # azmn).  sqrts(ns) = 1, so even+odd sums are the physical edge row.
        presf_ns = jnp.asarray(rt.presf_ns_scale) * fields.pressure[-1]
        gcon_edge = jnp.asarray(rt.bsqvac_edge) + presf_ns
        r1_edge = geometry.R_even[-1] + geometry.R_odd[-1]
        rbsq = gcon_edge * r1_edge / hs
        ru0, zu0 = geometry.theta_derivatives_full(s)
        forces = replace(
            forces,
            force_R_even=jnp.asarray(forces.force_R_even).at[-1].add(zu0[-1] * rbsq),
            force_R_odd=jnp.asarray(forces.force_R_odd).at[-1].add(zu0[-1] * rbsq),
            force_Z_even=jnp.asarray(forces.force_Z_even).at[-1].add(-ru0[-1] * rbsq),
            force_Z_odd=jnp.asarray(forces.force_Z_odd).at[-1].add(-ru0[-1] * rbsq),
        )
    passing = jnp.asarray(True)
    real_space_finite = _all_finite(forces) if collect_health else passing
    spectral = spectral_mhd_forces(
        forces, mpol=res.mpol, ntor=res.ntor, trig=rt.trig,
        include_edge=bool(rt.lfreeb),
    )
    if rt.lforbal:
        equif = radial_force_balance_error(
            fields=fields,
            phipf=setup.phipf,
            trig=rt.trig,
            s=s,
            signgs=setup.signgs,
        )
        spectral = apply_m1_force_balance(
            spectral,
            equif=equif,
            factor_R=cache.force_balance_R,
            factor_Z=cache.force_balance_Z,
        )

    # -- residue.f90 chain ---------------------------------------------------
    rotated = m1_residue_rotation(spectral, lconm1=setup.lconm1)
    zero_gate = m1_zero_condition(fsqz_previous=fsqz_previous, iterations_since_restart=iteration)
    released = zero_m1_z_force(rotated, zero_gate)
    spectral_finite = (
        _all_finite((spectral, rotated, released)) if collect_health else passing
    )
    scaled = scalxc_scale_force(released, s=s)
    scaled_finite = _all_finite(scaled) if collect_health else passing
    if setup.lthreed or setup.lasym:
        rhs = scale_m1_preconditioner_rhs(
            scaled, coefficients_R=cache.coefficients_R,
            coefficients_Z=cache.coefficients_Z, lconm1=setup.lconm1,
        )
    else:
        rhs = scaled
    rhs_finite = _all_finite(rhs) if collect_health else passing
    solved, radial_safe = apply_radial_preconditioner(
        rhs,
        matrices_R=cache.matrices_R,
        matrices_Z=cache.matrices_Z,
        jmax=rt.jmax,
        return_safe=True,
        pivot_data_R=cache.pivot_R,
        pivot_data_Z=cache.pivot_Z,
    )
    radial_solve_finite = _all_finite(solved) if collect_health else passing
    preconditioned = apply_lambda_preconditioner(solved, cache.faclam)
    health = ForcePipelineHealth(
        real_space_finite=real_space_finite,
        spectral_finite=spectral_finite,
        scaled_finite=scaled_finite,
        rhs_finite=rhs_finite,
        radial_preconditioner_safe=radial_safe,
        radial_solve_finite=radial_solve_finite,
        preconditioned_finite=(
            _all_finite(preconditioned) if collect_health else passing
        ),
    )
    return scaled, preconditioned, health


def _preconditioned_force_signed(
    state: SpectralState, cache: PreconditionerCache, rt: SolverRuntime,
    *, iteration: Array, fsqz_previous: Array,
) -> SpectralState:
    """1D-preconditioned force map ``state -> gc`` at a frozen ``cache``.

    Reproduces the ``gc`` that :func:`_evaluate` returns (same
    :func:`_force_pipeline`, same signed packing via :func:`_force_to_state`),
    but as a standalone pure function of ``state`` with everything else frozen:
    exactly the map whose Jacobian the 2D block preconditioner needs.
    :func:`jax.jvp` of this function is the exact block-tridiagonal
    Hessian-vector product (VMEC2000 ``precon2d.f`` ``compute_blocks``, without
    the finite-difference jogs).  Fixed-boundary only (``lfreeb=False``); the
    frozen ``iteration``/``fsqz_previous`` fix the ``residue.f90`` m=1 release
    gate so the map matches the ``gc`` produced at the linearization point.
    """
    setup = rt.setup
    s = setup.s_full
    (R_cos, R_sin, Z_cos, Z_sin), geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    _, preconditioned, _health = _force_pipeline(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields,
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        cache=cache, rt=rt, iteration=iteration, fsqz_previous=fsqz_previous,
    )
    return _force_to_state(preconditioned, rt)


_ALL_CHANNELS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def _newton_step(
    rt: SolverRuntime, state: SpectralState, gc_signed: SpectralState,
    cache: PreconditionerCache, iteration: Array, fsqz_previous: Array,
    fsq_raw: Array, gate: Array,
) -> tuple[SpectralState, Array]:
    """2D-preconditioner Newton direction, gated on the activation predicate.

    Returns ``(direction, active)``.  When ``active`` (finest grid,
    ``fsq_raw < threshold``, ``iteration >= start_iteration``, and the base
    ``gate``), ``direction`` is the damped-Newton replacement for the 1D force
    ``gc_signed``: it solves ``J delta = -gc_signed`` with matrix-free GMRES
    (``J = d(preconditioned force)/d(state)`` at ``state``, exact HVP via
    ``jax.jvp``), so ``state += cfg.step * delta`` is the block-preconditioned
    update (``precon2d.f`` ``block_precond``: ``gc <- -H^{-1} gc``).  Otherwise
    ``direction = gc_signed`` and ``active = False``.  The linear solve runs
    only over the non-trivial spectral channels (symmetric: R_cos/Z_sin/L_sin).
    Only reached when ``rt.prec2d is not None`` (the branch is otherwise never
    traced, keeping the 1D-only path byte-identical).
    """
    cfg = rt.prec2d
    if not cfg.finest:  # non-finest multigrid stage: never activate (static)
        return gc_signed, jnp.zeros((), dtype=bool)

    active = gate & (fsq_raw < cfg.threshold) & (iteration >= cfg.start_iteration)
    channels = _ALL_CHANNELS if rt.setup.lasym else ("R_cos", "Z_sin", "L_sin")
    inactive = tuple(c for c in _ALL_CHANNELS if c not in channels)

    def to_full(reduced: dict) -> SpectralState:
        # inactive (identically-zero) channels are frozen constants of the HVP
        full = dict(reduced)
        for c in inactive:
            full[c] = getattr(state, c) * 0.0
        return SpectralState(**{c: full[c] for c in _ALL_CHANNELS})

    def g_reduced(reduced: dict) -> dict:
        gc_full = _preconditioned_force_signed(
            to_full(reduced), cache, rt, iteration=iteration, fsqz_previous=fsqz_previous,
        )
        return {c: getattr(gc_full, c) for c in channels}

    x0 = {c: getattr(state, c) for c in channels}
    rhs = {c: -getattr(gc_signed, c) for c in channels}  # solve J delta = -g

    def do_newton(_):
        delta, _sol = newton_direction(g_reduced, x0, rhs, cfg)
        return SpectralState(**{
            c: (delta[c] if c in channels else getattr(gc_signed, c))
            for c in _ALL_CHANNELS
        })

    direction = lax.cond(active, do_newton, lambda _: gc_signed, operand=None)
    return direction, active


def _evaluate(
    state: SpectralState, cache: PreconditionerCache, iteration: Array,
    iter_last_reset: Array, fsqz_previous: Array, rt: SolverRuntime,
    fsq_rz_previous: Array | None = None,
    *, collect_health: bool = False,
    use_fft: bool = False,
    synthesis: tuple | None = None,
) -> _EvalResult:
    """One funct3d.f pass (fixed boundary), pure and jit-friendly.

    Order of operations follows ``funct3d.f``: synthesis -> Jacobian (sign
    flag carried, never branched on) -> ``bcovar`` fields/energies with the
    ``mod(iter2-iter1, ns4) == 0`` cache refresh -> constraint force with the
    cached ``tcon`` and static ``rcon0/zcon0`` -> ``forces`` -> ``tomnsps`` ->
    ``gc*scalxc`` -> ``residue`` (m=1 rotation + conditional Z-force zeroing,
    ``getfsq``, ``scale_m1`` + ``scalfor`` + ``faclam`` preconditioning,
    ``fsqr1/fsqz1/fsql1``).  On a Jacobian sign change VMEC skips everything
    past the Jacobian; here the computation proceeds (traced) and the caller
    discards results, while the cache refresh is suppressed exactly.

    ``synthesis`` optionally supplies the pass's already-computed
    ``((R_cos, R_sin, Z_cos, Z_sin), geometry, jacobian)`` for ``state``
    (funct3d.f computes them ONCE per pass and shares them with the vacuum
    gate).  It must have been produced by the same ``_geometry``/
    ``half_mesh_jacobian`` calls this function would make — the traced
    free-boundary lane hoists exactly those, so results are bit-identical.
    """
    setup = rt.setup
    res = rt.resolution
    s = setup.s_full
    ns = int(s.shape[0])
    hs = setup.hs

    if synthesis is None:
        (R_cos, R_sin, Z_cos, Z_sin), geometry = _geometry(
            state, rt, use_fft=use_fft
        )
        jacobian = half_mesh_jacobian(geometry, s=s)
    else:
        (R_cos, R_sin, Z_cos, Z_sin), geometry, jacobian = synthesis
    jac_changed = jacobian.jacobian_sign_changed

    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    energies = energies_and_force_norms(
        jacobian=jacobian, metrics=metrics, fields=fields, trig=rt.trig, s=s, signgs=setup.signgs,
    )

    # -- ns4-cadence refresh candidates (bcovar.f) --------------------------
    tcon_new = constraint_scaling(
        tcon0=rt.tcon0, geometry=geometry, jacobian=jacobian,
        total_pressure=fields.total_pressure, trig=rt.trig, s=s,
    )
    common = dict(
        r12_half=jacobian.r12[1:], bsq_half=fields.total_pressure[1:],
        bsupv_half=fields.bsupv[1:], sqrt_g_half=jacobian.sqrt_g[1:],
        angular_weight=rt.weights, delta_s=hs, ns=ns,
    )
    coefficients_R = precondn(
        dxds_half=jacobian.dZ_ds[1:], dxdu_half=jacobian.zu12[1:],
        dxdu_even_full=geometry.dZ_dtheta_even, dxdu_odd_full=geometry.dZ_dtheta_odd,
        x_odd_full=geometry.Z_odd, **common,
    )
    coefficients_Z = precondn(
        dxds_half=jacobian.dR_ds[1:], dxdu_half=jacobian.ru12[1:],
        dxdu_even_full=geometry.dR_dtheta_even, dxdu_odd_full=geometry.dR_dtheta_odd,
        x_odd_full=geometry.R_odd, **common,
    )
    if rt.lforbal and res.mpol >= 2:
        force_balance_R = force_balance_preconditioner_factor(
            axd_odd=coefficients_R.axd[:, 1],
            dxdu_half=jacobian.zu12,
            trig_multiplier=np.asarray(rt.trig.cosmu)[:, 1],
            jacobian=jacobian,
            fields=fields,
            trig=rt.trig,
            s=s,
            signgs=setup.signgs,
        )
        force_balance_Z = force_balance_preconditioner_factor(
            axd_odd=coefficients_Z.axd[:, 1],
            dxdu_half=jacobian.ru12,
            trig_multiplier=-np.asarray(rt.trig.sinmu)[:, 1],
            jacobian=jacobian,
            fields=fields,
            trig=rt.trig,
            s=s,
            signgs=setup.signgs,
        )
    else:
        force_balance_R = jnp.zeros((ns,), dtype=s.dtype)
        force_balance_Z = jnp.zeros((ns,), dtype=s.dtype)
    # jmax follows scalfor.f: ns-1 fixed boundary (rt.jmax default), ns once
    # the vacuum field is on (free-boundary lane) — activates the
    # EDGE_PEDESTAL / ZC(0,0) edge stiffening inside scalfor_matrices.
    mat_kwargs = dict(delta_s=hs, mpol=res.mpol, ntor=res.ntor, nfp=res.nfp, ns=ns,
                      jmax=int(rt.jmax))
    matrices_R = scalfor_matrices(coefficients_R, stabilize_edge_zc00=False, **mat_kwargs)
    matrices_Z = scalfor_matrices(coefficients_Z, stabilize_edge_zc00=True, **mat_kwargs)
    faclam_new = lamcal(
        guu_half=metrics.guu, guv_half=metrics.guv, gvv_half=metrics.gvv,
        sqrt_g_half=jacobian.sqrt_g, lamscale=fields.lamscale,
        angular_weight=rt.weights, mpol=res.mpol, ntor=res.ntor, nfp=res.nfp,
        lthreed=setup.lthreed,
    )
    fnorm1_new = preconditioned_force_norm(
        R_cos=state.R_cos, Z_sin=state.Z_sin, modes=rt.modes,
        R_sin=state.R_sin if setup.lasym else None,
        Z_cos=state.Z_cos if setup.lasym else None,
    )
    fresh = PreconditionerCache(
        tcon=tcon_new, fnorm=energies.fnorm, fnormL=energies.fnormL,
        fnorm1=fnorm1_new, coefficients_R=coefficients_R,
        coefficients_Z=coefficients_Z,
        force_balance_R=force_balance_R,
        force_balance_Z=force_balance_Z, matrices_R=matrices_R,
        matrices_Z=matrices_Z, faclam=faclam_new,
        pivot_R=scalfor_pivot_data(matrices_R, jmax=int(rt.jmax)),
        pivot_Z=scalfor_pivot_data(matrices_Z, jmax=int(rt.jmax)),
    )
    refresh = (((iteration - iter_last_reset) % NS4) == 0) & (~jac_changed)
    cache = _select(refresh, fresh, cache)

    # -- constraint + MHD forces + residue.f90 preconditioning chain --------
    # The mhd_forces -> tomnsps -> residue -> scalfor/faclam pipeline is shared
    # verbatim with the 2D-preconditioner force map (:func:`_force_pipeline`)
    # so the two can never drift; ``scaled`` feeds the invariant residuals,
    # ``preconditioned`` the fsqr1/fsqz1 residuals and the update force ``gc``.
    scaled, preconditioned, pipeline_health = _force_pipeline(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields,
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        cache=cache, rt=rt, iteration=iteration, fsqz_previous=fsqz_previous,
        collect_health=collect_health, use_fft=use_fft,
    )
    if rt.lfreeb:
        # residue.f90 medge rule: the edge rows join fsqr/fsqz only when
        # iter2 - iter1 < 50 and the previous fsqr+fsqz < 1e-6 (traced
        # condition; both variants are cheap sums).
        res_int = force_residuals(scaled, fnorm=cache.fnorm, fnormL=cache.fnormL,
                                  r1=energies.r1, include_edge=False)
        res_edge = force_residuals(scaled, fnorm=cache.fnorm, fnormL=cache.fnormL,
                                   r1=energies.r1, include_edge=True)
        fsq_rz_prev = jnp.asarray(1.0, dtype=s.dtype) if fsq_rz_previous is None \
            else jnp.asarray(fsq_rz_previous)
        medge = edge_force_condition(
            fsq_rz_previous=fsq_rz_prev,
            iterations_since_restart=iteration - iter_last_reset,
            free_boundary=True,
        )
        residuals = _select(medge, res_edge, res_int)
    else:
        residuals = force_residuals(scaled, fnorm=cache.fnorm, fnormL=cache.fnormL, r1=energies.r1, include_edge=False)
    pre = preconditioned_residuals(preconditioned, fnorm1=cache.fnorm1, delta_s=hs)

    passing = jnp.asarray(True)
    if collect_health:
        sqrt_g_inner = jnp.asarray(jacobian.sqrt_g)[1:]
        volume = jnp.asarray(energies.volume)
        energy_scale = jnp.maximum(
            jnp.asarray(energies.wb), jnp.asarray(energies.wp)
        )
        lamscale = jnp.asarray(fields.lamscale)
        health = NumericalHealth(
            geometry_finite=_all_finite((geometry, jacobian, metrics)),
            jacobian_nonzero=jnp.all(sqrt_g_inner != 0.0),
            fields_finite=_all_finite(fields),
            volume_valid=jnp.isfinite(volume) & (volume != 0.0),
            energy_scale_valid=(
                jnp.isfinite(energy_scale) & (energy_scale != 0.0)
            ),
            force_norm_finite=jnp.isfinite(jnp.asarray(energies.fnorm)),
            lambda_scale_valid=jnp.isfinite(lamscale) & (lamscale != 0.0),
            lambda_norm_finite=jnp.isfinite(jnp.asarray(energies.fnormL)),
            pipeline=pipeline_health,
            raw_r_sum_finite=jnp.isfinite(jnp.asarray(residuals.gcr2)),
            raw_z_sum_finite=jnp.isfinite(jnp.asarray(residuals.gcz2)),
            raw_lambda_sum_finite=jnp.isfinite(jnp.asarray(residuals.gcl2)),
            raw_r_residual_finite=jnp.isfinite(jnp.asarray(residuals.fsqr)),
            raw_z_residual_finite=jnp.isfinite(jnp.asarray(residuals.fsqz)),
            raw_lambda_residual_finite=jnp.isfinite(jnp.asarray(residuals.fsql)),
            cache_finite=_all_finite(cache),
        )
    else:
        health = NumericalHealth(
            geometry_finite=passing, jacobian_nonzero=passing,
            fields_finite=passing, volume_valid=passing,
            energy_scale_valid=passing, force_norm_finite=passing,
            lambda_scale_valid=passing, lambda_norm_finite=passing,
            pipeline=pipeline_health,
            raw_r_sum_finite=passing, raw_z_sum_finite=passing,
            raw_lambda_sum_finite=passing, raw_r_residual_finite=passing,
            raw_z_residual_finite=passing, raw_lambda_residual_finite=passing,
            cache_finite=passing,
        )

    sqrt_s0 = jnp.sqrt(jnp.maximum(s[0], 0.0))
    r00 = geometry.R_even[0, 0, 0] + sqrt_s0 * geometry.R_odd[0, 0, 0]
    z00 = geometry.Z_even[0, 0, 0] + sqrt_s0 * geometry.Z_odd[0, 0, 0]

    return _EvalResult(
        gc=_force_to_state(preconditioned, rt),
        residuals=residuals, pre=pre, wb=energies.wb, wp=energies.wp,
        r00=r00, z00=z00, jacobian_sign_changed=jac_changed, cache=cache,
        health=health,
    )


def evaluate_forces(
    state: SpectralState,
    runtime: SolverRuntime,
    *,
    cache: PreconditionerCache | None = None,
    iteration: int = 1, iter_last_reset: int = 1, fsqz_previous: float = 1.0,
) -> tuple[SpectralState, ForceResiduals, FunctDiagnostics]:
    """One funct3d pass: preconditioned force ``gc``, residuals, diagnostics.

    VMEC2000: ``funct3d.f`` + ``residue.f90``.  ``cache=None`` forces a
    preconditioner/norm/tcon refresh (as VMEC does whenever
    ``iter2 == iter1``); pass the returned ``diagnostics.cache`` back in to
    reproduce the ``ns4 = 25`` cadence.  The returned ``gc`` is in the same
    signed spectral packing as ``state`` and feeds
    :func:`vmex.core.step.momentum_update` directly.
    """
    if cache is None:
        cache = _zero_cache(runtime)
        iter_last_reset = iteration  # force refresh
    result = _evaluate(
        state, cache, jnp.asarray(iteration), jnp.asarray(iter_last_reset),
        jnp.asarray(fsqz_previous), runtime, collect_health=True,
    )
    diagnostics = FunctDiagnostics(
        preconditioned=result.pre,
        wb=result.wb, wp=result.wp, r00=result.r00, z00=result.z00,
        jacobian_sign_changed=result.jacobian_sign_changed, cache=result.cache,
        health=result.health,
    )
    return result.gc, result.residuals, diagnostics


def _evaluation_is_finite(result: _EvalResult) -> Array:
    """Whether every iteration-driving output of one ``funct3d`` pass is finite."""
    leaves = jax.tree.leaves((
        result.gc, result.residuals, result.pre, result.wb, result.wp,
        result.r00, result.z00, result.cache,
    ))
    finite = jnp.asarray(True)
    for leaf in leaves:
        finite = finite & jnp.all(jnp.isfinite(jnp.asarray(leaf)))
    return finite


def reguess_initial_axis(
    rt: SolverRuntime, state: SpectralState, *, use_fft: bool = False
) -> tuple[SolverRuntime, SpectralState, tuple[Array, Array, Array, Array]]:
    """Apply VMEC2000's first-pass magnetic-axis retry.

    Shared by fixed- and free-boundary drivers for a first bad Jacobian and
    for ``LMOVE_AXIS=T`` with a first raw-force sum above ``1e2``.  Besides
    rebuilding the ``profil3d`` state, this updates the setup's axis arrays
    and rebinds the constraint baselines to that state
    (``funct3d.f: iter2 == iter1``).
    """
    setup = rt.setup
    _, geometry = _geometry(state, rt, use_fft=use_fft)
    axis = guess_axis(
        geometry, s=setup.s_full, trig=rt.trig, signgs=setup.signgs
    )
    arrays = interior_guess(
        boundary_R_cos=setup.boundary_R_cos,
        boundary_R_sin=setup.boundary_R_sin,
        boundary_Z_cos=setup.boundary_Z_cos,
        boundary_Z_sin=setup.boundary_Z_sin,
        raxis_c=axis[0], raxis_s=axis[1], zaxis_c=axis[2], zaxis_s=axis[3],
        modes=rt.modes, trig=rt.trig, s=setup.s_full,
    )
    new_setup = replace(
        setup,
        raxis_c=axis[0], raxis_s=axis[1], zaxis_c=axis[2], zaxis_s=axis[3],
        R_cos=arrays[0], R_sin=arrays[1], Z_cos=arrays[2], Z_sin=arrays[3],
        lambda_cos=arrays[4], lambda_sin=arrays[5],
    )
    state = SpectralState(
        R_cos=arrays[0], R_sin=arrays[1], Z_cos=arrays[2], Z_sin=arrays[3],
        L_cos=arrays[4], L_sin=arrays[5],
    )
    rt = runtime_with_baselines(
        replace(rt, setup=new_setup), state, use_fft=use_fft
    )
    return rt, state, axis


# -- Iteration body (evolve.f + TimeStepControl + the eqsolve.f checks) ----------------------------------------------


def _make_body(
    rt: SolverRuntime,
    *,
    evaluation_state: SpectralState | None = None,
    use_fft: bool = False,
    evaluation_synthesis: tuple | None = None,
) -> Callable[[_LoopCarry], _LoopCarry]:
    """Build the traced single-iteration body shared by both lanes.

    ``evaluation_state`` is the narrowly scoped VMEC2000 free-boundary
    turn-on seam.  ``funct3d.f`` computes geometry and vacuum pressure, then
    performs the ``irst=2`` soft restart, and finally evaluates forces using
    the already-computed geometry.  Supplying the pre-restart state here
    reproduces that one pass while momentum still evolves ``carry.state``
    (the restored best state).  Normal iterations leave it as ``None``.

    ``evaluation_synthesis`` optionally hands the first funct3d pass its
    already-computed ``((R_cos, R_sin, Z_cos, Z_sin), geometry, jacobian)``
    (the ``_evaluate`` ``synthesis`` seam) — the free-boundary steady lane
    synthesizes geometry once per pass, feeds its Jacobian sign to the
    vacuum-source gate, and reuses it here, exactly as ``funct3d.f`` computes
    the Jacobian ONCE before ``bcovar``/IVAC0 and shares it with the force
    build.  It must correspond to the state the first pass evaluates
    (``carry.state``, or ``evaluation_state`` when given); the restart
    re-evaluation always re-synthesizes at the restored state.
    """
    ftol = rt.ftol
    max_iter = rt.max_iterations
    gamma = rt.gamma

    def body(carry: _LoopCarry) -> _LoopCarry:
        it = carry.iteration
        running = jnp.logical_not(carry.done)

        # ---- funct3d (evolve.f) -------------------------------------------
        state_e1 = carry.state if evaluation_state is None else evaluation_state
        e1 = _evaluate(state_e1, carry.cache, it, carry.iter1, carry.fsqz, rt,
                       carry.fsqr + carry.fsqz, use_fft=use_fft,
                       synthesis=evaluation_synthesis)
        jac1 = e1.jacobian_sign_changed
        nonfinite1 = (~jac1) & (~_evaluation_is_finite(e1))
        # On irst=2 funct3d skips residue: the module residuals stay stale.
        fsqr_c = jnp.where(jac1, carry.fsqr, e1.residuals.fsqr)
        fsqz_c = jnp.where(jac1, carry.fsqz, e1.residuals.fsqz)
        fsql_c = jnp.where(jac1, carry.fsql, e1.residuals.fsql)
        fsq0 = fsqr_c + fsqz_c + fsql_c

        converged = (~jac1) & (fsqr_c <= ftol) & (fsqz_c <= ftol) & (fsql_c <= ftol)
        bad_init = jac1 & (it == 1)
        # funct3d.f/eqsolve.f: LMOVE_AXIS=T and a finite first raw-force sum
        # above 1e2 set irst=4 and return to guess_axis before evolving xc.
        # ijacob=0 makes this a single retry, exactly like the Fortran guard.
        axis_reguess = (
            bool(rt.lmove_axis)
            & (~jac1)
            & (~nonfinite1)
            & (it == 1)
            & (carry.ijacob == 0)
            & (rt.resolution.ns >= 3)
            & (fsq0 > 1.0e2)
        )
        # Unlike a bad Jacobian, irst=4 does not return from evolve.f: the
        # triggering pass still performs its damping/momentum update, and
        # eqsolve carries that xcdot into the rebuilt-axis retry.  ``done``
        # below transfers control after that update; only its xc is discarded.
        stepping = running & (~converged) & (~bad_init) & (~nonfinite1)

        # ---- TimeStepControl (evolve.f) ------------------------------------
        first = it == carry.iter1
        fsq_prev = carry.fsq
        res0_f = jnp.where(first, fsq_prev, carry.res0)
        res1_f = jnp.where(first, fsq0, carry.res1)
        record_low = (fsq_prev <= res0_f) & (fsq0 <= res1_f)
        res0_n = jnp.minimum(res0_f, fsq_prev)
        res1_n = jnp.minimum(res1_f, fsq0)
        growth_gate = ~(record_low & (~jac1))     # IF/ELSE-IF chain in Fortran
        grew = (
            growth_gate
            & ((it - carry.iter1) > GROWTH_MIN_ITERATIONS)
            & ((fsq_prev > GROWTH_LIMIT * res0_n) | (fsq0 > GROWTH_LIMIT * res1_n))
        )
        kind = jnp.where(grew, RESTART_GROWTH,
                         jnp.where(jac1, RESTART_JACOBIAN, STEP_OK))
        restart = stepping & (kind != STEP_OK)
        store = stepping & (first | (record_low & (~jac1)))

        xstore_n = _select(store, carry.state, carry.xstore)
        state_r = _select(restart, xstore_n, carry.state)
        xcdot_r = _select(restart, jax.tree.map(jnp.zeros_like, carry.xcdot), carry.xcdot)
        delt_r = carry.time_step * jnp.where(
            restart & (kind == RESTART_JACOBIAN), JACOBIAN_RESET_FACTOR, 1.0
        ) * jnp.where(
            restart & (kind == RESTART_GROWTH), 1.0 / GROWTH_BACKOFF_DIVISOR, 1.0
        )
        ijacob_r = carry.ijacob + (restart & (kind == RESTART_JACOBIAN)).astype(carry.ijacob.dtype)
        iter1_r = jnp.where(restart, it, carry.iter1)

        # Re-evaluate at the restored state (TimeStepControl calls funct3d).
        e2 = lax.cond(
            restart,
            lambda args: _evaluate(
                args[0], args[1], it, it, args[2], rt, args[3],
                use_fft=use_fft,
            ),
            lambda args: e1,
            (state_r, e1.cache, fsqz_c, fsqr_c + fsqz_c),
        )
        reeval_bad = restart & e2.jacobian_sign_changed
        nonfinite2 = restart & (~e2.jacobian_sign_changed) & (~_evaluation_is_finite(e2))
        numerical_bad = nonfinite1 | nonfinite2

        fsqr_f = jnp.where(restart, e2.residuals.fsqr, fsqr_c)
        fsqz_f = jnp.where(restart, e2.residuals.fsqz, fsqz_c)
        fsql_f = jnp.where(restart, e2.residuals.fsql, fsql_c)
        fsqr1_f = jnp.where(restart, e2.pre.fsqr1, e1.pre.fsqr1)
        fsqz1_f = jnp.where(restart, e2.pre.fsqz1, e1.pre.fsqz1)
        fsql1_f = jnp.where(restart, e2.pre.fsql1, e1.pre.fsql1)
        wb_f = jnp.where(restart, e2.wb, e1.wb)
        wp_f = jnp.where(restart, e2.wp, e1.wp)
        r00_f = jnp.where(restart, e2.r00, e1.r00)
        z00_f = jnp.where(restart, e2.z00, e1.z00)
        cache_f = _select(restart, e2.cache, e1.cache)
        gc_f = _select(restart, e2.gc, e1.gc)

        # ---- damping + momentum step (evolve.f) ----------------------------
        fsq1 = fsqr1_f + fsqz1_f + fsql1_f
        control = StepControl(
            time_step=delt_r, inv_tau=carry.inv_tau, fsq_total_prev=fsq_prev,
            residual_best_precond=jnp.asarray(jnp.inf),
            residual_best_raw=jnp.asarray(jnp.inf),
            iter_last_reset=iter1_r, jacobian_resets=ijacob_r,
        )
        b1, fac, control2 = damping_coefficients(control, it, fsq1)
        state_n, xcdot_n = momentum_update(state_r, xcdot_r, gc_f, b1, fac, delt_r)

        # ---- 2D block preconditioner: matrix-free Newton step (precon2d.f) --
        # Only traced when enabled; otherwise the momentum step above stands
        # and the default 1D path is byte-identical.  When active, take a
        # damped Newton step (state += cfg.step * (-J^{-1} gc)) with zeroed
        # velocity, mirroring evolve.f's xcdot reset on prec2d activation.
        if rt.prec2d is not None:
            fsqz_prev_used = jnp.where(restart, fsqz_c, carry.fsqz)
            newton_dir, prec2d_active = _newton_step(
                rt, state_r, gc_f, cache_f, it, fsqz_prev_used,
                fsqr_f + fsqz_f + fsql_f, stepping & (~reeval_bad),
            )
            state_newton = jax.tree.map(
                lambda x, d: x + rt.prec2d.step * d, state_r, newton_dir
            )
            xcdot_newton = jax.tree.map(jnp.zeros_like, xcdot_r)
            state_n = _select(prec2d_active, state_newton, state_n)
            xcdot_n = _select(prec2d_active, xcdot_newton, xcdot_n)

        # ---- eqsolve.f escalation ------------------------------------------
        eq_reset = stepping & ((ijacob_r == 25) | (ijacob_r == 50))
        state_out = _select(eq_reset, xstore_n, _select(stepping, state_n, carry.state))
        xcdot_out = _select(
            eq_reset, jax.tree.map(jnp.zeros_like, carry.xcdot),
            _select(stepping, xcdot_n, carry.xcdot),
        )
        delt_n = delt_r * jnp.where(eq_reset, JACOBIAN_RESET_FACTOR, 1.0)
        ijacob_n = ijacob_r + eq_reset.astype(ijacob_r.dtype)
        iter1_n = jnp.where(eq_reset, it, iter1_r)

        jac75 = stepping & (ijacob_n >= 75)
        maxed = stepping & (~eq_reset) & (~jac75) & (it >= max_iter)

        stop_now = (
            running & (converged | bad_init | axis_reguess | numerical_bad)
            | jac75 | maxed | reeval_bad
        )
        done_n = carry.done | stop_now
        ier_n = jnp.where(
            carry.done, carry.ier,
            jnp.where(running & converged, SUCCESSFUL_TERM_FLAG,
            jnp.where(running & bad_init, BAD_JACOBIAN_FLAG,
            jnp.where(running & axis_reguess, AXIS_REGUESS_FLAG,
            jnp.where(running & numerical_bad, NONFINITE_FLAG,
            jnp.where(jac75, JAC75_FLAG,
            jnp.where(reeval_bad, MISC_ERROR_FLAG,
            jnp.where(maxed, MORE_ITER_FLAG, carry.ier))))))),
        ).astype(carry.ier.dtype)

        advance = (
            stepping & (~axis_reguess) & (~eq_reset) & (~jac75) & (~maxed)
            & (~reeval_bad) & (~numerical_bad)
        )
        iteration_n = jnp.where(advance, it + 1, it)

        # ---- trajectory row (printout.f values) ----------------------------
        w0 = wb_f + wp_f / (gamma - 1.0)
        row = jnp.stack([
            it.astype(wb_f.dtype), fsqr_f, fsqz_f, fsql_f,
            fsqr1_f, fsqz1_f, fsql1_f, r00_f, z00_f, w0 * _TWO_PI_SQ, delt_r,
        ])
        idx = jnp.clip(it - 1, 0, max_iter - 1)
        old_row = lax.dynamic_slice_in_dim(carry.trajectory, idx, 1, axis=0)[0]
        new_row = jnp.where(running, row, old_row)
        trajectory_n = lax.dynamic_update_slice_in_dim(carry.trajectory, new_row[None, :], idx, axis=0)

        gate = lambda mask, new, old: jnp.where(mask, new, old)  # noqa: E731
        return _LoopCarry(
            state=state_out, xcdot=xcdot_out,
            xstore=_select(stepping, xstore_n, carry.xstore),
            cache=_select(running, cache_f, carry.cache),
            time_step=gate(stepping, delt_n, carry.time_step),
            inv_tau=gate(stepping, control2.inv_tau, carry.inv_tau),
            fsq=gate(stepping, control2.fsq_total_prev, carry.fsq),
            res0=gate(stepping, res0_n, carry.res0),
            res1=gate(stepping, res1_n, carry.res1),
            fsqr=gate(running, fsqr_f, carry.fsqr),
            fsqz=gate(running, fsqz_f, carry.fsqz),
            fsql=gate(running, fsql_f, carry.fsql),
            fsqr1=gate(running, fsqr1_f, carry.fsqr1),
            fsqz1=gate(running, fsqz1_f, carry.fsqz1),
            fsql1=gate(running, fsql1_f, carry.fsql1),
            wb=gate(running, wb_f, carry.wb), wp=gate(running, wp_f, carry.wp),
            r00=gate(running, r00_f, carry.r00),
            iteration=iteration_n, iter1=gate(stepping, iter1_n, carry.iter1),
            ijacob=gate(stepping, ijacob_n, carry.ijacob),
            done=done_n, ier=ier_n, trajectory=trajectory_n,
        )

    return body


def _initial_carry(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    ijacob: int,
    xcdot: SpectralState | None = None,
    residuals: tuple[float | Array, float | Array, float | Array] | None = None,
) -> _LoopCarry:
    """Initial loop carry (``reset_params.f`` / ``initialize_radial.f``).

    ``initialize_radial.f`` resets ``fsq`` to one but intentionally leaves
    the module variables ``fsqr/fsqz/fsql`` unchanged across radial grids.
    Those retained values control the first-pass free-boundary edge gate in
    ``residue.f90``.  ``residuals=None`` is the cold-start
    ``reset_params.f`` value ``(1, 1, 1)``.
    """
    dtype = rt.setup.s_full.dtype
    one = jnp.asarray(1.0, dtype=dtype)
    zeros = (
        jax.tree.map(jnp.zeros_like, state)
        if xcdot is None else xcdot
    )
    delt0 = jnp.asarray(rt.time_step0, dtype=dtype)
    zero, inf = jnp.zeros((), dtype=dtype), jnp.asarray(jnp.inf, dtype=dtype)
    # NOTE: scalar counters/flags carry explicit (non-weak) dtypes so that the
    # initial carry has exactly the avals of the carry the jitted lanes
    # return; weak-typed Python scalars here would force a second lane
    # compilation on the first block round-trip.
    int_ = lambda v: jnp.asarray(v, dtype=jnp.int64)  # noqa: E731
    if residuals is None:
        fsqr0 = fsqz0 = fsql0 = one
    else:
        fsqr0, fsqz0, fsql0 = (
            jnp.asarray(value, dtype=dtype) for value in residuals
        )
    return _LoopCarry(
        state=state, xcdot=zeros, xstore=state, cache=_zero_cache(rt),
        time_step=delt0,
        inv_tau=jnp.full((NDAMP,), DAMPING_CAP, dtype=dtype) / delt0,
        fsq=one, res0=inf, res1=inf,
        fsqr=fsqr0, fsqz=fsqz0, fsql=fsql0,
        fsqr1=one, fsqz1=one, fsql1=one,
        wb=zero, wp=zero, r00=zero,
        iteration=int_(1), iter1=int_(1),
        ijacob=int_(int(ijacob)),
        done=jnp.zeros((), dtype=bool), ier=int_(NORM_TERM_FLAG),
        trajectory=jnp.zeros((rt.max_iterations, _TRAJ_COLS), dtype=dtype),
    )


# -- Result container and lanes --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VacuumOutput:
    """Publishable final NESTOR state (potential modes + surface fields).

    Arrays use NESTOR's reduced boundary grid.  Internal Green-function
    matrices and cached right-hand sides deliberately remain private.
    """

    potsin: np.ndarray
    potcos: np.ndarray | None
    xmpot: np.ndarray
    xnpot: np.ndarray
    bsubu: np.ndarray
    bsubv: np.ndarray
    bsupu: np.ndarray
    bsupv: np.ndarray


@dataclass(frozen=True)
class SolveResult:
    """Converged fixed-boundary solve output.

    ``rmnc/zmns`` (+ ``rmns/zmnc`` when ``lasym``) are physical (wout
    convention) spectral coefficients on the full mesh, mode-ordered like the
    wout ``xm/xn`` arrays; ``iotaf`` follows ``add_fluxes.f90`` for
    ``ncurr = 1``.  ``fsq_history`` has one row per iteration:
    ``(fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1)``.  ``wmhd`` is the printed
    ``WMHD = (wb + wp/(gamma-1)) * (2 pi)^2``.  ``vacuum`` is ``None`` for
    fixed-boundary solves.
    """

    converged: bool; iterations: int; ier_flag: int
    fsqr: float; fsqz: float; fsql: float
    wb: float; wp: float; wmhd: float; r00: float
    time_step: float; jacobian_resets: int
    state: SpectralState
    xm: np.ndarray; xn: np.ndarray
    rmnc: np.ndarray; zmns: np.ndarray
    rmns: np.ndarray | None; zmnc: np.ndarray | None
    iotaf: np.ndarray; fsq_history: np.ndarray
    vacuum: VacuumOutput | None = None


def _result_from_carry(carry: _LoopCarry, rt: SolverRuntime) -> SolveResult:
    """Host-side result assembly (wout-convention outputs)."""
    setup = rt.setup
    state = carry.state
    R_cos, R_sin, Z_cos, Z_sin = _physical_coefficients(
        carry.state, modes=rt.modes, lthreed=setup.lthreed, lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    scale = 1.0 / physical_to_internal_scale(rt.modes, rt.trig)
    rmnc = np.asarray(R_cos) * scale[None, :]
    zmns = np.asarray(Z_sin) * scale[None, :]
    rmns = np.asarray(R_sin) * scale[None, :] if setup.lasym else None
    zmnc = np.asarray(Z_cos) * scale[None, :] if setup.lasym else None

    # iotaf (add_fluxes.f90): prescribed profile for ncurr = 0; reconstructed
    # from the converged current-constrained chips for ncurr = 1.
    if int(setup.ncurr) == 1:
        _, geometry = _geometry(carry.state, rt)
        jacobian = half_mesh_jacobian(geometry, s=setup.s_full)
        metrics = metric_elements(geometry, s=setup.s_full)
        fields = magnetic_fields(
            geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
            s=setup.s_full, phips=setup.phips, phipf=setup.phipf,
            chips=setup.chips, signgs=setup.signgs, gamma=rt.gamma,
            mass=setup.mass, ncurr=setup.ncurr, enclosed_current=setup.icurv,
        )
        chips = np.asarray(fields.chips)
        phips = np.asarray(setup.phips)
        iotas = np.divide(chips, phips, out=np.zeros_like(chips), where=phips != 0.0)
        iotaf = np.zeros_like(iotas)
        iotaf[0] = 1.5 * iotas[1] - 0.5 * iotas[2]
        iotaf[1:-1] = 0.5 * (iotas[1:-1] + iotas[2:])
        iotaf[-1] = 1.5 * iotas[-1] - 0.5 * iotas[-2]
    else:
        iotaf = np.asarray(setup.iotaf)

    iterations = int(carry.iteration)
    trajectory = np.asarray(carry.trajectory)[:iterations]
    xm = np.asarray(rt.modes.m, dtype=float)
    xn = np.asarray(rt.modes.n, dtype=float) * float(rt.resolution.nfp)
    gamma = rt.gamma
    wb = float(carry.wb)
    wp = float(carry.wp)
    return SolveResult(
        converged=bool(int(carry.ier) == SUCCESSFUL_TERM_FLAG),
        iterations=iterations,
        ier_flag=int(carry.ier),
        fsqr=float(carry.fsqr), fsqz=float(carry.fsqz), fsql=float(carry.fsql),
        wb=wb, wp=wp, wmhd=float((wb + wp / (gamma - 1.0)) * _TWO_PI_SQ),
        r00=float(carry.r00),
        time_step=float(carry.time_step),
        jacobian_resets=int(carry.ijacob),
        state=state, xm=xm, xn=xn,
        rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc, iotaf=iotaf,
        fsq_history=trajectory[:, 1:7].copy(),
    )


def _emit_lines(rt: SolverRuntime, trajectory: np.ndarray, upto: int,
                printed: set[int], final: bool, emit) -> None:
    """Print screen lines at the VMEC2000 cadence (eqsolve.f/printout.f)."""
    lasym = rt.resolution.lasym
    for it in range(1, upto + 1):
        due = (it == 1) or (it % rt.nstep == 0) or (final and it == upto)
        if not due or it in printed:
            continue
        row = trajectory[it - 1]
        if int(row[0]) != it:      # row not (yet) written for this iteration
            continue
        emit(screen_line(
            it, float(row[1]), float(row[2]), float(row[3]),
            float(row[7]), float(row[10]), float(row[9]),
            z_axis=float(row[8]) if lasym else None,
        ), end="")
        printed.add(it)


@jax.jit
def _while_lane(carry: _LoopCarry, rt: SolverRuntime) -> _LoopCarry:
    """Whole-solve ``lax.while_loop`` lane, keyed structurally on ``rt``.

    Module-level ``jax.jit`` with the runtime passed as a pytree argument:
    two DIFFERENT runtimes with equal structure (same meta, same leaf
    shapes/dtypes) — e.g. two boundaries at one :class:`Resolution`, hot
    restarts, optimization iterates — share one XLA executable.
    """
    body = _make_body(rt)
    return lax.while_loop(lambda c: jnp.logical_not(c.done), body, carry)


@functools.partial(jax.jit, donate_argnums=(0,))
def _block_lane(carry: _LoopCarry, rt: SolverRuntime) -> _LoopCarry:
    """One ``BLOCK_SIZE``-iteration ``lax.scan`` block (CLI lane), structural.

    ``donate_argnums=(0,)``: the CLI lane drives the solve as a Python loop
    ``carry = _block_lane(carry, rt)``, so the input carry is dead after each
    call — donating it lets XLA alias the carry's output onto the input
    buffers, removing the transient 2x-carry high-water mark.  ``rt``
    (argument 1) is reused across blocks and is *not* donated.  Numerically
    identical to the non-donated lane.
    """
    body = _make_body(rt)
    return lax.scan(lambda cc, _: (body(cc), None), carry, None, length=BLOCK_SIZE)[0]


@jax.jit
def _while_lane_fft(carry: _LoopCarry, rt: SolverRuntime) -> _LoopCarry:
    """Whole-solve lane using separable Fourier synthesis."""
    body = _make_body(rt, use_fft=True)
    return lax.while_loop(lambda c: jnp.logical_not(c.done), body, carry)


@functools.partial(jax.jit, donate_argnums=(0,))
def _block_lane_fft(carry: _LoopCarry, rt: SolverRuntime) -> _LoopCarry:
    """Donated CLI block using separable Fourier synthesis."""
    body = _make_body(rt, use_fft=True)
    return lax.scan(
        lambda cc, _: (body(cc), None), carry, None, length=BLOCK_SIZE
    )[0]


# -- Overlapped stage compilation (multigrid compile scheduling) ------------------------------------------------------
#
# The ``NS_ARRAY`` ladder knows every rung's structure up front, so the
# multigrid driver AOT-compiles rung k+1's CLI block-lane executable on a
# background thread while rung k iterates (XLA compilation releases the
# GIL).  The standalone :class:`jax.stages.Compiled` object is stored here,
# keyed on the exact ``(lane, treedef, leaf-avals)`` structure, and consumed
# by :func:`_run_loop`; any mismatch falls back silently to the ordinary
# jitted lane.  Standalone executables survive ``jax.clear_caches()`` — the
# CLI's ``release_stage_cache`` point — which is why the prefetch stores
# them instead of relying only on the shared in-memory executable cache
# that ``Lowered.compile()`` happens to prime.  This is cache warming only:
# the prefetched executable is byte-identical XLA output for the identical
# lowering, so every iteration is bit-for-bit the on-demand result.

_LANE_EXECUTABLES: dict[Any, Any] = {}   #: prefetched AOT lane executables
_USED_LANE_KEYS: set = set()             #: lane structures already compiled/used


def _leaf_signature(leaf: Any) -> tuple:
    """Hashable (shape, dtype, weak-typed) triple of one pytree leaf."""
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return (tuple(leaf.shape), str(leaf.dtype), False)
    return ((), str(np.result_type(type(leaf))), True)


def _lane_signature(lane_name: str, carry: _LoopCarry, rt: SolverRuntime):
    """Structural key of one ``lane(carry, rt)`` call (treedef + avals)."""
    leaves, treedef = jax.tree_util.tree_flatten((carry, rt))
    return (lane_name, treedef, tuple(_leaf_signature(leaf) for leaf in leaves))


def _prefetch_block_lane(rt: SolverRuntime, *, use_fft: bool) -> bool:
    """AOT-compile the CLI block lane for ``rt``'s structure ahead of use.

    Called from the multigrid prefetch thread.  Builds the same initial
    carry :func:`_run_loop` would build (cheap structure assembly — no
    force evaluation) and compiles the lane via ``jit.lower(...).compile()``.
    Returns ``True`` when an executable for the structure is available
    (freshly built, or already compiled earlier in this process).
    """
    if jax.config.jax_disable_jit:
        return False
    lane = _block_lane_fft if use_fft else _block_lane
    lane_name = "block_fft" if use_fft else "block"
    carry = jax.tree.map(
        jnp.array, _initial_carry(_initial_state(rt.setup), rt, ijacob=0)
    )
    key = _lane_signature(lane_name, carry, rt)
    if key in _LANE_EXECUTABLES or key in _USED_LANE_KEYS:
        return True
    _LANE_EXECUTABLES[key] = lane.lower(carry, rt).compile()
    return True


def _release_used_lane_executables() -> None:
    """Drop lane executables already consumed by a finished rung.

    Companion of the multigrid ``release_stage_cache`` point (called next to
    ``jax.clear_caches()``): consumed standalone executables die with the
    jit caches, while a just-prefetched — not yet used — executable for the
    NEXT rung survives the release.
    """
    for key in list(_LANE_EXECUTABLES):
        if key in _USED_LANE_KEYS:
            del _LANE_EXECUTABLES[key]
    _USED_LANE_KEYS.clear()


def _resolve_use_fft(
    use_fft: bool | None, device: Any, resolution: Resolution
) -> bool:
    """Select the measured synthesis kernel without environment routing."""
    if use_fft is not None:
        return bool(use_fft)
    if int(resolution.mnmax) <= GPU_MAX_SPECTRAL_MODES:
        return False
    target = _placement_device(device, resolution)
    if target is None:
        target = jax.config.jax_default_device
    backend = (
        getattr(target, "platform", None)
        if target is not None
        else jax.default_backend()
    )
    return backend != "cpu" or platform.machine().lower() in ("arm64", "aarch64")


def _run_loop(state0: SpectralState, rt: SolverRuntime, *, mode: str,
              ijacob: int, verbose: bool, emit,
              use_fft: bool = False,
              emit_banner: bool = True,
              emit_legend: bool = True,
              initial_xcdot: SpectralState | None = None,
              initial_residuals: (
                  tuple[float | Array, float | Array, float | Array] | None
              ) = None) -> _LoopCarry:
    """Run the iteration loop in the requested lane; return the final carry.

    ``emit_legend=False`` suppresses the ``BEGIN FORCE ITERATIONS`` legend
    while keeping the ``NS = ...`` banner and column header (runvmec.f
    prints the legend once per run, not once per radial grid).
    """
    carry = _initial_carry(
        state0, rt, ijacob=ijacob, xcdot=initial_xcdot,
        residuals=initial_residuals,
    )

    if mode == "jit":
        return (_while_lane_fft if use_fft else _while_lane)(carry, rt)

    if mode != "cli":
        raise ValueError(f"unknown mode {mode!r}; expected 'cli' or 'jit'")
    # The donated CLI lane (_block_lane, donate_argnums=0) requires every leaf
    # of the input carry to be a distinct buffer; _initial_carry aliases some
    # (xstore=state, shared cache zeros).  One copy to distinct buffers here
    # (values bit-for-bit unchanged) makes the per-block donation valid and is
    # amortized over the whole solve.
    carry = jax.tree.map(jnp.array, carry)
    if verbose and emit_banner:
        # initialize_radial.f prints the total Fourier mode count (mnmax), not mpol.
        emit(stage_banner(rt.resolution.ns, rt.resolution.mnmax, rt.ftol, rt.max_iterations), end="")
        if emit_legend:
            emit(FORCE_ITERATIONS_BANNER, end="")
        emit(screen_header(lasym=rt.resolution.lasym, lfreeb=False), end="")

    printed: set[int] = set()
    max_passes = rt.max_iterations + 200
    lane = _block_lane_fft if use_fft else _block_lane
    # Prefetched-executable consumption + compile attribution.  The
    # structural key selects a background-compiled standalone executable
    # when one exists (identical lowering — bit-identical iterations); a
    # first-time structure without one is compiled on demand by the jitted
    # lane, and the notice makes that pause attributable on screen.
    executable = None
    if not jax.config.jax_disable_jit:
        key = _lane_signature("block_fft" if use_fft else "block", carry, rt)
        executable = _LANE_EXECUTABLES.get(key)
        if verbose and key not in _USED_LANE_KEYS:
            emit(compile_notice(rt.resolution.ns,
                                prefetched=executable is not None), end="")
        _USED_LANE_KEYS.add(key)
    for _ in range(max_passes):
        if executable is not None:
            try:
                carry = executable(carry, rt)
            except Exception:
                # Structural/placement drift (argument validation precedes
                # execution and donation, so the carry is intact): fall back
                # to the on-demand jitted lane for the rest of the rung.
                executable = None
                carry = lane(carry, rt)
        else:
            carry = lane(carry, rt)
        done = bool(carry.done)
        upto = int(carry.iteration) if done else int(carry.iteration) - 1
        # VMEC2000's irst=4 and first-bad-Jacobian transfers return to
        # eqsolve before printout.f, so the discarded first pass has no row.
        retry_transfer = done and int(carry.ier) in (
            AXIS_REGUESS_FLAG, BAD_JACOBIAN_FLAG,
        )
        if verbose and not retry_transfer:
            trajectory = np.asarray(carry.trajectory[:max(upto, 0)])
            _emit_lines(rt, trajectory, upto, printed, done, emit)
        if done:
            break
    return carry


def _solve_stage(rt: SolverRuntime, state0: SpectralState | None, *,
                 mode: str, verbose: bool, emit,
                 try_axis_reguess: bool = True,
                 use_fft: bool = False,
                 jacobian_retries: int = 2,
                 emit_legend: bool = True,
                 residual_continuation: (
                     tuple[float | Array, float | Array, float | Array] | None
                 ) = None) -> _LoopCarry:
    """Run one solve at a fixed runtime, with the eqsolve.f axis-retry.

    ``state0=None`` starts from the runtime's ``profil3d.f`` interior guess.
    On a first-iteration Jacobian sign change with ``ijacob == 0``
    (``eqsolve.f``), the axis is re-guessed from the failing geometry and the
    loop restarted once (``try_axis_reguess``).

    VMEC2000 stops after 75 Jacobian resets and tells the user to change
    ``DELT``.  VMEX makes that manual recovery deterministic: up to
    ``jacobian_retries`` attempts restart the best finite checkpoint with
    zero velocity and half the preceding initial step, capped at 0.5.  The
    equilibrium equations and convergence test are unchanged; setting
    ``jacobian_retries=0`` preserves the exact VMEC2000 fatal-stop policy.
    Returns the final carry; the caller maps ``carry.ier`` to
    results/exceptions (:func:`_finalize`).
    """
    if int(jacobian_retries) < 0:
        raise ValueError("jacobian_retries must be non-negative")

    def run_attempt(
        attempt_rt: SolverRuntime,
        attempt_state: SpectralState,
        *,
        emit_banner: bool,
        allow_axis_reguess: bool,
        attempt_residuals: (
            tuple[float | Array, float | Array, float | Array] | None
        ),
    ) -> tuple[_LoopCarry, SolverRuntime]:
        # A JAC75 recovery is a continuation from xstore, not a new
        # profil3d.f initialization.  Disable the iteration-1 LMOVE_AXIS
        # transfer for that attempt so the checkpoint cannot be replaced by a
        # reconstructed cold state merely because its force is still large.
        loop_rt = (
            attempt_rt
            if allow_axis_reguess
            else replace(attempt_rt, lmove_axis=False)
        )
        carry = _run_loop(
            attempt_state, loop_rt, mode=mode, ijacob=0,
            verbose=verbose, emit=emit, use_fft=use_fft,
            emit_banner=emit_banner, emit_legend=emit_legend,
            initial_residuals=attempt_residuals,
        )

        # eqsolve.f: on a first-iteration Jacobian sign change with
        # ijacob == 0, re-guess the axis from the current geometry and restart
        # once.  A high finite first force uses the same transfer while
        # preserving the triggering pass's momentum.
        retry_reason = int(carry.ier)
        if (
            allow_axis_reguess
            and try_axis_reguess
            and retry_reason in (BAD_JACOBIAN_FLAG, AXIS_REGUESS_FLAG)
            and int(carry.ijacob) == 0
            and attempt_rt.resolution.ns >= 3
        ):
            if verbose:
                if retry_reason == BAD_JACOBIAN_FLAG:
                    emit(" INITIAL JACOBIAN CHANGED SIGN!")
                emit(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS")
            attempt_rt, attempt_state, _axis = reguess_initial_axis(
                attempt_rt, attempt_state, use_fft=use_fft
            )
            if verbose:
                lasym = bool(attempt_rt.resolution.lasym)
                emit(improved_axis_block(
                    _axis[0], _axis[3],
                    raxis_cs=_axis[1] if lasym else None,
                    zaxis_cc=_axis[2] if lasym else None,
                ), end="")
            carry = _run_loop(
                attempt_state, attempt_rt, mode=mode, ijacob=1,
                verbose=verbose, emit=emit, use_fft=use_fft,
                emit_banner=False,
                initial_xcdot=(
                    carry.xcdot
                    if retry_reason == AXIS_REGUESS_FLAG else None
                ),
                initial_residuals=(
                    carry.fsqr, carry.fsqz, carry.fsql
                ),
            )
        return carry, attempt_rt

    attempt_state = _initial_state(rt.setup) if state0 is None else state0
    attempt_rt = rt
    carry, attempt_rt = run_attempt(
        attempt_rt, attempt_state, emit_banner=True, allow_axis_reguess=True,
        attempt_residuals=residual_continuation,
    )
    for attempt in range(1, int(jacobian_retries) + 1):
        if int(carry.ier) != JAC75_FLAG:
            break
        retry_step = min(0.5, 0.5 * float(attempt_rt.time_step0))
        attempt_state = carry.xstore
        attempt_rt = runtime_with_baselines(
            replace(attempt_rt, time_step0=retry_step),
            attempt_state,
        )
        if verbose:
            emit(
                " JACOBIAN RECOVERY RETRY "
                f"{attempt}/{int(jacobian_retries)}: "
                "RESTARTING BEST FINITE STATE WITH "
                f"DELT = {retry_step:.6g}"
            )
        carry, attempt_rt = run_attempt(
            attempt_rt, attempt_state, emit_banner=False,
            allow_axis_reguess=False,
            attempt_residuals=(carry.fsqr, carry.fsqz, carry.fsql),
        )
    return carry


def _finalize(carry: _LoopCarry, rt: SolverRuntime) -> SolveResult:
    """Map the final carry to a :class:`SolveResult` or a typed exception."""
    ier = int(carry.ier)
    fsq = (float(carry.fsqr), float(carry.fsqz), float(carry.fsql))
    if ier == SUCCESSFUL_TERM_FLAG:
        return _result_from_carry(carry, rt)
    if ier == MORE_ITER_FLAG:
        hint = "increase NITER or loosen FTOL"
        if (
            not rt.lforbal
            and rt.setup.ncurr == 1
            and not np.any(np.asarray(rt.setup.mass))
            and not np.any(np.asarray(rt.setup.icurv))
        ):
            hint += "; review LFORBAL=T for a weak vacuum m=1,n=0 direction"
        raise VmecConvergenceError(
            WERROR_MESSAGES[MORE_ITER_FLAG],
            hint=hint,
            iteration=int(carry.iteration), fsq=fsq, ftol=rt.ftol,
        )
    if ier == NONFINITE_FLAG:
        raise VmecNumericalError(
            "NON-FINITE FORCE EVALUATION (NaN OR Inf)",
            hint=(
                "check parsed PHIEDGE/APHI, profile values, and the initial "
                "axis/boundary; a zero effective toroidal flux makes the "
                "VMEC force normalizations singular"
            ),
            iteration=int(carry.iteration), fsq=fsq,
        )
    raise VmecJacobianError(
        WERROR_MESSAGES.get(ier, WERROR_MESSAGES[JAC75_FLAG]),
        hint="decrease DELT or improve the axis guess",
        ier_flag=ier if ier in WERROR_MESSAGES else JAC75_FLAG,
        iteration=int(carry.iteration), jacobian_resets=int(carry.ijacob),
        fsq=fsq,
    )


def solve(
    source: VmecInput | RunSetup,
    resolution: Resolution | None = None,
    *,
    ftol: float | None = None, max_iterations: int | None = None,
    mode: str = "cli",
    time_step: float | None = None, tcon0: float | None = None,
    gamma: float | None = None, nstep: int | None = None,
    lconm1: bool = True, verbose: bool = False, emit=print,
    initial_state: SpectralState | None = None,
    restart_from: Any = None,
    device: Any = AUTO,
    precon_type: str | None = None, prec2d_threshold: float | None = None,
    prec2d: Prec2DConfig | None = None,
    use_fft: bool | None = None,
    jacobian_retries: int = 2,
) -> SolveResult:
    """Single-grid fixed-boundary solve (VMEC2000 ``eqsolve.f``).

    ``source`` is a parsed :class:`vmex.core.input.VmecInput`
    (recommended; supplies the ``delt/tcon0/gamma/nstep/ftol/niter`` defaults,
    with the keywords overriding) or a prebuilt
    :class:`vmex.core.setup.RunSetup` (requires ``resolution``).  The
    resolution defaults to the first ``ns_array`` stage (``read_indata.f``
    grid rules).  Convergence requires ``fsqr, fsqz, fsql <= ftol``
    *simultaneously* (``evolve.f``).  ``mode="cli"`` runs a Python loop over
    jitted 10-iteration blocks with host residual checks and VMEC2000-format
    printing (``verbose=True``); ``mode="jit"`` runs one ``lax.while_loop``
    over the same traced body.

    Returns a :class:`SolveResult` on convergence.  ``jacobian_retries``
    (default 2) restarts the best finite checkpoint with a reduced ``DELT``
    after VMEC2000's 75-reset condition; set it to zero for the exact
    VMEC2000 fatal-stop policy.  Raises
    :class:`VmecJacobianError` when the initial Jacobian changes sign twice
    (after one ``guess_axis`` retry — the ``eqsolve.f`` ``ijacob == 0`` path)
    or at ``ijacob >= 75`` (``jac75_flag``), and :class:`VmecConvergenceError`
    when ``max_iterations`` is exhausted (``more_iter_flag``); both carry the
    final iteration and ``(fsqr, fsqz, fsql)`` diagnostics.

    ``initial_state`` hot-restarts the solve from a previous
    :class:`SpectralState` at the *same* resolution (e.g. ``result.state`` of
    an earlier solve on a perturbed boundary — VMEC++-style hot restart; use
    :func:`vmex.core.multigrid.interpolate_state` first when ``ns``
    differs).  The R/Z *edge row* of the provided state is replaced by the
    input's processed boundary (the edge never evolves in fixed-boundary
    mode, so keeping the old row would silently re-solve the old boundary);
    the interior and lambda are kept.

    ``restart_from`` is the general form: a ``wout_*.nc`` path (VMEX- or
    VMEC2000/PARVMEC-written), a :class:`~vmex.core.wout.WoutData`, a
    previous :class:`SolveResult`, or a :class:`SpectralState`.  The source
    is normalized with :func:`vmex.core.restart.restart_state` — wout mode
    tables are remapped onto this deck's ``mpol/ntor`` and the radial grid is
    resampled to this solve's ``ns`` — and then follows the
    ``initial_state`` path above.  Requires a :class:`VmecInput` source for
    wout inputs (the deck supplies the mode table and flux normalization);
    mutually exclusive with ``initial_state``.

    ``device`` places the jitted iteration lanes: ``"cpu"``/``"gpu"``/
    ``"cuda"``/``"tpu"`` or a ``jax.Device`` (always honored), ``"auto"``
    (default) to apply the measured small-work-to-CPU policy, or ``None`` to
    follow JAX placement.  The automatic policy never overrides an active
    ``jax.default_device`` context or user-pinned JAX platform.

    ``use_fft=None`` (default) selects separable FFT only above 512 modes on
    accelerators and ARM CPUs; smaller problems and x86 CPUs retain the dense
    real contraction. Explicit ``True``/``False`` always wins. The implicit
    API sets it False internally so its equilibrium and Jacobian share one
    lower-memory real-contraction executable.

    ``precon_type`` (``"NONE"`` default) with a finite ``prec2d_threshold`` —
    or an explicit ``prec2d``
    :class:`~vmex.core.preconditioner_2d.Prec2DConfig` — switches on the
    optional **2D block preconditioner** (VMEC2000 ``precon2d.f``): once
    ``fsqr + fsqz + fsql < prec2d_threshold`` the iteration replaces the 1D
    radial force direction by a matrix-free Newton step (exact Hessian-vector
    products via ``jax.jvp``, solved with :func:`solvax.gmres`), converging
    stiff cases (high beta/aspect/mode-number) in far fewer iterations.  The
    default (``NONE``) path is byte-identical to the 1D-only solver.
    """
    if resolution is None and isinstance(source, VmecInput):
        resolution = resolution_from_input(source)
    if resolution is None:
        raise ValueError("solve(RunSetup) requires a Resolution")
    if restart_from is not None:
        if initial_state is not None:
            raise ValueError(
                "pass either restart_from or initial_state, not both"
            )
        if not isinstance(source, VmecInput):
            raise ValueError(
                "solve(restart_from=...) requires a VmecInput source (the "
                "deck supplies the mode table and flux normalization)"
            )
        from .restart import restart_state

        initial_state = restart_state(
            restart_from, source, ns=int(resolution.ns)
        )
    use_fft_resolved = _resolve_use_fft(use_fft, device, resolution)
    initial_state = _put_numeric_leaves(
        initial_state, _placement_device(device, resolution)
    )
    with device_context(device, resolution):
        rt = prepare_runtime(
            source, resolution, ftol=ftol, max_iterations=max_iterations,
            time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
            lconm1=lconm1, precon_type=precon_type,
            prec2d_threshold=prec2d_threshold, prec2d=prec2d,
            use_fft=use_fft_resolved,
        )
    with device_context(device, rt.resolution):
        if initial_state is not None:
            ns, mnmax = rt.resolution.ns, rt.modes.mnmax
            if tuple(initial_state.R_cos.shape) != (ns, mnmax):
                raise ValueError(
                    f"initial_state has shape {tuple(initial_state.R_cos.shape)}, "
                    f"expected ({ns}, {mnmax}); interpolate with "
                    "vmex.core.multigrid.interpolate_state first"
                )
            initial_state = hot_restart_state(rt, initial_state)
            rt = runtime_with_baselines(
                rt, initial_state, use_fft=use_fft_resolved
            )  # funct3d.f iter2==iter1
        carry = _solve_stage(
            rt, initial_state, mode=mode, verbose=verbose, emit=emit,
            use_fft=use_fft_resolved,
            jacobian_retries=jacobian_retries,
        )
        return _finalize(carry, rt)
