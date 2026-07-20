"""Implicit differentiation of the fixed-boundary equilibrium (§6).

The converged equilibrium is a root of ``F(x, p) = 0`` with ``x`` the
:class:`~vmex.core.solver.SpectralState` and ``p`` the differentiable
run parameters (:class:`ImplicitParams`: dense INDATA boundary arrays,
``phiedge``, ``pres_scale``, ``curtor`` and the ``am/ai/ac`` profile
coefficients).  :func:`solve_implicit` wraps the opaque host solver
(:func:`vmex.core.solver.solve` / ``solve_multigrid``) in
``jax.custom_vjp``; the backward pass solves the adjoint linear system

    ``(dF/dx)^T lambda = g_x``

matrix-free (one ``jax.vjp`` linearization of the residual, re-applied by
GMRES) and returns ``g_p - lambda^T dF/dp`` with one more VJP — O(1) memory
in the forward iteration count.

Residual formulation (documented choice)
----------------------------------------
``F`` is the **self-consistently preconditioned force** ``gc`` of a single
fresh :func:`~vmex.core.solver.evaluate_forces` pass (``cache=None``):
the ``bcovar.f`` preconditioner/force norms/``tcon`` are *recomputed from the
current* ``(x, p)`` rather than frozen at the converged cache.  This makes
``F`` a fixed, smooth function of ``(x, p)`` — required by the implicit
function theorem — while remaining exactly as well-conditioned as VMEC's own
preconditioned iteration.  Correctness: ``gc = M(x, p) f(x, p)`` with ``f``
the raw (``scalxc``-scaled) spectral force and ``M`` the invertible linear
1D-preconditioner map (``scale_m1`` + ``scalfor`` tridiagonal solves +
``faclam``).  At the root, ``dF = M df + dM f = M df`` up to ``O(|f|) =
O(ftol)``, so the implicit gradients of the preconditioned residual equal
those of the raw force residual to solver accuracy, and GMRES on ``dF/dx``
inherits the preconditioning for free.  The raw-force formulation
(``formulation="raw"``) is kept for the informational with/without-
preconditioner comparison in the tests.

The m=1 constrained Z force is evaluated in its converged branch (zeroed —
``residue.f90`` zeroes ``gcz(m=1)`` once ``fsqz < 1e-6``, which always holds
at the fixed point), so the corresponding constrained combinations are *not*
degrees of freedom: they are frozen at their converged values, exactly
mirroring the forward solver's behavior near convergence.

Degrees of freedom / boundary handling
--------------------------------------
In fixed-boundary mode the R/Z edge spectral row never evolves: the full
state is assembled as ``x = mask*z + edge_mask*boundary(p) + frozen`` where
``z`` are the evolved dofs, the edge row comes (differentiably) from the
boundary parameters, and the remaining entries (structurally zero families,
released m=1 combinations, the lambda axis row overwritten by the ``totzsp``
closure) are frozen constants.  The dof mask is computed once per forward
solve from the *exact structural zero patterns* of ``gc`` (row support) and
of the ``x``-dependence of ``gc`` (column support, one VJP with a random
cotangent) at a generically perturbed state — see ``_dof_mask``.

Gradient checking solver-sensitive metrics
------------------------------------------
The adjoint gradient is the derivative of the fixed point of the *frozen*
residual ``F`` — the preconditioner/``tcon``/m=1 branch/dof mask are captured
once at the base parameters, not re-derived.  For a smooth bulk integral
(``wb``, ``aspect``) a naive central FD through the full host solver already
matches ``jax.grad`` to ``rtol <= 1e-6``.  But a **solver-sensitive** metric —
``iota`` (derived from the current-constrained ``chips`` at ``ncurr=1``), the
mirror ratio, the magnetic well, the Boozer/QI residual — reads the converged
state directly, and a naive re-solve at ``p ± h`` lets that convergence logic
re-form slightly differently on each side, an O(1) path perturbation that can
sign-flip the FD (``d(iota_edge)/d(RBC(-1,1))`` on ``li383_low_res``: adjoint
``-0.773``, naive FD ``+0.045``).  The naive FD is therefore *not* a valid
reference for these metrics; :func:`frozen_path_directional_fd` provides the
correct one (Newton-solve the frozen ``F`` at ``p ± h``), and it reproduces the
adjoint to solver accuracy — see ``tests/test_implicit_grad.py``.

Zero-crash typed errors through the callback
--------------------------------------------
The forward solve runs behind ``jax.pure_callback``, which converts any host
exception into an opaque ``jax.errors.JaxRuntimeError`` whose message embeds
the whole host traceback (measured ~3.7 KB) and *loses* the typed exception
(``__cause__`` is ``None``) — breaking the :mod:`vmex.core.errors`
zero-crash taxonomy.  The relay: ``_host_solve_and_mask`` catches any
:class:`~vmex.core.errors.VmecError`, stashes it in the module-level
``_HOST_ERROR`` slot (host callbacks are serialized per solve, so a single
slot suffices) and re-raises a SHORT sentinel ``RuntimeError``; the
``pure_callback`` call sites (``_callback_solve``) catch the runtime error,
pop the slot and re-raise the ORIGINAL typed exception with ``from None`` to
suppress the callback noise.  ``im.run`` / :func:`solve_implicit` therefore
fail with a short :class:`~vmex.core.errors.VmecConvergenceError` /
:class:`~vmex.core.errors.VmecJacobianError`.  Under ``jax.jit`` the
error instead surfaces at the jit boundary (where the
``optimize.least_squares`` zero-crash penalty lanes catch it); the sentinel
keeps even that message short.

Parameter map
-------------
``runtime_from_params`` rebuilds every p-dependent
:class:`~vmex.core.setup.RunSetup` field traceably: the ``readin.f``
boundary processing (with the theta-flip decision frozen from the reference
input — it is discrete), the ``profil1d.f`` flux/pressure/current profiles,
and the ``funct3d.f`` constraint baselines ``rcon0/zcon0`` (which depend on
the boundary only — the edge row of any admissible state).  Its output is
verified against :func:`~vmex.core.setup.run_setup` in
``tests/test_implicit_grad.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import types
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from solvax import gcrot as _solvax_gcrot
from solvax import gmres as _solvax_gmres

from .errors import VmecError
from .fields import magnetic_fields, metric_elements
from .fourier import Resolution
from .geometry import half_mesh_jacobian
from .input import VmecInput
from .multigrid import solve_multigrid
from .residuals import (
    m1_physical_to_constrained, m1_residue_rotation, scalxc_scale_force,
    zero_m1_z_force,
)
from .setup import RadialGrids, flux_profiles, interior_guess
from .solver import (
    SolveResult, SolverRuntime, SpectralState, _constraint_baselines,
    _force_to_state, _geometry, _initial_state, _physical_coefficients,
    _static_tables, evaluate_forces, prepare_runtime, resolution_from_input,
    solve,
)
from .fields import constraint_scaling
from .forces import mhd_forces, spectral_mhd_forces
from .statephysics import _field_chain as _field_chain_shared
from .transforms import (
    physical_to_internal_scale,
    register_pytree_dataclass as _register,
)

__all__ = [
    "ImplicitParams", "ImplicitConfig", "ImplicitSolution",
    "params_from_input", "input_with_params", "runtime_from_params",
    "make_config", "solve_implicit", "solve_implicit_with_aux",
    "implicit_state_pullback_multi_rhs", "run",
    "mhd_energy", "plasma_volume", "aspect_ratio", "iota_profile",
    "iota_axis", "iota_edge", "edge_iota", "residual_fn", "adjoint_matvec",
    "frozen_path_directional_fd",
]

Array = Any

_STATE_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


# ---------------------------------------------------------------------------
# Differentiable parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImplicitParams:
    """Differentiable run parameters (a JAX pytree).

    ``rbc/rbs/zbc/zbs`` are the *dense INDATA* boundary arrays, shape
    ``(2*ntor + 1, mpol)`` indexed ``[n + ntor, m]`` (physical, un-processed
    — exactly :class:`~vmex.core.input.VmecInput` layout, so e.g.
    ``RBC(0, 1)`` is ``rbc[ntor, 1]``).  ``am/ai/ac`` are the dense profile
    coefficient arrays; ``phiedge/pres_scale/curtor`` scalars.
    """

    rbc: Array
    rbs: Array
    zbc: Array
    zbs: Array
    phiedge: Array
    pres_scale: Array
    curtor: Array
    am: Array
    ai: Array
    ac: Array


_register(ImplicitParams)


def params_from_input(inp: VmecInput) -> ImplicitParams:
    """Extract the differentiable parameters of an input as a pytree.

    On an accelerator box the pytree is *committed* to the CPU
    (:func:`vmex.core.device.resolve_implicit_device`): every eager op of
    a ``jax.grad``/``jax.jacrev`` over :func:`run` then executes there, which
    is where the launch-bound implicit adjoint is fastest — measured 57 s
    (GPU) vs seconds (CPU) for one solovev ``value_and_grad`` (R24).
    A user ``JAX_PLATFORMS`` pin or an already-CPU backend stands the pin
    down; ``optimize.least_squares`` applies the same rule to its dof vector.
    """
    from .device import resolve_implicit_device

    dev = resolve_implicit_device(None, None)
    if dev is None:
        arr = lambda a: jnp.asarray(np.asarray(a, dtype=np.float64))  # noqa: E731
    else:
        arr = lambda a: jax.device_put(np.asarray(a, dtype=np.float64), dev)  # noqa: E731
    return ImplicitParams(
        rbc=arr(inp.rbc), rbs=arr(inp.rbs), zbc=arr(inp.zbc), zbs=arr(inp.zbs),
        phiedge=arr(inp.phiedge), pres_scale=arr(inp.pres_scale),
        curtor=arr(inp.curtor), am=arr(inp.am), ai=arr(inp.ai), ac=arr(inp.ac),
    )


def input_with_params(inp: VmecInput, params: ImplicitParams) -> VmecInput:
    """Host-side: a new :class:`VmecInput` with the parameter values applied."""
    arr = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    return dataclasses.replace(
        inp, rbc=arr(params.rbc), rbs=arr(params.rbs), zbc=arr(params.zbc),
        zbs=arr(params.zbs), phiedge=float(np.asarray(params.phiedge)),
        pres_scale=float(np.asarray(params.pres_scale)),
        curtor=float(np.asarray(params.curtor)),
        am=arr(params.am), ai=arr(params.ai), ac=arr(params.ac),
    )


# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ImplicitConfig:
    """Static (non-differentiable) context of one implicit solve."""

    inp: VmecInput
    resolution: Resolution
    ftol: float
    max_iterations: int
    mode: str = "cli"
    multigrid: bool = False
    lconm1: bool = True
    adjoint_tol: float = 1e-11
    adjoint_restart: int = 30
    adjoint_maxiter: int = 300
    #: seed repeated host solves from the last converged state of this config
    #: (optimization trials; the fixed point — hence the gradient — is
    #: unchanged, only the iteration count drops).  Makes the callback
    #: stateful across calls, so keep False for one-shot/diagnostic use.
    hot_restart: bool = False


def make_config(
    inp: VmecInput,
    *,
    ns: int | None = None,
    ftol: float | None = None,
    max_iterations: int | None = None,
    mode: str = "cli",
    multigrid: bool = False,
    lconm1: bool = True,
    adjoint_tol: float = 1e-11,
    adjoint_restart: int = 30,
    adjoint_maxiter: int = 300,
    hot_restart: bool = False,
) -> ImplicitConfig:
    """Build the static config; ``resolution`` is the (final-stage) grid."""
    if multigrid and ns is None:
        ns = int(np.max(np.asarray(inp.ns_array)))
    resolution = resolution_from_input(inp, ns=ns)
    if ftol is None:
        ftol = float(np.asarray(inp.ftol_array).ravel()[-1])
    if max_iterations is None:
        max_iterations = int(np.asarray(inp.niter_array).ravel()[-1])
    return ImplicitConfig(
        inp=inp, resolution=resolution, ftol=float(ftol),
        max_iterations=int(max_iterations), mode=str(mode),
        multigrid=bool(multigrid), lconm1=bool(lconm1),
        adjoint_tol=float(adjoint_tol), adjoint_restart=int(adjoint_restart),
        adjoint_maxiter=int(adjoint_maxiter), hot_restart=bool(hot_restart),
    )


@functools.lru_cache(maxsize=8)
def _template_runtime(cfg: ImplicitConfig) -> SolverRuntime:
    """Reference (host-built) runtime at the config's base input.

    Cached per config (identity hash — :class:`ImplicitConfig` is ``eq=False``):
    the template is p-independent, and caching both avoids rebuilding it on
    every trial solve of an optimization and keeps
    :func:`runtime_from_params` traceable (the host-side ``run_setup`` logic
    never runs under a trace once the template is a closure constant).
    """
    return prepare_runtime(
        cfg.inp, cfg.resolution, ftol=cfg.ftol,
        max_iterations=cfg.max_iterations, lconm1=cfg.lconm1,
    )


# ---------------------------------------------------------------------------
# Traceable parameter -> runtime map (readin.f + profil1d.f, differentiable)
# ---------------------------------------------------------------------------


def _boundary_from_params(params: ImplicitParams, cfg: ImplicitConfig):
    """Traceable ``readin.f`` boundary processing (symmetric inputs).

    Reproduces :func:`vmex.core.setup.boundary_from_input` for
    ``lasym = False`` with the theta-flip decision frozen from the reference
    input (it is a discrete function of ``p`` — constant in any neighborhood
    where the gradient exists).  Returns the internal-normalized,
    m=1-constrained signed helical arrays plus the physical ``r00``.
    """
    res = cfg.resolution
    if res.lasym:
        raise NotImplementedError(
            "implicit parameter map: lasym boundary processing (the readin.f "
            "delta rotation) is not implemented yet; run with lasym = False"
        )
    mpol, ntor = int(res.mpol), int(res.ntor)
    modes, trig = _static_tables(res)[0], _static_tables(res)[1]
    template = _template_runtime(cfg)
    lflip = bool(template.setup.lflip)

    rbc = jnp.asarray(params.rbc)
    zbs = jnp.asarray(params.zbs)

    # readin.f accumulation into the internal (|n|, m) blocks.
    shape = (ntor + 1, mpol)
    rbcc = jnp.zeros(shape, dtype=rbc.dtype)
    rbss = jnp.zeros(shape, dtype=rbc.dtype)
    zbcs = jnp.zeros(shape, dtype=rbc.dtype)
    zbsc = jnp.zeros(shape, dtype=rbc.dtype)
    lthreed = ntor > 0
    for m in range(mpol):
        if cfg.inp.lfreeb and 1 < cfg.inp.mfilter_fbdy < m:
            continue
        for n in range(-ntor, ntor + 1):
            if cfg.inp.lfreeb and 0 < cfg.inp.nfilter_fbdy < abs(n):
                continue
            ni, isgn, j = abs(n), (0 if n == 0 else (1 if n > 0 else -1)), n + ntor
            rbcc = rbcc.at[ni, m].add(rbc[j, m])
            if m > 0:
                zbsc = zbsc.at[ni, m].add(zbs[j, m])
            if lthreed:
                if m > 0:
                    rbss = rbss.at[ni, m].add(isgn * rbc[j, m])
                zbcs = zbcs.at[ni, m].add(-isgn * zbs[j, m])

    r00 = rbcc[0, 0]

    if lflip:  # flip_theta (init_geometry.f90), decision frozen from cfg.inp
        signs = jnp.asarray((-1.0) ** np.arange(mpol, dtype=float))[None, :]
        keep0 = lambda new, old: new.at[:, 0].set(old[:, 0])  # noqa: E731
        rbcc = keep0(signs * rbcc, rbcc)
        zbsc = keep0(-signs * zbsc, zbsc)
        rbss = keep0(-signs * rbss, rbss)
        zbcs = keep0(signs * zbcs, zbcs)

    # internal blocks -> signed helical packing (setup._helical_from_internal_blocks)
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    R_list, Z_list = [], []
    for m, n in zip(m_arr, n_arr):
        ni = abs(n)
        if m == 0 and n != 0:
            isgn = 1 if n > 0 else -1
            R_list.append(rbcc[ni, m])
            Z_list.append(-isgn * zbcs[ni, m])
        elif n == 0:
            R_list.append(rbcc[ni, m])
            Z_list.append(zbsc[ni, m])
        elif n > 0:
            R_list.append(0.5 * (rbcc[ni, m] + rbss[ni, m]))
            Z_list.append(0.5 * (zbsc[ni, m] - zbcs[ni, m]))
        else:
            R_list.append(0.5 * (rbcc[ni, m] - rbss[ni, m]))
            Z_list.append(0.5 * (zbsc[ni, m] + zbcs[ni, m]))
    scale = jnp.asarray(physical_to_internal_scale(modes, trig))
    R_cos = jnp.stack(R_list) * scale
    Z_sin = jnp.stack(Z_list) * scale
    zeros = jnp.zeros_like(R_cos)

    R_cos2, Z_sin2, _, _ = m1_physical_to_constrained(
        R_cos[None, :], Z_sin[None, :], None, None,
        modes=modes, lthreed=lthreed, lasym=False, lconm1=cfg.lconm1,
    )
    return R_cos2[0], zeros, zeros, Z_sin2[0], r00


def runtime_from_params(params: ImplicitParams, cfg: ImplicitConfig) -> SolverRuntime:
    """Differentiable (traceable) map ``p -> SolverRuntime``.

    Rebuilds every p-dependent :class:`RunSetup` field with jnp operations:
    the processed boundary, the ``profil1d.f`` flux/mass/current profiles
    (through :func:`vmex.core.setup.flux_profiles`, which is traced in
    ``phiedge/pres_scale/curtor/am/ai/ac`` and in ``r00``), the ``profil3d.f``
    interior guess (whose *edge row* is the boundary — the initial interior
    is an initializer only) and the constraint baselines ``rcon0/zcon0``
    (functions of the edge row alone).  All p-independent fields (grids,
    ``scalxc``, axis arrays, static metadata) come from the reference
    runtime.
    """
    template = _template_runtime(cfg)
    setup0 = template.setup
    inp = cfg.inp

    bR_cos, bR_sin, bZ_cos, bZ_sin, r00 = _boundary_from_params(params, cfg)

    grids = RadialGrids(
        s_full=setup0.s_full, s_half=setup0.s_half, sqrts=setup0.sqrts,
        shalf=setup0.shalf, sm=setup0.sm, sp=setup0.sp, hs=setup0.hs,
    )
    shim = types.SimpleNamespace(
        aphi=inp.aphi, bloat=inp.bloat, gamma=inp.gamma,
        spres_ped=inp.spres_ped,
        phiedge=params.phiedge, pres_scale=params.pres_scale,
        curtor=params.curtor,
        pmass_type=inp.pmass_type, am=params.am,
        am_aux_s=inp.am_aux_s, am_aux_f=inp.am_aux_f,
        piota_type=inp.piota_type, ai=params.ai,
        ai_aux_s=inp.ai_aux_s, ai_aux_f=inp.ai_aux_f,
        pcurr_type=inp.pcurr_type, ac=params.ac,
        ac_aux_s=inp.ac_aux_s, ac_aux_f=inp.ac_aux_f,
    )
    prof = flux_profiles(shim, grids, r00=r00, signgs=setup0.signgs,
                         lflip=setup0.lflip)

    modes, trig = _static_tables(cfg.resolution)[0], _static_tables(cfg.resolution)[1]
    state = interior_guess(
        boundary_R_cos=bR_cos, boundary_R_sin=bR_sin,
        boundary_Z_cos=bZ_cos, boundary_Z_sin=bZ_sin,
        raxis_c=setup0.raxis_c, raxis_s=setup0.raxis_s,
        zaxis_c=setup0.zaxis_c, zaxis_s=setup0.zaxis_s,
        modes=modes, trig=trig, s=setup0.s_full,
    )

    setup = dataclasses.replace(
        setup0,
        phips=prof["phips"], chips=prof["chips"], iotas=prof["iotas"],
        icurv=prof["icurv"], mass=prof["mass"], phipf=prof["phipf"],
        chipf=prof["chipf"], iotaf=prof["iotaf"], lamscale=prof["lamscale"],
        boundary_R_cos=bR_cos, boundary_R_sin=bR_sin,
        boundary_Z_cos=bZ_cos, boundary_Z_sin=bZ_sin,
        R_cos=state[0], R_sin=state[1], Z_cos=state[2], Z_sin=state[3],
        lambda_cos=state[4], lambda_sin=state[5],
    )
    rt = dataclasses.replace(template, setup=setup)
    rcon0, zcon0 = _constraint_baselines(_initial_state(setup), rt)
    return dataclasses.replace(rt, rcon0=rcon0, zcon0=zcon0)


# ---------------------------------------------------------------------------
# Residual, dof mask, state assembly
# ---------------------------------------------------------------------------


def _edge_mask(cfg: ImplicitConfig) -> SpectralState:
    """1.0 on the fixed R/Z edge spectral row, 0.0 elsewhere (static)."""
    res = cfg.resolution
    mn = int(np.asarray(_static_tables(res)[0].m).size)
    z = np.zeros((res.ns, mn))
    e = z.copy()
    e[-1, :] = 1.0
    mk = lambda a: jnp.asarray(a)  # noqa: E731
    return SpectralState(R_cos=mk(e), R_sin=mk(e), Z_cos=mk(e), Z_sin=mk(e),
                         L_cos=mk(z), L_sin=mk(z))


def _m1_pair_columns(cfg: ImplicitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Signed-packing column indices of the m=1 ``(+n, -n)`` pairs."""
    modes = _static_tables(cfg.resolution)[0]
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    index = {(mm, nn): k for k, (mm, nn) in enumerate(zip(m, n))}
    ntor = int(cfg.resolution.ntor)
    pos = np.asarray([index[(1, j)] for j in range(1, ntor + 1)], dtype=int)
    neg = np.asarray([index[(1, -j)] for j in range(1, ntor + 1)], dtype=int)
    return pos, neg


def _dof_projector(cfg: ImplicitConfig, dof_mask: SpectralState) -> Callable:
    """Symmetric idempotent projector onto the evolved-dof subspace.

    Elementwise ``dof_mask`` plus, for ``lconm1`` runs, the m=1 pair
    symmetrization: in the signed packing the released constrained Z force
    obeys ``gc(m=1, +n) = gc(m=1, -n)`` exactly (``force_Z_cs`` zeroed at
    convergence, so both entries gather ``sc/2``) — the *antisymmetric*
    combination is a frozen direction, not representable by an entry mask.
    ``P`` therefore averages the ``Z_sin`` pair columns (3D) resp.
    antisymmetrizes ``Z_cos`` (``lasym``, where ``force_Z_cc`` is zeroed and
    the gather is ``(cc +/- ss)/2``).  The mask is pre-equalized on the pair
    columns (see :func:`_dof_mask`), so ``P`` is symmetric and idempotent.
    """
    lconm1 = bool(cfg.lconm1)
    lthreed = int(cfg.resolution.ntor) > 0
    lasym = bool(cfg.resolution.lasym)
    pos, neg = (_m1_pair_columns(cfg) if (lconm1 and lthreed)
                else (np.zeros(0, int), np.zeros(0, int)))

    def P(t: SpectralState) -> SpectralState:
        out = {name: getattr(dof_mask, name) * getattr(t, name)
               for name in _STATE_FIELDS}
        if lconm1 and lthreed and pos.size:
            v = out["Z_sin"]
            sym = 0.5 * (v[:, pos] + v[:, neg])
            out["Z_sin"] = v.at[:, pos].set(sym).at[:, neg].set(sym)
            if lasym:
                v = out["Z_cos"]
                anti = 0.5 * (v[:, pos] - v[:, neg])
                out["Z_cos"] = v.at[:, pos].set(anti).at[:, neg].set(-anti)
        return SpectralState(**out)

    return P


def _assemble(z: SpectralState, rt_p: SolverRuntime, frozen: SpectralState,
              P: Callable, edge_mask: SpectralState) -> SpectralState:
    """``x = frozen + P(z - frozen) + edge_mask*(boundary(p) - frozen)``.

    At ``z = P(x*)`` and the solved parameters this reproduces ``x*``
    exactly; the edge term carries the differentiable boundary dependence
    (``edge_mask`` and the dof subspace are disjoint).
    """
    setup = rt_p.setup
    boundary = dict(R_cos=setup.boundary_R_cos, R_sin=setup.boundary_R_sin,
                    Z_cos=setup.boundary_Z_cos, Z_sin=setup.boundary_Z_sin)
    dz = P(jax.tree.map(lambda a, b: a - b, z, frozen))
    out = {}
    for name in _STATE_FIELDS:
        x = getattr(frozen, name) + getattr(dz, name)
        if name in boundary:
            e = getattr(edge_mask, name)
            x = x + e * (boundary[name][None, :] - getattr(frozen, name))
        out[name] = x
    return SpectralState(**out)


def residual_fn(cfg: ImplicitConfig, frozen: SpectralState,
                dof_mask: SpectralState,
                formulation: str = "preconditioned") -> Callable:
    """Return the implicit residual ``F(z, params) -> masked force pytree``.

    ``formulation="preconditioned"`` (default, used by the adjoint): the
    self-consistently preconditioned ``gc`` of a fresh
    :func:`evaluate_forces` pass — see the module docstring for why its
    implicit gradients coincide with the raw formulation.
    ``formulation="raw"``: the un-preconditioned (``scalxc``-scaled, m=1
    rotated/zeroed) spectral force — same root, same gradients, but the
    adjoint GMRES then runs without preconditioning (diagnostic only).
    """
    edge_mask = _edge_mask(cfg)
    P = _dof_projector(cfg, dof_mask)

    if formulation == "preconditioned":

        # jax.jit the residual so its linearization (``jax.vjp``/``jax.jvp``
        # in the adjoint and the forward-mode Jacobian) compiles as a single
        # reusable XLA sub-computation instead of being re-inlined into the
        # enclosing ``jax.grad``/``jacrev`` program.  Measured: this shrinks
        # the reverse-gradient *compile* working set — the dominant term of
        # the implicit-gradient peak (R16 profiling: the peak is XLA compile
        # memory, not runtime buffers) — by ~15-20% and speeds the compile,
        # bit-identically (the gradient value is unchanged).
        @jax.jit
        def F(z: SpectralState, params: ImplicitParams) -> SpectralState:
            rt_p = runtime_from_params(params, cfg)
            x = _assemble(z, rt_p, frozen, P, edge_mask)
            gc, _, _ = evaluate_forces(x, rt_p)
            return P(gc)

        return F

    if formulation != "raw":
        raise ValueError(f"unknown formulation {formulation!r}")

    def F_raw(z: SpectralState, params: ImplicitParams) -> SpectralState:
        rt_p = runtime_from_params(params, cfg)
        x = _assemble(z, rt_p, frozen, P, edge_mask)
        setup = rt_p.setup
        s = setup.s_full
        (R_cos, R_sin, Z_cos, Z_sin), geometry = _geometry(x, rt_p)
        jacobian = half_mesh_jacobian(geometry, s=s)
        metrics = metric_elements(geometry, s=s)
        fields = magnetic_fields(
            geometry=geometry, jacobian=jacobian, metrics=metrics,
            trig=rt_p.trig, s=s, phips=setup.phips, phipf=setup.phipf,
            chips=setup.chips, signgs=setup.signgs, gamma=rt_p.gamma,
            mass=setup.mass, ncurr=setup.ncurr, enclosed_current=setup.icurv,
        )
        tcon = constraint_scaling(
            tcon0=rt_p.tcon0, geometry=geometry, jacobian=jacobian,
            total_pressure=fields.total_pressure, trig=rt_p.trig, s=s,
        )
        forces = mhd_forces(
            geometry=geometry, jacobian=jacobian, metrics=metrics,
            fields=fields, R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
            modes=rt_p.modes, trig=rt_p.trig, s=s, phipf=setup.phipf,
            tcon=tcon, signgs=setup.signgs, rcon0=rt_p.rcon0, zcon0=rt_p.zcon0,
        )
        spectral = spectral_mhd_forces(
            forces, mpol=cfg.resolution.mpol, ntor=cfg.resolution.ntor,
            trig=rt_p.trig, include_edge=False,
        )
        rotated = m1_residue_rotation(spectral, lconm1=setup.lconm1)
        # converged branch: fsqz < threshold zeroes the constrained m=1 Z force
        released = zero_m1_z_force(rotated, jnp.asarray(True))
        scaled = scalxc_scale_force(released, s=s)
        return P(_force_to_state(scaled, rt_p))

    return F_raw


def _dof_mask(x_star: SpectralState, rt: SolverRuntime,
              cfg: ImplicitConfig, seed: int = 0) -> SpectralState:
    """Evolved-dof mask from the structural zero patterns of ``gc``.

    Host-side, once per forward solve.  A dof is an entry where (a) the
    residual can respond (row support: ``gc`` nonzero at a generically
    perturbed state — structural zeros stay exactly ``0.0`` in floating
    point) and (b) the residual depends on the entry (column support: one
    VJP of ``gc`` with a random cotangent).  This excludes, exactly: the
    fixed R/Z edge row (``include_edge=False``), the structurally zero
    families of symmetric runs, the released m=1 constrained Z combinations
    (zeroed force at convergence) and the lambda axis row (overwritten by the
    ``totzsp`` axis closure, so no equation depends on it).
    """
    rng = np.random.default_rng(seed)
    scale = max(float(max(np.max(np.abs(np.asarray(getattr(x_star, f))), initial=0.0)
                          for f in _STATE_FIELDS)), 1.0)

    def evaluate(x):
        gc, _, diag = evaluate_forces(x, rt)
        return gc, bool(np.asarray(diag.jacobian_sign_changed))

    def perturbed(k, eps):
        r = np.random.default_rng(seed + k)
        return jax.tree.map(
            lambda a: jnp.asarray(np.asarray(a) + eps * scale
                                  * r.standard_normal(np.shape(a))), x_star)

    gc_fn = lambda x: evaluate_forces(x, rt)[0]  # noqa: E731

    rows = jax.tree.map(lambda a: np.zeros(np.shape(a), dtype=bool), x_star)
    cols = jax.tree.map(lambda a: np.zeros(np.shape(a), dtype=bool), x_star)
    for k in range(2):
        # A perturbation that flips the Jacobian-sign flag suppresses the
        # bcovar cache refresh (faclam etc. stay zero) and corrupts the zero
        # patterns; shrink until the flag stays clear.
        for eps in (1e-7, 1e-9, 1e-11, 0.0):
            xp = perturbed(k, eps)
            gc, flipped = evaluate(xp)
            if not flipped:
                break
        else:  # pragma: no cover - converged state must have a valid Jacobian
            raise RuntimeError("Jacobian sign flag set at the converged state")
        rows = jax.tree.map(lambda r, g: r | (np.asarray(g) != 0.0), rows, gc)
        _, vjp = jax.vjp(gc_fn, xp)
        ct = jax.tree.map(
            lambda a: jnp.asarray(rng.standard_normal(np.shape(a))), x_star)
        g = vjp(ct)[0]
        cols = jax.tree.map(lambda c, gg: c | (np.asarray(gg) != 0.0), cols, g)

    edge = _edge_mask(cfg)
    mask_np = jax.tree.map(
        lambda r, c, e: (r & c & (np.asarray(e) == 0.0)).astype(np.float64),
        rows, cols, edge)
    # Equalize the m=1 pair columns so the pair projector commutes with the
    # mask (see _dof_projector); the pairs share one force value, so their
    # supports coincide structurally anyway.
    if cfg.lconm1 and int(cfg.resolution.ntor) > 0:
        pos, neg = _m1_pair_columns(cfg)
        for name in ("Z_sin",) + (("Z_cos",) if cfg.resolution.lasym else ()):
            arr = getattr(mask_np, name)
            both = arr[:, pos] * arr[:, neg]
            arr[:, pos] = both
            arr[:, neg] = both
    return jax.tree.map(jnp.asarray, mask_np)


# ---------------------------------------------------------------------------
# Forward solve (opaque host solver behind pure_callback) + custom VJP
# ---------------------------------------------------------------------------


# cfg -> last converged SpectralState
_HOT_CACHE: weakref.WeakKeyDictionary[ImplicitConfig, SpectralState] = \
    weakref.WeakKeyDictionary()

# cfg -> (params-bytes key, SolveResult): one-entry memo of the LAST solve.
# scipy trust-region drivers evaluate jac(x) at exactly the x that fun(x)
# just converged (DESC's ``_update_equilibrium``/``f_where_x`` pattern), so
# this removes one full equilibrium solve per accepted iterate (plan R25.1).
_LAST_SOLVE: weakref.WeakKeyDictionary[ImplicitConfig, tuple[bytes, SolveResult]] = \
    weakref.WeakKeyDictionary()

# cfg -> one-shot SpectralState seed for the NEXT host solve (plan R25.4):
# the optimizer's trial evaluation deposits the DESC-style first-order
# perturbation prediction ``x_ref + sum_j (dx)_j dz_j`` (arXiv:2203.15927,
# ``eq.perturb`` before ``eq.solve``) right before the solve that consumes
# (pops) it.  A missing/failed seed falls back silently to the plain
# ``_HOT_CACHE`` hot restart — only the initial guess changes, never the
# fixed point.
_PERTURB_SEED: weakref.WeakKeyDictionary[ImplicitConfig, SpectralState] = \
    weakref.WeakKeyDictionary()

# cfg -> {"solves": int, "iterations": int}: cumulative host forward-solve
# effort of a config (memo hits excluded) — the instrumentation behind the
# R25.4 warm-start benchmarks (``optimize.least_squares`` attaches it to the
# scipy result as ``solve_stats``).
_SOLVE_STATS: weakref.WeakKeyDictionary[ImplicitConfig, dict[str, int]] = \
    weakref.WeakKeyDictionary()

# Single-slot relay for typed host exceptions (module docstring, "Zero-crash
# typed errors through the callback"): ``_host_solve_and_mask`` deposits the
# :class:`VmecError` it caught right before raising the short sentinel, and
# ``_callback_solve`` pops and re-raises it.  Host callbacks are serialized
# per solve, so one slot cannot race.
_HOST_ERROR: list[VmecError] = []


def _params_key(params: ImplicitParams) -> bytes:
    return b"".join(np.asarray(leaf, dtype=np.float64).tobytes()
                    for leaf in jax.tree.leaves(params))


def _host_solve(cfg: ImplicitConfig, params: ImplicitParams) -> SolveResult:
    key = _params_key(params)
    hit = _LAST_SOLVE.get(cfg)
    if hit is not None and hit[0] == key:
        _PERTURB_SEED.pop(cfg, None)  # already solved: drop the stale seed
        return hit[1]
    inp2 = input_with_params(cfg.inp, params)
    hot = _HOT_CACHE.get(cfg) if cfg.hot_restart else None
    perturb = _PERTURB_SEED.pop(cfg, None) if cfg.hot_restart else None
    if cfg.multigrid:
        ns_arr = np.asarray(inp2.ns_array)
        ftol_arr = np.asarray(inp2.ftol_array, dtype=float).copy()
        ftol_arr[-1] = cfg.ftol
        # NITER-exhausted final stages still return a usable (penalized)
        # state — matching the optimize.least_squares trial-solve policy
        # (VMEC2000 behaves the same way).
        run = lambda init: solve_multigrid(  # noqa: E731
            inp2, ns_array=ns_arr, ftol_array=ftol_arr, mode=cfg.mode,
            lconm1=cfg.lconm1, raise_on_max_iterations=False,
            initial_state=init)
    else:
        run = lambda init: solve(  # noqa: E731
            inp2, cfg.resolution, ftol=cfg.ftol,
            max_iterations=cfg.max_iterations, mode=cfg.mode,
            lconm1=cfg.lconm1, initial_state=init)
    # Seed ladder: perturbation prediction -> plain hot restart -> cold.
    # A bad warm seed must not fail the trial (only the initial guess is at
    # stake — every rung converges to the same fixed point).
    attempts = [s for s in (perturb, hot) if s is not None] + [None]
    for k, init in enumerate(attempts):
        try:
            result = run(init)
            break
        except Exception:
            if k == len(attempts) - 1:
                raise
    if cfg.hot_restart and bool(result.converged):
        _HOT_CACHE[cfg] = result.state
    _LAST_SOLVE[cfg] = (key, result)
    stats = _SOLVE_STATS.setdefault(cfg, {"solves": 0, "iterations": 0})
    stats["solves"] += 1
    stats["iterations"] += int(result.iterations)
    return result


# structural-signature -> host dof mask.  The mask is a *structural* property
# (module docstring / ``_dof_mask``): it depends only on the resolution, the
# symmetry/lconm1 mode families and the ncurr force branch — NOT on the
# parameter values, and NOT on the ``ImplicitConfig`` object identity.  Keying
# by identity (the previous ``WeakKeyDictionary``) missed on every ``run`` /
# ``make_config`` call, because :class:`ImplicitConfig` is ``eq=False`` and
# ``make_config`` mints a fresh object per call, so an optimization loop that
# calls ``im.run`` hundreds of times at one resolution recomputed the mask
# (the eager ``evaluate_forces`` x2 + VJP, measured 20-40 s) every time.
# A plain dict keyed by the structural signature hits across all such calls.
_MASK_CACHE: dict[tuple, SpectralState] = {}


def _mask_cache_key(cfg: ImplicitConfig) -> tuple:
    """Structural identity of the dof mask (see :func:`_dof_mask`).

    Everything the mask's zero pattern depends on and nothing else, so two
    configs that differ only in parameter values, tolerances, iteration caps,
    the multigrid schedule or object identity share one entry.  ``resolution``
    is a value-hashable frozen dataclass (``mpol/ntor/ntheta/nzeta/nfp/lasym/
    ns``); ``lconm1`` drives the m=1 pair equalization; ``ncurr`` is carried as
    a conservative safety margin for the current-constrained force branch.
    """
    return (cfg.resolution, bool(cfg.lconm1), int(cfg.inp.ncurr))


def _host_solve_and_mask(cfg: ImplicitConfig, params_np) -> tuple:
    _HOST_ERROR.clear()  # fresh callback: drop any stale relayed error
    params = jax.tree.map(jnp.asarray, params_np)
    try:
        result = _host_solve(cfg, params)
    except VmecError as exc:
        # Relay the typed exception (module docstring, "Zero-crash typed
        # errors through the callback"): stash it and raise a SHORT sentinel
        # — pure_callback would otherwise bury it in a multi-KB traceback
        # dump with the typed class lost.
        _HOST_ERROR.append(exc)
        raise RuntimeError(
            f"host equilibrium solve failed with {type(exc).__name__} "
            "(re-raised typed at the pure_callback call site)") from None
    as_np = lambda t: jax.tree.map(  # noqa: E731
        lambda a: np.asarray(a, dtype=np.float64), t)
    # Prime the per-cfg-identity differentiable template *concretely* here (this
    # host callback runs outside any trace).  The jitted residual ``F`` later
    # calls ``runtime_from_params(params, cfg)`` -> ``_template_runtime(cfg)``
    # under a ``jax.jit`` trace, where the host-side ``run_setup`` cannot run;
    # its ``lru_cache`` must therefore be filled by a concrete call first.  The
    # forward solve used to fill it as a side effect of building ``rt`` for the
    # mask — preserve that guarantee even when the structural mask cache hits
    # and skips that rebuild (``_template_runtime`` is keyed by object identity,
    # so a fresh ``make_config``/``run`` config needs its own concrete prime).
    _template_runtime(cfg)
    # The dof mask captures *structural* zero patterns (resolution, symmetry,
    # lconm1 pairing) — invariant across the parameter values of one config,
    # so repeated solves of an optimization reuse the first solve's mask.
    # Keyed by structural signature (not object identity) so a fresh
    # ``make_config``/``run`` at the same resolution hits — see ``_MASK_CACHE``.
    cache_key = _mask_cache_key(cfg)
    mask = _MASK_CACHE.get(cache_key)
    if mask is None:
        rt = runtime_from_params(params, cfg)
        mask = as_np(_dof_mask(result.state, rt, cfg))
        _MASK_CACHE[cache_key] = mask
    return as_np(result.state), mask


def _callback_solve(params: ImplicitParams, cfg: ImplicitConfig):
    """``pure_callback`` host solve returning ``(state, dof_mask)``.

    Shared by :func:`solve_implicit`, ``_solve_implicit_fwd`` and
    :func:`solve_implicit_with_aux`.  Re-raises the ORIGINAL typed
    :class:`VmecError` deposited in ``_HOST_ERROR`` by
    ``_host_solve_and_mask`` (``from None`` suppresses the noisy
    ``JaxRuntimeError`` context) so eager callers see the short typed
    exception.  Under ``jax.jit`` this try/except is trace-time only and the
    (short, sentinel-carrying) runtime error surfaces at the jit boundary
    instead — see the module docstring.
    """
    try:
        return jax.pure_callback(
            functools.partial(_host_solve_and_mask, cfg),
            (_state_struct(cfg), _state_struct(cfg)), params,
        )
    except Exception:
        if _HOST_ERROR:
            raise _HOST_ERROR.pop() from None
        raise


def _state_struct(cfg: ImplicitConfig) -> SpectralState:
    mn = int(np.asarray(_static_tables(cfg.resolution)[0].m).size)
    s = jax.ShapeDtypeStruct((cfg.resolution.ns, mn), jnp.float64)
    return SpectralState(*(s,) * 6)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def solve_implicit(params: ImplicitParams, cfg: ImplicitConfig) -> SpectralState:
    """Differentiable equilibrium solve: ``params -> converged SpectralState``.

    Forward: the fast opaque host solver (``jax.pure_callback``; multigrid /
    adaptive control invisible to AD — only the fixed point defines the
    derivative).  Backward: preconditioned matrix-free adjoint (module
    docstring).  Compose with :func:`runtime_from_params` and the helpers
    (:func:`mhd_energy`, :func:`aspect_ratio`, ...) to build objectives, or
    use :func:`run`.
    """
    state, _ = _callback_solve(params, cfg)
    return state


def _solve_implicit_fwd(params, cfg):
    state, mask = _callback_solve(params, cfg)
    return state, (params, state, mask)


def solve_implicit_with_aux(params: ImplicitParams, cfg: ImplicitConfig):
    """Return ``(state, dof_mask)`` using the same callback as solve_implicit."""
    return _callback_solve(params, cfg)


def _adjoint_solve(A, b, cfg: ImplicitConfig, *, x0=None, max_restarts=None):
    """Adjoint linear solve ``(dF/dz)^T lambda = b`` via ``solvax.gmres``.

    ``solvax.gmres`` (roadmap R18b shared-solver consolidation) operates on
    flat ``(n,)`` vectors, so the :class:`SpectralState` pytree ``b`` is
    raveled to a flat vector and the matrix-free operator ``A`` wrapped to
    match; the flat GMRES is exactly the pytree GMRES because ``ravel_pytree``
    is a linear isomorphism.  Tolerances/limits mirror the previous
    ``jax.scipy.sparse.linalg.gmres`` call (``rtol = adjoint_tol``, ``atol =
    0``, Arnoldi cycle size ``adjoint_restart``, up to ``adjoint_maxiter``
    restarts), so the adjoint accuracy is unchanged.

    ``x0`` (pytree like ``b``) warm-starts GMRES — solvax checks the initial
    residual before the first Arnoldi cycle, so a warm start that already
    meets the tolerance costs exactly one matvec (the plan R25.2 corrector
    pass over the block-tridiagonal direct solve).  ``max_restarts``
    overrides ``cfg.adjoint_maxiter`` for such short corrector budgets.
    """
    b_flat, unravel = ravel_pytree(b)

    def matvec(v):
        return ravel_pytree(A(unravel(v)))[0]

    sol = _solvax_gmres(
        matvec, b_flat,
        x0=None if x0 is None else ravel_pytree(x0)[0],
        rtol=cfg.adjoint_tol, atol=0.0, restart=cfg.adjoint_restart,
        max_restarts=(cfg.adjoint_maxiter if max_restarts is None
                      else int(max_restarts)),
    )
    return unravel(sol.x), sol


# Recycle-space width for _recycled_solve (plan R25.3).  GCROT keeps k
# deflation directions in a fixed-shape (n, k) pair, so k trades warm-start
# overhead (k re-orthonormalization matvecs per solve) against deflation
# depth; 10 matches the solvax default and the GCRO-DR literature.
_RECYCLE_K = 10


def _recycled_solve(A, b, cfg: ImplicitConfig, recycle):
    """Linearized solve via :func:`solvax.gcrot` with subspace recycling.

    Same operator wrapping, tolerance (``rtol = adjoint_tol``, ``atol = 0``),
    cycle size and restart budget as :func:`_adjoint_solve`; check
    ``sol.converged`` — a warm recycle pair changes the iteration path, and
    a solve that exhausts ``adjoint_maxiter`` returns whatever residual it
    reached.  ``recycle`` is a ``(C, U)`` pair of shape ``(n, _RECYCLE_K)``
    (an all-zero pair degenerates to a cold start); the updated pair is
    returned on ``sol.recycle`` so callers can thread it through a sequence
    of solves sharing (or slowly varying) the operator — the plan R25.3
    per-dof implicit-Jacobian loop (opt-in via
    ``least_squares(..., recycle=True)``; see the measured caveat there).
    """
    b_flat, unravel = ravel_pytree(b)

    def matvec(v):
        return ravel_pytree(A(unravel(v)))[0]

    sol = _solvax_gcrot(
        matvec, b_flat, rtol=cfg.adjoint_tol, atol=0.0,
        m=cfg.adjoint_restart, k=_RECYCLE_K,
        max_restarts=cfg.adjoint_maxiter, recycle=recycle,
    )
    return unravel(sol.x), sol


def _solve_implicit_bwd(cfg, res, gbar):
    params, x_star, dof_mask = res
    frozen = jax.lax.stop_gradient(x_star)
    edge_mask = _edge_mask(cfg)
    P = _dof_projector(cfg, dof_mask)
    F = residual_fn(cfg, frozen, dof_mask)

    z_star = P(x_star)

    # (dF/dz)^T lambda = P gbar, matrix-free (one linearization reused).
    _, vjp_z = jax.vjp(lambda z: F(z, params), z_star)
    b = P(gbar)
    lam, _ = _adjoint_solve(lambda v: vjp_z(v)[0], b, cfg)

    # -lambda^T dF/dp ...
    _, vjp_p = jax.vjp(lambda prm: F(z_star, prm), params)
    g1 = vjp_p(jax.tree.map(jnp.negative, lam))[0]
    # ... plus the direct boundary(p) path through the fixed edge row.
    _, vjp_p2 = jax.vjp(
        lambda prm: _assemble(z_star, runtime_from_params(prm, cfg),
                              frozen, P, edge_mask), params)
    g2 = vjp_p2(gbar)[0]
    return (jax.tree.map(jnp.add, g1, g2),)


solve_implicit.defvjp(_solve_implicit_fwd, _solve_implicit_bwd)


def implicit_state_pullback_multi_rhs(
    params: ImplicitParams,
    cfg: ImplicitConfig,
    x_star: SpectralState,
    dof_mask: SpectralState,
    gbar_batch: SpectralState,
) -> ImplicitParams:
    """Batched state-cotangent pullback with shared implicit-linearization setup.

    This preserves the scalar solve_implicit VJP and only adds a helper for
    callers that already have several state cotangents for the same fixed
    point.  It reuses the residual/projector/VJP setup once, then applies the
    existing single-RHS GMRES per row.
    """
    frozen = jax.lax.stop_gradient(x_star)
    edge_mask = _edge_mask(cfg)
    P = _dof_projector(cfg, dof_mask)
    F = residual_fn(cfg, frozen, dof_mask)
    z_star = P(x_star)

    _, vjp_z = jax.vjp(lambda z: F(z, params), z_star)
    _, vjp_p = jax.vjp(lambda prm: F(z_star, prm), params)
    _, vjp_p2 = jax.vjp(
        lambda prm: _assemble(z_star, runtime_from_params(prm, cfg),
                              frozen, P, edge_mask), params)

    rhs_batch = jax.vmap(P)(gbar_batch)
    lam_batch = jax.vmap(
        lambda rhs: _adjoint_solve(lambda v: vjp_z(v)[0], rhs, cfg)[0]
    )(rhs_batch)
    g1_batch = jax.vmap(lambda lam: vjp_p(jax.tree.map(jnp.negative, lam))[0])(lam_batch)
    g2_batch = jax.vmap(lambda gbar: vjp_p2(gbar)[0])(gbar_batch)
    return jax.tree.map(jnp.add, g1_batch, g2_batch)


def adjoint_matvec(cfg: ImplicitConfig, params: ImplicitParams,
                   x_star: SpectralState, dof_mask: SpectralState,
                   formulation: str = "preconditioned") -> Callable:
    """``v -> (dF/dz)^T v`` for tests/diagnostics (both formulations)."""
    frozen = jax.lax.stop_gradient(x_star)
    F = residual_fn(cfg, frozen, dof_mask, formulation=formulation)
    z_star = _dof_projector(cfg, dof_mask)(x_star)
    _, vjp_z = jax.vjp(lambda z: F(z, params), z_star)
    return lambda v: vjp_z(v)[0]


# ---------------------------------------------------------------------------
# Differentiable derived quantities (objective building blocks)
# ---------------------------------------------------------------------------


# The shared geometry->fields->energies pipeline (statephysics.py), jitted
# here for the same reason as the implicit residual (see ``residual_fn``):
# the derived-quantity objectives (:func:`mhd_energy`, :func:`aspect_ratio`,
# ...) all route through it, so compiling it once as a reusable
# sub-computation cuts the enclosing ``jax.grad``/``jacrev`` compile working
# set (R16/R17.2 memory profiling).
_field_chain = jax.jit(_field_chain_shared)


def mhd_energy(state: SpectralState, rt: SolverRuntime) -> tuple[Array, Array]:
    """``(wb, wp)`` in the wout normalization (``bcovar.f``), differentiable."""
    _, _, _, _, energies = _field_chain(state, rt)
    return energies.wb, energies.wp


def plasma_volume(state: SpectralState, rt: SolverRuntime) -> Array:
    """Plasma volume ``volume_p`` [m^3] (``= (2 pi)^2 * hs * sum vp``).

    Quadrature note (Item I.7): this is the implicit module's historical
    differential-volume sum, pinned by :class:`ImplicitSolution.volume` and
    the FD-cached gradient tables of ``tests/test_implicit_grad.py``.  The
    canonical wout-parity boundary quadrature of the same scalar is
    :func:`vmex.core.statephysics.volume` (re-exported as
    ``optimize.volume``); the two agree to quadrature resolution.
    """
    _, _, _, _, energies = _field_chain(state, rt)
    return (2.0 * jnp.pi) ** 2 * jnp.abs(energies.volume)


def _edge_physical(state: SpectralState, rt: SolverRuntime):
    R_cos, R_sin, Z_cos, Z_sin = _physical_coefficients(
        state, modes=rt.modes, lthreed=rt.setup.lthreed,
        lasym=rt.setup.lasym, lconm1=rt.setup.lconm1,
    )
    scale = jnp.asarray(1.0 / physical_to_internal_scale(rt.modes, rt.trig))
    return (R_cos[-1] * scale, R_sin[-1] * scale,
            Z_cos[-1] * scale, Z_sin[-1] * scale)


def aspect_ratio(state: SpectralState, rt: SolverRuntime,
                 *, ntheta: int = 128, nzeta: int = 32) -> Array:
    """VMEC-convention aspect ratio ``Rmajor_p / Aminor_p`` (differentiable).

    ``Aminor_p = sqrt(<cross-section area>_zeta / pi)`` with the area from
    the shoelace integral ``-oint Z dR/dtheta dtheta`` on the boundary, and
    ``Rmajor_p = volume_p / (2 pi^2 Aminor_p^2)`` (``aspectratio.f``).

    Quadrature note (Item I.7): this is the implicit module's historical
    shoelace-on-a-fresh-grid variant, pinned by
    :class:`ImplicitSolution.aspect` and the FD-cached solovev gradient table
    of ``tests/test_implicit_grad.py``.  The canonical wout-parity
    ``aspectratio.f`` boundary quadrature (internal-grid ``wint`` weights,
    equal to the wout ``aspect`` scalar) is
    :func:`vmex.core.statephysics.aspect_ratio` (re-exported as
    ``optimize.aspect_ratio``); the two agree to quadrature resolution.
    """
    rmnc, rmns, zmnc, zmns = _edge_physical(state, rt)
    m = jnp.asarray(np.asarray(rt.modes.m, dtype=float))
    n = jnp.asarray(np.asarray(rt.modes.n, dtype=float) * rt.resolution.nfp)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / rt.resolution.nfp, nzeta,
                        endpoint=False)
    ang = (m[:, None, None] * theta[None, :, None]
           - n[:, None, None] * zeta[None, None, :])
    cos, sin = jnp.cos(ang), jnp.sin(ang)
    Z = jnp.einsum("k,ktz->tz", zmns, sin) + jnp.einsum("k,ktz->tz", zmnc, cos)
    dRdt = (-jnp.einsum("k,ktz->tz", m * rmnc, sin)
            + jnp.einsum("k,ktz->tz", m * rmns, cos))
    area = jnp.abs(jnp.mean(jnp.sum(-Z * dRdt, axis=0) * (2.0 * jnp.pi / ntheta)))
    aminor = jnp.sqrt(area / jnp.pi)
    vol = plasma_volume(state, rt)
    rmajor = vol / (2.0 * jnp.pi ** 2 * aminor ** 2)
    return rmajor / aminor


def iota_profile(state: SpectralState, rt: SolverRuntime) -> Array:
    """Full-mesh ``iotaf`` (``add_fluxes.f90``), differentiable.

    ``ncurr = 0``: the prescribed profile (p-dependent through ``ai``);
    ``ncurr = 1``: reconstructed from the converged current-constrained
    ``chips`` exactly as in the solver's result assembly.
    """
    setup = rt.setup
    if int(setup.ncurr) != 1:
        return jnp.asarray(setup.iotaf)
    _, _, _, fields, _ = _field_chain(state, rt)
    chips = fields.chips
    phips = jnp.asarray(setup.phips)
    safe = jnp.where(phips != 0.0, phips, 1.0)
    iotas = jnp.where(phips != 0.0, chips / safe, 0.0)
    iotaf = 0.5 * (iotas + jnp.roll(iotas, -1))
    iotaf = iotaf.at[0].set(1.5 * iotas[1] - 0.5 * iotas[2])
    iotaf = iotaf.at[-1].set(1.5 * iotas[-1] - 0.5 * iotas[-2])
    return iotaf


def iota_axis(state: SpectralState, rt: SolverRuntime) -> Array:
    return iota_profile(state, rt)[0]


def iota_edge(state: SpectralState, rt: SolverRuntime) -> Array:
    """Boundary rotational transform ``iotaf[-1]`` (differentiable).

    Naming note (Item I.7): the same physical scalar as
    :func:`vmex.core.statephysics.edge_iota` (``optimize.edge_iota``) —
    identical for ``ncurr = 1``; at ``ncurr = 0`` this evaluates the
    prescribed full-mesh ``iotaf`` endpoint while the wout-parity version
    extrapolates the half-mesh ``iotas``.  ``edge_iota`` is provided as an
    alias here so either spelling works in either module.
    """
    return iota_profile(state, rt)[-1]


edge_iota = iota_edge   # naming-flip alias (see the iota_edge docstring)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImplicitSolution:
    """Differentiable outputs of :func:`run` (a JAX pytree).

    ``runtime`` is the :class:`~vmex.core.solver.SolverRuntime` that
    :func:`run` built internally (``runtime_from_params(params, cfg)``), so
    objective callers can evaluate further ``(state, runtime)`` scalar
    targets without rebuilding it per evaluation.  It is deliberately **not**
    part of the pytree (registered as a dropped field): the solution's
    established pytree structure — six state leaves plus seven scalars — is
    unchanged, and a solution that round-trips through
    ``flatten``/``unflatten`` (e.g. across a ``jax.jit`` boundary) comes back
    with ``runtime = None``.  Inside a ``jax.grad``/``jax.value_and_grad``
    trace of :func:`run` the attribute is available and fully traced, so
    gradients flow through ``runtime``-consuming objectives exactly as
    through an explicit ``runtime_from_params`` rebuild.
    """

    state: SpectralState
    wb: Array
    wp: Array
    wmhd: Array
    volume: Array
    aspect: Array
    iota_axis: Array
    iota_edge: Array
    runtime: SolverRuntime | None = None


_register(ImplicitSolution, drop=("runtime",))


def run(
    source: VmecInput | str,
    params: ImplicitParams | None = None,
    *,
    ns: int | None = None,
    ftol: float | None = None,
    max_iterations: int | None = None,
    mode: str = "cli",
    multigrid: bool = False,
    lconm1: bool = True,
    adjoint_tol: float = 1e-11,
    adjoint_restart: int = 30,
    adjoint_maxiter: int = 300,
) -> ImplicitSolution:
    """Differentiable fixed-boundary equilibrium: input -> outputs pytree.

    ``params`` defaults to :func:`params_from_input`; pass a perturbed /
    traced :class:`ImplicitParams` to differentiate::

        inp = VmecInput.from_file("input.solovev")
        p0 = params_from_input(inp)
        grad = jax.grad(lambda p: run(inp, p).wb)(p0)

    ``wmhd`` follows the printed ``WMHD`` normalization; ``gamma = 1`` inputs
    get ``wmhd = nan`` (as in VMEC).  All outputs are differentiable in
    ``params`` (state via the implicit adjoint; scalars additionally through
    their explicit parameter dependence).

    The returned solution also carries the internally built
    :class:`~vmex.core.solver.SolverRuntime` as ``sol.runtime`` (a
    non-pytree convenience attribute, see :class:`ImplicitSolution`), so
    objective code can evaluate additional ``(state, runtime)`` targets —
    e.g. ``optimize.mean_iota(sol.state, sol.runtime)`` — without repeating
    ``runtime_from_params(params, make_config(...))`` per evaluation.
    """
    inp = VmecInput.from_file(source) if isinstance(source, str) else source
    cfg = make_config(
        inp, ns=ns, ftol=ftol, max_iterations=max_iterations, mode=mode,
        multigrid=multigrid, lconm1=lconm1, adjoint_tol=adjoint_tol,
        adjoint_restart=adjoint_restart, adjoint_maxiter=adjoint_maxiter,
    )
    if params is None:
        params = params_from_input(inp)
    state = solve_implicit(params, cfg)
    rt = runtime_from_params(params, cfg)
    wb, wp = mhd_energy(state, rt)
    vol = plasma_volume(state, rt)
    gamma = float(rt.gamma)
    wmhd = (wb + wp / (gamma - 1.0)) * (2.0 * np.pi) ** 2
    return ImplicitSolution(
        state=state, wb=wb, wp=wp, wmhd=wmhd, volume=vol,
        aspect=aspect_ratio(state, rt),
        iota_axis=iota_axis(state, rt), iota_edge=iota_edge(state, rt),
        runtime=rt,
    )


# ---------------------------------------------------------------------------
# Gradient diagnostics
# ---------------------------------------------------------------------------


def frozen_path_directional_fd(
    params: ImplicitParams,
    cfg: ImplicitConfig,
    metric_fn: Callable[[SpectralState, SolverRuntime], Array],
    tangent: ImplicitParams,
    *,
    h: float = 1e-4,
    newton_steps: int = 20,
    newton_rtol: float = 1e-11,
) -> tuple[float, dict]:
    """Central FD of ``metric_fn`` along ``tangent`` on the *frozen* solve path.

    The correct finite-difference reference for **solver-sensitive** metrics --
    ``iota`` (derived from the current-constrained ``chips`` at ``ncurr=1``),
    the mirror ratio, the magnetic well, the Boozer/QI residual -- whose value
    reads the converged solver state directly rather than through a smooth bulk
    integral (``wb``, ``aspect``, for which a naive re-solve FD is already
    exact and :func:`jax.grad` matches it to ``rtol <= 1e-6``).

    A naive full re-solve at ``params +/- h*tangent`` lets the solver's internal
    convergence logic -- the ``bcovar`` preconditioner, the ``tcon`` constraint
    scaling, the m=1 ``gcz`` zeroing branch (``residue.f90``), the dof mask, the
    multigrid schedule, and exactly where the ``ftol`` crossing lands -- re-form
    slightly differently at each perturbed point.  For a solver-sensitive metric
    that path variation is an O(1) contribution that can inflate or even
    sign-flip the finite difference (measured on ``li383_low_res``:
    ``d(iota_edge)/d(RBC(-1,1)) = -0.773`` from the adjoint, but the naive
    central FD reads ``+0.045`` -- wrong sign).

    The implicit adjoint deliberately does *not* differentiate through that
    logic: it linearizes the fixed point of the **frozen** residual ``F`` (the
    preconditioner / mask / branch captured once at ``params``; see the module
    docstring), which is the stable, physical gradient.  This helper reproduces
    exactly that path -- it captures ``F`` once at ``params`` and Newton-solves
    ``F(z, params +/- h*tangent) = 0`` (matrix-free, the same linearization the
    adjoint uses) from the converged ``z*`` before central-differencing
    ``metric_fn``.  The result therefore equals :func:`jax.grad` of the metric
    contracted with ``tangent`` to solver accuracy -- the gradient check a naive
    re-solve FD cannot provide for these metrics.

    Returns ``(fd, info)`` where ``info['newton_res']`` are the two frozen-solve
    residual norms; confirm they are small (an unconverged frozen solve
    invalidates the comparison).
    """
    x_star, dof_mask = solve_implicit_with_aux(params, cfg)
    frozen = jax.lax.stop_gradient(x_star)
    P = _dof_projector(cfg, dof_mask)
    edge_mask = _edge_mask(cfg)
    F = residual_fn(cfg, frozen, dof_mask)
    z_star = P(x_star)

    def _norm(t: SpectralState) -> float:
        return float(jnp.sqrt(sum(jnp.vdot(v, v).real for v in jax.tree.leaves(t))))

    def _eval(sign: float) -> tuple[float, float]:
        p_h = jax.tree.map(lambda a, d: a + sign * h * d, params, tangent)
        z = z_star
        r0 = max(_norm(F(z, p_h)), 1.0)
        for _ in range(newton_steps):
            fz = F(z, p_h)
            if _norm(fz) <= newton_rtol * r0:
                break
            # Newton step (dF/dz) delta = F(z, p_h), matrix-free forward solve.
            _, jvp = jax.linearize(lambda zz: F(zz, p_h), z)
            delta, _ = _adjoint_solve(jvp, fz, cfg)
            z = jax.tree.map(lambda a, b: a - b, z, delta)
        rt = runtime_from_params(p_h, cfg)
        x = _assemble(z, rt, frozen, P, edge_mask)
        return float(metric_fn(x, rt)), _norm(F(z, p_h))

    val_p, res_p = _eval(+1.0)
    val_m, res_m = _eval(-1.0)
    return (val_p - val_m) / (2.0 * h), {"newton_res": (res_p, res_m), "h": h}
