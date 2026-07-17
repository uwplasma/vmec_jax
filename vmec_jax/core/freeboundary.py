"""Free-boundary solve: NESTOR vacuum coupling around the core solver.

Implements the ``funct3d.f`` free-boundary block (VMEC2000) on top of the
fixed-boundary iteration of :mod:`vmec_jax.core.solver`:

- **Activation**: ``ivac`` starts at -1 and increments on every iteration
  with ``iter2 > 1`` and ``fsqr + fsqz <= 1e-3``; the first vacuum call
  promotes ``ivac`` 0 -> 1 (``vacuum.f``), prints the ``In VACUUM`` block,
  and triggers the soft-start restart (``restart_iter`` with ``irst = 2``:
  state <- best stored state, zero velocity, ``delt *= 0.9``,
  ``iter1 = iter2``, ``ijacob += 1``); ``eqsolve.f`` then prints the
  ``VACUUM PRESSURE TURNED ON`` banner and sets ``ivac = 2``.
- **Cadence**: ``ivacskip = mod(iter2 - iter1, nvacskip)`` (forced 0 while
  ``ivac <= 2``); on full steps (``ivacskip == 0``) the Green-function
  kernel/matrix is rebuilt and ``nvacskip = max(nvskip0,
  1/max(0.1, 1e11*(fsqr+fsqz)))``; on skip steps only the analytic source is
  refreshed against the cached matrix (``scalpot.f``).
- **Edge force**: ``bsqvac + presf(ns)`` enters the R/Z edge force rows via
  the :class:`~vmec_jax.core.solver.SolverRuntime` free-boundary seam
  (``lfreeb/bsqvac_edge/presf_ns_scale`` — see ``solver._evaluate``), the
  edge row is evolved (``jmax = ns``) and the ``rcon0/zcon0`` constraint
  baselines are damped by 0.9 per active iteration (``funct3d.f``).

The iteration itself runs the *same* traced body as the fixed-boundary
lanes.  Scheduling (plan item F.2): the pre-activation fixed-boundary
iterations run as one jitted ``lax.while_loop`` (:func:`_preactivation_lane`)
and the whole post-turn-on steady state — vacuum cadence, constraint damping,
``bsqvac_edge`` refresh and the eqsolve iteration — as another
(:func:`_make_vacuum_lane`); only the single turn-on pass is host-stepped,
because it applies the one-time soft restart and prints the banners.  The
per-iteration numerics are identical to the per-pass host driver.

Known divergence from VMEC2000 (documented): at turn-on VMEC computes the
turn-on iteration's forces from the pre-restart geometry while evolving the
restored state; here the restart is applied *before* the turn-on iteration,
so that iteration's forces come from the restored state.  The golden
free-boundary fixture is chaotic/unconverged past turn-on, so trajectories
are compared structurally, not pointwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from . import profiles as _profiles
from .errors import MORE_ITER_FLAG, SUCCESSFUL_TERM_FLAG
from .fields import magnetic_fields, metric_elements
from .fourier import ModeTable
from .geometry import half_mesh_jacobian
from .input import VmecInput
from .mgrid import MgridField
from .printing import (
    FORCE_ITERATIONS_BANNER, screen_header, screen_line, stage_banner,
    vacuum_banner,
)
from .solver import (
    SolveResult, SolverRuntime, SpectralState,
    _finalize, _geometry, _initial_carry, _initial_state, _make_body,
    _result_from_carry, _zero_cache, prepare_runtime, resolution_from_input,
)
from .transforms import register_pytree_dataclass as _register
from .vacuum import (
    VacuumBasis, VacuumBoundary, make_vacuum_solver, vacuum_basis,
    vacuum_channels,
)

__all__ = [
    "FreeBoundaryState",
    "boundary_from_coefficients",
    "solve_free_boundary",
]

Array = Any
MU0 = 4.0e-7 * np.pi

#: funct3d.f vacuum activation threshold on fsqr + fsqz.
ACTIVATION_FSQ = 1.0e-3


# ---------------------------------------------------------------------------
# Boundary surface synthesis (NESTOR surface.f)
# ---------------------------------------------------------------------------


def boundary_from_coefficients(
    *,
    rmnc: np.ndarray,
    zmns: np.ndarray,
    rmns: np.ndarray | None,
    zmnc: np.ndarray | None,
    modes: ModeTable,
    basis: VacuumBasis,
) -> VacuumBoundary:
    """Sample the boundary surface on the NESTOR grid (``surface.f``).

    ``rmnc``... are wout-convention edge coefficients over the signed
    ``modes`` table.  Angles: ``theta/zeta`` from ``basis`` (per-period
    ``zeta``); ``xn = n*nfp`` so all v-derivatives are geometric-phi
    derivatives, exactly as ``surface.f``.
    """
    xm = np.asarray(modes.m, dtype=float)
    xn = np.asarray(modes.n, dtype=float) * float(basis.nfp)
    th = np.asarray(basis.theta, dtype=float)[:, None]
    # ``basis.zeta`` spans [0, 2*pi) per field period; the geometric toroidal
    # angle is ``phi = zeta * onp`` (onp = 1/nfp).  The wout-convention phase
    # is ``m*theta - xn*phi`` with ``xn = n*nfp`` (so all v-derivatives below
    # are geometric-phi derivatives, matching surface.f and the ``onp`` folding
    # in ``vacuum.py``/``external_field_channels``).  Using ``zeta`` directly
    # here double-counts nfp and mis-places every n != 0 harmonic toroidally.
    ze = np.asarray(basis.zeta, dtype=float)[:, None] * float(basis.onp)
    arg = th * xm[None, :] - ze * xn[None, :]
    cosmn = np.cos(arg)
    sinmn = np.sin(arg)

    rc = np.asarray(rmnc, dtype=float)
    zs = np.asarray(zmns, dtype=float)
    R = cosmn @ rc
    Z = sinmn @ zs
    Ru = -(sinmn * xm[None, :]) @ rc
    Rv = (sinmn * xn[None, :]) @ rc
    Zu = (cosmn * xm[None, :]) @ zs
    Zv = -(cosmn * xn[None, :]) @ zs
    ruu = -(cosmn * (xm * xm)[None, :]) @ rc
    ruv = (cosmn * (xm * xn)[None, :]) @ rc
    rvv = -(cosmn * (xn * xn)[None, :]) @ rc
    zuu = -(sinmn * (xm * xm)[None, :]) @ zs
    zuv = (sinmn * (xm * xn)[None, :]) @ zs
    zvv = -(sinmn * (xn * xn)[None, :]) @ zs
    if rmns is not None and zmnc is not None:
        rs = np.asarray(rmns, dtype=float)
        zc = np.asarray(zmnc, dtype=float)
        R = R + sinmn @ rs
        Z = Z + cosmn @ zc
        Ru = Ru + (cosmn * xm[None, :]) @ rs
        Rv = Rv - (cosmn * xn[None, :]) @ rs
        Zu = Zu - (sinmn * xm[None, :]) @ zc
        Zv = Zv + (sinmn * xn[None, :]) @ zc
        ruu = ruu - (sinmn * (xm * xm)[None, :]) @ rs
        ruv = ruv + (sinmn * (xm * xn)[None, :]) @ rs
        rvv = rvv - (sinmn * (xn * xn)[None, :]) @ rs
        zuu = zuu - (cosmn * (xm * xm)[None, :]) @ zc
        zuv = zuv + (cosmn * (xm * xn)[None, :]) @ zc
        zvv = zvv - (cosmn * (xn * xn)[None, :]) @ zc

    shape = (int(basis.ntheta3), int(basis.nzeta))
    return VacuumBoundary(
        R=R.reshape(shape), Z=Z.reshape(shape),
        Ru=Ru.reshape(shape), Zu=Zu.reshape(shape),
        Rv=Rv.reshape(shape), Zv=Zv.reshape(shape),
        ruu=ruu.reshape(shape), ruv=ruv.reshape(shape), rvv=rvv.reshape(shape),
        zuu=zuu.reshape(shape), zuv=zuv.reshape(shape), zvv=zvv.reshape(shape),
    )


def _edge_fourier(state: SpectralState, rt: SolverRuntime):
    """Edge-row wout-convention coefficients (``convert.f`` before vacuum)."""
    from .residuals import m1_constrained_to_physical
    from .transforms import physical_to_internal_scale

    setup = rt.setup
    R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos,
        modes=rt.modes, lthreed=setup.lthreed, lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    scale = 1.0 / physical_to_internal_scale(rt.modes, rt.trig)
    rmnc = np.asarray(R_cos)[-1] * scale
    zmns = np.asarray(Z_sin)[-1] * scale
    if setup.lasym:
        rmns = np.asarray(R_sin)[-1] * scale
        zmnc = np.asarray(Z_cos)[-1] * scale
    else:
        rmns = zmnc = None
    return rmnc, zmns, rmns, zmnc


# ---------------------------------------------------------------------------
# Axis-filament plasma-current field (tolicu.f + belicu.f)
# ---------------------------------------------------------------------------


def axis_current_field(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    axis_r: np.ndarray,
    axis_z: np.ndarray,
    nfp: int,
    plascur: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Biot-Savart field of the net toroidal current on the magnetic axis.

    Port of the legacy parity-proven ``axis_current_field_vmec_filament``
    (VMEC ``tolicu.f`` axis filament across field periods + LIBSTELL
    ``bsc_b`` segment kernel with ``eps_sq`` regularization).  ``plascur``
    is VMEC's ``ctor`` (mu0*A, ``bcovar.f`` sign convention); the filament
    current is ``+plascur/mu0`` exactly as ``tolicu.f`` (the legacy port
    used the opposite sign because its ``plascur_edge_from_bcovar`` carried
    ``-signgs`` instead of ``bcovar.f``'s ``+signgs``).
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    axis_r = np.asarray(axis_r, dtype=float).reshape(-1)
    axis_z = np.asarray(axis_z, dtype=float).reshape(-1)
    ntheta, nv = R.shape
    current = float(plascur) / MU0
    if (not np.isfinite(current)) or current == 0.0:
        z = np.zeros_like(R)
        return z, z, z

    nfper = max(1, int(nfp))
    nvper = 64 if nv == 1 else nfper
    alvp = (2.0 * np.pi / float(max(1, nv))) / float(nfper)
    cosuv = np.cos(alvp * np.arange(nv, dtype=float))
    sinuv = np.sin(alvp * np.arange(nv, dtype=float))
    alp_per = 2.0 * np.pi / float(nvper)
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))

    # tolicu.f: axis points over all periods (loop closed below).
    x0 = axis_r[None, :] * cosuv[None, :]
    y0 = axis_r[None, :] * sinuv[None, :]
    xpts = np.zeros((3, nvper * nv), dtype=float)
    for kper in range(nvper):
        sl = slice(kper * nv, (kper + 1) * nv)
        xpts[0, sl] = cosper[kper] * x0 - sinper[kper] * y0
        xpts[1, sl] = sinper[kper] * x0 + cosper[kper] * y0
        xpts[2, sl] = axis_z
    # bsc_construct('fil_loop'): drop zero-length segments, close the loop.
    keep = [0]
    for i in range(1, xpts.shape[1]):
        d = xpts[:, keep[-1]] - xpts[:, i]
        if float(d @ d) != 0.0:
            keep.append(i)
    xnod = xpts[:, keep]
    if float((xnod[:, -1] - xpts[:, 0]) @ (xnod[:, -1] - xpts[:, 0])) != 0.0:
        xnod = np.concatenate([xnod, xpts[:, :1]], axis=1)
    if xnod.shape[1] < 2:
        z = np.zeros_like(R)
        return z, z, z

    dxnod = xnod[:, 1:] - xnod[:, :-1]
    lsqnod = np.sum(dxnod * dxnod, axis=0)
    eps_sq = max(np.finfo(float).eps * float(np.min(lsqnod[lsqnod > 0.0])), np.finfo(float).tiny)

    cos1 = np.broadcast_to(cosuv[None, :], (ntheta, nv)).reshape(-1)
    sin1 = np.broadcast_to(sinuv[None, :], (ntheta, nv)).reshape(-1)
    rp = R.reshape(-1)
    xobs = np.stack([rp * cos1, rp * sin1, Z.reshape(-1)], axis=1)

    capRv = xobs[:, None, :] - xnod.T[None, :, :]
    capR = np.sqrt(np.maximum(eps_sq, np.sum(capRv * capRv, axis=2)))
    R1p2 = capR[:, :-1] + capR[:, 1:]
    denom = np.maximum(R1p2 * R1p2 - lsqnod[None, :], eps_sq)
    Rfactor = 2.0 * R1p2 / (capR[:, :-1] * capR[:, 1:] * denom)
    crossv = np.cross(dxnod.T[None, :, :], capRv[:, :-1, :])
    bxyz = (current * 1.0e-7) * np.sum(crossv * Rfactor[:, :, None], axis=1)

    br = cos1 * bxyz[:, 0] + sin1 * bxyz[:, 1]
    bp = -sin1 * bxyz[:, 0] + cos1 * bxyz[:, 1]
    return br.reshape((ntheta, nv)), bp.reshape((ntheta, nv)), bxyz[:, 2].reshape((ntheta, nv))


# ---------------------------------------------------------------------------
# External-field projection (bextern.f)
# ---------------------------------------------------------------------------


def external_field_channels(
    *,
    boundary: VacuumBoundary,
    br: np.ndarray,
    bp: np.ndarray,
    bz: np.ndarray,
    basis: VacuumBasis,
    signgs: int,
) -> dict[str, np.ndarray]:
    """``bextern.f``: covariant components, normal source, and metric.

    Returns ``bexu/bexv`` (covariant, geometric-phi convention), ``bexni``
    (the weighted normal source ``-B.n * wint * (2*pi)^2``), and the
    physical surface metric ``guu/guv/gvv``.
    """
    R = np.asarray(boundary.R, dtype=float)
    Ru = np.asarray(boundary.Ru, dtype=float)
    Zu = np.asarray(boundary.Zu, dtype=float)
    Rv = np.asarray(boundary.Rv, dtype=float)
    Zv = np.asarray(boundary.Zv, dtype=float)
    sgn = float(int(signgs))
    snr = sgn * R * Zu
    snv = sgn * (Ru * Zv - Rv * Zu)
    snz = -sgn * R * Ru
    bexu = Ru * br + Zu * bz
    bexv = Rv * br + Zv * bz + R * bp
    bexn = -(br * snr + bp * snv + bz * snz)
    wint2 = np.asarray(basis.wint, dtype=float).reshape(R.shape)
    bexni = bexn * wint2 * ((2.0 * np.pi) ** 2)
    return {
        "bexu": bexu,
        "bexv": bexv,
        "bexn": bexn,
        "bexni": bexni,
        "guu": Ru * Ru + Zu * Zu,
        "guv": Ru * Rv + Zu * Zv,
        "gvv": R * R + Rv * Rv + Zv * Zv,
    }


# ---------------------------------------------------------------------------
# Per-iteration plasma scalars (bcovar.f tails consumed by vacuum)
# ---------------------------------------------------------------------------


@jax.jit
def _vacuum_scalars(state: SpectralState, rt: SolverRuntime):
    """``(ctor, rbtor, axis_r, axis_z, bsq_edge_extrap, pres_edge)``.

    - ``ctor = signgs*2*pi*(1.5*buco(ns) - 0.5*buco(ns-1))`` with
      ``buco = <B_u>`` (``bcovar.f``/``calc_fbal``);
    - ``rbtor = 1.5*bvco(ns) - 0.5*bvco(ns-1)``;
    - ``axis_r/axis_z``: ``r1/z1(js=1, theta=0, :)`` — the ``raxis_nestor``
      arrays of ``funct3d.f``;
    - ``bsq_edge_extrap = 1.5*bsq(ns) - 0.5*bsq(ns-1)`` on the angular grid
      (``bsqsav(:,3)`` for the DEL-BSQ diagnostic).
    """
    setup = rt.setup
    s = setup.s_full
    _, geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    wint = jnp.asarray(rt.weights)  # (ntheta_eff,), zeta-constant wint
    buco = jnp.sum(fields.bsubu * wint[None, :, None], axis=(1, 2))
    bvco = jnp.sum(fields.bsubv * wint[None, :, None], axis=(1, 2))
    sgn = jnp.asarray(float(setup.signgs))
    ctor = sgn * (2.0 * jnp.pi) * (1.5 * buco[-1] - 0.5 * buco[-2])
    rbtor = 1.5 * bvco[-1] - 0.5 * bvco[-2]
    axis_r = geometry.R_even[0, 0, :]
    axis_z = geometry.Z_even[0, 0, :]
    bsq = fields.total_pressure
    bsq_edge_extrap = 1.5 * bsq[-1] - 0.5 * bsq[-2]
    return ctor, rbtor, axis_r, axis_z, bsq_edge_extrap, fields.pressure[-1]


@jax.jit
def _iter_lane(carry, rt: SolverRuntime):
    """One jitted eqsolve iteration (shared traced body; per-``rt`` lane)."""
    return _make_body(rt)(carry)


# ---------------------------------------------------------------------------
# Free-boundary driver state
# ---------------------------------------------------------------------------


@dataclass
class FreeBoundaryState:
    """Host cadence state + NESTOR cache (``funct3d.f`` module variables)."""

    ivac: int = -1
    nvacskip: int = 1
    nvskip0: int = 1
    turned_on: bool = False
    banner_pending: bool = False
    delbsq: float = 1.0
    bsqvac: np.ndarray | None = None
    # NESTOR cache (amatsav / bvecsav of scalpot.f):
    mode_matrix: Any = None
    bvec_nonsing: Any = None
    potvac: np.ndarray | None = None
    ctor: float = 0.0
    rbtor: float = 0.0
    vacuum_calls: int = 0
    full_updates: int = 0


def _resolve_mgrid(inp: VmecInput, mgrid_path: str | Path | None) -> Path:
    p = Path(str(mgrid_path if mgrid_path is not None else inp.mgrid_file)).expanduser()
    return p


# ---------------------------------------------------------------------------
# Fused on-device vacuum update (JAX ports of the host-assembly functions)
# ---------------------------------------------------------------------------
#
# The NumPy functions above (``_edge_fourier``, ``boundary_from_coefficients``,
# ``axis_current_field``, ``external_field_channels``) remain the parity-proven
# reference used by the operator tests.  The ``*_jax`` mirrors below are their
# pure-``jnp`` equivalents so the whole per-iteration vacuum update can run as
# ONE jitted program (``_make_fused_vacuum``) with no NumPy<->JAX round-trips
# (``tests/test_freeboundary.py::test_fused_vacuum_matches_reference``
# A/B-locks them to the NumPy path at machine precision).


def _edge_fourier_jax(state: SpectralState, rt: SolverRuntime):
    """On-device edge wout-convention coefficients (JAX ``_edge_fourier``)."""
    from .residuals import m1_constrained_to_physical
    from .transforms import physical_to_internal_scale

    setup = rt.setup
    R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos,
        modes=rt.modes, lthreed=setup.lthreed, lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    scale = jnp.asarray(1.0 / physical_to_internal_scale(rt.modes, rt.trig))
    rmnc = R_cos[-1] * scale
    zmns = Z_sin[-1] * scale
    if setup.lasym:
        return rmnc, zmns, R_sin[-1] * scale, Z_cos[-1] * scale
    return rmnc, zmns, None, None


def _boundary_from_coefficients_jax(rmnc, zmns, rmns, zmnc, *, modes: ModeTable,
                                    basis: VacuumBasis) -> VacuumBoundary:
    """On-device boundary synthesis (JAX ``boundary_from_coefficients``)."""
    xm = jnp.asarray(np.asarray(modes.m, dtype=float))
    xn = jnp.asarray(np.asarray(modes.n, dtype=float) * float(basis.nfp))
    th = jnp.asarray(np.asarray(basis.theta, dtype=float))[:, None]
    ze = jnp.asarray(np.asarray(basis.zeta, dtype=float))[:, None] * float(basis.onp)
    arg = th * xm[None, :] - ze * xn[None, :]
    cosmn = jnp.cos(arg)
    sinmn = jnp.sin(arg)
    rc = jnp.asarray(rmnc)
    zs = jnp.asarray(zmns)
    R = cosmn @ rc
    Z = sinmn @ zs
    Ru = -(sinmn * xm[None, :]) @ rc
    Rv = (sinmn * xn[None, :]) @ rc
    Zu = (cosmn * xm[None, :]) @ zs
    Zv = -(cosmn * xn[None, :]) @ zs
    ruu = -(cosmn * (xm * xm)[None, :]) @ rc
    ruv = (cosmn * (xm * xn)[None, :]) @ rc
    rvv = -(cosmn * (xn * xn)[None, :]) @ rc
    zuu = -(sinmn * (xm * xm)[None, :]) @ zs
    zuv = (sinmn * (xm * xn)[None, :]) @ zs
    zvv = -(sinmn * (xn * xn)[None, :]) @ zs
    if rmns is not None and zmnc is not None:
        rs = jnp.asarray(rmns)
        zc = jnp.asarray(zmnc)
        R = R + sinmn @ rs
        Z = Z + cosmn @ zc
        Ru = Ru + (cosmn * xm[None, :]) @ rs
        Rv = Rv - (cosmn * xn[None, :]) @ rs
        Zu = Zu - (sinmn * xm[None, :]) @ zc
        Zv = Zv + (sinmn * xn[None, :]) @ zc
        ruu = ruu - (sinmn * (xm * xm)[None, :]) @ rs
        ruv = ruv + (sinmn * (xm * xn)[None, :]) @ rs
        rvv = rvv - (sinmn * (xn * xn)[None, :]) @ rs
        zuu = zuu - (cosmn * (xm * xm)[None, :]) @ zc
        zuv = zuv + (cosmn * (xm * xn)[None, :]) @ zc
        zvv = zvv - (cosmn * (xn * xn)[None, :]) @ zc
    shape = (int(basis.ntheta3), int(basis.nzeta))
    return VacuumBoundary(
        R=R.reshape(shape), Z=Z.reshape(shape),
        Ru=Ru.reshape(shape), Zu=Zu.reshape(shape),
        Rv=Rv.reshape(shape), Zv=Zv.reshape(shape),
        ruu=ruu.reshape(shape), ruv=ruv.reshape(shape), rvv=rvv.reshape(shape),
        zuu=zuu.reshape(shape), zuv=zuv.reshape(shape), zvv=zvv.reshape(shape),
    )


def _axis_current_tables(basis: VacuumBasis) -> dict[str, Any]:
    """Static filament tables for :func:`_axis_current_field_jax` (``tolicu.f``).

    For a non-degenerate axis every replicated node is distinct, so the
    ``tolicu.f`` ``bsc_construct`` keep-filtering is a no-op and the closed-loop
    node count is static (``nvper*nzeta + 1``) — verified at build time in
    :func:`_make_fused_vacuum`.  Returns the geometry-independent period/segment
    trig tables (device arrays) shared across every iteration.
    """
    nv = int(basis.nzeta)
    nfp = max(1, int(basis.nfp))
    nvper = 64 if nv == 1 else nfp
    alvp = (2.0 * np.pi / float(max(1, nv))) / float(nfp)
    cosuv = np.cos(alvp * np.arange(nv, dtype=float))
    sinuv = np.sin(alvp * np.arange(nv, dtype=float))
    alp_per = 2.0 * np.pi / float(nvper)
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))
    return {
        "nv": nv, "nvper": nvper,
        "cosuv": jnp.asarray(cosuv), "sinuv": jnp.asarray(sinuv),
        "cosper": jnp.asarray(cosper), "sinper": jnp.asarray(sinper),
    }


def _axis_current_field_jax(R, Z, axis_r, axis_z, current, tables: dict[str, Any]):
    """On-device axis-filament Biot-Savart (JAX ``tolicu.f`` + ``belicu.f``).

    Static-topology port of :func:`axis_current_field`: the ``nvper*nzeta``
    replicated nodes are all distinct (non-degenerate axis) and the loop is
    closed by appending the first node; ``eps_sq`` regularizes the LIBSTELL
    ``bsc_b`` segment kernel exactly as the NumPy reference.  ``current`` is the
    filament current ``ctor / mu0``.
    """
    nv = int(tables["nv"])
    nvper = int(tables["nvper"])
    cosuv = tables["cosuv"]
    sinuv = tables["sinuv"]
    cosper = tables["cosper"]
    sinper = tables["sinper"]
    ar = jnp.reshape(axis_r, (-1,))
    az = jnp.reshape(axis_z, (-1,))
    x0 = ar * cosuv
    y0 = ar * sinuv
    xper = cosper[:, None] * x0[None, :] - sinper[:, None] * y0[None, :]
    yper = sinper[:, None] * x0[None, :] + cosper[:, None] * y0[None, :]
    zper = jnp.broadcast_to(az[None, :], (nvper, nv))
    xpts = jnp.stack([xper.reshape(-1), yper.reshape(-1), zper.reshape(-1)], axis=0)
    xnod = jnp.concatenate([xpts, xpts[:, :1]], axis=1)  # bsc: close the loop
    dxnod = xnod[:, 1:] - xnod[:, :-1]
    lsqnod = jnp.sum(dxnod * dxnod, axis=0)
    eps = float(np.finfo(float).eps)
    tiny = float(np.finfo(float).tiny)
    eps_sq = jnp.maximum(eps * jnp.min(jnp.where(lsqnod > 0.0, lsqnod, jnp.inf)), tiny)
    ntheta = int(R.shape[0])
    cos1 = jnp.broadcast_to(cosuv[None, :], (ntheta, nv)).reshape(-1)
    sin1 = jnp.broadcast_to(sinuv[None, :], (ntheta, nv)).reshape(-1)
    rp = jnp.reshape(R, (-1,))
    xobs = jnp.stack([rp * cos1, rp * sin1, jnp.reshape(Z, (-1,))], axis=1)
    capRv = xobs[:, None, :] - xnod.T[None, :, :]
    capR = jnp.sqrt(jnp.maximum(eps_sq, jnp.sum(capRv * capRv, axis=2)))
    R1p2 = capR[:, :-1] + capR[:, 1:]
    denom = jnp.maximum(R1p2 * R1p2 - lsqnod[None, :], eps_sq)
    Rfactor = 2.0 * R1p2 / (capR[:, :-1] * capR[:, 1:] * denom)
    crossv = jnp.cross(dxnod.T[None, :, :], capRv[:, :-1, :])
    bxyz = (current * 1.0e-7) * jnp.sum(crossv * Rfactor[:, :, None], axis=1)
    br = cos1 * bxyz[:, 0] + sin1 * bxyz[:, 1]
    bp = -sin1 * bxyz[:, 0] + cos1 * bxyz[:, 1]
    return (br.reshape((ntheta, nv)), bp.reshape((ntheta, nv)),
            bxyz[:, 2].reshape((ntheta, nv)))


def _external_field_channels_jax(boundary: VacuumBoundary, br, bp, bz, *,
                                 basis: VacuumBasis, signgs: int):
    """On-device ``bextern.f`` channels (JAX ``external_field_channels``)."""
    R = boundary.R
    Ru = boundary.Ru
    Zu = boundary.Zu
    Rv = boundary.Rv
    Zv = boundary.Zv
    sgn = float(int(signgs))
    snr = sgn * R * Zu
    snv = sgn * (Ru * Zv - Rv * Zu)
    snz = -sgn * R * Ru
    bexu = Ru * br + Zu * bz
    bexv = Rv * br + Zv * bz + R * bp
    bexn = -(br * snr + bp * snv + bz * snz)
    wint2 = jnp.asarray(np.asarray(basis.wint, dtype=float).reshape((
        int(basis.ntheta3), int(basis.nzeta))))
    bexni = bexn * wint2 * ((2.0 * np.pi) ** 2)
    return {
        "bexu": bexu, "bexv": bexv, "bexn": bexn, "bexni": bexni,
        "guu": Ru * Ru + Zu * Zu,
        "guv": Ru * Rv + Zu * Zv,
        "gvv": R * R + Rv * Rv + Zv * Zv,
    }


@dataclass(frozen=True, eq=False)
class FusedVacuum:
    """Jitted whole-pipeline NESTOR update closures (host-side full/skip choice).

    ``full(state, rt, field)`` and ``skip(state, rt, field, bvec_nonsing,
    mode_matrix)`` each run the ENTIRE per-iteration vacuum update on-device —
    plasma scalars, boundary synthesis, mgrid + axis-current external field,
    NESTOR solve, surface field and the DEL-BSQ / banner reductions — returning
    a dict of device arrays.  Composing them into one jitted program removes the
    ~27 NumPy<->JAX host round-trips the step-by-step host driver incurred.
    """

    full: Any
    skip: Any


def _make_fused_vacuum(basis: VacuumBasis, *, modes: ModeTable, signgs: int,
                       solver_vac, axis_r0, axis_z0) -> FusedVacuum:
    """Build the jitted full/skip whole-pipeline vacuum updates for one basis."""
    axis_tb = _axis_current_tables(basis)
    _assert_static_filament_topology(basis, axis_r0, axis_z0)
    shape = (int(basis.ntheta3), int(basis.nzeta))
    phi_geom = jnp.asarray(
        (np.asarray(basis.zeta, dtype=float) * float(basis.onp)).reshape(shape)
    )
    wint2 = jnp.asarray(np.asarray(basis.wint, dtype=float).reshape(shape))
    two_pi = 2.0 * float(np.pi)
    sgn = float(int(signgs))

    def _pipeline(field, boundary, ctor, axis_r, axis_z):
        br_c, bp_c, bz_c = field.b_cyl(boundary.R, phi_geom, boundary.Z)
        br_a, bp_a, bz_a = _axis_current_field_jax(
            boundary.R, boundary.Z, axis_r, axis_z, ctor / MU0, axis_tb
        )
        ext = _external_field_channels_jax(
            boundary, br_c + br_a, bp_c + bp_a, bz_c + bz_a,
            basis=basis, signgs=signgs,
        )
        return ext

    def _diagnostics(bsqvac, bsubu_s, bsubv_s, bsq3, pres_ns, rt):
        gcon_edge = bsqvac + pres_ns * jnp.asarray(rt.presf_ns_scale)
        delbsq_num = jnp.sum(jnp.abs(gcon_edge - bsq3) * wint2)
        delbsq_den = jnp.sum(bsq3 * wint2)
        bsubuvac = jnp.sum(bsubu_s * wint2) * sgn * two_pi
        bsubvvac = jnp.sum(bsubv_s * wint2)
        return delbsq_num, delbsq_den, bsubuvac, bsubvvac

    def _full(state: SpectralState, rt: SolverRuntime, field: MgridField):
        ctor, rbtor, axis_r, axis_z, bsq3, pres_ns = _vacuum_scalars(state, rt)
        rmnc, zmns, rmns, zmnc = _edge_fourier_jax(state, rt)
        boundary = _boundary_from_coefficients_jax(
            rmnc, zmns, rmns, zmnc, modes=modes, basis=basis
        )
        ext = _pipeline(field, boundary, ctor, axis_r, axis_z)
        potvac, mode_matrix, bvec_nonsing, _rhs, _gsrc, _grp = solver_vac.full(
            boundary, ext["bexni"]
        )
        bsqvac, bsubu_s, bsubv_s, _bu, _bv = vacuum_channels(
            basis=basis, potvac=potvac, bexu=ext["bexu"], bexv=ext["bexv"],
            guu=ext["guu"], guv=ext["guv"], gvv=ext["gvv"],
        )
        delbsq_num, delbsq_den, bsubuvac, bsubvvac = _diagnostics(
            bsqvac, bsubu_s, bsubv_s, bsq3, pres_ns, rt
        )
        return {
            "bsqvac": bsqvac, "ctor": ctor, "rbtor": rbtor, "potvac": potvac,
            "mode_matrix": mode_matrix, "bvec_nonsing": bvec_nonsing,
            "delbsq_num": delbsq_num, "delbsq_den": delbsq_den,
            "bsubuvac": bsubuvac, "bsubvvac": bsubvvac,
        }

    def _skip(state: SpectralState, rt: SolverRuntime, field: MgridField,
              bvec_nonsing, mode_matrix):
        ctor, rbtor, axis_r, axis_z, bsq3, pres_ns = _vacuum_scalars(state, rt)
        rmnc, zmns, rmns, zmnc = _edge_fourier_jax(state, rt)
        boundary = _boundary_from_coefficients_jax(
            rmnc, zmns, rmns, zmnc, modes=modes, basis=basis
        )
        ext = _pipeline(field, boundary, ctor, axis_r, axis_z)
        potvac, _rhs = solver_vac.skip(
            boundary, ext["bexni"], bvec_nonsing, mode_matrix
        )
        bsqvac, bsubu_s, bsubv_s, _bu, _bv = vacuum_channels(
            basis=basis, potvac=potvac, bexu=ext["bexu"], bexv=ext["bexv"],
            guu=ext["guu"], guv=ext["guv"], gvv=ext["gvv"],
        )
        delbsq_num, delbsq_den, bsubuvac, bsubvvac = _diagnostics(
            bsqvac, bsubu_s, bsubv_s, bsq3, pres_ns, rt
        )
        return {
            "bsqvac": bsqvac, "ctor": ctor, "rbtor": rbtor, "potvac": potvac,
            "delbsq_num": delbsq_num, "delbsq_den": delbsq_den,
            "bsubuvac": bsubuvac, "bsubvvac": bsubvvac,
        }

    return FusedVacuum(full=jax.jit(_full), skip=jax.jit(_skip))


def _assert_static_filament_topology(basis: VacuumBasis, axis_r0, axis_z0) -> None:
    """Guard the static-topology assumption of :func:`_axis_current_field_jax`.

    Replays ``tolicu.f``'s ``bsc_construct`` keep-filtering on the initial axis;
    the fused path assumes every replicated node is distinct and the loop closes
    (true for any non-degenerate axis on ``nzeta > 1``).  Raises if a deck ever
    violates it, rather than silently diverging from the NumPy reference.
    """
    ar = np.asarray(axis_r0, dtype=float).reshape(-1)
    az = np.asarray(axis_z0, dtype=float).reshape(-1)
    nv = int(basis.nzeta)
    nfp = max(1, int(basis.nfp))
    nvper = 64 if nv == 1 else nfp
    alvp = (2.0 * np.pi / float(max(1, nv))) / float(nfp)
    cosuv = np.cos(alvp * np.arange(nv, dtype=float))
    sinuv = np.sin(alvp * np.arange(nv, dtype=float))
    alp_per = 2.0 * np.pi / float(nvper)
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))
    x0 = ar * cosuv
    y0 = ar * sinuv
    xpts = np.zeros((3, nvper * nv), dtype=float)
    for kper in range(nvper):
        sl = slice(kper * nv, (kper + 1) * nv)
        xpts[0, sl] = cosper[kper] * x0 - sinper[kper] * y0
        xpts[1, sl] = sinper[kper] * x0 + cosper[kper] * y0
        xpts[2, sl] = az
    keep = [0]
    for i in range(1, xpts.shape[1]):
        d = xpts[:, keep[-1]] - xpts[:, i]
        if float(d @ d) != 0.0:
            keep.append(i)
    closed = float((xpts[:, keep[-1]] - xpts[:, 0]) @ (xpts[:, keep[-1]] - xpts[:, 0])) != 0.0
    if len(keep) != xpts.shape[1] or not closed:
        raise NotImplementedError(
            "fused axis-current filament requires a non-degenerate axis "
            f"(kept {len(keep)}/{xpts.shape[1]} nodes, closed={closed}); "
            "this deck needs the NumPy axis_current_field path"
        )


#: Structural executable reuse for the fused vacuum program, mirroring
#: ``solver._static_tables``: repeated free-boundary solves at one resolution
#: (the warm benchmark's second solve, hot restarts, optimization iterates)
#: reuse ONE compiled NESTOR fused program instead of recompiling the
#: greenf/analyt/solve kernels (~5 s on CPU) every solve.  Keyed on the
#: hashable ``(resolution, signgs, mf, nf)``; the boundary/profile/mgrid values
#: enter the jitted program as traced arguments, so one executable serves every
#: solve at a given resolution.  The third element is the jitted steady-state
#: free-boundary loop (:func:`_make_vacuum_lane`) built over the same fused
#: vacuum closures.
_VACUUM_EXECUTABLE_CACHE: dict[tuple[Any, int, int, int], tuple[VacuumBasis, FusedVacuum, Any]] = {}


def _vacuum_executables(resolution, *, mf: int, nf: int, signgs: int, wint,
                        modes: ModeTable, axis_r0, axis_z0) -> tuple[VacuumBasis, FusedVacuum, Any]:
    """Return the cached ``(basis, fused vacuum, steady lane)`` for one resolution/signgs.

    ``wint``/``modes``/``axis_r0``/``axis_z0`` are resolution-determined build
    inputs (not part of the key); they are consumed only on a cache miss.
    """
    key = (resolution, int(signgs), int(mf), int(nf))
    cached = _VACUUM_EXECUTABLE_CACHE.get(key)
    if cached is not None:
        return cached
    basis = vacuum_basis(
        mf=int(mf), nf=int(nf), ntheta3=int(resolution.ntheta3),
        nzeta=int(resolution.nzeta), nfp=int(resolution.nfp),
        lasym=bool(resolution.lasym), wint=wint,
    )
    solver_vac = make_vacuum_solver(basis, signgs=int(signgs))
    fused = _make_fused_vacuum(
        basis, modes=modes, signgs=int(signgs), solver_vac=solver_vac,
        axis_r0=axis_r0, axis_z0=axis_z0,
    )
    lane = _make_vacuum_lane(fused)
    _VACUUM_EXECUTABLE_CACHE[key] = (basis, fused, lane)
    return basis, fused, lane


def _vacuum_step(
    *,
    carry,
    rt: SolverRuntime,
    fb: FreeBoundaryState,
    basis: VacuumBasis,
    fused_vac: FusedVacuum,
    field: MgridField,
    ivacskip: int,
    emit,
    verbose: bool,
) -> Array:
    """One NESTOR update (``vacuum.f``): returns ``bsqvac`` on the grid (device).

    The whole update — plasma scalars, boundary synthesis, mgrid + axis-current
    external field, NESTOR solve, surface field and DEL-BSQ reduction — runs as
    ONE jitted program (:class:`FusedVacuum`).  Only a few diagnostic scalars are
    pulled to the host (screen line + turn-on banner); ``bsqvac`` and the cached
    ``amatsav``/``bvecsav`` matrices stay on-device across iterations.
    """
    if int(ivacskip) == 0 or fb.mode_matrix is None:
        out = fused_vac.full(carry.state, rt, field)
        fb.mode_matrix = out["mode_matrix"]
        fb.bvec_nonsing = out["bvec_nonsing"]
        fb.full_updates += 1
    else:
        out = fused_vac.skip(carry.state, rt, field, fb.bvec_nonsing, fb.mode_matrix)
    bsqvac = out["bsqvac"]
    fb.potvac = out["potvac"]
    fb.ctor = float(out["ctor"])
    fb.rbtor = float(out["rbtor"])
    fb.vacuum_calls += 1

    if fb.ivac == 0:
        # vacuum.f first-call block: promote ivac and print grid/current info.
        fb.ivac = 1
        if verbose:
            emit(
                f"\n  In VACUUM, np = {basis.nfp:2d}  mf = {basis.mf:2d}  nf = {basis.nf:2d}"
                f" nu = {basis.nu_full:2d}  nv = {basis.nzeta:4d}\n"
            )
            fac = 1.0e-6 / MU0
            emit(
                f"  2*pi * a * -BPOL(vac) = {float(out['bsubuvac'])*fac:10.2E}"
                f" TOROIDAL CURRENT = {fb.ctor*fac:10.2E}\n"
                f"  R * BTOR(vac) = {float(out['bsubvvac']):10.2E}"
                f" R * BTOR(plasma) = {fb.rbtor:10.2E}\n"
            )

    # DEL-BSQ diagnostic (funct3d.f dbsq + printout.f delbsq).
    den = float(out["delbsq_den"])
    if den != 0.0:
        fb.delbsq = float(out["delbsq_num"]) / den
    fb.bsqvac = bsqvac
    return bsqvac


def _presf_ns_scale(inp: VmecInput, ns: int) -> float:
    """funct3d.f: ``presf_ns = pmass(1)/pmass(hs*(ns-1.5)) * pres(ns)``."""
    hs = 1.0 / float(ns - 1)
    sedge = hs * (float(ns) - 1.5)
    kwargs = dict(pres_scale=float(inp.pres_scale), bloat=float(inp.bloat),
                  spres_ped=1.0)
    p_edge = float(np.asarray(_profiles.pressure(
        inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, sedge, **kwargs)))
    if p_edge == 0.0:
        return 0.0
    p_one = float(np.asarray(_profiles.pressure(
        inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, 1.0, **kwargs)))
    return p_one / p_edge


# ---------------------------------------------------------------------------
# Batched iteration lanes (plan item F.2: NESTOR loop batching)
# ---------------------------------------------------------------------------
#
# The host driver used to dispatch ONE jitted iteration per Python pass, with
# several device->host syncs (`bool(carry.done)`, `int(carry.iteration)`,
# `float(carry.fsqr)`, ...) and per-pass `replace(rt, rcon0=0.9*rcon0, ...)`
# runtime rebuilds.  The two lanes below move that scaffolding on-device
# without touching the per-iteration numerics (same ops, same order):
#
# - :func:`_preactivation_lane` runs every fixed-boundary iteration BEFORE
#   vacuum activation as one ``lax.while_loop`` (the free-boundary analogue of
#   ``solver._while_lane``), exiting exactly when the host driver would have
#   entered the funct3d.f IVAC0 block (``iter2 > 1`` and
#   ``fsqr + fsqz <= 1e-3``) so the turn-on pass still runs host-side.
# - :func:`_make_vacuum_lane` runs the whole POST-turn-on steady state as one
#   ``lax.while_loop`` whose body replays the host pass verbatim in traced
#   form: ivac increment, 0.9 rcon0/zcon0 damping, the ivacskip/nvacskip
#   cadence, the full/skip fused NESTOR update (``lax.cond``), the
#   ``bsqvac_edge`` runtime update, then the shared ``_make_body`` iteration.
#   ``delbsq_traj`` records the per-iteration DEL-BSQ diagnostic so verbose
#   screen lines print the same value the per-pass driver printed.
#
# Only the single turn-on pass (first vacuum call, ``In VACUUM`` block, soft
# restart, ``VACUUM PRESSURE TURNED ON`` banner) stays host-stepped: it
# mutates the carry/runtime once, between the two lanes.


@jax.jit
def _preactivation_lane(carry, rt: SolverRuntime):
    """Fixed-boundary iterations up to vacuum activation, as one jitted loop.

    Iterates the shared traced body while the run is live and the funct3d.f
    activation condition (``iter2 > 1 and fsqr + fsqz <= 1e-3``) has not yet
    fired; the returned carry is exactly the carry the per-pass host driver
    held when it first entered the IVAC0 block (or finished the run).
    """
    body = _make_body(rt)

    def cond(c):
        activate = (c.iteration > 1) & (c.fsqr + c.fsqz <= ACTIVATION_FSQ)
        return jnp.logical_not(c.done | activate)

    return lax.while_loop(cond, body, carry)


@dataclass(frozen=True)
class _VacuumLoopCarry:
    """Traced carry of the steady-state free-boundary loop.

    Extends the solver ``_LoopCarry`` with the funct3d.f module state the
    host driver used to keep in :class:`FreeBoundaryState` / rebuilt
    runtimes: the damped ``rcon0/zcon0`` baselines, the NESTOR cache
    (``amatsav``/``bvecsav``), the vacuum cadence counters and the DEL-BSQ
    diagnostic (plus its per-iteration history for verbose printing).
    """

    carry: Any                  # solver._LoopCarry
    rcon0: Array; zcon0: Array
    bsqvac: Array               # NESTOR 0.5*|B|^2 on the boundary grid
    mode_matrix: Array          # amatsav (scalpot.f)
    bvec_nonsing: Array         # bvecsav (scalpot.f)
    potvac: Array
    ivac: Array; nvacskip: Array; nvskip0: Array
    delbsq: Array; delbsq_traj: Array
    ctor: Array; rbtor: Array
    vacuum_calls: Array; full_updates: Array


_register(_VacuumLoopCarry)


def _make_vacuum_lane(fused: FusedVacuum):
    """Build the jitted steady-state (post-turn-on) free-boundary loop.

    Closed over the per-basis :class:`FusedVacuum` (cached alongside it in
    ``_VACUUM_EXECUTABLE_CACHE``); ``rt``/``field``/carry enter as traced
    arguments, so one executable serves every solve at a given resolution.
    """

    def _pass(vc: _VacuumLoopCarry, rt: SolverRuntime, field) -> _VacuumLoopCarry:
        c = vc.carry
        it = c.iteration
        fsq_rz = c.fsqr + c.fsqz

        # -- funct3d.f IVAC0 block (traced; same order as the host pass) ----
        ivac = vc.ivac + ((it > 1) & (fsq_rz <= ACTIVATION_FSQ)).astype(vc.ivac.dtype)
        rcon0 = 0.9 * vc.rcon0
        zcon0 = 0.9 * vc.zcon0
        one = jnp.ones((), dtype=vc.nvacskip.dtype)
        ivacskip = jnp.where(
            ivac <= 2, jnp.zeros_like(vc.nvacskip),
            (it - c.iter1).astype(vc.nvacskip.dtype) % jnp.maximum(one, vc.nvacskip),
        )
        full = ivacskip == 0
        # int() truncation toward zero == astype for the positive operand.
        nvacskip = jnp.where(
            full,
            jnp.maximum(vc.nvskip0, (1.0 / jnp.maximum(1.0e-1, 1.0e11 * fsq_rz))
                        .astype(vc.nvacskip.dtype)),
            vc.nvacskip,
        )

        # -- vacuum.f update: full rebuild vs cached-matrix skip ------------
        rt_vac = replace(rt, rcon0=rcon0, zcon0=zcon0, bsqvac_edge=vc.bsqvac)

        def _full(_):
            out = fused.full(c.state, rt_vac, field)
            return (out["bsqvac"], out["ctor"], out["rbtor"], out["potvac"],
                    out["delbsq_num"], out["delbsq_den"],
                    out["mode_matrix"], out["bvec_nonsing"])

        def _skip(_):
            out = fused.skip(c.state, rt_vac, field, vc.bvec_nonsing,
                             vc.mode_matrix)
            return (out["bsqvac"], out["ctor"], out["rbtor"], out["potvac"],
                    out["delbsq_num"], out["delbsq_den"],
                    vc.mode_matrix, vc.bvec_nonsing)

        (bsqvac, ctor, rbtor, potvac, num, den, mode_matrix,
         bvec_nonsing) = lax.cond(full, _full, _skip, None)

        delbsq = jnp.where(den != 0.0, num / den, vc.delbsq)
        idx = jnp.clip(it - 1, 0, rt.max_iterations - 1)
        delbsq_traj = lax.dynamic_update_slice_in_dim(
            vc.delbsq_traj, delbsq[None], idx, axis=0)

        # -- one eqsolve iteration with the refreshed edge field ------------
        new_carry = _make_body(replace(rt_vac, bsqvac_edge=bsqvac))(c)

        return _VacuumLoopCarry(
            carry=new_carry, rcon0=rcon0, zcon0=zcon0, bsqvac=bsqvac,
            mode_matrix=mode_matrix, bvec_nonsing=bvec_nonsing, potvac=potvac,
            ivac=ivac, nvacskip=nvacskip, nvskip0=vc.nvskip0,
            delbsq=delbsq, delbsq_traj=delbsq_traj, ctor=ctor, rbtor=rbtor,
            vacuum_calls=vc.vacuum_calls + 1,
            full_updates=vc.full_updates + full.astype(vc.full_updates.dtype),
        )

    def _lane(vc: _VacuumLoopCarry, rt: SolverRuntime, field) -> _VacuumLoopCarry:
        return lax.while_loop(
            lambda v: jnp.logical_not(v.carry.done),
            lambda v: _pass(v, rt, field), vc)

    return jax.jit(_lane)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def solve_free_boundary(
    inp: VmecInput,
    *,
    mgrid_path: str | Path | None = None,
    external_field: MgridField | None = None,
    resolution=None,
    ftol: float | None = None,
    max_iterations: int | None = None,
    verbose: bool = False,
    emit=print,
    error_on_no_convergence: bool = True,
) -> SolveResult:
    """Single-grid free-boundary solve (``eqsolve.f`` + ``funct3d.f`` IVAC0).

    ``external_field`` overrides the mgrid file (any
    :class:`~vmec_jax.core.mgrid.MgridField`-compatible object with a
    ``b_cyl(r, phi, z)`` method — e.g. a direct-coil Biot-Savart field).
    Raises :class:`~vmec_jax.core.errors.MgridNotFoundError` when the deck's
    mgrid file is missing and no field is supplied (callers such as the CLI
    implement VMEC2000's warn-and-fall-back-to-fixed-boundary policy).

    ``error_on_no_convergence=False`` returns the final state instead of
    raising when NITER is exhausted (useful against unconverged goldens).
    """
    if not bool(inp.lfreeb):
        raise ValueError("solve_free_boundary requires an LFREEB=T input")
    if external_field is None:
        path = _resolve_mgrid(inp, mgrid_path)
        data_extcur = np.atleast_1d(np.asarray(inp.extcur if inp.extcur is not None else [], dtype=float))
        from .mgrid import read_mgrid

        data = read_mgrid(path)  # raises MgridNotFoundError when missing
        extcur = np.zeros((data.nextcur,), dtype=float)
        n_copy = min(data_extcur.size, data.nextcur)
        extcur[:n_copy] = data_extcur[:n_copy]
        if str(data.mgrid_mode).upper().startswith("R") or str(data.mgrid_mode).upper().startswith("N"):
            raw = np.asarray(data.raw_coil_cur, dtype=float)
            extcur = np.divide(extcur, raw, out=extcur, where=raw != 0.0)
        external_field = MgridField.from_mgrid_data(data, extcur=extcur)

    if resolution is None:
        resolution = resolution_from_input(inp)
    rt = prepare_runtime(inp, resolution, ftol=ftol, max_iterations=max_iterations)
    ns = int(resolution.ns)
    dtype = rt.setup.s_full.dtype

    _init_state = _initial_state(rt.setup)
    _axis_r0, _axis_z0 = _vacuum_scalars(_init_state, rt)[2:4]
    basis, fused_vac, vacuum_lane = _vacuum_executables(
        resolution, mf=int(inp.mpol) + 1, nf=int(inp.ntor),
        signgs=int(rt.setup.signgs), wint=np.asarray(rt.trig.wint, dtype=float),
        modes=rt.modes, axis_r0=_axis_r0, axis_z0=_axis_z0,
    )

    zeros_edge = jnp.zeros((basis.ntheta3, basis.nzeta), dtype=dtype)
    rt_fixed = replace(rt, lfreeb=False, bsqvac_edge=zeros_edge,
                       presf_ns_scale=jnp.asarray(0.0, dtype=dtype))
    rt_freeb = replace(
        rt, lfreeb=True, jmax=ns, bsqvac_edge=zeros_edge,
        presf_ns_scale=jnp.asarray(_presf_ns_scale(inp, ns), dtype=dtype),
    )

    fb = FreeBoundaryState(
        ivac=-1,
        nvacskip=max(1, int(inp.nvacskip)),
        nvskip0=max(1, int(inp.nvacskip)),
    )

    if verbose:
        emit(stage_banner(ns, resolution.mnmax, rt.ftol, rt.max_iterations), end="")
        emit(FORCE_ITERATIONS_BANNER, end="")
        emit(screen_header(lasym=resolution.lasym, lfreeb=True), end="")

    carry = _initial_carry(_init_state, rt_fixed, ijacob=0)
    printed: set[int] = set()
    #: per-iteration DEL-BSQ recorded by the batched steady-state lane; rows
    #: not covered (pre-activation, turn-on pass) fall back to ``fb.delbsq``
    #: exactly as the per-pass driver printed them.
    delbsq_rows: dict[int, float] = {}

    def _emit_due(final: bool) -> None:
        if not verbose:
            return
        upto = int(carry.iteration) if bool(carry.done) or final else int(carry.iteration) - 1
        trajectory = np.asarray(carry.trajectory[: max(upto, 0)])
        for it_p in range(1, upto + 1):
            due = (it_p == 1) or (it_p % rt.nstep == 0) or (final and it_p == upto)
            if not due or it_p in printed:
                continue
            row = trajectory[it_p - 1]
            if int(row[0]) != it_p:
                continue
            emit(screen_line(
                it_p, float(row[1]), float(row[2]), float(row[3]),
                float(row[7]), float(row[10]), float(row[9]),
                z_axis=float(row[8]) if resolution.lasym else None,
                del_bsq=delbsq_rows.get(it_p, float(fb.delbsq)),
            ), end="")
            printed.add(it_p)

    int_dtype = carry.iteration.dtype
    max_passes = rt.max_iterations + 400
    for _ in range(max_passes):
        if bool(carry.done):
            break
        if fb.ivac == -1:
            # F.2: every fixed-boundary iteration before vacuum activation
            # runs as ONE jitted while_loop; the lane exits precisely where
            # the per-pass driver would have entered the IVAC0 block below.
            carry = _preactivation_lane(carry, rt_fixed)
            _emit_due(final=False)
            if bool(carry.done):
                break
            # Activation is now due: fall through to the host-stepped
            # turn-on pass (first vacuum call, soft restart, banners).
        elif fb.turned_on and not fb.banner_pending:
            # F.2: the whole post-turn-on steady state runs as ONE jitted
            # while_loop (vacuum cadence + damping + iteration, traced).
            vc = _VacuumLoopCarry(
                carry=carry,
                rcon0=rt_freeb.rcon0, zcon0=rt_freeb.zcon0,
                bsqvac=rt_freeb.bsqvac_edge,
                mode_matrix=fb.mode_matrix, bvec_nonsing=fb.bvec_nonsing,
                potvac=fb.potvac,
                ivac=jnp.asarray(fb.ivac, dtype=int_dtype),
                nvacskip=jnp.asarray(fb.nvacskip, dtype=int_dtype),
                nvskip0=jnp.asarray(fb.nvskip0, dtype=int_dtype),
                delbsq=jnp.asarray(fb.delbsq, dtype=dtype),
                delbsq_traj=jnp.full((rt.max_iterations,), np.nan, dtype=dtype),
                ctor=jnp.asarray(fb.ctor, dtype=dtype),
                rbtor=jnp.asarray(fb.rbtor, dtype=dtype),
                vacuum_calls=jnp.asarray(fb.vacuum_calls, dtype=int_dtype),
                full_updates=jnp.asarray(fb.full_updates, dtype=int_dtype),
            )
            vc = vacuum_lane(vc, rt_freeb, external_field)
            carry = vc.carry
            rt_freeb = replace(rt_freeb, rcon0=vc.rcon0, zcon0=vc.zcon0,
                               bsqvac_edge=vc.bsqvac)
            fb.ivac = int(vc.ivac)
            fb.nvacskip = int(vc.nvacskip)
            fb.delbsq = float(vc.delbsq)
            fb.bsqvac = vc.bsqvac
            fb.mode_matrix = vc.mode_matrix
            fb.bvec_nonsing = vc.bvec_nonsing
            fb.potvac = vc.potvac
            fb.ctor = float(vc.ctor)
            fb.rbtor = float(vc.rbtor)
            fb.vacuum_calls = int(vc.vacuum_calls)
            fb.full_updates = int(vc.full_updates)
            if verbose:
                traj_db = np.asarray(vc.delbsq_traj)
                for i in np.flatnonzero(~np.isnan(traj_db)):
                    delbsq_rows[int(i) + 1] = float(traj_db[i])
            _emit_due(final=False)
            continue
        it = int(carry.iteration)
        iter1 = int(carry.iter1)
        fsq_rz = float(carry.fsqr) + float(carry.fsqz)

        # -- funct3d.f IVAC0 block (host) -----------------------------------
        if it > 1 and fsq_rz <= ACTIVATION_FSQ:
            fb.ivac += 1
        rt_use = rt_fixed
        if fb.ivac >= 0:
            # Damp the constraint baselines (funct3d: 0.9 per iteration).
            rt_fixed = replace(rt_fixed, rcon0=0.9 * rt_fixed.rcon0, zcon0=0.9 * rt_fixed.zcon0)
            rt_freeb = replace(rt_freeb, rcon0=0.9 * rt_freeb.rcon0, zcon0=0.9 * rt_freeb.zcon0)
            ivacskip = (it - iter1) % max(1, fb.nvacskip)
            if fb.ivac <= 2:
                ivacskip = 0
            if ivacskip == 0:
                fb.nvacskip = max(fb.nvskip0, int(1.0 / max(1.0e-1, 1.0e11 * fsq_rz)))
            bsqvac = _vacuum_step(
                carry=carry, rt=rt_freeb, fb=fb, basis=basis,
                fused_vac=fused_vac, field=external_field,
                ivacskip=ivacskip, emit=emit, verbose=verbose,
            )
            if fb.ivac >= 1 and not fb.turned_on:
                # funct3d.f soft start (restart_iter, irst = 2) applied on
                # the host: best state restored, velocity zeroed, delt*0.9,
                # iter1 = iter2, ijacob += 1.  Divergence from VMEC noted in
                # the module docstring (restart applied before this
                # iteration's force evaluation).
                fb.turned_on = True
                fb.banner_pending = True
                carry = replace(
                    carry,
                    state=carry.xstore,
                    xcdot=jax.tree.map(jnp.zeros_like, carry.xcdot),
                    time_step=carry.time_step * 0.9,
                    ijacob=carry.ijacob + 1,
                    iter1=carry.iteration,
                    # The preconditioner cache changes shape with jmax = ns;
                    # iter1 = iteration forces an immediate ns4 refresh, so
                    # the zeroed cache is never consumed.
                    cache=_zero_cache(rt_freeb),
                )
            if fb.ivac >= 1:
                rt_freeb = replace(rt_freeb, bsqvac_edge=jnp.asarray(bsqvac, dtype=dtype))
                rt_use = rt_freeb

        carry = _iter_lane(carry, rt_use)

        if fb.banner_pending:
            if verbose:
                emit(vacuum_banner(it), end="")
            fb.banner_pending = False
            fb.ivac = max(fb.ivac, 2)  # eqsolve.f: ivac = ivac + 1 after banner
        _emit_due(final=False)

    _emit_due(final=True)
    ier = int(carry.ier)
    if ier == MORE_ITER_FLAG and not error_on_no_convergence:
        result = _result_from_carry(carry, rt_freeb if fb.turned_on else rt_fixed)
        return replace(result, converged=False, ier_flag=MORE_ITER_FLAG)
    if ier == SUCCESSFUL_TERM_FLAG:
        return _result_from_carry(carry, rt_freeb if fb.turned_on else rt_fixed)
    return _finalize(carry, rt_freeb if fb.turned_on else rt_fixed)
