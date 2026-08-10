"""Free-boundary solve: NESTOR vacuum coupling around the core solver.

Implements the ``funct3d.f`` free-boundary block (VMEC2000) on top of the
fixed-boundary iteration of :mod:`vmex.core.solver`:

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
  the :class:`~vmex.core.solver.SolverRuntime` free-boundary seam
  (``lfreeb/bsqvac_edge/presf_ns_scale`` — see ``solver._evaluate``), the
  edge row is evolved (``jmax = ns``) and the ``rcon0/zcon0`` constraint
  baselines are damped by 0.9 per active iteration (``funct3d.f``).

The iteration itself runs the *same* traced body as the fixed-boundary
lanes.  Scheduling: the pre-activation fixed-boundary
iterations run as one jitted ``lax.while_loop`` (:func:`_preactivation_lane`)
and the whole post-turn-on steady state — vacuum cadence, constraint damping,
``bsqvac_edge`` refresh and the eqsolve iteration — as another
(:func:`_make_vacuum_lane`); only the single turn-on pass is host-stepped,
because it applies the one-time soft restart and prints the banners.  The
per-iteration numerics are identical to the per-pass host driver.

The one unusual turn-on pass preserves VMEC2000's ordering: its forces use
the geometry computed before the soft restart while the momentum update
evolves the restored best state.  Subsequent passes use the ordinary shared
iteration body.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import functools
import threading

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import linalg as jsp_linalg

from . import profiles as _profiles
from .device import AUTO, _placement_device, _put_numeric_leaves, device_context
from .errors import (
    AXIS_REGUESS_FLAG,
    JAC75_FLAG,
    MORE_ITER_FLAG,
    SUCCESSFUL_TERM_FLAG,
    VmecInputError,
)
from .fields import magnetic_fields, metric_elements
from .fourier import ModeTable
from .geometry import half_mesh_jacobian
from .input import VmecInput
from .mgrid import MgridField
from .preconditioner_2d import Prec2DConfig
from .printing import (
    FORCE_ITERATIONS_BANNER, compile_notice, improved_axis_block,
    screen_header, screen_line,
    stage_banner,
    vacuum_banner,
)
from .solver import (
    SolveResult, SolverRuntime, SpectralState, VacuumOutput,
    _LANE_EXECUTABLES, _USED_LANE_KEYS,
    _finalize, _geometry, _initial_carry, _initial_state, _leaf_signature,
    _make_body,
    _result_from_carry, _zero_cache, prepare_runtime, resolution_from_input,
    reguess_initial_axis, runtime_with_baselines,
    _resolve_use_fft,
)
from .transforms import register_pytree_dataclass as _register
from .vacuum import (
    VacuumBasis, VacuumBoundary, _analytic_terms, make_vacuum_solver, vacuum_basis,
    vacuum_channels,
)

__all__ = [
    "FreeBoundaryState",
    "VacuumOutput",
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
    # angle is ``phi = zeta * onp`` (onp = 1/nfp; matches surface.f and the
    # ``onp`` folding in ``vacuum.py``/``external_field_channels``).  Using
    # ``zeta`` directly here double-counts nfp and mis-places every n != 0
    # harmonic toroidally.
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


@functools.partial(jax.jit, static_argnames=("use_fft",))
def _jacobian_ok(state: SpectralState, rt: SolverRuntime,
                 *, use_fft: bool = False) -> Array:
    """funct3d.f Jacobian gate: True while the state's ``tau`` has one sign.

    ``funct3d.f`` calls ``jacobian`` immediately after synthesis and, when
    the sign changed (``irst = 2``, ``iequi = 0``), sets ``gc = 0`` and
    jumps past ``bcovar``, the whole ``IVAC0`` block and the force build;
    ``evolve.f`` then restores ``xc = xstore`` (``restart_iter``,
    ``iter1 = iter2``) and re-calls funct3d, whose IVAC0 block performs a
    FULL NESTOR update at the restored state (``ivacskip = 0``).  The
    free-boundary cadence uses this flag to substitute ``xstore`` as the
    NESTOR source on exactly those passes, so the vacuum never consumes a
    sign-changed state.  Evaluating NESTOR on such a transient hands
    ``analyt.f``'s ``log``/``sqrt`` kernels a degenerate boundary, and the
    resulting non-finite ``bsqvac`` turns funct3d.f's recoverable restart
    into a fatal NON-FINITE FORCE EVALUATION.

    Used by the host-stepped activation passes only.  The traced steady
    vacuum lane computes the same flag from its own single per-pass
    synthesis (``_make_vacuum_lane``), which the force evaluation then
    reuses — inlining this helper there duplicated a full geometry
    synthesis inside the compiled free-lane program.
    """
    _, geometry = _geometry(state, rt, use_fft=use_fft)
    return jnp.logical_not(
        half_mesh_jacobian(geometry, s=rt.setup.s_full).jacobian_sign_changed
    )


@functools.partial(jax.jit, static_argnames=("use_fft",))
def _iter_lane(carry, rt: SolverRuntime, *, use_fft: bool = False):
    """One jitted eqsolve iteration (shared traced body; per-``rt`` lane)."""
    return _make_body(rt, use_fft=use_fft)(carry)


@functools.partial(jax.jit, static_argnames=("use_fft",))
def _turnon_iter_lane(carry, rt: SolverRuntime, evaluation_state: SpectralState,
                      *, use_fft: bool = False):
    """VMEC2000 turn-on pass: old-geometry forces evolve restored ``xc``."""
    return _make_body(rt, evaluation_state=evaluation_state,
                      use_fft=use_fft)(carry)


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
    rbsq: np.ndarray | None = None
    # NESTOR cache (amatsav / bvecsav of scalpot.f):
    mode_matrix: Any = None
    mode_factor: Any = None
    mode_pivots: Any = None
    bvec_nonsing: Any = None
    potvac: np.ndarray | None = None
    surface_fields: Any = None
    ctor: float = 0.0
    rbtor: float = 0.0
    vacuum_calls: int = 0
    full_updates: int = 0


@dataclass(frozen=True)
class _FreeBoundaryStageResult:
    """One radial stage plus the vacuum continuation state for the next grid."""

    result: SolveResult
    vacuum: FreeBoundaryState
    continuation_state: SpectralState
    rcon0: Array
    zcon0: Array


def _vacuum_output(fb: FreeBoundaryState, basis: VacuumBasis) -> VacuumOutput | None:
    """Strip private NESTOR caches from the final public result."""
    if fb.potvac is None or fb.surface_fields is None:
        return None
    bsubu, bsubv, bsupu, bsupv = fb.surface_fields
    pot = np.asarray(fb.potvac, dtype=float).reshape(-1)
    mnpd = int(basis.mnpd)
    shape = (int(basis.ntheta3), int(basis.nzeta))
    return VacuumOutput(
        potsin=pot[:mnpd].copy(),
        potcos=(pot[mnpd:2 * mnpd].copy() if basis.lasym else None),
        xmpot=np.asarray(basis.xmpot, dtype=float).copy(),
        xnpot=(np.asarray(basis.n_raw, dtype=float) * float(basis.nfp)),
        bsubu=np.asarray(bsubu, dtype=float).reshape(shape).copy(),
        bsubv=np.asarray(bsubv, dtype=float).reshape(shape).copy(),
        bsupu=np.asarray(bsupu, dtype=float).reshape(shape).copy(),
        bsupv=np.asarray(bsupv, dtype=float).reshape(shape).copy(),
    )


def _resolve_mgrid(inp: VmecInput, mgrid_path: str | Path | None) -> Path:
    p = Path(str(mgrid_path if mgrid_path is not None else inp.mgrid_file)).expanduser()
    return p


def _external_field_from_input(
    inp: VmecInput, mgrid_path: str | Path | None = None,
) -> MgridField:
    """Load and scale the deck's external field once for one solve/ladder."""
    path = _resolve_mgrid(inp, mgrid_path)
    data_extcur = np.atleast_1d(np.asarray(
        inp.extcur if inp.extcur is not None else [], dtype=float))
    from .mgrid import read_mgrid

    data = read_mgrid(path)  # raises MgridNotFoundError when missing
    extcur = np.zeros((data.nextcur,), dtype=float)
    n_copy = min(data_extcur.size, data.nextcur)
    extcur[:n_copy] = data_extcur[:n_copy]
    if str(data.mgrid_mode).upper().startswith(("R", "N")):
        raw = np.asarray(data.raw_coil_cur, dtype=float)
        extcur = np.divide(extcur, raw, out=extcur, where=raw != 0.0)
    return MgridField.from_mgrid_data(data, extcur=extcur)


def _mgrid_planes(external_field: Any) -> int | None:
    """Toroidal planes per field period of a tabulated field, else ``None``.

    Only :class:`~vmex.core.mgrid.MgridField`-shaped objects (a 4-D ``br``
    table ``(nextcur, kp, jz, ir)``) constrain the solver's toroidal grid;
    analytic/direct-coil fields evaluate at arbitrary angles.
    """
    br = getattr(external_field, "br", None)
    if br is None:
        return None
    shape = tuple(getattr(br, "shape", ()))
    return int(shape[1]) if len(shape) == 4 else None


def free_boundary_resolution(
    inp: VmecInput, external_field: Any, *, ns: int | None = None,
):
    """``resolution_from_input`` plus VMEC2000's mgrid angular-grid policy.

    VMEC2000 requires the solver's ``NZETA`` to divide evenly into the mgrid
    file's toroidal planes per period (``mgrid_mod.f``:
    ``MOD(np0b, nv) /= 0 -> ier_flag = 9``) — its vacuum path samples the
    file planes by stride.  An incompatible pair must therefore never reach
    iteration one:

    * explicit incompatible ``NZETA`` -> typed :class:`VmecInputError`
      (VMEC2000 rejects the same deck before solving);
    * automatic resolution (``NZETA = 0``) -> the smallest divisor of the
      table's plane count at or above the ``read_indata.f`` floor
      ``2*ntor + 4`` (e.g. a 24-plane table with ``ntor = 9`` selects 24,
      not the floor 22); when even the full plane count is below the floor,
      the table cannot support the requested mode content -> typed error
      advising a finer mgrid.
    """
    resolution = resolution_from_input(inp, ns=ns)
    kp = _mgrid_planes(external_field)
    if kp is None or int(inp.ntor) == 0:
        return resolution
    if int(inp.nzeta) > 0:
        if kp % int(inp.nzeta) != 0:
            raise VmecInputError(
                f"NZETA = {int(inp.nzeta)} does not divide evenly into the "
                f"mgrid table's {kp} toroidal planes per period; VMEC2000 "
                "rejects this pairing before solving (mgrid_mod.f ier=9)",
                hint="set NZETA to a divisor of the mgrid plane count "
                     "(or NZETA = 0 for automatic compatible selection)",
            )
        return resolution
    floor = int(resolution.nzeta)  # 2*ntor + 4 automatic floor
    compatible = [d for d in range(1, kp + 1) if kp % d == 0 and d >= floor]
    if not compatible:
        raise VmecInputError(
            f"the mgrid table has {kp} toroidal planes per period, below "
            f"the automatic-resolution floor {floor} for NTOR = "
            f"{int(inp.ntor)}",
            hint="regenerate the mgrid with at least "
                 f"{floor} planes per period (kp a multiple works best)",
        )
    return replace(resolution, nzeta=int(compatible[0]))


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
    """Jitted whole-pipeline NESTOR update closures.

    ``full(state, rt, field)`` and ``skip(state, rt, field, bvec_nonsing,
    mode_factor, mode_pivots)`` each run the full per-iteration vacuum update
    as one jitted program. On GPU, only NESTOR's small dense
    assembly/factor/solve is CPU-sharded; plasma geometry, external field and
    surface fields stay on the accelerator, while the cached
    matrix/factors/RHS remain resident on the solve device between updates
    (they are consumed only there).

    ``cache_key`` is the owning ``_VACUUM_EXECUTABLE_CACHE`` key (set by
    :func:`_vacuum_executables`): the prefetch registry uses it to tell apart
    same-shaped closures built for different resolutions/signs, so a
    background-compiled executable can never be consumed by a different
    NESTOR program.
    """

    full: Any
    skip: Any
    cache_key: Any = None
    solve_device: Any = None


def _make_fused_vacuum(basis: VacuumBasis, *, modes: ModeTable, signgs: int,
                       solver_vac, axis_r0, axis_z0, solve_device=None,
                       output_device=None) -> FusedVacuum:
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

    def _move(tree, device):
        if device is None:
            return tree
        return jax.tree.map(lambda x: jax.device_put(x, device), tree)

    def _solve_full(boundary, bexni):
        boundary, bexni = _move((boundary, bexni), solve_device)
        mode_matrix, rhs, bvec_nonsing, _gsource, _grpmn = (
            solver_vac.assemble(boundary, bexni)
        )
        mode_factor, mode_pivots = jsp_linalg.lu_factor(mode_matrix)
        potvac = jsp_linalg.lu_solve((mode_factor, mode_pivots), rhs)
        # Placement only: the LU factors and cached matrix/RHS stay resident
        # on the solve device (CPU on a GPU lane) across vacuum skip steps —
        # they are consumed only by _solve_skip's CPU-sharded lu_solve, so
        # shipping them to the plasma device every full update and back on
        # every skip step is a pure dense-matrix transfer round-trip.  Values
        # are byte-identical; only potvac rejoins the plasma device.
        return (
            _move(potvac, output_device),
            mode_matrix, mode_factor, mode_pivots, bvec_nonsing,
        )

    def _solve_skip(boundary, bexni, bvec_nonsing, mode_factor, mode_pivots):
        boundary, bexni = _move((boundary, bexni), solve_device)
        bvec_nonsing, mode_factor, mode_pivots = _move(
            (bvec_nonsing, mode_factor, mode_pivots), solve_device,
        )  # no-op for the resident cache; guards continuation-supplied arrays
        bvec_analytic, _ = _analytic_terms(
            boundary, bexni, basis, signgs, include_kernel=False
        )
        rhs = bvec_nonsing + bvec_analytic
        potvac = jsp_linalg.lu_solve((mode_factor, mode_pivots), rhs)
        return _move(potvac, output_device)

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
        (potvac, mode_matrix, mode_factor, mode_pivots,
         bvec_nonsing) = _solve_full(
            boundary, ext["bexni"],
        )
        bsqvac, bsubu_s, bsubv_s, bsupu_s, bsupv_s = vacuum_channels(
            basis=basis, potvac=potvac, bexu=ext["bexu"], bexv=ext["bexv"],
            guu=ext["guu"], guv=ext["guv"], gvv=ext["gvv"],
        )
        delbsq_num, delbsq_den, bsubuvac, bsubvvac = _diagnostics(
            bsqvac, bsubu_s, bsubv_s, bsq3, pres_ns, rt
        )
        return {
            "bsqvac": bsqvac, "ctor": ctor, "rbtor": rbtor, "potvac": potvac,
            "rbsq": (bsqvac + pres_ns * jnp.asarray(rt.presf_ns_scale))
                    * boundary.R / jnp.asarray(rt.setup.hs),
            "mode_matrix": mode_matrix, "mode_factor": mode_factor,
            "mode_pivots": mode_pivots, "bvec_nonsing": bvec_nonsing,
            "surface_fields": (bsubu_s, bsubv_s, bsupu_s, bsupv_s),
            "delbsq_num": delbsq_num, "delbsq_den": delbsq_den,
            "bsubuvac": bsubuvac, "bsubvvac": bsubvvac,
        }

    def _skip(state: SpectralState, rt: SolverRuntime, field: MgridField,
              bvec_nonsing, mode_factor, mode_pivots):
        ctor, rbtor, axis_r, axis_z, bsq3, pres_ns = _vacuum_scalars(state, rt)
        rmnc, zmns, rmns, zmnc = _edge_fourier_jax(state, rt)
        boundary = _boundary_from_coefficients_jax(
            rmnc, zmns, rmns, zmnc, modes=modes, basis=basis
        )
        ext = _pipeline(field, boundary, ctor, axis_r, axis_z)
        potvac = _solve_skip(
            boundary, ext["bexni"], bvec_nonsing, mode_factor, mode_pivots
        )
        bsqvac, bsubu_s, bsubv_s, bsupu_s, bsupv_s = vacuum_channels(
            basis=basis, potvac=potvac, bexu=ext["bexu"], bexv=ext["bexv"],
            guu=ext["guu"], guv=ext["guv"], gvv=ext["gvv"],
        )
        delbsq_num, delbsq_den, bsubuvac, bsubvvac = _diagnostics(
            bsqvac, bsubu_s, bsubv_s, bsq3, pres_ns, rt
        )
        return {
            "bsqvac": bsqvac, "ctor": ctor, "rbtor": rbtor, "potvac": potvac,
            "rbsq": (bsqvac + pres_ns * jnp.asarray(rt.presf_ns_scale))
                    * boundary.R / jnp.asarray(rt.setup.hs),
            "surface_fields": (bsubu_s, bsubv_s, bsupu_s, bsupv_s),
            "delbsq_num": delbsq_num, "delbsq_den": delbsq_den,
            "bsubuvac": bsubuvac, "bsubvvac": bsubvvac,
        }

    return FusedVacuum(
        full=jax.jit(_full), skip=jax.jit(_skip), solve_device=solve_device,
    )


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
#: greenf/analyt/solve kernels (~5 s on CPU) every solve. Keyed on
#: resolution/sign/device structure; the boundary/profile/mgrid values
#: enter the jitted program as traced arguments, so one executable serves every
#: solve at a given resolution.  The third element is the jitted steady-state
#: free-boundary loop (:func:`_make_vacuum_lane`) built over the same fused
#: vacuum closures.
_VACUUM_EXECUTABLE_CACHE: dict[
    tuple[Any, ...], tuple[VacuumBasis, FusedVacuum, Any]
] = {}


def _vacuum_executables(resolution, *, mf: int, nf: int, signgs: int, wint,
                        modes: ModeTable, axis_r0, axis_z0,
                        use_fft: bool = False) -> tuple[VacuumBasis, FusedVacuum, Any]:
    """Return the cached ``(basis, fused vacuum, steady lane)`` for one resolution/signgs.

    ``wint``/``modes`` are resolution-determined build inputs consumed only on
    a cache miss.  The dynamic axis is validated on every call.  ``use_fft``
    is part of the cache key: the steady vacuum lane bakes the synthesis
    kernel into its traced body, so the two kernels must never share an
    executable.
    """
    plasma_device = next(iter(jnp.asarray(axis_r0).devices()))
    solve_device = (
        jax.devices("cpu")[0] if plasma_device.platform == "gpu" else None
    )
    key = (
        resolution, int(signgs), int(mf), int(nf), bool(use_fft),
        str(plasma_device), str(solve_device),
    )
    cached = _VACUUM_EXECUTABLE_CACHE.get(key)
    if cached is not None:
        _assert_static_filament_topology(cached[0], axis_r0, axis_z0)
        return cached
    basis = vacuum_basis(
        mf=int(mf), nf=int(nf), ntheta3=int(resolution.ntheta3),
        nzeta=int(resolution.nzeta), nfp=int(resolution.nfp),
        lasym=bool(resolution.lasym), wint=wint,
    )
    solver_vac = make_vacuum_solver(basis, signgs=int(signgs))
    fused = replace(_make_fused_vacuum(
        basis, modes=modes, signgs=int(signgs), solver_vac=solver_vac,
        axis_r0=axis_r0, axis_z0=axis_z0, solve_device=solve_device,
        output_device=plasma_device if solve_device is not None else None,
    ), cache_key=key)
    lane = _make_vacuum_lane(fused, use_fft=use_fft)
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

    The whole update runs as one jitted program (:class:`FusedVacuum`). On a
    GPU lane the dense NESTOR block is CPU-sharded and its cached
    matrix/factor/RHS stay resident there across skip steps, while ``bsqvac``
    lives on the selected plasma device. Only a few diagnostic scalars reach
    Python.
    """
    ns_notice = int(rt.resolution.ns)
    if (
        int(ivacskip) == 0
        or fb.mode_matrix is None
        or fb.mode_factor is None
        or fb.mode_pivots is None
    ):
        if fused_vac.cache_key is None:
            out = fused_vac.full(carry.state, rt, field)
        else:
            out = _call_lane(
                ("fb_vacfull", fused_vac.cache_key),
                fused_vac.full, (carry.state, rt, field),
                notice=(emit, ns_notice, "NESTOR full-update")
                if verbose else None,
            )
        fb.mode_matrix = out["mode_matrix"]
        fb.mode_factor = out["mode_factor"]
        fb.mode_pivots = out["mode_pivots"]
        fb.bvec_nonsing = out["bvec_nonsing"]
        fb.full_updates += 1
    else:
        skip_args = (carry.state, rt, field, fb.bvec_nonsing,
                     fb.mode_factor, fb.mode_pivots)
        if fused_vac.cache_key is None:
            out = fused_vac.skip(*skip_args)
        else:
            out = _call_lane(
                ("fb_vacskip", fused_vac.cache_key),
                fused_vac.skip, skip_args,
                notice=(emit, ns_notice, "NESTOR skip-update")
                if verbose else None,
            )
    bsqvac = out["bsqvac"]
    fb.rbsq = out["rbsq"]
    fb.potvac = out["potvac"]
    fb.surface_fields = out["surface_fields"]
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
# Overlapped lane compilation (free-boundary compile scheduling)
# ---------------------------------------------------------------------------
#
# A cold free-boundary rung pays a SERIAL chain of lane compiles: the first
# force pass (``_iter_lane``), the pre-activation while-loop, the fused
# NESTOR full update, the turn-on pass, and finally the steady-state vacuum
# loop — each blocking iteration progress while it builds.  Every one of
# those structures is known the moment the stage's runtimes exist (the
# activation ITERATION is runtime-dependent, the lane STRUCTURES are not),
# so a background thread AOT-compiles the post-entry lanes while the main
# thread compiles/runs the entry phase, and the multigrid driver prefetches
# the next rung's whole set while the current rung iterates — the
# free-boundary counterpart of ``solver._prefetch_block_lane``.
#
# The standalone :class:`jax.stages.Compiled` objects live in the solver's
# shared ``_LANE_EXECUTABLES`` registry, keyed on ``(tag, treedef,
# leaf-avals)`` where the tag carries the lane name plus — for the
# per-basis NESTOR closures — the owning ``_VACUUM_EXECUTABLE_CACHE`` key,
# so same-shaped programs with different baked constants can never be
# confused.  Consumption (:func:`_call_lane`) falls back silently to the
# ordinary jitted lane on any mismatch.  This is cache warming only: a
# prefetched executable is XLA output for the IDENTICAL lowering, so every
# iteration is bit-for-bit the on-demand result.  Standalone executables
# survive ``jax.clear_caches()`` (the driver's ``release_stage_cache``
# point), and consumed ones are dropped there by the shared
# ``solver._release_used_lane_executables``.

#: Cheap proxy keys of stage lane-sets a prefetch thread was already
#: launched for in this process (mirrors ``multigrid._PREFETCH_ATTEMPTED``).
_FB_PREFETCH_ATTEMPTED: set = set()

#: Keys whose background compile is currently in flight, mapped to the event
#: set when it finishes.  Serves two purposes: duplicate-launch suppression
#: between the within-stage and next-rung prefetch threads, and the
#: join-in-progress path of :func:`_call_lane` — a background compile that
#: started EARLIER always finishes before a fresh main-thread compile of the
#: same lowering would, so the main thread waits for it instead of
#: duplicating the work.
_FB_INFLIGHT: dict = {}
_FB_LOCK = threading.Lock()


def _fb_lane_signature(tag: Any, args: tuple) -> tuple:
    """Structural key of one ``lane(*args)`` call (tag + treedef + avals)."""
    leaves, treedef = jax.tree_util.tree_flatten(args)
    return (tag, treedef, tuple(_leaf_signature(leaf) for leaf in leaves))


def _call_lane(tag: Any, lane: Any, args: tuple,
               static: dict[str, Any] | None = None,
               *, notice: tuple | None = None) -> Any:
    """Run ``lane(*args, **static)``, preferring a prefetched executable.

    ``static`` holds jit static kwargs (baked into a prefetched executable's
    lowering; the tag encodes them).  A background compile still in flight
    for exactly this structure is joined rather than duplicated (it started
    earlier, so it finishes first; the worker signals the event in a
    ``finally``, so the wait always terminates).  A structural/placement
    mismatch — the executable's argument validation precedes execution, so
    ``args`` are intact — falls back to the ordinary jitted lane.

    ``notice = (emit, ns, label)`` extends the fixed-boundary
    ``compile_notice`` convention to the free lanes: the first call for a
    given structure in this process emits the one-line attribution BEFORE
    the compile (or before joining an in-flight background compile), so a
    file-redirected cluster log shows what a silent multi-minute pause is
    building.  The caller passes it only when verbose; the emit must flush
    (the CLI sink does).
    """
    kwargs = static or {}
    if jax.config.jax_disable_jit:
        return lane(*args, **kwargs)
    if notice is None and not _LANE_EXECUTABLES and not _FB_PREFETCH_ATTEMPTED:
        return lane(*args, **kwargs)
    key = _fb_lane_signature(tag, args)
    with _FB_LOCK:
        pending = _FB_INFLIGHT.get(key)
    if notice is not None and key not in _USED_LANE_KEYS:
        emit, ns, label = notice
        emit(compile_notice(
            int(ns), lane=str(label),
            prefetched=(pending is not None or key in _LANE_EXECUTABLES),
        ), end="")
    _USED_LANE_KEYS.add(key)
    if pending is not None:
        pending.wait()
    executable = _LANE_EXECUTABLES.get(key)
    if executable is not None:
        try:
            return executable(*args)
        except Exception:
            pass
    return lane(*args, **kwargs)


def _prefetch_stage_lane_set(
    inp: VmecInput, *, external_field: Any, ns: int, ftol: float | None,
    max_iterations: int | None, time_step: float | None, tcon0: float | None,
    gamma: float | None, nstep: int | None, lconm1: bool,
    precon_type: str | None, prec2d_threshold: float | None, prec2d: Any,
    use_fft: bool, device: Any, lmove_axis: bool | None,
    vacuum_on: bool | None, entry_lanes: bool, cancel: threading.Event,
) -> None:
    """AOT-compile one free-boundary stage's lane set (background worker).

    Builds runtime/carry templates exactly as ``_solve_free_boundary_stage``
    does (cheap structure assembly — no force evaluation) and compiles the
    lanes the stage will need: the entry lanes first (``entry_lanes`` is set
    by the cross-rung launch only — the within-stage launch skips them
    because the main thread is already compiling them), then the
    post-activation set ordered by the scheduling rationale on the job list
    below.  ``vacuum_on`` prunes the set when the stage's entry mode is
    known (``None`` compiles both).  Any failure leaves the affected lane
    to on-demand compilation.
    """
    resolution = free_boundary_resolution(inp, external_field, ns=int(ns))
    with device_context(device, resolution):
        rt = prepare_runtime(
            inp, resolution, ftol=ftol, max_iterations=max_iterations,
            time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
            lconm1=lconm1, precon_type=precon_type,
            prec2d_threshold=prec2d_threshold, prec2d=prec2d, use_fft=use_fft,
        )
        if lmove_axis is not None and bool(rt.lmove_axis) != bool(lmove_axis):
            rt = replace(rt, lmove_axis=bool(lmove_axis))
        state = _initial_state(rt.setup)
        axis_r0, axis_z0 = _vacuum_scalars(state, rt)[2:4]
        basis, fused, vacuum_lane = _vacuum_executables(
            resolution, mf=int(inp.mpol) + 1, nf=int(inp.ntor),
            signgs=int(rt.setup.signgs),
            wint=np.asarray(rt.trig.wint, dtype=float),
            modes=rt.modes, axis_r0=axis_r0, axis_z0=axis_z0, use_fft=use_fft,
        )
        ns_i = int(resolution.ns)
        dtype = rt.setup.s_full.dtype
        zeros_edge = jnp.zeros((basis.ntheta3, basis.nzeta), dtype=dtype)
        rt_fixed = replace(rt, lfreeb=False, bsqvac_edge=zeros_edge,
                           presf_ns_scale=jnp.asarray(0.0, dtype=dtype))
        rt_freeb = replace(
            rt, lfreeb=True, jmax=ns_i, bsqvac_edge=zeros_edge,
            presf_ns_scale=jnp.asarray(
                _presf_ns_scale(inp, ns_i), dtype=dtype),
        )
        carry_fixed = _initial_carry(state, rt_fixed, ijacob=0)
        carry_freeb = _initial_carry(state, rt_freeb, ijacob=0)
        out = jax.eval_shape(fused.full, state, rt_freeb, external_field)
        z = lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype)  # noqa: E731
        zs = (z if fused.solve_device is None else
              (lambda sd: jax.device_put(z(sd), fused.solve_device)))
        int_dtype = carry_freeb.iteration.dtype
        vc = _VacuumLoopCarry(
            carry=carry_freeb,
            rcon0=rt_freeb.rcon0, zcon0=rt_freeb.zcon0,
            bsqvac=zeros_edge, rbsq=z(out["rbsq"]),
            mode_matrix=zs(out["mode_matrix"]),
            mode_factor=zs(out["mode_factor"]),
            mode_pivots=zs(out["mode_pivots"]),
            bvec_nonsing=zs(out["bvec_nonsing"]), potvac=z(out["potvac"]),
            surface_fields=tuple(z(s) for s in out["surface_fields"]),
            ivac=jnp.asarray(1, dtype=int_dtype),
            nvacskip=jnp.asarray(1, dtype=int_dtype),
            nvskip0=jnp.asarray(1, dtype=int_dtype),
            delbsq=jnp.asarray(1.0, dtype=dtype),
            delbsq_traj=jnp.full((rt.max_iterations,), np.nan, dtype=dtype),
            ctor=jnp.asarray(0.0, dtype=dtype),
            rbtor=jnp.asarray(0.0, dtype=dtype),
            vacuum_calls=jnp.asarray(0, dtype=int_dtype),
            full_updates=jnp.asarray(0, dtype=int_dtype),
        )

        # Job order is first-use order for the entry lanes (the next rung
        # needs them the moment it starts), then LONGEST-FIRST for the
        # post-activation set: the steady vacuum loop is both the costliest
        # compile and the stage's critical path, and ``_call_lane`` joining
        # an in-flight compile means starting it as early as possible always
        # wins, while the cheaper fused-full/turn-on lanes compile on the
        # main thread in parallel exactly as they would without prefetch.
        fft = {"use_fft": use_fft}
        jobs: list[tuple[Any, Any, tuple, dict[str, Any]]] = []
        if entry_lanes and vacuum_on is not True:
            jobs += [
                (("fb_iter", use_fft), _iter_lane, (carry_fixed, rt_fixed), fft),
                (("fb_preact", use_fft), _preactivation_lane,
                 (carry_fixed, rt_fixed), fft),
            ]
        if entry_lanes and vacuum_on is not False:
            jobs.append(
                (("fb_iter", use_fft), _iter_lane, (carry_freeb, rt_freeb), fft))
        jobs.append((("fb_vac", fused.cache_key), vacuum_lane,
                     (vc, rt_freeb, external_field), {}))
        if vacuum_on is not True:
            jobs.append((("fb_turnon", use_fft), _turnon_iter_lane,
                         (carry_freeb, rt_freeb, state), fft))
        jobs.append((("fb_vacfull", fused.cache_key), fused.full,
                     (state, rt_freeb, external_field), {}))

        for tag, lane, args, static in jobs:
            if cancel.is_set():
                return
            key = _fb_lane_signature(tag, args)
            with _FB_LOCK:
                if (key in _LANE_EXECUTABLES or key in _USED_LANE_KEYS
                        or key in _FB_INFLIGHT):
                    continue
                done = threading.Event()
                _FB_INFLIGHT[key] = done
            try:
                _LANE_EXECUTABLES[key] = lane.lower(*args, **static).compile()
            finally:
                done.set()
                with _FB_LOCK:
                    _FB_INFLIGHT.pop(key, None)


def _launch_free_lane_prefetch(
    inp: VmecInput, *, external_field: Any, ns: int, ftol: float | None,
    max_iterations: int | None, time_step: float | None, tcon0: float | None,
    gamma: float | None, nstep: int | None, lconm1: bool,
    precon_type: str | None, prec2d_threshold: float | None, prec2d: Any,
    use_fft: bool, device: Any, lmove_axis: bool | None,
    vacuum_on: bool | None, entry_lanes: bool,
) -> tuple[threading.Thread, threading.Event] | None:
    """Start a stage lane-set prefetch thread; ``None`` when not applicable.

    Cache warming only (see the section comment): any failure inside the
    worker leaves every lane to on-demand compilation.  Returns the thread
    and its cancel event; setting the event stops the worker at the next
    lane boundary (an in-progress XLA compile cannot be interrupted).
    """
    if jax.config.jax_disable_jit:
        return None
    attempt_key = (
        int(ns), None if ftol is None else float(ftol),
        None if max_iterations is None else int(max_iterations),
        bool(use_fft), int(inp.mpol), int(inp.ntor), int(inp.nfp),
        bool(inp.lasym), bool(lconm1), time_step, tcon0, gamma, nstep,
        precon_type, prec2d_threshold, prec2d is None, str(device),
        lmove_axis, vacuum_on, bool(entry_lanes),
    )
    if attempt_key in _FB_PREFETCH_ATTEMPTED:
        return None
    _FB_PREFETCH_ATTEMPTED.add(attempt_key)
    cancel = threading.Event()

    def worker() -> None:
        try:
            _prefetch_stage_lane_set(
                inp, external_field=external_field, ns=ns, ftol=ftol,
                max_iterations=max_iterations, time_step=time_step,
                tcon0=tcon0, gamma=gamma, nstep=nstep, lconm1=lconm1,
                precon_type=precon_type, prec2d_threshold=prec2d_threshold,
                prec2d=prec2d, use_fft=use_fft, device=device,
                lmove_axis=lmove_axis, vacuum_on=vacuum_on,
                entry_lanes=entry_lanes, cancel=cancel,
            )
        except Exception:      # silent fallback: on-demand compilation
            pass

    thread = threading.Thread(
        target=worker, name="vmex-fb-lane-prefetch", daemon=True)
    thread.start()
    return thread, cancel


# ---------------------------------------------------------------------------
# Batched iteration lanes (NESTOR loop batching)
# ---------------------------------------------------------------------------
#
# The two lanes below keep the per-pass host driver's scaffolding on-device
# without touching the per-iteration numerics (same ops, same order):
#
# - :func:`_preactivation_lane` runs every fixed-boundary iteration BEFORE
#   vacuum activation as one ``lax.while_loop``, exiting exactly where the
#   host driver would enter the funct3d.f IVAC0 block.
# - :func:`_make_vacuum_lane` runs the whole POST-turn-on steady state as one
#   ``lax.while_loop`` replaying the host pass verbatim in traced form: ivac
#   increment, 0.9 rcon0/zcon0 damping, the ivacskip/nvacskip cadence, the
#   full/skip fused NESTOR update, the ``bsqvac_edge`` runtime update, then
#   the shared ``_make_body`` iteration.  ``delbsq_traj`` records the
#   per-iteration DEL-BSQ diagnostic for verbose screen lines.
#
# Only the single turn-on pass (first vacuum call, ``In VACUUM`` block, soft
# restart, ``VACUUM PRESSURE TURNED ON`` banner) stays host-stepped: it
# mutates the carry/runtime once, between the two lanes.


@functools.partial(jax.jit, static_argnames=("use_fft",))
def _preactivation_lane(carry, rt: SolverRuntime, *, use_fft: bool = False):
    """Fixed-boundary iterations up to vacuum activation, as one jitted loop.

    Iterates the shared traced body while the run is live and the funct3d.f
    activation condition (``iter2 > 1 and fsqr + fsqz <= 1e-3``) has not yet
    fired; the returned carry is exactly the carry the per-pass host driver
    held when it first entered the IVAC0 block (or finished the run).
    """
    body = _make_body(rt, use_fft=use_fft)

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
    rbsq: Array                 # carried forces.f edge-pressure product
    mode_matrix: Array          # amatsav (scalpot.f)
    mode_factor: Array          # DGETRF factors of amatsav
    mode_pivots: Array
    bvec_nonsing: Array         # bvecsav (scalpot.f)
    potvac: Array
    surface_fields: tuple[Array, Array, Array, Array]
    ivac: Array; nvacskip: Array; nvskip0: Array
    delbsq: Array; delbsq_traj: Array
    ctor: Array; rbtor: Array
    vacuum_calls: Array; full_updates: Array


_register(_VacuumLoopCarry)


def _make_vacuum_lane(fused: FusedVacuum, *, use_fft: bool = False):
    """Build the jitted steady-state (post-turn-on) free-boundary loop.

    Closed over the per-basis :class:`FusedVacuum` (cached alongside it in
    ``_VACUUM_EXECUTABLE_CACHE``); ``rt``/``field``/carry enter as traced
    arguments, so one executable serves every solve at a given resolution.
    """

    def _pass(vc: _VacuumLoopCarry, rt: SolverRuntime, field) -> _VacuumLoopCarry:
        c = vc.carry
        it = c.iteration
        fsq_rz = c.fsqr + c.fsqz

        # funct3d.f computes geometry and its Jacobian ONCE per pass, before
        # bcovar/IVAC0, and the force build shares them.  Synthesize once
        # here: the Jacobian sign feeds the vacuum-source gate below and the
        # same synthesis is handed to the force evaluation at the end of the
        # pass (``evaluation_synthesis``), instead of inlining a second full
        # geometry synthesis into this traced lane (the old ``_jacobian_ok``
        # call).  Same ops on the same state — rows are bit-identical.
        coeffs, geometry = _geometry(c.state, rt, use_fft=use_fft)
        jacobian = half_mesh_jacobian(geometry, s=rt.setup.s_full)

        # NESTOR must never consume a sign-changed state (full funct3d.f/
        # evolve.f rationale: _jacobian_ok): feed it xstore and force a full
        # update whenever the current state is invalid.
        good = jnp.logical_not(jacobian.jacobian_sign_changed)
        vac_state = jax.tree.map(
            lambda a, b: jnp.where(good, a, b), c.state, c.xstore)

        # -- funct3d.f IVAC0 block (traced; same order as the host pass) ----
        ivac = vc.ivac + ((it > 1) & (fsq_rz <= ACTIVATION_FSQ)).astype(vc.ivac.dtype)
        rcon0 = 0.9 * vc.rcon0
        zcon0 = 0.9 * vc.zcon0
        one = jnp.ones((), dtype=vc.nvacskip.dtype)
        ivacskip = jnp.where(
            (ivac <= 2) | jnp.logical_not(good), jnp.zeros_like(vc.nvacskip),
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
            out = fused.full(vac_state, rt_vac, field)
            return (out["bsqvac"], out["rbsq"], out["ctor"], out["rbtor"], out["potvac"],
                    out["surface_fields"],
                    out["delbsq_num"], out["delbsq_den"],
                    out["mode_matrix"], out["mode_factor"],
                    out["mode_pivots"], out["bvec_nonsing"])

        def _skip(_):
            out = fused.skip(vac_state, rt_vac, field, vc.bvec_nonsing,
                             vc.mode_factor, vc.mode_pivots)
            return (out["bsqvac"], out["rbsq"], out["ctor"], out["rbtor"], out["potvac"],
                    out["surface_fields"],
                    out["delbsq_num"], out["delbsq_den"],
                    vc.mode_matrix, vc.mode_factor,
                    vc.mode_pivots, vc.bvec_nonsing)

        (bsqvac, rbsq, ctor, rbtor, potvac, surface_fields,
         num, den, mode_matrix, mode_factor, mode_pivots,
         bvec_nonsing) = lax.cond(
             full, _full, _skip, None)

        delbsq = jnp.where(den != 0.0, num / den, vc.delbsq)
        idx = jnp.clip(it - 1, 0, rt.max_iterations - 1)
        delbsq_traj = lax.dynamic_update_slice_in_dim(
            vc.delbsq_traj, delbsq[None], idx, axis=0)

        # -- one eqsolve iteration with the refreshed edge field ------------
        # The force pass reuses the pass's single synthesis (funct3d.f
        # ordering: forces consume the geometry computed before IVAC0).
        new_carry = _make_body(
            replace(rt_vac, bsqvac_edge=bsqvac), use_fft=use_fft,
            evaluation_synthesis=(coeffs, geometry, jacobian),
        )(c)

        return _VacuumLoopCarry(
            carry=new_carry, rcon0=rcon0, zcon0=zcon0, bsqvac=bsqvac,
            rbsq=rbsq,
            mode_matrix=mode_matrix, mode_factor=mode_factor,
            mode_pivots=mode_pivots, bvec_nonsing=bvec_nonsing,
            potvac=potvac,
            surface_fields=surface_fields,
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


def _solve_free_boundary_stage(
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
    initial_state: SpectralState | None = None,
    vacuum_continuation: FreeBoundaryState | None = None,
    time_step: float | None = None,
    tcon0: float | None = None,
    gamma: float | None = None,
    nstep: int | None = None,
    lconm1: bool = True,
    precon_type: str | None = None,
    prec2d_threshold: float | None = None,
    prec2d: Prec2DConfig | None = None,
    jacobian_retries: int = 2,
    constraint_continuation: tuple[Array, Array] | None = None,
    reuse_vacuum_cache: bool = False,
    allow_initial_axis_reguess: bool = True,
    use_fft: bool = False,
    emit_legend: bool = True,
    residual_continuation: (
        tuple[float | Array, float | Array, float | Array] | None
    ) = None,
    prefetch_compile: bool = False,
    prefetch_device: Any = None,
) -> _FreeBoundaryStageResult:
    """Internal single-grid free-boundary stage with multigrid continuation.

    ``external_field`` overrides the mgrid file (any
    :class:`~vmex.core.mgrid.MgridField`-compatible object with a
    ``b_cyl(r, phi, z)`` method — e.g. a direct-coil Biot-Savart field).
    Raises :class:`~vmex.core.errors.MgridNotFoundError` when the deck's
    mgrid file is missing and no field is supplied (callers such as the CLI
    implement VMEC2000's warn-and-fall-back-to-fixed-boundary policy).

    ``error_on_no_convergence=False`` returns the final state instead of
    raising when NITER is exhausted (useful against unconverged goldens).
    After VMEC2000's fatal 75-Jacobian-reset condition,
    ``jacobian_retries`` restarts the best finite plasma checkpoint with
    zero velocity and a halved/capped ``DELT``.  Vacuum/NESTOR structures are
    rebuilt for that checkpoint.  Set zero for the exact VMEC2000 stop.

    ``prefetch_compile=True`` overlaps this stage's post-entry lane
    compilation with its entry phase (see the free-boundary compile
    scheduling section above; results are bit-identical).
    ``prefetch_device`` is the multigrid driver's placement policy, consumed
    only by the prefetch worker.
    """
    if int(jacobian_retries) < 0:
        raise ValueError("jacobian_retries must be non-negative")
    if not bool(inp.lfreeb):
        raise ValueError("solve_free_boundary requires an LFREEB=T input")
    if external_field is None:
        external_field = _external_field_from_input(inp, mgrid_path)

    if resolution is None:
        resolution = free_boundary_resolution(inp, external_field)
    else:
        # A supplied resolution must still satisfy VMEC2000's angular
        # compatibility rule against a tabulated field — never let an
        # incompatible pairing reach iteration one (it produces NaN after
        # vacuum activation; VMEC2000 rejects it before solving).
        kp = _mgrid_planes(external_field)
        if (kp is not None and int(resolution.ntor) > 0
                and kp % int(resolution.nzeta) != 0):
            raise VmecInputError(
                f"solver NZETA = {int(resolution.nzeta)} does not divide "
                f"evenly into the mgrid table's {kp} toroidal planes per "
                "period (mgrid_mod.f ier=9)",
                hint="use free_boundary_resolution(), NZETA = 0, or a "
                     "divisor of the mgrid plane count",
            )
    rt = prepare_runtime(
        inp, resolution, ftol=ftol, max_iterations=max_iterations,
        time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
        lconm1=lconm1, precon_type=precon_type,
        prec2d_threshold=prec2d_threshold, prec2d=prec2d,
        use_fft=use_fft,
    )
    if not allow_initial_axis_reguess:
        rt = replace(rt, lmove_axis=False)
    ns = int(resolution.ns)
    dtype = rt.setup.s_full.dtype

    if initial_state is None:
        _init_state = _initial_state(rt.setup)
    else:
        expected = (ns, rt.modes.mnmax)
        if tuple(initial_state.R_cos.shape) != expected:
            raise ValueError(
                f"initial_state has shape {tuple(initial_state.R_cos.shape)}, "
                f"expected {expected}; interpolate with "
                "vmex.core.multigrid.interpolate_state first"
            )
        _init_state = initial_state
        # Unlike fixed boundary, do not replace the edge row: it is an evolved
        # unknown.  On a reset-style start (IVAC=-1), funct3d.f seeds the
        # constraint baseline from this state at iter2==iter1.  Across an
        # already-active radial transition, allocate_ns creates new zeroed
        # rcon0/zcon0 arrays and funct3d deliberately does not reseed them
        # (the seeding predicate requires IVAC<=0).
        if constraint_continuation is not None:
            rt = replace(
                rt, rcon0=jnp.asarray(constraint_continuation[0]),
                zcon0=jnp.asarray(constraint_continuation[1]))
        elif (vacuum_continuation is not None
              and vacuum_continuation.turned_on):
            rt = replace(
                rt, rcon0=jnp.zeros_like(rt.rcon0),
                zcon0=jnp.zeros_like(rt.zcon0))
        else:
            rt = runtime_with_baselines(rt, _init_state, use_fft=use_fft)
    _initial_ijacob = 0
    # ``eqsolve.f`` retries a supplied axis when the first Jacobian changes
    # sign.  The fixed-boundary driver already did this in ``_solve_stage``;
    # free boundary must do it *before* constructing the fused axis-current
    # filament, otherwise a recoverable bad axis can fail the static-topology
    # guard before iteration 1.
    _, _initial_geometry = _geometry(_init_state, rt)
    _initial_jacobian = half_mesh_jacobian(_initial_geometry, s=rt.setup.s_full)
    if (
        allow_initial_axis_reguess
        and bool(_initial_jacobian.jacobian_sign_changed)
        and ns >= 3
    ):
        if verbose:
            emit(" INITIAL JACOBIAN CHANGED SIGN!")
            emit(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS")
        rt, _init_state, _axis = reguess_initial_axis(rt, _init_state)
        if verbose:
            emit(improved_axis_block(
                _axis[0], _axis[3],
                raxis_cs=_axis[1] if resolution.lasym else None,
                zaxis_cc=_axis[2] if resolution.lasym else None,
            ), end="")
        _initial_ijacob = 1
        _, _retry_geometry = _geometry(_init_state, rt)
        _retry_jacobian = half_mesh_jacobian(_retry_geometry, s=rt.setup.s_full)
        if bool(_retry_jacobian.jacobian_sign_changed):
            # Preserve the normal typed solver failure and its remedy hint;
            # do not let the later axis-filament topology guard obscure it.
            from .errors import BAD_JACOBIAN_FLAG, VmecJacobianError, WERROR_MESSAGES
            raise VmecJacobianError(
                WERROR_MESSAGES[BAD_JACOBIAN_FLAG],
                hint="decrease DELT or provide a better RAXIS_*/ZAXIS_* guess",
                ier_flag=BAD_JACOBIAN_FLAG, iteration=1,
                jacobian_resets=1, fsq=(1.0, 1.0, 1.0),
            )
    _axis_r0, _axis_z0 = _vacuum_scalars(_init_state, rt)[2:4]
    basis, fused_vac, vacuum_lane = _vacuum_executables(
        resolution, mf=int(inp.mpol) + 1, nf=int(inp.ntor),
        signgs=int(rt.setup.signgs), wint=np.asarray(rt.trig.wint, dtype=float),
        modes=rt.modes, axis_r0=_axis_r0, axis_z0=_axis_z0, use_fft=use_fft,
    )

    zeros_edge = jnp.zeros((basis.ntheta3, basis.nzeta), dtype=dtype)
    carried_edge = zeros_edge
    if (vacuum_continuation is not None
            and vacuum_continuation.turned_on):
        if vacuum_continuation.rbsq is not None:
            # rbsq is a persistent angular array in VMEC2000.  At iter2=1
            # funct3d skips IVAC0, so forces.f consumes the coarse-stage rbsq
            # verbatim.  Express that product as an equivalent bsqvac on the
            # new geometry for the shared force seam.
            _, carried_geometry = _geometry(_init_state, rt)
            r_edge = carried_geometry.R_even[-1] + carried_geometry.R_odd[-1]
            pres_ns = _vacuum_scalars(_init_state, rt)[5]
            rbsq = jnp.asarray(vacuum_continuation.rbsq, dtype=dtype)
            carried_edge = jnp.where(
                r_edge != 0.0, rbsq * jnp.asarray(rt.setup.hs) / r_edge,
                jnp.zeros_like(r_edge),
            ) - pres_ns * jnp.asarray(_presf_ns_scale(inp, ns), dtype=dtype)
        elif vacuum_continuation.bsqvac is not None:
            carried_edge = jnp.asarray(vacuum_continuation.bsqvac, dtype=dtype)
        if tuple(carried_edge.shape) != tuple(zeros_edge.shape):
            raise ValueError(
                "carried vacuum-pressure grid has shape "
                f"{tuple(carried_edge.shape)}, expected {tuple(zeros_edge.shape)}"
            )
    rt_fixed = replace(rt, lfreeb=False, bsqvac_edge=zeros_edge,
                       presf_ns_scale=jnp.asarray(0.0, dtype=dtype))
    rt_freeb = replace(
        rt, lfreeb=True, jmax=ns, bsqvac_edge=carried_edge,
        presf_ns_scale=jnp.asarray(_presf_ns_scale(inp, ns), dtype=dtype),
    )

    if vacuum_continuation is None:
        fb = FreeBoundaryState(
            ivac=-1,
            nvacskip=max(1, int(inp.nvacskip)),
            nvskip0=max(1, int(inp.nvacskip)),
        )
    else:
        # runvmec.f does not reset IVAC or the adaptive NVACSKIP at an
        # initialize_radial.f transition.  Carry those scalar controls (and
        # DEL-BSQ for screen parity).  A changed radial grid discards the
        # dynamic NESTOR matrix/vector; an equal-grid rerun preserves them,
        # matching initialize_radial.f's early return.
        fb = FreeBoundaryState(
            ivac=int(vacuum_continuation.ivac),
            nvacskip=max(1, int(vacuum_continuation.nvacskip)),
            nvskip0=max(1, int(vacuum_continuation.nvskip0)),
            turned_on=bool(vacuum_continuation.turned_on),
            delbsq=float(vacuum_continuation.delbsq),
            bsqvac=vacuum_continuation.bsqvac,
            rbsq=vacuum_continuation.rbsq,
            ctor=float(vacuum_continuation.ctor),
            rbtor=float(vacuum_continuation.rbtor),
            mode_matrix=(vacuum_continuation.mode_matrix
                         if reuse_vacuum_cache else None),
            mode_factor=(vacuum_continuation.mode_factor
                         if reuse_vacuum_cache else None),
            mode_pivots=(vacuum_continuation.mode_pivots
                         if reuse_vacuum_cache else None),
            bvec_nonsing=(vacuum_continuation.bvec_nonsing
                          if reuse_vacuum_cache else None),
            potvac=(vacuum_continuation.potvac
                    if reuse_vacuum_cache else None),
            surface_fields=(vacuum_continuation.surface_fields
                            if reuse_vacuum_cache else None),
        )
    vacuum_active = fb.turned_on

    # Overlapped lane compilation (opt-in; see the compile-scheduling section
    # above): while the main thread compiles/runs this stage's entry lanes, a
    # background thread AOT-compiles the lanes activation will need.  The
    # thread is joined before every exit below — a mid-compile
    # ``jax.clear_caches()`` from the driver's ``release_stage_cache`` point
    # is the one interaction the overlap must never have.
    prefetch = None
    if prefetch_compile:
        prefetch = _launch_free_lane_prefetch(
            inp, external_field=external_field, ns=ns, ftol=float(rt.ftol),
            max_iterations=int(rt.max_iterations),
            time_step=float(rt.time_step0), tcon0=float(rt.tcon0),
            gamma=float(rt.gamma), nstep=int(rt.nstep), lconm1=lconm1,
            precon_type=precon_type, prec2d_threshold=prec2d_threshold,
            prec2d=prec2d, use_fft=use_fft, device=prefetch_device,
            lmove_axis=bool(rt.lmove_axis), vacuum_on=bool(vacuum_active),
            entry_lanes=False,
        )

    def _lane_notice(label: str):
        """Verbose ``compile_notice`` payload for the free lanes: every
        free-lane compile pause is attributed in the (flushed) log BEFORE
        the compile starts."""
        return (emit, ns, label) if verbose else None

    try:
        if verbose:
            emit(stage_banner(ns, resolution.mnmax, rt.ftol, rt.max_iterations), end="")
            # runvmec.f prints the residual legend once per run; later radial
            # rungs of a ladder keep only the NS banner and the column header.
            if emit_legend:
                emit(FORCE_ITERATIONS_BANNER, end="")
            emit(screen_header(lasym=resolution.lasym, lfreeb=True), end="")

        # An already-active multigrid stage evaluates with the free-boundary edge
        # row on its first pass, so its zero cache must have jmax=ns (not ns-1).
        rt_initial = rt_freeb if vacuum_active else rt_fixed
        carry = _initial_carry(
            _init_state,
            rt_initial,
            ijacob=_initial_ijacob,
            residuals=residual_continuation,
        )
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

        # VMEC2000's LMOVE_AXIS path is triggered by the *first force pass*, not
        # only by a bad Jacobian.  Run that pass explicitly before entering the
        # batched pre-/post-vacuum lanes.  If the shared solver body returns the
        # internal irst=4 transfer, rebuild both the plasma profiles and the
        # axis-dependent NESTOR filament/executables, then repeat iteration 1
        # once with ijacob=1.  The discarded triggering pass is not printed.
        carry = _call_lane(("fb_iter", use_fft), _iter_lane,
                           (carry, rt_initial), {"use_fft": use_fft},
                           notice=_lane_notice("free-iteration"))
        if (allow_initial_axis_reguess
                and int(carry.ier) == AXIS_REGUESS_FLAG
                and int(carry.ijacob) == 0 and ns >= 3):
            if verbose:
                emit(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS")
            rt, _init_state, _axis = reguess_initial_axis(rt, _init_state)
            if verbose:
                emit(improved_axis_block(
                    _axis[0], _axis[3],
                    raxis_cs=_axis[1] if resolution.lasym else None,
                    zaxis_cc=_axis[2] if resolution.lasym else None,
                ), end="")
            _initial_ijacob = 1

            _axis_r0, _axis_z0 = _vacuum_scalars(_init_state, rt)[2:4]
            basis, fused_vac, vacuum_lane = _vacuum_executables(
                resolution, mf=int(inp.mpol) + 1, nf=int(inp.ntor),
                signgs=int(rt.setup.signgs),
                wint=np.asarray(rt.trig.wint, dtype=float),
                modes=rt.modes, axis_r0=_axis_r0, axis_z0=_axis_z0,
                # The rebuild must keep the same synthesis kernel: dropping
                # this silently swapped an FFT-selected run onto the dense
                # vacuum lane for the rest of the stage.
                use_fft=use_fft,
            )
            zeros_edge = jnp.zeros(
                (basis.ntheta3, basis.nzeta), dtype=dtype)
            carried_edge = zeros_edge
            if (vacuum_continuation is not None
                    and vacuum_continuation.turned_on):
                if vacuum_continuation.rbsq is not None:
                    _, carried_geometry = _geometry(_init_state, rt)
                    r_edge = (
                        carried_geometry.R_even[-1] + carried_geometry.R_odd[-1]
                    )
                    pres_ns = _vacuum_scalars(_init_state, rt)[5]
                    rbsq = jnp.asarray(vacuum_continuation.rbsq, dtype=dtype)
                    carried_edge = jnp.where(
                        r_edge != 0.0,
                        rbsq * jnp.asarray(rt.setup.hs) / r_edge,
                        jnp.zeros_like(r_edge),
                    ) - pres_ns * jnp.asarray(
                        _presf_ns_scale(inp, ns), dtype=dtype)
                elif vacuum_continuation.bsqvac is not None:
                    carried_edge = jnp.asarray(
                        vacuum_continuation.bsqvac, dtype=dtype)
            rt_fixed = replace(
                rt, lfreeb=False, bsqvac_edge=zeros_edge,
                presf_ns_scale=jnp.asarray(0.0, dtype=dtype),
            )
            rt_freeb = replace(
                rt, lfreeb=True, jmax=ns, bsqvac_edge=carried_edge,
                presf_ns_scale=jnp.asarray(
                    _presf_ns_scale(inp, ns), dtype=dtype),
            )
            # The NESTOR matrix/vector were built for the discarded axis.
            fb.mode_matrix = None
            fb.mode_factor = None
            fb.mode_pivots = None
            fb.bvec_nonsing = None
            fb.potvac = None
            rt_initial = rt_freeb if vacuum_active else rt_fixed
            retry_xcdot = carry.xcdot
            carry = _initial_carry(
                _init_state, rt_initial, ijacob=_initial_ijacob,
                xcdot=retry_xcdot,
                residuals=(carry.fsqr, carry.fsqz, carry.fsql),
            )
            carry = _call_lane(("fb_iter", use_fft), _iter_lane,
                               (carry, rt_initial), {"use_fft": use_fft},
                               notice=_lane_notice("free-iteration"))
        _emit_due(final=False)

        int_dtype = carry.iteration.dtype
        max_passes = rt.max_iterations + 400
        for _ in range(max_passes):
            if bool(carry.done):
                break
            if fb.ivac == -1:
                # Every fixed-boundary iteration before vacuum activation
                # runs as ONE jitted while_loop; the lane exits precisely where
                # the per-pass driver would have entered the IVAC0 block below.
                carry = _call_lane(("fb_preact", use_fft), _preactivation_lane,
                                   (carry, rt_fixed), {"use_fft": use_fft},
                                   notice=_lane_notice("pre-vacuum loop"))
                _emit_due(final=False)
                if bool(carry.done):
                    break
                # Activation is now due: fall through to the host-stepped
                # turn-on pass (first vacuum call, soft restart, banners).
            elif fb.turned_on and int(carry.iteration) == 1:
                # funct3d.f wraps the entire IVAC0 block in ``iter2 > 1``.  At a
                # new radial grid, iteration 1 therefore uses the coarse stage's
                # carried bsqvac once; iteration 2 performs the first full update
                # against freshly selected resolution-specific NESTOR programs.
                carry = _call_lane(("fb_iter", use_fft), _iter_lane,
                                   (carry, rt_freeb), {"use_fft": use_fft},
                                   notice=_lane_notice("free-iteration"))
                _emit_due(final=False)
                continue
            elif (fb.turned_on and not fb.banner_pending
                  and fb.mode_matrix is not None
                  and fb.mode_factor is not None
                  and fb.mode_pivots is not None):
                # The whole post-turn-on steady state runs as ONE jitted
                # while_loop (vacuum cadence + damping + iteration, traced).
                zeros_sur = jnp.zeros_like(rt_freeb.bsqvac_edge)
                vc = _VacuumLoopCarry(
                    carry=carry,
                    rcon0=rt_freeb.rcon0, zcon0=rt_freeb.zcon0,
                    bsqvac=rt_freeb.bsqvac_edge,
                    rbsq=jnp.asarray(fb.rbsq, dtype=dtype),
                    mode_matrix=fb.mode_matrix, mode_factor=fb.mode_factor,
                    mode_pivots=fb.mode_pivots,
                    bvec_nonsing=fb.bvec_nonsing,
                    potvac=fb.potvac,
                    surface_fields=((zeros_sur,) * 4 if fb.surface_fields is None
                                    else fb.surface_fields),
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
                vc = _call_lane(("fb_vac", fused_vac.cache_key), vacuum_lane,
                                (vc, rt_freeb, external_field),
                                notice=_lane_notice("steady vacuum loop"))
                carry = vc.carry
                rt_freeb = replace(rt_freeb, rcon0=vc.rcon0, zcon0=vc.zcon0,
                                   bsqvac_edge=vc.bsqvac)
                fb.ivac = int(vc.ivac)
                fb.nvacskip = int(vc.nvacskip)
                fb.delbsq = float(vc.delbsq)
                fb.bsqvac = vc.bsqvac
                fb.rbsq = vc.rbsq
                fb.mode_matrix = vc.mode_matrix
                fb.mode_factor = vc.mode_factor
                fb.mode_pivots = vc.mode_pivots
                fb.bvec_nonsing = vc.bvec_nonsing
                fb.potvac = vc.potvac
                fb.surface_fields = vc.surface_fields
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

            # NESTOR never consumes a sign-changed state: the restored xstore is
            # the vacuum source and the update is forced full (_jacobian_ok).
            jacobian_good = bool(
                _jacobian_ok(carry.state, rt, use_fft=use_fft))
            vacuum_source = carry if jacobian_good else replace(
                carry, state=carry.xstore)

            # -- funct3d.f IVAC0 block (host) -----------------------------------
            if it > 1 and fsq_rz <= ACTIVATION_FSQ:
                fb.ivac += 1
            rt_use = rt_fixed
            turnon_evaluation_state = None
            if fb.ivac >= 0:
                # Damp the constraint baselines (funct3d: 0.9 per iteration).
                rt_fixed = replace(rt_fixed, rcon0=0.9 * rt_fixed.rcon0, zcon0=0.9 * rt_fixed.zcon0)
                rt_freeb = replace(rt_freeb, rcon0=0.9 * rt_freeb.rcon0, zcon0=0.9 * rt_freeb.zcon0)
                ivacskip = (it - iter1) % max(1, fb.nvacskip)
                if fb.ivac <= 2 or not jacobian_good:
                    ivacskip = 0
                if ivacskip == 0:
                    fb.nvacskip = max(fb.nvskip0, int(1.0 / max(1.0e-1, 1.0e11 * fsq_rz)))
                bsqvac = _vacuum_step(
                    carry=vacuum_source, rt=rt_freeb, fb=fb, basis=basis,
                    fused_vac=fused_vac, field=external_field,
                    ivacskip=ivacskip, emit=emit, verbose=verbose,
                )
                if fb.ivac >= 1 and not fb.turned_on:
                    # funct3d.f soft start (restart_iter, irst = 2) applied on
                    # the host: best state restored, velocity zeroed, delt*0.9,
                    # iter1 = iter2, ijacob += 1.  funct3d.f has already computed
                    # geometry at the pre-restart state, so preserve it for this
                    # pass's force evaluation (the momentum base remains xstore).
                    fb.turned_on = True
                    fb.banner_pending = True
                    turnon_evaluation_state = vacuum_source.state
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

            if turnon_evaluation_state is None:
                carry = _call_lane(("fb_iter", use_fft), _iter_lane,
                                   (carry, rt_use), {"use_fft": use_fft},
                                   notice=_lane_notice("free-iteration"))
            else:
                carry = _call_lane(
                    ("fb_turnon", use_fft), _turnon_iter_lane,
                    (carry, rt_use, turnon_evaluation_state),
                    {"use_fft": use_fft},
                    notice=_lane_notice("vacuum turn-on"))

            if fb.banner_pending:
                if verbose:
                    emit(vacuum_banner(it), end="")
                fb.banner_pending = False
                fb.ivac = max(fb.ivac, 2)  # eqsolve.f: ivac = ivac + 1 after banner
            _emit_due(final=False)

        _emit_due(final=True)
        ier = int(carry.ier)
        if ier == JAC75_FLAG and int(jacobian_retries) > 0:
            retry_step = min(0.5, 0.5 * float(rt.time_step0))
            if verbose:
                emit(
                    " JACOBIAN RECOVERY RETRY: RESTARTING BEST FINITE "
                    f"STATE WITH DELT = {retry_step:.6g}"
                )
            # Retire this stage's prefetch before recursing: the retry's
            # halved DELT is static meta, so none of the pending lanes match
            # its structures (the finally below re-joins harmlessly).
            if prefetch is not None:
                prefetch[1].set()
                prefetch[0].join()
            current_rt = rt_freeb if fb.turned_on else rt_fixed
            return _solve_free_boundary_stage(
                inp,
                mgrid_path=mgrid_path,
                external_field=external_field,
                resolution=resolution,
                ftol=ftol,
                max_iterations=max_iterations,
                verbose=verbose,
                emit=emit,
                error_on_no_convergence=error_on_no_convergence,
                initial_state=carry.xstore,
                vacuum_continuation=fb,
                time_step=retry_step,
                tcon0=tcon0,
                gamma=gamma,
                nstep=nstep,
                lconm1=lconm1,
                precon_type=precon_type,
                prec2d_threshold=prec2d_threshold,
                prec2d=prec2d,
                # The retry must keep the same synthesis kernel: dropping this
                # silently fell back to the dense transform exactly on the
                # high-mode recoveries where the FFT selection matters most.
                use_fft=use_fft,
                jacobian_retries=int(jacobian_retries) - 1,
                constraint_continuation=(
                    current_rt.rcon0, current_rt.zcon0
                ) if fb.turned_on else None,
                # NESTOR's matrix and axis-current filament depend on the
                # checkpoint geometry; never reuse the failed attempt's compiled
                # dynamic cache even at an equal radial resolution.
                reuse_vacuum_cache=False,
                allow_initial_axis_reguess=False,
                residual_continuation=(carry.fsqr, carry.fsqz, carry.fsql),
                prefetch_compile=prefetch_compile,
                prefetch_device=prefetch_device,
            )
        if ier == MORE_ITER_FLAG and not error_on_no_convergence:
            result = _result_from_carry(carry, rt_freeb if fb.turned_on else rt_fixed)
            return _FreeBoundaryStageResult(
                replace(result, converged=False, ier_flag=MORE_ITER_FLAG,
                        vacuum=_vacuum_output(fb, basis)), fb,
                carry.xstore, rt_freeb.rcon0 if fb.turned_on else rt_fixed.rcon0,
                rt_freeb.zcon0 if fb.turned_on else rt_fixed.zcon0)
        if ier == SUCCESSFUL_TERM_FLAG:
            return _FreeBoundaryStageResult(
                replace(
                    _result_from_carry(
                        carry, rt_freeb if fb.turned_on else rt_fixed),
                    vacuum=_vacuum_output(fb, basis),
                ), fb,
                carry.xstore, rt_freeb.rcon0 if fb.turned_on else rt_fixed.rcon0,
                rt_freeb.zcon0 if fb.turned_on else rt_fixed.zcon0)
        return _FreeBoundaryStageResult(
            replace(
                _finalize(carry, rt_freeb if fb.turned_on else rt_fixed),
                vacuum=_vacuum_output(fb, basis),
            ), fb,
            carry.xstore, rt_freeb.rcon0 if fb.turned_on else rt_fixed.rcon0,
            rt_freeb.zcon0 if fb.turned_on else rt_fixed.zcon0)
    finally:
        if prefetch is not None:
            prefetch[1].set()      # stop at the next lane boundary
            prefetch[0].join()




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
    initial_state: SpectralState | None = None,
    device: Any = AUTO,
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
) -> SolveResult:
    """Single-grid free-boundary solve (``eqsolve.f`` + ``funct3d.f`` IVAC0).

    ``initial_state`` hot-restarts at the same radial resolution.  Its entire
    plasma state, including the evolved edge, is preserved; use
    :func:`vmex.core.multigrid.interpolate_state` first when the resolution
    differs.  As in VMEC2000 reset-file starts, vacuum activation is repeated.

    ``device`` has the same explicit/default placement semantics as
    :func:`vmex.core.solver.solve`.  Use
    :func:`vmex.core.multigrid.solve_free_boundary_multigrid` for the complete
    ``NS_ARRAY`` ladder.  Time-step, constraint, print-cadence, m=1-constraint,
    optional 2D-preconditioner overrides, and bounded ``jacobian_retries``
    recovery mirror :func:`vmex.core.solver.solve`. The returned
    ``result.vacuum`` contains the final NESTOR potential modes and surface
    fields, without internal matrix caches.
    """
    if resolution is None:
        # The angular grid depends on the field table (VMEC2000's NZETA
        # divisibility rule), so the field must be loaded before the
        # resolution is minted; a supplied resolution is validated against
        # the field inside the stage instead.
        if external_field is None:
            external_field = _external_field_from_input(inp, mgrid_path)
        resolution = free_boundary_resolution(inp, external_field)
    target = _placement_device(device, resolution)
    external_field = _put_numeric_leaves(external_field, target)
    initial_state = _put_numeric_leaves(initial_state, target)
    with device_context(device, resolution):
        stage = _solve_free_boundary_stage(
            inp, mgrid_path=mgrid_path, external_field=external_field,
            resolution=resolution, ftol=ftol, max_iterations=max_iterations,
            verbose=verbose, emit=emit,
            error_on_no_convergence=error_on_no_convergence,
            initial_state=initial_state, vacuum_continuation=None,
            time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
            lconm1=lconm1,
            precon_type=precon_type, prec2d_threshold=prec2d_threshold,
            prec2d=prec2d,
            jacobian_retries=jacobian_retries,
            constraint_continuation=None, reuse_vacuum_cache=False,
            use_fft=_resolve_use_fft(use_fft, device, resolution),
        )
    return stage.result
