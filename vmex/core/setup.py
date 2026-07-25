"""Run setup: radial grids, 1D flux/profile arrays, boundary processing, and
the initial interior guess — everything a fixed-boundary solve needs before
the first iteration.

VMEC2000 counterparts
---------------------
- ``Sources/Initialization_Cleanup/profil1d.f`` — radial grids (``hs``,
  ``sqrts``, ``shalf``, ``sm/sp``), the 1D flux profiles ``phips/chips`` and
  ``phipf/chipf``, ``iotas/iotaf`` (``piota``), the enclosed-current profile
  ``icurv`` (``pcurr`` + ``CURTOR`` scaling) and the ``mass`` profile
  (``pmass`` in internal ``mu0*Pa`` units), plus ``lamscale``:
  :func:`radial_grids`, :func:`flux_profiles`.
- ``Sources/Initialization_Cleanup/magnetic_fluxes.f`` — ``torflux``/
  ``torflux_deriv`` from the ``APHI`` polynomial and ``polflux_deriv =
  piota(tf) * torflux_deriv``.
- ``Sources/Input_Output/readin.f`` — boundary-coefficient processing: the
  ``lasym`` ``delta`` rotation, conversion to the internal ``rbcc/rbss/...``
  blocks, the Jacobian-sign check ``lflip = (rtest*ztest < 0)`` with
  ``signgs = -1``, and the ``lconm1`` m = 1 constraint
  (``rbss = (rbss + zbcs)/2`` etc.): :func:`boundary_from_input`.
- ``Sources/Initialization_Cleanup/init_geometry.f90`` — ``flip_theta``
  (``theta -> pi - theta`` sign factors ``(-1)**m``).
- ``Sources/Initialization_Cleanup/profil3d.f`` — the interior guess: odd/even
  ``scalxc`` factors and the interpolation of boundary + axis into the volume
  (``rmn(js,m>0) = rmn_bdy * sqrts(js)**m``; ``rmn(js,m=0) = s*rmn_bdy +
  (1-s)*axis``): :func:`interior_guess`, :func:`run_setup`.
- ``Sources/Initialization_Cleanup/guess_axis.f`` — the axis re-guess grid
  search used after a first bad-Jacobian start: :func:`guess_axis`.

Representation conventions (parity-critical)
--------------------------------------------
All spectral outputs are in the signed-(m, n) helical packing of
:func:`vmex.core.fourier.mode_table`, in VMEC *internal* normalization
(divided by ``mscale(m)*nscale(|n|)``) and — for the boundary and the initial
state — in the m = 1 *constrained* basis evolved by the solver
(``residue.f90`` / ``readin.f`` ``lconm1``).  Geometry synthesis
(:func:`vmex.core.geometry.real_space_geometry`) consumes the *physical*
basis; :func:`geometry_state` performs the conversion
(:func:`vmex.core.residuals.m1_constrained_to_physical` + the 3D lambda
axis closure).

Boundary/axis parsing and the ``guess_axis`` grid search are one-time host
NumPy code (data-dependent argmax); every produced array is a ``jnp`` array
and :func:`interior_guess` — the state-producing path — is jit-compatible.
Ported from the parity-proven legacy implementation
(``vmex.init_guess``, ``vmex.boundary``, ``vmex.energy``,
``vmex.solvers.fixed_boundary.profiles``); equivalence is enforced in
``tests/test_setup_ab.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import jax.numpy as jnp

from . import profiles as prof
from .fourier import ModeTable, Resolution, TrigTables, mode_table, trig_tables
from .geometry import RealSpaceGeometry, apply_lambda_axis_closure, real_space_geometry
from .input import VmecInput
from .residuals import m1_constrained_to_physical, m1_physical_to_constrained
from .transforms import (
    odd_m_sqrt_s_scaling, physical_to_internal_scale,
    register_pytree_dataclass as _register,
)

__all__ = [
    "RadialGrids",
    "ProcessedBoundary",
    "RunSetup",
    "radial_grids",
    "boundary_from_input",
    "flux_profiles",
    "interior_guess",
    "guess_axis",
    "geometry_state",
    "run_setup",
]

Array = Any

#: guess_axis.f ``limpts``: grid points per direction in the axis scan.
GUESS_AXIS_GRID_POINTS = 61


# ---------------------------------------------------------------------------
# Radial grids (profil1d.f)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadialGrids:
    """VMEC radial full/half meshes (VMEC2000: ``profil1d.f``).

    With ``hs = 1/(ns-1)`` and 1-based Fortran index ``i``:

    - ``s_full(i) = hs*(i-1)`` — full mesh, ``s in [0, 1]``;
    - ``s_half(i) = hs*|i-1.5|`` — half mesh; the axis slot repeats the first
      interior value (``s_half(1) = s_half(2) = hs/2``, matching
      :func:`vmex.core.geometry.sqrt_s_half_mesh`);
    - ``sqrts = sqrt(s_full)`` with ``sqrts(ns) = 1`` exactly (round-off
      guard); ``shalf = sqrt(s_half)``;
    - ``sm(i) = shalf(i)/sqrts(i)``, ``sp(i) = shalf(i+1)/sqrts(i)`` with
      ``sm(1) = 0``, ``sp(1) = sm(2)`` and ``shalf(ns+1) = 1`` — the odd-m
      half-mesh interpolation weights used by ``lamcal``/``precondn``;
    - ``hs`` — the uniform radial step (0-d array).
    """

    s_full: Array
    s_half: Array
    sqrts: Array
    shalf: Array
    sm: Array
    sp: Array
    hs: Array


@dataclass(frozen=True)
class ProcessedBoundary:
    """Boundary coefficients after the ``readin.f`` processing chain.

    ``R_cos/R_sin/Z_cos/Z_sin`` are signed-(m, n) helical coefficients aligned
    with :func:`vmex.core.fourier.mode_table`, in internal normalization
    (divided by ``mscale*nscale``), theta-flipped when ``lflip`` and in the
    m = 1 constrained basis when ``lconm1`` (``readin.f``:
    ``rbss = (rbss + zbcs)/2`` etc.).  ``r00`` is the physical ``RBC(0, 0)``
    (VMEC ``r00 = rmn_bdy(0,0,rcc)``); ``signgs`` is always ``-1``
    (``readin.f``); ``lflip`` records the theta flip decision
    ``rtest*ztest < 0``.
    """

    R_cos: Array
    R_sin: Array
    Z_cos: Array
    Z_sin: Array
    r00: Array
    signgs: int
    lflip: bool


@dataclass(frozen=True)
class RunSetup:
    """Everything a fixed-boundary solve needs before iterating.

    Radial grids (``profil1d.f``; see :class:`RadialGrids` for definitions):
    ``s_full, s_half, sqrts, shalf, sm, sp, hs`` plus ``scalxc`` — the odd-m
    ``1/sqrt(s)`` factors, shape ``(ns, mpol)`` (``profil3d.f``; equals
    :func:`vmex.core.transforms.odd_m_sqrt_s_scaling`).

    1D profiles (``profil1d.f``, half mesh unless noted; index 0 = axis slot
    is zeroed exactly as in Fortran):

    - ``phips/chips``: ``torflux_edge * torflux_deriv/polflux_deriv(s_half)``
      with ``torflux_edge = signgs*phiedge/(2*pi)`` (normalized by
      ``torflux(1)``); ``chips`` (and ``iotas``) are negated when ``lflip``.
    - ``iotas``: ``piota(min(torflux(s_half), 1))``; ``iotaf/phipf/chipf``:
      the full-mesh companions (*not* flipped — profil1d.f quirk).
    - ``icurv``: ``Itor * pcurr(tf)`` with ``Itor = signgs*mu0*curtor /
      (2*pi*pcurr(1))`` (zero when ``|pcurr(1)| <= eps*|curtor|``).  Computed
      unconditionally; consumed by ``add_fluxes`` only when ``ncurr = 1``.
    - ``mass``: ``mu0*pres_scale*pmass(tf) * (|phips|*r00)**gamma`` with the
      ``spres_ped`` clamp (``pmass(spres_ped)`` for ``s_half > spres_ped``).
    - ``lamscale = sqrt(hs * sum(phips(2:ns)**2))``.

    Boundary/axis: ``boundary_R_cos/...`` per :class:`ProcessedBoundary`;
    ``raxis_c/raxis_s/zaxis_c/zaxis_s`` are the *physical* axis coefficients
    actually used (input arrays, or the :func:`guess_axis` result when the
    input axis was all-zero and ``infer_axis_if_missing``).

    Initial state (``profil3d.f``): ``R_cos/R_sin/Z_cos/Z_sin`` shape
    ``(ns, mnmax)`` — internal-normalized, m = 1-*constrained* spectral
    coefficients (the solver's evolution basis) — and ``lambda_cos/
    lambda_sin`` (zero; ``profil1d.f`` zeroes the lambda block of ``xc``).
    Use :func:`geometry_state` to obtain the physical-basis inputs of
    :func:`vmex.core.geometry.real_space_geometry`.

    Static metadata: ``signgs`` (always -1), ``lflip``, ``lasym``, ``lthreed``,
    ``lconm1`` and ``ncurr`` (0: prescribed iota, 1: prescribed current).
    """

    # -- radial grids (profil1d.f / profil3d.f) --
    s_full: Array
    s_half: Array
    sqrts: Array
    shalf: Array
    sm: Array
    sp: Array
    hs: Array
    scalxc: Array

    # -- 1D profiles (profil1d.f) --
    phips: Array
    chips: Array
    iotas: Array
    icurv: Array
    mass: Array
    phipf: Array
    chipf: Array
    iotaf: Array
    lamscale: Array

    # -- boundary (readin.f) and axis --
    boundary_R_cos: Array
    boundary_R_sin: Array
    boundary_Z_cos: Array
    boundary_Z_sin: Array
    raxis_c: Array
    raxis_s: Array
    zaxis_c: Array
    zaxis_s: Array

    # -- initial spectral state (profil3d.f), m=1-constrained internal --
    R_cos: Array
    R_sin: Array
    Z_cos: Array
    Z_sin: Array
    lambda_cos: Array
    lambda_sin: Array

    # -- static metadata --
    signgs: int
    lflip: bool
    lasym: bool
    lthreed: bool
    lconm1: bool
    ncurr: int


_register(RadialGrids)
_register(ProcessedBoundary, meta=("signgs", "lflip"))
_register(RunSetup, meta=("signgs", "lflip", "lasym", "lthreed", "lconm1", "ncurr"))


def radial_grids(ns: int, *, dtype=jnp.float64) -> RadialGrids:
    """Build the VMEC radial meshes (VMEC2000: ``profil1d.f``).

    ``s_full(i) = hs*(i-1)``, ``s_half(i) = hs*|i-1.5|`` (1-based ``i``),
    ``sqrts/shalf`` their square roots (``sqrts(ns) = 1`` exactly), and the
    ``sm/sp`` odd-m interpolation weights (see :class:`RadialGrids`).
    """
    ns = int(ns)
    if ns < 2:
        one = jnp.ones((max(ns, 1),), dtype=dtype)
        zero = jnp.zeros_like(one)
        return RadialGrids(s_full=zero, s_half=zero, sqrts=one, shalf=one,
                           sm=zero, sp=one, hs=jnp.asarray(1.0, dtype=dtype))
    s_full = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    hs = s_full[1] - s_full[0]
    idx = jnp.arange(ns, dtype=dtype)
    s_half = hs * jnp.abs(idx - 0.5)
    sqrts = jnp.sqrt(s_full).at[-1].set(jnp.asarray(1.0, dtype=dtype))
    shalf = jnp.sqrt(s_half)
    # profil1d.f: sm(i) = shalf(i)/sqrts(i); sp(i) = shalf(i+1)/sqrts(i) with
    # shalf(ns+1) = 1; sm(1) = 0 and sp(1) = sm(2) (axis conventions).
    shalf_up = jnp.concatenate([shalf[1:], jnp.ones((1,), dtype=dtype)])
    sqrts_safe = jnp.where(sqrts > 0.0, sqrts, 1.0)
    sm = jnp.where(idx > 0, shalf / sqrts_safe, 0.0)
    sp = shalf_up / sqrts_safe
    sp = sp.at[0].set(sm[1])
    return RadialGrids(s_full=s_full, s_half=s_half, sqrts=sqrts, shalf=shalf,
                       sm=sm, sp=sp, hs=hs)


# ---------------------------------------------------------------------------
# Boundary processing (readin.f + init_geometry.f90 flip_theta)
# ---------------------------------------------------------------------------


def _lasym_delta_rotation(rbc, rbs, zbc, zbs, *, mpol: int, ntor: int):
    """``readin.f`` lasym normalization: rotate theta so ``RBS(0,1) = ZBC(0,1)``.

    ``delta = atan((rbs(0,1) - zbc(0,1)) / (|rbc(0,1)| + |zbs(0,1)|))``; every
    (n, m) pair is rotated by ``m*delta`` (cos/sin mixing).  A denominator of
    zero (degenerate m=1 content) keeps the input unrotated.
    """
    if mpol < 2:
        return rbc, rbs, zbc, zbs
    denom = abs(rbc[ntor, 1]) + abs(zbs[ntor, 1])
    if denom == 0.0:
        return rbc, rbs, zbc, zbs
    delta = float(np.arctan((rbs[ntor, 1] - zbc[ntor, 1]) / denom))
    if delta == 0.0:
        return rbc, rbs, zbc, zbs
    m = np.arange(mpol, dtype=float)[None, :]
    cos_md, sin_md = np.cos(m * delta), np.sin(m * delta)
    rbc_new = rbc * cos_md + rbs * sin_md
    rbs_new = rbs * cos_md - rbc * sin_md
    zbc_new = zbc * cos_md + zbs * sin_md
    zbs_new = zbs * cos_md - zbc * sin_md
    return rbc_new, rbs_new, zbc_new, zbs_new


def _internal_blocks_from_input(inp: VmecInput) -> tuple[dict[str, np.ndarray], bool]:
    """``readin.f``: dense INDATA arrays -> internal ``rbcc/rbss/...`` blocks.

    Implements, in order: the lasym ``delta`` rotation, the accumulation

        ``rbcc(|n|,m) += rbc(n,m)``;  ``zbsc(|n|,m) += zbs(n,m)`` (m>0);
        3D: ``rbss(|n|,m) += sgn(n)*rbc`` (m>0), ``zbcs(|n|,m) -= sgn(n)*zbs``;
        lasym: ``rbsc += rbs`` (m>0), ``zbcc += zbc`` and (3D)
        ``rbcs -= sgn(n)*rbs``, ``zbss += sgn(n)*zbc`` (m>0)

    (with the free-boundary ``mfilter_fbdy/nfilter_fbdy`` skips), the theta
    flip when ``rtest*ztest < 0`` (``flip_theta``: sign ``(-1)**m`` factors,
    ``init_geometry.f90``), and the ``lconm1`` m = 1 constraint.  Returns the
    block dict and ``lflip``.
    """
    mpol, ntor = int(inp.mpol), int(inp.ntor)
    lthreed, lasym = ntor > 0, bool(inp.lasym)
    rbc = np.asarray(inp.rbc, dtype=float).copy()
    rbs = np.asarray(inp.rbs, dtype=float).copy()
    zbc = np.asarray(inp.zbc, dtype=float).copy()
    zbs = np.asarray(inp.zbs, dtype=float).copy()
    if lasym:
        rbc, rbs, zbc, zbs = _lasym_delta_rotation(rbc, rbs, zbc, zbs,
                                                   mpol=mpol, ntor=ntor)

    shape = (ntor + 1, mpol)
    blocks = {name: np.zeros(shape) for name in
              ("rbcc", "rbss", "rbcs", "rbsc", "zbcc", "zbss", "zbcs", "zbsc")}
    for m in range(mpol):
        if inp.lfreeb and 1 < inp.mfilter_fbdy < m:
            continue
        for n in range(-ntor, ntor + 1):
            if inp.lfreeb and 0 < inp.nfilter_fbdy < abs(n):
                continue
            ni, isgn = abs(n), (0 if n == 0 else (1 if n > 0 else -1))
            j = n + ntor
            blocks["rbcc"][ni, m] += rbc[j, m]
            if m > 0:
                blocks["zbsc"][ni, m] += zbs[j, m]
            if lthreed:
                if m > 0:
                    blocks["rbss"][ni, m] += isgn * rbc[j, m]
                blocks["zbcs"][ni, m] -= isgn * zbs[j, m]
            if lasym:
                if m > 0:
                    blocks["rbsc"][ni, m] += rbs[j, m]
                blocks["zbcc"][ni, m] += zbc[j, m]
                if lthreed:
                    blocks["rbcs"][ni, m] -= isgn * rbs[j, m]
                    if m > 0:
                        blocks["zbss"][ni, m] += isgn * zbc[j, m]

    # readin.f: rtest/ztest Jacobian-sign check on the m=1 row; signgs = -1.
    lflip = False
    if mpol > 1:
        rtest = float(np.sum(blocks["rbcc"][:, 1]))
        ztest = float(np.sum(blocks["zbsc"][:, 1]))
        lflip = (rtest * ztest) < 0.0
    if lflip:
        # flip_theta (init_geometry.f90): theta -> pi - theta.
        signs = (-1.0) ** np.arange(mpol, dtype=float)[None, :]  # (-1)**m
        for name, fac in (("rbcc", 1), ("zbsc", -1), ("rbss", -1), ("zbcs", 1),
                          ("rbsc", -1), ("zbcc", 1), ("rbcs", 1), ("zbss", -1)):
            blocks[name][:, 1:] = (fac * signs * blocks[name])[:, 1:]
    return blocks, lflip


def _helical_from_internal_blocks(blocks: dict[str, np.ndarray], modes: ModeTable):
    """Internal ``rbcc/...`` blocks -> signed-(m, n) helical coefficients.

    Inverse of the ``readin.f`` accumulation on the ``mode_table`` packing
    (legacy ``vmex.boundary._boundary_helical_from_internal``): ``n = 0``
    rows map directly; for ``n != 0``, ``cos(mu)cos(nv)`` etc. combine as
    ``R_cos(m, +/-n) = (rbcc +/- rbss)/2`` and the m = 0 modes (stored only
    for ``n >= 0``) carry the whole ``rbcc/zbcs`` content.
    """
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    K = m_arr.size
    out = {key: np.zeros((K,)) for key in ("R_cos", "R_sin", "Z_cos", "Z_sin")}
    b = blocks
    for k, (m, n) in enumerate(zip(m_arr, n_arr)):
        ni = abs(n)
        if m == 0 and n != 0:
            isgn = 1 if n > 0 else -1
            out["R_cos"][k] = b["rbcc"][ni, m]
            out["R_sin"][k] = -isgn * b["rbcs"][ni, m]
            out["Z_cos"][k] = b["zbcc"][ni, m]
            out["Z_sin"][k] = -isgn * b["zbcs"][ni, m]
        elif n == 0:
            out["R_cos"][k] = b["rbcc"][ni, m]
            out["R_sin"][k] = b["rbsc"][ni, m]
            out["Z_cos"][k] = b["zbcc"][ni, m]
            out["Z_sin"][k] = b["zbsc"][ni, m]
        elif n > 0:
            out["R_cos"][k] = 0.5 * (b["rbcc"][ni, m] + b["rbss"][ni, m])
            out["R_sin"][k] = 0.5 * (b["rbsc"][ni, m] - b["rbcs"][ni, m])
            out["Z_cos"][k] = 0.5 * (b["zbcc"][ni, m] + b["zbss"][ni, m])
            out["Z_sin"][k] = 0.5 * (b["zbsc"][ni, m] - b["zbcs"][ni, m])
        else:
            out["R_cos"][k] = 0.5 * (b["rbcc"][ni, m] - b["rbss"][ni, m])
            out["R_sin"][k] = 0.5 * (b["rbsc"][ni, m] + b["rbcs"][ni, m])
            out["Z_cos"][k] = 0.5 * (b["zbcc"][ni, m] - b["zbss"][ni, m])
            out["Z_sin"][k] = 0.5 * (b["zbsc"][ni, m] + b["zbcs"][ni, m])
    return out


def boundary_from_input(
    inp: VmecInput,
    *,
    modes: ModeTable,
    trig: TrigTables,
    lconm1: bool = True,
) -> ProcessedBoundary:
    """Process INDATA boundary coefficients into the solver representation.

    VMEC2000: ``readin.f`` — lasym ``delta`` rotation, internal-block
    accumulation, theta flip / ``signgs = -1`` determination, the ``lconm1``
    m = 1 constraint, and the ``mscale*nscale`` internal normalization
    applied by ``profil3d.f`` (``t1 = 1/(mscale(m)*nscale(n))``).

    Host NumPy (one-time parsing); all outputs are ``jnp`` arrays.
    """
    blocks, lflip = _internal_blocks_from_input(inp)
    r00 = float(blocks["rbcc"][0, 0])
    helical = _helical_from_internal_blocks(blocks, modes)
    scale = physical_to_internal_scale(modes, trig)
    R_cos = jnp.asarray(helical["R_cos"] * scale)
    R_sin = jnp.asarray(helical["R_sin"] * scale)
    Z_cos = jnp.asarray(helical["Z_cos"] * scale)
    Z_sin = jnp.asarray(helical["Z_sin"] * scale)
    lthreed = int(inp.ntor) > 0
    # readin.f lconm1 conversion (same rotation as residue.f90; the m=1 modes
    # share one mscale*nscale factor, so it commutes with the scaling above).
    R_cos2, Z_sin2, R_sin2, Z_cos2 = m1_physical_to_constrained(
        R_cos[None, :], Z_sin[None, :], R_sin[None, :], Z_cos[None, :],
        modes=modes, lthreed=lthreed, lasym=bool(inp.lasym), lconm1=bool(lconm1),
    )
    return ProcessedBoundary(
        R_cos=R_cos2[0], R_sin=R_sin2[0], Z_cos=Z_cos2[0], Z_sin=Z_sin2[0],
        r00=jnp.asarray(r00), signgs=-1, lflip=bool(lflip),
    )


# ---------------------------------------------------------------------------
# 1D flux and profile arrays (profil1d.f, magnetic_fluxes.f)
# ---------------------------------------------------------------------------


def _torflux_functions(aphi: np.ndarray):
    """Return ``(torflux, torflux_deriv)`` from the ``APHI`` polynomial.

    VMEC2000: ``magnetic_fluxes.f`` — ``torflux_deriv(x) = sum_i i*aphi(i)*
    x**(i-1)`` and ``torflux(x)`` its 101-point trapezoid integral on
    ``[0, x]``.  The default ``aphi = [1, 0, ...]`` short-circuits to the
    identity map (``profil1d.f`` behavior, exact in floating point).
    """
    aphi = np.asarray(aphi, dtype=float).ravel()
    if aphi.size == 0:
        aphi = np.asarray([1.0])
    if aphi[0] == 1.0 and not np.any(aphi[1:]):
        return (lambda x: jnp.asarray(x)), (lambda x: jnp.ones_like(jnp.asarray(x)))
    dcoef = aphi * np.arange(1, aphi.size + 1, dtype=float)

    def torflux_deriv(x):
        x = jnp.asarray(x)
        y = jnp.zeros_like(x)
        for c in dcoef[::-1]:
            y = y * x + c
        return y

    def torflux(x):
        x = jnp.asarray(x)
        h = 1e-2 * x
        xi = h[..., None] * jnp.arange(101.0)
        vals = torflux_deriv(xi)
        return h * (jnp.sum(vals, axis=-1) - 0.5 * (vals[..., 0] + vals[..., -1]))

    return torflux, torflux_deriv


def flux_profiles(
    inp: VmecInput,
    grids: RadialGrids,
    *,
    r00: Array,
    signgs: int = -1,
    lflip: bool = False,
) -> dict[str, Array]:
    """Evaluate the profil1d.f 1D profile arrays on the radial meshes.

    VMEC2000: ``profil1d.f`` — with ``torflux_edge = signgs*phiedge/(2*pi)``
    (divided by ``torflux(1)`` when nonzero) and ``tf = min(torflux(s), 1)``:

    - half mesh (index 0 zeroed): ``phips = torflux_edge*torflux_deriv``,
      ``chips = torflux_edge*piota(tf)*torflux_deriv`` (``polflux_deriv``),
      ``iotas = piota(tf)``, ``icurv = Itor*pcurr(tf)`` with
      ``Itor = signgs*mu0*curtor/(2*pi*pcurr(1))`` (0 when ``|pcurr(1)| <=
      eps*|curtor|``), ``mass = mu0*pres_scale*pmass * (|phips|*r00)**gamma``
      with the ``spres_ped`` clamp;
    - full mesh: ``phipf/chipf/iotaf`` analogously;
    - ``lamscale = sqrt(hs*sum(phips(2:)**2))``;
    - ``lflip`` negates ``iotas`` and ``chips`` (only — profil1d.f leaves the
      full-mesh arrays unflipped).

    ``lrfp`` (RFP mode) is not supported.  Returns a dict of ``jnp`` arrays
    keyed ``phips, chips, iotas, icurv, mass, phipf, chipf, iotaf, lamscale``.
    """
    dtype = grids.s_full.dtype
    ns = int(grids.s_full.shape[0])
    torflux, torflux_deriv = _torflux_functions(inp.aphi)

    two_pi = 2.0 * np.pi
    torflux_edge = jnp.asarray(signgs * inp.phiedge / two_pi, dtype=dtype)
    tf1 = torflux(jnp.asarray(1.0, dtype=dtype))
    torflux_edge = jnp.where(tf1 != 0.0, torflux_edge / jnp.where(tf1 != 0.0, tf1, 1.0),
                             torflux_edge)

    def iota_at(x):
        return prof.iota(inp.piota_type, inp.ai, inp.ai_aux_s, inp.ai_aux_f, x,
                         bloat=inp.bloat)

    def current_at(x):
        return prof.current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, x,
                            bloat=inp.bloat)

    def pmass_pa_at(x):
        # spres_ped clamp applied on the half-mesh coordinate below, not here.
        return prof.pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, x,
                             pres_scale=inp.pres_scale, bloat=inp.bloat, spres_ped=1.0)

    not_axis = jnp.arange(ns) != 0

    # -- half mesh (profil1d.f DO i = 2, ns; index 1 slots stay 0) --
    tf_half = jnp.minimum(torflux(grids.s_half), 1.0)
    td_half = torflux_deriv(grids.s_half)
    iota_half = iota_at(tf_half)
    phips = jnp.where(not_axis, torflux_edge * td_half, 0.0)
    chips = jnp.where(not_axis, torflux_edge * (iota_half * td_half), 0.0)
    iotas = jnp.where(not_axis, iota_half, 0.0)

    # lamscale (profil1d.f): normalizes lambda for the Hessian scaling.
    lamscale = jnp.sqrt(grids.hs * jnp.sum(phips[1:] ** 2))

    if bool(lflip):
        iotas, chips = -iotas, -chips

    # -- full mesh --
    tf_full = jnp.minimum(torflux(grids.s_full), 1.0)
    td_full = torflux_deriv(grids.s_full)
    iotaf = iota_at(tf_full)
    phipf = torflux_edge * td_full
    chipf = torflux_edge * (iotaf * td_full)

    # -- current profile: scale to match CURTOR (profil1d.f Itor) --
    pedge = current_at(jnp.asarray(1.0, dtype=dtype))
    currv = prof.MU0 * jnp.asarray(inp.curtor, dtype=dtype)
    eps = float(np.finfo(np.float64).eps)
    valid = jnp.abs(pedge) > jnp.abs(eps * inp.curtor)
    itor = jnp.where(
        valid, signgs * currv / (two_pi * jnp.where(valid, pedge, 1.0)), 0.0
    )
    icurv = jnp.where(not_axis, itor * current_at(tf_half), 0.0)

    # -- mass profile in internal units (profil1d.f; mu0 conversion) --
    spres_ped = abs(float(inp.spres_ped))
    p_half_pa = pmass_pa_at(tf_half)
    if spres_ped < 1.0:
        p_ped_pa = pmass_pa_at(jnp.asarray(spres_ped, dtype=dtype))
        p_half_pa = jnp.where(grids.s_half > spres_ped, p_ped_pa, p_half_pa)
    # NaN-safe under reverse-mode AD (core.implicit traces this function in
    # phiedge/r00): vpnorm = 0 at the axis slot would make d(vpnorm**gamma)
    # produce 0 * inf = nan inside the discarded jnp.where branch.
    vpnorm = jnp.abs(phips) * jnp.asarray(r00, dtype=dtype)
    if float(inp.gamma) == 0.0:
        gamma_factor = jnp.ones_like(vpnorm)
    else:
        safe_vpnorm = jnp.where(not_axis, vpnorm, 1.0)
        gamma_factor = jnp.where(not_axis, safe_vpnorm ** inp.gamma, 0.0)
    mass = jnp.where(not_axis, prof.MU0 * p_half_pa * gamma_factor, 0.0)

    return dict(phips=phips, chips=chips, iotas=iotas, icurv=icurv, mass=mass,
                phipf=phipf, chipf=chipf, iotaf=iotaf, lamscale=lamscale)


# ---------------------------------------------------------------------------
# Interior guess (profil3d.f)
# ---------------------------------------------------------------------------


def interior_guess(
    *,
    boundary_R_cos: Array,
    boundary_R_sin: Array,
    boundary_Z_cos: Array,
    boundary_Z_sin: Array,
    raxis_c: Array,
    raxis_s: Array,
    zaxis_c: Array,
    zaxis_s: Array,
    modes: ModeTable,
    trig: TrigTables,
    s: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """profil3d.f interior interpolation of boundary + axis into the volume.

    VMEC2000: ``profil3d.f`` (``lreset``) — starting from ``xc = 0``:

    - ``m > 0``:  ``coeff(js) = coeff_bdy * sqrts(js)**m``  (``facj``);
    - ``m = 0``:  ``coeff(js) = s(js)*coeff_bdy + (1 - s(js))*axis`` where the
      axis coefficients enter internally scaled (``rax1*t1``); in the signed
      helical packing the net mapping is ``R_cos <- raxis_c/nscale``,
      ``R_sin <- raxis_s/nscale``, ``Z_cos <- zaxis_c/nscale``,
      ``Z_sin <- zaxis_s/nscale`` (the two profil3d.f minus signs — the
      internal ``zcs/rcs`` storage and the ``sin(m*u - n*v)`` helical basis at
      ``m = 0`` — cancel);
    - lambda: zero (``profil1d.f`` zeroes the lambda block of ``xc``).

    Boundary coefficients must already be processed by
    :func:`boundary_from_input` (internal-scaled, flipped, m = 1-constrained);
    the returned state is in the same (evolution) basis.  Pure JAX and
    jit-compatible: ``modes``/``trig`` are static tables, ``s`` has a static
    shape.

    Returns ``(R_cos, R_sin, Z_cos, Z_sin, lambda_cos, lambda_sin)``,
    each of shape ``(ns, mnmax)``.
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    dtype = s.dtype
    m = np.asarray(modes.m, dtype=int)
    n_axis = int(np.asarray(modes.n).max(initial=0)) + 1  # = ntor + 1
    # mode_table ordering: the first ntor+1 modes are (m=0, n=0..ntor).
    assert np.all(m[:n_axis] == 0) and not np.any(m[n_axis:] == 0)

    # profil3d.f: psqrts(:, ns) = 1 (avoid round-off at the edge).
    rho = jnp.sqrt(jnp.maximum(s, 0.0))
    if ns >= 1:
        rho = rho.at[-1].set(jnp.asarray(1.0, dtype=dtype))
    m_j = jnp.asarray(m)[None, :]
    facj = jnp.where(m_j > 0, rho[:, None] ** m_j, jnp.ones((ns, m.size), dtype=dtype))

    R_cos = facj * jnp.asarray(boundary_R_cos, dtype=dtype)[None, :]
    R_sin = facj * jnp.asarray(boundary_R_sin, dtype=dtype)[None, :]
    Z_cos = facj * jnp.asarray(boundary_Z_cos, dtype=dtype)[None, :]
    Z_sin = facj * jnp.asarray(boundary_Z_sin, dtype=dtype)[None, :]

    # m = 0 axis blend, internal axis scale t1 = 1/(mscale(0)*nscale(n)).
    axis_scale = jnp.asarray(1.0 / np.asarray(trig.nscale)[:n_axis], dtype=dtype)
    blend = s[:, None]
    sm0 = 1.0 - blend

    def blend_m0(full, boundary, axis):
        row = blend * jnp.asarray(boundary, dtype=dtype)[None, :n_axis] \
            + sm0 * (jnp.asarray(axis, dtype=dtype) * axis_scale)[None, :]
        return full.at[:, :n_axis].set(row)

    R_cos = blend_m0(R_cos, boundary_R_cos, raxis_c)
    R_sin = blend_m0(R_sin, boundary_R_sin, raxis_s)
    Z_cos = blend_m0(Z_cos, boundary_Z_cos, zaxis_c)
    Z_sin = blend_m0(Z_sin, boundary_Z_sin, zaxis_s)

    lambda_cos = jnp.zeros((ns, m.size), dtype=dtype)
    lambda_sin = jnp.zeros((ns, m.size), dtype=dtype)
    return R_cos, R_sin, Z_cos, Z_sin, lambda_cos, lambda_sin


def _axis_inference_state(
    boundary: ProcessedBoundary, *, modes: ModeTable, s: Array
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Pre-blend scaled state used to seed :func:`guess_axis` (no axis yet).

    Legacy ``vmex.init_guess`` convention (the parity-proven inference
    lane): ``m > 0`` modes scale like ``sqrt(s)**m`` (profil3d.f ``facj``),
    the ``m = 0`` cos rows stay at the boundary value and the ``m = 0`` sin
    rows scale linearly in ``s`` — i.e. the profil3d.f interpolation before
    any axis information is blended in.
    """
    s = jnp.asarray(s)
    dtype = s.dtype
    m_j = jnp.asarray(np.asarray(modes.m, dtype=int))[None, :]
    rho = jnp.sqrt(jnp.maximum(s, 0.0))
    rho_m = rho[:, None] ** m_j
    fac_cos = jnp.where(m_j > 0, rho_m, jnp.ones_like(rho_m))
    fac_sin = jnp.where(m_j > 0, rho_m, s[:, None] * jnp.ones_like(rho_m))
    zeros = jnp.zeros((int(s.shape[0]), int(m_j.shape[1])), dtype=dtype)
    return (
        fac_cos * jnp.asarray(boundary.R_cos, dtype=dtype)[None, :],
        fac_sin * jnp.asarray(boundary.R_sin, dtype=dtype)[None, :],
        fac_sin * jnp.asarray(boundary.Z_cos, dtype=dtype)[None, :],
        fac_sin * jnp.asarray(boundary.Z_sin, dtype=dtype)[None, :],
        zeros,
        zeros,
    )


# ---------------------------------------------------------------------------
# Axis re-guess (guess_axis.f)
# ---------------------------------------------------------------------------


def guess_axis(
    geometry: RealSpaceGeometry,
    *,
    s: Array,
    trig: TrigTables,
    signgs: int = -1,
    grid_points: int = GUESS_AXIS_GRID_POINTS,
) -> tuple[Array, Array, Array, Array]:
    """Re-guess the magnetic axis after a bad-Jacobian start (``guess_axis.f``).

    VMEC2000: ``Sources/Initialization_Cleanup/guess_axis.f`` — in each zeta
    plane, scan a ``limpts x limpts`` (R, Z) grid over the LCFS bounding box
    for the axis position maximizing the *minimum* over theta of the Jacobian
    proxy

        ``tau = signgs * (ru12*(zs + z_axis') - zu12*(rs + r_axis'))``

    built from the boundary surface (``js = ns``) and the mid surface
    (``js = ns12 = (ns+1)/2``) with ``rs = (r1b - r12)/ds + r_axis``; the
    symmetric case mirrors ``theta in (pi, 2*pi)`` from the reduced grid and
    scans only half the zeta planes, and up-down-symmetric planes fix
    ``z = 0``.  The winning per-plane positions are Fourier-projected with the
    ``nscale``-normalized zeta tables (halving ``n = 0`` and ``n = nzeta/2``).

    ``geometry`` supplies the ``totzsps`` channels (``r1/z1`` even/odd with the
    internal odd representation, and the theta derivatives for ``ru0/zu0``) —
    exactly the inputs of the Fortran routine.  Host NumPy code (data-dependent
    grid search; runs once after a failed start), ported verbatim from the
    parity-proven ``vmex.init_guess._recompute_axis_from_state_vmec``.

    Returns physical axis coefficient arrays
    ``(raxis_c, raxis_s, zaxis_c, zaxis_s)``, each of length ``ntor + 1``.
    """
    lasym = bool(trig.lasym)
    ntheta1, ntheta2, ntheta3 = int(trig.ntheta1), int(trig.ntheta2), int(trig.ntheta3)
    s_np = np.asarray(s, dtype=float)
    ns = int(s_np.shape[0])
    if ns < 2:
        raise ValueError("guess_axis requires ns >= 2")
    r1_even = np.asarray(geometry.R_even, dtype=float)
    r1_odd = np.asarray(geometry.R_odd, dtype=float)
    z1_even = np.asarray(geometry.Z_even, dtype=float)
    z1_odd = np.asarray(geometry.Z_odd, dtype=float)
    nzeta = int(r1_even.shape[2])
    sqrts = np.sqrt(np.maximum(s_np, 0.0))
    hs = float(s_np[1] - s_np[0])
    ns12 = (ns + 1) // 2 - 1                # Fortran ns12 = (ns+1)/2, 0-based
    ds = float((ns - 1 - ns12) * hs)        # (ns - ns12)*hs in Fortran indices

    # ru0/zu0 = full-mesh theta derivatives (funct3d.f convention).
    sq3 = sqrts[:, None, None]
    ru0 = np.asarray(geometry.dR_dtheta_even) + sq3 * np.asarray(geometry.dR_dtheta_odd)
    zu0 = np.asarray(geometry.dZ_dtheta_even) + sq3 * np.asarray(geometry.dZ_dtheta_odd)

    # Boundary and mid-surface planes on the reduced theta grid.
    r1b_red = r1_even[ns - 1, :ntheta3, :] + r1_odd[ns - 1, :ntheta3, :]
    z1b_red = z1_even[ns - 1, :ntheta3, :] + z1_odd[ns - 1, :ntheta3, :]
    r12_red = r1_even[ns12, :ntheta3, :] + sqrts[ns12] * r1_odd[ns12, :ntheta3, :]
    z12_red = z1_even[ns12, :ntheta3, :] + sqrts[ns12] * z1_odd[ns12, :ntheta3, :]
    ru12_red = 0.5 * (ru0[ns - 1, :ntheta3, :] + ru0[ns12, :ntheta3, :])
    zu12_red = 0.5 * (zu0[ns - 1, :ntheta3, :] + zu0[ns12, :ntheta3, :])

    arrays = {name: np.zeros((ntheta1, nzeta)) for name in
              ("r1b", "z1b", "r12", "z12", "ru12", "zu12")}
    for name, red in (("r1b", r1b_red), ("z1b", z1b_red), ("r12", r12_red),
                      ("z12", z12_red), ("ru12", ru12_red), ("zu12", zu12_red)):
        arrays[name][:ntheta3, :] = red
    r1b, z1b = arrays["r1b"], arrays["z1b"]
    r12, z12 = arrays["r12"], arrays["z12"]
    ru12, zu12 = arrays["ru12"], arrays["zu12"]

    if not lasym:
        # Stellarator symmetry: R(v,-u) = R(2pi-v,u), Z(v,-u) = -Z(2pi-v,u).
        for iv in range(nzeta):
            ivminus = (nzeta - iv) % nzeta
            for iu in range(ntheta2, ntheta1):
                iu_r = ntheta1 - iu
                r1b[iu, iv] = r1b_red[iu_r, ivminus]
                z1b[iu, iv] = -z1b_red[iu_r, ivminus]
                r12[iu, iv] = r12_red[iu_r, ivminus]
                z12[iu, iv] = -z12_red[iu_r, ivminus]
                ru12[iu, iv] = -ru12_red[iu_r, ivminus]
                zu12[iu, iv] = zu12_red[iu_r, ivminus]

    rcom = np.zeros((nzeta,))
    zcom = np.zeros((nzeta,))
    axis_r0 = r1_even[0, 0, :]   # r1(js=1, theta=0, :) — current axis guess
    axis_z0 = z1_even[0, 0, :]

    grid_count = max(int(grid_points), 0)
    grid_frac = np.arange(grid_count, dtype=float) / float(max(grid_count - 1, 1))
    planes = range(nzeta) if lasym else range(nzeta // 2 + 1)
    for iv in planes:
        rmin, rmax = float(np.min(r1b[:, iv])), float(np.max(r1b[:, iv]))
        zmin, zmax = float(np.min(z1b[:, iv])), float(np.max(z1b[:, iv]))
        rbest = 0.5 * (rmax + rmin)
        zbest = 0.5 * (zmax + zmin)

        rs = (r1b[:, iv] - r12[:, iv]) / ds + axis_r0[iv]
        zs = (z1b[:, iv] - z12[:, iv]) / ds + axis_z0[iv]
        tau0 = ru12[:, iv] * zs - zu12[:, iv] * rs

        if grid_count > 0:
            r_grid = rmin + (rmax - rmin) * grid_frac
            if (not lasym) and (iv == 0 or iv == nzeta // 2):
                z_grid = np.zeros((1,))          # up-down symmetric plane
            else:
                z_grid = zmin + (zmax - zmin) * grid_frac
            tau = int(signgs) * (
                tau0[None, None, :]
                - ru12[:, iv][None, None, :] * z_grid[:, None, None]
                + zu12[:, iv][None, None, :] * r_grid[None, :, None]
            )
            min_tau = np.min(tau, axis=2)        # (nz_grid, nr_grid)
            max_tau = float(np.max(min_tau))
            if max_tau > 0.0:
                best_mask = min_tau == max_tau
                first_flat = int(np.argmax(best_mask.reshape(-1)))
                iz_best, ir_best = divmod(first_flat, int(r_grid.size))
                rbest = float(r_grid[ir_best])
                zbest = float(z_grid[iz_best])
                # Ties: prefer the smallest |z| (guess_axis.f mintemp==mintau).
                row_has_best = np.any(best_mask, axis=1)
                if np.any(row_has_best):
                    z_abs = np.abs(z_grid)
                    best_abs = float(np.min(z_abs[row_has_best]))
                    z_rows = np.nonzero(row_has_best & (z_abs == best_abs))[0]
                    if z_rows.size and abs(zbest) > abs(float(z_grid[int(z_rows[0])])):
                        zbest = float(z_grid[int(z_rows[0])])
            elif max_tau == 0.0:
                zero_rows = np.any(min_tau == 0.0, axis=1)
                if np.any(zero_rows):
                    z_abs = np.abs(z_grid)
                    better = zero_rows & (z_abs < abs(zbest))
                    if np.any(better):
                        best_abs = float(np.min(z_abs[better]))
                        z_rows = np.nonzero(better & (z_abs == best_abs))[0]
                        if z_rows.size:
                            zbest = float(z_grid[int(z_rows[0])])
        rcom[iv] = rbest
        zcom[iv] = zbest

    if not lasym:
        # rcom(iv) = rcom(nzeta+2-iv), zcom(iv) = -zcom(nzeta+2-iv).
        for iv in range(nzeta // 2 + 1, nzeta):
            src = nzeta - iv
            rcom[iv] = rcom[src]
            zcom[iv] = -zcom[src]

    # Fourier-project rcom/zcom (guess_axis.f; cosnv carries nscale).
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    dzeta = 2.0 / float(nzeta)
    raxis_c = dzeta * (cosnv.T @ rcom) / nscale
    zaxis_s = -dzeta * (sinnv.T @ zcom) / nscale
    raxis_s = -dzeta * (sinnv.T @ rcom) / nscale
    zaxis_c = dzeta * (cosnv.T @ zcom) / nscale
    ntor = int(nscale.size - 1)
    raxis_c[0] *= 0.5
    zaxis_c[0] *= 0.5
    if (nzeta % 2 == 0) and (nzeta // 2 <= ntor):
        raxis_c[nzeta // 2] *= 0.5
        zaxis_c[nzeta // 2] *= 0.5
    return (jnp.asarray(raxis_c), jnp.asarray(raxis_s),
            jnp.asarray(zaxis_c), jnp.asarray(zaxis_s))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _geometry_state_arrays(
    state: tuple[Array, Array, Array, Array, Array, Array],
    *,
    modes: ModeTable,
    lthreed: bool,
    lasym: bool,
    lconm1: bool,
) -> dict[str, Array]:
    """Evolution-basis ``(R_cos, R_sin, Z_cos, Z_sin, L_cos, L_sin)`` ->
    the keyword dict of :func:`vmex.core.geometry.real_space_geometry`."""
    R_cos, R_sin, Z_cos, Z_sin, lambda_cos, lambda_sin = state
    R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
        R_cos, Z_sin, R_sin, Z_cos,
        modes=modes, lthreed=lthreed, lasym=lasym, lconm1=lconm1,
    )
    ntor = int(np.abs(np.asarray(modes.n)).max(initial=0))
    return dict(
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        lambda_cos=lambda_cos,
        lambda_sin=apply_lambda_axis_closure(lambda_sin, modes=modes, ntor=ntor),
    )


def geometry_state(setup: RunSetup, *, modes: ModeTable) -> dict[str, Array]:
    """Convert the evolution-basis state into geometry-synthesis inputs.

    Applies :func:`vmex.core.residuals.m1_constrained_to_physical`
    (``residue.f90`` — undo the internal m = 1 constraint) and the 3D lambda
    axis closure (``totzsp_mod.f``); the result is the keyword dict expected
    by :func:`vmex.core.geometry.real_space_geometry`.
    """
    return _geometry_state_arrays(
        (setup.R_cos, setup.R_sin, setup.Z_cos, setup.Z_sin,
         setup.lambda_cos, setup.lambda_sin),
        modes=modes, lthreed=setup.lthreed, lasym=setup.lasym, lconm1=setup.lconm1,
    )


def _axis_arrays(inp: VmecInput, dtype) -> tuple[Array, Array, Array, Array]:
    """Dense physical axis arrays from the input (``read_indata_namelist``)."""
    return tuple(jnp.asarray(np.asarray(a, dtype=float), dtype=dtype) for a in
                 (inp.raxis_c, inp.raxis_s, inp.zaxis_c, inp.zaxis_s))


def run_setup(
    inp: VmecInput,
    resolution: Resolution,
    *,
    lconm1: bool = True,
    infer_axis_if_missing: bool = True,
    use_fft: bool = False,
) -> RunSetup:
    """Build the complete pre-iteration setup for one radial resolution.

    VMEC2000: the ``readin.f`` boundary processing + ``profil1d.f`` +
    ``profil3d.f`` initialization sequence of ``runvmec.f``/``eqsolve.f``.

    Parameters
    ----------
    inp:
        Parsed input (:class:`vmex.core.input.VmecInput`); its fields are
        consumed as concrete host numbers (setup runs once per solve).
    resolution:
        Static resolution; ``resolution.ns`` selects the multigrid stage.
    lconm1:
        Apply the m = 1 constraint conversion (VMEC2000 default T).
    infer_axis_if_missing:
        When every input axis coefficient is zero, run :func:`guess_axis` on
        the zero-axis interior guess and re-blend.  This is an explicit
        geometry/setup convenience; production solver construction passes
        ``False`` because VMEC2000 starts from the supplied zero axis and
        calls ``guess_axis`` exactly once after its first bad-Jacobian or
        high-force control transfer.

    Returns
    -------
    :class:`RunSetup` (see its docstring for the field-by-field contract).
    """
    modes = mode_table(resolution.mpol, resolution.ntor)
    trig = trig_tables(resolution)
    grids = radial_grids(resolution.ns)
    dtype = grids.s_full.dtype

    boundary = boundary_from_input(inp, modes=modes, trig=trig, lconm1=lconm1)
    profiles_1d = flux_profiles(
        inp, grids, r00=boundary.r00, signgs=boundary.signgs, lflip=boundary.lflip
    )

    raxis_c, raxis_s, zaxis_c, zaxis_s = _axis_arrays(inp, dtype)

    def build_state(axis):
        return interior_guess(
            boundary_R_cos=boundary.R_cos, boundary_R_sin=boundary.R_sin,
            boundary_Z_cos=boundary.Z_cos, boundary_Z_sin=boundary.Z_sin,
            raxis_c=axis[0], raxis_s=axis[1], zaxis_c=axis[2], zaxis_s=axis[3],
            modes=modes, trig=trig, s=grids.s_full,
        )

    axis = (raxis_c, raxis_s, zaxis_c, zaxis_s)
    state = build_state(axis)

    axis_missing = not any(bool(np.any(np.asarray(a) != 0.0)) for a in axis)
    if axis_missing and bool(infer_axis_if_missing) and resolution.ns >= 2:
        geom = real_space_geometry(
            **_geometry_state_arrays(
                _axis_inference_state(boundary, modes=modes, s=grids.s_full),
                modes=modes, lthreed=bool(resolution.lthreed),
                lasym=bool(resolution.lasym), lconm1=bool(lconm1),
            ),
            modes=modes, trig=trig, s=grids.s_full, use_fft=use_fft,
        )
        axis = guess_axis(geom, s=grids.s_full, trig=trig, signgs=boundary.signgs)
        raxis_c, raxis_s, zaxis_c, zaxis_s = axis
        state = build_state(axis)

    return RunSetup(
        s_full=grids.s_full, s_half=grids.s_half, sqrts=grids.sqrts,
        shalf=grids.shalf, sm=grids.sm, sp=grids.sp, hs=grids.hs,
        scalxc=odd_m_sqrt_s_scaling(grids.s_full, resolution.mpol),
        phips=profiles_1d["phips"], chips=profiles_1d["chips"],
        iotas=profiles_1d["iotas"], icurv=profiles_1d["icurv"],
        mass=profiles_1d["mass"], phipf=profiles_1d["phipf"],
        chipf=profiles_1d["chipf"], iotaf=profiles_1d["iotaf"],
        lamscale=profiles_1d["lamscale"],
        boundary_R_cos=boundary.R_cos, boundary_R_sin=boundary.R_sin,
        boundary_Z_cos=boundary.Z_cos, boundary_Z_sin=boundary.Z_sin,
        raxis_c=raxis_c, raxis_s=raxis_s, zaxis_c=zaxis_c, zaxis_s=zaxis_s,
        R_cos=state[0], R_sin=state[1], Z_cos=state[2], Z_sin=state[3],
        lambda_cos=state[4], lambda_sin=state[5],
        signgs=boundary.signgs, lflip=boundary.lflip,
        lasym=bool(resolution.lasym), lthreed=bool(resolution.lthreed),
        lconm1=bool(lconm1), ncurr=int(inp.ncurr),
    )
