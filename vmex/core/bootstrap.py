"""Differentiable Redl (2021) bootstrap current (R26.g, steps 1-2).

Implements the pure-formula layer and the two geometry lanes of
``notes_r26g_redl_spec.md``:

- :func:`compute_trapped_fraction` — effective trapped fraction, epsilon and
  the flux-surface averages ``<B^2>``, ``<1/B>`` from ``|B|``/``sqrt(g)`` on
  an angular grid.  Differentiable rewrite of simsopt
  ``simsopt.mhd.bootstrap.compute_trapped_fraction`` (fixed-order
  Gauss-Legendre lambda quadrature instead of adaptive ``scipy.integrate.quad``;
  plain grid max/min instead of spline-refined extrema; double-where guard on
  the ``sqrt(1 - lambda B)`` near-singularity so reverse-mode AD stays finite).
- :func:`j_dot_B_redl` — the Redl et al., Phys. Plasmas 28, 022502 (2021)
  ``<J.B>`` formula (eqs. 10-16, 19-21 with the Sauter 18b-18e
  collisionalities), transcribed verbatim from simsopt
  ``j_dot_B_Redl`` including the quasisymmetry isomorphism as simsopt applies
  it: ``iota -> iota - nfp*helicity_n`` everywhere iota appears, ``G``
  unshifted (spec section 3 note).
- :class:`KineticProfiles` / :func:`profile_value_and_dds` — polynomial
  ``ne/Te/Ti/Zeff`` profiles in ``s`` (lowest order first, simsopt
  ``ProfilePolynomial`` convention), value + analytic d/ds via Horner.
- :func:`redl_geometry_from_wout` — parity lane mirroring simsopt
  ``RedlGeomVmec.__call__``: half-mesh linear interpolation of
  ``iotas/bvco/buco/gmnc/bmnc`` onto the requested surfaces, cosine synthesis
  of ``|B|``/``sqrt(g)`` on a uniform ``(theta, phi)`` grid, then
  :func:`compute_trapped_fraction`.
- :func:`redl_geometry_from_state` — traceable lane on
  ``(SpectralState, SolverRuntime)``: ``|B|`` and ``sqrt(g)`` from the
  solver's half-mesh internal grid (mirrored from the reduced ``[0, pi]``
  theta grid exactly as ``QuasisymmetryRatioResidual._pointwise_state``),
  iota/G/I from ``_iotas_half``/``surface_currents``.

Steps 3-4 of the spec (this module, second landing):

- :func:`vmec_j_dot_B` / :func:`vmec_j_dot_B_from_wout` — the equilibrium
  ``<J.B>`` via the MHD identity (spec section 6.2, validated against the
  Zenodo ``convertSfincsToVmecCurrentProfile`` script and the wout
  ``jdotb``):

      <J.B>(s) = [<B^2> dI/ds + mu0 I dp/ds] / (2 pi psi_a),
      I(s) = signgs*(2 pi/mu0)*buco(s) [A],  psi_a = phi(1)/(2 pi) [Wb/rad]

  (the ``signgs`` matches VMEC's ``ctor = signgs*(2 pi/mu0)*buco(ns)``
  convention, so the identity reproduces the sign of the wout ``jdotb``).
- :class:`RedlBootstrapMismatch` — the paper/simsopt ``f_boot`` normalized
  residual ``R_j = (Jv_j - Jr_j)/sqrt(sum_k (Jv_k + Jr_k)^2)`` comparing
  ``<J.B>_vmec`` against :func:`j_dot_B_redl`, with the same wout /
  traceable-state dual-lane shape as ``QuasisymmetryRatioResidual`` (so it
  composes with ``least_squares`` at ``jac=None`` and ``jac="implicit"``).
- :func:`self_consistent_bootstrap` — fixed-boundary Picard iteration
  ``AC/CURTOR <- current profile implied by the Redl <J.B>`` (the Zenodo
  script's "smooth method": solve ``[<B^2> d/ds + mu0 dp/ds] I = 2 pi psi_a
  J_Redl`` with ``I(0) = 0``, then re-fit the ``I'(s)`` power series).

Units follow simsopt/the spec: ``ne`` [1/m^3], ``Te/Ti`` [eV], ``G/I/R``
[T*m], ``psi_edge`` [Wb/rad], output ``<J.B>`` [A*T/m^2].

Note (spec section 6.1b): the spec's stated decision was to hoist
``_field_chain``/``_iotas_half``/``_half_grid``/``_interp_half_grid`` into a
shared module; since R26a they live in :mod:`vmex.core.statephysics`
(``optimize`` re-exports them for backward compatibility).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from .fields import surface_currents
from .input import VmecInput
from .profiles import MU0
from .solver import SolverRuntime, SpectralState
from .statephysics import (
    _as_1d,
    _field_chain,
    _half_grid,
    _interp_half_grid,
    _iotas_half,
    _mode_matrix,
)
from .wout import read_wout

__all__ = [
    "ELEMENTARY_CHARGE",
    "KineticProfiles",
    "profile_value_and_dds",
    "compute_trapped_fraction",
    "RedlGeometry",
    "redl_geometry_from_wout",
    "redl_geometry_from_state",
    "j_dot_B_redl",
    "vmec_j_dot_B",
    "vmec_j_dot_B_from_wout",
    "RedlBootstrapMismatch",
    "BootstrapPicardResult",
    "self_consistent_bootstrap",
]

Array = Any

#: CODATA elementary charge [C]; converts eV -> J in the <J.B> assembly.
ELEMENTARY_CHARGE = 1.602176634e-19


# ===========================================================================
# Kinetic profiles (polynomials in s, simsopt ProfilePolynomial convention)
# ===========================================================================


def profile_value_and_dds(coeffs: Array, s: Array) -> tuple[Array, Array]:
    """Polynomial profile value and analytic d/ds at ``s`` (traceable).

    ``coeffs`` are polynomial coefficients in ``s``, lowest order first
    (simsopt ``ProfilePolynomial``): ``p(s) = sum_k coeffs[k] * s**k``.
    Evaluated with the paired Horner recurrence, so the derivative is exact
    and both outputs are differentiable in ``coeffs`` and ``s``.
    """
    coeffs = jnp.atleast_1d(jnp.asarray(coeffs))
    s = jnp.asarray(s)
    value = jnp.zeros_like(s)
    dds = jnp.zeros_like(s)
    for k in range(int(coeffs.shape[0]) - 1, -1, -1):
        dds = dds * s + value
        value = value * s + coeffs[k]
    return value, dds


@dataclass(frozen=True)
class KineticProfiles:
    """Prescribed kinetic profiles for the Redl formula (frozen pytree).

    Polynomial coefficients in ``s`` (lowest order first).  These are *not*
    VMEC inputs; they parameterize the bootstrap objective (spec section 6.6).
    Units: ``ne_coeffs`` [1/m^3], ``Te_coeffs``/``Ti_coeffs`` [eV],
    ``Zeff_coeffs`` dimensionless (default: constant 1, hydrogen).

    Example (paper profiles): ``ne = n0*(1 - s^5)``, ``Te = Ti = T0*(1 - s)``::

        KineticProfiles(ne_coeffs=4.13e20 * np.array([1, 0, 0, 0, 0, -1]),
                        Te_coeffs=12.0e3 * np.array([1, -1]),
                        Ti_coeffs=12.0e3 * np.array([1, -1]))
    """

    ne_coeffs: Array
    Te_coeffs: Array
    Ti_coeffs: Array
    Zeff_coeffs: Array = 1.0


jax.tree_util.register_dataclass(
    KineticProfiles,
    data_fields=["ne_coeffs", "Te_coeffs", "Ti_coeffs", "Zeff_coeffs"],
    meta_fields=[],
)


# ===========================================================================
# Trapped fraction (differentiable rewrite of simsopt compute_trapped_fraction)
# ===========================================================================


def compute_trapped_fraction(modB: Array, sqrtg: Array, *, n_lambda: int = 64):
    r"""Effective trapped fraction and flux-surface averages per surface.

    ``f_t = 1 - (3/4) <B^2> \int_0^{1/Bmax} lambda dlambda / <sqrt(1 - lambda B)>``
    with ``<.>`` the flux-surface average (weight ``|sqrt(g)|``; only ratios
    of averages enter, so the uniform Jacobian sign cancels).

    Args:
        modB: ``(nsurf, ntheta, nzeta)`` array of :math:`|B|` on a uniform
            angular grid (leading surface axis — note simsopt uses trailing).
        sqrtg: same shape, the Jacobian :math:`\sqrt{g}` on the grid.
        n_lambda: fixed Gauss-Legendre quadrature order for the lambda
            integral on ``[0, 1/Bmax]`` (replaces simsopt's adaptive quad;
            the nodes exclude the ``lambda = 1/Bmax`` endpoint).

    Returns:
        ``(Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t)`` — 1D arrays of
        length ``nsurf`` (same tuple as simsopt).  ``Bmin/Bmax`` are hard grid
        extrema: piecewise-smooth gradients, adequate for trust-region least
        squares (same stance as :func:`vmex.core.optimize.mirror_ratio`).
    """
    modB = jnp.asarray(modB)
    sqrtg = jnp.asarray(sqrtg)
    if modB.ndim != 3 or modB.shape != sqrtg.shape:
        raise ValueError("modB and sqrtg must both have shape (nsurf, ntheta, nzeta)")

    w = jnp.abs(sqrtg)
    Vp = jnp.mean(w, axis=(1, 2))
    fsa_B2 = jnp.mean(modB * modB * w, axis=(1, 2)) / Vp
    fsa_1overB = jnp.mean(w / modB, axis=(1, 2)) / Vp
    Bmax = jnp.max(modB, axis=(1, 2))
    Bmin = jnp.min(modB, axis=(1, 2))
    epsilon = (Bmax - Bmin) / (Bmax + Bmin)

    # Gauss-Legendre nodes/weights on [0, 1] (host constants, order is static).
    nodes, weights = np.polynomial.legendre.leggauss(int(n_lambda))
    x = jnp.asarray(0.5 * (nodes + 1.0), dtype=modB.dtype)      # (nl,) in (0, 1)
    wq = jnp.asarray(0.5 * weights, dtype=modB.dtype)

    lam = x[None, :] / Bmax[:, None]                            # (nsurf, nl)
    arg = 1.0 - lam[:, :, None, None] * modB[:, None, :, :]
    # Double-where guard: at the |B| = Bmax grid point arg -> 0 as
    # lambda -> 1/Bmax; d/dx sqrt(x) is infinite there, so reverse-mode AD
    # must never see sqrt evaluated at (or differentiated through) arg <= 0.
    positive = arg > 0.0
    safe = jnp.where(positive, arg, 1.0)
    root = jnp.where(positive, jnp.sqrt(safe), 0.0)
    fsa_root = (jnp.mean(root * w[:, None, :, :], axis=(2, 3))
                / Vp[:, None])                                  # <sqrt(1 - lam B)>
    integral = jnp.sum(wq[None, :] * lam / fsa_root, axis=1) / Bmax
    f_t = 1.0 - 0.75 * fsa_B2 * integral
    return Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t


# ===========================================================================
# Geometry container + the two lanes
# ===========================================================================


@dataclass(frozen=True)
class RedlGeometry:
    """Per-surface geometry inputs of the Redl formula (frozen pytree).

    All arrays are 1D over ``surfaces`` (values of normalized toroidal flux).
    ``G``/``I`` are the Boozer covariant averages (wout ``bvco``/``buco``,
    [T*m]); ``R = (G + iota*I) * <1/B>`` is the effective major radius for the
    Sauter collisionality; ``psi_edge = -phi(s=1)/(2 pi)`` [Wb/rad] (wout sign
    convention).  ``nfp`` is static metadata.
    """

    surfaces: Array
    iota: Array
    G: Array
    I: Array  # noqa: E741 - Boozer I
    R: Array
    epsilon: Array
    f_t: Array
    fsa_B2: Array
    fsa_1overB: Array
    Bmin: Array
    Bmax: Array
    psi_edge: Array
    nfp: int = 1


jax.tree_util.register_dataclass(
    RedlGeometry,
    data_fields=["surfaces", "iota", "G", "I", "R", "epsilon", "f_t",
                 "fsa_B2", "fsa_1overB", "Bmin", "Bmax", "psi_edge"],
    meta_fields=["nfp"],
)


def redl_geometry_from_wout(
    wout,
    surfaces,
    *,
    ntheta: int = 64,
    nphi: int = 65,
    n_lambda: int = 64,
) -> RedlGeometry:
    """Redl geometry inputs from a wout dataset (simsopt ``RedlGeomVmec`` lane).

    ``wout`` may be a :class:`~vmex.core.wout.WoutData`, a path to a
    ``wout_*.nc`` file, or anything exposing a ``.wout`` attribute (e.g.
    :class:`~vmex.core.optimize.Equilibrium`).  Mirrors simsopt
    ``RedlGeomVmec.__call__``: linear interpolation of the half-mesh tables
    ``iotas/bvco/buco/gmnc/bmnc`` (axis slot dropped) onto ``surfaces``,
    cosine synthesis of ``|B|``/``sqrt(g)`` from the Nyquist mode tables on
    ``theta in [0, 2 pi)`` x ``phi in [0, 2 pi/nfp)`` grids, then
    :func:`compute_trapped_fraction`.  jnp throughout; this is the
    validation/finite-difference lane, not the implicit path.
    """
    if isinstance(wout, (str, Path)):
        wout = read_wout(wout)
    wout = getattr(wout, "wout", wout)
    if bool(getattr(wout, "lasym", False)):
        raise NotImplementedError("redl_geometry_from_wout supports lasym = False only")

    surfaces = _as_1d(surfaces)
    nfp = int(wout.nfp)
    iotas = _as_1d(np.asarray(wout.iotas, dtype=float))
    ns = int(iotas.shape[0])
    s_half = _half_grid(ns, iotas.dtype)

    def half(values):
        return _interp_half_grid(values[1:], surfaces, s_half)

    iota = half(iotas)
    G = half(_as_1d(np.asarray(wout.bvco, dtype=float)))
    I = half(_as_1d(np.asarray(wout.buco, dtype=float)))  # noqa: E741 - Boozer I

    xm = _as_1d(np.asarray(wout.xm_nyq, dtype=float))
    xn = _as_1d(np.asarray(wout.xn_nyq, dtype=float))
    mn = int(xm.shape[0])
    bmnc = half(_mode_matrix(wout, "bmnc", ns=ns, mn=mn))
    gmnc = half(_mode_matrix(wout, "gmnc", ns=ns, mn=mn))

    theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, int(nphi), endpoint=False)
    angle = (theta1d[:, None, None] * xm[None, None, :]
             - phi1d[None, :, None] * xn[None, None, :])
    cosangle = jnp.cos(angle)
    modB = jnp.einsum("sm,tpm->stp", bmnc, cosangle)
    sqrtg = jnp.einsum("sm,tpm->stp", gmnc, cosangle)

    Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(
        modB, sqrtg, n_lambda=n_lambda)
    R = (G + iota * I) * fsa_1overB
    psi_edge = -_as_1d(np.asarray(wout.phi, dtype=float))[-1] / (2.0 * jnp.pi)
    return RedlGeometry(
        surfaces=surfaces, iota=iota, G=G, I=I, R=R, epsilon=epsilon, f_t=f_t,
        fsa_B2=fsa_B2, fsa_1overB=fsa_1overB, Bmin=Bmin, Bmax=Bmax,
        psi_edge=psi_edge, nfp=nfp)


class _HalfMeshFields(NamedTuple):
    """Traceable half-mesh (js = 1..ns-1) raw fields of one core state.

    ``bmag``/``w`` are ``(ns-1, ntheta1, nzeta)`` on the mirrored full theta
    circle; the per-surface arrays are 1D of length ``ns-1``.  ``p_int`` is
    the kinetic pressure in internal units (``mu0*Pa``); ``phi_edge`` is
    ``phi(s=1)/(2 pi)`` [Wb/rad] (the un-negated wout convention — the Redl
    ``psi_edge`` is its negative).
    """

    s_half: Array
    bmag: Array
    w: Array
    iota: Array
    G: Array
    I: Array  # noqa: E741 - Boozer I
    p_int: Array
    phi_edge: Array
    signgs: float
    nfp: int


def _half_mesh_fields(state: SpectralState, rt: SolverRuntime) -> _HalfMeshFields:
    """One ``_field_chain`` pass -> the half-mesh inputs of both bootstrap lanes.

    ``|B|`` and ``sqrt(g)`` on the solver's half-mesh internal grid
    (``|B|^2 = 2 (total_pressure - pressure)``, bcovar.f) with the reduced
    ``[0, pi]`` theta grid mirrored to the full circle exactly as
    ``QuasisymmetryRatioResidual._pointwise_state`` does; the axis row
    ``js = 0`` is dropped *before* any singular division (0*inf AD note,
    optimize.py).  iota comes from ``_iotas_half`` (ncurr=1-aware), G/I from
    :func:`~vmex.core.fields.surface_currents` (``bvco``/``buco``).
    """
    setup = rt.setup
    if bool(setup.lasym):
        raise NotImplementedError(
            "bootstrap state-lane evaluation supports lasym = False only")
    s = jnp.asarray(setup.s_full)
    nfp = int(rt.resolution.nfp)
    _, jacobian, _, fields, _ = _field_chain(state, rt)

    # Mirror the reduced [0, pi] theta grid to the full circle
    # (stellarator-symmetry map X(2 pi - theta, -zeta) = X(theta, zeta)).
    ntheta2 = int(np.shape(fields.total_pressure)[1])
    nzeta = int(np.shape(fields.total_pressure)[2])
    ntheta1 = max(2 * (ntheta2 - 1), 1)
    i_full = np.arange(ntheta1)
    i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
    k = np.arange(nzeta)
    k_src = np.where(i_full[:, None] < ntheta2, k[None, :],
                     (nzeta - k[None, :]) % nzeta)
    i_src = np.broadcast_to(i_src[:, None], (ntheta1, nzeta))

    def full(a):
        # Drop the zeroed axis row (js = 0) before the divisions downstream.
        return jnp.asarray(a)[1:, i_src, k_src]

    bsq2 = 2.0 * (jnp.asarray(fields.total_pressure)
                  - jnp.asarray(fields.pressure)[:, None, None])
    tiny = jnp.asarray(jnp.finfo(bsq2.dtype).tiny, dtype=bsq2.dtype)
    bmag = jnp.sqrt(jnp.maximum(full(bsq2), tiny))     # half mesh js = 1..ns-1
    w = jnp.abs(full(jacobian.sqrt_g))

    iota_h = _iotas_half(state, rt)[1:]
    cur = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv,
                           trig=rt.trig, s=s, signgs=setup.signgs)
    G_h = jnp.asarray(cur.bvco)[1:]
    I_h = jnp.asarray(cur.buco)[1:]
    p_int = jnp.asarray(fields.pressure)[1:]           # mu0*Pa, half mesh

    hs = s[1] - s[0]
    phi_edge = float(setup.signgs) * hs * jnp.sum(jnp.asarray(setup.phipf)[1:])
    return _HalfMeshFields(
        s_half=0.5 * (s[:-1] + s[1:]), bmag=bmag, w=w, iota=iota_h, G=G_h,
        I=I_h, p_int=p_int, phi_edge=phi_edge, signgs=float(setup.signgs),
        nfp=nfp)


def _geometry_from_half(hm: _HalfMeshFields, surfaces, *, n_lambda: int) -> RedlGeometry:
    """Interpolate half-mesh fields onto ``surfaces`` -> :class:`RedlGeometry`."""
    iota = _interp_half_grid(hm.iota, surfaces, hm.s_half)
    G = _interp_half_grid(hm.G, surfaces, hm.s_half)
    I = _interp_half_grid(hm.I, surfaces, hm.s_half)  # noqa: E741 - Boozer I
    modB = _interp_half_grid(hm.bmag, surfaces, hm.s_half)
    sqrtg = _interp_half_grid(hm.w, surfaces, hm.s_half)

    Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(
        modB, sqrtg, n_lambda=n_lambda)
    R = (G + iota * I) * fsa_1overB
    return RedlGeometry(
        surfaces=surfaces, iota=iota, G=G, I=I, R=R, epsilon=epsilon, f_t=f_t,
        fsa_B2=fsa_B2, fsa_1overB=fsa_1overB, Bmin=Bmin, Bmax=Bmax,
        psi_edge=-hm.phi_edge, nfp=hm.nfp)


def redl_geometry_from_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces=None,
    n_lambda: int = 64,
) -> RedlGeometry:
    """Redl geometry inputs as a traceable function of ``(state, runtime)``.

    ``|B|`` and ``sqrt(g)`` are taken on the solver's half-mesh internal grid
    (see :func:`_half_mesh_fields`); ``psi_edge = -signgs*hs*sum(phipf[1:])``.
    Per-surface scalars and the angular fields are linearly interpolated from
    the half mesh onto ``surfaces`` before :func:`compute_trapped_fraction`.

    Default ``surfaces``: ``linspace(0.05, 0.95, 16)`` — interior only; do
    not sample ``s -> 1`` where ``Te, Ti -> 0`` blows up the collisionality
    (spec section 6.1b).
    """
    if surfaces is None:
        surfaces = np.linspace(0.05, 0.95, 16)
    surfaces = _as_1d(surfaces)
    return _geometry_from_half(_half_mesh_fields(state, rt), surfaces,
                               n_lambda=n_lambda)


# ===========================================================================
# Redl (2021) <J.B> formula (transcribed verbatim from simsopt j_dot_B_Redl)
# ===========================================================================


def j_dot_B_redl(
    profiles: KineticProfiles,
    geom: RedlGeometry,
    helicity_n: int,
) -> tuple[Array, dict[str, Array]]:
    """Bootstrap ``<J.B>`` [A*T/m^2] from the Redl (2021) formulae (pure jnp).

    Equation-by-equation transcription of simsopt ``j_dot_B_Redl`` (which
    generated the arXiv:2205.02914 results): Sauter eqs. (18b)-(18e) for the
    Coulomb logarithms and collisionalities, Redl eqs. (10)-(16) and
    (19)-(21) for ``L31/L32/L34/alpha``.  The quasisymmetry isomorphism is
    applied exactly as simsopt does: ``iota -> iota - N`` with
    ``N = nfp*helicity_n`` in the collisionality geometry factor and in the
    ``1/(psi_edge*(iota - N))`` prefactor; ``G`` (and the ``R`` provided by
    the geometry lanes) is used unshifted.

    ``helicity_n`` is 0 for quasi-axisymmetry, +/-1 for quasi-helical
    symmetry.  ``Te/Ti`` are clamped at 1 eV and ``ne`` at 1e17 1/m^3
    (belt-and-suspenders against sampling s -> 1 where the profiles vanish
    and the collisionality blows up; spec section 6.1b).

    Returns ``(jdotB, details)`` with ``details`` a dict of every
    intermediate quantity (simsopt names).
    """
    s = jnp.asarray(geom.surfaces)
    ne_s, d_ne_d_s = profile_value_and_dds(profiles.ne_coeffs, s)
    Te_s, d_Te_d_s = profile_value_and_dds(profiles.Te_coeffs, s)
    Ti_s, d_Ti_d_s = profile_value_and_dds(profiles.Ti_coeffs, s)
    Zeff_s, _ = profile_value_and_dds(profiles.Zeff_coeffs, s)

    ne_s = jnp.maximum(ne_s, 1e17)
    Te_s = jnp.maximum(Te_s, 1.0)
    Ti_s = jnp.maximum(Ti_s, 1.0)
    ni_s = ne_s / Zeff_s
    pe_s = ne_s * Te_s
    pi_s = ni_s * Ti_s

    # Sauter eqs. (18d)-(18e):
    ln_Lambda_e = 31.3 - jnp.log(jnp.sqrt(ne_s) / Te_s)
    ln_Lambda_ii = 30.0 - jnp.log(Zeff_s ** 3 * jnp.sqrt(ni_s) / (Ti_s ** 1.5))

    # Sauter eqs. (18b)-(18c) with the isomorphism substitution iota -> iota - N:
    helicity_N = int(geom.nfp) * int(helicity_n)
    iota_N = geom.iota - helicity_N
    geometry_factor = jnp.abs(geom.R / iota_N)
    nu_e = (geometry_factor * (6.921e-18) * ne_s * Zeff_s * ln_Lambda_e
            / (Te_s * Te_s * (geom.epsilon ** 1.5)))
    nu_i = (geometry_factor * (4.90e-18) * ni_s * (Zeff_s ** 4) * ln_Lambda_ii
            / (Ti_s * Ti_s * (geom.epsilon ** 1.5)))

    f_t = geom.f_t
    sqrt_nu_e = jnp.sqrt(nu_e)
    sqrt_nu_i = jnp.sqrt(nu_i)
    # AD-safe sqrt(Zeff - 1): Zeff = 1 exactly is the common case and
    # d/dx sqrt(x) is infinite at x = 0 (double-where guard).
    zm1 = Zeff_s - 1.0
    zm1_pos = zm1 > 0.0
    sqrt_zm1 = jnp.where(zm1_pos, jnp.sqrt(jnp.where(zm1_pos, zm1, 1.0)), 0.0)

    # Redl eq (11):
    X31 = f_t / (1 + (0.67 * (1 - 0.7 * f_t) * sqrt_nu_e) / (0.56 + 0.44 * Zeff_s)
                 + (0.52 + 0.086 * sqrt_nu_e) * (1 + 0.87 * f_t) * nu_e / (1 + 1.13 * sqrt_zm1))

    # Redl eq (10):
    Zfac = Zeff_s ** 1.2 - 0.71
    L31 = (1 + 0.15 / Zfac) * X31 \
        - 0.22 / Zfac * (X31 ** 2) \
        + 0.01 / Zfac * (X31 ** 3) \
        + 0.06 / Zfac * (X31 ** 4)

    # Redl eq (14):
    X32e = f_t / ((1 + 0.23 * (1 - 0.96 * f_t) * sqrt_nu_e / jnp.sqrt(Zeff_s)
                   + 0.13 * (1 - 0.38 * f_t) * nu_e / (Zeff_s * Zeff_s)
                   * (jnp.sqrt(1 + 2 * sqrt_zm1)
                      + f_t * f_t * jnp.sqrt((0.075 + 0.25 * zm1 ** 2) * nu_e))))

    # Redl eq (13):
    F32ee = (0.1 + 0.6 * Zeff_s) * (X32e - X32e ** 4) \
        / (Zeff_s * (0.77 + 0.63 * (1 + zm1 ** 1.1))) \
        + 0.7 / (1 + 0.2 * Zeff_s) * (X32e ** 2 - X32e ** 4 - 1.2 * (X32e ** 3 - X32e ** 4)) \
        + 1.3 / (1 + 0.5 * Zeff_s) * (X32e ** 4)

    # Redl eq (16):
    X32ei = f_t / (1 + 0.87 * (1 + 0.39 * f_t) * sqrt_nu_e / (1 + 2.95 * zm1 ** 2)
                   + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * zm1))

    # Redl eq (15):
    F32ei = -(0.4 + 1.93 * Zeff_s) / (Zeff_s * (0.8 + 0.6 * Zeff_s)) * (X32ei - X32ei ** 4) \
        + 5.5 / (1.5 + 2 * Zeff_s) * (X32ei ** 2 - X32ei ** 4 - 0.8 * (X32ei ** 3 - X32ei ** 4)) \
        - 1.3 / (1 + 0.5 * Zeff_s) * (X32ei ** 4)

    # Redl eq (12):
    L32 = F32ei + F32ee

    # Redl eq (19):
    L34 = L31

    # Redl eq (20):
    alpha0 = -(0.62 + 0.055 * zm1) * (1 - f_t) \
        / ((0.53 + 0.17 * zm1) * (1 - (0.31 - 0.065 * zm1) * f_t - 0.25 * f_t * f_t))
    # Redl eq (21):
    alpha = ((alpha0 + 0.7 * Zeff_s * jnp.sqrt(f_t * nu_i)) / (1 + 0.18 * sqrt_nu_i)
             - 0.002 * nu_i * nu_i * (f_t ** 6)) \
        / (1 + 0.004 * nu_i * nu_i * (f_t ** 6))

    # Factor of ELEMENTARY_CHARGE converts temperatures from eV to J:
    pref = -geom.G * ELEMENTARY_CHARGE / (geom.psi_edge * iota_N)
    dnds_term = pref * (ne_s * Te_s + ni_s * Ti_s) * L31 * (d_ne_d_s / ne_s)
    dTeds_term = pref * pe_s * (L31 + L32) * (d_Te_d_s / Te_s)
    dTids_term = pref * pi_s * (L31 + L34 * alpha) * (d_Ti_d_s / Ti_s)
    jdotB = dnds_term + dTeds_term + dTids_term

    details = {
        "s": s, "ne_s": ne_s, "ni_s": ni_s, "Zeff_s": Zeff_s,
        "Te_s": Te_s, "Ti_s": Ti_s,
        "d_ne_d_s": d_ne_d_s, "d_Te_d_s": d_Te_d_s, "d_Ti_d_s": d_Ti_d_s,
        "ln_Lambda_e": ln_Lambda_e, "ln_Lambda_ii": ln_Lambda_ii,
        "nu_e_star": nu_e, "nu_i_star": nu_i,
        "X31": X31, "X32e": X32e, "X32ei": X32ei,
        "F32ee": F32ee, "F32ei": F32ei,
        "L31": L31, "L32": L32, "L34": L34,
        "alpha0": alpha0, "alpha": alpha,
        "dnds_term": dnds_term, "dTeds_term": dTeds_term,
        "dTids_term": dTids_term, "jdotB": jdotB,
    }
    return jdotB, details


# ===========================================================================
# Traceable <J.B>_vmec — the MHD identity (spec section 6.2)
# ===========================================================================


def _dds_half(f: Array, hs) -> Array:
    """d/ds of half-mesh radial samples on the half mesh itself.

    Central differences in the interior, one-sided two-point stencils at the
    ends (the spec's "half -> full central differences, ends extrapolated"
    evaluated back on the half mesh — identical values, no regridding).
    Works on 1D per-surface arrays (leading radial axis).
    """
    f = jnp.asarray(f)
    d0 = (f[1] - f[0]) / hs
    dn = (f[-1] - f[-2]) / hs
    return jnp.concatenate([d0[None], (f[2:] - f[:-2]) / (2.0 * hs), dn[None]])


def _jv_from_half(hm: _HalfMeshFields, surfaces) -> Array:
    """``<J.B>_vmec`` at ``surfaces`` from half-mesh fields (see :func:`vmec_j_dot_B`)."""
    Vp = jnp.mean(hm.w, axis=(1, 2))
    fsa_B2 = jnp.mean(hm.bmag * hm.bmag * hm.w, axis=(1, 2)) / Vp
    hs = hm.s_half[1] - hm.s_half[0]
    jv_half = hm.signgs * (fsa_B2 * _dds_half(hm.I, hs)
                           + hm.I * _dds_half(hm.p_int, hs)) / (MU0 * hm.phi_edge)
    return _interp_half_grid(jv_half, surfaces, hm.s_half)


def vmec_j_dot_B(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces=None,
) -> Array:
    """Traceable equilibrium ``<J.B>`` [A*T/m^2] via the MHD identity.

    For a VMEC equilibrium the flux-surface-averaged parallel current obeys
    (spec section 6.2; validated in the Zenodo
    ``convertSfincsToVmecCurrentProfile`` script against VMEC's own
    ``jdotb``)

        <J.B>(s) = [<B^2> dI/ds + mu0 I dp/ds] / (2 pi psi_a)

    with ``I(s) = signgs*(2 pi/mu0)*buco(s)`` [A] (VMEC's
    ``ctor = signgs*(2 pi/mu0)*buco(ns)`` sign convention), ``p`` in Pa and
    ``psi_a = phi(1)/(2 pi)`` [Wb/rad].  In core internal units
    (``fields.pressure`` is ``mu0*Pa``, ``buco`` [T*m]) this reduces to
    ``signgs*(<B^2> buco' + buco p_int') / (mu0 psi_a)``.  Radial derivatives
    are half-mesh central differences (one-sided at the ends), then the
    half-mesh profile is linearly interpolated onto ``surfaces`` (default:
    ``linspace(0.05, 0.95, 16)``, matching
    :func:`redl_geometry_from_state`).

    Agrees with the wout-engine ``jdotb`` (jxbforce.f) to a few 1e-4
    relative on the Zenodo finite-beta optima — no ``bsubs`` port needed.
    """
    if surfaces is None:
        surfaces = np.linspace(0.05, 0.95, 16)
    return _jv_from_half(_half_mesh_fields(state, rt), _as_1d(surfaces))


def vmec_j_dot_B_from_wout(wout, surfaces, *, geom: RedlGeometry | None = None,
                           ntheta: int = 64, nphi: int = 65) -> Array:
    """The section-6.2 identity evaluated from wout tables (parity lane).

    Same formula as :func:`vmec_j_dot_B` with ``buco``/``pres``/``phi`` read
    from the wout dataset and ``<B^2>`` synthesized from ``bmnc``/``gmnc`` at
    the requested ``surfaces``; pass ``geom`` (a :class:`RedlGeometry` from
    :func:`redl_geometry_from_wout` at the *same* surfaces) to reuse its
    ``fsa_B2`` instead.  This is the validation lane — the wout ``jdotb``
    itself (jxbforce.f) is what :class:`RedlBootstrapMismatch` consumes.
    """
    if isinstance(wout, (str, Path)):
        wout = read_wout(wout)
    wout = getattr(wout, "wout", wout)
    surfaces = _as_1d(surfaces)

    ns = int(wout.ns)
    s_full = jnp.linspace(0.0, 1.0, ns)
    hs = s_full[1] - s_full[0]
    s_half = _half_grid(ns, s_full.dtype)
    buco = _as_1d(np.asarray(wout.buco, dtype=float))[1:]
    p_int = MU0 * _as_1d(np.asarray(wout.pres, dtype=float))[1:]
    phi_edge = _as_1d(np.asarray(wout.phi, dtype=float))[-1] / (2.0 * jnp.pi)
    signgs = float(wout.signgs)

    if geom is not None:
        fsa_B2 = jnp.asarray(geom.fsa_B2)
    else:
        xm = _as_1d(np.asarray(wout.xm_nyq, dtype=float))
        xn = _as_1d(np.asarray(wout.xn_nyq, dtype=float))
        mn = int(xm.shape[0])
        bmnc = _interp_half_grid(_mode_matrix(wout, "bmnc", ns=ns, mn=mn)[1:],
                                 surfaces, s_half)
        gmnc = _interp_half_grid(_mode_matrix(wout, "gmnc", ns=ns, mn=mn)[1:],
                                 surfaces, s_half)
        theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
        phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / int(wout.nfp), int(nphi),
                             endpoint=False)
        angle = (theta1d[:, None, None] * xm[None, None, :]
                 - phi1d[None, :, None] * xn[None, None, :])
        cosangle = jnp.cos(angle)
        modB = jnp.einsum("sm,tpm->stp", bmnc, cosangle)
        sqrtg = jnp.abs(jnp.einsum("sm,tpm->stp", gmnc, cosangle))
        fsa_B2 = (jnp.mean(modB * modB * sqrtg, axis=(1, 2))
                  / jnp.mean(sqrtg, axis=(1, 2)))

    dI = _interp_half_grid(_dds_half(buco, hs), surfaces, s_half)
    dp = _interp_half_grid(_dds_half(p_int, hs), surfaces, s_half)
    I = _interp_half_grid(buco, surfaces, s_half)  # noqa: E741 - Boozer I
    return signgs * (fsa_B2 * dI + I * dp) / (MU0 * phi_edge)


# ===========================================================================
# f_boot objective (spec section 6.3; simsopt VmecRedlBootstrapMismatch)
# ===========================================================================


class RedlBootstrapMismatch:
    """Normalized ``<J.B>_vmec`` vs ``<J.B>_Redl`` mismatch (paper ``f_boot``).

    Residual vector (simsopt ``VmecRedlBootstrapMismatch.residuals``
    verbatim)::

        R_j = (Jv(s_j) - Jr(s_j)) / sqrt(sum_k (Jv(s_k) + Jr(s_k))**2)

    over the geometry ``surfaces``, so ``sum(R**2)`` is the paper's

        f_boot = int ds [<J.B>_vmec - <J.B>_Redl]^2
                 / int ds [<J.B>_vmec + <J.B>_Redl]^2

    (bounded by 1; the denominator depends on the dofs — the simsopt/paper
    self-normalizing convention, kept as is).  Dual-lane shape mirroring
    :class:`~vmex.core.optimize.QuasisymmetryRatioResidual`:

    - :meth:`J`/:meth:`residuals` (wout lane, FD-only): ``Jr`` from
      :func:`redl_geometry_from_wout`, ``Jv`` from the wout ``jdotb``
      (jxbforce.f) interpolated onto ``surfaces`` — simsopt parity.
    - :meth:`residuals_state` (traceable lane): one
      :func:`_half_mesh_fields` pass feeds both the Redl geometry and the
      section-6.2 identity ``Jv``, so the term composes with
      ``least_squares(jac="implicit")`` (profiles fixed, geometry traced).

    ``helicity_n`` is 0 for QA, -1/+1 for QH (units of ``nfp``, simsopt
    convention).  Default ``surfaces``: ``linspace(0.05, 0.95, 16)``
    (interior only — never sample ``s -> 1``; spec section 6.1b).  Usage as
    an objective term (target 0; the normalization already scales it)::

        terms = [(qs.residuals_state, 0.0, 1.0),
                 (boot.residuals_state, 0.0, w_boot), ...]
    """

    name = "f_boot"

    def __init__(
        self,
        profiles: KineticProfiles,
        helicity_n: int,
        surfaces=None,
        *,
        ntheta: int = 64,
        nphi: int = 65,
        n_lambda: int = 64,
    ):
        self.profiles = profiles
        self.helicity_n = int(helicity_n)
        self.surfaces = np.atleast_1d(np.asarray(
            np.linspace(0.05, 0.95, 16) if surfaces is None else surfaces,
            dtype=float))
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)
        self.n_lambda = int(n_lambda)

    @staticmethod
    def _residual_vector(jv: Array, jr: Array) -> jnp.ndarray:
        denom_sq = jnp.sum((jv + jr) ** 2)
        tiny = jnp.asarray(jnp.finfo(jnp.asarray(denom_sq).dtype).tiny)
        return (jv - jr) / jnp.sqrt(jnp.maximum(denom_sq, tiny))

    # -- wout-table evaluation (simsopt parity; FD lane) -----------------------

    def residuals(self, wout) -> jnp.ndarray:
        """Residual vector from a wout-like object / path / ``Equilibrium``."""
        wout = getattr(wout, "wout", wout)
        if isinstance(wout, (str, Path)):
            wout = read_wout(wout)
        surfaces = _as_1d(self.surfaces)
        geom = redl_geometry_from_wout(wout, surfaces, ntheta=self.ntheta,
                                       nphi=self.nphi, n_lambda=self.n_lambda)
        jr, _ = j_dot_B_redl(self.profiles, geom, self.helicity_n)
        s_full = jnp.linspace(0.0, 1.0, int(wout.ns))
        jv = jnp.interp(surfaces, s_full,
                        _as_1d(np.asarray(wout.jdotb, dtype=float)))
        return self._residual_vector(jv, jr)

    def profile(self, wout) -> jnp.ndarray:
        """Per-surface squared residuals (``sum = total``)."""
        r = self.residuals(wout)
        return r * r

    def total(self, wout) -> Array:
        """Scalar ``f_boot = sum(residuals**2)``."""
        r = self.residuals(wout)
        return jnp.sum(r * r)

    def J(self, eq) -> jnp.ndarray:
        """Objective-term entry point for :func:`~vmex.core.optimize.least_squares`."""
        return self.residuals(eq)

    __call__ = J  # the instance itself can be an objective term

    # -- traceable (state, runtime) evaluation --------------------------------

    def residuals_state(self, state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
        """Traceable residual vector (implicit lane; ``sum(r**2) = total_state``).

        Same ``R_j`` with ``Jr`` from :func:`redl_geometry_from_state` and
        ``Jv`` from :func:`vmec_j_dot_B` — one shared field-chain pass.
        Agrees with :meth:`residuals` at discretization level (solver
        internal grid + the identity ``Jv`` vs the 64x65 wout synthesis +
        jxbforce ``jdotb``).
        """
        surfaces = _as_1d(self.surfaces)
        hm = _half_mesh_fields(state, rt)
        geom = _geometry_from_half(hm, surfaces, n_lambda=self.n_lambda)
        jr, _ = j_dot_B_redl(self.profiles, geom, self.helicity_n)
        jv = _jv_from_half(hm, surfaces)
        return self._residual_vector(jv, jr)

    def profile_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable per-surface squared residuals."""
        r = self.residuals_state(state, rt)
        return r * r

    def total_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable scalar ``f_boot``."""
        r = self.residuals_state(state, rt)
        return jnp.sum(r * r)


# ===========================================================================
# Fixed-boundary Picard self-consistency loop (spec section 6.5, secondary lane)
# ===========================================================================


@dataclass(frozen=True)
class BootstrapPicardResult:
    """Outcome of :func:`self_consistent_bootstrap`.

    ``input`` carries the final ``AC``/``CURTOR`` (``ncurr=1``,
    ``pcurr_type="power_series"``); ``equilibrium`` is the last solved
    :class:`~vmex.core.optimize.Equilibrium`; ``history`` is one dict per
    iteration (``curtor``, ``delta``, ``f_boot``).  ``delta`` is the
    self-consistency error ``max|I'_Redl - I'_applied| / max|I'_Redl|``
    between the Redl-implied and the currently applied current profile.
    """

    input: VmecInput
    equilibrium: Any
    converged: bool
    iterations: int
    history: tuple


def _picard_dds_matrix(ns: int, ds: float) -> np.ndarray:
    """4th-order d/ds collocation matrix on ``linspace(0, 1, ns)`` (host).

    Verbatim port of the finite-difference matrix of the Zenodo
    ``convertSfincsToVmecCurrentProfile`` script (centered 5-point interior
    stencil, one-sided 5-point rows at both boundaries); requires
    ``ns >= 5``.
    """
    dds = (np.diag(1.0 / (12 * ds) * np.ones(ns - 2), -2)
           - np.diag(2.0 / (3 * ds) * np.ones(ns - 1), -1)
           + np.diag(2.0 / (3 * ds) * np.ones(ns - 1), 1)
           - np.diag(1.0 / (12 * ds) * np.ones(ns - 2), 2))
    dds[0, :5] = np.array([-25.0 / 12, 4.0, -3.0, 4.0 / 3, -1.0 / 4]) / ds
    dds[1, :5] = np.array([-1.0 / 4, -5.0 / 6, 3.0 / 2, -1.0 / 2, 1.0 / 12]) / ds
    dds[-1, -5:] = np.array([1.0 / 4, -4.0 / 3, 3.0, -4.0, 25.0 / 12]) / ds
    dds[-2, -5:] = np.array([-1.0 / 12, 1.0 / 2, -3.0 / 2, 5.0 / 6, 1.0 / 4]) / ds
    return dds


def self_consistent_bootstrap(
    inp: VmecInput,
    profiles: KineticProfiles,
    helicity_n: int,
    *,
    n_iter: int = 10,
    tol: float = 1e-3,
    relax: float = 1.0,
    degree: int = 12,
    s_eval=None,
    solve_kwargs: dict | None = None,
    verbose: bool = False,
) -> BootstrapPicardResult:
    """Fixed-boundary Picard iteration to a bootstrap-consistent current profile.

    The Zenodo ``convertSfincsToVmecCurrentProfile`` "smooth method" with
    the Redl ``<J.B>`` in place of SFINCS (spec section 6.5, secondary
    lane).  Host-side loop (no AD through it):

    1. solve the equilibrium (hot-restarted from the previous iterate);
    2. ``Jr = j_dot_B_redl(profiles, redl_geometry_from_wout(...))`` on the
       interior ``s_eval`` grid (default ``linspace(0.02, 0.98, 49)``),
       pinned to 0 at ``s = 0, 1`` and interpolated onto the full mesh;
    3. invert the section-6.2 identity for the enclosed current: solve
       ``[<B^2> d/ds + mu0 dp/ds] I = 2 pi psi_a Jr`` with ``I(0) = 0``
       (dense collocation solve, :func:`_picard_dds_matrix`), then recompute
       the *smooth* ``dI/ds = (2 pi psi_a Jr - mu0 I dp/ds)/<B^2>``;
    4. under-relax ``I' <- (1 - relax) I'_prev + relax I'_new``, refit the
       VMEC ``AC`` power series (``numpy.polynomial.polyfit``, ascending —
       ``pcurr_type="power_series"`` parameterizes ``I'``) and set
       ``CURTOR = I(1)``;
    5. stop when ``delta = max|I'_new - I'_applied| / max|I'_new| <= tol``.

    ``relax = 1.0`` is the plain fixed point (paper expectation: <= 5
    iterations at beta <= 2.5%); use ``relax = 0.5`` when the current
    dominates iota (e.g. tokamaks / the beta = 5% QH), where the
    ``<J.B> ~ 1/iota ~ 1/I`` feedback makes the undamped map marginally
    stable.  ``degree`` is clipped to ``ns - 2``.  The first solve uses
    ``inp``'s own current settings (any ``pcurr_type``); every refit
    switches to ``ncurr=1`` + ``power_series``.
    """
    from . import optimize as opt

    solve_kwargs = dict(solve_kwargs or {})
    if s_eval is None:
        s_eval = np.linspace(0.02, 0.98, 49)
    s_eval = np.atleast_1d(np.asarray(s_eval, dtype=float))
    relax = float(relax)
    if not 0.0 < relax <= 1.0:
        raise ValueError(f"relax must be in (0, 1], got {relax}")

    mismatch = RedlBootstrapMismatch(profiles, helicity_n,
                                     surfaces=np.clip(s_eval, 0.05, 0.95))
    history: list[dict] = []
    state = None
    dIds_applied = None   # I'(s_full) currently driving the equilibrium
    eq = None
    converged = False

    for it in range(int(n_iter)):
        try:
            eq = opt.solve_equilibrium(inp, initial_state=state, **solve_kwargs)
        except Exception:
            if state is None:
                raise
            eq = opt.solve_equilibrium(inp, **solve_kwargs)  # cold fallback
        state = eq.state
        w = eq.wout
        ns = int(w.ns)
        if ns < 5:
            raise ValueError("self_consistent_bootstrap requires ns >= 5")
        s_full = np.linspace(0.0, 1.0, ns)
        ds = s_full[1] - s_full[0]
        s_half = s_full[1:] - 0.5 * ds

        geom = redl_geometry_from_wout(w, s_eval)
        jr, _ = j_dot_B_redl(profiles, geom, helicity_n)
        jr_full = np.interp(s_full, np.concatenate([[0.0], s_eval, [1.0]]),
                            np.concatenate([[0.0], np.asarray(jr), [0.0]]))
        fsa_B2_full = np.interp(s_full, s_eval, np.asarray(geom.fsa_B2))

        pres = np.asarray(w.pres, dtype=float)[1:]                # Pa, half mesh
        dp_half = np.empty_like(pres)
        dp_half[1:-1] = (pres[2:] - pres[:-2]) / (2 * ds)
        dp_half[0] = (pres[1] - pres[0]) / ds
        dp_half[-1] = (pres[-1] - pres[-2]) / ds
        dpds = np.interp(s_full, s_half, dp_half)
        psi_a = float(np.asarray(w.phi)[-1]) / (2.0 * np.pi)

        dds = _picard_dds_matrix(ns, ds)
        matrix = (np.diag(fsa_B2_full) @ dds + np.diag(MU0 * dpds)) / (2 * np.pi * psi_a)
        matrix[0, :] = 0.0
        matrix[0, 0] = 1.0
        rhs = jr_full.copy()
        rhs[0] = 0.0
        I_solve = np.linalg.solve(matrix, rhs)
        dIds_new = (jr_full * 2 * np.pi * psi_a - MU0 * I_solve * dpds) / fsa_B2_full

        scale = max(float(np.max(np.abs(dIds_new))), np.finfo(float).tiny)
        delta = (np.inf if dIds_applied is None
                 else float(np.max(np.abs(dIds_new - dIds_applied))) / scale)
        f_boot = float(mismatch.total(w))
        history.append(dict(curtor=float(inp.curtor), delta=delta, f_boot=f_boot))
        if verbose:
            print(f"[self_consistent_bootstrap] it {it}: curtor={inp.curtor:.6e} "
                  f"delta={delta:.3e} f_boot={f_boot:.3e}")
        if delta <= float(tol):
            converged = True
            break

        dIds_use = (dIds_new if dIds_applied is None
                    else (1.0 - relax) * dIds_applied + relax * dIds_new)
        deg = int(min(int(degree), ns - 2))
        coeffs = np.polynomial.polynomial.polyfit(s_full, dIds_use, deg)
        ac = np.zeros(max(21, deg + 1))
        ac[:deg + 1] = coeffs
        curtor = float(np.trapezoid(dIds_use, s_full))
        inp = dataclasses.replace(inp, ncurr=1, pcurr_type="power_series",
                                  ac=ac, curtor=curtor)
        dIds_applied = dIds_use

    return BootstrapPicardResult(
        input=inp, equilibrium=eq, converged=converged,
        iterations=len(history), history=tuple(history))
