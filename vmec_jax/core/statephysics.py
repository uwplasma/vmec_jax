"""Shared state-physics primitives of the derived-quantity/objective modules.

One home (R26a consolidation) for the small private helpers that
:mod:`~vmec_jax.core.optimize`, :mod:`~vmec_jax.core.implicit`,
:mod:`~vmec_jax.core.bootstrap`, :mod:`~vmec_jax.core.stability` and
:mod:`~vmec_jax.core.omnigenity` all need — formerly byte-identical copies in
``optimize.py``/``implicit.py`` plus re-inlined recipes in ``stability.py``
(the shared module the bootstrap spec section 6.1b called ``_state_diag.py``):

- :func:`_field_chain` — geometry -> Jacobian -> metric -> fields -> energies
  of a core ``(SpectralState, SolverRuntime)`` pair, the evaluation chain
  behind every solver-native scalar target;
- :func:`_iotas_half` / :func:`_iotas_half_from_fields` — the ``ncurr``-aware
  half-mesh rotational transform (``add_fluxes.f90`` conventions);
- the **canonical wout-parity scalar targets** (plan_pre_vmex Item I.7):
  :func:`aspect_ratio` / :func:`volume` (``aspectratio.f`` boundary
  quadrature, equal to the wout ``aspect``/``volume_p`` scalars) and
  :func:`mean_iota` / :func:`edge_iota` (wout ``iotas``/``iotaf[-1]``
  conventions), re-exported unchanged by :mod:`~vmec_jax.core.optimize`.
  :mod:`~vmec_jax.core.implicit` keeps its historical ``aspect_ratio`` /
  ``plasma_volume`` variants (shoelace boundary area on a fresh grid /
  ``sum(vp)``) because :class:`~vmec_jax.core.implicit.ImplicitSolution`
  fields and the FD-cached gradient tables of ``tests/test_implicit_grad.py``
  pin those exact quadratures — the two families agree to quadrature
  resolution, and each module's docstrings cross-reference the other;
- the half-mesh radial sampling primitives :func:`_half_grid` /
  :func:`_interp_half_grid` and the wout-table utilities :func:`_as_1d` /
  :func:`_mode_matrix`;
- the ``L_grad_B`` primitives (plan_pre_vmex Item E): :func:`_lgradb_grid`
  (the pointwise magnetic-gradient scale length from wout-convention
  coefficient tables, pure jnp — shared by the wout-lane and traceable
  objectives) and :func:`_lgradb_state_tables` (those tables rebuilt
  *traceably* from ``(state, runtime)``: physical ``rmnc/zmns`` from the
  spectral state and the ``wrout.f`` Nyquist analysis of ``B^u/B^v`` as jnp
  einsums over host-constant trig tables).

This module sits directly above :mod:`vmec_jax.core.solver` in the import
graph; it must not import the objective modules.  ``optimize`` re-exports
these names for backward compatibility.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import numpy as np

import jax.numpy as jnp

from .fields import energies_and_force_norms, magnetic_fields, metric_elements
from .fourier import Resolution, mode_table, trig_tables
from .geometry import half_mesh_jacobian
from .residuals import m1_constrained_to_physical
from .solver import SolverRuntime, SpectralState, _geometry
from .transforms import physical_to_internal_scale

Array = Any


# ---------------------------------------------------------------------------
# Small array / wout-table utilities
# ---------------------------------------------------------------------------


def _as_1d(values, dtype=np.float64) -> jnp.ndarray:
    try:
        seq = list(values)  # type: ignore[arg-type]
    except TypeError:
        seq = [values]
    return jnp.asarray(np.asarray(seq, dtype=dtype))


def _half_grid(ns: int, dtype) -> jnp.ndarray:
    s_full = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    return 0.5 * (s_full[:-1] + s_full[1:])


def _interp_half_grid(samples: jnp.ndarray, surfaces: jnp.ndarray, s_half: jnp.ndarray) -> jnp.ndarray:
    """Linear interpolation of half-mesh radial samples onto ``surfaces``."""
    if int(s_half.shape[0]) == 1:
        return jnp.broadcast_to(samples[:1], (surfaces.shape[0],) + samples.shape[1:])
    idx_hi = jnp.clip(jnp.searchsorted(s_half, surfaces, side="left"), 1, s_half.shape[0] - 1)
    idx_lo = idx_hi - 1
    x0, x1 = s_half[idx_lo], s_half[idx_hi]
    denom = jnp.where(x1 != x0, x1 - x0, jnp.ones_like(x1))
    t = ((surfaces - x0) / denom).reshape((surfaces.shape[0],) + (1,) * (samples.ndim - 1))
    return samples[idx_lo] + t * (samples[idx_hi] - samples[idx_lo])


def _mode_matrix(wout, name: str, *, ns: int, mn: int, optional: bool = False) -> jnp.ndarray:
    """A ``(ns, mn)`` coefficient table from a wout-like object (either layout)."""
    value = getattr(wout, name, None)
    if value is None:
        if optional:
            return jnp.zeros((ns, mn), dtype=jnp.float64)
        raise AttributeError(f"wout-like object lacks required table {name!r}")
    arr = jnp.asarray(np.ascontiguousarray(np.asarray(value, dtype=np.float64)))
    if arr.shape == (ns, mn):
        return arr
    if arr.shape == (mn, ns):
        return arr.T
    raise ValueError(f"{name}: unexpected shape {arr.shape}, expected {(ns, mn)}")


# ---------------------------------------------------------------------------
# The state -> field-state evaluation chain and its derived profiles
# ---------------------------------------------------------------------------


def _field_chain(state: SpectralState, rt: SolverRuntime):
    """Geometry -> Jacobian -> metric -> fields -> energies of a core state."""
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
    energies = energies_and_force_norms(
        jacobian=jacobian, metrics=metrics, fields=fields, trig=rt.trig,
        s=s, signgs=setup.signgs,
    )
    return geometry, jacobian, metrics, fields, energies


def _iotas_half_from_fields(setup, fields) -> jnp.ndarray:
    """Half-mesh iota from an already-evaluated field state (``add_fluxes.f90``).

    ``ncurr = 0``: the prescribed profile; ``ncurr = 1``: reconstructed from
    the current-constrained ``chips`` of the field state (differentiable),
    exactly as the solver/wout writer do.  Index 0 is the (zeroed) axis slot.
    """
    if int(setup.ncurr) != 1:
        return jnp.asarray(setup.iotas)
    phips = jnp.asarray(setup.phips)
    safe = jnp.where(phips != 0.0, phips, 1.0)
    return jnp.where(phips != 0.0, jnp.asarray(fields.chips) / safe, 0.0)


def _iotas_half(state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
    """Half-mesh rotational transform of a core state (``add_fluxes.f90``)."""
    setup = rt.setup
    if int(setup.ncurr) != 1:
        return jnp.asarray(setup.iotas)
    _, _, _, fields, _ = _field_chain(state, rt)
    return _iotas_half_from_fields(setup, fields)


# ---------------------------------------------------------------------------
# Canonical wout-parity scalar targets (plan_pre_vmex Item I.7)
# ---------------------------------------------------------------------------


def _aspect_scalars(state: SpectralState, rt: SolverRuntime):
    """``aspectratio.f`` scalars ``(Aminor_p, Rmajor_p, aspect, volume_p)``.

    Boundary-surface quadrature identical to the wout writer
    (:func:`vmec_jax.core.postprocess.aspect_ratio_scalars`), kept in JAX.
    """
    geometry, _, _, _, _ = _field_chain(state, rt)
    sqrts_edge = jnp.asarray(rt.setup.sqrts)[-1]
    rb = jnp.asarray(geometry.R_even)[-1] + sqrts_edge * jnp.asarray(geometry.R_odd)[-1]
    zub = (jnp.asarray(geometry.dZ_dtheta_even)[-1]
           + sqrts_edge * jnp.asarray(geometry.dZ_dtheta_odd)[-1])
    wint = jnp.asarray(rt.trig.wint)
    t1 = rb * zub * wint
    volume_p = 2.0 * jnp.pi ** 2 * jnp.abs(jnp.sum(rb * t1))
    area = 2.0 * jnp.pi * jnp.abs(jnp.sum(t1))
    area_safe = jnp.where(area != 0.0, area, 1.0)
    aminor = jnp.sqrt(area_safe / jnp.pi)
    rmajor = volume_p / (2.0 * jnp.pi * area_safe)
    return aminor, rmajor, rmajor / aminor, volume_p


def aspect_ratio(state: SpectralState, rt: SolverRuntime) -> Array:
    """VMEC aspect ratio ``Rmajor_p / Aminor_p`` (``aspectratio.f`` convention).

    ``Aminor_p = sqrt(<cross-section area> / pi)``, ``Rmajor_p =
    volume_p / (2 pi <area>)`` from the boundary surface quadrature; equals
    the wout ``aspect`` scalar of the same state.  This is the canonical
    (wout-parity) implementation, re-exported as
    ``vmec_jax.core.optimize.aspect_ratio``;
    :func:`vmec_jax.core.implicit.aspect_ratio` is the implicit module's
    historical shoelace-quadrature variant of the same scalar (see there).
    """
    return _aspect_scalars(state, rt)[2]


def volume(state: SpectralState, rt: SolverRuntime) -> Array:
    """Plasma volume ``volume_p`` [m^3] (wout convention, boundary quadrature).

    Canonical (wout-parity) implementation, re-exported as
    ``vmec_jax.core.optimize.volume``;
    :func:`vmec_jax.core.implicit.plasma_volume` is the implicit module's
    ``sum(vp)`` variant of the same scalar (see there).
    """
    return _aspect_scalars(state, rt)[3]


def mean_iota(state: SpectralState, rt: SolverRuntime) -> Array:
    """Mean rotational transform over the half-mesh surfaces (axis excluded).

    Matches the legacy optimization ``mean_iota`` convention
    (``mean(iotas[1:])``, i.e. the mean of the wout ``iotas`` profile).
    """
    iotas = _iotas_half(state, rt)
    return jnp.mean(iotas[1:])


def edge_iota(state: SpectralState, rt: SolverRuntime) -> Array:
    """Rotational transform at the boundary (wout ``iotaf[-1]`` convention:
    linear extrapolation of the half mesh, ``1.5 iotas[-1] - 0.5 iotas[-2]``).

    Naming note (Item I.7): ``optimize.edge_iota`` and
    :func:`vmec_jax.core.implicit.iota_edge` are the same physical scalar —
    identical for ``ncurr = 1`` (both reconstruct iota from the converged
    ``chips``); at ``ncurr = 0`` this wout-parity version extrapolates the
    prescribed half-mesh ``iotas`` while the implicit variant evaluates the
    prescribed full-mesh ``iotaf`` endpoint directly.  ``iota_edge`` is
    provided as an alias here (and ``edge_iota`` in ``implicit``) so either
    spelling works in either module.
    """
    iotas = _iotas_half(state, rt)
    return 1.5 * iotas[-1] - 0.5 * iotas[-2]


iota_edge = edge_iota   # naming-flip alias (see the edge_iota docstring)


# ---------------------------------------------------------------------------
# Magnetic-gradient scale length L_grad_B (plan_pre_vmex Item E)
# ---------------------------------------------------------------------------


def _lgradb_grid(
    *,
    xm: jnp.ndarray,
    xn: jnp.ndarray,
    xm_nyq: jnp.ndarray,
    xn_nyq: jnp.ndarray,
    rmnc: jnp.ndarray,
    zmns: jnp.ndarray,
    bsupumnc: jnp.ndarray,
    bsupvmnc: jnp.ndarray,
    ns: int,
    nfp: int,
    s_index: int = -1,
    ntheta: int = 24,
    nphi: int = 24,
) -> jnp.ndarray:
    """Pointwise ``L_grad_B`` on one half-mesh surface, from wout-convention tables.

    ``L_grad_B = |B| sqrt(2 / (grad B : grad B))`` with ``grad B : grad B``
    the squared Frobenius norm of the Cartesian field-gradient tensor.  Inputs
    are the wout-normalized coefficient tables: full-mesh ``rmnc/zmns`` on the
    main modes ``(xm, xn)`` and half-mesh ``bsupumnc/bsupvmnc`` on the Nyquist
    modes ``(xm_nyq, xn_nyq)`` (``xn`` conventions include the ``nfp``
    factor).  ``B^u``/``B^v`` are synthesized spectrally, the coordinate basis
    vectors and their derivatives spectrally from ``rmnc/zmns``, and radial
    derivatives use the native half/full-mesh finite differences (exact
    half-mesh derivative of the full-mesh ``R/Z``; central — one-sided at the
    edges — differences of the half-mesh field).  Returns the
    ``(ntheta, nphi)`` array of ``L_grad_B`` values on a uniform
    ``(theta, phi)`` grid of the surface selected by ``s_index`` (indexing the
    ``ns``-long half-mesh arrays; default edge).  Pure jnp — shared by the
    wout-lane :func:`vmec_jax.core.optimize.l_grad_b` and the traceable
    :func:`vmec_jax.core.optimize.l_grad_b_state` (via
    :func:`_lgradb_state_tables`), which therefore agree to float round-off.
    Symmetric configurations only (lasym sine partners are ignored).
    """
    ns = int(ns)
    j = max(1, min(int(s_index) % ns, ns - 1))
    hs = 1.0 / (ns - 1)

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / int(nfp), int(nphi), endpoint=False)

    def tables(m, n):
        ang = (theta[:, None, None] * m[None, None, :]
               - phi[None, :, None] * n[None, None, :])
        return jnp.cos(ang), jnp.sin(ang)

    cosang, sinang = tables(xm, xn)

    def series(coeff, parity, second: bool = True):
        """Value + angular derivatives of a cos/sin(m theta - n phi) series."""
        base, alt = (cosang, sinang) if parity == "cos" else (sinang, cosang)
        s1 = -1.0 if parity == "cos" else 1.0
        val = jnp.einsum("m,tpm->tp", coeff, base)
        d_t = s1 * jnp.einsum("m,tpm,m->tp", coeff, alt, xm)
        d_p = -s1 * jnp.einsum("m,tpm,m->tp", coeff, alt, xn)
        if not second:
            return val, d_t, d_p, None, None, None
        d_tt = -jnp.einsum("m,tpm,m->tp", coeff, base, xm * xm)
        d_tp = jnp.einsum("m,tpm,m->tp", coeff, base, xm * xn)
        d_pp = -jnp.einsum("m,tpm,m->tp", coeff, base, xn * xn)
        return val, d_t, d_p, d_tt, d_tp, d_pp

    # Full-mesh R/Z -> half-mesh values + radial derivatives (exact on half mesh).
    R, Ru, Rv, Ruu, Ruv, Rvv = series(0.5 * (rmnc[j - 1] + rmnc[j]), "cos")
    Z, Zu, Zv, Zuu, Zuv, Zvv = series(0.5 * (zmns[j - 1] + zmns[j]), "sin")
    Rs, Rsu, Rsv, _, _, _ = series((rmnc[j] - rmnc[j - 1]) / hs, "cos", second=False)
    Zs, Zsu, Zsv, _, _, _ = series((zmns[j] - zmns[j - 1]) / hs, "sin", second=False)

    cphi, sphi = jnp.cos(phi)[None, :], jnp.sin(phi)[None, :]

    def cart(vR, vP, vZ):
        """Cylindrical (R, phi, Z) components -> Cartesian (x, y, z)."""
        return jnp.stack([vR * cphi - vP * sphi, vR * sphi + vP * cphi, vZ], axis=-1)

    zero = jnp.zeros_like(R)
    e_s, e_u, e_v = cart(Rs, zero, Zs), cart(Ru, zero, Zu), cart(Rv, R, Zv)
    # d(e_u)/du, d(e_u)/dv=d(e_v)/du, d(e_v)/dv, d(e_u)/ds, d(e_v)/ds
    deu_u = cart(Ruu, zero, Zuu)
    deu_v = cart(Ruv, Ru, Zuv)
    dev_v = cart(Rvv - R, 2.0 * Rv, Zvv)
    deu_s = cart(Rsu, zero, Zsu)
    dev_s = cart(Rsv, Rs, Zsv)

    # Half-mesh contravariant field (Nyquist modes) + radial derivative.
    cosn, sinn = tables(xm_nyq, xn_nyq)

    def nyq(coeff):
        return (jnp.einsum("m,tpm->tp", coeff, cosn),
                -jnp.einsum("m,tpm,m->tp", coeff, sinn, xm_nyq),
                jnp.einsum("m,tpm,m->tp", coeff, sinn, xn_nyq))

    bu, bu_t, bu_p = nyq(bsupumnc[j])
    bv, bv_t, bv_p = nyq(bsupvmnc[j])
    lo, hi = (j - 1, j + 1) if 1 < j < ns - 1 else ((j, j + 1) if j == 1 else (j - 1, j))
    span = hs * (hi - lo)
    bu_s, _, _ = nyq((bsupumnc[hi] - bsupumnc[lo]) / span)
    bv_s, _, _ = nyq((bsupvmnc[hi] - bsupvmnc[lo]) / span)

    B = bu[..., None] * e_u + bv[..., None] * e_v
    dB = jnp.stack([
        bu_s[..., None] * e_u + bv_s[..., None] * e_v
        + bu[..., None] * deu_s + bv[..., None] * dev_s,
        bu_t[..., None] * e_u + bv_t[..., None] * e_v
        + bu[..., None] * deu_u + bv[..., None] * deu_v,
        bu_p[..., None] * e_u + bv_p[..., None] * e_v
        + bu[..., None] * deu_v + bv[..., None] * dev_v,
    ], axis=-2)                                        # (t, p, coord, cart)

    basis = jnp.stack([e_s, e_u, e_v], axis=-2)
    g = jnp.einsum("...ic,...jc->...ij", basis, basis)
    ginv = jnp.linalg.inv(g)
    grad_sq = jnp.einsum("...ic,...ij,...jc->...", dB, ginv, dB)
    tiny = jnp.asarray(jnp.finfo(grad_sq.dtype).tiny, dtype=grad_sq.dtype)
    bmag = jnp.sqrt(jnp.maximum(jnp.sum(B * B, axis=-1), tiny))
    return bmag * jnp.sqrt(2.0 / jnp.maximum(grad_sq, tiny))


@functools.lru_cache(maxsize=None)
def _nyquist_analysis_constants(res: Resolution):
    """Host-NumPy ``wrout.f`` Nyquist-analysis constants for a resolution.

    The wout engine analyzes the internal-grid fields on the Nyquist-extended
    trig tables (:func:`vmec_jax.core.wout.wout_from_state`); this rebuilds
    the identical mode table and integration-weighted trig products —
    :func:`vmec_jax.core.nyquist._wrout_theta_tables` /
    ``_wrout_zeta_tables`` / ``_wrout_dmult`` on
    ``mode_table(max(ntheta1/2, mpol-1) + 1, max(nzeta/2, ntor))`` — as
    static NumPy constants so :func:`_wrout_cos_coeffs_state` can run the
    same analysis as traceable jnp einsums.  Cached per (hashable)
    :class:`~vmec_jax.core.fourier.Resolution`.
    """
    from .nyquist import _wrout_dmult, _wrout_theta_tables, _wrout_zeta_tables

    mnyq = max(res.ntheta1 // 2, res.mpol - 1)
    nnyq = max(res.nzeta // 2, res.ntor)
    trig_nyq = trig_tables(dataclasses.replace(res, mpol=mnyq + 1, ntor=nnyq))
    modes_nyq = mode_table(mnyq + 1, nnyq)
    cosmui, sinmui = _wrout_theta_tables(trig_nyq)
    cosnv, sinnv = _wrout_zeta_tables(trig_nyq)
    m_idx = np.asarray(modes_nyq.m, dtype=int)
    n_idx = np.abs(np.asarray(modes_nyq.n, dtype=int))
    sgn = np.where(np.asarray(modes_nyq.n, dtype=int) < 0, -1.0, 1.0)
    dmult = _wrout_dmult(modes_nyq, trig_nyq)
    nt2 = int(trig_nyq.ntheta2)
    return modes_nyq, (cosmui, sinmui, cosnv, sinnv, m_idx, n_idx, sgn, dmult, nt2)


def _wrout_cos_coeffs_state(f: jnp.ndarray, consts) -> jnp.ndarray:
    """Traceable ``wrout.f`` Nyquist cosine analysis, shape ``(ns, mnmax_nyq)``.

    The jnp mirror of :func:`vmec_jax.core.nyquist.wrout_cos_coeffs`
    (identical math, host-constant tables from
    :func:`_nyquist_analysis_constants`), so gradients flow through the
    analyzed internal-grid field ``f``.
    """
    cosmui, sinmui, cosnv, sinnv, m_idx, n_idx, sgn, dmult, nt2 = consts
    f = jnp.asarray(f)[:, :nt2, :]
    f_theta_cos = jnp.einsum("sik,im->smk", f, jnp.asarray(cosmui))
    f_theta_sin = jnp.einsum("sik,im->smk", f, jnp.asarray(sinmui))
    cos_zeta = jnp.einsum("smk,kn->smn", f_theta_cos, jnp.asarray(cosnv))
    sin_zeta = jnp.einsum("smk,kn->smn", f_theta_sin, jnp.asarray(sinnv))
    coeff = cos_zeta[:, m_idx, n_idx] + jnp.asarray(sgn)[None, :] * sin_zeta[:, m_idx, n_idx]
    return coeff * jnp.asarray(dmult)[None, :]


def _lgradb_state_tables(state: SpectralState, rt: SolverRuntime) -> dict:
    """Traceable wout-convention ``L_grad_B`` input tables from ``(state, rt)``.

    Rebuilds exactly the tables the wout engine feeds the wout-lane
    :func:`vmec_jax.core.optimize.l_grad_b`, fully in jnp:

    - ``rmnc/zmns``: the m=1 constraint undone
      (:func:`~vmec_jax.core.residuals.m1_constrained_to_physical`) and the
      internal ``mscale*nscale`` normalization multiplied back
      (``wout_from_state``'s ``mode_scale``);
    - ``bsupumnc/bsupvmnc``: the half-mesh contravariant field of
      :func:`_field_chain` pushed through the ``wrout.f`` Nyquist cosine
      analysis (:func:`_wrout_cos_coeffs_state`).  The (never-used) axis row
      is zeroed first so no axis-slot garbage can leak into reverse-mode AD.

    Returns the keyword dict for :func:`_lgradb_grid` (minus the sampling
    controls).  Symmetric configurations only (``lasym = False``).
    """
    setup = rt.setup
    if bool(setup.lasym):
        raise NotImplementedError(
            "l_grad_b_state supports lasym = False only (the wout-lane "
            "l_grad_b ignores the lasym sine partners too)")
    res = rt.resolution
    nfp = int(res.nfp)
    ns = int(np.shape(state.R_cos)[0])

    R_cos_p, Z_sin_p, _, _ = m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos,
        modes=rt.modes, lthreed=bool(setup.lthreed), lasym=bool(setup.lasym),
        lconm1=bool(setup.lconm1),
    )
    mode_scale = 1.0 / physical_to_internal_scale(rt.modes, rt.trig)
    rmnc = jnp.asarray(R_cos_p) * jnp.asarray(mode_scale)[None, :]
    zmns = jnp.asarray(Z_sin_p) * jnp.asarray(mode_scale)[None, :]

    _, _, _, fields, _ = _field_chain(state, rt)
    modes_nyq, consts = _nyquist_analysis_constants(res)
    bsupu = jnp.asarray(fields.bsupu).at[0].set(0.0)
    bsupv = jnp.asarray(fields.bsupv).at[0].set(0.0)
    bsupumnc = _wrout_cos_coeffs_state(bsupu, consts)
    bsupvmnc = _wrout_cos_coeffs_state(bsupv, consts)

    return dict(
        xm=jnp.asarray(np.asarray(rt.modes.m, dtype=float)),
        xn=jnp.asarray(np.asarray(rt.modes.n, dtype=float) * nfp),
        xm_nyq=jnp.asarray(np.asarray(modes_nyq.m, dtype=float)),
        xn_nyq=jnp.asarray(np.asarray(modes_nyq.n, dtype=float) * nfp),
        rmnc=rmnc, zmns=zmns, bsupumnc=bsupumnc, bsupvmnc=bsupvmnc,
        ns=ns, nfp=nfp,
    )
