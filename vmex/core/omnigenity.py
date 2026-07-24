"""Traceable omnigenity/QI objective (R26h.h2).

A quasi-isodynamic (QI) omnigenity residual evaluated as a pure, traceable
function of a converged ``(SpectralState, SolverRuntime)`` pair, so it
composes with the implicit-gradient least-squares lane
(``vmex.core.optimize.least_squares(..., jac="implicit")``) exactly like
:class:`~vmex.core.optimize.QuasisymmetryRatioResidual` — no wout tables,
no host booz_xform round-trip.

Two pieces:

1. :func:`boozer_bmnc_state` — a minimal *traceable* Boozer ``|B|`` transform.
   The classic BOOZ_XFORM construction (Hirshman's Fortran code; equation
   numbers below follow Landreman's booz_xform documentation) is evaluated
   in pure ``jax.numpy`` from the solver's internal half-mesh field tables:
   the periodic part of the Boozer generating potential ``w`` is integrated
   spectrally from the covariant field components (``dw/dtheta = B_theta``,
   ``dw/dzeta = B_zeta``), then ``nu = (w - I*lambda) / (G + iota*I)`` gives
   the Boozer angles ``theta_B = theta + lambda + iota*nu``,
   ``zeta_B = zeta + nu``, and the Boozer ``|B|`` harmonics come from the
   angle-transform quadrature ``bmnc_b = <|B| cos(m theta_B - n zeta_B) *
   d(theta_B, zeta_B)/d(theta, zeta)>`` — the same equations booz_xform
   solves, here end-to-end differentiable.

2. :func:`omnigenity_residual` / :class:`QIResidual` — a smooth omnigenity
   distance for poloidally-closed-contour (``M = 0, N = 1``) omnigenity.
   The formulation is the constructed-QI-target distance of
   **Goodman et al., "Constructing precisely quasi-isodynamic magnetic
   fields", J. Plasma Phys. 89, 905890504 (2023), arXiv:2211.09829**,
   distilled to its level-set form: on each surface, ``|B|`` is sampled along
   Boozer field lines ``theta_B = alpha + iota * phi_B`` over one field
   period and the residual stacks, per surface,

   - **bounce-distance uniformity** (``well_weight``): for every trapping
     level ``B*``, the bounce distance ``delta(alpha, B*)`` between the two
     monotone branches of the magnetic well (smooth occupancy integrals of
     the running-maximum branch envelopes) minus its field-line average —
     the Cary–Shasharina omnigenity condition (Cary & Shasharina, PRL 78,
     674 (1997)) that Goodman's "shuffle" step enforces;
   - **extremum alignment** (``extremum_weight``): the per-field-line
     ``B_min``/``B_max`` minus their field-line averages — poloidal closure
     of the extremal ``|B|`` contours (Goodman's "align the maxima" step;
     also the flat-``B_max`` condition of Dudt et al., J. Plasma Phys. 90,
     905900120 (2024), arXiv:2305.08026);
   - **single-well monotonicity** (``squash_weight``): the pointwise distance
     between ``|B|`` and its monotone branch envelopes — Goodman's "squash"
     distance, penalizing side extrema (multiple wells per period).

   Each piece is an exact zero of an exactly QI field, every operation is
   smooth or piecewise-smooth (sigmoid occupancies, running maxima), and the
   full pipeline is jit/grad/jvp-transparent for the implicit lane.

Scope notes
-----------
- Stellarator-symmetric states only (``lasym = False``), like the other
  traceable objectives.
- Requested surfaces are snapped to the *nearest half-mesh surface* (the
  Boozer transform is a per-surface construction — same convention as
  :func:`vmex.core.optimize.boozer_modes_from_wout`), not interpolated.
- The wout-engine analogue for cross-checks is
  :func:`vmex.core.optimize.quasi_isodynamic_residual_from_wout`
  (host booz_xform_jax, finite-difference-only).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import jax
import jax.numpy as jnp

from .fields import magnetic_fields, metric_elements, surface_currents
from .geometry import half_mesh_jacobian
from .solver import SolverRuntime, SpectralState, _geometry
from .statephysics import _as_1d, _iotas_half_from_fields
from .transforms import physical_to_internal_scale

__all__ = [
    "boozer_bmnc_state",
    "omnigenity_residual",
    "QIResidual",
]

Array = Any


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _pad_spectrum_axis(hat: jnp.ndarray, axis: int, n_fine: int) -> jnp.ndarray:
    """Zero-pad an FFT spectrum along ``axis`` (Nyquist split, real-safe).

    Standard trigonometric interpolation: positive/negative frequency blocks
    are copied, an even-length Nyquist coefficient is split half-and-half
    onto ``+N/2`` and ``-N/2`` of the fine spectrum.  The caller rescales by
    ``n_fine / n_coarse`` per padded axis (ifft normalization).
    """
    n = int(hat.shape[axis])
    if n_fine == n:
        return hat
    if n_fine < n:
        raise ValueError(f"padding target {n_fine} smaller than source {n}")

    def take(sl):
        index = [slice(None)] * hat.ndim
        index[axis] = sl
        return hat[tuple(index)]

    n_pos = (n + 1) // 2                      # frequencies 0 .. ceil(n/2)-1
    shape = list(hat.shape)
    shape[axis] = n_fine
    out = jnp.zeros(shape, dtype=hat.dtype)
    index = [slice(None)] * hat.ndim
    index[axis] = slice(0, n_pos)
    out = out.at[tuple(index)].set(take(slice(0, n_pos)))
    if n % 2 == 0:                            # split the Nyquist mode
        nyq = 0.5 * take(slice(n // 2, n // 2 + 1))
        index[axis] = slice(n // 2, n // 2 + 1)
        out = out.at[tuple(index)].set(nyq)
        index[axis] = slice(n_fine - n // 2, n_fine - n // 2 + 1)
        out = out.at[tuple(index)].set(nyq)
        n_neg = n // 2 - 1                    # strictly negative, non-Nyquist
    else:
        n_neg = n // 2
    if n_neg > 0:
        index[axis] = slice(n_fine - n_neg, n_fine)
        out = out.at[tuple(index)].set(take(slice(n - n_neg, n)))
    return out


def _pad_spectrum(hat: jnp.ndarray, nt_fine: int, nz_fine: int) -> jnp.ndarray:
    """Zero-pad a 2D FFT spectrum ``(..., ntheta, nzeta)`` with rescaling."""
    nt, nz = int(hat.shape[-2]), int(hat.shape[-1])
    out = _pad_spectrum_axis(hat, hat.ndim - 2, nt_fine)
    out = _pad_spectrum_axis(out, hat.ndim - 1, nz_fine)
    return out * (float(nt_fine * nz_fine) / float(nt * nz))


def _mirror_maps(ntheta2: int, nzeta: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Stellarator-symmetry mirror of the reduced ``[0, pi]`` theta grid.

    Same map as ``QuasisymmetryRatioResidual._pointwise_state``:
    ``X(2 pi - theta, -zeta) = X(theta, zeta)`` for even (cos-series) fields.
    """
    ntheta1 = max(2 * (ntheta2 - 1), 1)
    i_full = np.arange(ntheta1)
    i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
    k = np.arange(nzeta)
    k_src = np.where(i_full[:, None] < ntheta2, k[None, :], (nzeta - k[None, :]) % nzeta)
    i_src = np.broadcast_to(i_src[:, None], (ntheta1, nzeta))
    return ntheta1, i_src, k_src


def _lambda_half_weights(ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Odd-m half-mesh interpolation weights ``(smw, spw)`` per half row.

    ``lambda_wout_from_full_mesh`` (``wrout.f``) interpolates odd-m lambda
    coefficients to half row ``js`` (between full ``js-1`` and ``js``) as
    ``0.5 * (smw[js] * lam[js] + spw[js] * lam[js-1])`` with the
    ``sqrt(s)``-representation weights below; even m uses plain averaging.
    """
    hs = 1.0 / (ns - 1)
    js = np.arange(ns, dtype=float)
    shalf = np.sqrt(hs * np.abs(js - 0.5))            # half surface js
    sqrts = np.sqrt(hs * js)
    smw = np.zeros(ns)
    spw = np.zeros(ns)
    smw[1:] = shalf[1:] / sqrts[1:]
    spw[2:] = shalf[2:] / sqrts[1:-1]
    if ns > 1:
        spw[1] = smw[1]                               # wrout.f: sp(1) = sm(2)
    return smw, spw


# ---------------------------------------------------------------------------
# Traceable Boozer |B| spectrum from a core state
# ---------------------------------------------------------------------------


def boozer_bmnc_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces,
    mboz: int = 16,
    nboz: int = 16,
    oversample: int = 2,
) -> dict[str, Array]:
    """Boozer ``|B|`` cosine spectrum of selected surfaces, fully traceable.

    The jnp analogue of :func:`vmex.core.optimize.boozer_modes_from_wout`
    (booz_xform algorithm, module docstring) evaluated from the solver's
    internal tables: ``|B|``, the covariant components ``B_theta``/``B_zeta``
    and the Boozer profile averages ``I``/``G`` live on the half-mesh
    ``(theta, zeta)`` grid (mirrored to the full theta circle), lambda comes
    from the state's spectral coefficients with the exact wout half-mesh
    rescale, and the generating potential ``w`` is integrated spectrally
    (FFT) — ``m != 0`` modes from ``B_theta``, ``m = 0`` modes from
    ``B_zeta``, the booz_xform mode split.

    ``surfaces`` are normalized-flux values snapped to the nearest half-mesh
    surfaces (one Boozer construction per requested value, duplicates kept so
    outputs align with ``surfaces``).  ``oversample`` refines the quadrature
    grid by trigonometric (FFT zero-pad) interpolation before the Boozer
    angle transform, reducing the aliasing of ``cos(m theta_B - n zeta_B)``
    products; ``mboz``/``nboz`` are capped at the fine grid's Nyquist.

    Returns ``{bmnc_b (nsurf, nmodes), xm_b, xn_b (physical), iota_b, nfp,
    s_b}`` — ``bmnc_b/xm_b/xn_b/iota_b/nfp`` are the spectrum inputs of
    :func:`omnigenity_residual` and of
    :func:`vmex.core.optimize.quasi_isodynamic_residual`.
    """
    setup = rt.setup
    if bool(setup.lasym):
        raise NotImplementedError(
            "boozer_bmnc_state supports stellarator-symmetric states only "
            "(lasym = False)")
    if int(oversample) < 1:
        raise ValueError("oversample must be >= 1")
    s = jnp.asarray(setup.s_full)
    ns = int(s.shape[0])
    if ns < 3:
        raise ValueError(f"boozer_bmnc_state needs ns >= 3, got ns = {ns}")
    nfp = int(rt.resolution.nfp)
    dtype = s.dtype

    # -- surface selection: nearest half-mesh rows (static, shape-only) -----
    s_half_np = (np.arange(ns - 1) + 0.5) / (ns - 1)
    surf_np = np.atleast_1d(np.asarray(list(np.ravel(surfaces)), dtype=float))
    if surf_np.size == 0:
        raise ValueError("surfaces must be non-empty")
    rows = np.asarray([int(np.argmin(np.abs(s_half_np - v))) + 1 for v in surf_np])

    # -- half-mesh field tables (the QS-residual field chain) ----------------
    _, geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    iota_prof = _iotas_half_from_fields(setup, fields)
    cur = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv,
                           trig=rt.trig, s=s, signgs=setup.signgs)
    G_prof, I_prof = jnp.asarray(cur.bvco), jnp.asarray(cur.buco)

    # |B| on the half mesh (bcovar.f: bsq = |B|^2/2 + p).
    bsq2 = 2.0 * (jnp.asarray(fields.total_pressure)
                  - jnp.asarray(fields.pressure)[:, None, None])
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)

    # -- select rows, mirror the reduced theta grid to the full circle -------
    ntheta2 = int(np.shape(fields.total_pressure)[1])
    nzeta = int(np.shape(fields.total_pressure)[2])
    ntheta1, i_src, k_src = _mirror_maps(ntheta2, nzeta)

    def full(a):
        return jnp.asarray(a)[rows][:, i_src, k_src]

    bmag = jnp.sqrt(jnp.maximum(full(bsq2), tiny))          # (nsurf, nt1, nz)
    bsubu = full(fields.bsubu)
    bsubv = full(fields.bsubv)
    iota = iota_prof[rows]
    G = G_prof[rows]
    I = I_prof[rows]  # noqa: E741 - Boozer I

    # -- physical lambda modes on the selected half surfaces ------------------
    # wrout.f / lambda_wout_from_full_mesh: internal full-mesh L_sin ->
    # wout-normalized modes, lamscale/phipf rescale, parity-weighted half-mesh
    # interpolation (plain average for even m, sqrt(s)-representation weights
    # for odd m).
    m_modes = np.asarray(rt.modes.m, dtype=int)
    xn_modes = np.asarray(rt.modes.n, dtype=float) * float(nfp)
    mode_scale = jnp.asarray(1.0 / physical_to_internal_scale(rt.modes, rt.trig))
    phipf = jnp.asarray(setup.phipf)
    safe_phipf = jnp.where(phipf != 0.0, phipf, 1.0)
    lam_full = (jnp.asarray(state.L_sin) * mode_scale[None, :]
                * (jnp.asarray(setup.lamscale) / safe_phipf)[:, None])
    smw_np, spw_np = _lambda_half_weights(ns)
    even = (m_modes % 2 == 0)
    hi_w = np.where(even[None, :], 1.0, smw_np[rows][:, None])
    lo_w = np.where(even[None, :], 1.0, spw_np[rows][:, None])
    lam_mn = 0.5 * (jnp.asarray(hi_w) * lam_full[rows]
                    + jnp.asarray(lo_w) * lam_full[rows - 1])   # (nsurf, mn)

    # -- generating potential w (periodic part) by spectral integration ------
    # dw/dtheta = B_theta, dw/dzeta = B_zeta (physical toroidal angle; the
    # grid spans one field period, so the zeta wavenumbers carry nfp).
    kt_c = np.fft.fftfreq(ntheta1) * ntheta1
    kz_c = np.fft.fftfreq(nzeta) * nzeta * nfp
    bu_hat = jnp.fft.fft2(bsubu, axes=(1, 2))
    bv_hat = jnp.fft.fft2(bsubv, axes=(1, 2))
    kt2 = jnp.asarray(kt_c)[None, :, None]
    kz2 = jnp.asarray(kz_c)[None, None, :]
    w_hat = jnp.where(
        kt2 != 0.0, bu_hat / jnp.where(kt2 != 0.0, 1j * kt2, 1.0),
        jnp.where(kz2 != 0.0, bv_hat / jnp.where(kz2 != 0.0, 1j * kz2, 1.0), 0.0))

    # -- fine quadrature grid (trigonometric interpolation) ------------------
    nt_f = int(oversample) * ntheta1
    nz_f = int(oversample) * nzeta if nzeta > 1 else 1
    kt_f = jnp.asarray(np.fft.fftfreq(nt_f) * nt_f)[None, :, None]
    kz_f = jnp.asarray(np.fft.fftfreq(nz_f) * nz_f * nfp)[None, None, :]
    w_hat_f = _pad_spectrum(w_hat, nt_f, nz_f)
    w = jnp.real(jnp.fft.ifft2(w_hat_f, axes=(1, 2)))
    dw_dth = jnp.real(jnp.fft.ifft2(1j * kt_f * w_hat_f, axes=(1, 2)))
    dw_dze = jnp.real(jnp.fft.ifft2(1j * kz_f * w_hat_f, axes=(1, 2)))
    bmod = jnp.real(jnp.fft.ifft2(
        _pad_spectrum(jnp.fft.fft2(bmag, axes=(1, 2)), nt_f, nz_f), axes=(1, 2)))

    theta_f = jnp.asarray(2.0 * np.pi * np.arange(nt_f) / nt_f, dtype=dtype)
    zeta_f = jnp.asarray(2.0 * np.pi * np.arange(nz_f) / (nz_f * nfp), dtype=dtype)
    ang = (theta_f[:, None, None] * jnp.asarray(m_modes, dtype=dtype)
           - zeta_f[None, :, None] * jnp.asarray(xn_modes, dtype=dtype))
    sin_tab, cos_tab = jnp.sin(ang), jnp.cos(ang)           # (nt_f, nz_f, mn)
    lam = jnp.einsum("sm,tzm->stz", lam_mn, sin_tab)
    dlam_dth = jnp.einsum("sm,tzm->stz", lam_mn * jnp.asarray(m_modes, dtype=dtype), cos_tab)
    dlam_dze = -jnp.einsum("sm,tzm->stz", lam_mn * jnp.asarray(xn_modes, dtype=dtype), cos_tab)

    # -- Boozer angles + transform Jacobian (booz_xform eqs. (3), (10), (12))
    GI = G + iota * I
    GI_safe = jnp.where(GI != 0.0, GI, 1.0)
    one_over_GI = (1.0 / GI_safe)[:, None, None]
    nu = one_over_GI * (w - I[:, None, None] * lam)
    dnu_dth = one_over_GI * (dw_dth - I[:, None, None] * dlam_dth)
    dnu_dze = one_over_GI * (dw_dze - I[:, None, None] * dlam_dze)
    theta_B = theta_f[None, :, None] + lam + iota[:, None, None] * nu
    zeta_B = zeta_f[None, None, :] + nu
    jac_fac = ((1.0 + dlam_dth) * (1.0 + dnu_dze)
               + (iota[:, None, None] - dlam_dze) * dnu_dth)

    # -- Boozer mode list + separable Fourier quadrature ----------------------
    mboz = int(min(mboz, max(nt_f // 2 - 1, 0)))
    nboz = int(min(nboz, max((nz_f - 1) // 2, 0)))
    m_list = [0] * (nboz + 1) + [m for m in range(1, mboz + 1) for _ in range(2 * nboz + 1)]
    n_list = list(range(nboz + 1)) + [n for _ in range(1, mboz + 1)
                                      for n in range(-nboz, nboz + 1)]
    xm_b = np.asarray(m_list, dtype=float)
    xn_b = np.asarray(n_list, dtype=float) * float(nfp)

    marr = jnp.asarray(np.arange(mboz + 1), dtype=dtype)
    karr = jnp.asarray(np.arange(nboz + 1), dtype=dtype) * float(nfp)
    cosm = jnp.cos(theta_B[..., None] * marr)               # (nsurf, nt, nz, mb+1)
    sinm = jnp.sin(theta_B[..., None] * marr)
    cosn = jnp.cos(zeta_B[..., None] * karr)                # (nsurf, nt, nz, nb+1)
    sinn = jnp.sin(zeta_B[..., None] * karr)
    F = bmod * jac_fac
    Xc = jnp.einsum("stzm,stz,stzk->smk", cosm, F, cosn)
    Xs = jnp.einsum("stzm,stz,stzk->smk", sinm, F, sinn)
    m_idx = np.asarray(m_list, dtype=int)
    k_idx = np.abs(np.asarray(n_list, dtype=int))
    sgn = jnp.asarray(np.where(np.asarray(n_list) < 0, -1.0, 1.0), dtype=dtype)
    ff = np.full(len(m_list), 2.0 / (nt_f * nz_f))
    ff[0] = 1.0 / (nt_f * nz_f)                             # the (0, 0) mode
    # cos(m th_B - n ze_B) = cos(m th_B) cos(|n| ze_B) + sgn(n) sin(m th_B) sin(|n| ze_B)
    bmnc_b = jnp.asarray(ff) * (Xc[:, m_idx, k_idx] + sgn[None, :] * Xs[:, m_idx, k_idx])

    return {
        "bmnc_b": bmnc_b,
        "xm_b": xm_b,
        "xn_b": xn_b,
        "iota_b": iota,
        "buco_b": I,
        "bvco_b": G,
        "gi_b": GI,
        "nfp": nfp,
        "s_b": jnp.asarray(s_half_np, dtype=dtype)[rows - 1],
    }


# ---------------------------------------------------------------------------
# Omnigenity residual on Boozer |B| harmonics
# ---------------------------------------------------------------------------


def omnigenity_residual(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    weights: Iterable[float] | None = None,
    nphi: int = 97,
    nalpha: int = 25,
    n_levels: int = 16,
    softness: float = 2.0e-2,
    well_weight: float = 1.0,
    extremum_weight: float = 1.0,
    squash_weight: float = 1.0,
) -> dict[str, Array]:
    """Smooth constructed-QI-target omnigenity residual (module docstring).

    ``|B|`` is synthesized from Boozer harmonics along field lines
    ``theta_B = alpha + iota * phi_B`` on ``nalpha`` labels over one field
    period (``nphi`` periodic points), normalized per surface to ``[0, 1]``
    by the surface extrema.  Each field line is split at its minimum into
    two monotone branch envelopes (periodic running maxima); the residual
    stacks the three Goodman-construction distances:

    - ``well``: bounce distance ``delta(alpha, B*) = d_left + d_right``
      (sigmoid occupancy integrals of the branch envelopes at ``n_levels``
      trapping levels, in field-period fraction units) minus its
      ``alpha``-average — Cary–Shasharina bounce-distance omnigenity;
    - ``extremum``: per-line min/max of ``|B|`` minus their
      ``alpha``-averages — poloidally closed extremal contours;
    - ``squash``: pointwise ``envelope - |B|`` monotonicity defect —
      one magnetic well per field period.

    All three vanish on an exactly QI field.  ``softness`` is the sigmoid
    level width in normalized ``|B|`` units.  Returns ``residuals1d`` (flat
    least-squares vector), ``total = sum(residuals1d**2)`` and diagnostics.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    xm_b = jnp.asarray(np.asarray(xm_b, dtype=float))
    xn_b = jnp.asarray(np.asarray(xn_b, dtype=float))
    iota_b = jnp.atleast_1d(jnp.asarray(iota_b, dtype=jnp.float64))
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if nphi < 8 or nalpha < 2 or n_levels < 2:
        raise ValueError("omnigenity residual needs nphi >= 8, nalpha >= 2, n_levels >= 2")
    nsurf = int(bmnc_b.shape[0])
    dtype = bmnc_b.dtype
    w_arr = jnp.ones((nsurf,), dtype=dtype) if weights is None else _as_1d(weights)
    if int(w_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as surfaces")
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    eps = jnp.maximum(jnp.asarray(float(softness), dtype=dtype),
                      jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))

    # -- |B| along field lines over one field period (periodic phi grid) -----
    period = 2.0 * np.pi / float(nfp)
    phi = jnp.asarray(period * np.arange(nphi) / nphi, dtype=dtype)
    alpha = jnp.asarray(2.0 * np.pi * np.arange(nalpha) / nalpha, dtype=dtype)
    theta = alpha[None, :, None] + iota_b[:, None, None] * phi[None, None, :]
    angle = (theta[..., None] * xm_b - phi[None, None, :, None] * xn_b)
    b = jnp.einsum("sapm,sm->sap", jnp.cos(angle), bmnc_b)   # (nsurf, nalpha, nphi)

    bmin = jnp.min(b, axis=(1, 2), keepdims=True)
    bmax = jnp.max(b, axis=(1, 2), keepdims=True)
    bhat = (b - bmin) / jnp.maximum(bmax - bmin, tiny)

    # -- monotone branch envelopes about the per-line minimum ----------------
    imin = jnp.argmin(bhat, axis=-1)                          # (nsurf, nalpha)
    nk = nphi // 2 + 1
    offs = jnp.arange(nk, dtype=jnp.int32)
    idx_l = jnp.mod(imin[:, :, None] - offs[None, None, :], nphi)
    idx_r = jnp.mod(imin[:, :, None] + offs[None, None, :], nphi)
    raw_l = jnp.take_along_axis(bhat, idx_l, axis=-1)         # (nsurf, nalpha, nk)
    raw_r = jnp.take_along_axis(bhat, idx_r, axis=-1)
    env_l = jax.lax.cummax(raw_l, axis=raw_l.ndim - 1)
    env_r = jax.lax.cummax(raw_r, axis=raw_r.ndim - 1)

    sqrt_w = jnp.sqrt(w_arr)[:, None, None]
    pieces: list[jnp.ndarray] = []

    # -- bounce-distance uniformity (Cary-Shasharina / Goodman "shuffle") ----
    levels = jnp.linspace(0.0, 1.0, int(n_levels) + 2, dtype=dtype)[1:-1]
    occ_l = jax.nn.sigmoid((levels[None, None, None, :] - env_l[..., None]) / eps)
    occ_r = jax.nn.sigmoid((levels[None, None, None, :] - env_r[..., None]) / eps)
    delta = (jnp.sum(occ_l, axis=2) + jnp.sum(occ_r, axis=2)) / float(nphi)
    well_res = (delta - jnp.mean(delta, axis=1, keepdims=True)) * sqrt_w * float(well_weight)
    pieces.append(jnp.ravel(well_res) / np.sqrt(float(nalpha * n_levels)))

    # -- extremum alignment (poloidally closed B_min / B_max contours) -------
    line_min = env_l[..., 0]                                  # = bhat at the minimum
    line_max = jnp.maximum(env_l[..., -1], env_r[..., -1])
    ext = jnp.stack([line_min, line_max], axis=-1)            # (nsurf, nalpha, 2)
    ext_res = (ext - jnp.mean(ext, axis=1, keepdims=True)) * sqrt_w * float(extremum_weight)
    pieces.append(jnp.ravel(ext_res) / np.sqrt(float(2 * nalpha)))

    # -- single-well monotonicity (Goodman "squash" distance) ----------------
    squash = jnp.concatenate([env_l - raw_l, (env_r - raw_r)[..., 1:]], axis=-1)
    squash_res = squash * sqrt_w * float(squash_weight)
    pieces.append(jnp.ravel(squash_res) / np.sqrt(float(nalpha * (2 * nk - 1))))

    residuals1d = jnp.concatenate(pieces)
    return {
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "bhat": bhat,
        "delta": delta,
        "line_min": line_min,
        "line_max": line_max,
        "levels": levels,
        "phi": phi,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Objective-term wrapper (QuasisymmetryRatioResidual-style interface)
# ---------------------------------------------------------------------------


class QIResidual:
    """Traceable quasi-isodynamic omnigenity residual (module docstring).

    Composition of :func:`boozer_bmnc_state` (traceable Boozer ``|B|``
    spectrum on the requested surfaces) and :func:`omnigenity_residual`
    (smooth Goodman constructed-QI-target distance).  The interface mirrors
    :class:`~vmex.core.optimize.QuasisymmetryRatioResidual`: the instance
    is a :func:`~vmex.core.optimize.least_squares` objective term for
    both gradient modes — ``jac=None`` calls :meth:`J` on the converged
    :class:`~vmex.core.optimize.Equilibrium`, ``jac="implicit"`` picks up
    the traceable :meth:`residuals_state` vector (full pointwise
    Gauss-Newton geometry, exact implicit gradients).

    Example::

        qi = QIResidual(np.linspace(0.25, 0.9, 4))
        result = least_squares([(qi, 0.0, 10.0), ...], inp, jac="implicit")
    """

    name = "qi"

    def __init__(
        self,
        surfaces,
        *,
        weights: Iterable[float] | None = None,
        mboz: int = 16,
        nboz: int = 16,
        oversample: int = 2,
        nphi: int = 97,
        nalpha: int = 25,
        n_levels: int = 16,
        softness: float = 2.0e-2,
        well_weight: float = 1.0,
        extremum_weight: float = 1.0,
        squash_weight: float = 1.0,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.weights = None if weights is None else np.asarray(list(weights), dtype=float)
        if self.weights is not None and self.weights.shape[0] != self.surfaces.shape[0]:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz = int(mboz)
        self.nboz = int(nboz)
        self.oversample = int(oversample)
        self.nphi = int(nphi)
        self.nalpha = int(nalpha)
        self.n_levels = int(n_levels)
        self.softness = float(softness)
        self.well_weight = float(well_weight)
        self.extremum_weight = float(extremum_weight)
        self.squash_weight = float(squash_weight)

    # -- traceable (state, runtime) evaluation --------------------------------

    def compute_state(self, state: SpectralState, rt: SolverRuntime) -> dict[str, Array]:
        """Full diagnostics dict (Boozer spectrum + residual pieces)."""
        booz = boozer_bmnc_state(
            state, rt, surfaces=self.surfaces, mboz=self.mboz, nboz=self.nboz,
            oversample=self.oversample)
        out = omnigenity_residual(
            bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], nfp=booz["nfp"], weights=self.weights,
            nphi=self.nphi, nalpha=self.nalpha, n_levels=self.n_levels,
            softness=self.softness, well_weight=self.well_weight,
            extremum_weight=self.extremum_weight, squash_weight=self.squash_weight)
        out.update(booz)
        return out

    def residuals_state(self, state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
        """Traceable flat residual vector with ``sum(r**2) = total_state``."""
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable scalar omnigenity objective ``sum(residuals**2)``."""
        return self.compute_state(state, rt)["total"]

    # -- Equilibrium entry points (jac=None objective term) -------------------

    def J(self, eq) -> jnp.ndarray:
        """Objective-term entry point for ``least_squares`` (residual vector)."""
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = J

    def residuals(self, eq) -> jnp.ndarray:
        """Alias of :meth:`J` (simsopt-style vocabulary)."""
        return self.J(eq)

    def total(self, eq) -> Array:
        """Scalar omnigenity objective of a converged equilibrium."""
        return self.total_state(eq.state, eq.runtime)
