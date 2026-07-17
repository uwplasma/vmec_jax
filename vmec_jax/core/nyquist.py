"""Nyquist-table, bsubs, jxbforce and Mercier post-processing (core-native).

VMEC2000 counterparts
---------------------
- ``Sources/Input_Output/wrout.f`` — the Nyquist-resolution half-mesh Fourier
  tables ``gmnc/bmnc/bsubumnc/bsubvmnc/bsupumnc/bsupvmnc/bsubsmns`` (plus the
  lasym sine partners), including the jxbforce low-pass filtering of
  ``bsubu/bsubv`` VMEC applies before writing them:
  :func:`wout_field_tables`.
- ``Sources/Input_Output/bss.f`` — the covariant radial field ``B_s`` on the
  half mesh from the metric cross terms ``g_su/g_sv``:
  :func:`bsubs_half_mesh` / :func:`bsubs_full_mesh_for_wrout`.
- ``Sources/Input_Output/jxbforce.f`` — the low-pass filter transforms, the
  ``bsubsu/bsubsv`` angle derivatives and the 1D current diagnostics
  ``jdotb/bdotb/bdotgradv``.
- ``Sources/Input_Output/mercier.f`` (+ ``vmercier.f``) — the Mercier
  stability profiles ``DMerc/DShear/DWell/DCurr/DGeod``:
  :func:`mercier_and_jxb`.
- ``Sources/General/bcovar.f`` (IEQUI=1 block) — the surface-average
  correction of ``bsubv`` used for the lasym output lane:
  :func:`apply_bsubv_equif_correction`.

The math is extracted from the parity-proven legacy modules
``vmec_jax.io.wout_files.{nyquist,jxbforce,mercier,bsubs}`` (validated against
golden VMEC2000 ``wout`` files in ``tests/test_wout_golden.py``),
re-hosted on the core types: :class:`vmec_jax.core.geometry.RealSpaceGeometry`
/ :class:`~vmec_jax.core.geometry.HalfMeshJacobian` supply the parity
geometry channels, :class:`vmec_jax.core.fields.MagneticFields` the half-mesh
field state, and :class:`vmec_jax.core.fourier.TrigTables` (built at the
Nyquist resolution) the trig/weight tables.

Normalization note (parity-critical): VMEC2000's *output* transforms
normalize lasym integrals on the full theta grid (``dnorm = 1/(nzeta *
ntheta1)``, fixaray.f SPH012314), while the core solver trig tables carry the
reduced-interval ``dnorm`` for both symmetry modes (see
:mod:`vmec_jax.core.fourier`).  :func:`_analysis_theta_tables` therefore
rebuilds the integration-weighted theta tables with the output convention.

Host NumPy throughout: this is one-shot output post-processing, not solver
code.  The ``LBSUBS = T`` corrected-``bsubs`` collocation lane of
``jxbforce.f`` (off by default in VMEC2000 and unused by every reference
deck) is not ported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fourier import ModeTable, TrigTables, mode_table
from .geometry import HalfMeshJacobian, RealSpaceGeometry

__all__ = [
    "WoutFieldTables",
    "nyquist_limits",
    "bsubs_half_mesh",
    "bsubs_full_mesh_for_wrout",
    "apply_bsubv_equif_correction",
    "mercier_and_jxb",
    "wout_field_tables",
]

MU0 = 4.0e-7 * np.pi


# ---------------------------------------------------------------------------
# Mode/trig bookkeeping (fixaray.f Nyquist limits)
# ---------------------------------------------------------------------------


def nyquist_limits(trig: TrigTables) -> tuple[int, int]:
    """Grid Nyquist cutoffs ``(mnyq, nnyq)`` (``fixaray.f``).

    ``mnyq = ntheta2 - 1 = ntheta1/2`` and ``nnyq = nzeta/2`` — geometric grid
    limits, independent of the retained solver modes.
    """
    ntheta2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    return max(ntheta2 - 1, 0), max(nzeta // 2, 0)


def _analysis_theta_tables(trig: TrigTables) -> tuple[np.ndarray, np.ndarray]:
    """Integration-weighted theta tables in the *output* dnorm convention.

    ``fixaray.f``: ``dnorm = 1/(nzeta*ntheta1)`` on the full grid for lasym
    runs (SPH012314), ``1/(nzeta*(ntheta2-1))`` on the endpoint-half-weighted
    reduced grid otherwise; ``cosmui`` rows 0 and ``ntheta2-1`` carry the
    half-weights in both modes.  Returns ``(cosmui, sinmui)`` restricted to
    the reduced grid, shape ``(ntheta2, mnyq+1)``.
    """
    nt2 = int(trig.ntheta2)
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    if nt3 > nt2:  # lasym
        dnorm = 1.0 / (nzeta * nt3)
    else:
        dnorm = 1.0 / (nzeta * (nt2 - 1))
    cosmui = dnorm * np.asarray(trig.cosmu, dtype=float)[:nt2, :].copy()
    sinmui = dnorm * np.asarray(trig.sinmu, dtype=float)[:nt2, :].copy()
    cosmui[0, :] *= 0.5
    cosmui[nt2 - 1, :] *= 0.5
    return cosmui, sinmui


def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
    """Half-mesh ``sqrt(s)`` with the axis slot repeating the first interior."""
    s_arr = np.asarray(s_full, dtype=float)
    if s_arr.shape[0] < 2:
        return np.sqrt(np.maximum(s_arr, 0.0))
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    return np.sqrt(np.maximum(np.concatenate([sh[:1], sh], axis=0), 0.0))


# ---------------------------------------------------------------------------
# symoutput (wrout.f) parity split / extension
# ---------------------------------------------------------------------------


def symoutput_split(*, f: np.ndarray, trig: TrigTables, reversed_sym: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``symoutput`` split into symmetric/antisymmetric parts on ``[0, pi]``.

    The stellarator reflection is ``(theta, zeta) -> (2*pi - theta, -zeta)``;
    ``reversed_sym`` selects the odd-under-reflection kernels (``bsubs``).
    Returns two ``(ns, ntheta2, nzeta)`` arrays.
    """
    f = np.asarray(f, dtype=float)
    nt2, nt1 = int(trig.ntheta2), int(trig.ntheta1)
    nzeta = int(f.shape[2])
    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = f[:, ir0, :][:, :, kk]
    if reversed_sym:
        return 0.5 * (f_half - f_ref), 0.5 * (f_half + f_ref)
    return 0.5 * (f_half + f_ref), 0.5 * (f_half - f_ref)


# ---------------------------------------------------------------------------
# wrout.f Nyquist analysis (cos/sin coefficient tables)
# ---------------------------------------------------------------------------


def _wrout_zeta_tables(trig: TrigTables) -> tuple[np.ndarray, np.ndarray]:
    """``cosnv/sinnv`` with the wrout Nyquist half-weight on the last column."""
    cosnv = np.asarray(trig.cosnv, dtype=float).copy()
    sinnv = np.asarray(trig.sinnv, dtype=float)
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv[:, nnyq] *= 0.5
    return cosnv, sinnv


def _wrout_theta_tables(trig: TrigTables) -> tuple[np.ndarray, np.ndarray]:
    """``cosmui/sinmui`` (output dnorm) with the wrout mnyq half-weight."""
    cosmui, sinmui = _analysis_theta_tables(trig)
    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    return cosmui, sinmui


def _wrout_dmult(modes: ModeTable, trig: TrigTables) -> np.ndarray:
    """Per-mode wrout output normalization ``dmult`` (wrout.f ``tmult``)."""
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    tmult = 0.5
    if int(trig.ntheta3) > int(trig.ntheta2):
        # wrout.f doubles tmult for LASYM (paired with the full-grid dnorm).
        tmult *= 2.0
    dmult = np.asarray(trig.mscale, dtype=float)[m] * np.asarray(trig.nscale, dtype=float)[np.abs(n)] * tmult
    return np.where((m == 0) | (n == 0), 2.0 * dmult, dmult)


def wrout_cos_coeffs(*, f: np.ndarray, modes: ModeTable, trig: TrigTables) -> np.ndarray:
    """wrout.f Nyquist analysis for cosine coefficients, shape ``(ns, mnmax)``."""
    f = np.asarray(f, dtype=float)[:, : int(trig.ntheta2), :]
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    cosmui, sinmui = _wrout_theta_tables(trig)
    cosnv, sinnv = _wrout_zeta_tables(trig)
    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)
    sgn = np.where(n < 0, -1.0, 1.0)
    coeff = cos_zeta[:, m, np.abs(n)] + sgn[None, :] * sin_zeta[:, m, np.abs(n)]
    return coeff * _wrout_dmult(modes, trig)[None, :]


def wrout_sin_coeffs(*, f: np.ndarray, modes: ModeTable, trig: TrigTables) -> np.ndarray:
    """wrout.f Nyquist analysis for sine coefficients, shape ``(ns, mnmax)``."""
    f = np.asarray(f, dtype=float)[:, : int(trig.ntheta2), :]
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    cosmui, sinmui = _wrout_theta_tables(trig)
    cosnv, sinnv = _wrout_zeta_tables(trig)
    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True)
    sgn = np.where(n < 0, -1.0, 1.0)
    coeff = cos_zeta[:, m, np.abs(n)] - sgn[None, :] * sin_zeta[:, m, np.abs(n)]
    return coeff * _wrout_dmult(modes, trig)[None, :]


# ---------------------------------------------------------------------------
# jxbforce.f low-pass filtering of bsubu/bsubv
# ---------------------------------------------------------------------------


def filter_bsubuv_symmetric(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig: TrigTables,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """jxbforce.f low-pass filter of ``bsubu/bsubv`` (``lasym = False``).

    The half-mesh covariant fields are analyzed with the solver band limits
    ``mmax_force = mpol - 1`` / ``nmax_force = ntor`` and resynthesized; VMEC
    treats odd-m content in the ``1/shalf`` internal representation
    (``jxbforce.f``: ``bsubu(js,:,1)/shalf(js)``), reproduced here via the
    even/odd parity channels ``even = f``, ``odd = shalf * f``.
    Returns reduced-grid ``(ns, ntheta2, nzeta)`` arrays.
    """
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    nt2 = int(trig.ntheta2)
    mmax, nmax = int(mmax_force), int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    cosmui_full, sinmui_full = _analysis_theta_tables(trig)
    cosmui = cosmui_full[:, : mmax + 1]
    sinmui = sinmui_full[:, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    dmult = np.ones((mmax + 1, nmax + 1), dtype=float)
    mnyq, nnyq = nyquist_limits(trig)
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    # jxbforce.f handles odd m in the 1/shalf internal representation; with
    # the default parity channels (odd = shalf * even) the 1/shalf division
    # cancels exactly, so the filter reduces to a band-limited projection
    # (no radial dependence — ``s`` is kept in the signature for symmetry
    # with :func:`filter_bsubuv_lasym`).

    def _filter_field(f: np.ndarray) -> np.ndarray:
        f_red = f[:, :nt2, :]
        f_theta_cos = np.einsum("sik,im->smk", f_red, cosmui, optimize=True)
        f_theta_sin = np.einsum("sik,im->smk", f_red, sinmui, optimize=True)
        c1 = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True) * dmult[None, :, :]
        c2 = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True) * dmult[None, :, :]
        tmp_cos = np.einsum("smn,im->sin", c1, cosmu, optimize=True)
        tmp_sin = np.einsum("smn,im->sin", c2, sinmu, optimize=True)
        return np.einsum("sin,kn->sik", tmp_cos, cosnv, optimize=True) + np.einsum(
            "sin,kn->sik", tmp_sin, sinnv, optimize=True
        )

    return _filter_field(bsubu), _filter_field(bsubv)


def filter_bsubuv_lasym(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig: TrigTables,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """jxbforce.f low-pass filter of ``bsubu/bsubv`` for ``lasym = True``.

    Mirrors the Fortran surface loop: ``fsym_fft`` parity split to the reduced
    grid, the band-limited transform/inverse for both parities, and the
    ``fext_fft`` extension back to the full theta grid.  Long-double
    accumulation (the sums are cancellation-limited near the edge).
    Returns full-grid ``(ns, ntheta3, nzeta)`` arrays.
    """
    acc = np.longdouble
    bsubu = np.asarray(bsubu, dtype=acc)
    bsubv = np.asarray(bsubv, dtype=acc)
    ns, _, nzeta = bsubu.shape
    nt2, nt1 = int(trig.ntheta2), int(trig.ntheta1)
    nt3 = max(int(trig.ntheta3), nt2)
    mmax, nmax = int(mmax_force), int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu[:, :nt3, :].astype(float), bsubv[:, :nt3, :].astype(float)

    cosmui_full, sinmui_full = _analysis_theta_tables(trig)
    cosmui = np.asarray(cosmui_full, dtype=acc)[:, : mmax + 1]
    sinmui = np.asarray(sinmui_full, dtype=acc)[:, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc)[:, : nmax + 1]
    mnyq, nnyq = nyquist_limits(trig)

    bsubu_out = np.zeros((ns, nt3, nzeta), dtype=acc)
    bsubv_out = np.zeros((ns, nt3, nzeta), dtype=acc)

    for js in range(ns):
        bu = bsubu[js, :nt3, :].T  # Fortran (zeta, theta) ordering
        bv = bsubv[js, :nt3, :].T
        bu_ch = np.stack([bu, bu], axis=-1)  # (nzeta, nt3, parity)
        bv_ch = np.stack([bv, bv], axis=-1)

        bu_s = np.zeros((nzeta, nt2, 2), dtype=acc)
        bu_a = np.zeros((nzeta, nt2, 2), dtype=acc)
        bv_s = np.zeros((nzeta, nt2, 2), dtype=acc)
        bv_a = np.zeros((nzeta, nt2, 2), dtype=acc)
        for i in range(nt2):
            ir = 0 if i == 0 else (nt1 - i)
            for kz in range(nzeta):
                kzr = 0 if kz == 0 else (nzeta - kz)
                bu_a[kz, i, :] = 0.5 * (bu_ch[kz, i, :] - bu_ch[kzr, ir, :])
                bu_s[kz, i, :] = 0.5 * (bu_ch[kz, i, :] + bu_ch[kzr, ir, :])
                bv_a[kz, i, :] = 0.5 * (bv_ch[kz, i, :] - bv_ch[kzr, ir, :])
                bv_s[kz, i, :] = 0.5 * (bv_ch[kz, i, :] + bv_ch[kzr, ir, :])

        bsubua = np.zeros((nzeta, nt2, 2), dtype=acc)
        bsubva = np.zeros((nzeta, nt2, 2), dtype=acc)
        for m in range(mmax + 1):
            mparity = m & 1
            for n in range(nmax + 1):
                dnorm1 = acc(1.0)
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5

                bsubumn1 = bsubumn2 = bsubvmn1 = bsubvmn2 = acc(0.0)
                bsubumn3 = bsubumn4 = bsubvmn3 = bsubvmn4 = acc(0.0)
                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        bsubumn1 += tcosi1 * bu_s[k, j, mparity]
                        bsubumn2 += tcosi2 * bu_s[k, j, mparity]
                        bsubvmn1 += tcosi1 * bv_s[k, j, mparity]
                        bsubvmn2 += tcosi2 * bv_s[k, j, mparity]
                        bsubumn3 += tsini1 * bu_a[k, j, mparity]
                        bsubumn4 += tsini2 * bu_a[k, j, mparity]
                        bsubvmn3 += tsini1 * bv_a[k, j, mparity]
                        bsubvmn4 += tsini2 * bv_a[k, j, mparity]

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubua[k, j, 0] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubva[k, j, 0] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2
                        tsin1 = sinmu[j, m] * cosnv[k, n]
                        tsin2 = cosmu[j, m] * sinnv[k, n]
                        bsubua[k, j, 1] += tsin1 * bsubumn3 + tsin2 * bsubumn4
                        bsubva[k, j, 1] += tsin1 * bsubvmn3 + tsin2 * bsubvmn4

        bu_full = np.zeros((nzeta, nt3), dtype=acc)
        bv_full = np.zeros((nzeta, nt3), dtype=acc)
        bu_full[:, :nt2] = bsubua[:, :, 0] + bsubua[:, :, 1]
        bv_full[:, :nt2] = bsubva[:, :, 0] + bsubva[:, :, 1]
        for i in range(nt2, nt3):
            ir = nt1 - i
            for kz in range(nzeta):
                kzr = 0 if kz == 0 else (nzeta - kz)
                bu_full[kz, i] = bsubua[kzr, ir, 0] - bsubua[kzr, ir, 1]
                bv_full[kz, i] = bsubva[kzr, ir, 0] - bsubva[kzr, ir, 1]
        bsubu_out[js] = bu_full.T
        bsubv_out[js] = bv_full.T

    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


# ---------------------------------------------------------------------------
# bss.f — covariant radial field on the half mesh
# ---------------------------------------------------------------------------


def bsubs_half_mesh(
    *,
    geometry: RealSpaceGeometry,
    jacobian: HalfMeshJacobian,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """``B_s = B^u g_su + B^v g_sv`` on the half mesh (``bss.f``).

    The half-mesh geometry follows ``jacobian.f`` conventions: ``ru12/zu12/
    rs/zs`` come straight from :func:`~vmec_jax.core.geometry.half_mesh_jacobian`
    (identical formulas), ``rv12/zv12`` are the half-mesh averages of the
    toroidal derivatives, and ``rs12/zs12`` add the ``d(shalf)/ds`` odd-m
    chain-rule terms (``dphids = 0.25``); every half-mesh array mirrors
    ``js = 2`` into the axis slot.
    """
    s = np.asarray(s, dtype=float)
    if s.shape[0] < 2:
        return np.zeros_like(np.asarray(bsupu, dtype=float))
    sh = _pshalf_from_s(s)[:, None, None]
    sh[0] = 0.0  # match the legacy shalf array (unused: rows below start at 1)

    R_odd = np.asarray(geometry.R_odd, dtype=float)
    Z_odd = np.asarray(geometry.Z_odd, dtype=float)
    Rv_e = np.asarray(geometry.dR_dzeta_even, dtype=float)
    Rv_o = np.asarray(geometry.dR_dzeta_odd, dtype=float)
    Zv_e = np.asarray(geometry.dZ_dzeta_even, dtype=float)
    Zv_o = np.asarray(geometry.dZ_dzeta_odd, dtype=float)

    ru12 = np.asarray(jacobian.ru12, dtype=float)
    zu12 = np.asarray(jacobian.zu12, dtype=float)
    rs = np.asarray(jacobian.dR_ds, dtype=float)
    zs = np.asarray(jacobian.dZ_ds, dtype=float)

    rv12 = np.zeros_like(rs)
    zv12 = np.zeros_like(rs)
    rs12 = np.zeros_like(rs)
    zs12 = np.zeros_like(rs)
    dphids = 0.25
    rv12[1:] = 0.5 * (Rv_e[1:] + Rv_e[:-1] + sh[1:] * (Rv_o[1:] + Rv_o[:-1]))
    zv12[1:] = 0.5 * (Zv_e[1:] + Zv_e[:-1] + sh[1:] * (Zv_o[1:] + Zv_o[:-1]))
    rs12[1:] = rs[1:] + dphids * (R_odd[1:] + R_odd[:-1]) / sh[1:]
    zs12[1:] = zs[1:] + dphids * (Z_odd[1:] + Z_odd[:-1]) / sh[1:]

    for arr in (rs12, zs12, rv12, zv12):
        arr[0] = arr[1]
    # jacobian.f half-mesh arrays already carry the axis copy (row 0 = row 1).

    g_su = rs12 * ru12 + zs12 * zu12
    g_sv = rs12 * rv12 + zs12 * zv12
    return np.asarray(bsupu, dtype=float) * g_su + np.asarray(bsupv, dtype=float) * g_sv


def bsubs_full_mesh_for_wrout(*, bsubs_half: np.ndarray) -> np.ndarray:
    """Half-mesh ``bsubs`` -> the full-mesh convention written by ``wrout.f``."""
    bsubs_full = np.asarray(bsubs_half, dtype=float).copy()
    ns = int(bsubs_full.shape[0])
    if ns > 0:
        bsubs_full[0] = 0.0
    if ns > 2:
        bsubs_full[1:-1] = 0.5 * (bsubs_full[1:-1] + bsubs_full[2:])
        bsubs_full[0] = 2.0 * bsubs_full[1] - bsubs_full[2]
        bsubs_full[-1] = 2.0 * bsubs_full[-1] - bsubs_full[-2]
    return bsubs_full


# ---------------------------------------------------------------------------
# bcovar.f IEQUI=1 bsubv correction (lasym output lane)
# ---------------------------------------------------------------------------


def bsubv_lambda_full_mesh(
    *,
    geometry: RealSpaceGeometry,
    jacobian: HalfMeshJacobian,
    metrics,
    fields,
    s: np.ndarray,
    phipf: np.ndarray,
) -> np.ndarray:
    """Full-mesh ``bsubv_e`` of ``bcovar.f`` (the lambda-force reconstruction).

    ``bsubv`` rebuilt on the full mesh from the lambda derivatives (``lvv =
    gvv/sqrt(g)``) and blended with the plain forward average using the v8.49
    damping ``bdamp = 2*0.05*(1 - s)`` — the source of the lasym IEQUI
    correction (see :func:`apply_bsubv_equif_correction`).  Mirrors
    :func:`vmec_jax.core.forces.lambda_force_kernels` before its ``-lamscale``
    output scaling.
    """
    s = np.asarray(s, dtype=float)
    ns = int(s.shape[0])
    sqrt_g = np.asarray(jacobian.sqrt_g, dtype=float)
    lamscale = float(np.asarray(fields.lamscale))
    phipf = np.asarray(phipf, dtype=float)

    lu_even = lamscale * np.asarray(geometry.dlambda_dtheta_even, dtype=float) + phipf[:, None, None]
    lu_odd = lamscale * np.asarray(geometry.dlambda_dtheta_odd, dtype=float)

    phipog = np.where(sqrt_g != 0.0, 1.0 / np.where(sqrt_g != 0.0, sqrt_g, 1.0), 0.0)
    if ns >= 1:
        phipog[0] = 0.0
    pshalf = _pshalf_from_s(s)[:, None, None]
    lvv = phipog * np.asarray(metrics.gvv, dtype=float)
    lvv_sh = lvv * pshalf
    bsubv_cross = np.asarray(metrics.guv, dtype=float) * np.asarray(fields.bsupu, dtype=float)

    if ns < 2:
        return np.zeros_like(sqrt_g)
    bsubv_full = np.zeros_like(sqrt_g)
    bsubv_full[:-1] = 0.5 * (
        (lvv[:-1] + lvv[1:]) * lu_even[:-1]
        + (lvv_sh[:-1] + lvv_sh[1:]) * lu_odd[:-1]
        + bsubv_cross[:-1] + bsubv_cross[1:]
    )
    bsubv_full[-1] = 0.5 * (lvv[-1] * lu_even[-1] + lvv_sh[-1] * lu_odd[-1] + bsubv_cross[-1])

    bsubv_avg = np.zeros_like(sqrt_g)
    bsubv_h = np.asarray(fields.bsubv, dtype=float)
    bsubv_avg[:-1] = 0.5 * (bsubv_h[:-1] + bsubv_h[1:])
    bsubv_avg[-1] = 0.5 * bsubv_h[-1]

    bdamp = (2.0 * 0.05 * (1.0 - s))[:, None, None]
    return bdamp * bsubv_full + (1.0 - bdamp) * bsubv_avg


def apply_bsubv_equif_correction(
    *,
    bsubv: np.ndarray,
    bsubv_e: np.ndarray,
    trig: TrigTables,
) -> np.ndarray:
    """``bcovar.f`` IEQUI=1 half-mesh reconstruction of ``bsubv``.

    Backward recurrence ``bsubv(js) = 2*bsubv_e(js) - bsubv(js+1)`` followed
    by a per-surface constant shift preserving the surface average
    ``fpsi = <bsubv>`` (``pwint`` weights, zero on the axis row).
    """
    bsubv = np.asarray(bsubv, dtype=float)
    bsubv_e = np.asarray(bsubv_e, dtype=float)
    ns = int(bsubv.shape[0])
    if ns < 3:
        return bsubv
    wint = np.asarray(trig.wint, dtype=float)
    fpsi = np.array([float(np.sum(bsubv[js] * wint)) for js in range(ns)])
    out = bsubv.copy()
    for js in range(ns - 2, 0, -1):
        out[js] = 2.0 * bsubv_e[js] - out[js + 1]
    for js in range(1, ns):
        out[js] = out[js] + (fpsi[js] - float(np.sum(out[js] * wint)))
    return out


# ---------------------------------------------------------------------------
# jxbforce.f bsubsu/bsubsv angle derivatives
# ---------------------------------------------------------------------------


def _bsubs_angle_derivatives(
    *,
    bsubs_use: np.ndarray,
    trig: TrigTables,
    mmax_force: int,
    nmax_force: int,
    lasym: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Band-limited ``(d(bsubs)/du, d(bsubs)/dv)`` reconstruction (jxbforce.f).

    ``bsubs_use`` is the full-mesh (angle-full-grid for lasym) field; returns
    reduced-grid arrays for symmetric runs and full-grid for lasym.
    """
    mmax, nmax = int(mmax_force), int(nmax_force)
    nt2 = int(trig.ntheta2)
    cosmui_full, sinmui_full = _analysis_theta_tables(trig)
    cosmui = cosmui_full[:, : mmax + 1]
    sinmui = sinmui_full[:, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosmum = np.asarray(trig.cosmum, dtype=float)[:nt2, : mmax + 1]
    sinmum = np.asarray(trig.sinmum, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=float)[:, : nmax + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=float)[:, : nmax + 1]

    dmult = np.ones((mmax + 1, nmax + 1), dtype=float)
    mnyq, nnyq = nyquist_limits(trig)
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    def _sin_analysis(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c1 = np.einsum(
            "smk,kn->smn", np.einsum("sik,im->smk", f, sinmui, optimize=True), cosnv, optimize=True
        ) * dmult[None, :, :]
        c2 = np.einsum(
            "smk,kn->smn", np.einsum("sik,im->smk", f, cosmui, optimize=True), sinnv, optimize=True
        ) * dmult[None, :, :]
        return c1, c2

    def _cos_analysis(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c3 = np.einsum(
            "smk,kn->smn", np.einsum("sik,im->smk", f, cosmui, optimize=True), cosnv, optimize=True
        ) * dmult[None, :, :]
        c4 = np.einsum(
            "smk,kn->smn", np.einsum("sik,im->smk", f, sinmui, optimize=True), sinnv, optimize=True
        ) * dmult[None, :, :]
        return c3, c4

    def _synth(c_a, table_a, zeta_a, c_b, table_b, zeta_b):
        return np.einsum(
            "sin,kn->sik", np.einsum("smn,im->sin", c_a, table_a, optimize=True), zeta_a, optimize=True
        ) + np.einsum(
            "sin,kn->sik", np.einsum("smn,im->sin", c_b, table_b, optimize=True), zeta_b, optimize=True
        )

    if not bool(lasym):
        c1, c2 = _sin_analysis(bsubs_use[:, :nt2, :])
        bsubsu = _synth(c1, cosmum, cosnv, c2, sinmum, sinnv)
        bsubsv = _synth(c1, sinmu, sinnvn, c2, cosmu, cosnvn)
        return bsubsu, bsubsv

    bsubs_sym, bsubs_asym = symoutput_split(f=bsubs_use, trig=trig, reversed_sym=True)
    c1, c2 = _sin_analysis(bsubs_sym)
    bsubsu_s = _synth(c1, cosmum, cosnv, c2, sinmum, sinnv)
    bsubsv_s = _synth(c1, sinmu, sinnvn, c2, cosmu, cosnvn)
    c3, c4 = _cos_analysis(bsubs_asym)
    bsubsu_a = _synth(c3, sinmum, cosnv, c4, cosmum, sinnv)
    bsubsv_a = _synth(c3, cosmu, sinnvn, c4, sinmu, cosnvn)
    bsubsu = _extend_parity_to_full(bsubsu_s, bsubsu_a, trig=trig)
    bsubsv = _extend_parity_to_full(bsubsv_s, bsubsv_a, trig=trig)
    return bsubsu, bsubsv


def _extend_parity_to_full(par0: np.ndarray, par1: np.ndarray, *, trig: TrigTables) -> np.ndarray:
    """Extend reduced-grid parity components to the full lasym theta grid."""
    nt1, nt2 = int(trig.ntheta1), int(trig.ntheta2)
    nt3 = max(int(trig.ntheta3), nt2)
    nzeta = int(np.asarray(par0).shape[2])
    full = np.zeros((par0.shape[0], nt3, nzeta), dtype=float)
    full[:, :nt2, :] = par0 + par1
    if nt3 == nt2:
        return full
    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    mask = ir0 >= nt2
    if np.any(mask):
        full[:, ir0[mask], :] = par0[:, mask, :][:, :, kk] - par1[:, mask, :][:, :, kk]
    return full


# ---------------------------------------------------------------------------
# mercier.f / jxbforce.f 1D diagnostics
# ---------------------------------------------------------------------------


def mercier_and_jxb(
    *,
    geometry: RealSpaceGeometry,
    s: np.ndarray,
    lasym: bool,
    mpol: int,
    ntor: int,
    pres: np.ndarray,
    vp: np.ndarray,
    phips: np.ndarray,
    iotas: np.ndarray,
    bsq: np.ndarray,
    sqrtg: np.ndarray,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubs_half: np.ndarray,
    trig: TrigTables,
    signgs: int,
) -> tuple[np.ndarray, ...]:
    """Mercier profiles and jxbforce 1D current diagnostics.

    Port of ``mercier.f``/``vmercier.f`` and the ``jdotb/bdotb/bdotgradv``
    block of ``jxbforce.f``, consuming the core half-mesh field state:
    ``pres`` in internal units (mu0*Pa), ``bsq`` the total pressure,
    ``bsubu/bsubv`` the jxbforce-filtered covariant fields (reduced grid for
    symmetric runs, full grid for lasym), ``bsubs_half`` from
    :func:`bsubs_half_mesh`.  Returns
    ``(DMerc, DShear, DCurr, DWell, DGeod, jdotb, bdotb, bdotgradv)``.
    """
    pres = np.asarray(pres, dtype=float)
    ns = int(pres.shape[0])
    if ns < 3:
        zero = np.zeros((ns,), dtype=float)
        return tuple(zero.copy() for _ in range(8))
    hs = 1.0 / float(ns - 1)
    ohs = 1.0 / hs
    s = np.asarray(s, dtype=float)
    vp = np.asarray(vp, dtype=float)
    phips = np.asarray(phips, dtype=float)
    iotas = np.asarray(iotas, dtype=float)
    bsq = np.asarray(bsq, dtype=float)
    sqrtg = np.asarray(sqrtg, dtype=float)
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    sign_jac = float(np.sign(float(signgs))) if int(signgs) != 0 else 1.0
    mmax_force = max(int(mpol) - 1, 0)
    nmax_force = int(ntor)

    wint = np.asarray(trig.wint, dtype=float)

    def sum_w(arr: np.ndarray) -> float:
        return float(np.einsum("ij,ij->", np.asarray(arr, dtype=float), wint, optimize=True))

    # mercier.f flux normalizations (phip_real, vp_real).
    phip_real = (2.0 * np.pi) * phips * sign_jac
    vp_real = np.zeros_like(phip_real)
    vp_real[1:] = sign_jac * (2.0 * np.pi) ** 2 * vp[1:] / phip_real[1:]

    # jxbforce.f: bsubs averaged to the full mesh, axis zeroed.
    bsubs_use = np.asarray(bsubs_half, dtype=float).copy()
    if ns > 2:
        bsubs_use[1:-1] = 0.5 * (bsubs_use[1:-1] + bsubs_use[2:])
    bsubs_use[0] = 0.0

    if bool(lasym):
        # jxbforce.f filters the (already output-filtered) covariant fields
        # again inside the lasym force loop; replicated for golden parity.
        bsubu, bsubv = filter_bsubuv_lasym(
            bsubu=bsubu, bsubv=bsubv, trig=trig,
            mmax_force=mmax_force, nmax_force=nmax_force, s=s,
        )

    bsubsu, bsubsv = _bsubs_angle_derivatives(
        bsubs_use=bsubs_use, trig=trig, mmax_force=mmax_force,
        nmax_force=nmax_force, lasym=bool(lasym),
    )

    # jxbforce.f: sqrt(g)*J contravariant components and sqrt(g)*J.B.
    itheta = np.zeros_like(bsubs_use)
    izeta = np.zeros_like(bsubs_use)
    itheta[1:-1] = bsubsv[1:-1] - ohs * (bsubv[2:] - bsubv[1:-1])
    izeta[1:-1] = -bsubsu[1:-1] + ohs * (bsubu[2:] - bsubu[1:-1])
    izeta[0] = 2.0 * izeta[1] - izeta[2]
    izeta[-1] = 2.0 * izeta[-2] - izeta[-3]
    itheta /= MU0
    izeta /= MU0
    bdotk = np.zeros_like(bsubs_use)
    bdotk[1:-1] = itheta[1:-1] * 0.5 * (bsubu[2:] + bsubu[1:-1]) + izeta[1:-1] * 0.5 * (bsubv[2:] + bsubv[1:-1])
    bdotk_merc = MU0 * bdotk  # mercier.f consumes sqrt(g)*J.B in internal units

    # -- mercier.f radial stability terms -----------------------------------
    R_even, R_odd = np.asarray(geometry.R_even, float), np.asarray(geometry.R_odd, float)
    Ru_even, Ru_odd = np.asarray(geometry.dR_dtheta_even, float), np.asarray(geometry.dR_dtheta_odd, float)
    Zu_even, Zu_odd = np.asarray(geometry.dZ_dtheta_even, float), np.asarray(geometry.dZ_dtheta_odd, float)
    Rv_even, Rv_odd = np.asarray(geometry.dR_dzeta_even, float), np.asarray(geometry.dR_dzeta_odd, float)
    Zv_even, Zv_odd = np.asarray(geometry.dZ_dzeta_even, float), np.asarray(geometry.dZ_dzeta_odd, float)

    DMerc, Dshear, Dcurr, Dwell, Dgeod = (np.zeros((ns,), dtype=float) for _ in range(5))
    shear, vpp, presp, ip, torcur = (np.zeros((ns,), dtype=float) for _ in range(5))
    torcur[1:] = sign_jac * (2.0 * np.pi) * np.einsum("sij,ij->s", bsubu[1:], wint, optimize=True)
    phip_full_h = 0.5 * (phip_real[2:] + phip_real[1:-1])
    denom = 1.0 / (hs * phip_full_h)
    shear[1:-1] = (iotas[2:] - iotas[1:-1]) * denom
    vpp[1:-1] = (vp_real[2:] - vp_real[1:-1]) * denom
    presp[1:-1] = (pres[2:] - pres[1:-1]) * denom
    ip[1:-1] = (torcur[2:] - torcur[1:-1]) * denom

    b2 = 2.0 * (bsq - pres[:, None, None])
    two_pi_sq = (2.0 * np.pi) ** 2
    for i in range(1, ns - 1):
        phip_full = 0.5 * (phip_real[i + 1] + phip_real[i])
        gsqrt_raw = 0.5 * (sqrtg[i] + sqrtg[i + 1])
        gsqrt_full = gsqrt_raw / phip_full
        sqs = float(np.sqrt(s[i]))
        r1f = R_even[i] + sqs * R_odd[i]
        rtf = Ru_even[i] + sqs * Ru_odd[i]
        ztf = Zu_even[i] + sqs * Zu_odd[i]
        rzf = Rv_even[i] + sqs * Rv_odd[i]
        zzf = Zv_even[i] + sqs * Zv_odd[i]
        gtt = rtf * rtf + ztf * ztf
        gpp = (gsqrt_full * gsqrt_full) / (gtt * r1f * r1f + (rtf * zzf - rzf * ztf) ** 2)
        b2i = 0.5 * (b2[i + 1] + b2[i])
        tpp = sum_w(gsqrt_full / b2i) * two_pi_sq
        tbb = sum_w(b2i * gsqrt_full * gpp) * two_pi_sq
        bdotj_norm = np.where(gsqrt_raw != 0.0, bdotk_merc[i] / gsqrt_raw, 0.0)
        jdotb_i = bdotj_norm * gpp * gsqrt_full
        tjb = sum_w(jdotb_i) * two_pi_sq
        tjj = sum_w(jdotb_i * bdotj_norm / b2i) * two_pi_sq

        Dshear[i] = 0.25 * shear[i] * shear[i]
        Dcurr[i] = -shear[i] * (tjb - ip[i] * tbb)
        Dwell[i] = presp[i] * (vpp[i] - presp[i] * tpp) * tbb
        Dgeod[i] = tjb * tjb - tbb * tjj
        DMerc[i] = Dshear[i] + Dcurr[i] + Dwell[i] + Dgeod[i]

    # -- jxbforce.f 1D current diagnostics -----------------------------------
    jdotb = np.zeros((ns,), dtype=float)
    bdotb = np.zeros((ns,), dtype=float)
    bdotgradv = np.zeros((ns,), dtype=float)
    dnorm1 = float((2.0 * np.pi) ** 2)
    for js in range(1, ns - 1):
        denom_v = vp[js + 1] + vp[js]
        if denom_v == 0.0:
            continue
        tjnorm = (2.0 / denom_v / dnorm1) * sign_jac
        sqgb2 = sqrtg[js + 1] * (bsq[js + 1] - pres[js + 1]) + sqrtg[js] * (bsq[js] - pres[js])
        jdotb[js] = dnorm1 * tjnorm * sum_w(bdotk[js])
        bdotb[js] = dnorm1 * tjnorm * sum_w(sqgb2)
        bdotgradv[js] = 0.5 * dnorm1 * tjnorm * (phips[js] + phips[js + 1])
    jdotb[0] = 2.0 * jdotb[1] - jdotb[2]
    jdotb[-1] = 2.0 * jdotb[-2] - jdotb[-3]
    bdotb[0] = 2.0 * bdotb[2] - bdotb[1]
    bdotb[-1] = 2.0 * bdotb[-2] - bdotb[-3]
    bdotgradv[0] = 2.0 * bdotgradv[1] - bdotgradv[2]
    bdotgradv[-1] = 2.0 * bdotgradv[-2] - bdotgradv[-3]

    return DMerc, Dshear, Dcurr, Dwell, Dgeod, jdotb, bdotb, bdotgradv


# ---------------------------------------------------------------------------
# wrout.f Nyquist table assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WoutFieldTables:
    """wrout.f Nyquist-resolution output tables + jxbforce/Mercier profiles.

    Fourier tables are in file convention, shape ``(ns, mnmax_nyq)``; the
    sine partners are ``None`` for stellarator-symmetric runs.  ``xm_nyq`` /
    ``xn_nyq`` carry the ``nfp`` factor on ``xn``.  Radial profiles are
    ``(ns,)`` in wout units except ``jdotb/bdotb`` (internal jxbforce.f
    normalization, as written by VMEC2000).
    """

    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    gmnc: np.ndarray
    bmnc: np.ndarray
    bsubumnc: np.ndarray
    bsubvmnc: np.ndarray
    bsupumnc: np.ndarray
    bsupvmnc: np.ndarray
    bsubsmns: np.ndarray
    gmns: np.ndarray | None
    bmns: np.ndarray | None
    bsubumns: np.ndarray | None
    bsubvmns: np.ndarray | None
    bsupumns: np.ndarray | None
    bsupvmns: np.ndarray | None
    bsubsmnc: np.ndarray | None
    jdotb: np.ndarray
    bdotb: np.ndarray
    bdotgradv: np.ndarray
    DMerc: np.ndarray
    DShear: np.ndarray
    DCurr: np.ndarray
    DWell: np.ndarray
    DGeod: np.ndarray


def _zero_first_surface(*arrays: np.ndarray) -> None:
    for arr in arrays:
        if arr.shape[0] > 0:
            arr[0, :] = 0.0


def wout_field_tables(
    *,
    geometry: RealSpaceGeometry,
    jacobian: HalfMeshJacobian,
    metrics,
    fields,
    trig: TrigTables,
    s: np.ndarray,
    mpol: int,
    ntor: int,
    nfp: int,
    lasym: bool,
    vp: np.ndarray,
    phips: np.ndarray,
    iotas: np.ndarray,
    phipf: np.ndarray,
    signgs: int,
) -> WoutFieldTables:
    """Build every legacy-engine wout quantity from the core field state.

    Reproduces the wrout.f output sequence: bss.f ``bsubs``; the jxbforce.f
    low-pass filtering of ``bsubu/bsubv`` (with the lasym IEQUI ``bsubv``
    correction feeding the antisymmetric channel); the Nyquist coefficient
    analysis of ``sqrt(g)``, ``|B|``, ``B^u/B^v``, ``B_u/B_v`` and ``B_s``;
    and the jxbforce.f/mercier.f 1D diagnostics.

    Inputs are the core pipeline objects evaluated on the Nyquist-extended
    trig tables: ``fields.pressure`` in internal units, ``vp/phips/iotas``
    the half-mesh profiles (axis slots zeroed), ``phipf`` the internal
    full-mesh ``phip`` (as passed to ``magnetic_fields``).
    """
    s = np.asarray(s, dtype=float)
    ns = int(s.shape[0])
    mnyq, nnyq = nyquist_limits(trig)
    nyq_modes = mode_table(max(mnyq, max(int(mpol) - 1, 0)) + 1, max(nnyq, int(ntor)))
    xm_nyq = np.asarray(nyq_modes.m, dtype=float)
    xn_nyq = np.asarray(nyq_modes.n, dtype=float) * float(int(nfp))
    mmax_force = max(int(mpol) - 1, 0)
    nmax_force = int(ntor)

    sqrtg = np.asarray(jacobian.sqrt_g, dtype=float)
    bsq = np.asarray(fields.total_pressure, dtype=float)
    pres = np.asarray(fields.pressure, dtype=float)
    bsupu = np.asarray(fields.bsupu, dtype=float)
    bsupv = np.asarray(fields.bsupv, dtype=float)
    bsubu_raw = np.asarray(fields.bsubu, dtype=float)
    bsubv_raw = np.asarray(fields.bsubv, dtype=float)
    bmag = np.sqrt(2.0 * np.abs(bsq - pres[:, None, None]))

    # -- bss.f ----------------------------------------------------------------
    bsubs_half = bsubs_half_mesh(geometry=geometry, jacobian=jacobian,
                                 bsupu=bsupu, bsupv=bsupv, s=s)
    bsubs_full = bsubs_full_mesh_for_wrout(bsubs_half=bsubs_half)

    # -- jxbforce.f low-pass filtering of the covariant components ------------
    if bool(lasym):
        bsubv_e = bsubv_lambda_full_mesh(
            geometry=geometry, jacobian=jacobian, metrics=metrics,
            fields=fields, s=s, phipf=phipf,
        )
        bsubv_asym_source = apply_bsubv_equif_correction(
            bsubv=bsubv_raw, bsubv_e=bsubv_e, trig=trig,
        )
        bsubu_out, bsubv_out = filter_bsubuv_lasym(
            bsubu=bsubu_raw, bsubv=bsubv_raw, trig=trig,
            mmax_force=mmax_force, nmax_force=nmax_force, s=s,
        )
        _, bsubv_asym_source = filter_bsubuv_lasym(
            bsubu=bsubu_raw, bsubv=bsubv_asym_source, trig=trig,
            mmax_force=mmax_force, nmax_force=nmax_force, s=s,
        )
    else:
        bsubu_out, bsubv_out = filter_bsubuv_symmetric(
            bsubu=bsubu_raw, bsubv=bsubv_raw, trig=trig,
            mmax_force=mmax_force, nmax_force=nmax_force, s=s,
        )
        bsubv_asym_source = None

    # -- Nyquist coefficient tables (wrout.f) ----------------------------------
    if not bool(lasym):
        gmnc = wrout_cos_coeffs(f=sqrtg, modes=nyq_modes, trig=trig)
        bmnc = wrout_cos_coeffs(f=bmag, modes=nyq_modes, trig=trig)
        bsupumnc = wrout_cos_coeffs(f=bsupu, modes=nyq_modes, trig=trig)
        bsupvmnc = wrout_cos_coeffs(f=bsupv, modes=nyq_modes, trig=trig)
        bsubumnc = wrout_cos_coeffs(f=bsubu_out, modes=nyq_modes, trig=trig)
        bsubvmnc = wrout_cos_coeffs(f=bsubv_out, modes=nyq_modes, trig=trig)
        bsubsmns = wrout_sin_coeffs(f=bsubs_full, modes=nyq_modes, trig=trig)
        _zero_first_surface(gmnc, bmnc, bsupumnc, bsupvmnc, bsubumnc, bsubvmnc)
        if ns > 2:
            bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]
        gmns = bmns = bsubumns = bsubvmns = bsupumns = bsupvmns = bsubsmnc = None
    else:
        pairs = {}
        for name, field, reversed_sym in (
            ("g", sqrtg, False), ("b", bmag, False),
            ("bsubu", bsubu_out, False), ("bsubv", bsubv_out, False),
            ("bsupu", bsupu, False), ("bsupv", bsupv, False),
            ("bsubs", bsubs_full, True),
        ):
            pairs[name] = symoutput_split(f=field, trig=trig, reversed_sym=reversed_sym)
        if bsubv_asym_source is not None:
            # wrout.f uses the IEQUI-corrected field for the antisymmetric
            # bsubv channel only (the symmetric channel keeps the filter output).
            pairs["bsubv"] = (pairs["bsubv"][0],
                              symoutput_split(f=bsubv_asym_source, trig=trig)[1])

        gmnc = wrout_cos_coeffs(f=pairs["g"][0], modes=nyq_modes, trig=trig)
        bmnc = wrout_cos_coeffs(f=pairs["b"][0], modes=nyq_modes, trig=trig)
        bsubumnc = wrout_cos_coeffs(f=pairs["bsubu"][0], modes=nyq_modes, trig=trig)
        bsubvmnc = wrout_cos_coeffs(f=pairs["bsubv"][0], modes=nyq_modes, trig=trig)
        bsupumnc = wrout_cos_coeffs(f=pairs["bsupu"][0], modes=nyq_modes, trig=trig)
        bsupvmnc = wrout_cos_coeffs(f=pairs["bsupv"][0], modes=nyq_modes, trig=trig)
        bsubsmns = wrout_sin_coeffs(f=pairs["bsubs"][0], modes=nyq_modes, trig=trig)
        gmns = wrout_sin_coeffs(f=pairs["g"][1], modes=nyq_modes, trig=trig)
        bmns = wrout_sin_coeffs(f=pairs["b"][1], modes=nyq_modes, trig=trig)
        bsubumns = wrout_sin_coeffs(f=pairs["bsubu"][1], modes=nyq_modes, trig=trig)
        bsubvmns = wrout_sin_coeffs(f=pairs["bsubv"][1], modes=nyq_modes, trig=trig)
        bsupumns = wrout_sin_coeffs(f=pairs["bsupu"][1], modes=nyq_modes, trig=trig)
        bsupvmns = wrout_sin_coeffs(f=pairs["bsupv"][1], modes=nyq_modes, trig=trig)
        bsubsmnc = wrout_cos_coeffs(f=pairs["bsubs"][1], modes=nyq_modes, trig=trig)

        # wrout.f lasym conventions: covariant tables restricted to the solver
        # band and doubled (the reduced-grid analysis halves them).
        m_arr = np.asarray(nyq_modes.m, dtype=int)
        n_arr = np.asarray(nyq_modes.n, dtype=int)
        mask_bsub = (m_arr >= int(mpol)) | (np.abs(n_arr) > int(ntor))
        for arr in (bsubumnc, bsubumns, bsubvmnc, bsubvmns):
            arr[:, mask_bsub] = 0.0
        bsubumnc *= 2.0
        bsubvmnc *= 2.0
        bsubumns *= 2.0
        bsubvmns *= 2.0
        _zero_first_surface(gmnc, bmnc, bsubumnc, bsubvmnc, bsupumnc, bsupvmnc,
                            gmns, bmns, bsubumns, bsubvmns, bsupumns, bsupvmns)
        if ns > 2:
            bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]
            bsubsmnc[0, :] = 2.0 * bsubsmnc[1, :] - bsubsmnc[2, :]

    # -- jxbforce.f / mercier.f 1D diagnostics ---------------------------------
    DMerc, DShear, DCurr, DWell, DGeod, jdotb, bdotb, bdotgradv = mercier_and_jxb(
        geometry=geometry, s=s, lasym=bool(lasym), mpol=int(mpol), ntor=int(ntor),
        pres=pres, vp=np.asarray(vp, dtype=float), phips=np.asarray(phips, dtype=float),
        iotas=np.asarray(iotas, dtype=float), bsq=bsq, sqrtg=sqrtg,
        bsubu=bsubu_out, bsubv=bsubv_out, bsubs_half=bsubs_half,
        trig=trig, signgs=int(signgs),
    )

    return WoutFieldTables(
        xm_nyq=xm_nyq, xn_nyq=xn_nyq,
        gmnc=gmnc, bmnc=bmnc, bsubumnc=bsubumnc, bsubvmnc=bsubvmnc,
        bsupumnc=bsupumnc, bsupvmnc=bsupvmnc, bsubsmns=bsubsmns,
        gmns=gmns, bmns=bmns, bsubumns=bsubumns, bsubvmns=bsubvmns,
        bsupumns=bsupumns, bsupvmns=bsupvmns, bsubsmnc=bsubsmnc,
        jdotb=jdotb, bdotb=bdotb, bdotgradv=bdotgradv,
        DMerc=DMerc, DShear=DShear, DCurr=DCurr, DWell=DWell, DGeod=DGeod,
    )
