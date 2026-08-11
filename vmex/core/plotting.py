"""Publication-style diagnostic plots for new-core VMEC outputs (§5.1).

Self-contained matplotlib (Agg) figure set read from a ``wout_*.nc`` file
(or an in-memory :class:`vmex.core.wout.WoutData`):

- ``summary``   3x3 publication diagnostic set: rotational transform (full
  mesh), pressure, parallel (bootstrap) current ``<J.B>``, Mercier ``DMerc``
  and Glasser ``D_R`` with magnetic well on the right axis, a 3-D LCFS,
  a Velasco-style polar second-adiabatic-invariant map ``J(alpha, s)``, ``|B|``
  in Boozer coordinates at mid radius and on the LCFS (line contours with a
  field line of slope iota), and an equilibrium scalar card;
- ``surfaces``  flux-surface cross-sections at several zeta over one field
  period, with the magnetic axis marked;
- ``modB``      ``|B|`` contours in (zeta, theta) at mid radius and boundary;
- ``profiles``  iota / pressure / current profiles plus the ``fsqt``
  force-residual convergence trace;
- ``3d``        3-D plasma boundary colored by ``|B|`` (jet colormap).

Both stellarator-symmetric and ``lasym`` (asymmetric) equilibria are
supported: the sine/cosine partner tables (``rmns``, ``zmnc``, ``bmns``,
...) are included whenever present.  ``D_R`` follows the existing lasym
guard of :func:`vmex.core.stability.glasser_d_r_state` and is omitted (with
a panel note) for asymmetric equilibria.  All figures use the Agg backend
at ``dpi >= 200`` and are closed after saving.  The Boozer transform behind
the summary panels runs in-process (``booz_xform_jax``) so ``vmex --plot``
needs no separate ``--booz`` pass.

Public API
----------
``plot_wout(path_or_WoutData, outdir, which=(...)) -> dict[str, Path]``
``plot_boozmn(path, outdir) -> dict[str, Path]``
plus the per-figure helpers each of those dispatches to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "plot_wout",
    "plot_boozmn",
    "plot_summary",
    "plot_surfaces",
    "plot_modB",
    "plot_profiles",
    "plot_boundary_3d",
    "plot_boozmn_modB",
    "plot_boozmn_spectrum",
    "plot_boozmn_mode_profiles",
    "boozer_modB_on_surface",
]

_DPI = 200            # publication resolution for every saved PNG
_CMAP_3D = "jet"      # 3-D |B| surfaces (house style; STELLOPT convention)
_CMAP_MODB = "jet"    # non-filled |B| contour panels (booz_xform convention)
_CMAP_J = "viridis"   # J(alpha, s) invariant map (perceptually uniform)

#: Okabe-Ito colorblind-considerate cycle for 1-D profile lines.
_LINE_COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#000000", "#F0E442",
)

_MU0 = 4.0e-7 * np.pi


# ==========================================================================
# matplotlib / input handling
# ==========================================================================

def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _publication_rc() -> dict[str, Any]:
    """rcParams for a consistent publication figure style (>= 11 pt text)."""
    from cycler import cycler

    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 11.0,
        "axes.prop_cycle": cycler(color=list(_LINE_COLORS)),
        "lines.linewidth": 1.8,
        "lines.markersize": 4.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
    }


def _rc_context():
    import matplotlib

    return matplotlib.rc_context(_publication_rc())


def _as_wout(wout):
    """Accept a WoutData instance or a path to ``wout_*.nc``."""
    if hasattr(wout, "rmnc") and hasattr(wout, "xm"):
        return wout, "wout"
    from .wout import read_wout

    path = Path(wout)
    stem = path.stem
    name = stem[5:] if stem.startswith("wout_") else stem
    return read_wout(str(path)), name


def _ensure_outdir(outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# ==========================================================================
# Fourier evaluation on surfaces (file conventions: phase = m*theta - xn*phi,
# with xn already carrying the nfp factor)
# ==========================================================================

def _coeff_pair(wout, primary: str, secondary: str, s_index: int | None = None):
    """Cos/sin coefficient tables; the lasym partner is zeros when absent."""
    first = np.asarray(getattr(wout, primary), dtype=float)
    second = getattr(wout, secondary, None)
    if second is None or not bool(getattr(wout, "lasym", False)):
        second = np.zeros_like(first)
    else:
        second = np.asarray(second, dtype=float)
    if s_index is None:
        return first, second
    return first[int(s_index)], second[int(s_index)]


def _eval_modes(cos_coeff, sin_coeff, xm, xn, theta, phi, *, dtheta: int = 0, dphi: int = 0):
    """Evaluate ``sum_k [c_k cos(m_k*theta - n_k*phi) + s_k sin(...)]``.

    ``theta``/``phi`` are 1-D; the result has shape (ntheta, nphi).  With
    ``dtheta=1`` or ``dphi=1`` the corresponding first angular derivative of
    the series is returned instead.
    """
    xm = np.asarray(xm, dtype=float)
    xn = np.asarray(xn, dtype=float)
    # (mn, ntheta, nphi) phase table; grids here are small (<=260x260).
    angle = (
        xm[:, None, None] * np.asarray(theta)[None, :, None]
        - xn[:, None, None] * np.asarray(phi)[None, None, :]
    )
    cos_coeff = None if cos_coeff is None else np.asarray(cos_coeff, dtype=float)
    sin_coeff = None if sin_coeff is None else np.asarray(sin_coeff, dtype=float)
    if dtheta == 0 and dphi == 0:
        terms = [(cos_coeff, np.cos(angle)), (sin_coeff, np.sin(angle))]
    else:
        factor = xm if dtheta else -xn
        terms = [
            (None if cos_coeff is None else cos_coeff * factor.reshape((1,) * (cos_coeff.ndim - 1) + (-1,)), -np.sin(angle)),
            (None if sin_coeff is None else sin_coeff * factor.reshape((1,) * (sin_coeff.ndim - 1) + (-1,)), np.cos(angle)),
        ]
    out = None
    for coeff, basis in terms:
        if coeff is None:
            continue
        term = np.tensordot(coeff, basis, axes=(-1, 0))
        out = term if out is None else out + term
    assert out is not None
    return out


def surface_rz(wout, *, s_index: int, theta: np.ndarray, phi: np.ndarray):
    """R, Z on one full-mesh surface, shape (ntheta, nphi)."""
    rmnc, rmns = _coeff_pair(wout, "rmnc", "rmns", s_index)
    zmns, zmnc = _coeff_pair(wout, "zmns", "zmnc", s_index)
    R = _eval_modes(rmnc, rmns, wout.xm, wout.xn, theta, phi)
    Z = _eval_modes(zmnc, zmns, wout.xm, wout.xn, theta, phi)
    return R, Z


def surface_modB(wout, *, s_index: int, theta: np.ndarray, phi: np.ndarray):
    """``|B|`` on one half-mesh surface (Nyquist tables), shape (ntheta, nphi)."""
    bmnc, bmns = _coeff_pair(wout, "bmnc", "bmns", s_index)
    return _eval_modes(bmnc, bmns, wout.xm_nyq, wout.xn_nyq, theta, phi)


def axis_rz(wout, phi: np.ndarray):
    """Magnetic-axis curve R(phi), Z(phi) from the axis Fourier arrays."""
    phi = np.asarray(phi, dtype=float)
    raxis_cc = np.asarray(wout.raxis_cc, dtype=float)
    zaxis_cs = np.asarray(wout.zaxis_cs, dtype=float)
    n = np.arange(raxis_cc.size, dtype=float)
    angle = (-n[:, None] * float(wout.nfp)) * phi[None, :]
    raxis_cs = getattr(wout, "raxis_cs", None)
    zaxis_cc = getattr(wout, "zaxis_cc", None)
    raxis_cs = np.zeros_like(raxis_cc) if raxis_cs is None else np.asarray(raxis_cs, dtype=float)
    zaxis_cc = np.zeros_like(zaxis_cs) if zaxis_cc is None else np.asarray(zaxis_cc, dtype=float)
    R = np.sum(raxis_cc[:, None] * np.cos(angle) + raxis_cs[:, None] * np.sin(angle), axis=0)
    Z = np.sum(zaxis_cs[:, None] * np.sin(angle) + zaxis_cc[:, None] * np.cos(angle), axis=0)
    return R, Z


def _half_mesh_s(ns: int) -> np.ndarray:
    return (np.arange(1, ns, dtype=float) - 0.5) / float(ns - 1)


def _pi_ticks(ax, axis: str = "y") -> None:
    ticks = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


# ==========================================================================
# Glasser D_R reconstruction from wout tables (mercier.f integrals)
# ==========================================================================

def _glasser_d_r_from_wout(wout, *, ntheta: int | None = None, nzeta: int | None = None) -> dict[str, Any]:
    """Glasser--Greene--Johnson ``D_R`` profile reconstructed from a wout file.

    Re-evaluates the ``mercier.f`` surface integrals (``tpp/tbb/tjb/tjj``)
    from the wout Fourier tables on a uniform angular grid and assembles

        ``H   = S (tjb - tbb * mu0 <J.B>/<B.B>)``
        ``D_R = -DMerc + (H - S^2/2)^2 / S^2``     (0 where the shear vanishes)

    exactly as the traceable :func:`vmex.core.stability.glasser_d_r_state`
    (validated against it to ~1e-7 on the bundled decks).  The reconstruction
    is self-checking: the same integrals must reproduce the stored ``DMerc``
    profile; on mismatch (or for ``lasym`` equilibria, which the traceable
    Glasser lane rejects) the result is flagged invalid so callers can omit
    the curve instead of plotting an unvalidated one.
    """
    if bool(getattr(wout, "lasym", False)):
        return {"valid": False, "note": "not validated for lasym", "d_r": None}

    ns = int(wout.ns)
    if ns < 5:
        return {"valid": False, "note": "too few surfaces for D_R", "d_r": None}
    nfp = int(wout.nfp)
    hs = 1.0 / (ns - 1)
    ohs = 1.0 / hs
    sign_jac = float(np.sign(int(wout.signgs))) if int(wout.signgs) != 0 else 1.0

    xm_nyq = np.asarray(wout.xm_nyq, dtype=float)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=float)
    xm = np.asarray(wout.xm, dtype=float)
    xn = np.asarray(wout.xn, dtype=float)
    if ntheta is None:
        ntheta = int(min(256, max(64, 4 * (int(xm_nyq.max()) + 1))))
    if nzeta is None:
        n_over_nfp = int(np.max(np.abs(xn_nyq))) // max(nfp, 1)
        nzeta = int(min(256, max(64, 4 * (n_over_nfp + 1))))
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2.0 * np.pi * np.arange(nzeta) / (nzeta * nfp)

    pres = _MU0 * np.asarray(wout.pres, dtype=float)  # internal units (mu0*Pa)
    phips = np.asarray(wout.phips, dtype=float)
    vp = np.asarray(wout.vp, dtype=float)
    iotas = np.asarray(wout.iotas, dtype=float)
    buco = np.asarray(wout.buco, dtype=float)
    jdotb = np.asarray(wout.jdotb, dtype=float)
    bdotb = np.asarray(wout.bdotb, dtype=float)
    dmerc_stored = np.asarray(wout.DMerc, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        phip_real = 2.0 * np.pi * phips * sign_jac
        vp_real = np.zeros_like(phip_real)
        vp_real[1:] = sign_jac * (2.0 * np.pi) ** 2 * vp[1:] / phip_real[1:]
    torcur = sign_jac * 2.0 * np.pi * buco

    shear = np.zeros(ns)
    vpp = np.zeros(ns)
    presp = np.zeros(ns)
    ip = np.zeros(ns)
    phip_full_h = 0.5 * (phip_real[2:] + phip_real[1:-1])
    if np.any(phip_full_h == 0.0):
        return {"valid": False, "note": "vanishing phip", "d_r": None}
    denom = 1.0 / (hs * phip_full_h)
    shear[1:-1] = (iotas[2:] - iotas[1:-1]) * denom
    vpp[1:-1] = (vp_real[2:] - vp_real[1:-1]) * denom
    presp[1:-1] = (pres[2:] - pres[1:-1]) * denom
    ip[1:-1] = (torcur[2:] - torcur[1:-1]) * denom

    # Half-mesh real-space tables from the (already jxbforce-filtered) wout
    # Nyquist coefficients; full-mesh geometry from rmnc/zmns.
    bmag = _eval_modes(wout.bmnc, None, xm_nyq, xn_nyq, theta, zeta)
    b2 = bmag * bmag
    gsqrt = _eval_modes(wout.gmnc, None, xm_nyq, xn_nyq, theta, zeta)
    bsubu = _eval_modes(wout.bsubumnc, None, xm_nyq, xn_nyq, theta, zeta)
    bsubv = _eval_modes(wout.bsubvmnc, None, xm_nyq, xn_nyq, theta, zeta)

    # Full-mesh bsubs (sine parity) band-limited to the jxbforce force modes.
    keep = (xm_nyq <= max(int(wout.mpol) - 1, 0)) & (np.abs(xn_nyq) <= int(wout.ntor) * nfp)
    bsmns = np.asarray(wout.bsubsmns, dtype=float) * keep[None, :]
    bsubsu = _eval_modes(None, bsmns, xm_nyq, xn_nyq, theta, zeta, dtheta=1)
    bsubsv = _eval_modes(None, bsmns, xm_nyq, xn_nyq, theta, zeta, dphi=1)

    itheta = np.zeros_like(bsubu)
    izeta = np.zeros_like(bsubu)
    itheta[1:-1] = bsubsv[1:-1] - ohs * (bsubv[2:] - bsubv[1:-1])
    izeta[1:-1] = -bsubsu[1:-1] + ohs * (bsubu[2:] - bsubu[1:-1])
    bdotk = np.zeros_like(bsubu)
    bdotk[1:-1] = (
        itheta[1:-1] * 0.5 * (bsubu[2:] + bsubu[1:-1])
        + izeta[1:-1] * 0.5 * (bsubv[2:] + bsubv[1:-1])
    )

    R = _eval_modes(wout.rmnc, None, xm, xn, theta, zeta)
    Rt = _eval_modes(wout.rmnc, None, xm, xn, theta, zeta, dtheta=1)
    Rz = _eval_modes(wout.rmnc, None, xm, xn, theta, zeta, dphi=1)
    Zt = _eval_modes(None, wout.zmns, xm, xn, theta, zeta, dtheta=1)
    Zz = _eval_modes(None, wout.zmns, xm, xn, theta, zeta, dphi=1)

    two_pi_sq = (2.0 * np.pi) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bdotb != 0.0, _MU0 * jdotb / np.where(bdotb != 0.0, bdotb, 1.0), 0.0)
    dmerc_recon = np.zeros(ns)
    d_r = np.zeros(ns)
    for i in range(1, ns - 1):
        phip_full = 0.5 * (phip_real[i + 1] + phip_real[i])
        gsqrt_raw = 0.5 * (gsqrt[i] + gsqrt[i + 1])
        gsqrt_full = gsqrt_raw / phip_full
        gtt = Rt[i] ** 2 + Zt[i] ** 2
        gpp = gsqrt_full**2 / (gtt * R[i] ** 2 + (Rt[i] * Zz[i] - Rz[i] * Zt[i]) ** 2)
        b2i = 0.5 * (b2[i + 1] + b2[i])
        tpp = float(np.mean(gsqrt_full / b2i)) * two_pi_sq
        tbb = float(np.mean(b2i * gsqrt_full * gpp)) * two_pi_sq
        bdotj_norm = np.where(gsqrt_raw != 0.0, bdotk[i] / np.where(gsqrt_raw != 0.0, gsqrt_raw, 1.0), 0.0)
        jdotb_i = bdotj_norm * gpp * gsqrt_full
        tjb = float(np.mean(jdotb_i)) * two_pi_sq
        tjj = float(np.mean(jdotb_i * bdotj_norm / b2i)) * two_pi_sq

        dmerc_recon[i] = (
            0.25 * shear[i] ** 2
            - shear[i] * (tjb - ip[i] * tbb)
            + presp[i] * (vpp[i] - presp[i] * tpp) * tbb
            + tjb**2 - tbb * tjj
        )
        if shear[i] != 0.0:
            h_glasser = shear[i] * (tjb - tbb * ratio[i])
            d_r[i] = -dmerc_stored[i] + (h_glasser - 0.5 * shear[i] ** 2) ** 2 / shear[i] ** 2

    # Self-check: the reconstructed integrals must reproduce the stored DMerc.
    interior = slice(2, ns - 1)
    scale = float(np.max(np.abs(dmerc_stored[interior])))
    if scale == 0.0:
        mismatch = float(np.max(np.abs(dmerc_recon[interior])))
    else:
        mismatch = float(np.max(np.abs(dmerc_recon[interior] - dmerc_stored[interior]))) / scale
    if not np.isfinite(mismatch) or mismatch > 2.0e-2:
        return {
            "valid": False,
            "note": f"D_R self-check failed (DMerc mismatch {mismatch:.1e})",
            "d_r": None,
        }
    return {"valid": True, "note": "", "d_r": d_r, "dmerc_recon": dmerc_recon, "mismatch": mismatch}


# ==========================================================================
# In-process Boozer transform + second-adiabatic-invariant map
# ==========================================================================

def _boozer_summary_data(
    wout,
    *,
    n_surfaces: int = 9,
    mboz: int = 16,
    nboz: int = 12,
) -> dict[str, Any]:
    """Run ``booz_xform_jax`` on a spread of half-mesh surfaces.

    Returns the Boozer ``|B|`` spectra ``bmnc_b``/``bmns_b`` with shape
    ``(ns_b, nmode)``, the per-surface ``iota``/``G``/``I`` profiles, the
    computed-surface flux labels ``s_b``, and the indices of the surfaces
    closest to ``s = 0.5`` and to the LCFS for the ``|B|`` contour panels.
    """
    from booz_xform_jax import Booz_xform

    bx = Booz_xform(
        verbose=0,
        mboz=int(mboz),
        nboz=int(nboz) if int(wout.ntor) > 0 else 1,
    )
    bx.read_wout_data(wout)
    ns_in = int(bx.ns_in)
    s_in = np.asarray(bx.s_in, dtype=float)
    if s_in.size != ns_in:
        full = np.linspace(0.0, 1.0, ns_in + 1)
        s_in = 0.5 * (full[:-1] + full[1:])
    targets = np.linspace(0.1, 1.0, int(n_surfaces))
    indices = sorted(
        {int(np.argmin(np.abs(s_in - t))) for t in targets}
        | {int(np.argmin(np.abs(s_in - 0.5))), ns_in - 1}
    )
    bx.compute_surfs = indices
    bx.run()

    bmns_raw = getattr(bx, "bmns_b", None)
    bmns_b = (
        np.asarray(bmns_raw, dtype=float).T
        if bmns_raw is not None and np.size(bmns_raw) else None
    )
    s_b = np.asarray(bx.s_b, dtype=float)
    return {
        "bmnc_b": np.asarray(bx.bmnc_b, dtype=float).T,
        "bmns_b": bmns_b,
        "xm_b": np.asarray(bx.xm_b, dtype=int),
        "xn_b": np.asarray(bx.xn_b, dtype=int),
        "iota_b": np.asarray(bx.iota, dtype=float)[np.asarray(indices, dtype=int)],
        "G_b": np.asarray(bx.Boozer_G, dtype=float),
        "I_b": np.asarray(bx.Boozer_I, dtype=float),
        "s_b": s_b,
        "nfp": int(bx.nfp),
        "index_mid": int(np.argmin(np.abs(s_b - 0.5))),
        "index_lcfs": int(s_b.size - 1),
    }


def _boozer_surface_modB(booz: dict[str, Any], k: int, theta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Boozer ``|B|(theta_B, zeta_B)`` on computed surface ``k``."""
    bmns = None if booz["bmns_b"] is None else booz["bmns_b"][k]
    return _eval_modes(booz["bmnc_b"][k], bmns, booz["xm_b"], booz["xn_b"], theta, zeta)


def _j_invariant_map(
    booz: dict[str, Any],
    *,
    pitch_fraction: float = 0.5,
    nalpha: int = 96,
    points_per_period: int = 64,
    quadrature_order: int = 32,
) -> dict[str, Any]:
    """Second adiabatic invariant ``J(alpha, s)`` at fixed trapping class.

    The polar presentation and pitch convention follow Fig. 10 of Rodríguez,
    Helander & Goodman, J. Plasma Phys. 90, 905900212 (2024): on each surface,
    ``1/lambda = Bmin + lambda_n * (Bmax - Bmin)``.  Thus one normalized
    trapped-particle class ``lambda_n`` is followed radially even when the
    trapping interval changes with ``s``.  Omnigenity makes ``J`` independent
    of ``alpha``, so its contours in ``x=s*cos(alpha)``, ``y=s*sin(alpha)``
    become concentric circles.  ``J`` is normalized to ``J/(v R0)`` and the
    bounce integrals reuse the differentiable sine-mapped Gauss-Legendre
    kernel of :func:`vmex.core.bounce.bounce_action`, also used by DESC.
    """
    from .bounce import bounce_action_from_boozer

    bmnc_b = booz["bmnc_b"]
    nsurf = int(bmnc_b.shape[0])
    nfp = int(booz["nfp"])
    iota_b = booz["iota_b"]

    # Surface-local trapping ranges implement the fixed-lambda_n convention.
    theta = np.linspace(0.0, 2.0 * np.pi, 61)
    zeta = np.linspace(0.0, 2.0 * np.pi / nfp, 61)
    b_all = np.stack([_boozer_surface_modB(booz, k, theta, zeta) for k in range(nsurf)])
    b_min = b_all.min(axis=(1, 2)); b_max = b_all.max(axis=(1, 2))
    if (not np.all(np.isfinite(b_min)) or not np.all(np.isfinite(b_max))
            or np.any(b_min <= 0.0) or np.any(b_max <= b_min)):
        raise ValueError("Boozer |B| range is degenerate; cannot choose a pitch")
    b_star = b_min + float(pitch_fraction) * (b_max - b_min)
    pitch = 1.0 / b_star

    # Trace enough field periods to close at least one poloidal transit even
    # for small-iota / axisymmetric-boundary decks (well length ~ 2*pi/iota).
    iota_typical = float(np.median(np.abs(iota_b)))
    num_periods = int(min(40, max(2, np.ceil(1.2 * nfp * (1.0 + 1.0 / max(iota_typical, 0.2))))))

    alpha = np.linspace(0.0, 2.0 * np.pi, int(nalpha), endpoint=False)
    j_map = np.full((nsurf, alpha.size), np.nan)
    for k in range(nsurf):  # per-surface loop keeps the phase tables small
        out = bounce_action_from_boozer(
            bmnc_b=bmnc_b[k : k + 1],
            xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=iota_b[k : k + 1],
            G_b=booz["G_b"][k : k + 1], I_b=booz["I_b"][k : k + 1],
            nfp=nfp, alpha=alpha, pitch=pitch[k : k + 1],
            points_per_period=int(points_per_period),
            num_periods=num_periods,
            bmns_b=None if booz["bmns_b"] is None else booz["bmns_b"][k : k + 1],
            quadrature_order=int(quadrature_order),
        )
        action = np.asarray(out["action"])[0, :, 0, :]       # (nalpha, nwells)
        usable = np.asarray(out["usable_mask"])[0, :, 0, :]
        count = usable.sum(axis=-1)
        total = np.where(usable, np.where(np.isfinite(action), action, 0.0), 0.0).sum(axis=-1)
        j_map[k] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return {
        "alpha": alpha,
        "s_b": booz["s_b"],
        "j_map": j_map,
        "pitch": pitch,
        "pitch_fraction": float(pitch_fraction),
        "b_min": b_min,
        "b_max": b_max,
    }


# ==========================================================================
# Summary figure
# ==========================================================================

_S_LABEL = r"$s = \psi/\psi_b$"


def _profile_panel(ax, x, y, *, xlabel: str, ylabel: str, title: str, color=None) -> None:
    ax.plot(x, y, ".-", color=color or _LINE_COLORS[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _stability_panel(ax, wout, d_r_info: dict[str, Any], *, s_plot_ignore: float):
    """DMerc and D_R with magnetic well on a color-matched right axis."""
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    dmerc = np.asarray(wout.DMerc, dtype=float)
    lo = max(2, int(round(s_plot_ignore * ns)))
    sl = slice(lo, ns - 1)
    lines = [ax.plot(
        s[sl], dmerc[sl], marker="o", markersize=3.5, linestyle="-",
        color=_LINE_COLORS[0], label=r"$D_{Merc}$ ($>0$ stable)")[0]]
    finite = dmerc[sl][np.isfinite(dmerc[sl])]
    peak = float(np.max(np.abs(finite))) if finite.size else 1.0
    if d_r_info.get("valid"):
        d_r = np.asarray(d_r_info["d_r"], dtype=float)
        lines.append(ax.plot(
            s[sl], d_r[sl], marker="s", markersize=3.2, linestyle="--",
            color=_LINE_COLORS[1], label=r"$D_R$ ($\leq 0$ stable)")[0])
        finite_r = d_r[sl][np.isfinite(d_r[sl])]
        if finite_r.size:
            peak = max(peak, float(np.max(np.abs(finite_r))))
    else:
        lines.append(ax.plot(
            [], [], " ", label=f"$D_R$: {d_r_info.get('note', 'unavailable')}")[0])
    if peak > 30.0:
        ax.set_yscale("symlog", linthresh=max(1.0e-3, 1.0e-3 * peak))
    ax.axhline(0.0, color="0.4", linewidth=0.8, zorder=1)
    ax.set_xlabel(_S_LABEL)
    ax.set_ylabel(r"$D_{Merc}$, $D_R$")
    ax.set_xlim(0.0, 1.0)

    well_ax = ax.twinx()
    vp = np.abs(np.asarray(wout.vp, dtype=float))[1:]
    s_half = _half_mesh_s(int(wout.ns))
    v0 = 1.5 * vp[0] - 0.5 * vp[1]
    v1 = 1.5 * vp[-1] - 0.5 * vp[-2]
    if v0 != 0.0:
        well = 100.0 * (v0 - vp) / v0
        lines.append(well_ax.plot(
            s_half, well, marker="^", markersize=3.2, linestyle="-.",
            color=_LINE_COLORS[2], label=rf"magnetic well ({100.0 * (v0 - v1) / v0:.2f}% edge)")[0])
    else:
        lines.append(well_ax.plot([], [], " ", label="magnetic well: $V'(0)=0$")[0])
    well_ax.set_ylabel("magnetic well [%]", color=_LINE_COLORS[2])
    well_ax.tick_params(axis="y", colors=_LINE_COLORS[2])
    well_ax.spines["right"].set_color(_LINE_COLORS[2])
    ax.set_title("Mercier, resistive interchange, and magnetic well")
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    return well_ax


def _j_map_panel(ax, fig, j_info: dict[str, Any], r_major: float) -> None:
    """Velasco-style polar map of ``J/(v R0)`` in ``x=s cos(alpha), y=s sin(alpha)``."""
    length_scale = abs(float(r_major))
    if not np.isfinite(length_scale) or length_scale <= np.finfo(float).tiny:
        length_scale = 1.0
    j_norm = j_info["j_map"] / length_scale
    masked = np.ma.masked_invalid(j_norm)
    if masked.count() < 4 or np.ptp(masked.compressed()) == 0.0:
        ax.text(
            0.5, 0.5, "no trapped-particle wells\nresolved at this pitch",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title("second adiabatic invariant")
        ax.set_xlabel(r"$s\cos\alpha$")
        ax.set_ylabel(r"$s\sin\alpha$")
        return
    alpha = np.concatenate([j_info["alpha"], j_info["alpha"][:1] + 2.0 * np.pi])
    periodic = np.ma.concatenate([masked, masked[:, :1]], axis=1)
    alpha2d, s2d = np.meshgrid(alpha, j_info["s_b"])
    x, y = s2d * np.cos(alpha2d), s2d * np.sin(alpha2d)
    levels = np.linspace(float(periodic.min()), float(periodic.max()), 15)
    filled = ax.contourf(x, y, periodic, levels=levels, cmap=_CMAP_J, extend="both")
    ax.contour(x, y, periodic, levels=levels, colors="0.25", linewidths=0.35, alpha=0.65)
    fig.colorbar(filled, ax=ax, pad=0.02, label=r"$J\,/\,(v R_0)$")
    radius = max(1.0, float(np.max(j_info["s_b"])))
    ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.75)
    ax.axvline(0.0, color="white", linewidth=0.6, alpha=0.75)
    ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$s\cos\alpha$"); ax.set_ylabel(r"$s\sin\alpha$")
    ax.set_title(rf"second adiabatic invariant, $\lambda_n={j_info['pitch_fraction']:.2f}$")


def _boozer_modB_panel(ax, fig, booz: dict[str, Any], k: int, *, title: str) -> None:
    """Non-filled ``|B|`` contours in Boozer angles with one field line."""
    nfp = int(booz["nfp"])
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    zeta = np.linspace(0.0, 2.0 * np.pi / nfp, 181)
    B = _boozer_surface_modB(booz, k, theta, zeta)
    zeta2d, theta2d = np.meshgrid(zeta, theta)
    cs = ax.contour(zeta2d, theta2d, B, levels=21, cmap=_CMAP_MODB, linewidths=1.1)
    fig.colorbar(cs, ax=ax, pad=0.02, label=r"$|B|$ [T]")
    iota_k = float(booz["iota_b"][k])
    line_theta = np.mod(iota_k * zeta, 2.0 * np.pi)
    # NaN-out the wrap discontinuities so the mod jump draws no vertical bar.
    jump = np.abs(np.diff(line_theta)) > np.pi
    line_theta[1:][jump] = np.nan
    ax.plot(
        zeta, line_theta, color="black", linewidth=1.6,
        linestyle="--", label=rf"field line ($\iota = {iota_k:.2f}$)",
    )
    ax.set_xlim(0.0, 2.0 * np.pi / nfp)
    ax.set_ylim(0.0, 2.0 * np.pi)
    _pi_ticks(ax, "y")
    ax.set_xlabel(r"Boozer toroidal angle $\zeta_B$")
    ax.set_ylabel(r"Boozer poloidal angle $\theta_B$")
    ax.set_title(title)
    # The wrapped field line hugs the bottom-left for positive iota and the
    # top for negative iota; park the legend in the opposite corner.
    ax.legend(loc="upper left" if iota_k >= 0.0 else "lower left", framealpha=0.85)


def _fmt_compact(value: float) -> str:
    """Compact scientific format: 0 stays ``0``, exponents lose zero padding."""
    if value == 0.0:
        return "0"
    text = f"{value:.2e}"
    mantissa, exponent = text.split("e")
    return f"{mantissa}e{int(exponent)}"


def _scalar_card_panel(ax, wout) -> None:
    """Equilibrium scalar card (threed1-style global quantities)."""
    ax.set_axis_off()
    iotaf = np.asarray(wout.iotaf, dtype=float)
    rows = [
        ("field periods", f"{int(wout.nfp)}"),
        ("resolution", f"ns={int(wout.ns)}, mpol={int(wout.mpol)}, ntor={int(wout.ntor)}"),
        ("aspect ratio", f"{float(wout.aspect):.3f}"),
        (r"$R_0$ / $a$ [m]", f"{float(wout.Rmajor_p):.3f} / {float(wout.Aminor_p):.3f}"),
        (r"volume [m$^3$]", f"{float(wout.volume_p):.3f}"),
        (r"$\langle |B| \rangle$ [T]", f"{float(wout.volavgB):.3f}"),
        (r"$\beta$ total", _fmt_compact(float(wout.betatotal))),
        (r"$\beta$ pol / tor", f"{_fmt_compact(float(wout.betapol))} / {_fmt_compact(float(wout.betator))}"),
        ("toroidal current [A]", _fmt_compact(float(wout.ctor))),
        (r"$\iota$ axis / edge", f"{float(iotaf[0]):.4f} / {float(iotaf[-1]):.4f}"),
        ("asymmetric", "yes" if bool(getattr(wout, "lasym", False)) else "no"),
    ]
    ax.text(
        0.0, 0.99, "equilibrium summary", transform=ax.transAxes,
        fontsize=12.0, fontweight="bold", va="top",
    )
    step = 0.86 / max(len(rows) - 1, 1)
    for i, (key, value) in enumerate(rows):
        y = 0.88 - step * i
        ax.text(0.0, y, key, transform=ax.transAxes, fontsize=11.0, va="top", color="0.25")
        ax.text(0.46, y, value, transform=ax.transAxes, fontsize=11.0, va="top")


def _boundary_3d_panel(ax, wout, *, ntheta: int, nzeta: int):
    """Draw a smooth jet-mapped LCFS on an existing 3-D axis."""
    import matplotlib
    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = matplotlib.colormaps[_CMAP_3D]
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta))
    phi = np.linspace(0.0, 2.0 * np.pi, int(nzeta))
    R, Z = surface_rz(wout, s_index=int(wout.ns) - 1, theta=theta, phi=phi)
    B = surface_modB(wout, s_index=int(wout.ns) - 1, theta=theta, phi=phi)
    phi2d = np.meshgrid(phi, theta)[0]
    X, Y = R * np.cos(phi2d), R * np.sin(phi2d)
    norm = Normalize(float(B.min()), float(B.max()))
    ax.plot_surface(
        X, Y, Z, facecolors=cmap(norm(B)), rstride=1, cstride=1,
        antialiased=False, linewidth=0.0, shade=False,
    )
    scale = 0.7 * max(np.abs(X).max(), np.abs(Y).max())
    ax.auto_scale_xyz([-scale, scale], [-scale, scale], [-scale, scale])
    ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
    return cm.ScalarMappable(cmap=cmap, norm=norm)


def _summary_figure(wout, *, s_plot_ignore: float = 0.2):
    """Build the 3x3 summary figure; returns ``(fig, meta)`` for inspection."""
    plt = _import_matplotlib()
    wout, _ = _as_wout(wout)
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    meta: dict[str, Any] = {}

    with _rc_context():
        fig = plt.figure(figsize=(15.0, 11.5), layout="constrained")
        grid = fig.add_gridspec(3, 3)
        axes = np.empty((3, 3), dtype=object)
        for row in range(3):
            for column in range(3):
                projection = "3d" if (row, column) == (1, 1) else None
                axes[row, column] = fig.add_subplot(grid[row, column], projection=projection)

        # 1. rotational transform -- full mesh only.
        _profile_panel(
            axes[0, 0], s, np.asarray(wout.iotaf, dtype=float),
            xlabel=_S_LABEL, ylabel=r"$\iota$", title="rotational transform (full mesh)",
        )

        # pressure (kept from the classic threed1 set).
        _profile_panel(
            axes[0, 1], s, 1.0e-3 * np.asarray(wout.presf, dtype=float),
            xlabel=_S_LABEL, ylabel=r"$p$ [kPa]", title="pressure",
            color=_LINE_COLORS[1],
        )

        # 5. parallel (bootstrap) current profile <J.B>.
        _profile_panel(
            axes[0, 2], s, 1.0e-3 * np.asarray(wout.jdotb, dtype=float),
            xlabel=_S_LABEL, ylabel=r"$\langle \mathbf{J}\cdot\mathbf{B} \rangle$ [kA T/m$^2$]",
            title=r"parallel (bootstrap) current", color=_LINE_COLORS[2],
        )

        # Stability profiles share one panel; right-axis color identifies W.
        d_r_info = _glasser_d_r_from_wout(wout)
        meta["d_r"] = d_r_info
        meta["well_axis"] = _stability_panel(
            axes[1, 0], wout, d_r_info, s_plot_ignore=s_plot_ignore)

        # The summary includes the LCFS overview; --plot also writes it alone.
        summary_nzeta = min(480, max(240, 80 * int(wout.nfp)))
        boundary_map = _boundary_3d_panel(
            axes[1, 1], wout, ntheta=120, nzeta=summary_nzeta)
        axes[1, 1].set_title("3-D plasma boundary")
        fig.colorbar(
            boundary_map, ax=axes[1, 1], pad=0.0, fraction=0.045,
            shrink=0.72, label=r"$|B|$ [T]",
        )

        # 2 + 4. Boozer-based panels: J(alpha, s) map and |B| contours.
        booz_note = ""
        try:
            booz = _boozer_summary_data(wout)
        except Exception as exc:  # noqa: BLE001 - summary stays usable without booz
            booz = None
            booz_note = f"Boozer transform unavailable:\n{type(exc).__name__}"
        if booz is not None:
            try:
                j_info = _j_invariant_map(booz)
                meta["j_map"] = j_info
                _j_map_panel(axes[1, 2], fig, j_info, float(wout.Rmajor_p))
            except Exception as exc:  # noqa: BLE001
                axes[1, 2].text(
                    0.5, 0.5, f"J map unavailable:\n{type(exc).__name__}",
                    ha="center", va="center", transform=axes[1, 2].transAxes,
                )
                axes[1, 2].set_title("second adiabatic invariant")
                axes[1, 2].set_xlabel(r"$s\cos\alpha$")
                axes[1, 2].set_ylabel(r"$s\sin\alpha$")
                axes[1, 2].set_aspect("equal", adjustable="box")
            s_mid = float(booz["s_b"][booz["index_mid"]])
            _boozer_modB_panel(
                axes[2, 1], fig, booz, booz["index_mid"],
                title=rf"$|B|$ in Boozer angles, $s = {s_mid:.2f}$",
            )
            _boozer_modB_panel(
                axes[2, 2], fig, booz, booz["index_lcfs"],
                title=r"$|B|$ in Boozer angles, LCFS",
            )
            meta["booz"] = booz
        else:
            axes[1, 2].text(
                0.5, 0.5, booz_note, ha="center", va="center", transform=axes[1, 2].transAxes)
            axes[1, 2].set_title("second adiabatic invariant")
            axes[1, 2].set_xlabel(r"$s\cos\alpha$"); axes[1, 2].set_ylabel(r"$s\sin\alpha$")
            axes[1, 2].set_aspect("equal", adjustable="box")
            for ax, title in (
                (axes[2, 1], r"$|B|$ in Boozer angles, $s = 0.5$"),
                (axes[2, 2], r"$|B|$ in Boozer angles, LCFS"),
            ):
                ax.text(0.5, 0.5, booz_note, ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                ax.set_xlabel(r"Boozer toroidal angle $\zeta_B$")
                ax.set_ylabel(r"Boozer poloidal angle $\theta_B$")

        # 6. equilibrium scalar card (threed1-style global quantities).
        _scalar_card_panel(axes[2, 0], wout)

        meta["axes"] = {
            "iota": axes[0, 0], "pressure": axes[0, 1], "jdotb": axes[0, 2],
            "stability": axes[1, 0], "boundary_3d": axes[1, 1], "j_invariant": axes[1, 2],
            "card": axes[2, 0], "boozer_mid": axes[2, 1], "boozer_lcfs": axes[2, 2],
        }
    return fig, meta


def plot_summary(wout, out_path: str | Path, *, s_plot_ignore: float = 0.2) -> Path:
    """Publication summary figure (see :func:`_summary_figure` for panels)."""
    plt = _import_matplotlib()
    fig, _meta = _summary_figure(wout, s_plot_ignore=s_plot_ignore)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


# ==========================================================================
# Remaining per-figure plotters (wout)
# ==========================================================================

def plot_surfaces(
    wout,
    out_path: str | Path,
    *,
    nzeta: int = 8,
    nradii: int = 8,
    ntheta: int = 160,
) -> Path:
    """Flux-surface cross-sections at ``nzeta`` slices over one field period."""
    plt = _import_matplotlib()
    wout, _ = _as_wout(wout)
    ns, nfp = int(wout.ns), int(wout.nfp)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)
    iradii = np.unique(np.round(np.linspace(0, ns - 1, nradii)).astype(int))
    Raxis, Zaxis = axis_rz(wout, phi)

    ncols = min(4, nzeta)
    nrows = int(np.ceil(nzeta / ncols))
    with _rc_context():
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False,
            layout="constrained",
        )
        flat = axes.ravel()
        for iz in range(nzeta):
            ax = flat[iz]
            for irad in iradii:
                R, Z = surface_rz(wout, s_index=int(irad), theta=theta, phi=phi[iz : iz + 1])
                ax.plot(R[:, 0], Z[:, 0], "-", linewidth=1.0)
            ax.plot(Raxis[iz], Zaxis[iz], "x", color="black", markersize=6)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$R$ [m]")
            ax.set_ylabel(r"$Z$ [m]")
            ax.set_title(rf"$\phi$ = {phi[iz]:.2f}")
        for iz in range(nzeta, flat.size):
            flat[iz].set_axis_off()
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return out_path


def plot_modB(
    wout,
    out_path: str | Path,
    *,
    ntheta: int = 90,
    nphi: int = 180,
) -> Path:
    """``|B|`` line contours in (phi, theta) at mid radius and the boundary."""
    plt = _import_matplotlib()
    wout, _ = _as_wout(wout)
    ns = int(wout.ns)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), nphi)
    phi2d, theta2d = np.meshgrid(phi, theta)

    with _rc_context():
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), layout="constrained")
        for ax, irad, title in (
            (axes[0], ns // 2, "mid radius"),
            (axes[1], ns - 1, "plasma boundary"),
        ):
            B = surface_modB(wout, s_index=int(irad), theta=theta, phi=phi)
            cf = ax.contour(phi2d, theta2d, B, levels=25, cmap=_CMAP_MODB, linewidths=1.0)
            fig.colorbar(cf, ax=ax, label=r"$|B|$ [T]")
            ax.set_title(f"|B| on {title} (one field period)")
            ax.set_xlabel(r"toroidal angle $\phi$")
            ax.set_ylabel(r"poloidal angle $\theta$")
            _pi_ticks(ax, "y")
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return out_path


def plot_profiles(wout, out_path: str | Path) -> Path:
    """Radial profiles (iota, pressure, currents) and fsqt convergence."""
    plt = _import_matplotlib()
    wout, _ = _as_wout(wout)
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    s_half = _half_mesh_s(ns)

    with _rc_context():
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), layout="constrained")

        ax = axes[0, 0]
        ax.plot(s, np.asarray(wout.iotaf, dtype=float), ".-")
        ax.set_ylabel(r"$\iota$")
        ax.set_title("rotational transform (full mesh)")

        ax = axes[0, 1]
        ax.plot(s, np.asarray(wout.presf, dtype=float), ".-", label="presf (full)")
        ax.plot(s_half, np.asarray(wout.pres, dtype=float)[1:], ".", ms=3, label="pres (half)")
        ax.set_ylabel("pressure [Pa]")
        ax.legend()

        ax = axes[0, 2]
        ax.plot(s, np.asarray(wout.jcuru, dtype=float), ".-", label="jcuru")
        ax.plot(s, np.asarray(wout.jcurv, dtype=float), ".-", label="jcurv")
        ax.set_ylabel("current density [A]")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(s_half, np.asarray(wout.buco, dtype=float)[1:], ".-", label="buco")
        ax.plot(s_half, np.asarray(wout.bvco, dtype=float)[1:], ".-", label="bvco")
        ax.set_ylabel(r"$\langle B_u \rangle$, $\langle B_v \rangle$")
        ax.legend()

        ax = axes[1, 1]
        phi_flux = np.asarray(wout.phi, dtype=float)
        chi_flux = np.asarray(wout.chi, dtype=float)
        ax.plot(s, phi_flux, ".-", label=r"$\phi$ (toroidal)")
        ax.plot(s, chi_flux, ".-", label=r"$\chi$ (poloidal)")
        ax.set_ylabel("flux [Wb]")
        ax.legend()

        for ax in axes.ravel()[:5]:
            ax.set_xlabel(_S_LABEL)

        # fsqt convergence trace (VMEC stores up to 100 sampled residuals).
        ax = axes[1, 2]
        fsqt = np.asarray(getattr(wout, "fsqt", np.zeros(0)), dtype=float).ravel()
        wdot = np.asarray(getattr(wout, "wdot", np.zeros(0)), dtype=float).ravel()
        mask = fsqt > 0.0
        if np.any(mask):
            last = int(np.max(np.nonzero(mask)[0])) + 1
            it = np.arange(1, last + 1)
            ax.semilogy(it, np.maximum(fsqt[:last], 1e-30), ".-", label="fsqt")
            wmask = wdot[:last] > 0.0
            if np.any(wmask):
                ax.semilogy(it[wmask], wdot[:last][wmask], ".-", alpha=0.7, label="wdot")
            ftolv = float(getattr(wout, "ftolv", 0.0) or 0.0)
            if ftolv > 0.0:
                ax.axhline(ftolv, color="k", ls="--", lw=0.8)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "no fsqt history", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("stored iteration sample")
        ax.set_ylabel("force residual")
        ax.set_title("convergence (fsqt)")

        out_path = Path(out_path)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return out_path


def plot_boundary_3d(
    wout,
    out_path: str | Path,
    *,
    ntheta: int = 180,
    nzeta: int | None = None,
) -> Path:
    """3-D plasma boundary colored by ``|B|`` (full torus, jet colormap)."""
    plt = _import_matplotlib()
    wout, _ = _as_wout(wout)
    nfp = int(wout.nfp)
    if nzeta is None:
        nzeta = min(720, max(360, 120 * nfp))

    with _rc_context():
        fig = plt.figure(figsize=(5.6, 4.8), frameon=False)
        ax = fig.add_subplot(111, projection="3d")
        sm = _boundary_3d_panel(ax, wout, ntheta=ntheta, nzeta=int(nzeta))
        cax = fig.add_axes([0.21, 0.86, 0.60, 0.03])
        fig.colorbar(sm, orientation="horizontal", cax=cax).set_label(r"$|B|$ [T]")
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    return out_path


# ==========================================================================
# plot_wout dispatcher
# ==========================================================================

_WOUT_FIGURES = {
    "summary": ("summary", plot_summary),
    "surfaces": ("surfaces", plot_surfaces),
    "modB": ("modB", plot_modB),
    "profiles": ("profiles", plot_profiles),
    "3d": ("boundary3d", plot_boundary_3d),
}


def plot_wout(
    wout,
    outdir: str | Path,
    which: Sequence[str] = ("summary", "surfaces", "modB", "profiles", "3d"),
    *,
    name: str | None = None,
) -> dict[str, Path]:
    """Write the requested diagnostic figures for a WOUT file.

    Parameters
    ----------
    wout:
        Path to ``wout_*.nc`` or a :class:`~vmex.core.wout.WoutData`.
    outdir:
        Output directory (created if missing).
    which:
        Any subset of ``("summary", "surfaces", "modB", "profiles", "3d")``.
    name:
        Basename prefix for the figures (default: case name from the path).

    Returns a mapping from figure key to the written PNG path.
    """
    data, default_name = _as_wout(wout)
    label = name or default_name
    outdir = _ensure_outdir(outdir)
    unknown = [key for key in which if key not in _WOUT_FIGURES]
    if unknown:
        raise ValueError(f"Unknown figure keys {unknown}; choose from {sorted(_WOUT_FIGURES)}")
    results: dict[str, Path] = {}
    for key in which:
        suffix, fn = _WOUT_FIGURES[key]
        results[key] = fn(data, outdir / f"{label}_{suffix}.png")
    return results


# ==========================================================================
# Boozer (boozmn) figures
# ==========================================================================

def _load_boozmn(boozmn):
    """Accept a Booz_xform object or a ``boozmn_*.nc`` path."""
    if hasattr(boozmn, "bmnc_b") and hasattr(boozmn, "xm_b"):
        return boozmn
    try:
        from booz_xform_jax import Booz_xform
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Boozer plotting requires booz_xform_jax; run `pip install booz_xform_jax`."
        ) from exc
    bx = Booz_xform(verbose=0)
    bx.read_boozmn(str(boozmn))
    return bx


def _boozer_amplitudes(bx):
    bmnc = np.asarray(bx.bmnc_b, dtype=float)
    bmns_raw = getattr(bx, "bmns_b", None)
    bmns = np.zeros_like(bmnc) if bmns_raw is None or np.size(bmns_raw) == 0 else np.asarray(bmns_raw, dtype=float)
    if bmns.shape != bmnc.shape:
        bmns = np.zeros_like(bmnc)
    amp = np.sqrt(bmnc**2 + bmns**2)
    return amp, bmnc, bmns, np.asarray(bx.xm_b, dtype=int), np.asarray(bx.xn_b, dtype=int)


def _boozer_mode_group(m: int, n: int, nfp: int) -> tuple[str, str]:
    if m == 0 and n == 0:
        return "B00", "black"
    if n == 0:
        return "QA (n=0)", "tab:green"
    if m == 0:
        return "Mirror (m=0)", "darkgoldenrod"
    if n == nfp * m:
        return "QH + (n=NFP m)", "tab:purple"
    if n == -nfp * m:
        return "QH - (n=-NFP m)", "tab:cyan"
    return "Other", "tab:red"


def _boozer_modB_grid(bx, *, js: int, ntheta: int = 90, nphi: int = 180):
    _amp, bmnc, bmns, xm, xn = _boozer_amplitudes(bx)
    nfp = int(getattr(bx, "nfp", 1) or 1)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi / nfp, nphi)
    B = _eval_modes(bmnc[:, js], bmns[:, js], xm, xn, theta, phi)
    return theta, phi, B


def boozer_modB_on_surface(boozmn, *, s_index: int = -1, ntheta: int = 90, nphi: int = 180):
    """Boozer ``|B|(theta_B, phi_B)`` on one surface of a Boozer transform.

    Accepts a ``booz_xform_jax.Booz_xform`` object or a ``boozmn_*.nc`` path
    (as produced by :func:`vmex.core.boozer.run_booz_xform`).  ``s_index``
    indexes the computed Boozer surfaces; ``-1`` (the default) selects the
    outermost surface, i.e. ``|B|`` in Boozer coordinates on the LCFS.

    Returns ``(theta_B, phi_B, B)`` where ``B`` has shape ``(ntheta, nphi)``
    over one field period, suitable for a ``jet`` contour plot.
    """
    bx = _load_boozmn(boozmn)
    ns_b = int(np.asarray(bx.bmnc_b).shape[1])
    if ns_b < 1:
        raise ValueError("Boozer output contains no computed surfaces")
    js = int(s_index) + (ns_b if s_index < 0 else 0)
    if js < 0 or js >= ns_b:
        raise IndexError(f"s_index {s_index} outside Boozer range 0..{ns_b - 1}")
    return _boozer_modB_grid(bx, js=js, ntheta=ntheta, nphi=nphi)


def plot_boozmn_modB(
    boozmn, out_path: str | Path, *, ntheta: int = 90, nphi: int = 180,
    cmap: str = _CMAP_MODB,
) -> Path:
    """Boozer-coordinate ``|B|`` line contours at mid radius and the edge.

    ``cmap`` selects the contour colormap (default ``jet``, the STELLOPT /
    booz_xform convention).  Contours are always non-filled.
    """
    plt = _import_matplotlib()
    bx = _load_boozmn(boozmn)
    ns_b = int(np.asarray(bx.bmnc_b).shape[1])
    if ns_b < 1:
        raise ValueError("Boozer output contains no computed surfaces")
    selected = [("mid radius", ns_b // 2), ("outermost surface", ns_b - 1)]
    if selected[0][1] == selected[1][1]:
        selected = selected[1:]

    with _rc_context():
        fig, axes = plt.subplots(
            1, len(selected), figsize=(6.6 * len(selected), 4.6), squeeze=False,
            layout="constrained",
        )
        for ax, (title, js) in zip(axes[0], selected):
            theta, phi, B = _boozer_modB_grid(bx, js=js, ntheta=ntheta, nphi=nphi)
            phi2d, theta2d = np.meshgrid(phi, theta)
            cs = ax.contour(phi2d, theta2d, B, levels=24, cmap=cmap, linewidths=1.0)
            fig.colorbar(cs, ax=ax, label=r"$|B|$ [T]")
            ax.set_title(title)
            ax.set_xlabel(r"Boozer toroidal angle $\phi_B$")
            ax.set_ylabel(r"Boozer poloidal angle $\theta_B$")
            _pi_ticks(ax, "y")
            ax.set_ylim(0, 2 * np.pi)
        fig.suptitle("Boozer-coordinate |B| contours")
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return out_path


def plot_boozmn_mode_profiles(boozmn, out_path: str | Path, *, max_modes: int = 80) -> Path:
    """Radial Boozer ``|B|`` mode amplitudes grouped by symmetry family."""
    plt = _import_matplotlib()
    bx = _load_boozmn(boozmn)
    amp, _bmnc, _bmns, xm, xn = _boozer_amplitudes(bx)
    nfp = int(getattr(bx, "nfp", 1) or 1)
    s_b = np.asarray(getattr(bx, "s_b", ()), dtype=float)
    if s_b.size != amp.shape[1]:
        s_b = np.linspace(0.0, 1.0, amp.shape[1])
    order = np.argsort(-amp[:, -1])[: max(1, min(int(max_modes), amp.shape[0]))]

    with _rc_context():
        fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.2), layout="constrained")
        seen: set[str] = set()
        for idx in order:
            group, color = _boozer_mode_group(int(xm[idx]), int(xn[idx]), nfp)
            label = group if group not in seen else None
            seen.add(group)
            ax.semilogy(
                s_b, np.maximum(amp[idx], 1e-16), color=color,
                alpha=0.9 if label else 0.35, linewidth=1.6 if group != "Other" else 0.9,
                label=label,
            )
        ax.set_xlabel("normalized toroidal flux s")
        ax.set_ylabel(r"$|B_{mn}|$ [T]")
        ax.set_title("Boozer |B| radial spectra by symmetry family")
        if s_b.size > 1 and not np.isclose(s_b.min(), s_b.max()):
            ax.set_xlim(float(s_b.min()), float(s_b.max()))
        ax.legend(loc="best")
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return out_path


def plot_boozmn_spectrum(boozmn, out_path: str | Path, *, surface_index: int = -1, nmodes: int = 40) -> Path:
    """Largest Boozer ``|B|`` Fourier amplitudes on one surface (log bar chart)."""
    plt = _import_matplotlib()
    bx = _load_boozmn(boozmn)
    amp, _bmnc, _bmns, xm, xn = _boozer_amplitudes(bx)
    nfp = int(getattr(bx, "nfp", 1) or 1)
    ns_b = int(amp.shape[1])
    js = int(surface_index) + (ns_b if surface_index < 0 else 0)
    if js < 0 or js >= ns_b:
        raise IndexError(f"surface_index {surface_index} outside Boozer range 0..{ns_b - 1}")
    order = np.argsort(-amp[:, js])[: max(1, min(int(nmodes), amp.shape[0]))]
    colors = [_boozer_mode_group(int(xm[i]), int(xn[i]), nfp)[1] for i in order]

    with _rc_context():
        fig, ax = plt.subplots(1, 1, figsize=(max(8.2, 0.24 * len(order)), 5.0), layout="constrained")
        x = np.arange(len(order))
        ax.bar(x, np.maximum(amp[order, js], 1e-16), color=colors, width=0.8)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"({int(xm[i])},{int(xn[i])})" for i in order], rotation=75, ha="right", fontsize=9)
        ax.set_xlabel("Boozer mode (m, n)")
        ax.set_ylabel(r"$|B_{mn}|$ [T]")
        ax.set_title(f"Boozer |B| spectrum, surface {js + 1}/{ns_b}")
        legend = {}
        for i in order:
            group, color = _boozer_mode_group(int(xm[i]), int(xn[i]), nfp)
            legend[group] = color
        ax.legend(
            handles=[plt.Line2D([0], [0], color=c, lw=4, label=g) for g, c in legend.items()],
            loc="best",
        )
        ax.grid(True, axis="y", alpha=0.25)
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=_DPI)
        plt.close(fig)
    return out_path


def plot_boozmn(
    boozmn_path: str | Path,
    outdir: str | Path,
    which: Iterable[str] = ("modB", "mode_profiles", "spectrum"),
    *,
    name: str | None = None,
) -> dict[str, Path]:
    """Write Boozer diagnostic figures for a ``boozmn_*.nc`` file.

    Returns a mapping from figure key (``modB``, ``mode_profiles``,
    ``spectrum``) to the written PNG path.
    """
    boozmn_path = Path(boozmn_path)
    label = name or boozmn_path.stem
    outdir = _ensure_outdir(outdir)
    bx = _load_boozmn(boozmn_path)
    plotters = {
        "modB": (plot_boozmn_modB, f"{label}_modB.png"),
        "mode_profiles": (plot_boozmn_mode_profiles, f"{label}_mode_profiles.png"),
        "spectrum": (plot_boozmn_spectrum, f"{label}_spectrum.png"),
    }
    results: dict[str, Path] = {}
    for key in which:
        if key not in plotters:
            raise ValueError(f"Unknown boozmn figure key {key!r}; choose from {sorted(plotters)}")
        fn, filename = plotters[key]
        results[key] = fn(bx, outdir / filename)
    return results
