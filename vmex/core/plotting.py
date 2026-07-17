"""Publication-style diagnostic plots for new-core VMEC outputs (§5.1).

Self-contained matplotlib (Agg) port of the figure set from the legacy
``vmex.plotting`` module, reading everything from a ``wout_*.nc`` file
(or an in-memory :class:`vmex.core.wout.WoutData`):

- ``summary``   3x3 overview: iota, pressure, buco/bvco, jcuru/jcurv, DMerc,
  and ``|B|`` line contours at mid radius and at the plasma boundary;
- ``surfaces``  flux-surface cross-sections at several zeta over one field
  period, with the magnetic axis marked;
- ``modB``      ``|B|`` contours in (zeta, theta) at mid radius and boundary;
- ``profiles``  iota / pressure / current profiles plus the ``fsqt``
  force-residual convergence trace;
- ``3d``        3-D plasma boundary colored by ``|B|``.

Both stellarator-symmetric and ``lasym`` (asymmetric) equilibria are
supported: the sine/cosine partner tables (``rmns``, ``zmnc``, ``bmns``,
...) are included whenever present.  All figures use the Agg backend,
dpi <= 150, and are closed after saving.

Public API
----------
``plot_wout(path_or_WoutData, outdir, which=(...)) -> dict[str, Path]``
``plot_boozmn(path, outdir) -> dict[str, Path]``
plus the per-figure helpers each of those dispatches to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

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

_DPI = 110  # <=150 per plan; keeps every figure well under 400 kB.


# ==========================================================================
# matplotlib / input handling
# ==========================================================================

def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


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


def _eval_modes(cos_coeff, sin_coeff, xm, xn, theta, phi):
    """Evaluate sum_k [c_k cos(m_k*theta - n_k*phi) + s_k sin(...)].

    ``theta``/``phi`` are 1-D; the result has shape (ntheta, nphi).
    """
    xm = np.asarray(xm, dtype=float)
    xn = np.asarray(xn, dtype=float)
    # (mn, ntheta, nphi) phase table; grids here are small (<=200x260).
    angle = (
        xm[:, None, None] * np.asarray(theta)[None, :, None]
        - xn[:, None, None] * np.asarray(phi)[None, None, :]
    )
    return np.tensordot(np.asarray(cos_coeff, dtype=float), np.cos(angle), axes=(0, 0)) + np.tensordot(
        np.asarray(sin_coeff, dtype=float), np.sin(angle), axes=(0, 0)
    )


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
    labels = ["0", "π/2", "π", "3π/2", "2π"]
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


# ==========================================================================
# Per-figure plotters (wout)
# ==========================================================================

_S_LABEL = r"$s = \psi/\psi_b$"


def plot_summary(wout, out_path: str | Path, *, s_plot_ignore: float = 0.2) -> Path:
    """3x3 overview figure (profiles + two ``|B|`` contour panels)."""
    plt = _import_matplotlib()
    wout, _ = _as_wout(wout)
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    s_half = _half_mesh_s(ns)

    fig, axes = plt.subplots(3, 3, figsize=(13, 7))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    ax.plot(s, np.asarray(wout.iotaf, dtype=float), ".-")
    ax.set_xlabel(_S_LABEL)
    ax.set_ylabel(r"$\iota$")
    ax.set_title("rotational transform")

    ax = axes[0, 1]
    ax.plot(s, np.asarray(wout.presf, dtype=float), ".-", label="presf (full)")
    ax.plot(s_half, np.asarray(wout.pres, dtype=float)[1:], ".-", label="pres (half)")
    ax.legend(fontsize="x-small")
    ax.set_xlabel(_S_LABEL)
    ax.set_title("pressure [Pa]")

    for ax, x_vals, y_vals, title in (
        (axes[0, 2], s_half, np.asarray(wout.buco, dtype=float)[1:], "buco"),
        (axes[1, 0], s_half, np.asarray(wout.bvco, dtype=float)[1:], "bvco"),
        (axes[1, 1], s, np.asarray(wout.jcuru, dtype=float), "jcuru"),
        (axes[1, 2], s, np.asarray(wout.jcurv, dtype=float), "jcurv"),
    ):
        ax.plot(x_vals, y_vals, ".-")
        ax.set_title(title)
        ax.set_xlabel(_S_LABEL)

    ax = axes[2, 0]
    dmerc = np.asarray(wout.DMerc, dtype=float)
    ign = int(s_plot_ignore * ns)
    ax.plot(s[ign:-2], dmerc[ign:-2], ".-")
    ax.set_title("DMerc")
    ax.set_xlabel(_S_LABEL)

    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    phi = np.linspace(0.0, 2.0 * np.pi, 80)
    iotaf = np.asarray(wout.iotaf, dtype=float)
    for col, (irad, title) in enumerate(((ns // 2, "Mid radius |B|"), (ns - 1, "Boundary |B|"))):
        B = surface_modB(wout, s_index=int(irad), theta=theta, phi=phi)
        ax = axes[2, 1 + col]
        cf = ax.contour(*np.meshgrid(phi, theta), B, 20, cmap="viridis", linewidths=0.8)
        fig.colorbar(cf, ax=ax)
        iota_val = float(iotaf[irad])
        span = float(phi.max())
        line = [0.0, span * iota_val] if iota_val > 0 else [-span * iota_val, 0.0]
        ax.plot([0.0, span], line, "k", linewidth=0.9)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_title(title)
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\theta$")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    for iz in range(nzeta):
        ax = flat[iz]
        for irad in iradii:
            R, Z = surface_rz(wout, s_index=int(irad), theta=theta, phi=phi[iz : iz + 1])
            ax.plot(R[:, 0], Z[:, 0], "-", linewidth=0.9)
        ax.plot(Raxis[iz], Zaxis[iz], "xr", markersize=5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R [m]", fontsize=9)
        ax.set_ylabel("Z [m]", fontsize=9)
        ax.set_title(rf"$\phi$ = {phi[iz]:.2f}", fontsize=10)
    for iz in range(nzeta, flat.size):
        flat[iz].set_axis_off()
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_modB(
    wout,
    out_path: str | Path,
    *,
    ntheta: int = 90,
    nphi: int = 180,
) -> Path:
    """``|B|`` contours in (phi, theta) at mid radius and the plasma boundary."""
    plt = _import_matplotlib()
    wout, _ = _as_wout(wout)
    ns = int(wout.ns)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi / int(wout.nfp), nphi)
    phi2d, theta2d = np.meshgrid(phi, theta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.patch.set_facecolor("white")
    for ax, irad, title in (
        (axes[0], ns // 2, "mid radius"),
        (axes[1], ns - 1, "plasma boundary"),
    ):
        B = surface_modB(wout, s_index=int(irad), theta=theta, phi=phi)
        cf = ax.contour(phi2d, theta2d, B, levels=25, cmap="viridis", linewidths=1.0)
        fig.colorbar(cf, ax=ax, label="|B| [T]")
        ax.set_title(f"|B| on {title} (one field period)")
        ax.set_xlabel(r"toroidal angle $\phi$")
        ax.set_ylabel(r"poloidal angle $\theta$")
        _pi_ticks(ax, "y")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_profiles(wout, out_path: str | Path) -> Path:
    """Radial profiles (iota, pressure, currents) and fsqt convergence."""
    plt = _import_matplotlib()
    wout, _ = _as_wout(wout)
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    s_half = _half_mesh_s(ns)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    ax.plot(s, np.asarray(wout.iotaf, dtype=float), ".-", label=r"$\iota$ (full)")
    ax.plot(s_half, np.asarray(wout.iotas, dtype=float)[1:], ".", ms=3, label=r"$\iota$ (half)")
    ax.set_ylabel(r"$\iota$")
    ax.legend(fontsize="x-small")

    ax = axes[0, 1]
    ax.plot(s, np.asarray(wout.presf, dtype=float), ".-", label="presf")
    ax.plot(s_half, np.asarray(wout.pres, dtype=float)[1:], ".", ms=3, label="pres")
    ax.set_ylabel("pressure [Pa]")
    ax.legend(fontsize="x-small")

    ax = axes[0, 2]
    ax.plot(s, np.asarray(wout.jcuru, dtype=float), ".-", label="jcuru")
    ax.plot(s, np.asarray(wout.jcurv, dtype=float), ".-", label="jcurv")
    ax.set_ylabel("current density [A]")
    ax.legend(fontsize="x-small")

    ax = axes[1, 0]
    ax.plot(s_half, np.asarray(wout.buco, dtype=float)[1:], ".-", label="buco")
    ax.plot(s_half, np.asarray(wout.bvco, dtype=float)[1:], ".-", label="bvco")
    ax.set_ylabel(r"$\langle B_u \rangle$, $\langle B_v \rangle$")
    ax.legend(fontsize="x-small")

    ax = axes[1, 1]
    phi_flux = np.asarray(wout.phi, dtype=float)
    chi_flux = np.asarray(wout.chi, dtype=float)
    ax.plot(s, phi_flux, ".-", label=r"$\phi$ (toroidal)")
    ax.plot(s, chi_flux, ".-", label=r"$\chi$ (poloidal)")
    ax.set_ylabel("flux [Wb]")
    ax.legend(fontsize="x-small")

    for ax in axes.ravel()[:5]:
        ax.set_xlabel(_S_LABEL)
        ax.grid(True, alpha=0.25)

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
        ax.legend(fontsize="x-small")
    else:
        ax.text(0.5, 0.5, "no fsqt history", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("stored iteration sample")
    ax.set_ylabel("force residual")
    ax.set_title("convergence (fsqt)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_boundary_3d(
    wout,
    out_path: str | Path,
    *,
    ntheta: int = 60,
    nzeta: int | None = None,
) -> Path:
    """3-D plasma boundary colored by ``|B|`` (full torus)."""
    plt = _import_matplotlib()
    from matplotlib import cm
    from matplotlib.colors import Normalize

    wout, _ = _as_wout(wout)
    ns, nfp = int(wout.ns), int(wout.nfp)
    if nzeta is None:
        nzeta = min(360, max(120, 60 * nfp))
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, int(nzeta))
    R, Z = surface_rz(wout, s_index=ns - 1, theta=theta, phi=phi)
    B = surface_modB(wout, s_index=ns - 1, theta=theta, phi=phi)
    phi2d = np.meshgrid(phi, theta)[0]
    X, Y = R * np.cos(phi2d), R * np.sin(phi2d)
    B_scaled = (B - B.min()) / (B.max() - B.min() + 1e-30)

    fig = plt.figure(figsize=(5.2, 4.4), frameon=False)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, Y, Z, facecolors=cm.viridis(B_scaled), rstride=1, cstride=1,
        antialiased=False, linewidth=0.0,
    )
    scale = 0.7 * max(np.abs(X).max(), np.abs(Y).max())
    ax.auto_scale_xyz([-scale, scale], [-scale, scale], [-scale, scale])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    cax = fig.add_axes([0.21, 0.86, 0.60, 0.03])
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=Normalize(float(B.min()), float(B.max())))
    sm.set_array([])
    fig.colorbar(sm, orientation="horizontal", cax=cax).set_label("|B| [T]")
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
    cmap: str = "viridis",
) -> Path:
    """Boozer-coordinate ``|B|`` contours at mid radius and the outermost surface.

    ``cmap`` selects the contour colormap (pass ``"jet"`` for the STELLOPT /
    booz_xform convention).
    """
    plt = _import_matplotlib()
    bx = _load_boozmn(boozmn)
    ns_b = int(np.asarray(bx.bmnc_b).shape[1])
    if ns_b < 1:
        raise ValueError("Boozer output contains no computed surfaces")
    selected = [("mid radius", ns_b // 2), ("outermost surface", ns_b - 1)]
    if selected[0][1] == selected[1][1]:
        selected = selected[1:]

    fig, axes = plt.subplots(1, len(selected), figsize=(6.4 * len(selected), 4.4), squeeze=False)
    for ax, (title, js) in zip(axes[0], selected):
        theta, phi, B = _boozer_modB_grid(bx, js=js, ntheta=ntheta, nphi=nphi)
        phi2d, theta2d = np.meshgrid(phi, theta)
        cs = ax.contour(phi2d, theta2d, B, levels=24, cmap=cmap, linewidths=1.0)
        fig.colorbar(cs, ax=ax, label="|B| [T]")
        ax.set_title(title)
        ax.set_xlabel(r"Boozer toroidal angle $\phi_B$")
        ax.set_ylabel(r"Boozer poloidal angle $\theta_B$")
        _pi_ticks(ax, "y")
        ax.set_ylim(0, 2 * np.pi)
    fig.suptitle("Boozer-coordinate |B| contours", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
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

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.0))
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
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
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

    fig, ax = plt.subplots(1, 1, figsize=(max(8.0, 0.24 * len(order)), 4.8))
    x = np.arange(len(order))
    ax.bar(x, np.maximum(amp[order, js], 1e-16), color=colors, width=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"({int(xm[i])},{int(xn[i])})" for i in order], rotation=75, ha="right", fontsize=8)
    ax.set_xlabel("Boozer mode (m, n)")
    ax.set_ylabel(r"$|B_{mn}|$ [T]")
    ax.set_title(f"Boozer |B| spectrum, surface {js + 1}/{ns_b}")
    legend = {}
    for i in order:
        group, color = _boozer_mode_group(int(xm[i]), int(xn[i]), nfp)
        legend[group] = color
    ax.legend(
        handles=[plt.Line2D([0], [0], color=c, lw=4, label=g) for g, c in legend.items()],
        fontsize=8, loc="best",
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
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
