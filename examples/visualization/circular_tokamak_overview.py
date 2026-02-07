"""Generate overview plots for a bundled axisymmetric tokamak-like case.

This script renders VMEC-style plots from the bundled VMEC2000 reference wout:
- Nested flux surfaces in a poloidal cross-section (phi=0)
- 3D LCFS surface colored by |B|
- |B| on the LCFS (theta,phi) map
- iota and pressure profiles

Outputs are written into `docs/_static/figures/` so they can be used by README
and the Sphinx docs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmec_jax.api as vj


def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="circular_tokamak", help="Bundled example case name.")
    ap.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "docs" / "_static" / "figures"),
        help="Output directory for PNGs.",
    )
    ap.add_argument("--ntheta", type=int, default=256)
    ap.add_argument("--nphi", type=int, default=256)
    ap.add_argument("--nsurf", type=int, default=9, help="Number of surfaces in the nested cross-section panel.")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    outdir = _ensure_outdir(Path(args.outdir))
    ex = vj.load_example(args.case, with_wout=True)
    if ex.wout is None or ex.state is None:
        raise RuntimeError(f"Missing bundled wout for case={args.case}")

    wout = ex.wout
    state = ex.state
    static = ex.static
    indata = ex.indata

    # Use physical toroidal angle phi in [0, 2pi).
    theta = np.linspace(0.0, 2.0 * np.pi, int(args.ntheta), endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, int(args.nphi), endpoint=False)

    s_index_lcfs = int(wout.ns) - 1

    # Nested flux surfaces in a poloidal cross-section (phi=0).
    phi0 = np.asarray([0.0], dtype=float)
    nsurf = max(int(args.nsurf), 2)
    s_indices = np.linspace(0, s_index_lcfs, nsurf).round().astype(int)
    nested = []
    for si in s_indices:
        R, Z = vj.surface_rz_from_wout_physical(wout, theta=theta, phi=phi0, s_index=int(si), nyq=False)
        nested.append((R[:, 0], Z[:, 0], int(si)))

    # |B| on the LCFS surface.
    # Use vmec_jax field evaluation from the wout-derived state.
    B = vj.bmag_from_state_physical(
        state,
        static,
        indata=indata,
        theta=theta,
        phi=phi,
        s_index=s_index_lcfs,
    )  # (ntheta,nphi)

    # LCFS surface for 3D plot: evaluate R,Z on a (theta,phi) grid.
    # We use wout Fourier series for geometry so this matches VMECPlot2.
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    R3, Z3 = vj.surface_rz_from_wout_physical(
        wout,
        theta=theta,
        phi=phi,
        s_index=s_index_lcfs,
        nyq=False,
    )
    X = R3 * np.cos(PH)
    Y = R3 * np.sin(PH)

    # Profiles from wout.
    s = np.asarray(wout.phi) / float(wout.phi[-1]) if hasattr(wout, "phi") else np.linspace(0.0, 1.0, int(wout.ns))
    iota = np.asarray(getattr(wout, "iotaf", getattr(wout, "iotas", None)))
    pres = np.asarray(getattr(wout, "presf", getattr(wout, "pres", None)))
    if iota is None or pres is None:
        raise RuntimeError("wout missing iota/pressure arrays")

    # Plot.
    import matplotlib as mpl

    # Force a non-interactive backend so this script works in CI/headless runs.
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Nested cross-sections
    ax0 = fig.add_subplot(gs[0, 0])
    for R, Z, si in nested:
        ax0.plot(R, Z, lw=1.5, alpha=0.85)
    ax0.set_title("Nested flux surfaces (phi=0)")
    ax0.set_xlabel("R")
    ax0.set_ylabel("Z")
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, alpha=0.25)

    # 3D surface colored by |B|
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    cmap = mpl.cm.viridis
    vmin = float(np.nanmin(B))
    vmax = float(np.nanmax(B))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    facecolors = cmap(norm(B))
    # Downsample for speed / file size.
    stride_t = max(int(args.ntheta // 128), 1)
    stride_p = max(int(args.nphi // 128), 1)
    ax1.plot_surface(
        X[::stride_t, ::stride_p],
        Y[::stride_t, ::stride_p],
        Z3[::stride_t, ::stride_p],
        facecolors=facecolors[::stride_t, ::stride_p],
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(B)
    cb = fig.colorbar(mappable, ax=ax1, shrink=0.6, pad=0.1)
    cb.set_label("|B|")
    ax1.set_title("LCFS 3D surface colored by |B|")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    vj.fix_matplotlib_3d(ax1)

    # |B| map
    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.pcolormesh(phi, theta, B, shading="auto", cmap="viridis")
    ax2.set_title("|B| on LCFS")
    ax2.set_xlabel("phi")
    ax2.set_ylabel("theta")
    cb2 = fig.colorbar(im, ax=ax2)
    cb2.set_label("|B|")

    # Profiles
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(np.linspace(0.0, 1.0, iota.size), iota, lw=2, label="iota")
    ax3.plot(np.linspace(0.0, 1.0, pres.size), pres, lw=2, label="pressure")
    ax3.set_title("Profiles")
    ax3.set_xlabel("s (normalized)")
    ax3.legend(loc="best", frameon=True)

    outpath = outdir / f"{args.case}_overview.png"
    fig.suptitle(f"{args.case}: VMEC2000 wout visualization", fontsize=14)
    fig.savefig(outpath, dpi=int(args.dpi))
    print(f"[vmec_jax] wrote {outpath}")


if __name__ == "__main__":
    main()
