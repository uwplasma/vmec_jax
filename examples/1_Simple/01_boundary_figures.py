"""Simple example: boundary evaluation + publication-style figures.

This script:
  1) parses a VMEC `&INDATA` input file,
  2) evaluates the LCFS boundary surface R(θ,ζ), Z(θ,ζ),
  3) writes an `.npz` artifact,
  4) produces several publication-ready figures (requires matplotlib).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.fourier import eval_fourier
from vmec_jax.static import build_static


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required for this example (pip install -e .[plots])") from e


def _set_pub_style(plt):
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--outdir", type=str, default="figures_step0", help="Output directory for .npz and figures")
    p.add_argument("--nslices", type=int, default=4, help="Number of zeta slices to plot")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, indata = load_config(args.input)
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)

    R = np.asarray(eval_fourier(bdy.R_cos, bdy.R_sin, static.basis))
    Z = np.asarray(eval_fourier(bdy.Z_cos, bdy.Z_sin, static.basis))
    theta = np.asarray(static.grid.theta)
    zeta = np.asarray(static.grid.zeta)

    np.savez(
        outdir / "boundary_step0.npz",
        theta=theta,
        zeta=zeta,
        R=R,
        Z=Z,
        nfp=int(cfg.nfp),
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
    )

    plt = _import_matplotlib()
    _set_pub_style(plt)

    # --- Figure 1: cross-sections at several zeta slices
    nslices = max(1, int(args.nslices))
    idx = np.linspace(0, zeta.size - 1, nslices, dtype=int)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for j in idx:
        ax.plot(R[:, j], Z[:, j], lw=1.5, label=rf"$\zeta={zeta[j]:.2f}$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title("VMEC LCFS cross-sections (field period)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "boundary_cross_sections.png")
    fig.savefig(outdir / "boundary_cross_sections.pdf")
    plt.close(fig)

    # --- Figure 2: R(θ) and Z(θ) at zeta=0
    j0 = 0
    fig, ax = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)
    ax[0].plot(theta, R[:, j0], lw=2.0)
    ax[0].set_ylabel("R(θ)")
    ax[1].plot(theta, Z[:, j0], lw=2.0)
    ax[1].set_ylabel("Z(θ)")
    ax[1].set_xlabel("θ")
    fig.suptitle("LCFS at ζ=0")
    fig.tight_layout()
    fig.savefig(outdir / "boundary_RZ_vs_theta.png")
    fig.savefig(outdir / "boundary_RZ_vs_theta.pdf")
    plt.close(fig)

    # --- Figure 3: simple 3D surface (downsampled)
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return

    # Build Cartesian points.
    phi = zeta / cfg.nfp
    cosphi = np.cos(phi)[None, :]
    sinphi = np.sin(phi)[None, :]
    X = R * cosphi
    Y = R * sinphi
    ZZ = Z

    # Downsample for faster plotting
    ds_t = max(1, int(R.shape[0] // 128))
    ds_z = max(1, int(R.shape[1] // 128))

    fig = plt.figure(figsize=(7.0, 5.5))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot_surface(
        X[::ds_t, ::ds_z],
        Y[::ds_t, ::ds_z],
        ZZ[::ds_t, ::ds_z],
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        alpha=0.9,
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title("LCFS surface (one field period)")
    fig.tight_layout()
    fig.savefig(outdir / "boundary_surface_3d.png")
    fig.savefig(outdir / "boundary_surface_3d.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
