from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

from vmec_jax.driver import load_example
from vmec_jax.plotting import (
    axis_rz_from_wout,
    bmag_from_wout,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    select_zeta_slices,
    surface_rz_from_wout,
    surface_stack,
    zeta_grid,
)


CASES = {
    "n3are": "n3are_R7.75B5.7_lowres",
}


def _plot_profiles(wout, outdir: Path, *, tag: str):
    prof = profiles_from_wout(wout)
    s = prof["s"]
    s_half = prof["s_half"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(s, prof["iotaf"], lw=2.0)
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("iota")
    axes[0].set_title("Rotational transform")

    axes[1].plot(s, prof["presf"], lw=2.0, label="presf")
    axes[1].plot(s_half, prof["pres"][1:], lw=1.5, label="pres (half)")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("pressure")
    axes[1].set_title("Pressure profile")
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    path = outdir / f"{tag}_profiles.png"
    return fig, path


def _plot_cross_sections(wout, outdir: Path, *, tag: str):
    theta = closed_theta_grid(400)
    zeta = zeta_grid(64)
    zeta_slices = select_zeta_slices(zeta, n=4)

    R, Z = surface_stack(wout, theta=theta, zeta_list=zeta_slices, s_index=int(wout.ns) - 1)
    Raxis, Zaxis = axis_rz_from_wout(wout, zeta=zeta_slices)

    fig, ax = plt.subplots(figsize=(6, 6))
    for j, z in enumerate(zeta_slices):
        ax.plot(R[:, j], Z[:, j], lw=1.6, label=f"zeta={z:.2f}")
        ax.plot(Raxis[j], Zaxis[j], "x", ms=6, color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title("LCFS cross-sections (one field period)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = outdir / f"{tag}_cross_sections.png"
    return fig, path


def _plot_bmag_surface(wout, outdir: Path, *, tag: str):
    theta = closed_theta_grid(200)
    zeta = zeta_grid(200)
    B = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=int(wout.ns) - 1)
    zeta2d, theta2d = np.meshgrid(zeta, theta)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cf = ax.contourf(zeta2d, theta2d, B, levels=32, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="|B|")
    ax.set_xlabel("zeta")
    ax.set_ylabel("theta")
    ax.set_title("|B| on LCFS")
    fig.tight_layout()
    path = outdir / f"{tag}_bmag_surface.png"
    return fig, path


def _plot_3d_surface(wout, outdir: Path, *, tag: str):
    theta = closed_theta_grid(160)
    zeta = zeta_grid(260)
    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=int(wout.ns) - 1, nyq=False)
    B = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=int(wout.ns) - 1)

    zeta2d, theta2d = np.meshgrid(zeta, theta)
    X = R * np.cos(zeta2d)
    Y = R * np.sin(zeta2d)

    B_rescaled = (B - B.min()) / (B.max() - B.min() + 1e-12)
    colors = cm.viridis(B_rescaled)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("LCFS 3D surface colored by |B|")
    fix_matplotlib_3d(ax)
    fig.tight_layout()
    path = outdir / f"{tag}_3d_bmag.png"
    return fig, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES.keys(), default="n3are")
    parser.add_argument("--save", action="store_true", help="Save figures instead of showing them.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "docs/_static/figures"),
        help="Directory for saved figures (when --save is set).",
    )
    args = parser.parse_args()

    case_name = CASES[args.case]
    ex = load_example(case_name, root=REPO_ROOT, with_wout=True)
    if ex.wout is None:
        raise SystemExit("wout file not found for the selected case.")
    wout = ex.wout

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag = args.case
    figs = [
        _plot_profiles(wout, outdir, tag=tag),
        _plot_cross_sections(wout, outdir, tag=tag),
        _plot_bmag_surface(wout, outdir, tag=tag),
        _plot_3d_surface(wout, outdir, tag=tag),
    ]

    if args.save:
        for fig, path in figs:
            fig.savefig(path, dpi=220)
            plt.close(fig)
            print(f"Wrote {path}")
    else:
        for fig, _ in figs:
            fig.show()
        plt.show()


if __name__ == "__main__":
    main()
