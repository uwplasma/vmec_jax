"""Side-by-side VMEC2000 vs vmec_jax visualization for the n3are case.

This script:
1) Loads the bundled VMEC2000 reference `wout` for n3are.
2) Runs a lightweight vmec_jax fixed-boundary solve (or uses the initial guess).
3) Generates side-by-side figures for LCFS cross-sections, 3D surface, |B| on LCFS,
   and iota/pressure profiles.

The vmec_jax side reflects the *current* solver capability (not full parity yet).
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

from vmec_jax.modes import vmec_mode_table
from vmec_jax.plotting import (
    axis_rz_from_state_physical,
    axis_rz_from_wout_physical,
    bmag_from_state_physical,
    bmag_from_state_vmec_realspace,
    bmag_from_wout_physical,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    surface_rz_from_state_physical,
    surface_rz_from_wout_physical,
    vmecplot2_cross_section_indices,
    zeta_grid_field_period,
)
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.geom import eval_geom
from vmec_jax.diagnostics import print_jacobian_stats
from vmec_jax.wout import read_wout


def _load_vmec2000_wout():
    wout_path = REPO_ROOT / "examples" / "data" / "wout_n3are_R7.75B5.7_lowres.nc"
    if not wout_path.exists():
        raise SystemExit(f"Missing wout reference: {wout_path}")
    return read_wout(wout_path)



def _plot_cross_sections(ax, *, R, Z, Raxis, Zaxis, title: str):
    labels = [r"$\phi=0$", r"$\phi=\pi/2$", r"$\phi=\pi$", r"$\phi=3\pi/2$"]
    for j in range(R.shape[1]):
        ax.plot(R[:, j], Z[:, j], lw=1.6, label=labels[j])
        ax.plot(Raxis[j], Zaxis[j], "x", ms=6, color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)


def _plot_bmag_surface(ax, *, B, theta, phi, title: str):
    phi2d, theta2d = np.meshgrid(phi, theta)
    cf = ax.contourf(phi2d, theta2d, B, levels=20, cmap="viridis")
    ax.set_xlabel("phi")
    ax.set_ylabel("theta")
    ax.set_title(title)
    return cf


def _plot_3d_surface(ax, *, R, Z, B, phi, title: str):
    phi2d, theta2d = np.meshgrid(phi, np.linspace(0, 2 * np.pi, num=R.shape[0]))
    X = R * np.cos(phi2d)
    Y = R * np.sin(phi2d)
    B_rescaled = (B - B.min()) / (B.max() - B.min() + 1e-12)
    colors = cm.jet(B_rescaled)
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.auto_scale_xyz([X.min(), X.max()], [X.min(), X.max()], [X.min(), X.max()])
    fix_matplotlib_3d(ax)


def _plot_profiles(ax_iota, ax_pres, *, s, s_half, iota, presf, pres, title_prefix: str):
    ax_iota.plot(s, iota, lw=2.0)
    ax_iota.set_xlabel("s")
    ax_iota.set_ylabel("iota")
    ax_iota.set_title(f"{title_prefix}: iota")

    ax_pres.plot(s, presf, lw=2.0, label="presf")
    ax_pres.plot(s_half, pres[1:], lw=1.5, label="pres (half)")
    ax_pres.set_xlabel("s")
    ax_pres.set_ylabel("pressure")
    ax_pres.set_title(f"{title_prefix}: pressure")
    ax_pres.legend(frameon=False, fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(REPO_ROOT / "examples/data/input.n3are_R7.75B5.7_lowres"))
    p.add_argument("--outdir", type=str, default=str(REPO_ROOT / "docs/_static/figures"))
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--step-size", type=float, default=1e-5)
    p.add_argument("--solve", action="store_true", help="Run the vmec_jax fixed-boundary solver (slower).")
    p.add_argument("--no-solve", action="store_true", help="Use the initial guess only (fast).")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # VMEC2000 reference
    wout = _load_vmec2000_wout()

    # vmec_jax current output
    use_initial_guess = not bool(args.solve) or bool(args.no_solve)

    run = run_fixed_boundary(
        Path(args.input),
        solver="gd",
        max_iter=int(args.max_iter),
        step_size=float(args.step_size),
        use_initial_guess=use_initial_guess,
    )
    state = run.state
    static = run.static
    indata = run.indata
    signgs = run.signgs
    modes = vmec_mode_table(static.cfg.mpol, static.cfg.ntor)

    # --- Cross sections ---
    theta_cs = closed_theta_grid(200)
    phi_cs = zeta_grid_field_period(8, nfp=int(wout.nfp))
    idx = vmecplot2_cross_section_indices(phi_cs.size)
    phi_slices = phi_cs[idx]

    R_vmec, Z_vmec = surface_rz_from_wout_physical(wout, theta=theta_cs, phi=phi_slices, s_index=int(wout.ns) - 1)
    Raxis_vmec, Zaxis_vmec = axis_rz_from_wout_physical(wout, phi=phi_slices)

    R_jax, Z_jax = surface_rz_from_state_physical(
        state, modes, theta=theta_cs, phi=phi_slices, s_index=int(static.cfg.ns) - 1, nfp=int(static.cfg.nfp)
    )
    Raxis_jax, Zaxis_jax = axis_rz_from_state_physical(state, modes, phi=phi_slices, nfp=int(static.cfg.nfp))

    jax_title = "vmec_jax (initial guess)" if use_initial_guess else "vmec_jax (solver)"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    _plot_cross_sections(axes[0], R=R_vmec, Z=Z_vmec, Raxis=Raxis_vmec, Zaxis=Zaxis_vmec, title="VMEC2000")
    _plot_cross_sections(axes[1], R=R_jax, Z=Z_jax, Raxis=Raxis_jax, Zaxis=Zaxis_jax, title=jax_title)
    fig.tight_layout()
    fig.savefig(outdir / "n3are_compare_cross_sections.png", dpi=220)
    plt.close(fig)

    # --- B magnitude on LCFS ---
    theta_b = closed_theta_grid(30)
    phi_b = np.linspace(0.0, 2.0 * np.pi, num=65, endpoint=True)
    B_vmec = bmag_from_wout_physical(wout, theta=theta_b, phi=phi_b, s_index=int(wout.ns) - 1)
    sqrtg_floor = None
    if use_initial_guess:
        geom_init = eval_geom(state, static)
        abs_sg = np.abs(np.asarray(geom_init.sqrtg))
        floor = max(1e-3, 0.5 * float(np.median(abs_sg)))
        sqrtg_floor = floor
        print("[vmec_jax] initial guess Jacobian stats:")
        print_jacobian_stats(geom_init.sqrtg, indent="  ")
        print(f"[vmec_jax] initial guess sqrtg_floor={floor:.3e}")

    B_jax = bmag_from_state_physical(
        state,
        static,
        indata,
        theta=theta_b,
        phi=phi_b,
        s_index=int(static.cfg.ns) - 1,
        signgs=int(wout.signgs),
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
        lamscale=float(np.asarray(run.flux.lamscale)),
        sqrtg_floor=sqrtg_floor,
    )
    B_jax_vmec = bmag_from_state_vmec_realspace(
        state,
        static,
        indata,
        s_index=int(static.cfg.ns) - 1,
        signgs=int(wout.signgs),
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
        lamscale=float(np.asarray(run.flux.lamscale)),
    )
    print(
        f"[vmec_jax] B range (vmec_jax VMEC-grid) min={B_jax_vmec.min():.3e} max={B_jax_vmec.max():.3e}"
    )

    print(f"[vmec_jax] B range (VMEC2000) min={B_vmec.min():.3e} max={B_vmec.max():.3e}")
    print(f"[vmec_jax] B range (vmec_jax) min={B_jax.min():.3e} max={B_jax.max():.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cf0 = _plot_bmag_surface(axes[0], B=B_vmec, theta=theta_b, phi=phi_b, title="VMEC2000")
    cf1 = _plot_bmag_surface(axes[1], B=B_jax, theta=theta_b, phi=phi_b, title=jax_title)
    fig.colorbar(cf0, ax=axes[0], shrink=0.85, label="|B|")
    fig.colorbar(cf1, ax=axes[1], shrink=0.85, label="|B|")
    fig.tight_layout()
    fig.savefig(outdir / "n3are_compare_bmag_surface.png", dpi=220)
    plt.close(fig)

    # --- Profiles ---
    prof_vmec = profiles_from_wout(wout)
    s = prof_vmec["s"]
    s_half = prof_vmec["s_half"]

    prof_jax = run.profiles
    iota_jax = np.asarray(prof_jax.get("iota", np.zeros_like(np.asarray(static.s))))
    presf_jax = np.asarray(prof_jax.get("pressure", np.zeros_like(np.asarray(static.s))))
    pres_jax = np.asarray(prof_jax.get("pressure", np.zeros_like(np.asarray(static.s))))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    _plot_profiles(
        axes[0, 0],
        axes[0, 1],
        s=s,
        s_half=s_half,
        iota=prof_vmec["iotaf"],
        presf=prof_vmec["presf"],
        pres=prof_vmec["pres"],
        title_prefix="VMEC2000",
    )
    _plot_profiles(
        axes[1, 0],
        axes[1, 1],
        s=np.asarray(static.s),
        s_half=s_half,
        iota=iota_jax,
        presf=presf_jax,
        pres=pres_jax,
        title_prefix=jax_title,
    )
    fig.tight_layout()
    fig.savefig(outdir / "n3are_compare_profiles.png", dpi=220)
    plt.close(fig)

    # --- 3D LCFS ---
    theta_3d = closed_theta_grid(80)
    nzeta = int(150 * int(wout.nfp))
    phi_3d = np.linspace(0.0, 2.0 * np.pi, num=nzeta, endpoint=True)

    R_vmec_3d, Z_vmec_3d = surface_rz_from_wout_physical(
        wout, theta=theta_3d, phi=phi_3d, s_index=int(wout.ns) - 1
    )
    B_vmec_3d = bmag_from_wout_physical(wout, theta=theta_3d, phi=phi_3d, s_index=int(wout.ns) - 1)

    R_jax_3d, Z_jax_3d = surface_rz_from_state_physical(
        state, modes, theta=theta_3d, phi=phi_3d, s_index=int(static.cfg.ns) - 1, nfp=int(static.cfg.nfp)
    )
    B_jax_3d = bmag_from_state_physical(
        state,
        static,
        indata,
        theta=theta_3d,
        phi=phi_3d,
        s_index=int(static.cfg.ns) - 1,
        signgs=int(wout.signgs),
        phipf=np.asarray(wout.phipf),
        chipf=np.asarray(wout.chipf),
        lamscale=float(np.asarray(run.flux.lamscale)),
    )

    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_3d_surface(ax0, R=R_vmec_3d, Z=Z_vmec_3d, B=B_vmec_3d, phi=phi_3d, title="VMEC2000")
    _plot_3d_surface(ax1, R=R_jax_3d, Z=Z_jax_3d, B=B_jax_3d, phi=phi_3d, title=jax_title)
    fig.tight_layout()
    fig.savefig(outdir / "n3are_compare_3d.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
