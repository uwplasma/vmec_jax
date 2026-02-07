"""Showcase: input -> fixed-boundary solve -> wout -> plots -> parity printout.

This script is the recommended starting point for new users.
Edit the `input.*` file to change the configuration; the script is intentionally
light on flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmec_jax.api as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="shaped_tokamak_pressure")
    p.add_argument("--solver", default="vmecpp_iter", choices=["vmecpp_iter", "vmec_gn", "gd", "lbfgs"])
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--no-solve", action="store_true", help="Use initial guess only.")
    p.add_argument("--outdir", default=None, help="Defaults to examples/outputs/showcase/<case>/")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / f"input.{args.case}"
    ref_wout_path = root / "data" / f"wout_{args.case}_reference.nc"
    outdir = Path(args.outdir) if args.outdir else (root / "outputs" / "showcase" / args.case)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, indata = vj.load_input(input_path)
    run = vj.run_fixed_boundary(
        input_path,
        solver=str(args.solver),
        max_iter=int(args.max_iter),
        use_initial_guess=bool(args.no_solve),
        verbose=True,
    )
    out_wout_path = outdir / f"wout_{args.case}_vmec_jax.nc"
    wout_new = vj.write_wout_from_fixed_boundary_run(out_wout_path, run, include_fsq=True)

    wout_ref = vj.read_wout(ref_wout_path) if ref_wout_path.exists() else None
    print(f"[vmec_jax] wrote wout: {out_wout_path}")
    if wout_ref is not None:
        fsq_ref = float(wout_ref.fsqr + wout_ref.fsqz + wout_ref.fsql)
        fsq_new = float(wout_new.fsqr + wout_new.fsqz + wout_new.fsql)
        print(f"[vmec_jax] fsq_total: ref={fsq_ref:.3e} new={fsq_new:.3e}")

    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("This example requires matplotlib (pip install -e .[plots]).") from e

    theta = vj.closed_theta_grid(256)
    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    ns = int(wout_new.ns)
    s_index_lcfs = ns - 1
    s_indices = np.linspace(0, s_index_lcfs, 9).round().astype(int)

    # 1) Nested flux surfaces (phi=0) from reference and new wout.
    fig, ax = plt.subplots(1, 2 if wout_ref is not None else 1, figsize=(10, 4), constrained_layout=True)
    ax = np.atleast_1d(ax)
    for si in s_indices:
        Rn, Zn = vj.surface_rz_from_wout_physical(wout_new, theta=theta, phi=np.asarray([0.0]), s_index=int(si), nyq=False)
        ax[-1].plot(Rn[:, 0], Zn[:, 0], lw=1.2)
        if wout_ref is not None:
            Rr, Zr = vj.surface_rz_from_wout_physical(wout_ref, theta=theta, phi=np.asarray([0.0]), s_index=int(si), nyq=False)
            ax[0].plot(Rr[:, 0], Zr[:, 0], lw=1.2)
    if wout_ref is not None:
        ax[0].set_title("VMEC2000 (reference)")
        ax[1].set_title("vmec_jax (new wout)")
    else:
        ax[0].set_title("vmec_jax (new wout)")
    for a in ax:
        a.set_aspect("equal", "box")
        a.set_xlabel("R")
        a.set_ylabel("Z")
    fig.savefig(outdir / "surfaces_nested_phi0.png", dpi=180)
    plt.close(fig)

    # 2) |B| on LCFS from the state (avoids relying on Nyquist bmnc in minimal wouts).
    st_new = vj.state_from_wout(wout_new)
    B_new = vj.bmag_from_state_physical(st_new, run.static, indata=indata, theta=theta, phi=phi, s_index=s_index_lcfs)
    B_ref = None
    if wout_ref is not None:
        st_ref = vj.state_from_wout(wout_ref)
        B_ref = vj.bmag_from_state_physical(st_ref, run.static, indata=indata, theta=theta, phi=phi, s_index=s_index_lcfs)

    fig, ax = plt.subplots(1, 2 if B_ref is not None else 1, figsize=(10, 4), constrained_layout=True)
    ax = np.atleast_1d(ax)
    vmin = float(np.min(B_new if B_ref is None else np.minimum(B_new, B_ref)))
    vmax = float(np.max(B_new if B_ref is None else np.maximum(B_new, B_ref)))
    ax[-1].imshow(B_new, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax[-1].set_title("vmec_jax |B| (LCFS)")
    if B_ref is not None:
        ax[0].imshow(B_ref, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax[0].set_title("VMEC2000 |B| (LCFS)")
    for a in ax:
        a.set_xlabel("phi index")
        a.set_ylabel("theta index")
    fig.savefig(outdir / "bmag_lcfs.png", dpi=180)
    plt.close(fig)

    # 3) 3D LCFS surface colored by |B| (new wout).
    th3 = vj.closed_theta_grid(120)
    ph3 = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    Rlcfs, Zlcfs = vj.surface_rz_from_wout_physical(wout_new, theta=th3, phi=ph3, s_index=s_index_lcfs, nyq=False)
    Blcfs = vj.bmag_from_state_physical(st_new, run.static, indata=indata, theta=th3, phi=ph3, s_index=s_index_lcfs)
    X = Rlcfs * np.cos(ph3[None, :])
    Y = Rlcfs * np.sin(ph3[None, :])
    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot_surface(
        X,
        Y,
        Zlcfs,
        facecolors=plt.cm.viridis((Blcfs - Blcfs.min()) / max(Blcfs.ptp(), 1e-12)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    vj.fix_matplotlib_3d(ax3)
    ax3.set_title("LCFS 3D colored by |B| (vmec_jax)")
    fig.savefig(outdir / "lcfs_3d_bmag.png", dpi=180)
    plt.close(fig)

    print(f"[vmec_jax] wrote plots under: {outdir}")


if __name__ == "__main__":
    main()

