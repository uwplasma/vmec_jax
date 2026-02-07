"""Axisymmetric (tokamak) showcase: input -> fixed-boundary solve -> wout.nc + plots + parity summary.

Runs a small suite of bundled axisymmetric inputs, writes a `wout_*.nc` for each,
and compares a few key quantities against bundled VMEC2000 reference `wout` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vmec_jax.driver import run_fixed_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.plotting import (
    bmag_from_state_physical,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    surface_rz_from_wout_physical,
)
from vmec_jax.wout import read_wout, state_from_wout


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "examples" / "data"
    out_root = repo_root / "examples" / "outputs" / "axisym_showcase"

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=out_root)
    p.add_argument(
        "--solver",
        type=str,
        default="vmecpp_iter",
        choices=["vmecpp_iter", "vmec_gn"],
        help="Solver used for fixed-boundary solve (defaults to VMEC++-style iteration).",
    )
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Optional step size override. If omitted, uses a solver-specific default (e.g. DELT from input).",
    )
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    cases = [
        ("circular_tokamak", data_dir / "input.circular_tokamak", data_dir / "wout_circular_tokamak_reference.nc"),
        ("shaped_tokamak_pressure", data_dir / "input.shaped_tokamak_pressure", data_dir / "wout_shaped_tokamak_pressure_reference.nc"),
        ("vmecpp_solovev", data_dir / "input.vmecpp_solovev", data_dir / "wout_vmecpp_solovev_reference.nc"),
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)

    if bool(args.no_plots):
        plt = None
    else:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception:
            raise SystemExit("This example requires matplotlib for plots (or pass --no-plots).")

    def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        num = float(np.sqrt(np.mean((a - b) ** 2)))
        den = float(np.sqrt(np.mean(b**2)))
        if den != 0.0:
            return num / den
        # If the reference is identically zero, report absolute RMS error instead
        # of `inf` so axisymmetric cases with zero pressure are readable.
        return num

    for case, input_path, wout_ref_path in cases:
        assert input_path.exists(), input_path
        assert wout_ref_path.exists(), wout_ref_path

        print(f"\n== {case} ==")
        case_out = args.outdir / case
        case_out.mkdir(parents=True, exist_ok=True)

        run_kws = dict(
            solver=str(args.solver),
            max_iter=int(args.max_iter),
            verbose=True,
        )
        if args.step_size is not None:
            run_kws["step_size"] = float(args.step_size)
        run = run_fixed_boundary(input_path, **run_kws)
        wout_new_path = case_out / f"wout_{case}_vmec_jax.nc"
        wout_new = write_wout_from_fixed_boundary_run(wout_new_path, run, include_fsq=True)

        wout_ref = read_wout(wout_ref_path)
        print(f"[axisym_showcase] wrote:      {wout_new_path}")
        print(f"[axisym_showcase] reference:  {wout_ref_path}")

        # Scalar summary (reference uses VMEC2000 scalars; new uses computed Step-10 scalars).
        fsq_ref = float(wout_ref.fsqr + wout_ref.fsqz + wout_ref.fsql)
        fsq_new = float(wout_new.fsqr + wout_new.fsqz + wout_new.fsql)
        print(f"[axisym_showcase] fsq_total: ref={fsq_ref:.3e} new={fsq_new:.3e} rel={abs(fsq_new-fsq_ref)/max(abs(fsq_ref),1e-30):.3e}")

        # A few simple parity checks that are meaningful even before full parity.
        err_rmnc = _rel_rms(np.asarray(wout_new.rmnc), np.asarray(wout_ref.rmnc))
        err_zmns = _rel_rms(np.asarray(wout_new.zmns), np.asarray(wout_ref.zmns))
        print(f"[axisym_showcase] geom rms:   rmnc_rel_rms={err_rmnc:.3e}  zmns_rel_rms={err_zmns:.3e}")

        pref = profiles_from_wout(wout_ref)
        pnew = profiles_from_wout(wout_new)
        err_iota = _rel_rms(pnew['iotaf'], pref['iotaf'])
        err_pres = _rel_rms(pnew['pres'], pref['pres'])
        print(f"[axisym_showcase] prof rms:   iotaf_rel_rms={err_iota:.3e}  pres_rel_rms={err_pres:.3e}")

        if bool(args.no_plots):
            continue

        # Plot grids (vmecPlot2-like).
        theta = closed_theta_grid(256)
        phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        ns = int(wout_ref.ns)
        s_idx_list = [0, max(1, ns // 4), max(1, ns // 2), max(1, (3 * ns) // 4), ns - 1]

        # Evaluate |B| from state (avoid relying on Nyquist `bmnc` in minimal wouts).
        indata = run.indata
        static = run.static
        st_ref = state_from_wout(wout_ref)
        st_new = state_from_wout(wout_new)

        # 1) Nested surfaces cross-section (phi=0).
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for s_idx in s_idx_list:
            R0, Z0 = surface_rz_from_wout_physical(wout_ref, theta=theta, phi=np.asarray([0.0]), s_index=int(s_idx), nyq=False)
            ax[0].plot(R0[:, 0], Z0[:, 0], lw=1.0)
            R1, Z1 = surface_rz_from_wout_physical(wout_new, theta=theta, phi=np.asarray([0.0]), s_index=int(s_idx), nyq=False)
            ax[1].plot(R1[:, 0], Z1[:, 0], lw=1.0)
        ax[0].set_title("VMEC2000 (reference)")
        ax[1].set_title("vmec_jax (new wout)")
        for a in ax:
            a.set_aspect("equal", "box")
            a.set_xlabel("R")
            a.set_ylabel("Z")
        fig.suptitle(f"{case}: nested surfaces (phi=0)")
        fig.savefig(case_out / "surfaces_nested_phi0.png", dpi=180)
        plt.close(fig)

        # 2) LCFS cross-sections over one field period.
        phi_slices = np.asarray([0.0, 0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ph in phi_slices:
            R0, Z0 = surface_rz_from_wout_physical(wout_ref, theta=theta, phi=np.asarray([ph]), s_index=ns - 1, nyq=False)
            ax[0].plot(R0[:, 0], Z0[:, 0], lw=1.0, label=f"phi={ph:.2f}")
            R1, Z1 = surface_rz_from_wout_physical(wout_new, theta=theta, phi=np.asarray([ph]), s_index=ns - 1, nyq=False)
            ax[1].plot(R1[:, 0], Z1[:, 0], lw=1.0, label=f"phi={ph:.2f}")
        ax[0].set_title("VMEC2000 (reference)")
        ax[1].set_title("vmec_jax (new wout)")
        for a in ax:
            a.set_aspect("equal", "box")
            a.set_xlabel("R")
            a.set_ylabel("Z")
            a.legend(loc="best", fontsize=8)
        fig.suptitle(f"{case}: LCFS cross-sections (one field period)")
        fig.savefig(case_out / "lcfs_cross_sections.png", dpi=180)
        plt.close(fig)

        # 3) |B| on the LCFS (theta, phi) computed from state (no Nyquist dependency).
        B_ref = bmag_from_state_physical(st_ref, static, indata=indata, theta=theta, phi=phi, s_index=ns - 1)
        B_new = bmag_from_state_physical(st_new, static, indata=indata, theta=theta, phi=phi, s_index=ns - 1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        vmin = float(np.min([np.min(B_ref), np.min(B_new)]))
        vmax = float(np.max([np.max(B_ref), np.max(B_new)]))
        im0 = ax[0].imshow(B_ref, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, extent=[phi[0], phi[-1], theta[0], theta[-1]])
        im1 = ax[1].imshow(B_new, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, extent=[phi[0], phi[-1], theta[0], theta[-1]])
        ax[0].set_title("VMEC2000 (reference)")
        ax[1].set_title("vmec_jax (from new wout)")
        for a in ax:
            a.set_xlabel("phi")
            a.set_ylabel("theta")
        fig.colorbar(im1, ax=ax.ravel().tolist(), label="|B|")
        fig.suptitle(f"{case}: |B| on LCFS (state-derived)")
        fig.savefig(case_out / "bmag_lcfs.png", dpi=180)
        plt.close(fig)

        # 4) 3D LCFS surface colored by |B| (state-derived).
        th3 = closed_theta_grid(120)
        ph3 = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        Rlcfs, Zlcfs = surface_rz_from_wout_physical(wout_new, theta=th3, phi=ph3, s_index=ns - 1, nyq=False)
        Blcfs = bmag_from_state_physical(st_new, static, indata=indata, theta=th3, phi=ph3, s_index=ns - 1)
        X = Rlcfs * np.cos(ph3[None, :])
        Y = Rlcfs * np.sin(ph3[None, :])
        fig = plt.figure(figsize=(6, 5), constrained_layout=True)
        ax3 = fig.add_subplot(111, projection="3d")
        surf = ax3.plot_surface(X, Y, Zlcfs, facecolors=plt.cm.viridis((Blcfs - Blcfs.min()) / max(Blcfs.ptp(), 1e-12)), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax3.set_title(f"{case}: LCFS 3D colored by |B| (vmec_jax)")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        fix_matplotlib_3d(ax3)
        fig.savefig(case_out / "lcfs_3d_bmag.png", dpi=180)
        plt.close(fig)

        print(f"[axisym_showcase] wrote plots under: {case_out}")

    print(f"\n[axisym_showcase] done. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()
