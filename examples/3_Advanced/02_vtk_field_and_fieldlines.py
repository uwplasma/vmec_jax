"""Advanced example: export surface B field + field line trace to ParaView (VTK).

Writes:
- a surface StructuredGrid (.vts) with point data:
  - Bx, By, Bz (vector B)
  - |B|
  - R, Z
- a PolyData polyline (.vtp) with a single field line.

This uses only the minimal, dependency-free VTK writers in `vmec_jax.visualization`.
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
from vmec_jax.config import load_config
from vmec_jax.field import b_cartesian_from_bsup
from vmec_jax.fieldlines import trace_fieldline_on_surface
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout
from vmec_jax.visualization import write_vtp_polyline, write_vts_structured_grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--wout", type=str, default="", help="Optional wout_*.nc for equilibrium state")
    p.add_argument("--outdir", type=str, default="vtk_out", help="Output directory")
    p.add_argument("--s-index", type=int, default=-1, help="Radial surface index to export (default: edge)")
    p.add_argument("--hi-res", action="store_true", help="Use a higher-res angular grid for smoother VTK output")
    p.add_argument("--export-volume", action="store_true", help="Also export a full (s,theta,zeta) volume .vts")
    p.add_argument("--theta0", type=float, default=0.0, help="Field-line start theta")
    p.add_argument("--phi0", type=float, default=0.0, help="Field-line start physical toroidal angle phi")
    p.add_argument("--n-steps", type=int, default=2000, help="Field-line steps")
    p.add_argument("--dphi", type=float, default=2e-3, help="Field-line step in phi")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, _indata = load_config(args.input)

    wout_path = Path(args.wout) if args.wout else None
    if wout_path is None:
        inp = Path(args.input).name
        case = inp[len("input.") :] if inp.startswith("input.") else inp
        candidate = REPO_ROOT / "examples" / f"wout_{case}_reference.nc"
        if candidate.exists():
            wout_path = candidate
    if wout_path is None:
        raise SystemExit(
            "No --wout provided and no bundled reference wout found for this input. "
            "Pass --wout /path/to/wout_*.nc (requires netCDF4)."
        )

    wout = read_wout(str(wout_path))

    # Optionally increase angular resolution for smoother VTK output (keep mode content unchanged).
    if args.hi_res:
        from dataclasses import replace

        ntheta = max(int(cfg.ntheta), 4 * int(wout.mpol) + 16)
        ntheta = 2 * (ntheta // 2)
        nzeta = max(int(cfg.nzeta), 4 * int(wout.ntor) + 16)
        if nzeta <= 0:
            nzeta = 1
        cfg = replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))

    static = build_static(cfg)

    state = state_from_wout(wout)
    g = eval_geom(state, static)

    # Reconstruct Nyquist fields from wout on our (theta,zeta) grid.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)
    bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))

    B = b_cartesian_from_bsup(g, bsupu, bsupv, zeta=static.grid.zeta, nfp=cfg.nfp)
    B_np = np.asarray(B)
    Bmag = np.sqrt(np.sum(B_np**2, axis=-1))

    s_index = int(args.s_index)
    if s_index < 0:
        s_index = cfg.ns - 1
    if s_index < 0 or s_index >= cfg.ns:
        raise SystemExit(f"s-index out of range: {s_index}")

    R = np.asarray(g.R[s_index])
    Z = np.asarray(g.Z[s_index])
    # physical toroidal angle phi = zeta / nfp
    phi = np.asarray(static.grid.zeta) / cfg.nfp
    cosphi = np.cos(phi)[None, :]
    sinphi = np.sin(phi)[None, :]
    x = R * cosphi
    y = R * sinphi
    z = Z

    surface_path = outdir / f"surface_s{s_index:03d}.vts"
    write_vts_structured_grid(
        surface_path,
        x=x,
        y=y,
        z=z,
        point_data={
            "B": B_np[s_index],
            "Bx": B_np[s_index][..., 0],
            "By": B_np[s_index][..., 1],
            "Bz": B_np[s_index][..., 2],
            "Bmag": Bmag[s_index],
            "R": R,
            "Z": Z,
        },
    )
    print(f"wrote: {surface_path}")

    if args.export_volume:
        # Full volume StructuredGrid for slicing in ParaView.
        Rv = np.asarray(g.R)
        Zv = np.asarray(g.Z)
        phi_v = np.asarray(static.grid.zeta) / cfg.nfp
        cosphi_v = np.cos(phi_v)[None, None, :]
        sinphi_v = np.sin(phi_v)[None, None, :]
        xv = Rv * cosphi_v
        yv = Rv * sinphi_v
        zv = Zv
        vol_path = outdir / "volume.vts"
        write_vts_structured_grid(
            vol_path,
            x=xv,
            y=yv,
            z=zv,
            point_data={
                "B": B_np,
                "Bx": B_np[..., 0],
                "By": B_np[..., 1],
                "Bz": B_np[..., 2],
                "Bmag": Bmag,
            },
        )
        print(f"wrote: {vol_path}")

    fl = trace_fieldline_on_surface(
        R=R,
        Z=Z,
        bsupu=bsupu[s_index],
        bsupv=bsupv[s_index],
        Bmag=Bmag[s_index],
        nfp=cfg.nfp,
        theta0=float(args.theta0),
        phi0=float(args.phi0),
        n_steps=int(args.n_steps),
        dphi=float(args.dphi),
    )
    line_path = outdir / f"fieldline_s{s_index:03d}.vtp"
    write_vtp_polyline(
        line_path,
        points=np.stack([fl.x, fl.y, fl.z], axis=1),
        point_data={"Bmag": fl.Bmag},
    )
    print(f"wrote: {line_path}")

    print("\nParaView tips:")
    print("- Open the .vts and .vtp files; color the surface by `Bmag` and show vectors for `B`.")
    print("- Use 'Tube' on the fieldline polyline for visibility.")


if __name__ == "__main__":
    main()
