#!/usr/bin/env python
"""Parse a VMEC input file and evaluate the boundary surface.

This is the first sanity check for the JAX port:
- namelist parsing
- mode tables
- basis evaluation

Output is stored in a portable .npz for quick plotting/comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import vmec_jax as vj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="VMEC input file")
    ap.add_argument("--out", default="boundary_demo.npz")
    ap.add_argument("--verbose", action="store_true", help="Print extra debug information")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        ap.error(f"Input file not found: {args.input}")

    cfg, indata = vj.load_config(str(inp))
    modes = vj.vmec_mode_table(cfg.mpol, cfg.ntor)
    grid = vj.make_angle_grid(cfg.ntheta, cfg.nzeta, cfg.nfp)

    bdy = vj.boundary_from_indata(indata, modes)
    basis = vj.build_helical_basis(modes, grid)

    R = np.asarray(vj.eval_fourier(bdy.R_cos, bdy.R_sin, basis))
    Z = np.asarray(vj.eval_fourier(bdy.Z_cos, bdy.Z_sin, basis))

    dR_dtheta = np.asarray(vj.eval_fourier_dtheta(bdy.R_cos, bdy.R_sin, basis))
    dZ_dtheta = np.asarray(vj.eval_fourier_dtheta(bdy.Z_cos, bdy.Z_sin, basis))
    dR_dphi = np.asarray(vj.eval_fourier_dzeta_phys(bdy.R_cos, bdy.R_sin, basis))
    dZ_dphi = np.asarray(vj.eval_fourier_dzeta_phys(bdy.Z_cos, bdy.Z_sin, basis))

    # quick scalar sanity metrics
    print("==== vmec_jax step-0 boundary eval ====")
    K = len(modes.m)
    print(f"mpol={cfg.mpol} ntor={cfg.ntor} nfp={cfg.nfp}  K(mn)={K}  ntheta={cfg.ntheta} nzeta={cfg.nzeta}")
    print(f"grid: theta in [0,2pi) ({cfg.ntheta} pts), zeta in [0,2pi) one field period ({cfg.nzeta} pts)")
    print(f"R: min={R.min():.6g} max={R.max():.6g}  mean={R.mean():.6g}")
    print(f"Z: min={Z.min():.6g} max={Z.max():.6g}  mean={Z.mean():.6g}")

    if args.verbose:
        from vmec_jax.diagnostics import summarize_many

        print("\n-- derivatives (physical phi) --")
        summarize_many(
            [
                ("dR/dtheta", dR_dtheta),
                ("dZ/dtheta", dZ_dtheta),
                ("dR/dphi", dR_dphi),
                ("dZ/dphi", dZ_dphi),
            ],
            indent="  ",
        )

        # Show the largest boundary Fourier coefficients (cos blocks).
        mag = np.sqrt(bdy.R_cos**2 + bdy.R_sin**2 + bdy.Z_cos**2 + bdy.Z_sin**2)
        idx = np.argsort(-mag)[:10]
        print("\n-- largest boundary coefficients (by |(Rcos,Zsin)|) --")
        print("  rank  (m,n)    Rcos         Rsin         Zcos         Zsin")
        for r, j in enumerate(idx, start=1):
            print(
                f"  {r:>4d}  ({int(modes.m[j]):>2d},{int(modes.n[j]):>3d})"
                f"  {bdy.R_cos[j]:>+11.4e} {bdy.R_sin[j]:>+11.4e}"
                f"  {bdy.Z_cos[j]:>+11.4e} {bdy.Z_sin[j]:>+11.4e}"
            )
    print("saving:", args.out)

    np.savez(
        args.out,
        theta=grid.theta,
        zeta=grid.zeta,
        m=modes.m,
        n=modes.n,
        R=R,
        Z=Z,
        dR_dtheta=dR_dtheta,
        dZ_dtheta=dZ_dtheta,
        dR_dphi=dR_dphi,
        dZ_dphi=dZ_dphi,
        R_cos=bdy.R_cos,
        R_sin=bdy.R_sin,
        Z_cos=bdy.Z_cos,
        Z_sin=bdy.Z_sin,
    )


if __name__ == "__main__":
    main()
