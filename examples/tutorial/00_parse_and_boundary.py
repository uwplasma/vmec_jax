"""Step 0: evaluate boundary surface from an input file."""

from __future__ import annotations

import argparse

import numpy as np

import vmec_jax as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out", default="boundary_demo.npz")
    args = p.parse_args()

    cfg, indata = vj.load_config(args.input)
    modes = vj.vmec_mode_table(cfg.mpol, cfg.ntor)
    grid = vj.make_angle_grid(cfg.ntheta, cfg.nzeta, cfg.nfp)
    bdy = vj.boundary_from_indata(indata, modes)
    basis = vj.build_helical_basis(modes, grid)
    R = np.asarray(vj.eval_fourier(bdy.R_cos, bdy.R_sin, basis))
    Z = np.asarray(vj.eval_fourier(bdy.Z_cos, bdy.Z_sin, basis))

    np.savez(args.out, theta=grid.theta, zeta=grid.zeta, R=R, Z=Z, m=modes.m, n=modes.n)
    print(f"[vmec_jax] wrote {args.out}")


if __name__ == "__main__":
    main()

