"""Step-2: compute metric and Jacobian on the full (s,theta,zeta) grid.

This script validates that we can compute geometric derivatives, the covariant
metric tensor, and the Jacobian sqrt(g) in a single jitted kernel.

The main output is a .npz file containing diagnostics that we'll later compare
against VMEC2000's intermediate quantities.
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

from vmec_jax._compat import has_jax, enable_x64
from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.geom import eval_geom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--out", type=str, default="geom_step2.npz")
    p.add_argument("--verbose", action="store_true", help="Print extra debug information")
    p.add_argument("--dump_full", action="store_true", help="Store R,Z,derivatives in the .npz (bigger file)")
    args = p.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        raise SystemExit(f"error: input file not found: {inpath}")

    if has_jax():
        enable_x64(True)

    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    g = eval_geom(st0, static)

    print("\n==== vmec_jax step-2 geom metrics ====")
    print(f"ns={cfg.ns} ntheta={cfg.ntheta} nzeta={cfg.nzeta} nfp={cfg.nfp}")
    from vmec_jax.diagnostics import summarize_array, print_summary, slice_excluding_axis

    sqrtg = np.asarray(g.sqrtg)
    gtt = np.asarray(g.g_tt)
    gpp = np.asarray(g.g_pp)

    print_summary(summarize_array("sqrtg (all)", sqrtg), indent="")
    print_summary(summarize_array("g_tt  (all)", gtt), indent="")
    print_summary(summarize_array("g_pp  (all)", gpp), indent="")

    # Axis is a coordinate singularity: sqrtg and some metric components can go to 0 at s=0.
    sqrtg_noaxis = slice_excluding_axis(sqrtg, axis_dim=0)
    gtt_noaxis = slice_excluding_axis(gtt, axis_dim=0)
    print("\n-- excluding the axis surface s=0 --")
    print_summary(summarize_array("sqrtg (s>=1)", sqrtg_noaxis), indent="")
    print_summary(summarize_array("g_tt  (s>=1)", gtt_noaxis), indent="")

    # Approximate volume integral (for sanity): V ≈ ∫ sqrtg ds dtheta dphi.
    # Here zeta spans [0, 2π), but phi = zeta / nfp spans [0, 2π/nfp).
    if cfg.ns > 1:
        ds = 1.0 / (cfg.ns - 1)
    else:
        ds = 1.0
    dtheta = 2.0 * np.pi / cfg.ntheta
    # Here zeta spans one field period [0,2π). physical phi = zeta / nfp spans [0, 2π/nfp)
    dphi = 2.0 * np.pi / (cfg.nfp * cfg.nzeta)
    V_per_period = float(np.sum(sqrtg) * ds * dtheta * dphi)
    V_total = V_per_period * float(cfg.nfp)
    print(f"V (rough, per field period) = {V_per_period:.8e}")
    print(f"V (rough, full torus)      = {V_total:.8e}")

    if args.verbose:
        # A few extra sanity checks
        nneg = int(np.sum(sqrtg < 0))
        nzero = int(np.sum(sqrtg == 0))
        print("\n-- extra checks --")
        print(f"sqrtg: #neg={nneg}  #zero={nzero}  total={sqrtg.size}")
        # Show a small slice at s=1
        s1 = min(1, cfg.ns - 1)
        print(f"sqrtg(s={s1}): min={sqrtg[s1].min():.6g} max={sqrtg[s1].max():.6g}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = dict(
        sqrtg=np.asarray(g.sqrtg),
        g_ss=np.asarray(g.g_ss),
        g_st=np.asarray(g.g_st),
        g_sp=np.asarray(g.g_sp),
        g_tt=np.asarray(g.g_tt),
        g_tp=np.asarray(g.g_tp),
        g_pp=np.asarray(g.g_pp),
        s=np.asarray(static.s),
        theta=np.asarray(static.grid.theta),
        zeta=np.asarray(static.grid.zeta),
        m=np.asarray(static.modes.m),
        n=np.asarray(static.modes.n),
        nfp=np.asarray(cfg.nfp),
    )
    if args.dump_full:
        save.update(
            R=np.asarray(g.R),
            Z=np.asarray(g.Z),
            L=np.asarray(g.L),
            R_s=np.asarray(g.R_s),
            Z_s=np.asarray(g.Z_s),
            L_s=np.asarray(g.L_s),
            R_theta=np.asarray(g.R_theta),
            Z_theta=np.asarray(g.Z_theta),
            L_theta=np.asarray(g.L_theta),
            R_phi=np.asarray(g.R_phi),
            Z_phi=np.asarray(g.Z_phi),
            L_phi=np.asarray(g.L_phi),
        )
    np.savez(out, **save)
    print(f"saving: {out}")


if __name__ == "__main__":
    main()
