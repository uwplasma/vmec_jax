"""Advanced example: force residual diagnostics (Step-10 target).

This script computes vmec_jax's current *force-like* residual scalars derived
from the total-objective gradient and prints them alongside VMEC2000's
``wout``-reported ``fsqr/fsqz/fsql`` (when available).

Notes
-----
These are not yet exact VMEC residuals; they are intended as a regression and
solver-diagnostic stepping stone toward full ``residue/getfsq`` parity.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.energy import FluxProfiles, flux_profiles_from_indata
from vmec_jax.field import lamscale_from_phips, signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.profiles import eval_profiles
from vmec_jax.residuals import force_residuals_from_state
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout
from vmec_jax.config import load_config


def _fmt(x: float) -> str:
    return f"{x:.6e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--wout", type=str, default="", help="Optional wout_*.nc (VMEC2000 reference)")
    p.add_argument("--hi-res", action="store_true", help="Increase angular grid for diagnostics")
    p.add_argument("--max-iter", type=int, default=0, help="(reserved) solver iterations to run before reporting")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    cfg, indata = load_config(args.input)
    if args.hi_res and args.wout:
        wout_tmp = read_wout(args.wout)
        ntheta = max(int(cfg.ntheta), 4 * int(wout_tmp.mpol) + 16)
        ntheta = 2 * (ntheta // 2)
        nzeta = max(int(cfg.nzeta), 4 * int(wout_tmp.ntor) + 16)
        if nzeta <= 0:
            nzeta = 1
        cfg = replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))

    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    g0 = eval_geom(st0, static)
    signgs0 = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)
    flux0 = flux_profiles_from_indata(indata, static.s, signgs=signgs0)
    prof0 = eval_profiles(indata, static.s)
    pressure0 = prof0.get("pressure", np.zeros_like(np.asarray(static.s)))
    gamma0 = float(indata.get_float("GAMMA", 0.0))

    r0 = force_residuals_from_state(st0, static, flux=flux0, pressure=pressure0, gamma=gamma0)

    print("vmec_jax diagnostics (initial guess):")
    print(f"  fsqr_like = {_fmt(r0.fsqr_like)}")
    print(f"  fsqz_like = {_fmt(r0.fsqz_like)}")
    print(f"  fsql_like = {_fmt(r0.fsql_like)}")
    print(f"  fsq_like  = {_fmt(r0.fsq_like)}")
    print(f"  grad_rms  = {_fmt(r0.grad_rms)}")

    if args.wout:
        wout = read_wout(args.wout)
        st_w = state_from_wout(wout)
        flux_w = FluxProfiles(
            phipf=np.asarray(wout.phipf),
            chipf=np.asarray(wout.chipf),
            phips=np.asarray(wout.phips),
            signgs=int(wout.signgs),
            lamscale=lamscale_from_phips(np.asarray(wout.phips), np.asarray(static.s)),
        )
        r_w = force_residuals_from_state(st_w, static, flux=flux_w, pressure=np.asarray(wout.presf), gamma=float(wout.gamma))

        print("\nvmec_jax diagnostics (wout equilibrium state):")
        print(f"  fsqr_like = {_fmt(r_w.fsqr_like)}")
        print(f"  fsqz_like = {_fmt(r_w.fsqz_like)}")
        print(f"  fsql_like = {_fmt(r_w.fsql_like)}")
        print(f"  fsq_like  = {_fmt(r_w.fsq_like)}")
        print(f"  grad_rms  = {_fmt(r_w.grad_rms)}")

        print("\nVMEC2000 wout residual scalars (reference):")
        print(f"  fsqr = {_fmt(float(wout.fsqr))}")
        print(f"  fsqz = {_fmt(float(wout.fsqz))}")
        print(f"  fsql = {_fmt(float(wout.fsql))}")


if __name__ == "__main__":
    main()

