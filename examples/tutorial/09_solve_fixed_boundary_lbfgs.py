"""Step-7: fixed-boundary solve using L-BFGS (no external deps).

This script is a convenience wrapper around :func:`vmec_jax.solve.solve_fixed_boundary_lbfgs`.
It is useful for experimenting with faster convergence than plain gradient descent.
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
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.profiles import eval_profiles
from vmec_jax.solve import solve_fixed_boundary_lbfgs
from vmec_jax.static import build_static


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--max-iter", type=int, default=25)
    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--history-size", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if has_jax():
        enable_x64(True)

    cfg, indata = load_config(args.input)
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    prof = eval_profiles(indata, static.s)
    pressure = prof.get("pressure", np.zeros_like(np.asarray(static.s)))
    gamma = indata.get_float("GAMMA", 0.0)

    print("\n==== vmec_jax step-7: fixed-boundary solve (L-BFGS) ====")
    print(f"signgs={signgs}  gamma={gamma}")
    print(f"max_iter={args.max_iter}  step_size={args.step_size}  history={args.history_size}")

    res = solve_fixed_boundary_lbfgs(
        st0,
        static,
        phipf=flux.phipf,
        chipf=flux.chipf,
        signgs=signgs,
        lamscale=flux.lamscale,
        pressure=pressure,
        gamma=gamma,
        max_iter=int(args.max_iter),
        step_size=float(args.step_size),
        history_size=int(args.history_size),
        verbose=bool(args.verbose),
    )

    print("\n==== results ====")
    print(f"iters = {res.n_iter}")
    print(f"w0  = {res.w_history[0]:.8e}   wb0 = {res.wb_history[0]:.8e}   wp0 = {res.wp_history[0]:.8e}")
    print(f"wN  = {res.w_history[-1]:.8e}   wbN = {res.wb_history[-1]:.8e}   wpN = {res.wp_history[-1]:.8e}")
    print(f"rel_change_w = {(res.w_history[-1] - res.w_history[0]) / res.w_history[0]:.3e}")

    if args.verbose:
        print("\n==== history ====")
        for i in range(res.w_history.shape[0]):
            if i == 0:
                print(f"  {i:03d}: w={res.w_history[i]:.8e}")
            else:
                g = res.grad_rms_history[i - 1] if i - 1 < res.grad_rms_history.shape[0] else np.nan
                step = res.step_history[i - 1] if i - 1 < res.step_history.shape[0] else np.nan
                print(f"  {i:03d}: w={res.w_history[i]:.8e}  grad_rms={g:.3e}  step={step:.3e}")


if __name__ == "__main__":
    main()
