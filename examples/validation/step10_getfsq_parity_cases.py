"""Step-10 `getfsq` scalar parity report for bundled reference `wout` files.

This script is intentionally fast by default:
- It does not run a fixed-boundary solve unless `--solve` is provided.
- It evaluates the Step-10 pipeline on the *reference* VMEC state from `wout`,
  so kernel mismatches show up without solver noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmec_jax.api as vj
from vmec_jax.config import load_config
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.static import build_static
from vmec_jax.vmec_tomnsp import vmec_angle_grid
from vmec_jax.wout import read_wout, state_from_wout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--solve-metric",
        action="store_true",
        help="Alias for the default behavior (compute Step-10 scalars on reference wouts; no solve).",
    )
    p.add_argument("--solve", action="store_true", help="Also run a vmec_jax fixed-boundary solve per case.")
    p.add_argument("--solver", default="vmecpp_iter", help="Solver to use when `--solve` is enabled.")
    p.add_argument("--max-iter", type=int, default=10, help="Max iterations when `--solve` is enabled.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Include additional (potentially slower) bundled cases beyond the axisymmetric trio.",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]

    # Fast-by-default set: axisymmetric cases that load/compute quickly and make
    # it easier to diagnose conventions without 3D coupling.
    cases = [
        "circular_tokamak",
        "shaped_tokamak_pressure",
        "vmecpp_solovev",
    ]
    if args.all:
        # Keep this list small: this is a kernel parity script, not a benchmark.
        cases += [
            "li383_low_res",
            "n3are_R7.75B5.7_lowres",
        ]

    print("[vmec_jax] step10 getfsq parity (reference wout states)")
    print(f"[vmec_jax] cases={cases}")
    print("[vmec_jax] note: this script does not solve unless `--solve` is set")

    data_dir = root / "examples" / "data"

    for case in cases:
        input_path = data_dir / f"input.{case}"
        wout_path = data_dir / f"wout_{case}_reference.nc"
        if not wout_path.exists():
            wout_path = data_dir / f"wout_{case}.nc"
        if not wout_path.exists():
            print(f"[skip] {case}: no bundled reference wout")
            continue

        cfg, indata = load_config(str(input_path))
        grid = vmec_angle_grid(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(cfg.nfp),
            lasym=bool(cfg.lasym),
        )
        static = build_static(cfg, grid=grid)
        wout = read_wout(wout_path)
        state = state_from_wout(wout)

        g = eval_geom(state, static)
        signgs = signgs_from_sqrtg(np.asarray(g.sqrtg), axis_index=1)

        fsqr, fsqz, fsql = vj.step10_fsq_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            use_vmec_synthesis=True,
        )
        fsqr_ref = float(getattr(wout, "fsqr", np.nan))
        fsqz_ref = float(getattr(wout, "fsqz", np.nan))
        fsql_ref = float(getattr(wout, "fsql", np.nan))

        def _rel_err(x: float, y: float) -> float:
            denom = max(1e-16, abs(y))
            return abs(x - y) / denom

        print(
            f"[{case}] fsqr={fsqr:.6e} (ref {fsqr_ref:.6e}, rel {_rel_err(fsqr, fsqr_ref):.3e}) "
            f"fsqz={fsqz:.6e} (ref {fsqz_ref:.6e}, rel {_rel_err(fsqz, fsqz_ref):.3e}) "
            f"fsql={fsql:.6e} (ref {fsql_ref:.6e}, rel {_rel_err(fsql, fsql_ref):.3e})"
        )

        if args.solve:
            run = vj.run_fixed_boundary(
                input_path,
                solver=str(args.solver),
                max_iter=int(args.max_iter),
                verbose=True,
            )
            fsqr_s, fsqz_s, fsql_s = vj.step10_fsq_from_state(
                state=run.state,
                static=run.static,
                indata=run.indata,
                signgs=int(run.signgs),
                use_vmec_synthesis=True,
            )
            print(
                f"[{case}] (vmec_jax solve) fsqr={fsqr_s:.6e} fsqz={fsqz_s:.6e} fsql={fsql_s:.6e} "
                f"(solver={args.solver}, max_iter={args.max_iter})"
            )


if __name__ == "__main__":
    main()
