from __future__ import annotations

import argparse
import time
from pathlib import Path

from vmec_jax.driver import run_fixed_boundary


def _parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        type=str,
        default="circular_tokamak,ITERModel,circular_tokamak_aspect_100,purely_toroidal_field",
        help="Comma-separated bundled case names (input.<case> in examples/data/).",
    )
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--gn-damping", type=float, default=1e-6)
    p.add_argument("--gn-cg-tol", type=float, default=1e-10)
    p.add_argument("--gn-cg-maxiter", type=int, default=200)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--data-root", type=Path, default=root / "examples" / "data")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cases = [c.strip() for c in str(args.cases).split(",") if c.strip()]
    if not cases:
        raise SystemExit("No cases provided.")

    print("[vmec_jax] axisymmetric solve metrics (vmec_gn)")
    print(f"[vmec_jax] max_iter={args.max_iter} step_size={args.step_size}")
    print(f"[vmec_jax] gn_damping={args.gn_damping} gn_cg_tol={args.gn_cg_tol} gn_cg_maxiter={args.gn_cg_maxiter}")

    for case in cases:
        input_path = Path(args.data_root) / f"input.{case}"
        if not input_path.exists():
            print(f"[vmec_jax] case={case} missing input: {input_path}")
            continue

        t0 = time.time()
        run = run_fixed_boundary(
            input_path,
            solver="vmec_gn",
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            gn_damping=float(args.gn_damping),
            gn_cg_tol=float(args.gn_cg_tol),
            gn_cg_maxiter=int(args.gn_cg_maxiter),
            verbose=bool(args.verbose),
        )
        res = run.result
        fsqr2 = float(getattr(res, "fsqr2_history")[-1])
        fsqz2 = float(getattr(res, "fsqz2_history")[-1])
        fsql2 = float(getattr(res, "fsql2_history")[-1])
        fsq = fsqr2 + fsqz2 + fsql2
        dt = time.time() - t0
        print(f"[vmec_jax] case={case} fsq_final={fsq:.6e} (fsqr={fsqr2:.3e} fsqz={fsqz2:.3e} fsql={fsql2:.3e}) time={dt:.1f}s")


if __name__ == "__main__":
    main()

