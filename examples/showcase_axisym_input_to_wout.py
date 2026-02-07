"""Axisymmetric showcase: run bundled inputs, write wouts, plot, compare."""

from __future__ import annotations

import argparse

import vmec_jax.api as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=None)
    p.add_argument("--solver", default="vmecpp_iter", choices=["vmecpp_iter", "vmec_gn"])
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--step-size", type=float, default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    vj.axisym_showcase(
        outdir=args.outdir,
        solver=args.solver,
        max_iter=args.max_iter,
        step_size=args.step_size,
        plots=not args.no_plots,
        verbose=True,
    )


if __name__ == "__main__":
    main()

