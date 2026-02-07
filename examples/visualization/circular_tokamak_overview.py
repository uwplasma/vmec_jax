"""Write a single VMEC-style overview panel for a bundled case."""

from __future__ import annotations

import argparse

import vmec_jax.api as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="circular_tokamak")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    vj.write_axisym_overview(case=args.case, outdir=args.outdir)


if __name__ == "__main__":
    main()

