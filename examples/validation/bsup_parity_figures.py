"""Write bsup parity figures vs a VMEC2000 wout reference."""

from __future__ import annotations

import argparse

import vmec_jax.api as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--wout", required=True)
    p.add_argument("--outdir", default="figures_bsup_parity")
    args = p.parse_args()
    vj.write_bsup_parity_figures(input_path=args.input, wout_path=args.wout, outdir=args.outdir)


if __name__ == "__main__":
    main()

