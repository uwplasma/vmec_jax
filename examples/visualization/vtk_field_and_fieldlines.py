"""Export one surface B field + a fieldline to VTK."""

from __future__ import annotations

import argparse

import vmec_jax.api as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--wout", default=None)
    p.add_argument("--outdir", default="vtk_out")
    p.add_argument("--s-index", type=int, default=-1)
    p.add_argument("--hi-res", action="store_true")
    p.add_argument("--export-volume", action="store_true")
    args = p.parse_args()

    vj.export_vtk_surface_and_fieldline(
        input_path=args.input,
        wout_path=args.wout,
        outdir=args.outdir,
        s_index=args.s_index,
        hi_res=args.hi_res,
        export_volume=args.export_volume,
    )


if __name__ == "__main__":
    main()
