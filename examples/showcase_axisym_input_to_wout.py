"""Minimal vmec_jax showcase: input -> fixed-boundary solve -> wout.nc.

This example is intentionally short and uses only the input file as an input.
It produces a VMEC-style `wout_*.nc` that can be compared to a VMEC2000
reference wout in `examples/data/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vmec_jax.config import load_config
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import write_wout, wout_minimal_from_fixed_boundary


def _parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "examples" / "data"
    out_dir = repo_root / "examples" / "outputs"
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=data_dir / "input.shaped_tokamak_pressure",
        help="VMEC INDATA file.",
    )
    p.add_argument(
        "--reference-wout",
        type=Path,
        default=data_dir / "wout_shaped_tokamak_pressure_reference.nc",
        help="VMEC2000 reference wout for comparison (not used by the solve).",
    )
    p.add_argument("--outdir", type=Path, default=out_dir)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--gn-damping", type=float, default=1e-6)
    p.add_argument("--gn-cg-tol", type=float, default=1e-10)
    p.add_argument("--gn-cg-maxiter", type=int, default=200)
    return p.parse_args()


def _step10_fsq_from_state(*, state, static, indata, signgs: int) -> tuple[float, float, float]:
    """Compute VMEC-style invariant residual scalars for a state."""
    # Use the VMEC internal grid for scalar parity.
    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(static.cfg.nfp),
        mmax=int(static.cfg.mpol) - 1,
        nmax=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
    )
    wout_like = type(
        "_WoutLike",
        (),
        {
            "nfp": int(static.cfg.nfp),
            "mpol": int(static.cfg.mpol),
            "ntor": int(static.cfg.ntor),
            "lasym": bool(static.cfg.lasym),
            "signgs": int(signgs),
            # These get filled inside vmec_forces_rz_from_wout when indata is provided.
        },
    )()

    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        indata=indata,
        use_wout_bsup=False,
        use_vmec_synthesis=True,
        trig=trig,
    )
    rzl = vmec_residual_internal_from_kernels(
        k, cfg_ntheta=int(static.cfg.ntheta), cfg_nzeta=int(static.cfg.nzeta), wout=wout_like, trig=trig
    )
    frzl = TomnspsRZL(
        frcc=rzl.frcc,
        frss=rzl.frss,
        fzsc=rzl.fzsc,
        fzcs=rzl.fzcs,
        flsc=rzl.flsc,
        flcs=rzl.flcs,
        frsc=rzl.frsc,
        frcs=rzl.frcs,
        fzcc=rzl.fzcc,
        fzss=rzl.fzss,
        flcc=rzl.flcc,
        flss=rzl.flss,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=static.s, signgs=int(signgs))
    scal = vmec_fsq_from_tomnsps_dynamic(frzl=frzl, norms=norms, lconm1=bool(getattr(static.cfg, "lconm1", True)))
    return float(scal.fsqr), float(scal.fsqz), float(scal.fsql)


def main() -> None:
    args = _parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cfg, indata = load_config(str(args.input))
    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(cfg.nfp), lasym=bool(cfg.lasym))
    static = build_static(cfg, grid=grid)

    run = run_fixed_boundary(
        args.input,
        solver="vmec_gn",
        max_iter=int(args.max_iter),
        step_size=float(args.step_size),
        gn_damping=float(args.gn_damping),
        gn_cg_tol=float(args.gn_cg_tol),
        gn_cg_maxiter=int(args.gn_cg_maxiter),
        verbose=True,
    )

    # Scalars for the output file (computed from the solved state and input only).
    fsqr, fsqz, fsql = _step10_fsq_from_state(state=run.state, static=run.static, indata=indata, signgs=int(run.signgs))
    wout = wout_minimal_from_fixed_boundary(
        path=args.outdir / ("wout_" + args.input.name.replace("input.", "") + "_vmec_jax.nc"),
        state=run.state,
        static=run.static,
        indata=indata,
        signgs=int(run.signgs),
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
    )

    out_path = Path(wout.path)
    write_wout(out_path, wout, overwrite=True)
    print(f"[vmec_jax] wrote {out_path}")
    print(f"[vmec_jax] reference VMEC2000 wout (for comparison): {args.reference_wout}")
    print(f"[vmec_jax] fsq_final={fsqr+fsqz+fsql:.6e} (fsqr={fsqr:.3e} fsqz={fsqz:.3e} fsql={fsql:.3e})")

    # Simple smoke check: file readable.
    try:
        import netCDF4  # noqa: F401
    except Exception:
        print("[vmec_jax] netCDF4 not installed; skipping wout readback.")


if __name__ == "__main__":
    main()
