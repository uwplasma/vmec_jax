from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.config import load_config
from vmec_jax.diagnostics import print_summary, summarize_array
from vmec_jax.static import build_static
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar, vmec_fsq_from_tomnsps
from vmec_jax.vmec_forces import (
    vmec_forces_rz_from_wout_reference_fields,
    vmec_forces_rz_from_wout,
    vmec_residual_internal_from_kernels,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _parse_args():
    p = argparse.ArgumentParser(description="VMEC constraint pipeline parity report")
    p.add_argument("input", type=str, help="Path to input.* file")
    p.add_argument("wout", type=str, help="Path to wout_*.nc file")
    p.add_argument("--outdir", type=str, default="examples/outputs", help="Output directory")
    p.add_argument("--hi-res", action="store_true", help="Use a higher angular grid for diagnostics")
    p.add_argument("--full-fields", action="store_true", help="Use vmec_jax-computed fields instead of wout reference fields")
    return p.parse_args()


def main():
    args = _parse_args()
    enable_x64()

    input_path = Path(args.input).resolve()
    wout_path = Path(args.wout).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not wout_path.exists():
        raise FileNotFoundError(wout_path)

    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)

    if args.hi_res:
        cfg = replace(cfg, ntheta=max(int(cfg.ntheta), 128), nzeta=max(int(cfg.nzeta), 128))

    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg, grid=grid)
    st = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )

    if args.full_fields:
        k = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata)
    else:
        k = vmec_forces_rz_from_wout_reference_fields(state=st, static=static, wout=wout, indata=indata)

    rzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(cfg.ntheta),
        cfg_nzeta=int(cfg.nzeta),
        wout=wout,
        trig=trig,
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

    norms = vmec_force_norms_from_bcovar(bc=k.bc, trig=trig, wout=wout, s=static.s)
    scal = vmec_fsq_from_tomnsps(frzl=frzl, norms=norms, lconm1=bool(getattr(cfg, "lconm1", True)))

    print("== VMEC2000 wout scalars ==")
    print(f"fsqr={wout.fsqr:.3e}  fsqz={wout.fsqz:.3e}  fsql={wout.fsql:.3e}")
    print("== vmec_jax (VMEC-style tomnsps + getfsq) ==")
    print(f"fsqr={scal.fsqr:.3e}  fsqz={scal.fsqz:.3e}  fsql={scal.fsql:.3e}")

    abs_err_r = abs(scal.fsqr - wout.fsqr)
    abs_err_z = abs(scal.fsqz - wout.fsqz)
    abs_err_l = abs(scal.fsql - wout.fsql)
    print("== absolute errors ==")
    print(f"fsqr abs.err={abs_err_r:.3e}")
    print(f"fsqz abs.err={abs_err_z:.3e}")
    print(f"fsql abs.err={abs_err_l:.3e}")

    denom_r = max(abs(wout.fsqr), 1e-20)
    denom_z = max(abs(wout.fsqz), 1e-20)
    denom_l = max(abs(wout.fsql), 1e-20)
    print("== relative errors ==")
    print(f"fsqr rel.err={abs(scal.fsqr - wout.fsqr)/denom_r:.3e}")
    print(f"fsqz rel.err={abs(scal.fsqz - wout.fsqz)/denom_z:.3e}")
    print(f"fsql rel.err={abs(scal.fsql - wout.fsql)/denom_l:.3e}")

    if k.tcon is not None:
        tcon = np.asarray(k.tcon)
        print(f"tcon: min={tcon.min():.3e} max={tcon.max():.3e} mean={tcon.mean():.3e}")
    gcon = np.asarray(k.gcon)
    print_summary(summarize_array("gcon", gcon))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = wout_path.stem.replace("wout_", "")
    outpath = outdir / f"constraint_pipeline_report_{tag}.npz"
    np.savez(
        outpath,
        s=np.asarray(static.s),
        fsqr=float(scal.fsqr),
        fsqz=float(scal.fsqz),
        fsql=float(scal.fsql),
        fsqr_ref=float(wout.fsqr),
        fsqz_ref=float(wout.fsqz),
        fsql_ref=float(wout.fsql),
        fsqr_abs_err=float(abs_err_r),
        fsqz_abs_err=float(abs_err_z),
        fsql_abs_err=float(abs_err_l),
        gcon=np.asarray(k.gcon),
        tcon=np.asarray(k.tcon) if k.tcon is not None else np.zeros((0,)),
    )
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
