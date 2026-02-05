from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.config import load_config
from vmec_jax.diagnostics import print_summary, summarize_array
from vmec_jax.static import build_static
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar, vmec_fsq_from_tomnsps
from vmec_jax.vmec_forces import (
    vmec_forces_rz_from_wout,
    vmec_residual_internal_from_kernels,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def main():
    enable_x64()
    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples/data/input.circular_tokamak"
    wout_path = root / "examples/data/wout_circular_tokamak_reference.nc"

    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)

    # Use a moderate grid; increase if you want finer diagnostics.
    cfg_hi = replace(cfg, ntheta=max(int(cfg.ntheta), 128), nzeta=max(int(cfg.nzeta), 128))
    grid = vmec_angle_grid(ntheta=int(cfg_hi.ntheta), nzeta=int(cfg_hi.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg_hi, grid=grid)

    st = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=int(cfg_hi.ntheta),
        nzeta=int(cfg_hi.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )

    k = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata)
    rzl = vmec_residual_internal_from_kernels(k, cfg_ntheta=int(cfg_hi.ntheta), cfg_nzeta=int(cfg_hi.nzeta), wout=wout, trig=trig)
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
    scal = vmec_fsq_from_tomnsps(frzl=frzl, norms=norms)

    print("== VMEC2000 reference scalars ==")
    print(f"fsqr={wout.fsqr:.3e}  fsqz={wout.fsqz:.3e}  fsql={wout.fsql:.3e}")
    print("== vmec_jax (VMEC-style tomnsps + getfsq; parity WIP) ==")
    print(f"fsqr={scal.fsqr:.3e}  fsqz={scal.fsqz:.3e}  fsql={scal.fsql:.3e}")

    if k.tcon is not None:
        tcon = np.asarray(k.tcon)
        print(f"tcon: min={tcon.min():.3e} max={tcon.max():.3e} mean={tcon.mean():.3e}")
    gcon = np.asarray(k.gcon)
    print_summary(summarize_array("gcon", gcon))

    fr = np.asarray(frzl.frcc)
    fz = np.asarray(frzl.fzsc)
    fl = np.asarray(frzl.flsc)
    print_summary(summarize_array("||frcc||_2 per-surface", np.linalg.norm(fr.reshape(fr.shape[0], -1), axis=1)))
    print_summary(summarize_array("||fzsc||_2 per-surface", np.linalg.norm(fz.reshape(fz.shape[0], -1), axis=1)))
    print_summary(summarize_array("||flsc||_2 per-surface", np.linalg.norm(fl.reshape(fl.shape[0], -1), axis=1)))

    out = root / "examples/outputs"
    out.mkdir(exist_ok=True)
    np.savez(
        out / "step10_forces_rz_kernel_report.npz",
        s=np.asarray(static.s),
        frcc=np.asarray(frzl.frcc),
        frss=np.asarray(frzl.frss) if frzl.frss is not None else np.zeros((0,)),
        fzsc=np.asarray(frzl.fzsc),
        fzcs=np.asarray(frzl.fzcs) if frzl.fzcs is not None else np.zeros((0,)),
        flsc=np.asarray(frzl.flsc),
        flcs=np.asarray(frzl.flcs) if frzl.flcs is not None else np.zeros((0,)),
        fnorm=float(norms.fnorm),
        fnormL=float(norms.fnormL),
        fsqr=float(scal.fsqr),
        fsqz=float(scal.fsqz),
        fsql=float(scal.fsql),
        fsqr_ref=float(wout.fsqr),
        fsqz_ref=float(wout.fsqz),
        fsql_ref=float(wout.fsql),
        gcon=np.asarray(k.gcon),
        tcon=np.asarray(k.tcon) if k.tcon is not None else np.zeros((0,)),
    )
    print(f"Wrote {out / 'step10_forces_rz_kernel_report.npz'}")


if __name__ == "__main__":
    main()
