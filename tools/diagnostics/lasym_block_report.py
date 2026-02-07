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
from vmec_jax.vmec_forces import (
    vmec_forces_rz_from_wout_reference_fields,
    vmec_forces_rz_from_wout,
    vmec_residual_internal_from_kernels,
)
from vmec_jax.vmec_residue import vmec_apply_m1_constraints
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _parse_args():
    p = argparse.ArgumentParser(description="Lasym block contribution report (tomnsps + tomnspa)")
    p.add_argument("input", type=str, help="Path to input.* file")
    p.add_argument("wout", type=str, help="Path to wout_*.nc file")
    p.add_argument("--outdir", type=str, default="examples/outputs", help="Output directory")
    p.add_argument("--hi-res", action="store_true", help="Use a higher angular grid for diagnostics")
    p.add_argument("--full-fields", action="store_true", help="Use vmec_jax-computed fields instead of wout reference fields")
    p.add_argument("--topk", type=int, default=8, help="Show top-k (m,n) contributors per block")
    return p.parse_args()


def _frzl_from_kernels(k, *, cfg, wout, trig) -> TomnspsRZL:
    rzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(cfg.ntheta),
        cfg_nzeta=int(cfg.nzeta),
        wout=wout,
        trig=trig,
    )
    return TomnspsRZL(
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


def _sum_block(a):
    if a is None:
        return 0.0
    return float(np.sum(np.asarray(a) ** 2))


def _topk_block(label: str, arr, *, topk: int):
    if arr is None:
        return
    arr = np.asarray(arr)
    flat = np.sum(arr * arr, axis=0).reshape(-1)
    if flat.size == 0:
        return
    idx = np.argsort(flat)[::-1]
    total = float(np.sum(flat))
    print(f"-- Top {topk} (m,n) for {label} --")
    for rank in range(min(topk, idx.size)):
        k = int(idx[rank])
        m = k // arr.shape[2]
        n = k % arr.shape[2]
        val = float(flat[k])
        frac = val / total if total > 0 else 0.0
        print(f"  m={m:2d} n={n:2d}  sumsq={val:.3e}  frac={frac:.3e}")


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

    frzl = _frzl_from_kernels(k, cfg=cfg, wout=wout, trig=trig)
    frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(cfg, "lconm1", True)))

    if not bool(wout.lasym):
        print("lasym=False: tomnspa blocks are not populated for this case.")

    gcr_sym = _sum_block(frzl.frcc) + _sum_block(frzl.frss)
    gcz_sym = _sum_block(frzl.fzsc) + _sum_block(frzl.fzcs)
    gcl_sym = _sum_block(frzl.flsc) + _sum_block(frzl.flcs)

    gcr_asym = _sum_block(frzl.frsc) + _sum_block(frzl.frcs)
    gcz_asym = _sum_block(frzl.fzcc) + _sum_block(frzl.fzss)
    gcl_asym = _sum_block(frzl.flcc) + _sum_block(frzl.flss)

    print("== symmetric vs asymmetric block sums ==")
    print(f"gcr sym={gcr_sym:.3e}  gcr asym={gcr_asym:.3e}")
    print(f"gcz sym={gcz_sym:.3e}  gcz asym={gcz_asym:.3e}")
    print(f"gcl sym={gcl_sym:.3e}  gcl asym={gcl_asym:.3e}")

    print("== individual block sums ==")
    for label, arr in [
        ("frcc", frzl.frcc),
        ("frss", frzl.frss),
        ("frsc", frzl.frsc),
        ("frcs", frzl.frcs),
        ("fzsc", frzl.fzsc),
        ("fzcs", frzl.fzcs),
        ("fzcc", frzl.fzcc),
        ("fzss", frzl.fzss),
        ("flsc", frzl.flsc),
        ("flcs", frzl.flcs),
        ("flcc", frzl.flcc),
        ("flss", frzl.flss),
    ]:
        if arr is None:
            continue
        print(f"{label:4s} sumsq={_sum_block(arr):.3e}")

    for label, arr in [
        ("frcc", frzl.frcc),
        ("frss", frzl.frss),
        ("frsc", frzl.frsc),
        ("frcs", frzl.frcs),
        ("fzsc", frzl.fzsc),
        ("fzcs", frzl.fzcs),
        ("fzcc", frzl.fzcc),
        ("fzss", frzl.fzss),
        ("flsc", frzl.flsc),
        ("flcs", frzl.flcs),
        ("flcc", frzl.flcc),
        ("flss", frzl.flss),
    ]:
        _topk_block(label, arr, topk=int(args.topk))

    gcon = np.asarray(k.gcon)
    print_summary(summarize_array("gcon", gcon))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = wout_path.stem.replace("wout_", "")
    outpath = outdir / f"lasym_block_report_{tag}.npz"
    np.savez(
        outpath,
        s=np.asarray(static.s),
        gcr_sym=float(gcr_sym),
        gcr_asym=float(gcr_asym),
        gcz_sym=float(gcz_sym),
        gcz_asym=float(gcz_asym),
        gcl_sym=float(gcl_sym),
        gcl_asym=float(gcl_asym),
        gcon=np.asarray(k.gcon),
    )
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
