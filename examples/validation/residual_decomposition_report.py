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
    _constraint_kernels_from_state,
    vmec_forces_rz_from_wout_reference_fields,
    vmec_forces_rz_from_wout,
    vmec_residual_internal_from_kernels,
)
from vmec_jax.vmec_residue import (
    vmec_apply_m1_constraints,
    vmec_force_norms_from_bcovar_dynamic,
    vmec_fsq_from_tomnsps_dynamic,
    vmec_fsq_sums_from_tomnsps,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _parse_args():
    p = argparse.ArgumentParser(description="VMEC residual decomposition report")
    p.add_argument("input", type=str, help="Path to input.* file")
    p.add_argument("wout", type=str, help="Path to wout_*.nc file")
    p.add_argument("--outdir", type=str, default="examples/outputs", help="Output directory")
    p.add_argument("--hi-res", action="store_true", help="Use a higher angular grid for diagnostics")
    p.add_argument("--full-fields", action="store_true", help="Use vmec_jax-computed fields instead of wout reference fields")
    p.add_argument("--topk", type=int, default=8, help="Show top-k (m,n) contributors")
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


def _block_sum(a):
    if a is None:
        return None
    return np.sum(np.asarray(a) ** 2, axis=0)


def _mn_sums(frzl: TomnspsRZL):
    gcr = _block_sum(frzl.frcc)
    if frzl.frss is not None:
        gcr = gcr + _block_sum(frzl.frss)
    if getattr(frzl, "frsc", None) is not None:
        gcr = gcr + _block_sum(frzl.frsc)
    if getattr(frzl, "frcs", None) is not None:
        gcr = gcr + _block_sum(frzl.frcs)

    gcz = _block_sum(frzl.fzsc)
    if frzl.fzcs is not None:
        gcz = gcz + _block_sum(frzl.fzcs)
    if getattr(frzl, "fzcc", None) is not None:
        gcz = gcz + _block_sum(frzl.fzcc)
    if getattr(frzl, "fzss", None) is not None:
        gcz = gcz + _block_sum(frzl.fzss)

    gcl = _block_sum(frzl.flsc)
    if frzl.flcs is not None:
        gcl = gcl + _block_sum(frzl.flcs)
    if getattr(frzl, "flcc", None) is not None:
        gcl = gcl + _block_sum(frzl.flcc)
    if getattr(frzl, "flss", None) is not None:
        gcl = gcl + _block_sum(frzl.flss)

    return gcr, gcz, gcl


def _print_topk(label: str, arr, *, topk: int):
    if arr is None:
        return
    flat = np.asarray(arr).reshape(-1)
    if flat.size == 0:
        return
    idx = np.argsort(flat)[::-1]
    total = float(np.sum(flat))
    print(f"-- Top {topk} (m,n) for {label} --")
    for rank in range(min(topk, idx.size)):
        k = int(idx[rank])
        m = k // arr.shape[1]
        n = k % arr.shape[1]
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

    tcon0 = float(indata.get_float("TCON0", 0.0)) if indata is not None else 0.0
    con = _constraint_kernels_from_state(
        state=st,
        static=static,
        wout=wout,
        bc=k.bc,
        pru_0=k.pru_even,
        pru_1=k.pru_odd,
        pzu_0=k.pzu_even,
        pzu_1=k.pzu_odd,
        constraint_tcon0=tcon0,
    )

    s = np.asarray(static.s)
    psqrts = np.sqrt(np.maximum(s, 0.0))[:, None, None]

    brmn_con_e = con.rcon_force
    brmn_con_o = con.rcon_force * psqrts
    bzmn_con_e = con.zcon_force
    bzmn_con_o = con.zcon_force * psqrts

    brmn_phys_e = k.brmn_e - brmn_con_e
    brmn_phys_o = k.brmn_o - brmn_con_o
    bzmn_phys_e = k.bzmn_e - bzmn_con_e
    bzmn_phys_o = k.bzmn_o - bzmn_con_o

    z = np.zeros_like(np.asarray(k.armn_e))
    k_A = replace(
        k,
        brmn_e=z,
        brmn_o=z,
        crmn_e=z,
        crmn_o=z,
        bzmn_e=z,
        bzmn_o=z,
        czmn_e=z,
        czmn_o=z,
        arcon_e=z,
        arcon_o=z,
        azcon_e=z,
        azcon_o=z,
        gcon=np.zeros_like(np.asarray(k.gcon)),
    )
    k_B = replace(
        k,
        armn_e=z,
        armn_o=z,
        crmn_e=z,
        crmn_o=z,
        azmn_e=z,
        azmn_o=z,
        czmn_e=z,
        czmn_o=z,
        arcon_e=z,
        arcon_o=z,
        azcon_e=z,
        azcon_o=z,
        brmn_e=brmn_phys_e,
        brmn_o=brmn_phys_o,
        bzmn_e=bzmn_phys_e,
        bzmn_o=bzmn_phys_o,
        gcon=np.zeros_like(np.asarray(k.gcon)),
    )
    k_C = replace(
        k,
        armn_e=z,
        armn_o=z,
        brmn_e=z,
        brmn_o=z,
        azmn_e=z,
        azmn_o=z,
        bzmn_e=z,
        bzmn_o=z,
        arcon_e=z,
        arcon_o=z,
        azcon_e=z,
        azcon_o=z,
        gcon=np.zeros_like(np.asarray(k.gcon)),
    )
    k_con = replace(
        k,
        armn_e=z,
        armn_o=z,
        crmn_e=z,
        crmn_o=z,
        azmn_e=z,
        azmn_o=z,
        czmn_e=z,
        czmn_o=z,
        brmn_e=brmn_con_e,
        brmn_o=brmn_con_o,
        bzmn_e=bzmn_con_e,
        bzmn_o=bzmn_con_o,
        arcon_e=con.arcon_e,
        arcon_o=con.arcon_o,
        azcon_e=con.azcon_e,
        azcon_o=con.azcon_o,
        gcon=con.gcon,
        tcon=con.tcon,
    )

    frzl_total = _frzl_from_kernels(k, cfg=cfg, wout=wout, trig=trig)
    sums_total = vmec_fsq_sums_from_tomnsps(frzl=frzl_total, lconm1=bool(getattr(cfg, "lconm1", True)))

    norms_dyn = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=static.s, signgs=int(wout.signgs))
    scal_total = vmec_fsq_from_tomnsps_dynamic(frzl=frzl_total, norms=norms_dyn, lconm1=bool(getattr(cfg, "lconm1", True)))
    fsqr_total = float(scal_total.fsqr)
    fsqz_total = float(scal_total.fsqz)
    fsql_total = float(scal_total.fsql)
    r1 = float(norms_dyn.r1)
    fnorm = float(norms_dyn.fnorm)
    fnormL = float(norms_dyn.fnormL)

    print("== VMEC2000 wout scalars ==")
    print(f"fsqr={wout.fsqr:.3e}  fsqz={wout.fsqz:.3e}  fsql={wout.fsql:.3e}")
    print("== vmec_jax (VMEC-style tomnsps + getfsq) ==")
    print(f"fsqr={fsqr_total:.3e}  fsqz={fsqz_total:.3e}  fsql={fsql_total:.3e}")
    print("== sum-of-squares blocks (total) ==")
    print(f"gcr2={sums_total.gcr2:.3e}  gcz2={sums_total.gcz2:.3e}  gcl2={sums_total.gcl2:.3e}")

    def _component_sums(label: str, kc):
        frzl = _frzl_from_kernels(kc, cfg=cfg, wout=wout, trig=trig)
        sums = vmec_fsq_sums_from_tomnsps(frzl=frzl, lconm1=bool(getattr(cfg, "lconm1", True)))
        fsqr = r1 * fnorm * sums.gcr2
        fsqz = r1 * fnorm * sums.gcz2
        print(f"{label:12s} fsqr={fsqr:.3e}  fsqz={fsqz:.3e}  gcr2={sums.gcr2:.3e}  gcz2={sums.gcz2:.3e}")
        return sums

    print("== component-only norms (do not sum; cross-terms omitted) ==")
    sums_A = _component_sums("A-only", k_A)
    sums_B = _component_sums("B-only", k_B)
    sums_C = _component_sums("C-only", k_C)
    sums_con = _component_sums("constraint", k_con)

    # Lambda-only contribution (from blmn/clmn).
    k_L = replace(
        k,
        armn_e=z,
        armn_o=z,
        brmn_e=z,
        brmn_o=z,
        crmn_e=z,
        crmn_o=z,
        azmn_e=z,
        azmn_o=z,
        bzmn_e=z,
        bzmn_o=z,
        czmn_e=z,
        czmn_o=z,
        arcon_e=z,
        arcon_o=z,
        azcon_e=z,
        azcon_o=z,
        gcon=np.zeros_like(np.asarray(k.gcon)),
    )
    frzl_L = _frzl_from_kernels(k_L, cfg=cfg, wout=wout, trig=trig)
    sums_L = vmec_fsq_sums_from_tomnsps(frzl=frzl_L, lconm1=bool(getattr(cfg, "lconm1", True)))
    fsql_L = fnormL * sums_L.gcl2
    print(f"{'lambda':12s} fsql={fsql_L:.3e}  gcl2={sums_L.gcl2:.3e}")

    frzl_modes = vmec_apply_m1_constraints(frzl=frzl_total, lconm1=bool(getattr(cfg, "lconm1", True)))
    gcr_mn, gcz_mn, gcl_mn = _mn_sums(frzl_modes)
    _print_topk("gcr", gcr_mn, topk=int(args.topk))
    _print_topk("gcz", gcz_mn, topk=int(args.topk))
    _print_topk("gcl", gcl_mn, topk=int(args.topk))

    if con.tcon is not None:
        tcon = np.asarray(con.tcon)
        print(f"tcon: min={tcon.min():.3e} max={tcon.max():.3e} mean={tcon.mean():.3e}")
    gcon = np.asarray(con.gcon)
    print_summary(summarize_array("gcon", gcon))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = wout_path.stem.replace("wout_", "")
    outpath = outdir / f"residual_decomposition_report_{tag}.npz"
    np.savez(
        outpath,
        s=np.asarray(static.s),
        fsqr=float(fsqr_total),
        fsqz=float(fsqz_total),
        fsql=float(fsql_total),
        fsqr_ref=float(wout.fsqr),
        fsqz_ref=float(wout.fsqz),
        fsql_ref=float(wout.fsql),
        gcr2=float(sums_total.gcr2),
        gcz2=float(sums_total.gcz2),
        gcl2=float(sums_total.gcl2),
        gcr2_A=float(sums_A.gcr2),
        gcz2_A=float(sums_A.gcz2),
        gcr2_B=float(sums_B.gcr2),
        gcz2_B=float(sums_B.gcz2),
        gcr2_C=float(sums_C.gcr2),
        gcz2_C=float(sums_C.gcz2),
        gcr2_con=float(sums_con.gcr2),
        gcz2_con=float(sums_con.gcz2),
        gcl2_L=float(sums_L.gcl2),
        gcr_mn=np.asarray(gcr_mn),
        gcz_mn=np.asarray(gcz_mn),
        gcl_mn=np.asarray(gcl_mn),
        gcon=np.asarray(con.gcon),
        tcon=np.asarray(con.tcon) if con.tcon is not None else np.zeros((0,)),
    )
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
