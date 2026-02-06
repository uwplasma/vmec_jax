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
    p = argparse.ArgumentParser(description="Compare residual contributions for full vs reference fields")
    p.add_argument("input", type=str, help="Path to input.* file")
    p.add_argument("wout", type=str, help="Path to wout_*.nc file")
    p.add_argument("--outdir", type=str, default="examples/outputs", help="Output directory")
    p.add_argument("--hi-res", action="store_true", help="Use a higher angular grid for diagnostics")
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


def _component_sums(k, *, cfg, wout, trig):
    frzl = _frzl_from_kernels(k, cfg=cfg, wout=wout, trig=trig)
    sums = vmec_fsq_sums_from_tomnsps(frzl=frzl, lconm1=bool(getattr(cfg, "lconm1", True)))
    return sums


def _build_component_kernels(k, *, con, s):
    psqrts = np.sqrt(np.maximum(np.asarray(s), 0.0))[:, None, None]
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
    return k_A, k_B, k_C, k_con, k_L


def _compare_component(label, sums_ref, sums_full):
    gcr2 = sums_full.gcr2 - sums_ref.gcr2
    gcz2 = sums_full.gcz2 - sums_ref.gcz2
    gcl2 = sums_full.gcl2 - sums_ref.gcl2
    print(f"{label:12s} Δgcr2={gcr2:.3e}  Δgcz2={gcz2:.3e}  Δgcl2={gcl2:.3e}")


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

    k_ref = vmec_forces_rz_from_wout_reference_fields(state=st, static=static, wout=wout, indata=indata)
    k_full = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata)

    tcon0 = float(indata.get_float("TCON0", 0.0)) if indata is not None else 0.0
    con_ref = _constraint_kernels_from_state(
        state=st,
        static=static,
        wout=wout,
        bc=k_ref.bc,
        pru_0=k_ref.pru_even,
        pru_1=k_ref.pru_odd,
        pzu_0=k_ref.pzu_even,
        pzu_1=k_ref.pzu_odd,
        constraint_tcon0=tcon0,
    )
    con_full = _constraint_kernels_from_state(
        state=st,
        static=static,
        wout=wout,
        bc=k_full.bc,
        pru_0=k_full.pru_even,
        pru_1=k_full.pru_odd,
        pzu_0=k_full.pzu_even,
        pzu_1=k_full.pzu_odd,
        constraint_tcon0=tcon0,
    )

    kA_ref, kB_ref, kC_ref, kcon_ref, kL_ref = _build_component_kernels(k_ref, con=con_ref, s=static.s)
    kA_full, kB_full, kC_full, kcon_full, kL_full = _build_component_kernels(k_full, con=con_full, s=static.s)

    norms = vmec_force_norms_from_bcovar_dynamic(bc=k_ref.bc, trig=trig, s=static.s, signgs=int(wout.signgs))
    frzl_ref = _frzl_from_kernels(k_ref, cfg=cfg, wout=wout, trig=trig)
    frzl_full = _frzl_from_kernels(k_full, cfg=cfg, wout=wout, trig=trig)

    scal_ref = vmec_fsq_from_tomnsps_dynamic(frzl=frzl_ref, norms=norms, lconm1=bool(getattr(cfg, "lconm1", True)))
    scal_full = vmec_fsq_from_tomnsps_dynamic(frzl=frzl_full, norms=norms, lconm1=bool(getattr(cfg, "lconm1", True)))
    fsqr_ref = float(scal_ref.fsqr)
    fsqz_ref = float(scal_ref.fsqz)
    fsql_ref = float(scal_ref.fsql)
    fsqr_full = float(scal_full.fsqr)
    fsqz_full = float(scal_full.fsqz)
    fsql_full = float(scal_full.fsql)

    print("== VMEC2000 wout scalars ==")
    print(f"fsqr={wout.fsqr:.3e}  fsqz={wout.fsqz:.3e}  fsql={wout.fsql:.3e}")
    print("== vmec_jax scalars ==")
    print(f"reference  fsqr={fsqr_ref:.3e}  fsqz={fsqz_ref:.3e}  fsql={fsql_ref:.3e}")
    print(f"full       fsqr={fsqr_full:.3e}  fsqz={fsqz_full:.3e}  fsql={fsql_full:.3e}")

    sums_ref = vmec_fsq_sums_from_tomnsps(frzl=frzl_ref, lconm1=bool(getattr(cfg, "lconm1", True)))
    sums_full = vmec_fsq_sums_from_tomnsps(frzl=frzl_full, lconm1=bool(getattr(cfg, "lconm1", True)))
    print("== total sums: full - reference ==")
    _compare_component("total", sums_ref, sums_full)

    sums_A_ref = _component_sums(kA_ref, cfg=cfg, wout=wout, trig=trig)
    sums_A_full = _component_sums(kA_full, cfg=cfg, wout=wout, trig=trig)
    sums_B_ref = _component_sums(kB_ref, cfg=cfg, wout=wout, trig=trig)
    sums_B_full = _component_sums(kB_full, cfg=cfg, wout=wout, trig=trig)
    sums_C_ref = _component_sums(kC_ref, cfg=cfg, wout=wout, trig=trig)
    sums_C_full = _component_sums(kC_full, cfg=cfg, wout=wout, trig=trig)
    sums_con_ref = _component_sums(kcon_ref, cfg=cfg, wout=wout, trig=trig)
    sums_con_full = _component_sums(kcon_full, cfg=cfg, wout=wout, trig=trig)
    sums_L_ref = _component_sums(kL_ref, cfg=cfg, wout=wout, trig=trig)
    sums_L_full = _component_sums(kL_full, cfg=cfg, wout=wout, trig=trig)

    print("== component delta (full - reference) ==")
    _compare_component("A-only", sums_A_ref, sums_A_full)
    _compare_component("B-only", sums_B_ref, sums_B_full)
    _compare_component("C-only", sums_C_ref, sums_C_full)
    _compare_component("constraint", sums_con_ref, sums_con_full)
    _compare_component("lambda", sums_L_ref, sums_L_full)

    frzl_ref_modes = vmec_apply_m1_constraints(frzl=frzl_ref, lconm1=bool(getattr(cfg, "lconm1", True)))
    frzl_full_modes = vmec_apply_m1_constraints(frzl=frzl_full, lconm1=bool(getattr(cfg, "lconm1", True)))

    gcr_ref, gcz_ref, gcl_ref = _mn_sums(frzl_ref_modes)
    gcr_full, gcz_full, gcl_full = _mn_sums(frzl_full_modes)

    gcr_delta = np.abs(np.asarray(gcr_full) - np.asarray(gcr_ref))
    gcz_delta = np.abs(np.asarray(gcz_full) - np.asarray(gcz_ref))
    gcl_delta = np.abs(np.asarray(gcl_full) - np.asarray(gcl_ref))

    _print_topk("Δgcr", gcr_delta, topk=int(args.topk))
    _print_topk("Δgcz", gcz_delta, topk=int(args.topk))
    _print_topk("Δgcl", gcl_delta, topk=int(args.topk))

    print_summary(summarize_array("gcon_ref", np.asarray(con_ref.gcon)))
    print_summary(summarize_array("gcon_full", np.asarray(con_full.gcon)))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = wout_path.stem.replace("wout_", "")
    outpath = outdir / f"residual_compare_fields_report_{tag}.npz"
    np.savez(
        outpath,
        s=np.asarray(static.s),
        fsqr_ref=float(fsqr_ref),
        fsqz_ref=float(fsqz_ref),
        fsql_ref=float(fsql_ref),
        fsqr_full=float(fsqr_full),
        fsqz_full=float(fsqz_full),
        fsql_full=float(fsql_full),
        gcr2_ref=float(sums_ref.gcr2),
        gcz2_ref=float(sums_ref.gcz2),
        gcl2_ref=float(sums_ref.gcl2),
        gcr2_full=float(sums_full.gcr2),
        gcz2_full=float(sums_full.gcz2),
        gcl2_full=float(sums_full.gcl2),
        gcr2_A_ref=float(sums_A_ref.gcr2),
        gcr2_A_full=float(sums_A_full.gcr2),
        gcr2_B_ref=float(sums_B_ref.gcr2),
        gcr2_B_full=float(sums_B_full.gcr2),
        gcr2_C_ref=float(sums_C_ref.gcr2),
        gcr2_C_full=float(sums_C_full.gcr2),
        gcr2_con_ref=float(sums_con_ref.gcr2),
        gcr2_con_full=float(sums_con_full.gcr2),
        gcl2_L_ref=float(sums_L_ref.gcl2),
        gcl2_L_full=float(sums_L_full.gcl2),
        gcr_delta=np.asarray(gcr_delta),
        gcz_delta=np.asarray(gcz_delta),
        gcl_delta=np.asarray(gcl_delta),
        gcon_ref=np.asarray(con_ref.gcon),
        gcon_full=np.asarray(con_full.gcon),
        tcon_ref=np.asarray(con_ref.tcon) if con_ref.tcon is not None else np.zeros((0,)),
        tcon_full=np.asarray(con_full.tcon) if con_full.tcon is not None else np.zeros((0,)),
    )
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
