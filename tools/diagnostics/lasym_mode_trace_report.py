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
from vmec_jax.vmec_residue import vmec_apply_m1_constraints
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _parse_args():
    p = argparse.ArgumentParser(description="Trace lasym (m,n) contributions through tomnspa blocks")
    p.add_argument("input", type=str, help="Path to input.* file")
    p.add_argument("wout", type=str, help="Path to wout_*.nc file")
    p.add_argument("--m", type=int, required=True, help="Poloidal mode m to inspect")
    p.add_argument("--n", type=int, required=True, help="Toroidal mode n to inspect (VMEC n)")
    p.add_argument("--outdir", type=str, default="examples/outputs", help="Output directory")
    p.add_argument("--hi-res", action="store_true", help="Use a higher angular grid for diagnostics")
    p.add_argument("--full-fields", action="store_true", help="Use vmec_jax-computed fields instead of wout reference fields")
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


def _block_mode_sum(a, m: int, n: int) -> float:
    if a is None:
        return 0.0
    arr = np.asarray(a)
    if m < 0 or m >= arr.shape[1]:
        return 0.0
    if n < 0 or n >= arr.shape[2]:
        return 0.0
    return float(np.sum(arr[:, m, n] ** 2))


def _block_mode_series(a, m: int, n: int) -> np.ndarray:
    if a is None:
        return np.zeros((0,), dtype=float)
    arr = np.asarray(a)
    if m < 0 or m >= arr.shape[1] or n < 0 or n >= arr.shape[2]:
        return np.zeros((arr.shape[0],), dtype=float)
    return np.asarray(arr[:, m, n], dtype=float)


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
    return k_A, k_B, k_C, k_con


def _extract_mode_report(frzl: TomnspsRZL, *, m: int, n: int) -> dict[str, float]:
    return {
        "frcc": _block_mode_sum(frzl.frcc, m, n),
        "frss": _block_mode_sum(frzl.frss, m, n),
        "frsc": _block_mode_sum(frzl.frsc, m, n),
        "frcs": _block_mode_sum(frzl.frcs, m, n),
        "fzsc": _block_mode_sum(frzl.fzsc, m, n),
        "fzcs": _block_mode_sum(frzl.fzcs, m, n),
        "fzcc": _block_mode_sum(frzl.fzcc, m, n),
        "fzss": _block_mode_sum(frzl.fzss, m, n),
        "flsc": _block_mode_sum(frzl.flsc, m, n),
        "flcs": _block_mode_sum(frzl.flcs, m, n),
        "flcc": _block_mode_sum(frzl.flcc, m, n),
        "flss": _block_mode_sum(frzl.flss, m, n),
    }


def _extract_mode_series(frzl: TomnspsRZL, *, m: int, n: int) -> dict[str, np.ndarray]:
    return {
        "frcc": _block_mode_series(frzl.frcc, m, n),
        "frss": _block_mode_series(frzl.frss, m, n),
        "frsc": _block_mode_series(frzl.frsc, m, n),
        "frcs": _block_mode_series(frzl.frcs, m, n),
        "fzsc": _block_mode_series(frzl.fzsc, m, n),
        "fzcs": _block_mode_series(frzl.fzcs, m, n),
        "fzcc": _block_mode_series(frzl.fzcc, m, n),
        "fzss": _block_mode_series(frzl.fzss, m, n),
        "flsc": _block_mode_series(frzl.flsc, m, n),
        "flcs": _block_mode_series(frzl.flcs, m, n),
        "flcc": _block_mode_series(frzl.flcc, m, n),
        "flss": _block_mode_series(frzl.flss, m, n),
    }


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

    frzl = _frzl_from_kernels(k, cfg=cfg, wout=wout, trig=trig)
    frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(cfg, "lconm1", True)))
    mode_report = _extract_mode_report(frzl, m=args.m, n=args.n)
    mode_series = _extract_mode_series(frzl, m=args.m, n=args.n)

    k_A, k_B, k_C, k_con = _build_component_kernels(k, con=con, s=static.s)
    frzl_A = _frzl_from_kernels(k_A, cfg=cfg, wout=wout, trig=trig)
    frzl_B = _frzl_from_kernels(k_B, cfg=cfg, wout=wout, trig=trig)
    frzl_C = _frzl_from_kernels(k_C, cfg=cfg, wout=wout, trig=trig)
    frzl_con = _frzl_from_kernels(k_con, cfg=cfg, wout=wout, trig=trig)

    frzl_A = vmec_apply_m1_constraints(frzl=frzl_A, lconm1=bool(getattr(cfg, "lconm1", True)))
    frzl_B = vmec_apply_m1_constraints(frzl=frzl_B, lconm1=bool(getattr(cfg, "lconm1", True)))
    frzl_C = vmec_apply_m1_constraints(frzl=frzl_C, lconm1=bool(getattr(cfg, "lconm1", True)))
    frzl_con = vmec_apply_m1_constraints(frzl=frzl_con, lconm1=bool(getattr(cfg, "lconm1", True)))

    mode_A = _extract_mode_report(frzl_A, m=args.m, n=args.n)
    mode_B = _extract_mode_report(frzl_B, m=args.m, n=args.n)
    mode_C = _extract_mode_report(frzl_C, m=args.m, n=args.n)
    mode_con = _extract_mode_report(frzl_con, m=args.m, n=args.n)

    series_A = _extract_mode_series(frzl_A, m=args.m, n=args.n)
    series_B = _extract_mode_series(frzl_B, m=args.m, n=args.n)
    series_C = _extract_mode_series(frzl_C, m=args.m, n=args.n)
    series_con = _extract_mode_series(frzl_con, m=args.m, n=args.n)

    print(f"== Mode trace (m={args.m}, n={args.n}) ==")
    print("Total blocks (sum of squares over s):")
    for key, val in mode_report.items():
        if val != 0.0:
            print(f"  {key:4s}  {val:.3e}")

    def _print_component(label: str, rep: dict[str, float]):
        print(f"-- {label} component --")
        for key, val in rep.items():
            if val != 0.0:
                print(f"  {key:4s}  {val:.3e}")

    _print_component("A-only", mode_A)
    _print_component("B-only", mode_B)
    _print_component("C-only", mode_C)
    _print_component("constraint", mode_con)

    gcon = np.asarray(k.gcon)
    print_summary(summarize_array("gcon", gcon))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = wout_path.stem.replace("wout_", "")
    outpath = outdir / f"lasym_mode_trace_report_{tag}_m{args.m}_n{args.n}.npz"
    np.savez(
        outpath,
        s=np.asarray(static.s),
        m=int(args.m),
        n=int(args.n),
        mode_total=mode_report,
        mode_A=mode_A,
        mode_B=mode_B,
        mode_C=mode_C,
        mode_con=mode_con,
        series_total=mode_series,
        series_A=series_A,
        series_B=series_B,
        series_C=series_C,
        series_con=series_con,
        gcon=np.asarray(k.gcon),
    )
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
