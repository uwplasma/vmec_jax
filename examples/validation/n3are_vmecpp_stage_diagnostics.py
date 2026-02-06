"""Stage-by-stage diagnostics for vmec_jax VMEC++ fixed-point path on n3are.

This script compares key intermediate quantities for:
- initial guess
- solver output (vmecpp_iter)

It is intended to localize where parity starts to diverge.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vmec_jax.driver import run_fixed_boundary
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import half_mesh_avg_from_full_mesh
from vmec_jax.geom import eval_geom
from vmec_jax.profiles import eval_profiles
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import (
    vmec_apply_m1_constraints,
    vmec_apply_scalxc_to_tomnsps,
    vmec_force_norms_from_bcovar_dynamic,
    vmec_gcx2_from_tomnsps,
)
from vmec_jax.vmec_tomnsp import vmec_trig_tables
from vmec_jax.plotting import bmag_from_state_vmec_realspace, bmag_from_wout_physical, closed_theta_grid
from vmec_jax.wout import read_wout


@dataclass(frozen=True)
class WoutLike:
    nfp: int
    mpol: int
    ntor: int
    lasym: bool
    signgs: int
    phipf: np.ndarray
    phips: np.ndarray
    chipf: np.ndarray
    pres: np.ndarray


def _summary(a: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
    }


def _stage_metrics(run, state, trig, wout_like, wout_ref) -> dict[str, object]:
    static = run.static
    s = np.asarray(static.s)

    geom = eval_geom(state, static)
    sqrtg = np.asarray(geom.sqrtg)

    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        indata=None,
        constraint_tcon0=float(run.indata.get_float("TCON0", 0.0)),
        use_vmec_synthesis=True,
        trig=trig,
    )
    frzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
        wout=wout_like,
        trig=trig,
        apply_lforbal=False,
    )
    frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
    frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)

    gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
        frzl=frzl,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        apply_m1_constraints=False,
        include_edge=False,
        apply_scalxc=False,
        s=s,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=int(run.signgs))
    fsqr = float(np.asarray(norms.r1 * norms.fnorm * gcr2))
    fsqz = float(np.asarray(norms.r1 * norms.fnorm * gcz2))
    fsql = float(np.asarray(norms.fnormL * gcl2))

    b_vmec = np.asarray(
        bmag_from_state_vmec_realspace(
            state,
            static,
            run.indata,
            s_index=int(static.cfg.ns) - 1,
            signgs=int(wout_ref.signgs),
            phipf=np.asarray(wout_ref.phipf),
            chipf=np.asarray(wout_ref.chipf),
            lamscale=float(np.asarray(run.flux.lamscale)),
            sqrtg_floor=None,
        )
    )

    return {
        "sqrtg": _summary(sqrtg),
        "sqrtg_neg_fraction": float(np.mean(sqrtg < 0.0)),
        "frcc_norm": float(np.linalg.norm(np.asarray(frzl.frcc))),
        "fzsc_norm": float(np.linalg.norm(np.asarray(frzl.fzsc))),
        "flsc_norm": float(np.linalg.norm(np.asarray(frzl.flsc))),
        "fsqr": fsqr,
        "fsqz": fsqz,
        "fsql": fsql,
        "B_vmec_grid": _summary(b_vmec),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--input", type=Path, default=root / "examples/data/input.n3are_R7.75B5.7_lowres")
    p.add_argument("--wout", type=Path, default=root / "examples/data/wout_n3are_R7.75B5.7_lowres.nc")
    p.add_argument("--max-iter", type=int, default=10)
    p.add_argument("--step-size", type=float, default=1e-10)
    p.add_argument("--out", type=Path, default=root / "examples/outputs/n3are_vmecpp_stage_diagnostics.json")
    args = p.parse_args()

    run0 = run_fixed_boundary(args.input, solver="vmecpp_iter", use_initial_guess=True, verbose=False)
    run1 = run_fixed_boundary(
        args.input,
        solver="vmecpp_iter",
        max_iter=int(args.max_iter),
        step_size=float(args.step_size),
        verbose=False,
    )

    wout_ref = read_wout(args.wout)

    s = np.asarray(run0.static.s)
    flux = flux_profiles_from_indata(run0.indata, s, signgs=int(run0.signgs))
    chipf_wout = half_mesh_avg_from_full_mesh(np.asarray(flux.chipf))
    phips = np.asarray(flux.phips).copy()
    if phips.shape[0] >= 1:
        phips[0] = 0.0
    prof = eval_profiles(run0.indata, s)

    wout_like = WoutLike(
        nfp=int(run0.static.cfg.nfp),
        mpol=int(run0.static.cfg.mpol),
        ntor=int(run0.static.cfg.ntor),
        lasym=bool(run0.static.cfg.lasym),
        signgs=int(run0.signgs),
        phipf=np.asarray(flux.phipf),
        phips=phips,
        chipf=np.asarray(chipf_wout),
        pres=np.asarray(prof.get("pressure", np.zeros_like(s))),
    )

    trig = vmec_trig_tables(
        ntheta=int(run0.static.cfg.ntheta),
        nzeta=int(run0.static.cfg.nzeta),
        nfp=int(run0.static.cfg.nfp),
        mmax=int(run0.static.cfg.mpol) - 1,
        nmax=int(run0.static.cfg.ntor),
        lasym=bool(run0.static.cfg.lasym),
    )

    theta = closed_theta_grid(30)
    phi = np.linspace(0.0, 2.0 * np.pi, num=65, endpoint=True)
    b_ref = np.asarray(bmag_from_wout_physical(wout_ref, theta=theta, phi=phi, s_index=int(wout_ref.ns) - 1))

    report = {
        "initial": _stage_metrics(run0, run0.state, trig, wout_like, wout_ref),
        "after_solver": _stage_metrics(run1, run1.state, trig, wout_like, wout_ref),
        "solver": {
            "max_iter": int(args.max_iter),
            "step_size": float(args.step_size),
            "n_iter": int(run1.result.n_iter if run1.result is not None else 0),
            "w_final": float(run1.result.w_history[-1]) if run1.result is not None else float("nan"),
        },
        "reference": {
            "B_vmec2000_lcfs": _summary(b_ref)
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"[vmec_jax] wrote {args.out}")


if __name__ == "__main__":
    main()
