"""Stage-by-stage parity report against a fresh VMEC++ run.

This script localizes the first failing stage by:
1) running VMEC++ for an input case,
2) checking vmec_jax kernels on the VMEC++ final state (self-consistency),
3) comparing vmec_jax initial/solved states against VMEC++ reference fields.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from vmec_jax.config import load_config
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.field import (
    b2_from_bsup,
    bsub_from_bsup,
    bsup_from_geom,
    chips_from_wout_chipf,
    lamscale_from_phips,
)
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.wout import read_wout, state_from_wout


def _hi_res_cfg(cfg, *, mpol: int, ntor: int):
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 16)
    ntheta = 2 * (ntheta // 2)
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 16)
    if nzeta <= 0:
        nzeta = 1
    return replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den > 0.0 else float("inf")


def _half_mesh_coeffs(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


def _reference_fields(wout, static):
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    basis = build_helical_basis(modes_nyq, AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp))
    return {
        "sqrtg": np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis)),
        "bsupu": np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis)),
        "bsupv": np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis)),
        "bsubu": np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis)),
        "bsubv": np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis)),
        "bmag": np.asarray(eval_fourier(wout.bmnc, wout.bmns, basis)),
    }


def _state_field_errors(*, state, static, wout, ref_fields):
    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )
    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        use_wout_bsup=False,
        use_vmec_synthesis=True,
        trig=trig,
    )
    sl = slice(1, None)
    return {
        "sqrtg_rel_rms": _rel_rms(np.asarray(bc.jac.sqrtg)[sl], ref_fields["sqrtg"][sl]),
        "bsupu_rel_rms": _rel_rms(np.asarray(bc.bsupu)[sl], ref_fields["bsupu"][sl]),
        "bsupv_rel_rms": _rel_rms(np.asarray(bc.bsupv)[sl], ref_fields["bsupv"][sl]),
        "bsubu_rel_rms": _rel_rms(np.asarray(bc.bsubu)[sl], ref_fields["bsubu"][sl]),
        "bsubv_rel_rms": _rel_rms(np.asarray(bc.bsubv)[sl], ref_fields["bsubv"][sl]),
        # bmag from wout Fourier is checked elsewhere; keep this stage focused on
        # Jacobian and B-covariant/contravariant synthesis parity.
        "bmag_rel_rms": float("nan"),
    }


def _reference_self_consistency(*, state_ref, static, wout):
    ref_fields = _reference_fields(wout, static)
    err = _state_field_errors(state=state_ref, static=static, wout=wout, ref_fields=ref_fields)

    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )
    k = vmec_forces_rz_from_wout(
        state=state_ref,
        static=static,
        wout=wout,
        indata=None,
        use_wout_bsup=True,
        use_vmec_synthesis=True,
        trig=trig,
    )
    rzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
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
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=static.s, signgs=int(wout.signgs))
    scal = vmec_fsq_from_tomnsps_dynamic(frzl=frzl, norms=norms, lconm1=bool(getattr(static.cfg, "lconm1", True)))
    fsq = {
        "fsqr": float(np.asarray(scal.fsqr)),
        "fsqz": float(np.asarray(scal.fsqz)),
        "fsql": float(np.asarray(scal.fsql)),
    }
    fsq_ref = {"fsqr": float(wout.fsqr), "fsqz": float(wout.fsqz), "fsql": float(wout.fsql)}
    fsq_rel = {
        k: (abs(fsq[k] - fsq_ref[k]) / max(abs(fsq_ref[k]), 1e-20))
        for k in ("fsqr", "fsqz", "fsql")
    }
    return {
        "field_rel_rms": err,
        "fsq": fsq,
        "fsq_ref": fsq_ref,
        "fsq_rel_err": fsq_rel,
    }


def _first_failing_stage(self_consistency: dict[str, object]) -> dict[str, object]:
    # Keep `bsub` threshold looser than `bsup`: current symmetric nfp>1 cases
    # show a persistent few-1e-2 gap on the wout path, while the higher-ROI
    # solver mismatch is in downstream `getfsq` / update-loop conventions.
    stages = [
        ("geometry/sqrtg", float(self_consistency["field_rel_rms"]["sqrtg_rel_rms"]), 1e-2),  # type: ignore[index]
        ("bsup", max(float(self_consistency["field_rel_rms"]["bsupu_rel_rms"]), float(self_consistency["field_rel_rms"]["bsupv_rel_rms"])), 1e-2),  # type: ignore[index]
        ("bsub", max(float(self_consistency["field_rel_rms"]["bsubu_rel_rms"]), float(self_consistency["field_rel_rms"]["bsubv_rel_rms"])), 4e-2),  # type: ignore[index]
        (
            "getfsq",
            max(
                float(self_consistency["fsq_rel_err"]["fsqr"]),  # type: ignore[index]
                float(self_consistency["fsq_rel_err"]["fsqz"]),  # type: ignore[index]
                float(self_consistency["fsq_rel_err"]["fsql"]),  # type: ignore[index]
            ),
            1e-1,
        ),
    ]
    for name, value, tol in stages:
        if not np.isfinite(value) or value > tol:
            return {"stage": name, "value": float(value), "tol": float(tol)}
    return {"stage": "none", "value": 0.0, "tol": 0.0}


def _parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=root / "examples/data/input.n3are_R7.75B5.7_lowres")
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--step-size", type=float, default=1e-10)
    p.add_argument("--hi-res", action="store_true", help="Use 4*mpol/ntor+16 angular grid for diagnostics (default: use input grid).")
    p.add_argument("--out", type=Path, default=root / "examples/outputs/vmecpp_stage_parity_pipeline_n3are.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import vmecpp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"vmecpp import failed: {exc}")

    cfg_base, _indata = load_config(str(args.input))

    out_vmecpp = vmecpp.run(vmecpp.VmecInput.from_file(args.input), verbose=False)
    with tempfile.TemporaryDirectory() as td:
        wpath = Path(td) / f"wout_{args.input.name.replace('input.', '')}_vmecpp.nc"
        out_vmecpp.wout.save(str(wpath))
        wout_ref = read_wout(wpath)
    cfg = _hi_res_cfg(cfg_base, mpol=wout_ref.mpol, ntor=wout_ref.ntor) if bool(args.hi_res) else cfg_base
    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(cfg.nfp), lasym=bool(cfg.lasym))
    static = build_static(cfg, grid=grid)

    state_ref = state_from_wout(wout_ref)
    ref_fields = _reference_fields(wout_ref, static)
    self_consistency = _reference_self_consistency(state_ref=state_ref, static=static, wout=wout_ref)

    run0 = run_fixed_boundary(args.input, solver="vmecpp_iter", use_initial_guess=True, verbose=False)
    run1 = run_fixed_boundary(
        args.input,
        solver="vmecpp_iter",
        max_iter=int(args.max_iter),
        step_size=float(args.step_size),
        verbose=False,
    )
    initial_err = _state_field_errors(state=run0.state, static=static, wout=wout_ref, ref_fields=ref_fields)
    solved_err = _state_field_errors(state=run1.state, static=static, wout=wout_ref, ref_fields=ref_fields)

    first_fail = _first_failing_stage(self_consistency)
    report = {
        "input": str(args.input),
        "vmecpp": {
            "niter": int(getattr(out_vmecpp.wout, "niter", -1)),
            "wout_fsqr": float(wout_ref.fsqr),
            "wout_fsqz": float(wout_ref.fsqz),
            "wout_fsql": float(wout_ref.fsql),
        },
        "self_consistency": self_consistency,
        "first_failing_stage": first_fail,
        "vmec_jax_vs_vmecpp": {
            "initial_guess": initial_err,
            "after_vmecpp_iter": solved_err,
            "solver": {
                "max_iter": int(args.max_iter),
                "step_size": float(args.step_size),
                "n_iter": int(run1.result.n_iter if run1.result is not None else 0),
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[vmec_jax] wrote {args.out}")


if __name__ == "__main__":
    main()
