from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.config import load_config
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.vmec_tomnsp import vmec_angle_grid
from vmec_jax.wout import read_wout, state_from_wout


CASES = {
    "li383_low_res": ("examples/data/input.li383_low_res", "examples/data/wout_li383_low_res_reference.nc"),
    "n3are_R7.75B5.7_lowres": ("examples/data/input.n3are_R7.75B5.7_lowres", "examples/data/wout_n3are_R7.75B5.7_lowres.nc"),
}


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


def _case_paths(case: str):
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}. Choices: {', '.join(CASES)}")
    input_rel, wout_rel = CASES[case]
    return REPO_ROOT / input_rel, REPO_ROOT / wout_rel


def _summary_for_case(case: str) -> dict[str, float]:
    input_path, wout_path = _case_paths(case)
    cfg, _ = load_config(str(input_path))
    wout = read_wout(wout_path)

    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg, grid=grid)
    st = state_from_wout(wout)

    bc_eval = vmec_bcovar_half_mesh_from_wout(state=st, static=static, wout=wout, use_wout_bsup=False, use_vmec_synthesis=False)
    bc_vmec = vmec_bcovar_half_mesh_from_wout(state=st, static=static, wout=wout, use_wout_bsup=False, use_vmec_synthesis=True)

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    basis_nyq = build_helical_basis(modes_nyq, AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp))
    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    sl = slice(1, None)

    # Contravariant parity.
    bsupu_eval_err = _rel_rms(np.asarray(bc_eval.bsupu)[sl], bsupu_ref[sl])
    bsupv_eval_err = _rel_rms(np.asarray(bc_eval.bsupv)[sl], bsupv_ref[sl])
    bsupu_vmec_err = _rel_rms(np.asarray(bc_vmec.bsupu)[sl], bsupu_ref[sl])
    bsupv_vmec_err = _rel_rms(np.asarray(bc_vmec.bsupv)[sl], bsupv_ref[sl])

    # Covariant parity (full bcovar).
    bsubu_eval_err = _rel_rms(np.asarray(bc_eval.bsubu)[sl], bsubu_ref[sl])
    bsubv_eval_err = _rel_rms(np.asarray(bc_eval.bsubv)[sl], bsubv_ref[sl])
    bsubu_vmec_err = _rel_rms(np.asarray(bc_vmec.bsubu)[sl], bsubu_ref[sl])
    bsubv_vmec_err = _rel_rms(np.asarray(bc_vmec.bsubv)[sl], bsubv_ref[sl])

    # Metric-only bsub from wout bsup (isolates metric vs bsup).
    bsubu_metric_eval = np.asarray(bc_eval.guu) * bsupu_ref + np.asarray(bc_eval.guv) * bsupv_ref
    bsubv_metric_eval = np.asarray(bc_eval.guv) * bsupu_ref + np.asarray(bc_eval.gvv) * bsupv_ref
    bsubu_metric_vmec = np.asarray(bc_vmec.guu) * bsupu_ref + np.asarray(bc_vmec.guv) * bsupv_ref
    bsubv_metric_vmec = np.asarray(bc_vmec.guv) * bsupu_ref + np.asarray(bc_vmec.gvv) * bsupv_ref

    bsubu_metric_eval_err = _rel_rms(bsubu_metric_eval[sl], bsubu_ref[sl])
    bsubv_metric_eval_err = _rel_rms(bsubv_metric_eval[sl], bsubv_ref[sl])
    bsubu_metric_vmec_err = _rel_rms(bsubu_metric_vmec[sl], bsubu_ref[sl])
    bsubv_metric_vmec_err = _rel_rms(bsubv_metric_vmec[sl], bsubv_ref[sl])

    return {
        "bsupu_eval_rel_rms": bsupu_eval_err,
        "bsupv_eval_rel_rms": bsupv_eval_err,
        "bsupu_vmec_rel_rms": bsupu_vmec_err,
        "bsupv_vmec_rel_rms": bsupv_vmec_err,
        "bsubu_eval_rel_rms": bsubu_eval_err,
        "bsubv_eval_rel_rms": bsubv_eval_err,
        "bsubu_vmec_rel_rms": bsubu_vmec_err,
        "bsubv_vmec_rel_rms": bsubv_vmec_err,
        "bsubu_metric_eval_rel_rms": bsubu_metric_eval_err,
        "bsubv_metric_eval_rel_rms": bsubv_metric_eval_err,
        "bsubu_metric_vmec_rel_rms": bsubu_metric_vmec_err,
        "bsubv_metric_vmec_rel_rms": bsubv_metric_vmec_err,
    }


def main() -> None:
    enable_x64()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES.keys(), default="li383_low_res")
    args = parser.parse_args()

    summary = _summary_for_case(args.case)
    print(f"== {args.case} ==")
    for key, val in summary.items():
        print(f"{key:28s} {val:.3e}")


if __name__ == "__main__":
    main()
