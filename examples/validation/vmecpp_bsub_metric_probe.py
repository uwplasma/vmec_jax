"""Probe the remaining bsub parity gap against a fresh VMEC++ run.

This diagnostic isolates whether the dominant bsub mismatch comes from
contravariant B (bsup) or from half-mesh metric components (especially guu).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from vmec_jax.config import load_config
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.vmec_tomnsp import vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _hi_res_cfg(cfg, *, mpol: int, ntor: int):
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 16)
    ntheta = 2 * (ntheta // 2)
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 16)
    if nzeta <= 0:
        nzeta = 1
    return cfg.__class__(
        mpol=int(mpol),
        ntor=int(ntor),
        ns=int(cfg.ns),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
        lconm1=bool(cfg.lconm1),
        ntheta=int(ntheta),
        nzeta=int(nzeta),
    )


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    den = float(np.sqrt(np.mean(b**2)))
    if den == 0.0:
        return float("inf")
    return float(np.sqrt(np.mean((a - b) ** 2)) / den)


def _summary_stats(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a)
    finite = np.isfinite(a)
    if not np.any(finite):
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "p90": float("nan")}
    x = a[finite]
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90.0)),
    }


def _parse_args():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=root / "examples/data/input.n3are_R7.75B5.7_lowres")
    p.add_argument("--out", type=Path, default=root / "examples/outputs/vmecpp_bsub_metric_probe_n3are.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import vmecpp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"vmecpp import failed: {exc}")

    cfg, _ = load_config(str(args.input))

    out_vmecpp = vmecpp.run(vmecpp.VmecInput.from_file(args.input), verbose=False)
    with tempfile.TemporaryDirectory() as td:
        wpath = Path(td) / f"wout_{args.input.name.replace('input.', '')}_vmecpp.nc"
        out_vmecpp.wout.save(str(wpath))
        wout = read_wout(wpath)

    cfg = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(cfg.nfp), lasym=bool(cfg.lasym))
    static = build_static(cfg, grid=grid)
    state = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
        dtype=np.asarray(state.Rcos).dtype,
    )

    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        use_wout_bsup=False,
        use_vmec_synthesis=True,
        trig=trig,
    )

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    basis_nyq = build_helical_basis(modes_nyq, AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp))

    ref_bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    ref_bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    ref_bsubu = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    ref_bsubv = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    sl = slice(1, None)
    bsupu = np.asarray(bc.bsupu)
    bsupv = np.asarray(bc.bsupv)
    bsubu = np.asarray(bc.bsubu)
    bsubv = np.asarray(bc.bsubv)
    guu = np.asarray(bc.guu)
    guv = np.asarray(bc.guv)
    gvv = np.asarray(bc.gvv)

    # Decompose bsubu,bsubv into metric contributions using vmec_jax fields.
    bsubu_uu = guu * bsupu
    bsubu_uv = guv * bsupv
    bsubv_uv = guv * bsupu
    bsubv_vv = gvv * bsupv

    # Infer metric mismatch against reference bsub, holding guv fixed.
    eps = 1e-10
    denom_u = np.where(np.abs(ref_bsupu) > eps, ref_bsupu, np.nan)
    denom_v = np.where(np.abs(ref_bsupv) > eps, ref_bsupv, np.nan)
    guu_implied = (ref_bsubu - guv * ref_bsupv) / denom_u
    gvv_implied = (ref_bsubv - guv * ref_bsupu) / denom_v

    rel_d_guu = np.abs((guu_implied - guu) / np.where(np.abs(guu) > eps, guu, np.nan))
    rel_d_gvv = np.abs((gvv_implied - gvv) / np.where(np.abs(gvv) > eps, gvv, np.nan))

    report = {
        "input": str(args.input),
        "vmecpp": {
            "niter": int(getattr(out_vmecpp.wout, "niter", -1)),
            "wout_fsqr": float(wout.fsqr),
            "wout_fsqz": float(wout.fsqz),
            "wout_fsql": float(wout.fsql),
        },
        "field_rel_rms_outer": {
            "bsupu": _rel_rms(bsupu[sl], ref_bsupu[sl]),
            "bsupv": _rel_rms(bsupv[sl], ref_bsupv[sl]),
            "bsubu": _rel_rms(bsubu[sl], ref_bsubu[sl]),
            "bsubv": _rel_rms(bsubv[sl], ref_bsubv[sl]),
        },
        "bsub_decomposition_outer": {
            "bsubu_from_uu_rel_to_ref": _rel_rms(bsubu_uu[sl], ref_bsubu[sl]),
            "bsubu_from_uv_rel_to_ref": _rel_rms(bsubu_uv[sl], ref_bsubu[sl]),
            "bsubv_from_uv_rel_to_ref": _rel_rms(bsubv_uv[sl], ref_bsubv[sl]),
            "bsubv_from_vv_rel_to_ref": _rel_rms(bsubv_vv[sl], ref_bsubv[sl]),
        },
        "implied_metric_delta_outer": {
            "guu_abs_delta": _summary_stats((guu_implied - guu)[sl]),
            "gvv_abs_delta": _summary_stats((gvv_implied - gvv)[sl]),
            "guu_rel_delta": _summary_stats(rel_d_guu[sl]),
            "gvv_rel_delta": _summary_stats(rel_d_gvv[sl]),
        },
        "notes": [
            "This report keeps guv fixed and infers guu/gvv needed to match reference bsub.",
            "Large guu relative deltas with small gvv deltas indicate the remaining bsub gap is concentrated in the g_uu/R_u,Z_u path.",
            "All metrics above are evaluated on VMEC's internal grid and exclude the magnetic axis (js=1).",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[vmec_jax] wrote {args.out}")


if __name__ == "__main__":
    main()
