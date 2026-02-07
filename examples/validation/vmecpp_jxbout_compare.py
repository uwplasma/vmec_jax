"""Compare vmec_jax bcovar fields against VMEC++ jxbout arrays.

This script is a stage-level diagnostic on VMEC's internal grid
(``ntheta_eff x nzeta``), avoiding reliance on ``wout`` harmonic reconstruction.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
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


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den > 0.0 else float("inf")


def _fit_scale(a: np.ndarray, b: np.ndarray) -> float:
    aa = float(np.vdot(a, a).real)
    if aa == 0.0:
        return 1.0
    return float(np.vdot(a, b).real / aa)


def _stats(ours: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    ours = np.asarray(ours).reshape(-1)
    ref = np.asarray(ref).reshape(-1)
    rel_raw = _rel_rms(ours, ref)
    scale = _fit_scale(ours, ref)
    rel_scaled = _rel_rms(scale * ours, ref)
    with np.errstate(invalid="ignore"):
        corr = float(np.corrcoef(ours, ref)[0, 1])
    return {
        "rel_rms": rel_raw,
        "best_scale": scale,
        "rel_rms_scaled": rel_scaled,
        "corr": corr,
        "ours_rms": float(np.sqrt(np.mean(ours**2))),
        "ref_rms": float(np.sqrt(np.mean(ref**2))),
    }


def _reshape_jxbout(arr: np.ndarray, *, ns_like: int, nzeta: int, ntheta_eff: int) -> np.ndarray:
    """Reshape VMEC++ jxbout flat storage to (ns_like, ntheta_eff, nzeta).

    VMEC++ stores jxbout arrays flattened with radial-major order and internal
    angular layout `(ns, nzeta, ntheta_eff)`. Most vmec_jax kernels use
    `(ns, ntheta_eff, nzeta)`.
    """
    a = np.asarray(arr)
    return a.reshape(ns_like, nzeta, ntheta_eff).transpose(0, 2, 1)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=root / "examples/data/input.n3are_R7.75B5.7_lowres")
    p.add_argument("--out", type=Path, default=root / "examples/outputs/vmecpp_jxbout_compare_n3are.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import vmecpp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"vmecpp import failed: {exc}")

    cfg, _ = load_config(str(args.input))
    out = vmecpp.run(vmecpp.VmecInput.from_file(args.input), verbose=False)

    with tempfile.TemporaryDirectory() as td:
        wpath = Path(td) / f"wout_{args.input.name.replace('input.', '')}_vmecpp.nc"
        out.wout.save(str(wpath))
        wout = read_wout(wpath)

    grid = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
    )
    static = build_static(cfg, grid=grid)
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )
    state = state_from_wout(wout)

    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        use_wout_bsup=False,
        use_vmec_synthesis=True,
        trig=trig,
    )

    ns = int(wout.ns)
    nzeta = int(cfg.nzeta)
    raw_sqrtg3 = np.asarray(out.jxbout.sqrtg3)
    if raw_sqrtg3.ndim == 2:
        ns_jx, nZnT = int(raw_sqrtg3.shape[0]), int(raw_sqrtg3.shape[1])
        if ns_jx != ns:
            raise RuntimeError(f"unexpected jxbout sqrtg3 radial shape: {raw_sqrtg3.shape}, expected ns={ns}")
    elif raw_sqrtg3.ndim == 1:
        nZnT = int(raw_sqrtg3.shape[0] // ns)
    else:
        raise RuntimeError(f"unsupported jxbout sqrtg3 shape: {raw_sqrtg3.shape}")

    if nZnT % nzeta != 0:
        raise RuntimeError(f"cannot infer ntheta_eff: nZnT={nZnT} nzeta={nzeta}")
    ntheta_eff = int(nZnT // nzeta)
    if ntheta_eff <= 0:
        raise RuntimeError("failed to infer VMEC++ internal ntheta_eff from jxbout")

    ours_sqrtg = np.asarray(bc.jac.sqrtg)[:, :ntheta_eff, :]
    ours_bsupu = np.asarray(bc.bsupu)[:, :ntheta_eff, :]
    ours_bsupv = np.asarray(bc.bsupv)[:, :ntheta_eff, :]
    ours_bsubu = np.asarray(bc.bsubu)[1:, :ntheta_eff, :]
    ours_bsubv = np.asarray(bc.bsubv)[1:, :ntheta_eff, :]

    ref_sqrtg = _reshape_jxbout(raw_sqrtg3, ns_like=ns, nzeta=nzeta, ntheta_eff=ntheta_eff)
    ref_bsupu = _reshape_jxbout(np.asarray(out.jxbout.bsupu3), ns_like=ns, nzeta=nzeta, ntheta_eff=ntheta_eff)
    ref_bsupv = _reshape_jxbout(np.asarray(out.jxbout.bsupv3), ns_like=ns, nzeta=nzeta, ntheta_eff=ntheta_eff)
    ref_bsubu = _reshape_jxbout(np.asarray(out.jxbout.bsubu3), ns_like=ns - 1, nzeta=nzeta, ntheta_eff=ntheta_eff)
    ref_bsubv = _reshape_jxbout(np.asarray(out.jxbout.bsubv3), ns_like=ns - 1, nzeta=nzeta, ntheta_eff=ntheta_eff)

    # Baseline: wout Nyquist evaluation on the same internal grid.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    basis_nyq = build_helical_basis(
        modes_nyq,
        AngleGrid(theta=np.asarray(grid.theta[:ntheta_eff]), zeta=np.asarray(grid.zeta), nfp=int(wout.nfp)),
    )
    wout_sqrtg = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))
    wout_bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    wout_bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
    wout_bsubu = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))[1:]
    wout_bsubv = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))[1:]

    report = {
        "input": str(args.input),
        "vmecpp": {
            "niter": int(getattr(out.wout, "niter", -1)),
            "ns": ns,
            "nZnT": nZnT,
            "ntheta_cfg": int(cfg.ntheta),
            "nzeta_cfg": int(cfg.nzeta),
            "ntheta_eff_jxbout": ntheta_eff,
        },
        "field_stats": {
            "sqrtg3": _stats(ours_sqrtg, ref_sqrtg),
            "bsupu3": _stats(ours_bsupu, ref_bsupu),
            "bsupv3": _stats(ours_bsupv, ref_bsupv),
            "bsubu3": _stats(ours_bsubu, ref_bsubu),
            "bsubv3": _stats(ours_bsubv, ref_bsubv),
        },
        "wout_eval_vs_jxbout": {
            "sqrtg3": _stats(wout_sqrtg, ref_sqrtg),
            "bsupu3": _stats(wout_bsupu, ref_bsupu),
            "bsupv3": _stats(wout_bsupv, ref_bsupv),
            "bsubu3": _stats(wout_bsubu, ref_bsubu),
            "bsubv3": _stats(wout_bsubv, ref_bsubv),
        },
        "vmec_jax_vs_wout_eval": {
            "sqrtg3": _stats(ours_sqrtg, wout_sqrtg),
            "bsupu3": _stats(ours_bsupu, wout_bsupu),
            "bsupv3": _stats(ours_bsupv, wout_bsupv),
            "bsubu3": _stats(ours_bsubu, wout_bsubu),
            "bsubv3": _stats(ours_bsubv, wout_bsubv),
        },
        "notes": [
            "bsup*/bsub* comparisons are on the VMEC internal angular grid.",
            "jxbout is flattened as (ns, nzeta, ntheta_eff) and reshaped to (ns, ntheta_eff, nzeta) here.",
            "bsub* arrays in jxbout are half-mesh (ns-1), so axis is excluded on vmec_jax side.",
            "wout_eval_vs_jxbout quantifies Fourier-output-vs-internal-array differences independent of vmec_jax kernels.",
            "Use best_scale and rel_rms_scaled to separate convention/normalization offsets from shape mismatch.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[vmec_jax] wrote {args.out}")


if __name__ == "__main__":
    main()
