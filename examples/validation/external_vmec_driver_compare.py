"""Run VMEC2000 or VMEC++ via Python and compare against bundled references.

This script is meant for local validation when external codes are installed.
It can:
- run VMEC2000 (python wrapper) or VMEC++ (vmecpp) for a chosen input,
- save the resulting wout file,
- compare key wout fields to the bundled VMEC2000 reference,
- optionally compute b-field parity metrics using vmec_jax kernels.

Requires (depending on backend):
- vmec2000: vmec python extension + mpi4py + netCDF4
- vmecpp: vmecpp + netCDF4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.config import load_config
from vmec_jax.field import bsub_from_bsup, bsup_from_geom, chips_from_chipf, lamscale_from_phips
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


def _load_netcdf(path: Path):
    try:
        import netCDF4 as nc  # type: ignore

        return nc.Dataset(path, "r")
    except Exception:
        try:
            from scipy.io import netcdf_file  # type: ignore

            return netcdf_file(path, mmap=False)
        except Exception as exc:
            raise RuntimeError("Need netCDF4 or scipy to read wout files") from exc


def _hi_res_cfg(cfg, *, mpol: int, ntor: int):
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 16)
    ntheta = 2 * (ntheta // 2)
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 16)
    if nzeta <= 0:
        nzeta = 1
    return replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))


def _half_mesh_coeffs(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


def _bfield_parity(input_path: Path, wout_path: Path) -> dict[str, float]:
    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    cfg_hi = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg_hi)

    st = state_from_wout(wout)
    st_half = replace(
        st,
        Rcos=_half_mesh_coeffs(np.asarray(st.Rcos)),
        Rsin=_half_mesh_coeffs(np.asarray(st.Rsin)),
        Zcos=_half_mesh_coeffs(np.asarray(st.Zcos)),
        Zsin=_half_mesh_coeffs(np.asarray(st.Zsin)),
        Lcos=np.asarray(st.Lcos),
        Lsin=np.asarray(st.Lsin),
    )
    g = eval_geom(st_half, static)

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis_nyq = build_helical_basis(modes_nyq, grid)

    # bsup parity (computed from geometry + lambda)
    lamscale = lamscale_from_phips(wout.phips, static.s)
    chips = chips_from_chipf(wout.chipf)
    bsupu_calc, bsupv_calc = bsup_from_geom(
        g,
        phipf=wout.phipf,
        chipf=chips,
        nfp=wout.nfp,
        signgs=wout.signgs,
        lamscale=lamscale,
    )
    bsupu_calc = np.asarray(bsupu_calc)
    bsupv_calc = np.asarray(bsupv_calc)

    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))

    js0 = max(1, int(0.25 * (wout.ns - 1)))
    bsupu_err = _rel_rms(bsupu_calc[js0:], bsupu_ref[js0:])
    bsupv_err = _rel_rms(bsupv_calc[js0:], bsupv_ref[js0:])

    # bsub parity (computed from metric + bsup)
    bsubu_calc, bsubv_calc = bsub_from_bsup(g, bsupu_ref, bsupv_ref)
    bsubu_calc = np.asarray(bsubu_calc)
    bsubv_calc = np.asarray(bsubv_calc)
    bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
    bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    bsubu_err = _rel_rms(bsubu_calc[1:], bsubu_ref[1:])
    bsubv_err = _rel_rms(bsubv_calc[1:], bsubv_ref[1:])

    # |B| parity
    gtt = np.asarray(g.g_tt)
    gtp = np.asarray(g.g_tp)
    gpp = np.asarray(g.g_pp)
    B2 = gtt * bsupu_ref**2 + 2.0 * gtp * bsupu_ref * bsupv_ref + gpp * bsupv_ref**2
    Bmag_calc = np.sqrt(np.maximum(B2, 0.0))
    Bmag_ref = np.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))
    Bmag_err = _rel_rms(Bmag_calc[1:], Bmag_ref[1:])

    return {
        "bsupu_rel_rms_outer": float(bsupu_err),
        "bsupv_rel_rms_outer": float(bsupv_err),
        "bsubu_rel_rms": float(bsubu_err),
        "bsubv_rel_rms": float(bsubv_err),
        "bmag_rel_rms": float(Bmag_err),
    }


def _run_vmec2000(input_path: Path, tmp_dir: Path) -> Path:
    try:
        import vmec  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"vmec2000 python extension import failed: {exc}") from exc

    try:
        from mpi4py import MPI  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"mpi4py import failed: {exc}") from exc

    # Flags used by VMEC2000 python wrapper `runvmec`:
    restart_flag = 1
    readin_flag = 2
    timestep_flag = 4
    output_flag = 8
    reset_jacdt_flag = 32

    ictrl = np.zeros(5, dtype=np.int32)

    cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        ictrl[:] = 0
        ictrl[0] = restart_flag + readin_flag
        vmec.runvmec(ictrl, str(input_path), False, MPI.COMM_WORLD.py2f(), "")
        if int(ictrl[1]) != 0:
            raise RuntimeError(f"VMEC2000 readin failed (ictrl[1]={int(ictrl[1])})")
        vmec.cleanup(False)

        vmec.vmec_input.raxis_cc = 0
        vmec.vmec_input.raxis_cs = 0
        vmec.vmec_input.zaxis_cc = 0
        vmec.vmec_input.zaxis_cs = 0
        vmec.reinit()

        ictrl[:] = 0
        ictrl[0] = restart_flag + reset_jacdt_flag + timestep_flag + output_flag
        vmec.runvmec(ictrl, str(input_path), False, MPI.COMM_WORLD.py2f(), "")
        if int(ictrl[1]) != 11:
            raise RuntimeError(f"VMEC2000 run did not converge (ictrl[1]={int(ictrl[1])})")

        MPI.COMM_WORLD.Barrier()
        outs = list(Path(tmp_dir).glob("wout_*.nc"))
        if not outs:
            raise RuntimeError("VMEC2000 did not produce a wout_*.nc file")
        return outs[0]
    finally:
        vmec.cleanup(True)
        os.chdir(cwd)


def _run_vmecpp(input_path: Path, tmp_dir: Path, max_threads: int | None, verbose: bool) -> Path:
    try:
        import vmecpp  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"vmecpp import failed: {exc}") from exc

    vmec_input = vmecpp.VmecInput.from_file(input_path)
    output = vmecpp.run(vmec_input, max_threads=max_threads, verbose=verbose)

    wout_path = tmp_dir / f"wout_{input_path.name.replace('input.', '')}_vmecpp.nc"
    output.wout.save(str(wout_path))
    return wout_path


def _compare_wout(wout_path: Path, ref_path: Path) -> dict[str, float]:
    diffs: dict[str, float] = {}
    f1 = _load_netcdf(wout_path)
    f2 = _load_netcdf(ref_path)
    try:
        for field in ("iotaf", "rmnc", "zmns", "lmns", "bmnc"):
            if field in f1.variables and field in f2.variables:
                x1 = np.asarray(f1.variables[field][()])
                x2 = np.asarray(f2.variables[field][()])
                diffs[field] = float(np.max(np.abs(x2 - x1)))
    finally:
        f1.close()
        f2.close()
    return diffs


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["vmec2000", "vmecpp"], default="vmecpp")
    p.add_argument("--input", type=str, default="", help="Path to input.* file")
    p.add_argument(
        "--case",
        type=str,
        default="circular_tokamak",
        help="Case name under examples/data (input.<case>)",
    )
    p.add_argument("--reference", type=str, default="", help="Reference wout_*.nc to compare")
    p.add_argument("--max-threads", type=int, default=None, help="Max threads for VMEC++")
    p.add_argument("--verbose", action="store_true", help="Enable VMEC++ verbose output")
    p.add_argument("--no-bfield-parity", action="store_true", help="Skip vmec_jax b-field parity checks")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "examples/outputs"),
        help="Directory for JSON outputs",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input) if args.input else REPO_ROOT / "examples/data" / f"input.{args.case}"
    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    ref_path = Path(args.reference) if args.reference else REPO_ROOT / "examples/data" / f"wout_{args.case}_reference.nc"
    if not ref_path.exists():
        ref_path = None  # type: ignore[assignment]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if args.backend == "vmec2000":
            wout_path = _run_vmec2000(input_path, tmp_dir)
        else:
            wout_path = _run_vmecpp(input_path, tmp_dir, args.max_threads, args.verbose)

        summary: dict[str, object] = {
            "backend": args.backend,
            "input": str(input_path),
            "wout": str(wout_path),
        }

        if ref_path is not None and ref_path.exists():
            summary["reference"] = str(ref_path)
            summary["reference_diffs"] = _compare_wout(wout_path, ref_path)

        if not args.no_bfield_parity:
            summary["bfield_parity"] = _bfield_parity(input_path, wout_path)

        out_json = outdir / f"external_{args.backend}_{input_path.name.replace('input.', '')}.json"
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
