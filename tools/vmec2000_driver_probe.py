"""Probe VMEC2000 via its Python driver (vmec module).

This script mirrors the flow in vmec2000/python/tests/test_simple.py and
test_regression.py. It is intended to extract *inputs* and *outputs* from
VMEC2000 to support parity work in vmec_jax.

Notes
-----
- Requires the `vmec` Python extension from vmec2000 to be importable.
- Requires `mpi4py` to provide an MPI communicator for runvmec.
- Produces a small JSON summary and optional NPZ for quick comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VMEC2000_TESTS = Path("/Users/rogeriojorge/local/test/vmec2000/python/tests")


def _load_netcdf(path: Path):
    """Load a NetCDF file via netCDF4 (preferred) or scipy.io.netcdf_file."""
    try:
        import netCDF4 as nc  # type: ignore

        return nc.Dataset(path, "r")
    except Exception:
        try:
            from scipy.io import netcdf_file  # type: ignore

            return netcdf_file(path, mmap=False)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Need netCDF4 or scipy to read wout files") from exc


def _summarize_array(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a)
    if a.size == 0:
        return {"size": 0}
    return {
        "size": int(a.size),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Probe VMEC2000 Python driver")
    p.add_argument("--input", type=str, default="", help="Path to input.* file")
    p.add_argument(
        "--case",
        type=str,
        default="li383_low_res",
        help="Case name under vmec2000/python/tests (input.<case>)",
    )
    p.add_argument(
        "--reference",
        type=str,
        default="",
        help="Reference wout_*.nc to compare (optional)",
    )
    p.add_argument(
        "--read-input-only",
        action="store_true",
        help="Only read the input file (no time stepping / output).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "examples/outputs"),
        help="Directory for JSON/NPZ outputs",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input) if args.input else VMEC2000_TESTS / f"input.{args.case}"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    try:
        import vmec  # type: ignore
    except Exception as exc:
        print("Failed to import vmec. This requires the vmec2000 Python extension.")
        print(f"Import error: {exc}")
        return

    try:
        from mpi4py import MPI  # type: ignore

        fcomm = MPI.COMM_WORLD.py2f()
    except Exception as exc:
        print("Failed to import mpi4py; VMEC2000 driver requires an MPI communicator.")
        print(f"Import error: {exc}")
        return

    # Flags used by runvmec in vmec2000/python/tests.
    restart_flag = 1
    readin_flag = 2
    timestep_flag = 4
    output_flag = 8
    cleanup_flag = 16
    reset_jacdt_flag = 32

    ictrl = np.zeros(5, dtype=np.int32)
    verbose = True
    reset_file = ""

    if args.read_input_only:
        ictrl[:] = 0
        ictrl[0] = restart_flag + readin_flag
        vmec.runvmec(ictrl, str(input_path), verbose, fcomm, reset_file)
    else:
        ictrl[:] = 0
        ictrl[0] = restart_flag + readin_flag + timestep_flag + output_flag
        vmec.runvmec(ictrl, str(input_path), verbose, fcomm, reset_file)

    # Basic input diagnostics from vmec_input module.
    summary = {
        "input": str(input_path),
        "ictrl": ictrl.tolist(),
        "nfp": int(vmec.vmec_input.nfp),
        "mpol": int(vmec.vmec_input.mpol),
        "ntor": int(vmec.vmec_input.ntor),
    }

    # Probe a few arrays if present.
    for name in ("rbc", "zbs", "rbs", "zbc", "ai", "am", "ac"):
        if hasattr(vmec.vmec_input, name):
            arr = np.asarray(getattr(vmec.vmec_input, name))
            summary[f"{name}_summary"] = _summarize_array(arr)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # If we ran VMEC, it should have produced a wout_* file in CWD.
    wout_path = None
    if not args.read_input_only:
        wout_path = Path.cwd() / f"wout_{input_path.name.replace('input.', '')}.nc"
        if wout_path.exists():
            summary["wout"] = str(wout_path)
            f = _load_netcdf(wout_path)
            try:
                for field in ("fsqr", "fsqz", "fsql", "iotaf", "rmnc", "zmns", "lmns", "bmnc"):
                    if field in f.variables:
                        summary[f"wout_{field}"] = _summarize_array(f.variables[field][()])
            finally:
                f.close()

    # Optional comparison to reference wout.
    if args.reference:
        ref_path = Path(args.reference)
    else:
        ref_path = VMEC2000_TESTS / f"wout_{input_path.name.replace('input.', '')}_reference.nc"
    if ref_path.exists() and wout_path is not None and wout_path.exists():
        f1 = _load_netcdf(wout_path)
        f2 = _load_netcdf(ref_path)
        diffs = {}
        try:
            for field in ("iotaf", "rmnc", "zmns", "lmns", "bmnc"):
                if field in f1.variables and field in f2.variables:
                    x1 = np.asarray(f1.variables[field][()])
                    x2 = np.asarray(f2.variables[field][()])
                    diffs[field] = float(np.max(np.abs(x2 - x1)))
        finally:
            f1.close()
            f2.close()
        summary["reference"] = str(ref_path)
        summary["reference_diffs"] = diffs

    out_json = outdir / f"vmec2000_driver_probe_{input_path.name.replace('input.', '')}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")

    # Cleanup VMEC allocations. Use False when no timestep, True otherwise.
    vmec.cleanup(not args.read_input_only)


if __name__ == "__main__":
    main()
