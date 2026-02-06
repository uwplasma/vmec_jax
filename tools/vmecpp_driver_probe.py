"""Probe VMEC++ via its Python API and optionally compare to VMEC2000 wout files.

This script is intended for parity investigations in vmec_jax. It mirrors the
vmecpp Python examples and captures a compact JSON summary of inputs/outputs.
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


def _model_dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"repr": repr(obj)}


def _parse_args():
    p = argparse.ArgumentParser(description="Probe VMEC++ Python API")
    p.add_argument("--input", type=str, default="", help="Path to input file (JSON or classic INDATA)")
    p.add_argument(
        "--case",
        type=str,
        default="li383_low_res",
        help="Case name under vmec2000/python/tests (input.<case>)",
    )
    p.add_argument("--max-threads", type=int, default=None, help="Max threads for VMEC++")
    p.add_argument("--verbose", action="store_true", help="Enable VMEC++ verbose output")
    p.add_argument(
        "--save-wout",
        type=str,
        default="",
        help="Path to save wout_*.nc (optional). Defaults to wout_<case>_vmecpp.nc in output dir.",
    )
    p.add_argument(
        "--compare-to",
        type=str,
        default="",
        help="Reference wout_*.nc to compare (optional). Defaults to VMEC2000 reference.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "examples/outputs"),
        help="Directory for JSON/NPZ outputs",
    )
    p.add_argument("--inspect", action="store_true", help="Dump basic vmecpp module inspection info")
    p.add_argument("--inspect-only", action="store_true", help="Only write inspection info; skip VMEC++ run")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input) if args.input else VMEC2000_TESTS / f"input.{args.case}"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    try:
        import vmecpp  # type: ignore
    except Exception as exc:
        print("Failed to import vmecpp.")
        print(f"Import error: {exc}")
        return

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        info = {
            "vmecpp_module": repr(vmecpp),
            "vmecpp_dir": sorted([x for x in dir(vmecpp) if not x.startswith("_")])[:80],
            "VmecInput_doc": getattr(vmecpp.VmecInput, "__doc__", "") or "",
            "run_doc": getattr(vmecpp.run, "__doc__", "") or "",
        }
        inspect_path = outdir / f"vmecpp_inspect_{input_path.name.replace('input.', '')}.json"
        inspect_path.write_text(json.dumps(info, indent=2))
        print(f"Wrote {inspect_path}")

    if args.inspect_only:
        return

    vmec_input = vmecpp.VmecInput.from_file(input_path)
    output = vmecpp.run(vmec_input, max_threads=args.max_threads, verbose=args.verbose)

    summary = {
        "input": str(input_path),
        "vmec_input": _model_dump(vmec_input),
        "output_class": repr(output.__class__),
    }

    # Grab core wout scalars if present.
    if hasattr(output, "wout"):
        wout = output.wout
        summary["wout_class"] = repr(wout.__class__)
        for field in ("fsqr", "fsqz", "fsql", "wb", "wp", "volume_p"):
            if hasattr(wout, field):
                summary[f"wout_{field}"] = float(getattr(wout, field))

        save_path = Path(args.save_wout) if args.save_wout else outdir / f"wout_{input_path.name.replace('input.', '')}_vmecpp.nc"
        try:
            wout.save(str(save_path))
            summary["wout_path"] = str(save_path)
        except Exception as exc:
            summary["wout_save_error"] = str(exc)

    # Optional compare to VMEC2000 reference wout.
    ref_path = Path(args.compare_to) if args.compare_to else VMEC2000_TESTS / f"wout_{input_path.name.replace('input.', '')}_reference.nc"
    if summary.get("wout_path") and ref_path.exists():
        f1 = _load_netcdf(Path(summary["wout_path"]))
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

    out_json = outdir / f"vmecpp_driver_probe_{input_path.name.replace('input.', '')}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
