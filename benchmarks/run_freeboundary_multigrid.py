#!/usr/bin/env python3
"""Benchmark and parity-check free-boundary radial multigrid against VMEC2000.

The default case is the public, converged CTH-like fixture.  The report stores
only aggregate timings, residuals, iteration counts, and normalized output
errors; it never copies input-deck text or coil/mgrid data into the JSON.  This
makes the harness safe to use with private validation cases too::

    python benchmarks/run_freeboundary_multigrid.py \
      --deck /private/path/input.case --mgrid /private/path/mgrid.nc \
      --ns 7,15 --ftol 1e-8,1e-10 --niter 1000,2500 \
      --out /private/path/result.json

VMEC2000 runs in a temporary directory.  VMEX is measured cold and warm in
one process so the warm row excludes JAX compilation and mgrid loading caches.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_free_boundary_multigrid

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy"
DEFAULT_MGRID = REPO / "examples" / "data" / "mgrid_cth_like.nc"
DEFAULT_XVMEC = Path("/Users/rogerio/local/STELLOPT/VMEC2000/Release/xvmec2000")


def _numbers(value: str, cast):
    return [cast(x) for x in value.replace(",", " ").split()]


def _replace_array(text: str, name: str, values: list) -> str:
    replacement = f"  {name} = " + ", ".join(str(v) for v in values) + ","
    pattern = rf"(?im)^\s*{re.escape(name)}\s*=.*$"
    out, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"input deck has no active {name} assignment")
    return out


def _stage_iterations(stdout: str) -> list[dict]:
    stages = []
    matches = list(re.finditer(
        r"NS\s*=\s*(\d+).*?FTOLV\s*=\s*([\d.E+\-]+).*?NITER\s*=\s*(\d+)",
        stdout,
    ))
    for i, match in enumerate(matches):
        block = stdout[match.end(): matches[i + 1].start() if i + 1 < len(matches) else None]
        rows = re.findall(
            r"(?m)^\s*(\d+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)",
            block,
        )
        first = rows[0] if rows else (None, None, None, None)
        last = rows[-1] if rows else (None, None, None, None)
        stages.append({
            "ns": int(match.group(1)), "ftol": float(match.group(2)),
            "niter_cap": int(match.group(3)),
            "first_iteration": None if first[0] is None else int(first[0]),
            "first_fsqr": None if first[1] is None else float(first[1]),
            "first_fsqz": None if first[2] is None else float(first[2]),
            "first_fsql": None if first[3] is None else float(first[3]),
            "iterations": None if last[0] is None else int(last[0]),
            "fsqr": None if last[1] is None else float(last[1]),
            "fsqz": None if last[2] is None else float(last[2]),
            "fsql": None if last[3] is None else float(last[3]),
        })
    return stages


def _vmec_converged(stdout: str) -> bool:
    return (
        "EXECUTION TERMINATED NORMALLY" in stdout
        and "Try increasing NITER" not in stdout
    )


def _normalized_max_error(mine: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    return float(np.max(np.abs(mine - reference)) / scale)


def _peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(raw / 1e6 if platform.system() == "Darwin" else raw / 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--mgrid", type=Path, default=DEFAULT_MGRID)
    parser.add_argument("--xvmec", type=Path, default=DEFAULT_XVMEC)
    parser.add_argument("--ns", default="7,15")
    parser.add_argument("--ftol", default="1e-8,1e-10")
    parser.add_argument("--niter", default="1000,2500")
    parser.add_argument("--out", type=Path,
                        default=REPO / "benchmarks" / "freeboundary_multigrid.json")
    args = parser.parse_args()

    ns = _numbers(args.ns, int)
    ftol = _numbers(args.ftol, float)
    niter = _numbers(args.niter, int)
    if not (len(ns) == len(ftol) == len(niter)):
        raise ValueError("--ns, --ftol, and --niter must have equal lengths")
    for path in (args.deck, args.mgrid, args.xvmec):
        if not path.is_file():
            raise FileNotFoundError(path)

    text = args.deck.read_text()
    text = _replace_array(text, "NS_ARRAY", ns)
    text = _replace_array(text, "FTOL_ARRAY", ftol)
    text = _replace_array(text, "NITER_ARRAY", niter)
    case = "vmex_fbmg_benchmark"

    with tempfile.TemporaryDirectory(prefix="vmex-fbmg-") as td:
        work = Path(td)
        deck = work / f"input.{case}"
        deck.write_text(text)
        os.symlink(args.mgrid.resolve(), work / Path(args.mgrid).name)

        t0 = time.perf_counter()
        vmec = subprocess.run(
            [str(args.xvmec.resolve()), deck.name], cwd=work,
            capture_output=True, text=True, check=False,
        )
        vmec_wall = time.perf_counter() - t0
        if vmec.returncode != 0:
            raise RuntimeError(
                f"VMEC2000 failed with {vmec.returncode}:\n{vmec.stdout[-2000:]}\n"
                f"{vmec.stderr[-2000:]}"
            )

        inp = replace(
            VmecInput.from_file(deck), ns_array=np.asarray(ns),
            ftol_array=np.asarray(ftol), niter_array=np.asarray(niter),
        )

        def vmex_run():
            lines: list[str] = []
            start = time.perf_counter()
            result = solve_free_boundary_multigrid(
                inp, mgrid_path=args.mgrid.resolve(), verbose=True,
                emit=lambda value="", end="\n": lines.append(str(value) + end),
                raise_on_max_iterations=False,
            )
            jax.block_until_ready(result.state.R_cos)
            return result, "".join(lines), time.perf_counter() - start

        cold, cold_stdout, cold_wall = vmex_run()
        warm, warm_stdout, warm_wall = vmex_run()

        from vmex.core.wout import read_wout

        reference = read_wout(work / f"wout_{case}.nc")
        mine = {(int(m), int(n)): i for i, (m, n) in enumerate(zip(warm.xm, warm.xn))}
        indices = np.asarray([
            mine[(int(m), int(n))] for m, n in zip(reference.xm, reference.xn)
        ])
        parity = {
            "rmnc_scale_relative_max": _normalized_max_error(
                warm.rmnc[:, indices], np.asarray(reference.rmnc)),
            "zmns_scale_relative_max": _normalized_max_error(
                warm.zmns[:, indices], np.asarray(reference.zmns)),
            "iotaf_scale_relative_max": _normalized_max_error(
                warm.iotaf, np.asarray(reference.iotaf)),
            "wb_relative": abs(float(warm.wb) - float(reference.wb))
            / max(abs(float(reference.wb)), np.finfo(float).tiny),
        }

    report = {
        "schema": 1,
        "case": args.deck.name,
        "input_data_embedded": False,
        "ladder": {"ns": ns, "ftol": ftol, "niter": niter},
        "environment": {
            "platform": platform.platform(), "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__, "x64": bool(jax.config.jax_enable_x64),
        },
        "vmec2000": {
            "wall_s": vmec_wall,
            "converged": _vmec_converged(vmec.stdout),
            "vacuum_turnons": vmec.stdout.count("VACUUM PRESSURE TURNED ON"),
            "stages": _stage_iterations(vmec.stdout),
        },
        "vmex_cold": {
            "wall_s": cold_wall, "converged": bool(cold.converged),
            "vacuum_turnons": cold_stdout.count("VACUUM PRESSURE TURNED ON"),
            "stages": _stage_iterations(cold_stdout),
        },
        "vmex_warm": {
            "wall_s": warm_wall, "converged": bool(warm.converged),
            "vacuum_turnons": warm_stdout.count("VACUUM PRESSURE TURNED ON"),
            "stages": _stage_iterations(warm_stdout),
            "peak_rss_mb": _peak_rss_mb(),
        },
        "final_wout_parity": parity,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
