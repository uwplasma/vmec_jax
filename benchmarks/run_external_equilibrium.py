#!/usr/bin/env python
"""Run a DESC or VMEC2000 equilibrium with portable timing provenance."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import time


def _git_state(path: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _peak_rss_mib(children: bool = False) -> float:
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    value = resource.getrusage(who).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def _run_desc(args: argparse.Namespace) -> dict[str, object]:
    import jax
    from desc.vmec import VMECIO

    started = time.perf_counter()
    load_started = time.perf_counter()
    equilibrium = VMECIO.load(
        args.wout,
        L=args.L,
        M=args.M,
        N=args.N,
        spectral_indexing=args.spectral_indexing,
        profile=args.profile,
    )
    load_seconds = time.perf_counter() - load_started
    solve_started = time.perf_counter()
    equilibrium, result = equilibrium.solve(
        objective="force",
        ftol=args.tolerance,
        xtol=args.tolerance,
        gtol=args.tolerance,
        maxiter=args.maxiter,
        verbose=0,
    )
    solve_seconds = time.perf_counter() - solve_started
    save_started = time.perf_counter()
    VMECIO.save(equilibrium, args.output_wout, surfs=args.surfaces, verbose=0)
    save_seconds = time.perf_counter() - save_started
    return {
        "schema": "vmex.external-equilibrium-run/1",
        "engine": "desc",
        "input": args.wout.name,
        "output_wout": args.output_wout.name,
        "representation": {
            "L": args.L,
            "M": args.M,
            "N": args.N,
            "spectral_indexing": args.spectral_indexing,
            "profile": args.profile,
            "surfaces": args.surfaces,
        },
        "controls": {
            "maxiter": args.maxiter,
            "ftol": args.tolerance,
            "xtol": args.tolerance,
            "gtol": args.tolerance,
        },
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "timing_seconds": {
            "load": load_seconds,
            "solve": solve_seconds,
            "save": save_seconds,
            "total": time.perf_counter() - started,
        },
        "peak_rss_mib": _peak_rss_mib(),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "desc": metadata.version("desc-opt"),
            "jax": jax.__version__,
        },
        "devices": [str(device) for device in jax.devices()],
        "source": _git_state(args.source_repo),
    }


def _run_vmec2000(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True)
    local_input = args.output_dir / args.input.name
    shutil.copy2(args.input, local_input)
    started = time.perf_counter()
    completed = subprocess.run(
        [str(args.executable), local_input.name],
        cwd=args.output_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    wall_seconds = time.perf_counter() - started
    (args.output_dir / "stdout.txt").write_text(completed.stdout)
    (args.output_dir / "stderr.txt").write_text(completed.stderr)
    case = args.input.name.removeprefix("input.")
    threed_path = args.output_dir / f"threed1.{case}"
    wout_path = args.output_dir / f"wout_{case}.nc"
    if not threed_path.is_file():
        raise SystemExit("VMEC2000 did not produce threed1 output")
    threed = threed_path.read_text()
    iterations = re.findall(
        r"^\s*(\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)"
        r"\s+([0-9.]+E[+-]\d+)",
        threed,
        flags=re.MULTILINE,
    )
    if not iterations:
        raise SystemExit("VMEC2000 iteration history was not found")
    final_iteration, fsqr, fsqz, fsql = iterations[-1]
    version = re.search(r"VERSION\s+([^\s]+)", threed)
    compute = re.search(r"TOTAL COMPUTATIONAL TIME \(SEC\)\s+([0-9.]+)", threed)
    success = completed.returncode == 0 and "EXECUTION TERMINATED NORMALLY" in threed and wout_path.is_file()
    return {
        "schema": "vmex.external-equilibrium-run/1",
        "engine": "vmec2000",
        "input": args.input.name,
        "output_wout": wout_path.name,
        "success": success,
        "returncode": completed.returncode,
        "version": None if version is None else version.group(1),
        "iterations": int(final_iteration),
        "fsqr": float(fsqr),
        "fsqz": float(fsqz),
        "fsql": float(fsql),
        "timing_seconds": {
            "solve": None if compute is None else float(compute.group(1)),
            "total": wall_seconds,
        },
        "peak_rss_mib": _peak_rss_mib(children=True),
        "platform": platform.platform(),
        "source": _git_state(args.source_repo),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="engine", required=True)

    desc = subparsers.add_parser("desc", help="import, solve, and export with DESC")
    desc.add_argument("--wout", type=Path, required=True)
    desc.add_argument("--output-wout", type=Path, required=True)
    desc.add_argument("--report", type=Path, required=True)
    desc.add_argument("--source-repo", type=Path, required=True)
    desc.add_argument("--L", type=int, required=True)
    desc.add_argument("--M", type=int, required=True)
    desc.add_argument("--N", type=int, required=True)
    desc.add_argument("--surfaces", type=int, default=129)
    desc.add_argument("--spectral-indexing", default="fringe")
    desc.add_argument("--profile", default="iota")
    desc.add_argument("--maxiter", type=int, default=300)
    desc.add_argument("--tolerance", type=float, default=1.0e-12)

    vmec = subparsers.add_parser("vmec2000", help="solve an input deck with VMEC2000")
    vmec.add_argument("--executable", type=Path, required=True)
    vmec.add_argument("--input", type=Path, required=True)
    vmec.add_argument("--output-dir", type=Path, required=True)
    vmec.add_argument("--report", type=Path, required=True)
    vmec.add_argument("--source-repo", type=Path, required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    for path, message in (
        (args.source_repo / ".git", "source repository does not exist"),
        (getattr(args, "wout", None), "wout does not exist"),
        (getattr(args, "executable", None), "executable does not exist"),
        (getattr(args, "input", None), "input does not exist"),
    ):
        if path is not None and not path.exists():
            parser.error(f"{message}: {path}")
    if args.engine == "vmec2000":
        if not args.input.name.startswith("input."):
            parser.error("input filename must start with 'input.'")
        if args.output_dir.exists():
            parser.error(f"output directory already exists: {args.output_dir}")
        report = _run_vmec2000(args)
    else:
        report = _run_desc(args)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["success"]:
        raise SystemExit(f"{args.engine} did not terminate successfully")


if __name__ == "__main__":
    main()
