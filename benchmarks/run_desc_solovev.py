#!/usr/bin/env python
"""Solve the canonical VMEC-compatible Solov'ev case with DESC."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import time

import jax

from desc.vmec import VMECIO


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
    return {"desc_commit": commit, "desc_dirty": dirty}


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, required=True)
    parser.add_argument("--output-wout", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--desc-repo", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--surfaces", type=int, default=129)
    args = parser.parse_args()
    if not args.wout.is_file():
        parser.error(f"wout does not exist: {args.wout}")
    if not (args.desc_repo / ".git").exists():
        parser.error(f"DESC repository does not exist: {args.desc_repo}")

    started = time.perf_counter()
    load_started = time.perf_counter()
    equilibrium = VMECIO.load(
        args.wout,
        L=24,
        M=12,
        N=0,
        spectral_indexing="fringe",
        profile="iota",
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
    VMECIO.save(
        equilibrium,
        args.output_wout,
        surfs=args.surfaces,
        verbose=0,
    )
    save_seconds = time.perf_counter() - save_started
    report = {
        "schema": "vmex.desc-solovev-run/1",
        "source_wout": args.wout.name,
        "output_wout": args.output_wout.name,
        "representation": {
            "L": 24,
            "M": 12,
            "N": 0,
            "spectral_indexing": "fringe",
            "profile": "iota",
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
        "load_seconds": load_seconds,
        "solve_seconds": solve_seconds,
        "save_seconds": save_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_mib": _peak_rss_mib(),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "desc": metadata.version("desc-opt"),
            "jax": jax.__version__,
        },
        "devices": [str(device) for device in jax.devices()],
        **_git_state(args.desc_repo),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not result.success:
        raise SystemExit("DESC did not terminate successfully")


if __name__ == "__main__":
    main()
