#!/usr/bin/env python
"""Run native VMEC2000 on the canonical Solov'ev input with provenance."""

from __future__ import annotations

import argparse
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
    return {"vmec2000_commit": commit, "vmec2000_dirty": dirty}


def _peak_child_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    if not args.executable.is_file():
        parser.error(f"executable does not exist: {args.executable}")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if not args.input.name.startswith("input."):
        parser.error("input filename must start with 'input.'")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if not (args.source_repo / ".git").exists():
        parser.error(f"source repository does not exist: {args.source_repo}")

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
    compute = re.search(
        r"TOTAL COMPUTATIONAL TIME \(SEC\)\s+([0-9.]+)", threed
    )
    success = (
        completed.returncode == 0
        and "EXECUTION TERMINATED NORMALLY" in threed
        and wout_path.is_file()
    )
    report = {
        "schema": "vmex.vmec2000-solovev-run/1",
        "executable": str(args.executable),
        "input": str(args.input),
        "output_wout": str(wout_path),
        "success": success,
        "returncode": completed.returncode,
        "version": None if version is None else version.group(1),
        "iterations": int(final_iteration),
        "fsqr": float(fsqr),
        "fsqz": float(fsqz),
        "fsql": float(fsql),
        "compute_seconds": None if compute is None else float(compute.group(1)),
        "wall_seconds": wall_seconds,
        "peak_rss_mib": _peak_child_rss_mib(),
        "platform": platform.platform(),
        **_git_state(args.source_repo),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not success:
        raise SystemExit("VMEC2000 did not terminate normally")


if __name__ == "__main__":
    main()
