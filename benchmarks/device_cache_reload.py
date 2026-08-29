#!/usr/bin/env python3
"""Measure cold-cache, cache-reload, and warm VMEX device performance.

Each requested CPU/GPU lane runs :mod:`benchmarks.device_parity` in two fresh
processes sharing one otherwise-empty, lane-specific JAX compilation cache.
The first process measures a true cold cache plus an in-process warm repeat;
the second measures persistent-cache reload plus its own warm repeat.  The
child also measures one scalar implicit gradient, actual placement, host RSS,
and device peak memory.

The default uses the bounded parity smoke case and the MHD-energy gradient::

    python benchmarks/device_cache_reload.py --devices gpu \
        --output /tmp/vmex-wsl2-gpu.json

Use ``--full`` for the larger parity case.  A temporary cache is used by
default and removed after measurements; ``--cache-dir`` accepts only an empty
directory and is never cleared by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PARITY = REPO / "benchmarks" / "device_parity.py"


def _cache_stats(directory: Path) -> dict[str, int]:
    files = [entry for entry in directory.rglob("*") if entry.is_file()]
    return {
        "file_count": len(files),
        "bytes": sum(entry.stat().st_size for entry in files),
    }


def _parse_devices(spec: str) -> tuple[str, ...]:
    devices = tuple(dict.fromkeys(part.strip() for part in spec.split(",") if part.strip()))
    if not devices or any(device not in ("cpu", "gpu") for device in devices):
        raise ValueError("--devices must be a comma-separated subset of cpu,gpu")
    return devices


def _run_child(
    *,
    device: str,
    cache_dir: Path,
    metrics: str,
    full: bool,
    timeout: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PARITY),
        "--devices",
        device,
        "--metrics",
        metrics,
    ]
    if not full:
        command.append("--quick")
    environment = {
        **os.environ,
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
        "VMEX_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "VMEX_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    }
    before = _cache_stats(cache_dir)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "subprocess_wall_s": float(timeout),
            "cache_before": before,
            "cache_after": _cache_stats(cache_dir),
            "stderr": (error.stderr or "")[-2000:],
        }
    elapsed = time.perf_counter() - started
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "returncode": process.returncode,
            "subprocess_wall_s": elapsed,
            "cache_before": before,
            "cache_after": _cache_stats(cache_dir),
            "stderr": (process.stderr or process.stdout)[-2000:],
        }
    return {
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "subprocess_wall_s": elapsed,
        "cache_before": before,
        "cache_after": _cache_stats(cache_dir),
        "child": payload,
        "stderr": process.stderr[-2000:],
    }


def _lane_timings(process: dict[str, Any], device: str) -> dict[str, float] | None:
    try:
        lane = process["child"]["lanes"][device]
        metrics = lane["gradients"]
        return {
            "subprocess_wall_s": float(process["subprocess_wall_s"]),
            "forward_first_s": float(lane["forward"]["cold_wall_s"]),
            "forward_warm_s": float(lane["forward"]["warm_wall_s"]),
            "gradient_first_s": sum(float(item["cold_wall_s"]) for item in metrics.values()),
            "gradient_warm_s": sum(float(item["warm_wall_s"]) for item in metrics.values()),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _speedups(
    cold_process: dict[str, Any],
    reload_process: dict[str, Any],
    device: str,
) -> dict[str, float] | None:
    cold = _lane_timings(cold_process, device)
    reload = _lane_timings(reload_process, device)
    if cold is None or reload is None:
        return None

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / max(denominator, sys.float_info.min)

    return {
        "process_cache_reload_speedup": ratio(
            cold["subprocess_wall_s"], reload["subprocess_wall_s"]
        ),
        "forward_cache_reload_speedup": ratio(
            cold["forward_first_s"], reload["forward_first_s"]
        ),
        "gradient_cache_reload_speedup": ratio(
            cold["gradient_first_s"], reload["gradient_first_s"]
        ),
        "reload_forward_warm_speedup": ratio(
            reload["forward_first_s"], reload["forward_warm_s"]
        ),
        "reload_gradient_warm_speedup": ratio(
            reload["gradient_first_s"], reload["gradient_warm_s"]
        ),
    }


def _run_campaign(args: argparse.Namespace, cache_root: Path) -> dict[str, Any]:
    from _provenance import git_state

    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": git_state(REPO),
        "configuration": {
            "devices": list(args.devices),
            "metrics": args.metrics,
            "quick": not args.full,
            "cache": "temporary" if args.cache_dir is None else "user-supplied-empty",
            "cache_min_compile_time_s": 0,
            "cache_min_entry_size_bytes": -1,
        },
        "lanes": {},
    }
    failed = False
    for device in args.devices:
        lane_cache = cache_root / device
        lane_cache.mkdir(parents=True, exist_ok=True)
        cold = _run_child(
            device=device,
            cache_dir=lane_cache,
            metrics=args.metrics,
            full=args.full,
            timeout=args.timeout,
        )
        reload = _run_child(
            device=device,
            cache_dir=lane_cache,
            metrics=args.metrics,
            full=args.full,
            timeout=args.timeout,
        )
        skipped = False
        try:
            skipped = device in cold["child"]["skipped_devices"]
        except (KeyError, TypeError):
            pass
        lane_status = "skipped" if skipped else (
            "ok" if cold["status"] == reload["status"] == "ok" else "failed"
        )
        failed |= lane_status == "failed"
        result["lanes"][device] = {
            "status": lane_status,
            "cold_cache_process": cold,
            "cache_reload_process": reload,
            "speedups": _speedups(cold, reload, device),
        }
    result["status"] = "failed" if failed else "ok"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cpu,gpu")
    parser.add_argument("--metrics", default="mhd_energy")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.devices = _parse_devices(args.devices)
    except ValueError as error:
        parser.error(str(error))
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    if args.cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="vmex-device-cache-") as directory:
            result = _run_campaign(args, Path(directory))
    else:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        if any(args.cache_dir.iterdir()):
            parser.error("--cache-dir must be empty; VMEX will not clear user data")
        result = _run_campaign(args, args.cache_dir)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
