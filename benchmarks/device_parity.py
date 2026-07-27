#!/usr/bin/env python3
"""Audit CPU/GPU parity of a forward solve and implicit boundary gradients.

Hardware is selected through VMEX's public ``device=`` API; this script does
not set or require JAX platform environment variables.  The default runs on
every available CPU/GPU platform.  ``--devices cpu`` is the CPU-only lane.

Examples::

    python benchmarks/device_parity.py --quick --metrics mhd_energy --output /tmp/parity.json
    python benchmarks/device_parity.py --devices cpu,gpu --output parity.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import vmex
import jax

from vmex.core import implicit as im
from vmex.core import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual


DATA = REPO / "examples" / "data"
METRIC_NAMES = (
    "mhd_energy",
    "magnetic_well",
    "dmerc_interior_mean",
    "jdotb_interior_mean",
    "glasser_d_r_interior_mean",
    "quasisymmetry",
    "quasi_isodynamic",
)


def _device_of(value: Any) -> str:
    device = getattr(value, "device", None)
    return str(device() if callable(device) else device)


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = call()
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def _relative_difference(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), np.finfo(float).tiny)
    return abs(left - right) / scale


def _available_devices(gpu_id: int = 0) -> dict[str, Any]:
    devices = {"cpu": jax.devices("cpu")[0]}
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        return devices
    if gpu_id >= len(gpus):
        raise ValueError(
            f"--gpu-id {gpu_id} requested but only {len(gpus)} GPU(s) present")
    devices["gpu"] = gpus[gpu_id]
    return devices


def _requested_devices(spec: str, available: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    requested = list(available) if spec == "available" else [part.strip() for part in spec.split(",")]
    if not requested or any(part not in ("cpu", "gpu") for part in requested):
        raise ValueError("--devices must be 'available' or a comma-separated subset of cpu,gpu")
    selected = [kind for kind in dict.fromkeys(requested) if kind in available]
    skipped = {
        kind: f"no {kind.upper()} JAX device is available"
        for kind in dict.fromkeys(requested)
        if kind not in available
    }
    return selected, skipped


def _input(quick: bool) -> VmecInput:
    inp = VmecInput.from_file(DATA / "input.minimal_seed_nfp2")
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    # Move off the exactly axisymmetric, zero-transform saddle.
    rbc[inp.ntor + 1, 1] += 0.01
    zbs[inp.ntor + 1, 1] += 0.01
    return dataclasses.replace(
        inp,
        rbc=rbc,
        zbs=zbs,
        ncurr=0,
        ai=np.asarray([0.4, 0.1]),
        ns_array=np.asarray([7 if quick else 11]),
        ftol_array=np.asarray([1.0e-9 if quick else 1.0e-11]),
        niter_array=np.asarray([1200 if quick else 2000]),
    )


def _metrics(quick: bool) -> dict[str, Callable[[Any], Any]]:
    qs = opt.QuasisymmetryRatioResidual([0.5], helicity_m=1, helicity_n=0)
    qi = QIResidual(
        [0.5],
        mboz=3 if quick else 4,
        nboz=3 if quick else 4,
        oversample=1,
        nphi=13 if quick else 17,
        nalpha=7 if quick else 9,
        n_levels=4 if quick else 6,
    )
    return {
        "mhd_energy": lambda sol: sol.wb,
        "magnetic_well": lambda sol: opt.magnetic_well(sol.state, sol.runtime),
        "dmerc_interior_mean": lambda sol: opt.d_merc_state(
            sol.state, sol.runtime
        )[2:-1].mean(),
        "jdotb_interior_mean": lambda sol: opt.jdotb_state(
            sol.state, sol.runtime
        )[2:-1].mean(),
        "glasser_d_r_interior_mean": lambda sol: opt.glasser_d_r_state(
            sol.state, sol.runtime, shear_epsilon=1.0e-8
        )[2:-1].mean(),
        "quasisymmetry": lambda sol: qs.total_state(sol.state, sol.runtime),
        "quasi_isodynamic": lambda sol: qi.total_state(sol.state, sol.runtime),
    }


def _run_lane(
    kind: str,
    device: Any,
    inp: VmecInput,
    metric_names: list[str],
    *,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ns = int(inp.ns_array[-1])
    ftol = float(inp.ftol_array[-1])
    max_iterations = int(inp.niter_array[-1])
    index = (int(inp.ntor) + 1, 1)
    params = im.params_from_input(inp, device=device)

    solve = lambda: im.run(  # noqa: E731
        inp,
        params,
        ns=ns,
        ftol=ftol,
        max_iterations=max_iterations,
        device=device,
    )
    cold_solution, cold_s = _timed(solve)
    solution, warm_s = _timed(solve)
    state = np.concatenate([
        np.asarray(leaf).ravel() for leaf in jax.tree.leaves(solution.state)
    ])
    scalars = {
        name: float(np.asarray(getattr(solution, name)))
        for name in ("wb", "wp", "aspect", "volume", "iota_axis", "iota_edge")
    }
    lane = {
        "status": "ok",
        "requested_device": kind,
        "resolved_device": str(device),
        "parameter_device": _device_of(params.rbc),
        "forward": {
            "cold_wall_s": cold_s,
            "warm_wall_s": warm_s,
            "state_device": _device_of(solution.state.R_cos),
            "state_size": int(state.size),
            "state_l2_norm": float(np.linalg.norm(state)),
            "state_max_abs": float(np.max(np.abs(state))),
            "all_finite": bool(np.all(np.isfinite(state))),
            "scalars": scalars,
        },
        "gradients": {},
    }

    metrics = _metrics(quick)
    x0 = params.rbc[index]
    artifacts: dict[str, Any] = {"state": state, "metrics": {}}
    for name in metric_names:
        metric = metrics[name]

        def objective(x):
            perturbed = dataclasses.replace(params, rbc=params.rbc.at[index].set(x))
            return metric(im.run(
                inp,
                perturbed,
                ns=ns,
                ftol=ftol,
                max_iterations=max_iterations,
                device=device,
            ))

        value_and_grad = jax.value_and_grad(objective)
        (value, gradient), metric_cold_s = _timed(lambda: value_and_grad(x0))
        (value, gradient), metric_warm_s = _timed(lambda: value_and_grad(x0))
        value_f, gradient_f = float(value), float(gradient)
        lane["gradients"][name] = {
            "value": value_f,
            "d_d_rbc": gradient_f,
            "value_device": _device_of(value),
            "gradient_device": _device_of(gradient),
            "cold_wall_s": metric_cold_s,
            "warm_wall_s": metric_warm_s,
        }
        artifacts["metrics"][name] = (value_f, gradient_f)

    # Keep this reference alive until both timed solves have synchronized.
    del cold_solution
    return lane, artifacts


def _compare_lanes(
    cpu: dict[str, Any],
    gpu: dict[str, Any],
    *,
    rtol: float,
) -> dict[str, Any]:
    cpu_state, gpu_state = cpu["state"], gpu["state"]
    delta = gpu_state - cpu_state
    state_scale = max(float(np.linalg.norm(cpu_state)), np.finfo(float).tiny)
    forward = {
        "state_relative_l2": float(np.linalg.norm(delta)) / state_scale,
        "state_max_abs_difference": float(np.max(np.abs(delta))),
    }
    metrics = {}
    for name in cpu["metrics"]:
        cpu_value, cpu_gradient = cpu["metrics"][name]
        gpu_value, gpu_gradient = gpu["metrics"][name]
        metrics[name] = {
            "value_relative_difference": _relative_difference(cpu_value, gpu_value),
            "gradient_relative_difference": _relative_difference(cpu_gradient, gpu_gradient),
        }
        metrics[name]["passed"] = bool(
            metrics[name]["value_relative_difference"] <= rtol
            and metrics[name]["gradient_relative_difference"] <= rtol
        )
    passed = forward["state_relative_l2"] <= rtol and all(
        record["passed"] for record in metrics.values()
    )
    return {
        "status": "passed" if passed else "failed",
        "relative_tolerance": rtol,
        "forward": forward,
        "metrics": metrics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default="available",
        help="'available' (default) or a comma-separated subset of cpu,gpu",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(METRIC_NAMES),
        help=f"comma-separated subset of {','.join(METRIC_NAMES)}",
    )
    parser.add_argument("--quick", action="store_true", help="use a smaller smoke-test grid")
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="which GPU to benchmark (jax.devices('gpu')[GPU_ID]); errors if absent",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-7)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    available = _available_devices(gpu_id=int(args.gpu_id))
    try:
        devices, skipped = _requested_devices(args.devices, available)
    except ValueError as exc:
        _parser().error(str(exc))
    metric_names = [part.strip() for part in args.metrics.split(",") if part.strip()]
    unknown = [name for name in metric_names if name not in METRIC_NAMES]
    if not metric_names or unknown:
        _parser().error(f"unknown or empty --metrics selection: {unknown or args.metrics!r}")

    inp = _input(args.quick)
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {
            "python": platform.python_version(),
            "vmex": getattr(vmex, "__version__", "unknown"),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
        },
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "jax": {
            "default_backend": jax.default_backend(),
            "devices": {
                kind: [str(device) for device in jax.devices(kind)]
                for kind in available
            },
            "platform_environment": {
                key: os.environ.get(key) for key in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME")
            },
        },
        "configuration": {
            "quick": args.quick,
            "input": "examples/data/input.minimal_seed_nfp2",
            "ns": int(inp.ns_array[-1]),
            "ftol": float(inp.ftol_array[-1]),
            "max_iterations": int(inp.niter_array[-1]),
            "iota_profile": np.asarray(inp.ai).tolist(),
            "boundary_parameter": {
                "field": "rbc",
                "index": [int(inp.ntor) + 1, 1],
                "mode": {"n": 1, "m": 1},
            },
            "metrics": metric_names,
        },
        "skipped_devices": skipped,
        "lanes": {},
    }
    artifacts = {}
    for kind in devices:
        print(f"running {kind} lane ({', '.join(metric_names)})", file=sys.stderr, flush=True)
        lane, artifact = _run_lane(
            kind, available[kind], inp, metric_names, quick=args.quick
        )
        result["lanes"][kind] = lane
        artifacts[kind] = artifact

    if "cpu" in artifacts and "gpu" in artifacts:
        result["comparison"] = _compare_lanes(artifacts["cpu"], artifacts["gpu"], rtol=args.rtol)
    else:
        reason = "CPU/GPU comparison requires both lanes"
        if skipped:
            reason += "; " + "; ".join(skipped.values())
        result["comparison"] = {"status": "skipped", "reason": reason}

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    return int(result["comparison"]["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
