#!/usr/bin/env python3
"""Fresh-process resource profiles for the four production solver paths.

Each case runs in its own child process so process peak RSS is attributable.
Hardware is selected through VMEX's public ``device=`` API; this script does
not set JAX platform-selection environment variables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
MARKER = "VMEX_RESOURCE_PROFILE:"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _time_command() -> list[str]:
    if platform.system() == "Darwin":
        return ["/usr/bin/time", "-l"]
    if platform.system() == "Linux":
        return ["/usr/bin/time", "-v"]
    raise RuntimeError(f"unsupported operating system: {platform.system()}")


def _peak_rss_bytes(stderr: str) -> int:
    match = re.search(r"(\d+)\s+maximum resident set size", stderr)
    if match:
        return int(match.group(1))
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    if match:
        return int(match.group(1)) * 1024
    raise ValueError("OS peak RSS was absent from /usr/bin/time output")


def _checksum(tree: Any) -> str:
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        if array.dtype.kind not in "biufc":
            continue
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _device_memory() -> dict[str, dict[str, int]]:
    import jax

    keep = ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")
    return {
        str(device): {key: int(value) for key, value in stats.items() if key in keep}
        for device in jax.local_devices()
        if (stats := device.memory_stats())
    }


def _memory_stats(executables) -> dict[str, int] | None:
    totals: dict[str, int] = {}
    for executable in executables:
        stats = executable.memory_analysis()
        if stats is None:
            continue
        for field in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "generated_code_size_in_bytes",
        ):
            totals[field] = totals.get(field, 0) + int(getattr(stats, field, 0))
    return totals or None


def _compiled_memory() -> dict[str, int] | None:
    from vmex.core import solver

    return _memory_stats(solver._LANE_EXECUTABLES.values())


def _native_threads() -> int | None:
    status = Path("/proc/self/status")
    if status.is_file():
        match = re.search(r"^Threads:\s+(\d+)$", status.read_text(), re.MULTILINE)
        return int(match.group(1)) if match else None
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["ps", "-M", str(os.getpid())], capture_output=True, text=True
        )
        return max(0, len(result.stdout.splitlines()) - 1) if result.returncode == 0 else None
    return None


def _core_summary(result: Any) -> dict[str, Any]:
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "residuals": {
            name: float(getattr(result, name))
            for name in ("fsqr", "fsqz", "fsql")
        },
        "state_sha256": _checksum(result.state),
    }


def _timed(call):
    import jax

    started = time.perf_counter()
    result = call()
    for leaf in jax.tree.leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result, time.perf_counter() - started


def _device(args: argparse.Namespace) -> Any:
    if args.device == "none":
        return None
    if args.device in ("cpu", "gpu"):
        import jax

        devices = jax.devices(args.device)
        if args.device_index >= len(devices):
            raise ValueError(
                f"{args.device} index {args.device_index} is unavailable; "
                f"found {len(devices)} device(s)"
            )
        return devices[args.device_index]
    return args.device


def _profile_core(args: argparse.Namespace, *, free_boundary: bool) -> dict[str, Any]:
    import jax

    from vmex.core.input import VmecInput
    from vmex.core.multigrid import solve_free_boundary_multigrid, solve_multigrid
    from vmex.core import solver
    from benchmarks._provenance import file_sha256

    path = Path(args.free_input if free_boundary else args.fixed_input)
    inp = VmecInput.from_file(path)
    if bool(inp.lfreeb) != free_boundary:
        expected = "free" if free_boundary else "fixed"
        raise ValueError(f"{path.name} is not a {expected}-boundary input")

    def run():
        if free_boundary:
            return solve_free_boundary_multigrid(
                inp,
                mgrid_path=args.mgrid,
                device=_device(args),
                verbose=False,
                prefetch_compile=args.prefetch_compile,
                release_stage_cache=args.release_stage_cache,
            )
        return solve_multigrid(
            inp,
            device=_device(args),
            verbose=False,
            prefetch_compile=args.prefetch_compile,
            release_stage_cache=args.release_stage_cache,
        )

    compile_s = None
    if not free_boundary and args.prefetch_compile:
        resolution = solver.resolution_from_input(inp)
        runtime = solver.prepare_runtime(inp, resolution)
        use_fft = solver._resolve_use_fft(None, _device(args), resolution)
        started = time.perf_counter()
        solver._prefetch_block_lane(runtime, use_fft=use_fft)
        compile_s = time.perf_counter() - started
    cold, cold_exec_s = _timed(run)
    warm, warm_s = _timed(run)
    cold_s = (compile_s or 0.0) + cold_exec_s
    xla_memory = _compiled_memory()
    return {
        "input": path.name,
        "input_sha256": file_sha256(path),
        "mgrid": Path(args.mgrid).name if free_boundary else None,
        "mgrid_sha256": file_sha256(Path(args.mgrid)) if free_boundary else None,
        "resolution": {
            "ns_array": [int(value) for value in inp.ns_array],
            "mpol": int(inp.mpol),
            "ntor": int(inp.ntor),
            "lasym": bool(inp.lasym),
        },
        "prefetch_compile": args.prefetch_compile,
        "release_stage_cache": args.release_stage_cache,
        "cold_s": cold_s,
        "aot_compile_s": compile_s,
        "first_execution_s": cold_exec_s,
        "warm_s": warm_s,
        "compile_estimate_s": max(0.0, cold_s - warm_s),
        "cold": _core_summary(cold),
        "warm": _core_summary(warm),
        "xla_memory": xla_memory,
        "xla_memory_reason": (
            None if xla_memory else "no prefetched executable was retained"
        ),
        "device_memory": _device_memory(),
        "devices": [str(device) for device in jax.devices()],
    }


def _profile_implicit(args: argparse.Namespace) -> dict[str, Any]:
    import contextlib

    import jax
    import numpy as np

    from vmex.core import implicit
    from vmex.core.device import device_scope
    from vmex.core.input import VmecInput
    from benchmarks._provenance import file_sha256

    path = Path(args.implicit_input)
    inp = VmecInput.from_file(path)
    target = _device(args)
    explicit = args.device in ("cpu", "gpu")

    def scope():
        return device_scope(target) if explicit else contextlib.nullcontext()

    with scope():
        params = implicit.params_from_input(inp, device=None if explicit else target)

    def objective(p):
        return implicit.run(
            inp, p, multigrid=True, device=None if explicit else target
        ).aspect

    def run():
        with scope():
            value, gradient = jax.value_and_grad(objective)(params)
            jax.block_until_ready((value, gradient))
        return value, gradient

    (_cold_value, cold_gradient), cold_s = _timed(run)
    (warm_value, warm_gradient), warm_s = _timed(run)
    return {
        "input": path.name,
        "input_sha256": file_sha256(path),
        "resolution": {
            "ns_array": [int(value) for value in inp.ns_array],
            "mpol": int(inp.mpol),
            "ntor": int(inp.ntor),
            "lasym": bool(inp.lasym),
        },
        "cold_s": cold_s,
        "warm_s": warm_s,
        "compile_estimate_s": max(0.0, cold_s - warm_s),
        "value": float(np.asarray(warm_value)),
        "gradient_norm": float(
            np.sqrt(
                sum(
                    np.vdot(np.asarray(leaf), np.asarray(leaf)).real
                    for leaf in jax.tree.leaves(warm_gradient)
                )
            )
        ),
        "gradient_sha256": _checksum(warm_gradient),
        "cold_gradient_sha256": _checksum(cold_gradient),
        "xla_memory": None,
        "xla_memory_reason": "implicit callback graph is not exposed as one executable",
        "device_memory": _device_memory(),
        "devices": [str(device) for device in jax.devices()],
    }


def _profile_mirror(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from benchmarks.profile_high_resolution import _external_mirror_field
    from vmex.mirror import (
        MirrorBoundary,
        MirrorConfig,
        MirrorResolution,
        SplineMirrorDiscretization,
        solve_beta_scan,
    )

    ns, nxi, elements = args.mirror_resolution
    config = MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=0, nxi=nxi),
        z_min=-0.8,
        z_max=0.8,
        ftol=1e-12,
        max_iterations=args.mirror_max_iterations,
    )
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build_cgl(config, elements=elements)
    grid = discretization.grid
    on_axis = 0.08 + 0.02 * jnp.asarray(grid.z) ** 2
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = discretization.fit_boundary(
        MirrorBoundary.from_axis_field(flux, on_axis, grid), source_grid
    )

    def run():
        return solve_beta_scan(
            boundary,
            discretization,
            config,
            _external_mirror_field,
            jnp.asarray([0.0, 0.1]),
            axial_flux_derivative=flux,
            reference_field=float(on_axis[center]),
            exterior_ntheta=args.mirror_exterior_ntheta,
            exterior_order=6,
            exterior_spectral_side_density=True,
            device=_device(args),
        )

    cold, cold_s = _timed(run)
    warm, warm_s = _timed(run)
    return {
        "resolution": {
            "ns": ns,
            "nxi": nxi,
            "elements": elements,
            "exterior_ntheta": args.mirror_exterior_ntheta,
        },
        "betas": [0.0, 0.1],
        "cold_s": cold_s,
        "warm_s": warm_s,
        "compile_estimate_s": max(0.0, cold_s - warm_s),
        "converged": [bool(result.converged) for result in warm],
        "iterations": [int(result.iterations) for result in warm],
        "variational_max": [float(result.variational_max) for result in warm],
        "state_sha256": _checksum(
            tuple(result.coefficient_state for result in warm)
        ),
        "cold_state_sha256": _checksum(
            tuple(result.coefficient_state for result in cold)
        ),
        "xla_memory": None,
        "xla_memory_reason": "host-driven nonlinear solve has no single compiled executable",
        "device_memory": _device_memory(),
        "devices": [str(device) for device in jax.devices()],
    }


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import vmex

    from benchmarks._provenance import assert_repo_vmex

    handlers = {
        "fixed": lambda: _profile_core(args, free_boundary=False),
        "free": lambda: _profile_core(args, free_boundary=True),
        "implicit": lambda: _profile_implicit(args),
        "mirror": lambda: _profile_mirror(args),
    }
    payload = handlers[args.worker_case]()
    return {
        "case": args.worker_case,
        "requested_device": args.device,
        "requested_device_index": args.device_index,
        "vmex_version": vmex.__version__,
        "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
        "jax_version": jax.__version__,
        "native_threads_observed": _native_threads(),
        "platform_environment": {
            name: os.environ.get(name)
            for name in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME")
        },
        **payload,
    }


def _mirror_ladder(value: str) -> list[tuple[int, int, int]]:
    try:
        rows = [tuple(int(part) for part in row.split(":")) for row in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use ns:nxi:elements[,ns:nxi:elements]") from exc
    if not rows or any(len(row) != 3 or min(row) < 1 for row in rows):
        raise argparse.ArgumentTypeError("use positive ns:nxi:elements triples")
    return rows


def _child_command(args: argparse.Namespace, case: str, mirror: tuple[int, int, int] | None) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        case,
        "--device",
        args.device,
        "--device-index",
        str(args.device_index),
        "--fixed-input",
        str(Path(args.fixed_input).resolve()),
        "--free-input",
        str(Path(args.free_input).resolve()),
        "--mgrid",
        str(Path(args.mgrid).resolve()),
        "--implicit-input",
        str(Path(args.implicit_input).resolve()),
        "--mirror-max-iterations",
        str(args.mirror_max_iterations),
        "--mirror-exterior-ntheta",
        str(args.mirror_exterior_ntheta),
        "--prefetch-compile" if args.prefetch_compile else "--no-prefetch-compile",
        "--release-stage-cache" if args.release_stage_cache else "--no-release-stage-cache",
    ]
    if mirror is not None:
        command += ["--mirror-resolution", ":".join(str(value) for value in mirror)]
    return command


def _reference_provenance(args: argparse.Namespace) -> dict[str, Any]:
    from benchmarks._provenance import file_sha256, git_state

    references: dict[str, Any] = {}
    if args.vmec2000_executable:
        executable = Path(args.vmec2000_executable)
        references["vmec2000"] = {
            "available": executable.is_file(),
            "executable_sha256": file_sha256(executable) if executable.is_file() else None,
        }
        if args.vmec2000_source and (Path(args.vmec2000_source) / ".git").exists():
            references["vmec2000"]["source_checkout"] = git_state(
                Path(args.vmec2000_source)
            )
    if args.vmecpp_python:
        python = Path(args.vmecpp_python)
        row: dict[str, Any] = {
            "available": python.is_file(),
            "python_sha256": file_sha256(python) if python.is_file() else None,
            "threads": args.vmecpp_threads,
        }
        if python.is_file():
            result = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import importlib.metadata,json,pathlib,subprocess,vmecpp;"
                    "p=pathlib.Path(vmecpp.__file__).resolve();"
                    "root=next((x for x in p.parents if (x/'.git').exists()),None);"
                    "r=subprocess.run(['git','rev-parse','HEAD'],cwd=root,"
                    "capture_output=True,text=True) if root else None;"
                    "print(json.dumps({'version':importlib.metadata.version('vmecpp'),"
                    "'source_commit':r.stdout.strip() if r and r.returncode==0 else None}))",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            details = json.loads(result.stdout) if result.returncode == 0 else {}
            row.update(details)
        if args.vmecpp_source and (Path(args.vmecpp_source) / ".git").exists():
            row["source_checkout"] = git_state(Path(args.vmecpp_source))
        references["vmecpp"] = row
    return references


def _run_child(args: argparse.Namespace, case: str, mirror: tuple[int, int, int] | None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    started = time.perf_counter()
    proc = subprocess.run(
        _time_command() + _child_command(args, case, mirror),
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    wall_s = time.perf_counter() - started
    if proc.returncode:
        raise RuntimeError(
            f"{case} worker failed ({proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    line = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.startswith(MARKER)),
        None,
    )
    if line is None:
        raise RuntimeError(f"{case} worker did not emit a result")
    return {
        **json.loads(line.removeprefix(MARKER)),
        "subprocess_wall_s": wall_s,
        "peak_rss_bytes": _peak_rss_bytes(proc.stderr),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("auto", "none", "cpu", "gpu"), default="cpu"
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="index for an explicit CPU or GPU; hardware remains selected via device=",
    )
    parser.add_argument("--cases", default="fixed,free,implicit,mirror")
    parser.add_argument("--fixed-input", default=DATA / "input.li383_low_res")
    parser.add_argument("--free-input", default=DATA / "input.cth_like_free_bdy")
    parser.add_argument("--mgrid", default=DATA / "mgrid_cth_like.nc")
    parser.add_argument("--implicit-input", default=DATA / "input.solovev")
    parser.add_argument(
        "--mirror-ladder",
        type=_mirror_ladder,
        default=_mirror_ladder("5:7:4,7:13:7,9:17:9"),
    )
    parser.add_argument("--mirror-max-iterations", type=int, default=500)
    parser.add_argument("--mirror-exterior-ntheta", type=int, default=8)
    parser.add_argument(
        "--prefetch-compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--release-stage-cache", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--vmec2000-executable", type=Path)
    parser.add_argument("--vmec2000-source", type=Path)
    parser.add_argument("--vmecpp-python", type=Path)
    parser.add_argument("--vmecpp-source", type=Path)
    parser.add_argument("--vmecpp-threads", type=int, default=10)
    parser.add_argument(
        "--worker-case",
        choices=("fixed", "free", "implicit", "mirror"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mirror-resolution",
        type=lambda value: _mirror_ladder(value)[0],
        default=(5, 7, 4),
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.device_index < 0:
        parser.error("--device-index must be nonnegative")
    if args.device_index and args.device not in ("cpu", "gpu"):
        parser.error("--device-index requires --device cpu or gpu")
    if args.vmecpp_threads < 1:
        parser.error("--vmecpp-threads must be positive")
    if args.worker_case:
        print(MARKER + json.dumps(_worker(args), sort_keys=True), flush=True)
        return

    from benchmarks._provenance import git_state

    requested = [case.strip() for case in args.cases.split(",") if case.strip()]
    unknown = set(requested) - {"fixed", "free", "implicit", "mirror"}
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    cases: dict[str, dict[str, Any]] = {}
    for case in requested:
        ladder = args.mirror_ladder if case == "mirror" else [None]
        for mirror in ladder:
            key = (
                f"mirror-ns{mirror[0]}-nxi{mirror[1]}"
                if mirror is not None
                else case
            )
            print(f"profiling {key}", flush=True)
            cases[key] = _run_child(args, case, mirror)
    report = {
        "schema": "vmex.resource-profile/1",
        "provenance": git_state(REPO),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "thread_policy": "JAX runtime default; no platform environment override",
        },
        "references": _reference_provenance(args),
        "cases": cases,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
