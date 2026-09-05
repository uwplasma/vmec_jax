#!/usr/bin/env python
"""Measure what the batched force sweep costs in peak resident memory.

The polish setup evaluates the independent continuum force over two large
point sets: the overintegrated certificate grid, and the chart probes that
linearize the collocation residual.  Before 0.8.2 both ran as one flat
``vmap`` over every point at once, which at production stellarator
resolution is tens of GB in a single allocation and an OS kill rather than
a result.  :class:`~vmex.core.strong_force.ForceSweepPolicy` lets one build
run all three sweep strategies, so the comparison is three arms of one
commit rather than a diff between two:

``flat``
    one ``vmap`` over the whole grid — the pre-0.8.2 sweep.
``batched``
    ``lax.map`` over automatically sized batches, no remat boundary; fixes
    the forward sweep and leaves reverse-mode linearization unbounded.
``auto``
    the shipped policy: automatic batches with the per-point kernel
    checkpointed, so reverse passes replay it per batch.

Each arm runs in its own process because peak RSS is a per-process
high-water mark and an arm that is killed still has to report one.  The
parent reads that mark out of :func:`os.wait4`, which returns the rusage of
a child whether it exited or was killed by a signal.

Usage (all three arms, writing the committed artifact)::

    python benchmarks/polish_memory.py --input examples/data/input.solovev \\
        --output benchmarks/polish_memory_<host>.json

One arm in the current process (what the parent spawns)::

    python benchmarks/polish_memory.py --input ... --mode flat --stage chart
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import vmex  # noqa: E402
from vmex.core import implicit  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.multigrid import solve_multigrid  # noqa: E402
from vmex.core.polish import (  # noqa: E402
    build_low_order_preconditioner,
    make_strong_root_runtime,
    make_strong_structured_chart,
)
from vmex.core.polish_driver import PolishConfig  # noqa: E402
from vmex.core.radial_basis import BSplineBasis  # noqa: E402
from vmex.core.strong_force import (  # noqa: E402
    ForceSweepPolicy,
    certify_strong_force,
    force_sweep_batch,
    force_sweep_measurement,
    lift_high_order_state,
)

from _provenance import assert_repo_vmex, file_sha256, git_state  # noqa: E402

SCHEMA = "vmex.polish-sweep-memory/1"

#: Sweep strategy per arm.  ``flat`` reproduces the pre-0.8.2 sweep exactly.
MODES: dict[str, ForceSweepPolicy] = {
    "flat": ForceSweepPolicy(batch=False, checkpoint=False),
    "batched": ForceSweepPolicy(checkpoint=False),
    "auto": ForceSweepPolicy(),
}

#: How far into the polish setup an arm runs.  ``certificate`` stops after
#: the independent oracle (the forward-sweep cliff); ``chart`` continues
#: through the structured chart build (the reverse-mode cliff).
STAGES = ("certificate", "chart")


def _phase(message: str) -> None:
    """Timestamped, flushed phase marker so an OOM kill names its phase."""

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _rss_scale() -> int:
    """Bytes per ``ru_maxrss`` unit: Linux reports KiB, macOS bytes."""

    return 1024 if sys.platform.startswith("linux") else 1


def _self_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _rss_scale()


def _deck(args: argparse.Namespace) -> VmecInput:
    """Load the deck with the requested resolution overrides applied."""

    inp = VmecInput.from_file(args.input)
    if args.mpol is not None or args.ntor is not None:
        inp = inp.change_resolution(
            mpol=args.mpol if args.mpol is not None else int(inp.mpol),
            ntor=args.ntor if args.ntor is not None else int(inp.ntor),
        )
    if args.ns is not None:
        inp = dataclasses.replace(
            inp,
            ns_array=np.asarray([int(args.ns)]),
            ftol_array=np.asarray([float(inp.ftol_array[-1])]),
            niter_array=np.asarray([int(inp.niter_array[-1])]),
        )
    return inp


def run_arm(args: argparse.Namespace) -> dict[str, object]:
    """Run one sweep strategy through the polish setup in this process.

    The record is written out after every stage, not once at the end: an arm
    that the OS kills mid-sweep is the interesting arm, and its measurements
    up to that point must survive it.
    """

    policy = MODES[args.mode]
    started = time.perf_counter()
    inp = _deck(args)
    _phase(
        f"[{args.mode}] legacy solve start: mpol={int(inp.mpol)} "
        f"ntor={int(inp.ntor)} ns={np.asarray(inp.ns_array).tolist()}"
    )
    result = solve_multigrid(inp, polish_force_balance=False)
    legacy_seconds = time.perf_counter() - started
    _phase(f"[{args.mode}] legacy solve done in {legacy_seconds:.0f}s")

    config = PolishConfig(radial_degree=args.degree, radial_spans=args.radial_spans)
    ns = int(np.asarray(inp.ns_array)[-1])
    implicit_config = implicit.make_config(inp, ns=ns, lconm1=True, multigrid=False)
    params = implicit.params_from_input(inp)
    legacy_runtime = implicit.runtime_from_params(params, implicit_config)
    dof_mask = implicit._dof_mask(result.state, legacy_runtime, implicit_config)
    refined_state = implicit._refined_state(
        implicit_config, params, result.state, dof_mask
    )
    radial_basis = (
        None
        if config.radial_spans is None
        else BSplineBasis.clamped(
            np.linspace(0.0, 1.0, config.radial_spans + 1),
            degree=config.radial_degree,
        )
    )
    native = lift_high_order_state(
        refined_state, legacy_runtime, radial_basis=radial_basis,
        degree=config.radial_degree,
    )
    _phase(f"[{args.mode}] native lift done")
    setup_peak = _self_peak_rss_bytes()

    def checkpoint(record: dict[str, object]) -> None:
        if args.arm_output is not None:
            args.arm_output.write_text(
                json.dumps(record, indent=2, sort_keys=True))

    record: dict[str, object] = {
        "mode": args.mode,
        "stage": args.stage,
        "sweep_policy": dataclasses.asdict(policy),
        "resolution": {
            "mpol": int(inp.mpol),
            "ntor": int(inp.ntor),
            "ns": ns,
            "mnmax": int(np.asarray(native.m).size),
            "radial_basis_size": int(native.radial_basis.size),
        },
        "lift_peak_rss_bytes": setup_peak,
        "legacy_seconds": legacy_seconds,
    }
    checkpoint(record)

    with force_sweep_measurement(policy):
        certificate_started = time.perf_counter()
        _phase(f"[{args.mode}] certifying initial state")
        certificate = certify_strong_force(native)
        jax.block_until_ready(certificate.absolute_l2)
        record["certificate_seconds"] = time.perf_counter() - certificate_started
        record["certificate_absolute_l2"] = float(certificate.absolute_l2)
        record["certificate_peak_rss_bytes"] = _self_peak_rss_bytes()
        _phase(
            f"[{args.mode}] certificate done in "
            f"{record['certificate_seconds']:.0f}s; peak RSS "
            f"{record['certificate_peak_rss_bytes'] / 1024**3:.1f} GiB"
        )
        checkpoint(record)
        if args.stage == "chart":
            chart_started = time.perf_counter()
            low_preconditioner = build_low_order_preconditioner(
                native, params, implicit_config, refined_state, dof_mask,
                probe_chunk_size=4,
            )
            runtime = make_strong_root_runtime(
                native, low_preconditioner, dof_mask, balance_full_root=False,
            )
            _phase(f"[{args.mode}] runtime built; building chart")
            chart = make_strong_structured_chart(runtime)
            record["chart_size"] = int(chart.size)
            record["chart_seconds"] = time.perf_counter() - chart_started
            record["chart_peak_rss_bytes"] = _self_peak_rss_bytes()
            record["sweep_batch_points"] = int(
                force_sweep_batch(
                    native,
                    int(np.asarray(runtime.radial_nodes).size)
                    * int(np.asarray(runtime.theta).size)
                    * int(np.asarray(runtime.zeta).size),
                    policy,
                )
            )
            _phase(
                f"[{args.mode}] chart done in {record['chart_seconds']:.0f}s; "
                f"peak RSS {record['chart_peak_rss_bytes'] / 1024**3:.1f} GiB"
            )
            checkpoint(record)
    record["peak_rss_bytes"] = _self_peak_rss_bytes()
    record["total_seconds"] = time.perf_counter() - started
    checkpoint(record)
    return record


def _spawn(args: argparse.Namespace, mode: str) -> dict[str, object]:
    """Run one arm as a child and read its peak RSS out of ``wait4``.

    An arm that the OS kills for memory cannot write its own record; the
    kernel still accounts its high-water mark to the parent's child rusage,
    so the measurement survives exactly the failure it exists to measure.
    """

    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--input", str(args.input),
        "--mode", mode,
        "--stage", args.stage,
        "--degree", str(args.degree),
        "--arm-output", str(args.scratch / f"arm_{mode}.json"),
    ]
    for name in ("mpol", "ntor", "ns", "radial_spans"):
        value = getattr(args, name)
        if value is not None:
            command += [f"--{name.replace('_', '-')}", str(value)]
    started = time.perf_counter()
    child = subprocess.Popen(command, env={**os.environ})
    _, status, usage = os.wait4(child.pid, 0)
    seconds = time.perf_counter() - started
    signalled = os.WIFSIGNALED(status)
    arm_path = args.scratch / f"arm_{mode}.json"
    record: dict[str, object] = {
        "mode": mode,
        "peak_rss_bytes": int(usage.ru_maxrss) * _rss_scale(),
        "wall_seconds": seconds,
        "exit_status": (
            f"signal {os.WTERMSIG(status)}" if signalled
            else f"exit {os.WEXITSTATUS(status)}"
        ),
        "completed": (not signalled) and os.WEXITSTATUS(status) == 0,
    }
    if arm_path.exists():
        record["detail"] = json.loads(arm_path.read_text())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm-output", type=Path)
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--stage", choices=STAGES, default="chart")
    parser.add_argument("--mpol", type=int)
    parser.add_argument("--ntor", type=int)
    parser.add_argument("--ns", type=int)
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--radial-spans", type=int)
    parser.add_argument(
        "--scratch", type=Path, default=Path.cwd(),
        help="directory for the per-arm records the parent collects",
    )
    parser.add_argument(
        "--host", default="",
        help="prose description of the measurement machine for the artifact",
    )
    args = parser.parse_args()

    if args.mode is not None:
        record = run_arm(args)
        print(json.dumps(record, indent=2, sort_keys=True))
        return

    if args.output is None:
        parser.error("--output is required when running every arm")
    args.scratch.mkdir(parents=True, exist_ok=True)
    arms = [_spawn(args, mode) for mode in ("flat", "batched", "auto")]
    payload = {
        "schema": SCHEMA,
        "provenance": {
            **git_state(REPO),
            "vmex_version": vmex.__version__,
            "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
            "host": args.host,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "x64": bool(jax.config.jax_enable_x64),
            "jax_backend": jax.default_backend(),
        },
        "deck": {
            "name": Path(args.input).name,
            "sha256_prefix": file_sha256(Path(args.input))[:16],
        },
        "stage": args.stage,
        "arms": arms,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for arm in arms:
        print(
            f"{arm['mode']:>8}: peak RSS "
            f"{arm['peak_rss_bytes'] / 1024**3:7.2f} GiB  "
            f"{arm['exit_status']}  {arm['wall_seconds']:.0f}s"
        )


if __name__ == "__main__":
    main()
