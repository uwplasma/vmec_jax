#!/usr/bin/env python
"""Measure cold QA problem startup, warm gradient latency, and peak RSS."""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import resource
import statistics
import time

# Cold means no executable restored from a previous process.
os.environ["VMEX_COMPILATION_CACHE"] = "disabled"

import jax.numpy as jnp
import numpy as np

import vmex
from vmex import optimize as opt

from _provenance import assert_repo_vmex, file_sha256, git_state


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data" / "input.minimal_seed_nfp2"


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def _qa_input():
    inp = vmex.VmecInput.from_file(DATA)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.05, 0.05
    return replace(inp, rbc=rbc, zbs=zbs)


def _terms():
    qs = opt.QuasisymmetryRatioResidual(
        np.linspace(0.1, 1.0, 10), helicity_m=1, helicity_n=0)

    def iota_floor(state, runtime):
        return jnp.maximum(0.42 - opt.min_abs_iota(state, runtime), 0.0)

    return [
        (qs, 0.0, 1.0),
        (opt.aspect_ratio, 5.0, 1.0),
        (iota_floor, 0.0, 10.0),
        (opt.magnetic_well, 0.01, 1.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("scalar", "least-squares"),
                        default="scalar")
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    inp = _qa_input()
    terms = _terms()
    initial_solve_seconds = 0.0
    restart = None
    if args.lane == "least-squares":
        started = time.perf_counter()
        restart = opt.solve_equilibrium(inp)
        initial_solve_seconds = time.perf_counter() - started

    mpol = max(args.max_mode + 2, 5)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol,
        ntor=mpol,
        ntheta=2 * mpol + 6,
        nzeta=2 * mpol + 4,
    )
    started = time.perf_counter()
    if args.lane == "scalar":
        def loss(state, runtime):
            rows = opt.residuals_from_tuples(state, runtime, terms)
            return 0.5 * jnp.vdot(rows, rows)

        problem = opt.VmecProblem.from_loss(
            inp, loss, max_mode=args.max_mode, use_ess=True, ess_alpha=1.2,
            progress=False, evaluation_progress=False)
    else:
        problem = opt.VmecProblem.from_tuples(
            inp, terms, max_mode=args.max_mode, use_ess=True, ess_alpha=1.2,
            restart_from=restart, progress=False, evaluation_progress=False)
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    if args.lane == "scalar":
        evaluation = problem.compile_value_and_gradient(progress=False)
    else:
        evaluation = problem.compile_residual_and_jacobian(progress=False)
    compile_seconds = time.perf_counter() - started

    # FunctionProblem intentionally memoizes an identical decision vector, so
    # repeating x0 would measure a key lookup rather than the warm equilibrium
    # plus implicit-derivative path used by an optimizer.  Use deterministic,
    # distinct, in-bounds steps after compilation instead.
    direction = np.linspace(-0.5, 0.5, problem.x0.size)
    direction /= np.linalg.norm(direction)
    warm = []
    for repeat in range(args.repeats):
        point = problem.x0 + (repeat + 1) * 1.0e-5 * problem.scales * direction
        started = time.perf_counter()
        problem.value_and_grad(point)
        warm.append(time.perf_counter() - started)

    report = {
        "schema": "vmex.qa-optimization-startup/1",
        "command": (
            "JAX_ENABLE_X64=1 python benchmarks/qa_optimization_startup.py "
            f"--lane {args.lane} --max-mode {args.max_mode} "
            f"--repeats {args.repeats}"
        ),
        "persistent_compilation_cache": False,
        "lane": args.lane,
        "max_mode": args.max_mode,
        "repeats": args.repeats,
        "warm_distinct_points": True,
        "warm_relative_step": 1.0e-5,
        "dofs": int(problem.x0.size),
        "residual_rows": (
            None if evaluation.residual is None else int(evaluation.residual.size)
        ),
        "initial_solve_seconds": initial_solve_seconds,
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "cold_startup_seconds": initial_solve_seconds + build_seconds + compile_seconds,
        "warm_value_gradient_median_seconds": statistics.median(warm),
        "initial_value": float(evaluation.value),
        "initial_gradient_norm": float(np.linalg.norm(evaluation.gradient)),
        "peak_rss_mib": _peak_rss_mib(),
        "input_sha256": file_sha256(DATA),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "vmex": vmex.__version__,
            "jax": _version("jax"),
            "jaxlib": _version("jaxlib"),
            "numpy": np.__version__,
        },
        **git_state(REPO),
        "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
