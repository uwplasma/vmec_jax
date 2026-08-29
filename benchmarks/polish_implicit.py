#!/usr/bin/env python
"""Measure cold/warm tangent and adjoint cost for the strong-root IFT."""

from __future__ import annotations

import argparse
import dataclasses
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import resource
import statistics
import time

# A cold benchmark must not deserialize an executable from an earlier process.
os.environ["VMEX_COMPILATION_CACHE"] = "disabled"

import jax
import jax.numpy as jnp
import numpy as np

import vmex
from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.polish import (
    build_low_order_preconditioner,
    build_strong_mode_block_preconditioner,
    make_strong_root_runtime,
)
from vmex.core.polish_implicit import (
    PolishLinearConfig,
    implicit_polished_state,
    strong_root_adjoint,
    strong_root_tangent,
)
from vmex.core.strong_force import lift_high_order_state

from _provenance import assert_repo_vmex, git_state

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data" / "input.solovev"


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def _tree_dot(left, right) -> float:
    return float(
        sum(
            jnp.vdot(a, b).real
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
        )
    )


def _random_like(value, seed: int):
    leaves, structure = jax.tree.flatten(value)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree.unflatten(
        structure,
        [jax.random.normal(key, leaf.shape, leaf.dtype) for key, leaf in zip(keys, leaves)],
    )


def _timed(function, argument):
    started = time.perf_counter()
    result = jax.block_until_ready(function(argument))
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, default=5)
    parser.add_argument("--mpol", type=int, default=3)
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.ns < args.degree + 2:
        parser.error("ns must be at least degree + 2")

    inp = VmecInput.from_file(DATA).change_resolution(
        mpol=args.mpol,
        ntor=0,
        ntheta=max(12, 2 * args.mpol + 4),
        nzeta=4,
    )
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([args.ns]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )
    legacy_config = implicit.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    params = implicit.params_from_input(inp)
    state, mask = implicit.solve_implicit_with_aux(params, legacy_config)
    legacy_runtime = implicit.runtime_from_params(params, legacy_config)
    native = lift_high_order_state(state, legacy_runtime, degree=args.degree)
    adapter = build_low_order_preconditioner(
        native,
        params,
        legacy_config,
        state,
        mask,
        probe_chunk_size=4,
    )
    runtime = make_strong_root_runtime(native, adapter, mask)
    correction = jnp.zeros((runtime.layout.size,), dtype=jnp.float64)
    block_preconditioner = build_strong_mode_block_preconditioner(
        runtime, correction
    )
    native_tangent = _random_like(native, 41)
    output_cotangent = _random_like(native, 42)
    linear_config = PolishLinearConfig(
        rtol=2.0e-10,
        atol=2.0e-11,
        restart=runtime.layout.size,
        max_restarts=3,
    )

    tangent = jax.jit(
        lambda value: strong_root_tangent(
            runtime,
            correction,
            value,
            config=linear_config,
            preconditioner=block_preconditioner,
        )
    )
    adjoint = jax.jit(
        lambda value: strong_root_adjoint(
            runtime,
            correction,
            value,
            config=linear_config,
            preconditioner=block_preconditioner,
        )
    )

    def objective(value):
        polished = implicit_polished_state(
            value,
            correction,
            runtime,
            linear_config,
            block_preconditioner,
        )
        return sum(
            jnp.vdot(a, b).real
            for a, b in zip(
                jax.tree.leaves(polished),
                jax.tree.leaves(output_cotangent),
                strict=True,
            )
        )

    gradient = jax.jit(jax.grad(objective))
    rss_before = _peak_rss_mib()
    tangent_result, cold_tangent = _timed(tangent, native_tangent)
    rss_after_tangent = _peak_rss_mib()
    adjoint_result, cold_adjoint = _timed(adjoint, output_cotangent)
    rss_after_adjoint = _peak_rss_mib()
    gradient_result, cold_gradient = _timed(gradient, native)
    rss_after_gradient = _peak_rss_mib()
    warm_tangent = [_timed(tangent, native_tangent)[1] for _ in range(args.repeats)]
    warm_adjoint = [_timed(adjoint, output_cotangent)[1] for _ in range(args.repeats)]
    warm_gradient = [_timed(gradient, native)[1] for _ in range(args.repeats)]
    duality_left = _tree_dot(output_cotangent, tangent_result.native_tangent)
    duality_right = _tree_dot(adjoint_result.native_cotangent, native_tangent)
    duality_scale = max(abs(duality_left), abs(duality_right), 1.0e-300)
    gradient_difference = jax.tree.map(
        jnp.subtract, gradient_result, adjoint_result.native_cotangent
    )
    gradient_scale = max(
        abs(_tree_dot(gradient_result, gradient_result)),
        abs(_tree_dot(adjoint_result.native_cotangent, adjoint_result.native_cotangent)),
        1.0e-300,
    )

    report = {
        "schema": "vmex.polish-implicit-benchmark/1",
        "command": (
            "JAX_ENABLE_X64=1 python benchmarks/polish_implicit.py "
            f"--ns {args.ns} --mpol {args.mpol} --degree {args.degree} "
            f"--repeats {args.repeats}"
        ),
        "persistent_compilation_cache": False,
        "case": "solovev-structural-derivative-gate",
        "ns": args.ns,
        "mpol": args.mpol,
        "degree": args.degree,
        "free_dofs": runtime.layout.size,
        "mode_blocks": len(block_preconditioner.indices),
        "mode_block_build_seconds": block_preconditioner.build_seconds,
        "repeats": args.repeats,
        "cold_tangent_seconds": cold_tangent,
        "warm_tangent_median_seconds": statistics.median(warm_tangent),
        "tangent_peak_rss_increase_mib": rss_after_tangent - rss_before,
        "tangent_iterations": int(tangent_result.report.iterations),
        "tangent_residual_norm": float(tangent_result.report.residual_norm),
        "tangent_tolerance": float(tangent_result.report.tolerance),
        "cold_adjoint_seconds": cold_adjoint,
        "warm_adjoint_median_seconds": statistics.median(warm_adjoint),
        "adjoint_peak_rss_increase_mib": rss_after_adjoint - rss_after_tangent,
        "adjoint_iterations": int(adjoint_result.report.iterations),
        "adjoint_residual_norm": float(adjoint_result.report.residual_norm),
        "adjoint_tolerance": float(adjoint_result.report.tolerance),
        "cold_custom_vjp_seconds": cold_gradient,
        "warm_custom_vjp_median_seconds": statistics.median(warm_gradient),
        "custom_vjp_peak_rss_increase_mib": rss_after_gradient - rss_after_adjoint,
        "tangent_adjoint_duality_relative_error": abs(duality_left - duality_right)
        / duality_scale,
        "custom_vjp_relative_squared_error": _tree_dot(
            gradient_difference, gradient_difference
        )
        / gradient_scale,
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "vmex": vmex.__version__,
            "jax": jax.__version__,
            "jaxlib": _version("jaxlib"),
            "numpy": np.__version__,
        },
        **git_state(REPO),
        "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
