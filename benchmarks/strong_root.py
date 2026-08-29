#!/usr/bin/env python
"""Measure the square strong-root residual, JVP, rank, and peak memory."""

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

import jax
import jax.numpy as jnp
import numpy as np

import vmex
from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.radial_basis import BSplineBasis
from vmex.core.polish import (
    build_low_order_preconditioner,
    make_strong_physical_chart,
    make_strong_root_runtime,
    make_strong_structured_chart,
    strong_physical_residual,
    strong_root_rank,
    strong_root_residual,
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


def _timed(function, argument):
    started = time.perf_counter()
    result = jax.block_until_ready(function(argument))
    return result, time.perf_counter() - started


def _dominant_components(runtime, vector, *, limit=8):
    correction = runtime.layout.unpack(jnp.asarray(vector))
    components = []
    for field in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
        values = np.asarray(getattr(correction, field))
        for mode, coefficient_norm in enumerate(np.linalg.norm(values, axis=1)):
            components.append(
                {
                    "field": field,
                    "m": int(runtime.native.m[mode]),
                    "n": int(runtime.native.n[mode]),
                    "coefficient_l2": float(coefficient_norm),
                }
            )
    return sorted(
        components,
        key=lambda item: item["coefficient_l2"],
        reverse=True,
    )[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON after provenance is captured instead of using stdout",
    )
    parser.add_argument("--ns", type=int, default=5)
    parser.add_argument("--mpol", type=int, default=3)
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--radial-spans",
        type=int,
        help="explicit uniform high-order spline spans",
    )
    parser.add_argument(
        "--physical-chart",
        action="store_true",
        help="factor only the linear gauge operator and diagnose its reduced root",
    )
    parser.add_argument(
        "--structured-chart",
        action="store_true",
        help="use the O(n)-storage cylindrical-radial physical chart",
    )
    parser.add_argument(
        "--skip-full-rank",
        action="store_true",
        help="skip the diagnostic full-gauge dense Jacobian and SVD",
    )
    parser.add_argument(
        "--chart-balance-probes",
        type=int,
        default=8,
        help="fixed Rademacher probes for physical-chart equilibration",
    )
    args = parser.parse_args()
    if args.ns < args.degree + 2:
        parser.error("ns must be at least degree + 2")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.radial_spans is not None and args.radial_spans < 1:
        parser.error("radial-spans must be positive")
    if args.chart_balance_probes < 1:
        parser.error("chart-balance-probes must be positive")

    inp = VmecInput.from_file(args.input).change_resolution(
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
    config = implicit.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    params = implicit.params_from_input(inp)
    state, mask = implicit.solve_implicit_with_aux(params, config)
    legacy_runtime = implicit.runtime_from_params(params, config)
    radial_basis = (
        None
        if args.radial_spans is None
        else BSplineBasis.clamped(
            np.linspace(0.0, 1.0, args.radial_spans + 1),
            degree=args.degree,
            quadrature_order=args.degree + 3,
        )
    )
    native = lift_high_order_state(
        state,
        legacy_runtime,
        radial_basis=radial_basis,
        degree=args.degree,
    )
    adapter = build_low_order_preconditioner(
        native,
        params,
        config,
        state,
        mask,
        probe_chunk_size=4,
    )

    rss_before_runtime = _peak_rss_mib()
    started = time.perf_counter()
    runtime = make_strong_root_runtime(
        native,
        adapter,
        mask,
        balance_full_root=not args.structured_chart,
    )
    runtime_build_seconds = time.perf_counter() - started
    rss_after_runtime = _peak_rss_mib()
    zero = jnp.zeros((runtime.layout.size,), dtype=jnp.float64)
    initial_low_norm = float(jnp.linalg.norm(strong_root_residual(zero, runtime, 0.0)))
    physical_direction = jnp.linspace(-0.2, 0.3, runtime.layout.size)
    direction = physical_direction / jnp.asarray(runtime.coordinate_scale)
    residual = jax.jit(lambda value: strong_root_residual(value, runtime, 1.0))
    jvp = jax.jit(
        lambda value: jax.jvp(residual, (value,), (direction,))[1]
    )
    initial, cold_residual = _timed(residual, zero)
    rss_after_residual = _peak_rss_mib()
    tangent, cold_jvp = _timed(jvp, zero)
    rss_after_jvp = _peak_rss_mib()
    warm_residual = [_timed(residual, zero)[1] for _ in range(args.repeats)]
    warm_jvp = [_timed(jvp, zero)[1] for _ in range(args.repeats)]
    rank = None
    rank_seconds = None
    singular_values = None
    if not args.skip_full_rank:
        started = time.perf_counter()
        rank, singular_values = strong_root_rank(runtime, relative_tolerance=1.0e-8)
        rank_seconds = time.perf_counter() - started

    physical_chart_report = None
    if args.physical_chart or args.structured_chart:
        rss_before_chart = _peak_rss_mib()
        if args.structured_chart:
            chart = make_strong_structured_chart(
                runtime,
                balance_probes=args.chart_balance_probes,
            )
        else:
            chart = make_strong_physical_chart(runtime)
        rss_after_chart = _peak_rss_mib()
        physical_zero = jnp.zeros((chart.size,), dtype=jnp.float64)
        physical_residual = jax.jit(
            lambda value: strong_physical_residual(value, runtime, chart, 1.0)
        )
        _, physical_cold = _timed(physical_residual, physical_zero)
        physical_warm = [
            _timed(physical_residual, physical_zero)[1]
            for _ in range(args.repeats)
        ]
        started = time.perf_counter()
        physical_jacobian = jax.jacfwd(physical_residual)(physical_zero)
        physical_jacobian.block_until_ready()
        physical_left, physical_singular, physical_right = np.linalg.svd(
            np.asarray(physical_jacobian),
            full_matrices=False,
        )
        physical_rank_seconds = time.perf_counter() - started
        physical_rank = int(
            np.sum(physical_singular > 1.0e-8 * physical_singular[0])
        )
        weakest_triplet_residual = np.linalg.norm(
            np.asarray(physical_jacobian) @ physical_right[-1]
            - physical_singular[-1] * physical_left[:, -1]
        ) / max(
            physical_singular[0],
            np.finfo(physical_singular.dtype).tiny,
        )
        physical_chart_report = {
            "kind": "structured" if args.structured_chart else "dense-nullspace",
            "svd_backend": "numpy.linalg.svd",
            "free_dofs": chart.size,
            "gauge_rank": chart.gauge_rank,
            "chart_build_seconds": chart.build_seconds,
            "chart_peak_rss_increase_mib": rss_after_chart - rss_before_chart,
            "coordinate_scale_range": [
                float(jnp.min(chart.coordinate_scale)),
                float(jnp.max(chart.coordinate_scale)),
            ],
            "equation_scale_range": [
                float(jnp.min(chart.equation_scale)),
                float(jnp.max(chart.equation_scale)),
            ],
            "cold_residual_seconds": physical_cold,
            "warm_residual_median_seconds": statistics.median(physical_warm),
            "rank": physical_rank,
            "rank_seconds": physical_rank_seconds,
            "largest_singular_value": float(physical_singular[0]),
            "smallest_singular_value": float(physical_singular[-1]),
            "condition_number": float(
                physical_singular[0] / physical_singular[-1]
            ),
            "weakest_triplet_relative_residual": float(
                weakest_triplet_residual
            ),
            "weakest_right_components": _dominant_components(
                runtime,
                chart.lift(jnp.asarray(physical_right[-1])),
            ),
            "weakest_left_components": _dominant_components(
                runtime,
                jnp.asarray(chart.equation_basis)
                @ (jnp.asarray(chart.equation_scale) * physical_left[:, -1]),
            ),
        }

    step = 2.0e-5
    finite_difference = (residual(step * direction) - residual(-step * direction)) / (
        2.0 * step
    )
    jvp_relative_error = float(
        jnp.linalg.norm(tangent - finite_difference)
        / jnp.maximum(jnp.linalg.norm(finite_difference), jnp.finfo(jnp.float64).tiny)
    )
    report = {
        "case": args.input.name.removeprefix("input."),
        "ns": args.ns,
        "mpol": args.mpol,
        "degree": args.degree,
        "free_dofs": runtime.layout.size,
        "solve_grid": [
            int(runtime.radial_nodes.size),
            int(runtime.theta.size),
            int(runtime.zeta.size),
        ],
        "repeats": args.repeats,
        "compilation_cache": os.environ.get(
            "VMEX_COMPILATION_CACHE", "default"
        ),
        "runtime_build_seconds": runtime_build_seconds,
        "runtime_peak_rss_increase_mib": rss_after_runtime - rss_before_runtime,
        "cold_residual_seconds": cold_residual,
        "warm_residual_median_seconds": statistics.median(warm_residual),
        "residual_peak_rss_increase_mib": rss_after_residual - rss_after_runtime,
        "cold_jvp_seconds": cold_jvp,
        "warm_jvp_median_seconds": statistics.median(warm_jvp),
        "jvp_peak_rss_increase_mib": rss_after_jvp - rss_after_residual,
        "initial_scaled_residual_norm": float(jnp.linalg.norm(initial)),
        "initial_low_residual_norm": initial_low_norm,
        "operator_balance": float(runtime.operator_balance),
        "coordinate_scale_range": [
            float(jnp.min(runtime.coordinate_scale)),
            float(jnp.max(runtime.coordinate_scale)),
        ],
        "equation_scale_range": [
            float(jnp.min(runtime.equation_scale)),
            float(jnp.max(runtime.equation_scale)),
        ],
        "strong_block_sign": np.asarray(runtime.strong_block_sign).tolist(),
        "jvp_relative_error": jvp_relative_error,
        "rank": rank,
        "rank_seconds": rank_seconds,
        "largest_singular_value": (
            None if singular_values is None else float(singular_values[0])
        ),
        "smallest_singular_value": (
            None if singular_values is None else float(singular_values[-1])
        ),
        "condition_number": (
            None
            if singular_values is None
            else float(singular_values[0] / singular_values[-1])
        ),
        "physical_chart": physical_chart_report,
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
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
