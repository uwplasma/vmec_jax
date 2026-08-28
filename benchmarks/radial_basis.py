#!/usr/bin/env python
"""Measure cold/warm B-spline evaluation and reverse differentiation.

Run each degree in a fresh process.  The report records exact source and
runtime provenance; machine-specific timings are not committed as defaults.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import platform
from pathlib import Path
import resource
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import vmex
from vmex.core.radial_basis import BSplineBasis

from _provenance import assert_repo_vmex, git_state


REPO = Path(__file__).resolve().parents[1]


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _timed(function, argument):
    started = time.perf_counter()
    result = jax.block_until_ready(function(argument))
    return result, time.perf_counter() - started


def _peak_rss_mib() -> float:
    # Darwin reports bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--elements", type=int, default=64)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--points", type=int, default=257)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    basis = BSplineBasis.clamped(np.linspace(0.0, 1.0, args.elements + 1), degree=args.degree)
    coefficient_nodes = jnp.asarray(basis.collocation_nodes)
    mode_numbers = jnp.arange(1, args.modes + 1)[:, None]
    coefficients = jnp.sin(mode_numbers * coefficient_nodes[None, :])
    points = jnp.linspace(0.0, 1.0, args.points)

    def evaluate(values):
        return jnp.stack([basis.evaluate(values, points, derivative=order) for order in range(3)])

    def objective(values):
        result = evaluate(values)
        return jnp.mean(result**2)

    compiled_evaluate = jax.jit(evaluate)
    compiled_gradient = jax.jit(jax.grad(objective))
    rss_initial = _peak_rss_mib()
    _, cold_evaluate = _timed(compiled_evaluate, coefficients)
    rss_after_evaluate = _peak_rss_mib()
    _, cold_gradient = _timed(compiled_gradient, coefficients)
    rss_after_gradient = _peak_rss_mib()
    warm_evaluate = [_timed(compiled_evaluate, coefficients)[1] for _ in range(args.repeats)]
    warm_gradient = [_timed(compiled_gradient, coefficients)[1] for _ in range(args.repeats)]

    report = {
        "degree": args.degree,
        "elements": args.elements,
        "coefficients": basis.size,
        "modes": args.modes,
        "points": args.points,
        "repeats": args.repeats,
        "cold_evaluate_seconds": cold_evaluate,
        "warm_evaluate_median_seconds": statistics.median(warm_evaluate),
        "cold_gradient_seconds": cold_gradient,
        "warm_gradient_median_seconds": statistics.median(warm_gradient),
        "initial_peak_rss_mib": rss_initial,
        "evaluate_peak_rss_increase_mib": rss_after_evaluate - rss_initial,
        "gradient_peak_rss_increase_mib": rss_after_gradient - rss_after_evaluate,
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
