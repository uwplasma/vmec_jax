#!/usr/bin/env python
"""Measure cold/warm pointwise strong force, gradients, and peak memory."""

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
from vmex.core.radial_basis import BSplineBasis, evaluate_regularized_mode
from vmex.core.strong_force import HighOrderEquilibriumState, evaluate_strong_force

from _provenance import assert_repo_vmex, git_state

REPO = Path(__file__).resolve().parents[1]


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


def _state(degree: int, elements: int) -> HighOrderEquilibriumState:
    basis = BSplineBasis.clamped(np.linspace(0.0, 1.0, elements + 1), degree=degree)
    m = np.asarray([0, 1, 1, 2, 2])
    n = np.asarray([0, 0, 1, -1, 1])
    rng = np.random.default_rng(7)
    coefficients = 0.01 * rng.standard_normal((m.size, basis.size))
    R_cos = coefficients.copy()
    R_cos[0] += 10.0
    R_cos[1] += 1.0
    Z_sin = coefficients.copy()
    Z_sin[1] -= 1.0
    zeros = np.zeros_like(coefficients)
    return HighOrderEquilibriumState(
        radial_basis=basis,
        m=m,
        n=n,
        nfp=2,
        R_cos=jnp.asarray(R_cos),
        R_sin=jnp.asarray(zeros),
        Z_cos=jnp.asarray(zeros),
        Z_sin=jnp.asarray(Z_sin),
        L_cos=jnp.asarray(zeros),
        L_sin=jnp.asarray(0.01 * coefficients),
        phipf=jnp.full((basis.size,), 0.5),
        chipf=jnp.full((basis.size,), 0.05),
        pressure=jnp.linspace(2.0e4, 0.0, basis.size),
        jacobian_sign=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--elements", type=int, default=8)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    state = _state(args.degree, args.elements)
    rho = jnp.linspace(0.05, 0.98, args.points)
    theta = jnp.mod(jnp.arange(args.points) * 1.618, 2.0 * jnp.pi)
    zeta = jnp.mod(jnp.arange(args.points) * 2.414, 2.0 * jnp.pi)

    def evaluate(candidate):
        return evaluate_strong_force(candidate, rho, theta, zeta).force

    def objective(candidate):
        return jnp.mean(evaluate(candidate) ** 2)

    compiled_evaluate = jax.jit(evaluate)
    compiled_gradient = jax.jit(jax.grad(objective))
    rss_initial = _peak_rss_mib()
    _, cold_evaluate = _timed(compiled_evaluate, state)
    rss_after_evaluate = _peak_rss_mib()
    _, cold_gradient = _timed(compiled_gradient, state)
    rss_after_gradient = _peak_rss_mib()
    warm_evaluate = [_timed(compiled_evaluate, state)[1] for _ in range(args.repeats)]
    warm_gradient = [_timed(compiled_gradient, state)[1] for _ in range(args.repeats)]
    accuracy_points = jnp.linspace(1.0e-6, 1.0, 2001)
    radial_basis = state.radial_basis
    exp_coefficients = radial_basis.fit(jnp.exp(jnp.asarray(radial_basis.collocation_nodes)))
    second_derivative = evaluate_regularized_mode(
        radial_basis, exp_coefficients, accuracy_points, 2, derivative=2
    )
    exact_second_derivative = jnp.exp(accuracy_points) * (
        2.0 + 10.0 * accuracy_points + 4.0 * accuracy_points**2
    )
    radial_error = float(jnp.sqrt(jnp.mean((second_derivative - exact_second_derivative) ** 2)))

    report = {
        "degree": args.degree,
        "elements": args.elements,
        "radial_coefficients": state.radial_basis.size,
        "fourier_modes": int(state.m.size),
        "points": args.points,
        "repeats": args.repeats,
        "cold_evaluate_seconds": cold_evaluate,
        "warm_evaluate_median_seconds": statistics.median(warm_evaluate),
        "cold_gradient_seconds": cold_gradient,
        "warm_gradient_median_seconds": statistics.median(warm_gradient),
        "radial_second_derivative_l2_error": radial_error,
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
