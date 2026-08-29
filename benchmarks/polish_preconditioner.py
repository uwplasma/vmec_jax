#!/usr/bin/env python
"""Measure the reusable high/low raw-block preconditioner adapter."""

from __future__ import annotations

import argparse
import dataclasses
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
from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.polish import build_low_order_preconditioner
from vmex.core.radial_basis import BSplineBasis
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


def _norm(value) -> float:
    return float(
        jnp.sqrt(
            sum(jnp.vdot(leaf, leaf).real for leaf in jax.tree.leaves(value))
        )
    )


def _dot(left, right) -> float:
    return float(
        sum(
            jnp.vdot(a, b).real
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
        )
    )


def _timed(function, argument):
    started = time.perf_counter()
    result = jax.block_until_ready(function(argument))
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, default=5)
    parser.add_argument("--mpol", type=int, default=3)
    parser.add_argument("--ntor", type=int, default=0)
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.ns < args.degree + 2:
        parser.error("ns must be at least degree + 2")

    inp = VmecInput.from_file(DATA).change_resolution(
        mpol=args.mpol,
        ntor=args.ntor,
        ntheta=max(12, 2 * args.mpol + 4),
        nzeta=max(4, 2 * args.ntor + 1),
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
    runtime = implicit.runtime_from_params(params, config)
    native = lift_high_order_state(
        state,
        runtime,
        radial_basis=BSplineBasis.clamped(
            np.linspace(0.0, 1.0, args.ns - args.degree + 1),
            degree=args.degree,
            quadrature_order=args.degree + 3,
        ),
        degree=args.degree,
    )

    rss_before_factor = _peak_rss_mib()
    adapter = build_low_order_preconditioner(
        native,
        params,
        config,
        state,
        mask,
        probe_chunk_size=4,
    )
    rss_after_factor = _peak_rss_mib()

    zero = adapter.transfer.zeros_high(jnp.float64)
    leaves, structure = jax.tree.flatten(zero)
    keys = jax.random.split(jax.random.PRNGKey(19), len(leaves))
    rhs = jax.tree.unflatten(
        structure,
        [jax.random.normal(key, leaf.shape, leaf.dtype) for key, leaf in zip(keys, leaves)],
    )
    second_rhs = jax.tree.map(lambda value: 0.7 * value + 0.1, rhs)
    apply = jax.jit(adapter.apply)
    apply_transpose = jax.jit(adapter.apply_transpose)
    forward, cold_forward = _timed(apply, rhs)
    transpose, cold_transpose = _timed(apply_transpose, second_rhs)
    warm_forward = [_timed(apply, rhs)[1] for _ in range(args.repeats)]
    warm_transpose = [_timed(apply_transpose, second_rhs)[1] for _ in range(args.repeats)]

    low_rhs = adapter.transfer.restrict(rhs)
    low_solution = adapter.transfer.restrict(forward)
    low_residual = jax.tree.map(
        jnp.subtract,
        adapter.system.band_operator(low_solution),
        low_rhs,
    )
    roundtrip = adapter.transfer.restrict(adapter.transfer.prolong(low_rhs))
    roundtrip_residual = jax.tree.map(jnp.subtract, roundtrip, low_rhs)
    duality_scale = max(
        abs(_dot(forward, second_rhs)),
        abs(_dot(rhs, transpose)),
        1.0e-300,
    )

    report = {
        "case": "solovev",
        "ns": args.ns,
        "mpol": args.mpol,
        "ntor": args.ntor,
        "degree": args.degree,
        "mnmax": adapter.transfer.mnmax,
        "radial_coefficients": adapter.transfer.nbasis,
        "repeats": args.repeats,
        "factor_build_seconds": adapter.factor_build_seconds,
        "factor_peak_rss_increase_mib": rss_after_factor - rss_before_factor,
        "cold_forward_seconds": cold_forward,
        "warm_forward_median_seconds": statistics.median(warm_forward),
        "cold_transpose_seconds": cold_transpose,
        "warm_transpose_median_seconds": statistics.median(warm_transpose),
        "low_block_relative_residual": _norm(low_residual) / max(_norm(low_rhs), 1.0e-300),
        "transfer_roundtrip_relative_residual": _norm(roundtrip_residual)
        / max(_norm(low_rhs), 1.0e-300),
        "preconditioner_duality_relative_error": abs(
            _dot(forward, second_rhs) - _dot(rhs, transpose)
        )
        / duality_scale,
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
