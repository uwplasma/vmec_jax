#!/usr/bin/env python
"""Price the Gauss--Newton polish phase across the decks polishing ships on.

``POLISH = AUTO`` declines a solve whose predicted wall time exceeds
:attr:`~vmex.core.polish_driver.PolishConfig.auto_budget_seconds`.  That
threshold has to come from the cases the feature is actually shipped and
claimed on, not from a round number, so this runs each deck's polish setup
up to the point where the driver takes its measurement and records what it
found: the collocation size, the chart size, the measured seconds per
Gauss--Newton linear product, and the worst-case wall time the configured
iteration limits allow.

Nothing here runs the Gauss--Newton phase.  That is the point: the whole
question is what the driver can know *before* committing to it.

Usage::

    python benchmarks/polish_cost.py --output benchmarks/polish_cost_<host>.json \\
        --deck input.solovev --deck input.shaped_tokamak_pressure_polished \\
        --deck input.nfp2_QA_smooth_beta
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import vmex  # noqa: E402
from vmex.core import implicit  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.multigrid import solve_multigrid  # noqa: E402
from vmex.core.polish import (  # noqa: E402
    build_low_order_preconditioner,
    make_strong_root_runtime,
    make_strong_structured_chart,
    strong_collocation_residual,
    strong_projection_diagnostics,
)
from vmex.core.polish_driver import (  # noqa: E402
    PolishConfig,
    _collocation_variable_scale,
    _measure_polish_cost,
)
from vmex.core.radial_basis import BSplineBasis  # noqa: E402
from vmex.core.strong_force import certify_strong_force, lift_high_order_state  # noqa: E402

from _provenance import assert_repo_vmex, file_sha256, git_state  # noqa: E402

SCHEMA = "vmex.polish-cost/1"
DECKS = REPO / "examples" / "data"


def _phase(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def price(name: str, config: PolishConfig, *, mpol=None, ntor=None, ns=None) -> dict:
    """Run one deck's polish setup and take the driver's cost measurement."""

    path = DECKS / name
    inp = VmecInput.from_file(path)
    if mpol is not None or ntor is not None:
        inp = inp.change_resolution(
            mpol=mpol if mpol is not None else int(inp.mpol),
            ntor=ntor if ntor is not None else int(inp.ntor),
        )
    if ns is not None:
        inp = dataclasses.replace(
            inp,
            ns_array=np.asarray([int(ns)]),
            ftol_array=np.asarray([float(inp.ftol_array[-1])]),
            niter_array=np.asarray([int(inp.niter_array[-1])]),
        )
    _phase(f"{name}: legacy solve")
    started = time.perf_counter()
    result = solve_multigrid(inp, polish_force_balance=False)
    legacy_seconds = time.perf_counter() - started

    setup_started = time.perf_counter()
    final_ns = int(np.asarray(inp.ns_array)[-1])
    implicit_config = implicit.make_config(
        inp, ns=final_ns, lconm1=True, multigrid=False)
    params = implicit.params_from_input(inp)
    legacy_runtime = implicit.runtime_from_params(params, implicit_config)
    dof_mask = implicit._dof_mask(result.state, legacy_runtime, implicit_config)
    refined_state = implicit._refined_state(
        implicit_config, params, result.state, dof_mask)
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
        degree=config.radial_degree)
    certificate = certify_strong_force(native)
    low_preconditioner = build_low_order_preconditioner(
        native, params, implicit_config, refined_state, dof_mask,
        probe_chunk_size=4)
    runtime = make_strong_root_runtime(
        native, low_preconditioner, dof_mask, balance_full_root=False,
        radial_quadrature_order=config.radial_quadrature_order)
    _phase(f"{name}: building chart")
    chart_started = time.perf_counter()
    chart = make_strong_structured_chart(runtime)
    chart_seconds = time.perf_counter() - chart_started

    zero = jnp.zeros((int(chart.size),), dtype=jnp.asarray(native.R_cos).dtype)
    collocation = strong_collocation_residual(zero, runtime, chart)
    collocation_scale = jnp.asarray(max(
        float(jnp.linalg.norm(collocation)) / np.sqrt(float(collocation.size)),
        1.0e-12,
    ))
    variable_scale = jnp.asarray(_collocation_variable_scale(
        runtime, chart, collocation_scale, zero, int(collocation.size),
        config.collocation_scale_probes))
    setup_seconds = time.perf_counter() - setup_started

    _phase(f"{name}: timing the Gauss-Newton product")
    estimate = _measure_polish_cost(
        zero, runtime, chart, variable_scale, collocation_scale, config)
    projection = strong_projection_diagnostics(
        np.zeros((int(chart.size),)), runtime, chart)
    _phase(
        f"{name}: {estimate.seconds_per_product:.4G} s/product, predicted "
        f"{estimate.predicted_seconds / 3600.0:.2f} h worst case"
    )
    return {
        "deck": name,
        "sha256_prefix": file_sha256(path)[:16],
        "resolution": {
            "nfp": int(inp.nfp), "mpol": int(inp.mpol), "ntor": int(inp.ntor),
            "ns": final_ns,
            "mnmax": int(np.asarray(native.m).size),
            "radial_basis_size": int(native.radial_basis.size),
        },
        "axisymmetric": int(inp.ntor) == 0,
        "legacy_seconds": legacy_seconds,
        "setup_seconds": setup_seconds,
        "chart_seconds": chart_seconds,
        "chart_size": estimate.chart_size,
        "residual_rows": estimate.residual_rows,
        "seconds_per_linear_product": estimate.seconds_per_product,
        "worst_case_products": estimate.products,
        "predicted_solve_seconds": estimate.predicted_seconds,
        "initial_absolute_l2": float(certificate.absolute_l2),
        "initial_unresolved_fraction": float(projection.unresolved_fraction),
        "initial_helical_unresolved_fraction": float(
            projection.helical_unresolved_fraction),
        "initial_sampled_rms": float(projection.sampled_rms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--deck", action="append", default=[],
        help="deck file name under examples/data (repeatable)")
    parser.add_argument("--mpol", type=int)
    parser.add_argument("--ntor", type=int)
    parser.add_argument("--ns", type=int)
    parser.add_argument("--host", default="")
    args = parser.parse_args()

    config = PolishConfig()
    cases = [
        price(name, config, mpol=args.mpol, ntor=args.ntor, ns=args.ns)
        for name in args.deck
    ]
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
        "driver_defaults": {
            "max_nonlinear_iterations": config.max_nonlinear_iterations,
            "linear_restart": config.linear_restart,
            "linear_max_restarts": config.linear_max_restarts,
            "auto_budget_seconds": config.auto_budget_seconds,
        },
        "cases": cases,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for case in cases:
        print(
            f"{case['deck']:<44} chart {case['chart_size']:>6}  "
            f"{case['seconds_per_linear_product']:>9.4G} s/product  "
            f"worst case {case['predicted_solve_seconds'] / 3600.0:>8.2f} h"
        )


if __name__ == "__main__":
    main()
