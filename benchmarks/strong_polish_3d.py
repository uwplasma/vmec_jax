#!/usr/bin/env python
"""Probe the production 3-D strong-force polish on a bundled toroidal deck.

This mirrors :func:`vmex.core.polish_driver.polish_legacy_solution` (the
``solve_file(..., polish=True)`` path) step by step, but exposes the polish
controls on the command line and records the diagnostics that the driver
deliberately omits: the per-step Gauss--Newton history, the projection
diagnostics that split the initial residual into angular/radial content the
correction space can and cannot represent, and the full independent
certificate before and after the correction — for failed attempts too, which
the driver's fail path discards.  The solve itself runs through the same
jitted module lanes as production (``_gauss_newton_polish_lane``), so cache
behavior and results match ``solve_file`` bit for bit at equal settings.
``benchmarks/strong_polish.py`` remains the axisymmetric (ntor=0) harness;
this one keeps the deck's toroidal mode table and multigrid ladder.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
import numpy as np

from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_multigrid
from vmex.core.polish import (
    build_low_order_preconditioner,
    make_strong_root_runtime,
    make_strong_structured_chart,
    strong_collocation_residual,
    strong_projection_diagnostics,
)
from vmex.core.polish_driver import (
    PolishConfig,
    _collocation_variable_scale,
    _corrected_state,
    _gauss_newton_polish_lane,
    _PolishProgress,
    _polish_progress,
)
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import certify_strong_force, lift_high_order_state

from _provenance import git_state


def _phase(message: str) -> None:
    """Timestamped, flushed phase marker so an OOM kill names its phase."""

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _certificate_dict(certificate) -> dict:
    fields = (
        "absolute_l2",
        "normalized_l2",
        "normalized_p99",
        "normalized_linf",
        "radial_normalized_l2",
        "helical_normalized_l2",
        "near_axis_l2",
        "bulk_l2",
        "edge_l2",
        "angular_spectral_tail",
        "radial_refinement_difference",
        "minimum_signed_jacobian",
        "boundary_residual",
        "gauge_residual",
    )
    return {name: float(np.asarray(getattr(certificate, name))) for name in fields}


def _diagnostics_dict(diagnostics) -> dict:
    return {
        name: float(np.asarray(value))
        for name, value in diagnostics._asdict().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mpol", type=int, help="override the deck MPOL (new modes seed at zero)"
    )
    parser.add_argument(
        "--ntor", type=int, help="override the deck NTOR (new modes seed at zero)"
    )
    parser.add_argument(
        "--ns", type=int, help="replace the deck NS ladder with one final stage"
    )
    parser.add_argument("--ftol", type=float, help="override the final-stage FTOL")
    parser.add_argument("--niter", type=int, help="override the final-stage NITER")
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=3)
    parser.add_argument("--radial-spans", type=int)
    parser.add_argument("--radial-quadrature-order", type=int)
    parser.add_argument("--radial-refinement-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-nonlinear-iterations", type=int, default=80)
    parser.add_argument("--linear-restart", type=int, default=30)
    parser.add_argument("--linear-max-restarts", type=int, default=20)
    parser.add_argument("--collocation-scale-probes", type=int, default=8)
    parser.add_argument("--least-squares-initial-damping", type=float, default=1.0e-3)
    args = parser.parse_args()

    from solvax import LeastSquaresConfig

    started = time.perf_counter()
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
    if args.ftol is not None:
        ftol_array = np.asarray(inp.ftol_array, dtype=float).copy()
        ftol_array[-1] = args.ftol
        inp = dataclasses.replace(inp, ftol_array=ftol_array)
    if args.niter is not None:
        niter_array = np.asarray(inp.niter_array, dtype=int).copy()
        niter_array[-1] = args.niter
        inp = dataclasses.replace(inp, niter_array=niter_array)

    _phase(
        f"legacy solve start: mpol={int(inp.mpol)} ntor={int(inp.ntor)} "
        f"ns={np.asarray(inp.ns_array).tolist()}"
    )
    result = solve_multigrid(inp, polish_force_balance=False)
    legacy_seconds = time.perf_counter() - started
    _phase(f"legacy solve done in {legacy_seconds:.0f}s fsqr={float(result.fsqr):.2e}")
    ns = int(np.asarray(inp.ns_array)[-1])

    # polish_legacy_solution, instrumented.
    config = PolishConfig(
        tolerance=args.tolerance,
        validation_tolerance=args.validation_tolerance,
        radial_degree=args.degree,
        radial_spans=args.radial_spans,
        radial_quadrature_order=args.radial_quadrature_order,
        radial_refinement_tolerance=args.radial_refinement_tolerance,
        max_nonlinear_iterations=args.max_nonlinear_iterations,
        linear_restart=args.linear_restart,
        linear_max_restarts=args.linear_max_restarts,
        collocation_scale_probes=args.collocation_scale_probes,
        least_squares_initial_damping=args.least_squares_initial_damping,
        fail_policy="return_unpolished",
    )
    lift_started = time.perf_counter()
    implicit_config = implicit.make_config(inp, ns=ns, lconm1=True, multigrid=False)
    params = implicit.params_from_input(inp)
    legacy_runtime = implicit.runtime_from_params(params, implicit_config)
    dof_mask = implicit._dof_mask(result.state, legacy_runtime, implicit_config)
    refined_state = implicit._refined_state(
        implicit_config, params, result.state, dof_mask
    )
    _phase("legacy state refined")
    radial_basis = (
        None
        if config.radial_spans is None
        else BSplineBasis.clamped(
            np.linspace(0.0, 1.0, config.radial_spans + 1),
            degree=config.radial_degree,
            quadrature_order=(
                config.radial_degree + 3
                if config.radial_quadrature_order is None
                else config.radial_quadrature_order
            ),
        )
    )
    native = lift_high_order_state(
        refined_state,
        legacy_runtime,
        radial_basis=radial_basis,
        degree=config.radial_degree,
    )
    _phase("native lift done; certifying initial state")
    initial_certificate = certify_strong_force(native)
    _phase(
        "initial certificate done: "
        f"L2={float(initial_certificate.normalized_l2):.4e}"
    )
    low_preconditioner = build_low_order_preconditioner(
        native, params, implicit_config, refined_state, dof_mask,
        probe_chunk_size=4,
    )
    _phase("low-order preconditioner built")
    runtime = make_strong_root_runtime(
        native,
        low_preconditioner,
        dof_mask,
        balance_full_root=False,
        radial_quadrature_order=config.radial_quadrature_order,
    )
    _phase("strong-root runtime built")
    chart = make_strong_structured_chart(runtime)
    _phase(f"structured chart built: size={int(chart.size)}")
    zero = jnp.zeros((int(chart.size),), dtype=jnp.asarray(native.R_cos).dtype)
    initial_diagnostics = strong_projection_diagnostics(
        np.zeros((int(chart.size),)), runtime, chart
    )
    _phase("initial projection diagnostics done")
    setup_seconds = time.perf_counter() - lift_started

    # polish_collocation_least_squares internals, through the production lanes.
    polish_started = time.perf_counter()
    initial_collocation = strong_collocation_residual(zero, runtime, chart)
    collocation_scale = max(
        float(jnp.linalg.norm(initial_collocation))
        / np.sqrt(float(initial_collocation.size)),
        1.0e-12,
    )
    collocation_scale_array = jnp.asarray(collocation_scale)
    variable_scale = _collocation_variable_scale(
        runtime,
        chart,
        collocation_scale_array,
        zero,
        int(initial_collocation.size),
        config.collocation_scale_probes,
    )
    variable_scale_array = jnp.asarray(variable_scale)
    _phase("collocation and variable scales done; entering Gauss-Newton")
    least_squares_config = LeastSquaresConfig(
        rtol=config.tolerance,
        max_steps=config.max_nonlinear_iterations,
        initial_damping=config.least_squares_initial_damping,
        linear_rtol=1.0e-3,
        linear_max_steps=max(config.linear_restart * config.linear_max_restarts, 1),
    )
    # A production-resolution Gauss-Newton phase runs for hours inside one
    # lax.while_loop.  Route the driver's live heartbeat through the same
    # timestamped phase stamps so a long probe run stays attributable, and
    # so a run that has to be killed still leaves its progress in the log.
    reporter = _PolishProgress(
        lambda text, end="": _phase(text.strip()),
        product_budget=int(least_squares_config.max_steps)
        * int(least_squares_config.linear_max_steps),
    )
    with _polish_progress(reporter):
        solution = _gauss_newton_polish_lane(
            zero, runtime, chart, variable_scale_array,
            collocation_scale_array, least_squares_config, progress=True,
        )
        jax.block_until_ready(solution)
    polish_seconds = time.perf_counter() - polish_started
    _phase(f"Gauss-Newton done in {polish_seconds:.0f}s; certifying final state")
    vector = variable_scale_array * solution.x
    state = _corrected_state(vector, runtime, chart)
    final_certificate = certify_strong_force(state)
    _phase("final certificate done")
    certified = bool(
        float(final_certificate.normalized_l2) <= config.certificate_tolerance
        and float(final_certificate.radial_refinement_difference)
        <= config.radial_refinement_tolerance
        and float(final_certificate.minimum_signed_jacobian) > 0.0
    )
    steps = int(solution.steps)
    final_diagnostics = strong_projection_diagnostics(vector, runtime, chart)

    payload = {
        "schema": "vmex.strong-polish-3d-probe/1",
        "git": git_state(REPO),
        "argv": sys.argv[1:],
        "deck": str(args.input),
        "resolution": {
            "mpol": int(inp.mpol),
            "ntor": int(inp.ntor),
            "ns": ns,
            "nfp": int(inp.nfp),
            "chart_size": int(chart.size),
            "layout_size": int(runtime.layout.size),
            "radial_basis_size": int(native.radial_basis.size),
            "collocation": {
                "radial_nodes": int(np.asarray(runtime.radial_nodes).size),
                "ntheta": int(np.asarray(runtime.theta).size),
                "nzeta": int(np.asarray(runtime.zeta).size),
            },
        },
        "legacy": {
            "seconds": legacy_seconds,
            "fsqr": float(result.fsqr),
            "fsqz": float(result.fsqz),
            "fsql": float(result.fsql),
            "iterations": int(result.iterations),
        },
        "config": {
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
        },
        "setup_seconds": setup_seconds,
        "polish_seconds": polish_seconds,
        "certified": certified,
        "initial_certificate": _certificate_dict(initial_certificate),
        "final_certificate": _certificate_dict(final_certificate),
        "initial_projection": _diagnostics_dict(initial_diagnostics),
        "final_projection": _diagnostics_dict(final_diagnostics),
        "solver": {
            "steps": steps,
            "accepted_steps": int(solution.accepted_steps),
            "rejected_steps": int(solution.rejected_steps),
            "linear_iterations": int(solution.linear_iterations),
            "converged": bool(solution.converged),
            "cost": float(solution.cost),
            "gradient_norm": float(solution.gradient_norm),
            "damping": float(solution.damping),
            "collocation_scale": collocation_scale,
            "variable_scale_min": float(np.min(variable_scale)),
            "variable_scale_max": float(np.max(variable_scale)),
        },
        "history": {
            "cost": np.asarray(solution.history.cost)[: steps + 1].tolist(),
            "gradient_norm": np.asarray(solution.history.gradient_norm)[
                : steps + 1
            ].tolist(),
            "damping": np.asarray(solution.history.damping)[: steps + 1].tolist(),
            "ratio": np.asarray(solution.history.ratio)[:steps].tolist(),
            "accepted": np.asarray(solution.history.accepted)[:steps].tolist(),
            "linear_iterations": np.asarray(solution.history.linear_iterations)[
                :steps
            ].tolist(),
        },
        "total_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload["solver"], indent=1, sort_keys=True))
    print(
        f"certified={certified} initial → final normalized L2: "
        f"{payload['initial_certificate']['normalized_l2']:.4e} → "
        f"{payload['final_certificate']['normalized_l2']:.4e} "
        f"(refinement {payload['final_certificate']['radial_refinement_difference']:.3e}, "
        f"min sqrt(g) {payload['final_certificate']['minimum_signed_jacobian']:.3e})"
    )


if __name__ == "__main__":
    main()
