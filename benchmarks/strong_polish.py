#!/usr/bin/env python
"""Measure one structured-chart strong-root correction and certificate."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import platform
import resource
import time

import jax
import numpy as np

import vmex
from solvax import pseudo_transient_continuation
from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.polish import (
    build_low_order_preconditioner,
    make_strong_root_runtime,
    make_strong_structured_chart,
    strong_projection_diagnostics,
    strong_physical_residual,
)
from vmex.core.polish_driver import (
    PolishConfig,
    _corrected_state,
    _minimum_signed_jacobian,
    _ptc_config,
    polish_strong_root,
)
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import certify_strong_force, lift_high_order_state

from _provenance import assert_repo_vmex, git_state

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data" / "input.solovev"


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ns", type=int, default=11)
    parser.add_argument("--mpol", type=int, default=6)
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--radial-spans", type=int)
    parser.add_argument("--solve-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-stages", type=int, default=32)
    parser.add_argument("--max-nonlinear-iterations", type=int, default=80)
    parser.add_argument(
        "--preconditioner",
        choices=("none", "legacy", "mode-block"),
        default="mode-block",
    )
    parser.add_argument("--linear-restart", type=int, default=30)
    parser.add_argument("--linear-max-restarts", type=int, default=20)
    parser.add_argument("--no-arclength", action="store_true")
    parser.add_argument("--direct-endpoint", action="store_true")
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="build and certify the initial state without running a correction",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.ns < args.degree + 2:
        parser.error("ns must be at least degree + 2")
    if args.radial_spans is not None and args.radial_spans < 1:
        parser.error("radial-spans must be positive")

    started = time.perf_counter()
    rss_initial = _peak_rss_mib()
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
    implicit_config = implicit.make_config(
        inp,
        ftol=1.0e-10,
        max_iterations=1000,
    )
    params = implicit.params_from_input(inp)
    legacy_state, dof_mask = implicit.solve_implicit_with_aux(
        params, implicit_config
    )
    legacy_runtime = implicit.runtime_from_params(params, implicit_config)
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
        legacy_state,
        legacy_runtime,
        radial_basis=radial_basis,
        degree=args.degree,
    )
    initial_certificate = certify_strong_force(native)
    low_preconditioner = build_low_order_preconditioner(
        native,
        params,
        implicit_config,
        legacy_state,
        dof_mask,
        probe_chunk_size=4,
    )
    runtime = make_strong_root_runtime(
        native,
        low_preconditioner,
        dof_mask,
        balance_full_root=False,
    )
    chart = make_strong_structured_chart(runtime)
    zero = np.zeros((chart.size,), dtype=float)
    initial_projection = strong_projection_diagnostics(zero, runtime, chart)
    setup_seconds = time.perf_counter() - started
    rss_after_setup = _peak_rss_mib()
    polish_config = PolishConfig(
        tolerance=args.solve_tolerance,
        validation_tolerance=args.validation_tolerance,
        max_continuation_stages=args.max_stages,
        max_nonlinear_iterations=args.max_nonlinear_iterations,
        ptc_initial_dtau=1.0e12,
        preconditioner=args.preconditioner,
        linear_restart=args.linear_restart,
        linear_max_restarts=args.linear_max_restarts,
        use_pseudo_arclength=not args.no_arclength,
        fail_policy="return_unpolished",
    )
    if args.diagnostics_only:
        state = native
        final_vector = zero
        final_certificate = initial_certificate
        polish_report = {
            "converged": False,
            "termination_reason": "diagnostics-only",
            "final_alpha": 0.0,
            "initial_normalized_l2": float(initial_certificate.normalized_l2),
            "final_normalized_l2": float(initial_certificate.normalized_l2),
            "continuation_accepted": 0,
            "continuation_rejected": 0,
            "nonlinear_iterations": 0,
            "linear_iterations": 0,
            "residual_evaluations": 0,
            "arclength_steps": 0,
            "minimum_signed_jacobian": float(
                initial_certificate.minimum_signed_jacobian
            ),
            "factor_build_seconds": low_preconditioner.factor_build_seconds,
            "solve_seconds": 0.0,
        }
    elif args.direct_endpoint:
        margin = float(_minimum_signed_jacobian(zero, runtime, chart))
        direct = pseudo_transient_continuation(
            lambda value: strong_physical_residual(value, runtime, chart, 1.0),
            zero,
            admissible=lambda value: (
                _minimum_signed_jacobian(value, runtime, chart) >= 0.1 * margin
            ),
            config=_ptc_config(
                polish_config,
                residual_scale=np.sqrt(float(chart.size)),
            ),
        )
        state = _corrected_state(direct.x, runtime, chart)
        final_vector = direct.x
        final_certificate = certify_strong_force(state)
        polish_report = {
            "converged": bool(direct.converged and direct.linear_converged),
            "termination_reason": "direct-endpoint",
            "final_alpha": 1.0,
            "initial_normalized_l2": float(initial_certificate.normalized_l2),
            "final_normalized_l2": float(final_certificate.normalized_l2),
            "continuation_accepted": 0,
            "continuation_rejected": 0,
            "nonlinear_iterations": int(direct.steps),
            "linear_iterations": int(direct.linear_iterations),
            "residual_evaluations": int(direct.residual_evaluations),
            "arclength_steps": 0,
            "minimum_signed_jacobian": float(
                final_certificate.minimum_signed_jacobian
            ),
            "factor_build_seconds": low_preconditioner.factor_build_seconds,
            "solve_seconds": time.perf_counter() - started - setup_seconds,
        }
    else:
        result = polish_strong_root(
            runtime,
            config=polish_config,
            initial_certificate=initial_certificate,
            chart=chart,
        )
        jax.block_until_ready(result.native_equilibrium)
        final_certificate = result.strong_force
        final_vector = (
            chart.coordinate_basis.T @ result.correction
        ) / chart.coordinate_scale
        polish_report = dataclasses.asdict(result.polish_report)
    final_projection = strong_projection_diagnostics(
        final_vector, runtime, chart
    )
    report = {
        "schema": "vmex.strong-polish-benchmark/1",
        "case": args.input.name.removeprefix("input."),
        "ns": args.ns,
        "mpol": args.mpol,
        "degree": args.degree,
        "radial_spans": args.radial_spans,
        "full_dofs": runtime.layout.size,
        "physical_dofs": chart.size,
        "direct_endpoint": args.direct_endpoint,
        "diagnostics_only": args.diagnostics_only,
        "solve_grid": [
            int(runtime.radial_nodes.size),
            int(runtime.theta.size),
            int(runtime.zeta.size),
        ],
        "setup_seconds": setup_seconds,
        "setup_peak_rss_increase_mib": rss_after_setup - rss_initial,
        "total_seconds": time.perf_counter() - started,
        "total_peak_rss_increase_mib": _peak_rss_mib() - rss_initial,
        "initial_certificate": {
            "normalized_l2": float(initial_certificate.normalized_l2),
            "radial_refinement": float(
                initial_certificate.radial_refinement_difference
            ),
            "angular_tail": float(initial_certificate.angular_spectral_tail),
        },
        "final_certificate": {
            "normalized_l2": float(final_certificate.normalized_l2),
            "radial_refinement": float(
                final_certificate.radial_refinement_difference
            ),
            "angular_tail": float(final_certificate.angular_spectral_tail),
        },
        "projection_consistency": {
            "initial": {
                field: float(value)
                for field, value in initial_projection._asdict().items()
            },
            "final": {
                field: float(value)
                for field, value in final_projection._asdict().items()
            },
        },
        "polish_report": polish_report,
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "vmex": vmex.__version__,
            "jax": jax.__version__,
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
