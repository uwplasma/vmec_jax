#!/usr/bin/env python
"""Measure one structured-chart strong-root correction and certificate."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import platform
import resource
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import jax
import jax.numpy as jnp
import numpy as np

import vmex
from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.polish import (
    build_low_order_preconditioner,
    make_strong_root_runtime,
    make_strong_structured_chart,
    strong_collocation_residual,
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

DATA = REPO / "examples" / "data" / "input.solovev"


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def _collocation_variable_scale(
    residual,
    zero,
    row_count: int,
    probes: int,
) -> np.ndarray:
    if probes == 0:
        return np.ones(np.asarray(zero).shape, dtype=float)
    _, jvp = jax.linearize(residual, jnp.asarray(zero))
    transpose = jax.linear_transpose(jvp, jnp.asarray(zero))
    generator = np.random.default_rng(0)
    column_squared = np.zeros(np.asarray(zero).shape, dtype=float)
    for _ in range(probes):
        probe = generator.choice(np.asarray([-1.0, 1.0]), size=row_count)
        response = np.asarray(transpose(jnp.asarray(probe))[0])
        column_squared += response * response
    column_norm = np.sqrt(column_squared / float(probes))
    column_floor = max(1.0e-8 * float(np.max(column_norm)), 1.0e-12)
    return 1.0 / np.maximum(column_norm, column_floor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ns", type=int, default=11)
    parser.add_argument("--mpol", type=int, default=6)
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--radial-spans", type=int)
    parser.add_argument("--radial-quadrature-order", type=int)
    parser.add_argument("--solve-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--radial-refinement-tolerance", type=float, default=1.0e-3
    )
    parser.add_argument("--max-stages", type=int, default=32)
    parser.add_argument("--max-nonlinear-iterations", type=int, default=80)
    parser.add_argument(
        "--preconditioner",
        choices=("none", "legacy", "mode-block"),
        default="mode-block",
    )
    parser.add_argument("--linear-restart", type=int, default=30)
    parser.add_argument("--linear-max-restarts", type=int, default=20)
    parser.add_argument("--collocation-scale-probes", type=int, default=8)
    parser.add_argument("--no-arclength", action="store_true")
    parser.add_argument("--direct-endpoint", action="store_true")
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="build and certify the initial state without running a correction",
    )
    parser.add_argument(
        "--collocation-least-squares",
        action="store_true",
        help="diagnose the rectangular physical collocation residual with LSMR",
    )
    parser.add_argument(
        "--solvax-least-squares",
        action="store_true",
        help="solve rectangular collocation with JIT-native SOLVAX",
    )
    parser.add_argument(
        "--least-squares-initial-damping", type=float, default=1.0e-3
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.ns < args.degree + 2:
        parser.error("ns must be at least degree + 2")
    if args.radial_spans is not None and args.radial_spans < 1:
        parser.error("radial-spans must be positive")
    if (
        args.radial_quadrature_order is not None
        and args.radial_quadrature_order < 2
    ):
        parser.error("radial-quadrature-order must be at least 2")
    if sum(
        (
            args.direct_endpoint,
            args.diagnostics_only,
            args.collocation_least_squares,
            args.solvax_least_squares,
        )
    ) > 1:
        parser.error(
            "direct-endpoint, diagnostics-only, collocation-least-squares, "
            "and solvax-least-squares are mutually exclusive"
        )
    if args.collocation_scale_probes < 0:
        parser.error("collocation-scale-probes must be nonnegative")
    if args.radial_refinement_tolerance <= 0.0:
        parser.error("radial-refinement-tolerance must be positive")
    if args.least_squares_initial_damping <= 0.0:
        parser.error("least-squares-initial-damping must be positive")

    # Keep ``--help`` usable when this experimental benchmark's SOLVAX
    # dependency is absent or older than the requested algorithm.  Runtime
    # execution still imports and validates the dependency before any solve.
    import solvax
    from solvax import pseudo_transient_continuation

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
        radial_quadrature_order=args.radial_quadrature_order,
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
    elif args.solvax_least_squares:
        try:
            from solvax import (
                LeastSquaresConfig,
                gauss_newton_least_squares,
            )
        except ImportError as error:
            raise RuntimeError(
                "--solvax-least-squares requires the matrix-free least-squares "
                "API from uwplasma/SOLVAX#98"
            ) from error
        initial_collocation = strong_collocation_residual(zero, runtime, chart)
        collocation_scale = max(
            float(jnp.linalg.norm(initial_collocation))
            / np.sqrt(float(initial_collocation.size)),
            1.0e-12,
        )
        physical_residual = jax.jit(
            lambda value: strong_collocation_residual(
                value, runtime, chart
            )
            / collocation_scale
        )
        variable_scale = _collocation_variable_scale(
            physical_residual,
            zero,
            int(initial_collocation.size),
            args.collocation_scale_probes,
        )
        transformed_residual = jax.jit(
            lambda value: physical_residual(jnp.asarray(variable_scale) * value)
        )
        least_squares_config = LeastSquaresConfig(
            rtol=args.solve_tolerance,
            max_steps=args.max_nonlinear_iterations,
            initial_damping=args.least_squares_initial_damping,
            linear_rtol=1.0e-3,
            linear_max_steps=max(
                args.linear_restart * args.linear_max_restarts,
                1,
            ),
        )
        solve_started = time.perf_counter()
        native_result = jax.jit(
            lambda value: gauss_newton_least_squares(
                transformed_residual,
                value,
                config=least_squares_config,
            )
        )(zero)
        jax.block_until_ready(native_result)
        final_vector = jnp.asarray(variable_scale) * native_result.x
        state = _corrected_state(final_vector, runtime, chart)
        final_certificate = certify_strong_force(state)
        independently_certified = bool(
            float(final_certificate.normalized_l2)
            <= args.validation_tolerance
            and float(final_certificate.radial_refinement_difference)
            <= args.radial_refinement_tolerance
            and float(final_certificate.minimum_signed_jacobian) > 0.0
        )
        polish_report = {
            "converged": bool(native_result.converged)
            and independently_certified,
            "termination_reason": (
                "independently-certified"
                if bool(native_result.converged) and independently_certified
                else "solvax-collocation-least-squares"
            ),
            "final_alpha": 1.0,
            "initial_normalized_l2": float(initial_certificate.normalized_l2),
            "final_normalized_l2": float(final_certificate.normalized_l2),
            "continuation_accepted": int(native_result.accepted_steps),
            "continuation_rejected": int(native_result.rejected_steps),
            "nonlinear_iterations": int(native_result.steps),
            "linear_iterations": int(native_result.linear_iterations),
            "residual_evaluations": int(native_result.steps) + 1,
            "arclength_steps": 0,
            "minimum_signed_jacobian": float(
                final_certificate.minimum_signed_jacobian
            ),
            "factor_build_seconds": low_preconditioner.factor_build_seconds,
            "solve_seconds": time.perf_counter() - solve_started,
            "least_squares_cost": float(native_result.cost),
            "least_squares_optimality": float(native_result.gradient_norm),
            "least_squares_initial_optimality": float(
                native_result.history.gradient_norm[0]
            ),
            "least_squares_relative_optimality": float(
                native_result.gradient_norm
                / jnp.maximum(native_result.history.gradient_norm[0], 1.0e-300)
            ),
            "least_squares_success": bool(native_result.converged),
            "least_squares_damping": float(native_result.damping),
            "radial_refinement_tolerance": args.radial_refinement_tolerance,
            "variable_scale_min": float(np.min(variable_scale)),
            "variable_scale_max": float(np.max(variable_scale)),
            "variable_scale_probes": args.collocation_scale_probes,
        }
    elif args.collocation_least_squares:
        from scipy.optimize import least_squares
        from scipy.sparse.linalg import LinearOperator

        initial_collocation = strong_collocation_residual(zero, runtime, chart)
        collocation_scale = max(
            float(jnp.linalg.norm(initial_collocation))
            / np.sqrt(float(initial_collocation.size)),
            1.0e-12,
        )
        residual = jax.jit(
            lambda value: strong_collocation_residual(
                value, runtime, chart
            )
            / collocation_scale
        )
        cache: dict[str, object] = {}
        operator_counts = {"matvec": 0, "rmatvec": 0}

        def linearize(value):
            array = np.asarray(value)
            cached = cache.get("x")
            if cached is None or not np.array_equal(array, cached):
                point = jnp.asarray(array)
                result, jvp = jax.linearize(residual, point)
                transpose = jax.linear_transpose(jvp, point)
                cache.update(
                    x=array.copy(),
                    result=np.asarray(result),
                    jvp=jvp,
                    transpose=transpose,
                )
            return cache

        def scipy_residual(value):
            return linearize(value)["result"]

        def scipy_jacobian(value):
            current = linearize(value)
            jvp = current["jvp"]
            transpose = current["transpose"]

            def matvec(direction):
                operator_counts["matvec"] += 1
                return np.asarray(jvp(jnp.asarray(direction)))

            def rmatvec(cotangent):
                operator_counts["rmatvec"] += 1
                return np.asarray(transpose(jnp.asarray(cotangent))[0])

            return LinearOperator(
                (initial_collocation.size, chart.size),
                matvec=matvec,
                rmatvec=rmatvec,
                dtype=np.asarray(zero).dtype,
            )

        variable_scale = _collocation_variable_scale(
            residual,
            zero,
            int(initial_collocation.size),
            args.collocation_scale_probes,
        )

        callback_certificates = []

        def certificate_callback(intermediate_result):
            candidate = jnp.asarray(intermediate_result.x)
            candidate_state = _corrected_state(candidate, runtime, chart)
            certificate = certify_strong_force(candidate_state)
            callback_certificates.append(certificate)
            if (
                float(certificate.normalized_l2)
                <= args.validation_tolerance
                and float(certificate.radial_refinement_difference)
                <= args.radial_refinement_tolerance
                and float(certificate.minimum_signed_jacobian) > 0.0
            ):
                raise StopIteration

        solve_started = time.perf_counter()
        least_squares_result = least_squares(
            scipy_residual,
            zero,
            jac=scipy_jacobian,
            method="trf",
            tr_solver="lsmr",
            ftol=None,
            xtol=None,
            gtol=args.solve_tolerance,
            x_scale=variable_scale,
            max_nfev=args.max_nonlinear_iterations,
            callback=certificate_callback,
            verbose=0,
        )
        final_vector = jnp.asarray(least_squares_result.x)
        state = _corrected_state(final_vector, runtime, chart)
        final_certificate = certify_strong_force(state)
        independently_certified = bool(
            float(final_certificate.normalized_l2)
            <= args.validation_tolerance
            and float(final_certificate.radial_refinement_difference)
            <= args.radial_refinement_tolerance
            and float(final_certificate.minimum_signed_jacobian) > 0.0
        )
        polish_report = {
            "converged": independently_certified,
            "termination_reason": (
                "independently-certified"
                if independently_certified
                else "collocation-least-squares"
            ),
            "final_alpha": 1.0,
            "initial_normalized_l2": float(initial_certificate.normalized_l2),
            "final_normalized_l2": float(final_certificate.normalized_l2),
            "continuation_accepted": 0,
            "continuation_rejected": 0,
            "nonlinear_iterations": int(least_squares_result.nfev),
            "linear_iterations": operator_counts["matvec"],
            "transpose_iterations": operator_counts["rmatvec"],
            "residual_evaluations": int(least_squares_result.nfev),
            "arclength_steps": 0,
            "minimum_signed_jacobian": float(
                final_certificate.minimum_signed_jacobian
            ),
            "factor_build_seconds": low_preconditioner.factor_build_seconds,
            "solve_seconds": time.perf_counter() - solve_started,
            "least_squares_cost": float(least_squares_result.cost),
            "least_squares_optimality": float(least_squares_result.optimality),
            "least_squares_status": int(least_squares_result.status),
            "least_squares_success": bool(least_squares_result.success),
            "certificate_evaluations": len(callback_certificates) + 1,
            "radial_refinement_tolerance": args.radial_refinement_tolerance,
            "variable_scale_min": float(np.min(variable_scale)),
            "variable_scale_max": float(np.max(variable_scale)),
            "variable_scale_probes": args.collocation_scale_probes,
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
    solvax_source = None
    if args.solvax_least_squares:
        solvax_module = Path(solvax.__file__).resolve()
        solvax_repo = solvax_module.parents[2]
        if (solvax_repo / ".git").exists():
            source_state = git_state(solvax_repo)
            solvax_source = {
                "commit": source_state["measurement_commit"],
                "dirty": source_state["measurement_dirty"],
                "module": solvax_module.relative_to(solvax_repo).as_posix(),
            }
    report = {
        "schema": "vmex.strong-polish-benchmark/1",
        "case": args.input.name.removeprefix("input."),
        "ns": args.ns,
        "mpol": args.mpol,
        "degree": args.degree,
        "radial_spans": args.radial_spans,
        "radial_quadrature_order": args.radial_quadrature_order,
        "full_dofs": runtime.layout.size,
        "physical_dofs": chart.size,
        "direct_endpoint": args.direct_endpoint,
        "diagnostics_only": args.diagnostics_only,
        "collocation_least_squares": args.collocation_least_squares,
        "solvax_least_squares": args.solvax_least_squares,
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
            "radial_profile": {
                "rho": np.asarray(initial_certificate.radial_nodes).tolist(),
                "flux_surface_average_force_density": np.asarray(
                    initial_certificate.flux_surface_average
                ).tolist(),
                "flux_surface_normalized_l2": np.asarray(
                    initial_certificate.flux_surface_normalized_l2
                ).tolist(),
            },
        },
        "final_certificate": {
            "normalized_l2": float(final_certificate.normalized_l2),
            "radial_refinement": float(
                final_certificate.radial_refinement_difference
            ),
            "angular_tail": float(final_certificate.angular_spectral_tail),
            "radial_profile": {
                "rho": np.asarray(final_certificate.radial_nodes).tolist(),
                "flux_surface_average_force_density": np.asarray(
                    final_certificate.flux_surface_average
                ).tolist(),
                "flux_surface_normalized_l2": np.asarray(
                    final_certificate.flux_surface_normalized_l2
                ).tolist(),
            },
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
        "validation_tolerance": args.validation_tolerance,
        "radial_refinement_tolerance": args.radial_refinement_tolerance,
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "vmex": vmex.__version__,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "solvax": solvax.__version__,
        },
        "solvax_source": solvax_source,
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
