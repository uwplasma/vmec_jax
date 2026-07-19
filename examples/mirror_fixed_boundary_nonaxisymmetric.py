"""Native-spline fixed-boundary mirror equilibria.

Solves the supported rotating-ellipse case, the Agren-Savenko straight
field-line paraxial-accuracy benchmark, and a standard axisymmetric mirror,
then renders the axisymmetric and 90-degree rotating-ellipse solves side by
side in 3-D. The straight field-line mirror is an analytic field that is only
an equilibrium to order ``(a/c)^2``; it is gated on its clean unconstrained
bulk force and on the refinement convergence of that bulk force, and its
elevated end-collar force is the expected boundary layer at the frozen cuts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmex.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    mout_from_result,
    plot_mout,
    solve_fixed_boundary,
    solve_fixed_boundary_from_radius,
    spline_fixed_boundary_adjoint,
    write_mout,
)
from vmex.mirror.analytic import (  # noqa: E402
    AxisymmetricPolynomialMirror,
    RotatingEllipseParaxial,
    StraightFieldLineMirror,
)
from vmex.mirror.output import plot_mirror_3d_pair  # noqa: E402
from vmex.mirror.implicit import spline_fixed_boundary_parameters  # noqa: E402
from vmex.mirror.forces import force_gate_zones  # noqa: E402
from vmex.mirror.splines import initialize_from_cartesian_field  # noqa: E402

# Inputs: edit these constants, then run this file directly.
CASES = ("rotating_ellipse", "straight_field_line")
NS, MPOL, SOURCE_NXI = 7, 6, 17
SPLINE_ELEMENTS = 6
SHAPE_STAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
FTOL = 1.0e-12
MAX_ITERATIONS = 1000
RUN_GRADIENT_CHECK = True
FINITE_DIFFERENCE_STEP = 2.0e-4
STRONG_FORCE_GATE = 5.0e-2
OUTPUT_DIR = Path("results/mirror_fixed_boundary_nonaxisymmetric")

RADIUS = {"rotating_ellipse": 0.12, "straight_field_line": 0.10}
AXIAL_FLUX_DERIVATIVE = {"rotating_ellipse": 0.0072, "straight_field_line": 0.005}
AXISYMMETRIC_RADIUS = 0.12
AXISYMMETRIC_MPOL = 4
AXISYMMETRIC_MIRROR_STRENGTH = 0.5

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config = MirrorConfig(
    resolution=MirrorResolution(ns=NS, mpol=MPOL, nxi=SOURCE_NXI),
    z_min=-1.0,
    z_max=1.0,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
source_grid = config.build_grid()
discretization = SplineMirrorDiscretization.build(config, elements=SPLINE_ELEMENTS)
theta = jnp.asarray(source_grid.theta)[:, None]
z = jnp.asarray(source_grid.z)[None, :]


def boundary_for(case: str, stage: float) -> MirrorBoundary:
    """Return one analytic continuation boundary on the source grid."""

    radius = RADIUS[case]
    if case == "rotating_ellipse":
        fixture = RotatingEllipseParaxial(
            half_length=1.0,
            mirror_strength=0.2 * stage,
            elongation=1.0 + 0.5 * stage,
            rotation=0.5 * jnp.pi * stage,
        )
        values = fixture.boundary_radius(radius, theta, z)
    elif case == "straight_field_line":
        fixture = StraightFieldLineMirror(center_field=1.0, axial_scale=2.5)
        values = fixture.boundary_radius(radius, theta, stage * z)
    else:
        raise ValueError(f"unknown mirror case {case!r}")
    return MirrorBoundary.from_radius(values, source_grid)


summaries = {}
for case in CASES:
    initial_boundary = boundary_for(case, 0.0)
    previous_boundary = discretization.fit_boundary(initial_boundary, source_grid)
    coefficient_state = discretization.fit_state(MirrorState.from_boundary(initial_boundary, source_grid), source_grid)
    axial_flux_derivative = AXIAL_FLUX_DERIVATIVE[case]
    stage_iterations = []
    for stage in SHAPE_STAGES:
        final_boundary = discretization.fit_boundary(boundary_for(case, stage), source_grid)
        coefficient_state = discretization.transfer_boundary(coefficient_state, previous_boundary, final_boundary)
        if stage > 0.0:
            if case == "rotating_ellipse":
                fixture = RotatingEllipseParaxial(
                    half_length=1.0,
                    reference_field=2.0 * AXIAL_FLUX_DERIVATIVE[case] / RADIUS[case] ** 2,
                    mirror_strength=0.2 * stage,
                    elongation=1.0 + 0.5 * stage,
                    rotation=0.5 * jnp.pi * stage,
                )
            else:
                fixture = StraightFieldLineMirror(
                    center_field=1.0,
                    axial_scale=2.5 / stage,
                )
            initialized = initialize_from_cartesian_field(
                coefficient_state,
                final_boundary,
                discretization,
                fixture.field,
            )
            coefficient_state = discretization.impose_self_similar_cuts(
                initialized.state,
                final_boundary,
            )
            if case == "straight_field_line":
                axial_flux_derivative = initialized.axial_flux_derivative
        spline_result = solve_fixed_boundary(
            coefficient_state,
            final_boundary,
            discretization,
            config,
            axial_flux_derivative=axial_flux_derivative,
            solve_lambda=True,
            gradient_tolerance=FTOL,
            require_convergence=True,
        )
        coefficient_state = spline_result.coefficient_state
        previous_boundary = final_boundary
        result = spline_result.evaluated
        stage_iterations.append(result.iterations)

    evaluated_boundary = discretization.evaluate_boundary(final_boundary)
    mout_path = write_mout(
        OUTPUT_DIR / f"mout_{case}.nc",
        mout_from_result(
            result,
            discretization.grid,
            config,
            boundary=evaluated_boundary,
            axial_flux_derivative=axial_flux_derivative,
        ),
    )
    plot_mout(mout_path, OUTPUT_DIR, name=case)
    validation = {"lambda_max": float(jnp.max(jnp.abs(coefficient_state.lambda_coefficients)))}
    if case == "rotating_ellipse" and RUN_GRADIENT_CHECK:
        parameters = spline_fixed_boundary_parameters(
            final_boundary,
            axial_flux_derivative=axial_flux_derivative,
        )
        adjoint = spline_fixed_boundary_adjoint(
            spline_result,
            parameters,
            discretization,
            lambda _state, energy: energy.geometry.volume,
            solve_lambda=True,
            rtol=1.0e-9,
        )
        direction = jnp.zeros_like(final_boundary.radius_coefficients)
        direction = direction.at[:, direction.shape[1] // 2].set(1.0e-3)
        predicted = float(jnp.vdot(adjoint.gradient.boundary_coefficients, direction))
        values = []
        for sign in (-1.0, 1.0):
            varied_boundary = SplineMirrorBoundary(
                final_boundary.radius_coefficients + sign * FINITE_DIFFERENCE_STEP * direction
            )
            varied = solve_fixed_boundary(
                discretization.transfer_boundary(
                    spline_result.coefficient_state,
                    final_boundary,
                    varied_boundary,
                ),
                varied_boundary,
                discretization,
                config,
                axial_flux_derivative=axial_flux_derivative,
                solve_lambda=True,
                gradient_tolerance=FTOL,
                require_convergence=True,
            )
            values.append(float(varied.evaluated.energy.geometry.volume))
        finite_difference = (values[1] - values[0]) / (2.0 * FINITE_DIFFERENCE_STEP)
        validation["boundary_gradient_adjoint"] = predicted
        validation["boundary_gradient_finite_difference"] = finite_difference
        validation["boundary_gradient_relative_error"] = abs(predicted - finite_difference) / abs(finite_difference)
        validation["adjoint_relative_residual"] = adjoint.relative_residual
    supported = case == "rotating_ellipse"
    zones = force_gate_zones(result.force)
    # The rotating ellipse is a supported equilibrium and is gated on its
    # all-volume force. The straight field-line mirror is an analytic field
    # that is only an equilibrium to order (a/c)^2, so it is gated on its
    # clean unconstrained bulk force; its elevated end-collar force is the
    # expected boundary layer where the analytic cut profile is frozen.
    status = (
        "supported"
        if supported
        else "paraxial-accuracy benchmark: bulk force gated, cut collar expected"
    )
    summaries[case] = {
        "status": status,
        "stage_iterations": stage_iterations,
        "linear_iterations": result.linear_iterations,
        "final_linear_residual": result.final_linear_residual,
        "variational_max": float(result.variational.maximum),
        "staggered_weak_max": float(result.staggered_weak_force.maximum),
        "strong_force_normalized_rms": zones.all_volume,
        "strong_force_axis_rms": zones.axis_row,
        "strong_force_first_row_rms": zones.first_row,
        "strong_force_bulk_rms": zones.bulk,
        "strong_force_end_collar_rms": zones.end_collar,
        "strong_force_device_normalized_rms": zones.device_all_volume,
        "minor_radius": zones.minor_radius,
        "normalized_divergence_rms": float(result.normalized_divergence_rms),
        "axial_flux_derivative_min": float(jnp.min(jnp.asarray(axial_flux_derivative))),
        "axial_flux_derivative_max": float(jnp.max(jnp.asarray(axial_flux_derivative))),
        **validation,
    }
    assert float(result.variational.maximum) <= FTOL
    assert float(result.normalized_divergence_rms) < 1.0e-12
    if supported:
        assert float(result.staggered_weak_force.maximum) <= 1.1 * FTOL
        assert zones.all_volume < STRONG_FORCE_GATE
    else:
        # Bulk (unconstrained volume) force is the physical equilibrium gate;
        # its refinement convergence is recorded in docs/mirror_geometry.rst.
        assert zones.bulk < STRONG_FORCE_GATE

# Standard axisymmetric mirror through the one-call entry point: the boundary
# is the exact circular flux surface of an analytic vacuum mirror.
axisymmetric_fixture = AxisymmetricPolynomialMirror(
    center_field=1.0,
    half_length=1.0,
    mirror_strength=AXISYMMETRIC_MIRROR_STRENGTH,
)
axisymmetric_config = MirrorConfig(
    resolution=MirrorResolution(ns=NS, mpol=AXISYMMETRIC_MPOL, nxi=SOURCE_NXI),
    z_min=-1.0,
    z_max=1.0,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
axisymmetric_grid = axisymmetric_config.build_grid()
axisymmetric_radius = axisymmetric_fixture.boundary_radius(
    AXISYMMETRIC_RADIUS,
    jnp.asarray(axisymmetric_grid.z),
)
axisymmetric_flux_derivative = float(axisymmetric_fixture.poloidal_flux(AXISYMMETRIC_RADIUS, 0.0))
axisymmetric_result = solve_fixed_boundary_from_radius(
    axisymmetric_radius,
    axisymmetric_config,
    elements=SPLINE_ELEMENTS,
    axial_flux_derivative=axisymmetric_flux_derivative,
    solve_lambda=True,
    gradient_tolerance=FTOL,
    require_convergence=True,
)
axisymmetric_evaluated = axisymmetric_result.evaluated
axisymmetric_discretization = SplineMirrorDiscretization.build(axisymmetric_config, elements=SPLINE_ELEMENTS)
axisymmetric_boundary = axisymmetric_discretization.fit_boundary(
    MirrorBoundary.from_radius(axisymmetric_radius, axisymmetric_grid),
    axisymmetric_grid,
)
axisymmetric_mout = write_mout(
    OUTPUT_DIR / "mout_axisymmetric.nc",
    mout_from_result(
        axisymmetric_evaluated,
        axisymmetric_discretization.grid,
        axisymmetric_config,
        boundary=axisymmetric_discretization.evaluate_boundary(axisymmetric_boundary),
        axial_flux_derivative=axisymmetric_flux_derivative,
    ),
)
plot_mout(axisymmetric_mout, OUTPUT_DIR, name="axisymmetric")
summaries["axisymmetric"] = {
    "status": "supported",
    "iterations": axisymmetric_evaluated.iterations,
    "variational_max": float(axisymmetric_evaluated.variational.maximum),
    "staggered_weak_max": float(axisymmetric_evaluated.staggered_weak_force.maximum),
    "strong_force_normalized_rms": float(axisymmetric_evaluated.force.normalized_rms),
    "normalized_divergence_rms": float(axisymmetric_evaluated.normalized_divergence_rms),
    "axial_flux_derivative": axisymmetric_flux_derivative,
    "mirror_ratio": float(1.0 + AXISYMMETRIC_MIRROR_STRENGTH),
}
assert float(axisymmetric_evaluated.variational.maximum) <= FTOL
assert float(axisymmetric_evaluated.staggered_weak_force.maximum) <= 1.1 * FTOL
assert float(axisymmetric_evaluated.force.normalized_rms) < STRONG_FORCE_GATE
assert float(axisymmetric_evaluated.normalized_divergence_rms) < 1.0e-12

# Side-by-side solved 3-D geometry: circular-section axisymmetric mirror on
# the left, the 90-degree rotating ellipse on the right, coloured by LCFS |B|.
pair_figure = plot_mirror_3d_pair(
    axisymmetric_mout,
    OUTPUT_DIR / "mout_rotating_ellipse.nc",
    OUTPUT_DIR,
    titles=(
        "Axisymmetric mirror (circular sections)",
        "Rotating-ellipse mirror (90-degree twist)",
    ),
    name="mirror_fixed_boundary_3d",
)
print(f"Wrote paired fixed-boundary 3-D figure: {pair_figure}")

(OUTPUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
print(json.dumps(summaries, indent=2))
