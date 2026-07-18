"""Solve and plot a periodic B-spline stellarator-mirror hybrid."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmex.mirror import (
    MirrorConfig,
    MirrorResolution,
    build_stellarator_mirror_hybrid,
    plot_stellarator_mirror_hybrid,
    solve_fixed_boundary,
    trace_closed_field_line,
)

# Inputs: edit these values, then run this file directly.
NS, MPOL = 5, 3
SPLINE_COEFFICIENTS = 32
SPLINE_QUADRATURE_ORDER = 3
STRAIGHT_LENGTH = 8.0
RETURN_RADIUS = 2.5
SEMI_MAJOR = 0.45
SEMI_MINOR = 0.30
AXIAL_FLUX_DERIVATIVE = 0.02
CURRENT_DERIVATIVE = 0.002
FTOL = 1.0e-12
MAX_ITERATIONS = 1000
OUTPUT_DIR = Path("results/stellarator_mirror_hybrid")

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

resolution = MirrorResolution(ns=NS, mpol=MPOL, nxi=4)
config = MirrorConfig(
    resolution=resolution,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
setup = build_stellarator_mirror_hybrid(
    resolution,
    coefficient_count=SPLINE_COEFFICIENTS,
    straight_length=STRAIGHT_LENGTH,
    return_radius=RETURN_RADIUS,
    semi_major=SEMI_MAJOR,
    semi_minor=SEMI_MINOR,
    axial_flux_derivative=AXIAL_FLUX_DERIVATIVE,
    quadrature_order=SPLINE_QUADRATURE_ORDER,
)
result = solve_fixed_boundary(
    setup.initial_state,
    setup.boundary,
    setup.discretization,
    config,
    axial_flux_derivative=AXIAL_FLUX_DERIVATIVE,
    current_derivative=CURRENT_DERIVATIVE,
    solve_lambda=True,
    axis=setup.axis,
    require_convergence=True,
)
figure = plot_stellarator_mirror_hybrid(result, setup, OUTPUT_DIR)
field_line = trace_closed_field_line(
    result.evaluated.energy.field,
    setup.discretization,
    radial_index=NS - 2,
    turns=2,
)
summary = {
    "converged": result.evaluated.converged,
    "iterations": result.evaluated.iterations,
    "variational_max": float(result.evaluated.variational.maximum),
    "staggered_weak_max": float(result.evaluated.staggered_weak_force.maximum),
    "strong_force_normalized_rms": float(result.evaluated.force.normalized_rms),
    "strong_force_axis_rms": float(result.evaluated.force.axis_normalized_rms),
    "strong_force_first_row_rms": float(result.evaluated.force.first_row_normalized_rms),
    "strong_force_bulk_rms": float(result.evaluated.force.bulk_normalized_rms),
    "strong_force_components": [float(value) for value in result.evaluated.force.component_rms],
    "normalized_divergence_rms": float(result.evaluated.normalized_divergence_rms),
    "volume": float(result.evaluated.energy.geometry.volume),
    "axis_length": float(setup.axis.arc_length),
    "axis_closure_error": float(setup.axis.closure_error),
    "frame_closure_error": float(setup.axis.frame_closure_error),
    "iota_at_s_0p75": float(field_line.iota),
    "figure": str(figure),
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
