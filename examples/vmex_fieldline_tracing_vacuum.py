#!/usr/bin/env python
"""Compare VMEX and ESSOS field-line traces for a vacuum QA stellarator."""

from dataclasses import replace
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core import freeboundary_diff as fbd

from essos.coils import Coils
from essos.dynamics import LevelsetStoppingCriterion, Tracing
from essos.fields import BiotSavart
from essos.surfaces import SurfaceClassifier, surfacerzfourier_from_boundary

DATA = Path(__file__).resolve().parent / "data"
N_FIELDLINES, N_TOROIDAL_TURNS, TRACE_LENGTH, N_SAMPLES = 14, 400, 3000.0, 25000
# Cartesian coil/exterior traces use arclength, so rescaling B does not change coverage.
TRACE_TOLERANCE, OUTSIDE_OFFSET = 1.0e-7, 0.045
# At this seed angle, 0.049 m stayed bounded for the full trace; 0.050 m escaped.
MAX_SURFACE_DISTANCE = 0.15  # terminate a wandering coil-field line this far from the LCFS
NPHI, NTHETA, VC_DIGITS = 24, 24, 4
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    N_FIELDLINES, N_TOROIDAL_TURNS, N_SAMPLES, TRACE_TOLERANCE = 3, 2, 120, 1.0e-6
    TRACE_LENGTH = 20.0
    NPHI, NTHETA, VC_DIGITS = 8, 8, 3

print("Solving the vacuum QA equilibrium and loading its optimized ESSOS coils...")
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres").change_resolution(
    mpol=5, ntor=5, ntheta=16, nzeta=16)
inp = replace(inp, phiedge=-0.025, ns_array=np.array([17]),
              ftol_array=np.array([1e-9]), niter_array=np.array([4000]))
equilibrium = opt.solve_equilibrium(inp)
coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
biot_savart = BiotSavart(coils)
coil_field = jax.jit(lambda points: jax.vmap(biot_savart.B)(
    points.reshape(-1, 3)).reshape(points.shape))
exterior = equilibrium.exterior_field(external_field=coil_field, plasma="vacuum")

equilibrium.set_points_flux([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
axis, edge = equilibrium.field.get_points_cart()
# The exact magnetic axis is a coordinate singularity. These physical-space
# seeds form one uninterrupted line from just off-axis to OUTSIDE_OFFSET.
edge_radius = jnp.linalg.norm(edge - axis)
seed_fractions = jnp.linspace(0.02, 1.0 + OUTSIDE_OFFSET / edge_radius, N_FIELDLINES)
xyz_seeds = axis + seed_fractions[:, None] * (edge - axis)
inside = seed_fractions < 1.0
inside_xyz, outside_xyz = xyz_seeds[inside], xyz_seeds[~inside]
equilibrium.set_points_xyz(inside_xyz); flux_seeds = equilibrium.field.get_points_flux()

classifier_surface = surfacerzfourier_from_boundary(
    inp.rbc, inp.zbs, inp.nfp, nphi=32, ntheta=32)
classifier = SurfaceClassifier(
    classifier_surface, h=0.08, padding=MAX_SURFACE_DISTANCE + 0.03)
escape = LevelsetStoppingCriterion(classifier, maximum_distance=MAX_SURFACE_DISTANCE)

surface_data = fbd.surface_field_data_from_state(
    inp, equilibrium.state, runtime=equilibrium.runtime, nphi=NPHI, ntheta=NTHETA)
precision = fbd.plan_vc_precision(surface_data, digits=VC_DIGITS)
interface = fbd.FreeBoundaryDiffProblem.from_surface_data(
    surface_data, digits=VC_DIGITS, precision=precision)
B_surface = interface.total_B_out(coil_field); Bmag_surface = jnp.linalg.norm(B_surface, axis=0)
Bn_over_B = jnp.abs(interface.bnormal_residual(coil_field)) / Bmag_surface
print(f"True boundary B.n/B: mean = {100 * float(jnp.sum(interface.weights * Bn_over_B)):.3f}%, "
      f"max = {100 * float(jnp.max(Bn_over_B)):.3f}%")

def trace(label, field, seeds, *, cartesian=False, stop_outside=False):
    print(f"Tracing {label} (the first call compiles ESSOS)...")
    started = perf_counter()
    duration = TRACE_LENGTH if cartesian else 2.0 * jnp.pi * N_TOROIDAL_TURNS
    model = "FieldLineArclength" if cartesian else "FieldLineToroidal"
    result = Tracing(field=field, model=model, initial_conditions=seeds,
        maxtime=duration, timestep=duration / (N_SAMPLES - 1), times_to_trace=N_SAMPLES,
        atol=TRACE_TOLERANCE, rtol=TRACE_TOLERANCE,
        stopping_criteria=escape if stop_outside else None)
    jax.block_until_ready(result.trajectories_xyz)
    message = f"{label} ready in {perf_counter() - started:.1f} s"
    if stop_outside:
        message += f"; {int(jnp.sum(result.boundary_hits))}/{len(seeds)} lines reached the distance limit"
    print(message)
    return result

vmex_inside = trace("VMEX interior field", equilibrium.field_in_flux_coordinates(),
                    flux_seeds)
coil_trace = trace("ESSOS coil field from the same seed line", biot_savart,
                   xyz_seeds, cartesian=True, stop_outside=True)
vmex_outside = trace("VMEX exterior API outside the LCFS", exterior,
                     outside_xyz, cartesian=True, stop_outside=True)

print("Plotting 3D trajectories and the phi=0 Poincare comparison...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

surface = surfacerzfourier_from_boundary(inp.rbc, inp.zbs, inp.nfp, nphi=60, ntheta=60)
figure = plt.figure(figsize=(10.5, 4.5)); axis3d = figure.add_subplot(121, projection="3d")
surface.plot(ax=axis3d, show=False, color="lightsteelblue", alpha=0.30)
coils.plot(ax=axis3d, show=False, color="saddlebrown", linewidth=1.1)
vmex_inside.plot(ax=axis3d, show=False, n_trajectories_plot=len(inside_xyz),
                 color="#0072B2", linewidth=0.8)
vmex_outside.plot(ax=axis3d, show=False, n_trajectories_plot=len(outside_xyz),
                  color="#D55E00", linewidth=1.0)
axis3d.set_title("VMEX surface and ESSOS coils"); axis3d.set_axis_off()
poincare = figure.add_subplot(122)
vmex_inside.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#0072B2", s=0.01)
coil_colors = ["#009E73" if bool(value) else "#D55E00" for value in inside]
coil_trace.poincare_plot(shifts=[0.0], ax=poincare, show=False, color=coil_colors, s=0.01)
vmex_outside.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#CC79A7", s=0.01)
poincare.set(xlabel="R [m]", ylabel="Z [m]", title=r"Vacuum: $\phi=0$ Poincare")
from matplotlib.lines import Line2D
poincare.grid(alpha=0.25); poincare.legend(handles=[
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#0072B2", label="VMEX interior"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#009E73", label="ESSOS coils, interior seeds"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#D55E00", label="ESSOS coils, exterior seeds"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#CC79A7", label="VMEX exterior, exterior seeds"),
], fontsize=8, loc="best")
figure.tight_layout(); figure.savefig("vmex_fieldline_tracing_vacuum.png", dpi=200); plt.close(figure)
print("Wrote vmex_fieldline_tracing_vacuum.png")
