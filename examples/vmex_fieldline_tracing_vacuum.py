#!/usr/bin/env python
"""Compare VMEX and ESSOS field-line traces for a vacuum QA stellarator.

The commented ``Coils.from_simsopt`` line accepts a SIMSOPT coil JSON without
changing the VMEX exterior-field or ESSOS tracing workflow.
"""

from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core import virtual_casing as vc

from essos.coils import Coils
from essos.dynamics import LevelsetStoppingCriterion, trace_field_lines
from essos.fields import BiotSavart
from essos.surfaces import SurfaceClassifier, surfacerzfourier_from_boundary

DATA = Path(__file__).resolve().parent / "data"
N_FIELDLINES, N_TOROIDAL_TURNS, TRACE_LENGTH, N_SAMPLES = 14, 400, 3000.0, 25000
# Cartesian coil/exterior traces use arclength, so rescaling B does not change coverage.
TRACE_TOLERANCE, OUTSIDE_OFFSET = 1.0e-7, 0.045
# At this seed angle, 0.049 m stayed bounded for the full trace; 0.050 m escaped.
MAX_SURFACE_DISTANCE = 0.15  # terminate a wandering coil-field line this far from the LCFS
NPHI, NTHETA, VC_DIGITS = 24, 24, 4
TRACE_PROGRESS = True
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    N_FIELDLINES, N_TOROIDAL_TURNS, N_SAMPLES, TRACE_TOLERANCE = 3, 2, 120, 1.0e-6
    TRACE_LENGTH = 20.0
    NPHI, NTHETA, VC_DIGITS = 8, 8, 3
    TRACE_PROGRESS = False

print("Solving the vacuum QA equilibrium and loading its optimized ESSOS coils...")
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    inp = inp.change_resolution(mpol=5, ntor=5, ntheta=16, nzeta=16)
inp = replace(inp, phiedge=-0.025,
              ns_array=np.array([25 if os.environ.get("VMEX_EXAMPLES_CI") == "1" else 51]),
              ftol_array=np.array([1e-10 if os.environ.get("VMEX_EXAMPLES_CI") == "1" else 1e-14]),
              niter_array=np.array([8000]))
equilibrium = opt.solve_equilibrium(inp, verbose=True)
coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
# A SIMSOPT coil JSON can be used without changing the field/tracing code:
# coils = Coils.from_simsopt("coils.json", nfp=inp.nfp, stellsym=True)
biot_savart = BiotSavart(coils)
coil_field = jax.jit(lambda points: jax.vmap(biot_savart.B)(
    points.reshape(-1, 3)).reshape(points.shape))
exterior = equilibrium.exterior_field(external_field=coil_field, plasma="vacuum")

equilibrium.set_points_flux([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
axis, edge = equilibrium.field.get_points_cart()
# VMEX regularizes the physical field at the coordinate-degenerate axis. Sample
# the whole minor radius, then resolve the shorter exterior interval densely.
edge_radius = jnp.linalg.norm(edge - axis)
n_outside = max(1, N_FIELDLINES // 4); n_inside = N_FIELDLINES - n_outside
seed_fractions = jnp.concatenate((jnp.linspace(0.0, 1.0, n_inside),
    1.0 + jnp.linspace(1.0 / n_outside, 1.0, n_outside) * OUTSIDE_OFFSET / edge_radius))
xyz_seeds = axis + seed_fractions[:, None] * (edge - axis)
inside = seed_fractions <= 1.0
inside_xyz, outside_xyz = xyz_seeds[inside], xyz_seeds[~inside]
equilibrium.set_points_xyz(inside_xyz); flux_seeds = equilibrium.field.get_points_flux()

classifier_surface = surfacerzfourier_from_boundary(
    inp.rbc, inp.zbs, inp.nfp, nphi=32, ntheta=32)
classifier = SurfaceClassifier(
    classifier_surface, h=0.08, padding=MAX_SURFACE_DISTANCE + 0.03)
escape = LevelsetStoppingCriterion(classifier, maximum_distance=MAX_SURFACE_DISTANCE)

# The prescribed-interface API supplies an independent B.n/B check against
# the ESSOS coils here; it does not run a free-boundary equilibrium.
surface_data = vc.surface_field_data_from_state(
    inp, equilibrium.solution, runtime=equilibrium.solver_context, nphi=NPHI, ntheta=NTHETA)
precision = vc.plan_vc_precision(surface_data, digits=VC_DIGITS)
interface = vc.PlasmaVacuumInterface.from_surface_data(
    surface_data, digits=VC_DIGITS, precision=precision)
B_surface = interface.total_B_out(coil_field); Bmag_surface = jnp.linalg.norm(B_surface, axis=0)
Bn_over_B = jnp.abs(interface.bnormal_residual(coil_field)) / Bmag_surface
alignment = (jnp.sum(interface.weights * jnp.sum(B_surface * surface_data.B_total, axis=0))
             / jnp.sqrt(jnp.sum(interface.weights * Bmag_surface**2)
                        * jnp.sum(interface.weights * jnp.sum(surface_data.B_total**2, axis=0))))
print(f"True boundary B.n/B: mean = {100 * float(jnp.sum(interface.weights * Bn_over_B)):.3f}%, "
      f"max = {100 * float(jnp.max(Bn_over_B)):.3f}%")
print(f"Boundary field alignment = {float(alignment):.6f}; the Poincare comparison magnifies "
      f"the remaining error over {N_TOROIDAL_TURNS} toroidal turns")

vmex_inside = trace_field_lines(equilibrium.field_in_flux_coordinates(), flux_seeds,
    toroidal_turns=N_TOROIDAL_TURNS, samples=N_SAMPLES, tolerance=TRACE_TOLERANCE,
    progress=TRACE_PROGRESS, label="VMEX interior field")
coil_trace = trace_field_lines(biot_savart, xyz_seeds, length=TRACE_LENGTH,
    samples=N_SAMPLES, tolerance=TRACE_TOLERANCE, stopping_criteria=escape,
    progress=TRACE_PROGRESS, label="ESSOS coil field from the same seed line")
vmex_outside = trace_field_lines(exterior, outside_xyz, length=TRACE_LENGTH,
    samples=N_SAMPLES, tolerance=TRACE_TOLERANCE, stopping_criteria=escape,
    progress=TRACE_PROGRESS, label="VMEX exterior API outside the LCFS")

print("Plotting 3D trajectories and the phi=0 Poincare comparison...")
surface = surfacerzfourier_from_boundary(inp.rbc, inp.zbs, inp.nfp, nphi=60, ntheta=60)
figure = plt.figure(figsize=(8.6, 4.5)); grid = figure.add_gridspec(1, 2, width_ratios=(3, 1))
axis3d = figure.add_subplot(grid[0], projection="3d")
surface.plot(ax=axis3d, show=False, color="lightsteelblue", alpha=0.30)
coils.plot(ax=axis3d, show=False, color="saddlebrown", linewidth=1.1)
vmex_inside.plot(ax=axis3d, show=False, n_trajectories_plot=len(inside_xyz),
                 color="#0072B2", linewidth=0.35, alpha=0.65)
vmex_outside.plot(ax=axis3d, show=False, n_trajectories_plot=len(outside_xyz),
                  color="#D55E00", linewidth=0.35, alpha=0.65)
axis3d.set_title("VMEX surface and ESSOS coils"); axis3d.set_axis_off()
axis3d.set_box_aspect((1, 1, 1), zoom=1.20)
poincare = figure.add_subplot(grid[1])
inside_sections = vmex_inside.poincare_plot(
    shifts=[0.0], ax=poincare, show=False, color="#0072B2", s=0.01)
coil_colors = ["#009E73" if bool(value) else "#D55E00" for value in inside]
coil_sections = coil_trace.poincare_plot(
    shifts=[0.0], ax=poincare, show=False, color=coil_colors, s=0.01)
outside_sections = vmex_outside.poincare_plot(
    shifts=[0.0], ax=poincare, show=False, color="#CC79A7", s=0.01)
section = np.asarray(surface.gamma[0]); poincare.plot(
    np.hypot(section[:, 0], section[:, 1]), section[:, 2], "k-", lw=1.0)
poincare.set(xlabel="R [m]", ylabel="Z [m]", title=r"Vacuum: $\phi=0$ Poincare")
all_sections = inside_sections + coil_sections + outside_sections
r_values = np.concatenate([np.hypot(section[:, 0], section[:, 1])]
                          + [row[0] for row in all_sections])
z_values = np.concatenate([section[:, 2]] + [row[1] for row in all_sections])
poincare.set_xlim(r_values.min(), r_values.max()); poincare.set_ylim(z_values.min(), z_values.max())
poincare.set_aspect("equal", adjustable="box")
legend_handles = [
    Line2D([], [], color="k", lw=1.0, label="VMEX LCFS"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#0072B2", label="VMEX interior"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#009E73", label="ESSOS coils, interior seeds"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#D55E00", label="ESSOS coils, exterior seeds"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#CC79A7", label="VMEX exterior, exterior seeds"),
]
poincare.grid(alpha=0.25)
axis3d.legend(handles=legend_handles, fontsize=7, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
figure.subplots_adjust(left=0.01, right=0.98, bottom=0.05, top=0.92, wspace=0.02)
figure.savefig("vmex_fieldline_tracing_vacuum.png", dpi=200,
               bbox_inches="tight", pad_inches=0.04); plt.close(figure)
bounded = ~np.asarray(vmex_outside.boundary_hits)
crossings = np.asarray([len(row[0]) for row in outside_sections])
offsets = np.asarray((seed_fractions[~inside] - 1.0) * edge_radius)
print(f"Exterior trace QA: {bounded.sum()}/{len(bounded)} lines remained in the LCFS neighborhood; "
      f"offsets [m] = {offsets.round(5).tolist()}, crossings = {crossings.tolist()}")
print("Wrote vmex_fieldline_tracing_vacuum.png")
