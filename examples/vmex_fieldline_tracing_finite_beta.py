#!/usr/bin/env python
"""Trace a finite-beta QA field inside and outside its VMEX boundary.

The commented ``Coils.from_simsopt`` line accepts a SIMSOPT coil JSON without
changing the VMEX virtual-casing or ESSOS tracing workflow.
Preview: this script needs ESSOS branch ``rj/vmex-optimization-interfaces``.

Outside the CI smoke run, the phi=0 Poincare panel pair the README embeds is
also written straight into ``docs/_static/figures`` as lossless WebP, so
re-running this script reproduces the committed bytes.
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
from matplotlib.transforms import Bbox
import numpy as np
import vmex as vj
from vmex import optimize as opt
from vmex.core import virtual_casing as vc
from vmex.core.extender import VmecExtender

try:
    from essos.coils import Coils
    from essos.dynamics import LevelsetStoppingCriterion, trace_field_lines
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceClassifier, surfacerzfourier_from_boundary
except ImportError as error:
    raise ImportError(
        "This example needs ESSOS branch rj/vmex-optimization-interfaces "
        "(uwplasma/ESSOS#58)."
    ) from error

DATA = Path(__file__).resolve().parent / "data"
README_FIGURE = (Path(__file__).resolve().parents[1] / "docs" / "_static" / "figures"
                 / "readme_extender_exterior_islands.webp")
N_FIELDLINES, N_TOROIDAL_TURNS, TRACE_LENGTH, N_SAMPLES = 14, 400, 3000.0, 25000
# Cartesian coil/exterior traces use arclength, so rescaling B does not change coverage.
TRACE_TOLERANCE, OUTSIDE_OFFSET = 1.0e-7, 0.005
# Virtual casing is singular on the source surface. The fast field below is a
# local first-order continuation, so terminate it before extrapolation can
# create false islands. Stop before leaving the resolved exterior region.
MAX_SURFACE_DISTANCE = 0.055
NPHI, NTHETA, VC_DIGITS = 24, 24, 4
TRACE_PROGRESS = True
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    N_FIELDLINES, N_TOROIDAL_TURNS, N_SAMPLES, TRACE_TOLERANCE = 3, 2, 120, 1.0e-6
    TRACE_LENGTH = 20.0
    NPHI, NTHETA, VC_DIGITS = 8, 8, 3
    TRACE_PROGRESS = False

print("Solving the finite-beta QA equilibrium and loading its matched ESSOS coils...")
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_beta0p5_bootstrap")
inp = replace(inp, ns_array=np.array([31 if ci_smoke else 51]),
              ftol_array=np.array([1e-10 if ci_smoke else 1e-14]),
              niter_array=np.array([8000]))
equilibrium = opt.solve_equilibrium(inp, verbose=True)
coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json"))
# A SIMSOPT coil JSON can be used without changing virtual casing or tracing:
# coils = Coils.from_simsopt("coils.json", nfp=inp.nfp, stellsym=True)
biot_savart = BiotSavart(coils)
coil_field = jax.jit(lambda points: jax.vmap(biot_savart.B)(
    points.reshape(-1, 3)).reshape(points.shape))

print("Building the self-consistent coil + plasma-current exterior field...")
# A prescribed-interface virtual-casing calculation separates the converged
# total field into plasma-current and coil parts; no free boundary is solved.
surface_data = vc.surface_field_data_from_state(
    inp, equilibrium.solution, runtime=equilibrium.solver_context, nphi=NPHI, ntheta=NTHETA)
exterior = VmecExtender.from_surface_data(
    surface_data, external_field=coil_field, digits=VC_DIGITS)
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

precision = exterior.plasma_field.plan_surface_precision(digits=VC_DIGITS)
interface = vc.PlasmaVacuumInterface.from_surface_data(
    surface_data, digits=VC_DIGITS, precision=precision,
    virtual_casing_field=exterior.plasma_field)
B_surface = interface.total_B_out(coil_field); Bmag_surface = jnp.linalg.norm(B_surface, axis=0)
Bn_over_B = jnp.abs(interface.bnormal_residual(coil_field)) / Bmag_surface
alignment = (jnp.sum(interface.weights * jnp.sum(B_surface * surface_data.B_total, axis=0))
             / jnp.sqrt(jnp.sum(interface.weights * Bmag_surface**2)
                        * jnp.sum(interface.weights * jnp.sum(surface_data.B_total**2, axis=0))))
print(f"True boundary B.n/B: mean = {100 * float(jnp.sum(interface.weights * Bn_over_B)):.3f}%, "
      f"max = {100 * float(jnp.max(Bn_over_B)):.3f}%")
print(f"Boundary field alignment = {float(alignment):.6f}")
print("Preparing the near-surface virtual-casing continuation...")
exterior = exterior.with_near_surface_continuation(
    digits=VC_DIGITS, precision=precision, B_surface=interface.B_plasma)
coil_B_outside = coil_field(outside_xyz); total_B_outside = exterior.B(outside_xyz)
plasma_fraction = jnp.linalg.norm(total_B_outside - coil_B_outside, axis=1) / jnp.linalg.norm(total_B_outside, axis=1)
direction_difference = jnp.rad2deg(jnp.arccos(jnp.clip(jnp.sum(
    coil_B_outside * total_B_outside, axis=1) /
    (jnp.linalg.norm(coil_B_outside, axis=1) * jnp.linalg.norm(total_B_outside, axis=1)), -1.0, 1.0)))
print(f"Near-LCFS plasma-field fraction: median = {100 * float(jnp.median(plasma_fraction)):.2f}%, "
      f"max = {100 * float(jnp.max(plasma_fraction)):.2f}%; direction change: "
      f"median = {float(jnp.median(direction_difference)):.3f} deg, "
      f"max = {float(jnp.max(direction_difference)):.3f} deg")
del interface, precision, B_surface, Bmag_surface, Bn_over_B
jax.clear_caches()  # the equilibrium and on-surface diagnostic are not evaluated again
escape = None
vmex_inside = trace_field_lines(equilibrium.field_in_flux_coordinates(), flux_seeds,
    toroidal_turns=N_TOROIDAL_TURNS, samples=N_SAMPLES, tolerance=TRACE_TOLERANCE,
    progress=TRACE_PROGRESS, label="VMEX total field inside")
jax.clear_caches()
classifier_surface = surfacerzfourier_from_boundary(
    inp.rbc, inp.zbs, inp.nfp, nphi=32, ntheta=32)
classifier = SurfaceClassifier(
    classifier_surface, h=0.08, padding=MAX_SURFACE_DISTANCE + 0.03)
escape = LevelsetStoppingCriterion(classifier, maximum_distance=MAX_SURFACE_DISTANCE)
coil_trace = trace_field_lines(biot_savart, xyz_seeds, length=TRACE_LENGTH,
    samples=N_SAMPLES, tolerance=TRACE_TOLERANCE, stopping_criteria=escape,
    progress=TRACE_PROGRESS, label="ESSOS coil-only field from the same seed line")
vmex_outside = trace_field_lines(exterior, outside_xyz, length=TRACE_LENGTH,
    samples=N_SAMPLES, tolerance=TRACE_TOLERANCE, stopping_criteria=escape,
    progress=TRACE_PROGRESS, label="VMEX coil + virtual-casing field outside")

print("Plotting 3D trajectories and the phi=0 Poincare comparison...")
surface = surfacerzfourier_from_boundary(inp.rbc, inp.zbs, inp.nfp, nphi=60, ntheta=60)
figure = plt.figure(figsize=(10.6, 4.5)); grid = figure.add_gridspec(1, 3, width_ratios=(3, 1, 1))
axis3d = figure.add_subplot(grid[0], projection="3d")
surface.plot(ax=axis3d, show=False, color="lightsteelblue", alpha=0.30)
coils.plot(ax=axis3d, show=False, color="saddlebrown", linewidth=1.1)
vmex_inside.plot(ax=axis3d, show=False, n_trajectories_plot=len(inside_xyz),
                 color="#0072B2", linewidth=0.35, alpha=0.65)
vmex_outside.plot(ax=axis3d, show=False, n_trajectories_plot=len(outside_xyz),
                  color="#D55E00", linewidth=0.35, alpha=0.65)
axis3d.set_title(f"Self-consistent field, beta={float(equilibrium.wout.betatotal):.2%}")
axis3d.set_axis_off(); axis3d.set_box_aspect((1, 1, 1), zoom=1.20)
poincare_coils = figure.add_subplot(grid[1]); poincare_total = figure.add_subplot(grid[2])
# Marker size/opacity chosen so the sections — especially the exterior
# islands — stay legible at README scale.
POINCARE_STYLE = dict(s=1.2, alpha=0.95)
inside_sections = vmex_inside.poincare_plot(
    shifts=[0.0], ax=poincare_coils, show=False, color="#0072B2", **POINCARE_STYLE)
coil_colors = ["#009E73" if bool(value) else "#D55E00" for value in inside]
coil_sections = coil_trace.poincare_plot(
    shifts=[0.0], ax=poincare_coils, show=False, color=coil_colors, **POINCARE_STYLE)
vmex_inside.poincare_plot(
    shifts=[0.0], ax=poincare_total, show=False, color="#0072B2", **POINCARE_STYLE)
outside_sections = vmex_outside.poincare_plot(
    shifts=[0.0], ax=poincare_total, show=False, color="#CC79A7", **POINCARE_STYLE)
# Overlay the coil-only exterior sections on the same axes as coil + plasma;
# their separation is the virtual-casing plasma-current contribution.
for is_inside, (radius, height, _time) in zip(np.asarray(inside), coil_sections):
    if not is_inside:
        poincare_total.scatter(radius, height, color="#D55E00", marker="x",
                               s=5.0, linewidths=0.5, alpha=0.95)
coil_exterior_crossings = [len(section[0]) for is_inside, section in zip(np.asarray(inside), coil_sections)
                           if not is_inside]
if not any(coil_exterior_crossings):
    poincare_total.text(0.03, 0.97, "coil-only exterior:\n" r"no $\phi=0$ return",
        color="#D55E00", fontsize=6.5, va="top", transform=poincare_total.transAxes)
section = np.asarray(surface.gamma[0])
for panel, title in ((poincare_coils, "coils only"),
                     (poincare_total, "exterior: coils vs coils + plasma")):
    panel.plot(np.hypot(section[:, 0], section[:, 1]), section[:, 2], "k-", lw=1.0)
    panel.set(xlabel="R [m]", title=title); panel.grid(alpha=0.25)
    # poincare_plot sets oversized labels on the current axes; renormalize.
    panel.xaxis.label.set_fontsize(10); panel.yaxis.label.set_fontsize(10)
poincare_coils.set_ylabel("Z [m]"); poincare_total.set_ylabel("")
poincare_total.tick_params(labelleft=False)
all_sections = inside_sections + coil_sections + outside_sections
r_values = np.concatenate([np.hypot(section[:, 0], section[:, 1])]
                          + [row[0] for row in all_sections])
z_values = np.concatenate([section[:, 2]] + [row[1] for row in all_sections])
for panel in (poincare_coils, poincare_total):
    panel.set_xlim(r_values.min(), r_values.max()); panel.set_ylim(z_values.min(), z_values.max())
    panel.set_aspect("equal", adjustable="box")
legend_handles = [
    Line2D([], [], color="k", lw=1.0, label="VMEX LCFS"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#0072B2", label="VMEX total field, interior seeds"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#009E73", label="coils only, interior seeds"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#D55E00", label="coils only, exterior seeds"),
    Line2D([], [], marker="o", markersize=2, linestyle="none", color="#CC79A7", label="coils + plasma, exterior seeds"),
]
axis3d.legend(handles=legend_handles, fontsize=7, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
title = figure.suptitle(r"Finite-beta field lines at $\phi=0$", y=0.98)
figure.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.90, wspace=0.08)
figure.savefig("vmex_fieldline_tracing_finite_beta.png", dpi=200,
               bbox_inches="tight", pad_inches=0.04)
if not ci_smoke:
    # The README shows only the phi=0 Poincare pair: crop it from the same
    # render rather than by hand, so the committed figure has one generator.
    # The figure title straddles the 3-D panel and the pair; hide it so its
    # descender does not stray into the crop.
    title.set_visible(False)
    renderer = figure.canvas.get_renderer()
    pair = Bbox.union([panel.get_tightbbox(renderer) for panel in (poincare_coils, poincare_total)])
    figure.savefig(README_FIGURE, dpi=200, bbox_inches=pair.transformed(figure.dpi_scale_trans.inverted()),
                   pad_inches=0.04, pil_kwargs={"lossless": True})
    print(f"Wrote {README_FIGURE}")
plt.close(figure)
bounded = ~np.asarray(vmex_outside.boundary_hits)
crossings = np.asarray([len(row[0]) for row in outside_sections])
offsets = np.asarray((seed_fractions[~inside] - 1.0) * edge_radius)
print(f"Exterior trace QA: {bounded.sum()}/{len(bounded)} lines remained in the LCFS neighborhood; "
      f"offsets [m] = {offsets.round(5).tolist()}, total-field crossings = {crossings.tolist()}, "
      f"coil-only crossings = {coil_exterior_crossings}")
if not np.any(bounded):
    print("No closed exterior flux surface was verified for this finite-beta coil set.")
print("Wrote vmex_fieldline_tracing_finite_beta.png")
