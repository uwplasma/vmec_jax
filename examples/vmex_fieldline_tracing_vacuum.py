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

from essos.coils import Coils
from essos.dynamics import Tracing
from essos.fields import BiotSavart
from essos.surfaces import surfacerzfourier_from_boundary

DATA = Path(__file__).resolve().parent / "data"
N_FIELDLINES, MAXTIME, N_SAMPLES = 8, 800.0, 8000
TRACE_TOLERANCE, OUTSIDE_OFFSET = 1.0e-7, 0.04
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    N_FIELDLINES, MAXTIME, N_SAMPLES, TRACE_TOLERANCE = 3, 20.0, 120, 1.0e-6

print("Solving the vacuum QA equilibrium and loading its optimized ESSOS coils...")
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres").change_resolution(
    mpol=5, ntor=5, ntheta=16, nzeta=16)
inp = replace(inp, phiedge=-0.025, ns_array=np.array([17]),
              ftol_array=np.array([1e-9]), niter_array=np.array([4000]))
equilibrium = opt.solve_equilibrium(inp)
coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
biot_savart = BiotSavart(coils); coil_field = jax.jit(jax.vmap(biot_savart.B))
exterior = equilibrium.exterior_field(external_field=coil_field, plasma="vacuum")

flux_seeds = jnp.stack((jnp.linspace(0.05, 0.95, N_FIELDLINES),
                        jnp.zeros(N_FIELDLINES), jnp.zeros(N_FIELDLINES)), axis=1)
equilibrium.set_points_flux(flux_seeds); xyz_seeds = equilibrium.field.get_points_cart()
equilibrium.set_points_flux([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
axis, edge = equilibrium.field.get_points_cart()
outside_seed = edge + OUTSIDE_OFFSET * (edge - axis) / jnp.linalg.norm(edge - axis)

def trace(label, field, seeds, *, flux_coordinates=False):
    print(f"Tracing {label} (the first call compiles ESSOS)...")
    started = perf_counter()
    result = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=seeds,
        maxtime=MAXTIME, timestep=MAXTIME / (N_SAMPLES - 1), times_to_trace=N_SAMPLES,
        atol=TRACE_TOLERANCE, rtol=TRACE_TOLERANCE)
    if flux_coordinates:
        result.trajectories = result.trajectories_xyz = jax.vmap(jax.vmap(field.to_xyz))(
            result.trajectories)
    jax.block_until_ready(result.trajectories_xyz)
    print(f"{label} ready in {perf_counter() - started:.1f} s")
    return result

vmex_inside = trace("VMEX interior field", equilibrium.field_in_flux_coordinates(),
                    flux_seeds, flux_coordinates=True)
coil_inside = trace("ESSOS coil field from the same interior seeds", biot_savart, xyz_seeds)
coil_outside = trace("ESSOS coil field outside the LCFS", biot_savart, outside_seed[None])
vmex_outside = trace("VMEX exterior API outside the LCFS", exterior, outside_seed[None])

print("Plotting 3D trajectories and the phi=0 Poincare comparison...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

surface = surfacerzfourier_from_boundary(inp.rbc, inp.zbs, inp.nfp, nphi=60, ntheta=60)
figure = plt.figure(figsize=(10.5, 4.5)); axis3d = figure.add_subplot(121, projection="3d")
surface.plot(ax=axis3d, show=False, color="lightsteelblue", alpha=0.30)
coils.plot(ax=axis3d, show=False, color="saddlebrown", linewidth=1.1)
vmex_inside.plot(ax=axis3d, show=False, n_trajectories_plot=N_FIELDLINES,
                 color="#0072B2", linewidth=0.8)
coil_outside.plot(ax=axis3d, show=False, n_trajectories_plot=1, color="#D55E00", linewidth=1.0)
axis3d.set_title("VMEX surface and ESSOS coils"); axis3d.set_axis_off()
poincare = figure.add_subplot(122)
vmex_inside.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#0072B2", s=5)
coil_inside.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#009E73", s=2)
coil_outside.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#D55E00", s=7)
vmex_outside.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#CC79A7", s=2)
poincare.set(xlabel="R [m]", ylabel="Z [m]", title=r"Vacuum: $\phi=0$ Poincare")
from matplotlib.lines import Line2D
poincare.grid(alpha=0.25); poincare.legend(handles=[
    Line2D([], [], marker="o", linestyle="none", color="#0072B2", label="VMEX interior"),
    Line2D([], [], marker="o", linestyle="none", color="#009E73", label="ESSOS coils, same seeds"),
    Line2D([], [], marker="o", linestyle="none", color="#D55E00", label="ESSOS outside"),
    Line2D([], [], marker="o", linestyle="none", color="#CC79A7", label="VMEX exterior API"),
], fontsize=8, loc="best")
figure.tight_layout(); figure.savefig("vmex_fieldline_tracing_vacuum.png", dpi=200); plt.close(figure)
print("Wrote vmex_fieldline_tracing_vacuum.png")
