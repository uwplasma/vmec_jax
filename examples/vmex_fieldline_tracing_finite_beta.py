#!/usr/bin/env python
"""Trace a finite-beta QA field inside and outside its VMEX boundary."""

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
PRES_SCALE = 1400.0  # approximately 2% volume-average beta
N_FIELDLINES, MAXTIME, N_SAMPLES = 8, 800.0, 8000
TRACE_TOLERANCE, OUTSIDE_OFFSET = 1.0e-7, 0.04
NPHI, NTHETA, VC_DIGITS = 24, 24, 4
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    N_FIELDLINES, MAXTIME, N_SAMPLES, TRACE_TOLERANCE = 3, 20.0, 120, 1.0e-6
    NPHI, NTHETA, VC_DIGITS = 8, 8, 3

print("Solving the finite-beta QA equilibrium and loading its matched ESSOS coils...")
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres").change_resolution(
    mpol=5, ntor=5, ntheta=16, nzeta=16)
am = np.zeros(21); am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1-s)
inp = replace(inp, phiedge=-0.025, pmass_type="power_series", am=am, pres_scale=PRES_SCALE,
              ns_array=np.array([17]), ftol_array=np.array([1e-9]), niter_array=np.array([4000]))
equilibrium = opt.solve_equilibrium(inp)
coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA_finite_beta.json"))
biot_savart = BiotSavart(coils); coil_field = jax.jit(jax.vmap(biot_savart.B))

print("Building the self-consistent coil + plasma-current exterior field...")
exterior = equilibrium.exterior_field(
    external_field=coil_field, nphi=NPHI, ntheta=NTHETA, digits=VC_DIGITS)
flux_seeds = jnp.stack((jnp.linspace(0.05, 0.95, N_FIELDLINES),
                        jnp.zeros(N_FIELDLINES), jnp.zeros(N_FIELDLINES)), axis=1)
equilibrium.set_points_flux(flux_seeds); xyz_seeds = equilibrium.field.get_points_cart()
equilibrium.set_points_flux([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
axis, edge = equilibrium.field.get_points_cart()
outside_seed = edge + OUTSIDE_OFFSET * (edge - axis) / jnp.linalg.norm(edge - axis)

def trace(label, field, seeds, *, flux_coordinates=False, model="FieldLineAdaptative"):
    print(f"Tracing {label} (the first call compiles ESSOS)...")
    started = perf_counter()
    result = Tracing(field=field, model=model, initial_conditions=seeds,
        maxtime=MAXTIME, timestep=MAXTIME / (N_SAMPLES - 1), times_to_trace=N_SAMPLES,
        atol=TRACE_TOLERANCE, rtol=TRACE_TOLERANCE)
    if flux_coordinates:
        result.trajectories = result.trajectories_xyz = jax.vmap(jax.vmap(field.to_xyz))(
            result.trajectories)
    jax.block_until_ready(result.trajectories_xyz)
    print(f"{label} ready in {perf_counter() - started:.1f} s")
    return result

vmex_inside = trace("VMEX total field inside", equilibrium.field_in_flux_coordinates(),
                    flux_seeds, flux_coordinates=True)
coil_only = trace("ESSOS coil-only field from the same interior seeds", biot_savart, xyz_seeds)
vmex_outside = trace("VMEX coil + virtual-casing field outside", exterior,
                     outside_seed[None], model="FieldLine")

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
vmex_outside.plot(ax=axis3d, show=False, n_trajectories_plot=1, color="#D55E00", linewidth=1.0)
axis3d.set_title(f"Self-consistent field, beta={float(equilibrium.wout.betatotal):.2%}")
axis3d.set_axis_off(); poincare = figure.add_subplot(122)
vmex_inside.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#0072B2", s=5)
coil_only.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#009E73", s=2)
vmex_outside.poincare_plot(shifts=[0.0], ax=poincare, show=False, color="#D55E00", s=7)
poincare.set(xlabel="R [m]", ylabel="Z [m]", title=r"Finite beta: $\phi=0$ Poincare")
from matplotlib.lines import Line2D
poincare.grid(alpha=0.25); poincare.legend(handles=[
    Line2D([], [], marker="o", linestyle="none", color="#0072B2", label="VMEX total field"),
    Line2D([], [], marker="o", linestyle="none", color="#009E73", label="coils only"),
    Line2D([], [], marker="o", linestyle="none", color="#D55E00", label="coils + plasma outside"),
], fontsize=8, loc="best")
figure.tight_layout(); figure.savefig("vmex_fieldline_tracing_finite_beta.png", dpi=200); plt.close(figure)
print("Wrote vmex_fieldline_tracing_finite_beta.png")
