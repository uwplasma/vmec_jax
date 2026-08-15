#!/usr/bin/env python
"""Optimize ESSOS coils for a fixed finite-beta VMEX equilibrium."""

from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import freeboundary_diff as fbd

from essos.coils import Coils
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import surfacerzfourier_from_boundary

PRES_SCALE = 1400.0  # approximately 2% volume-average beta
NPHI, NTHETA, VC_DIGITS = 24, 24, 4
MAXITER = 80
NORMAL_FIELD_WEIGHT, PRESSURE_BALANCE_WEIGHT = 5.0e3, 5.0e3
LENGTH_TARGET, LENGTH_WEIGHT = 5.0, 0.2
CURVATURE_LIMIT, CURVATURE_WEIGHT = 5.0, 1.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.08, 1.0e3
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = 0.20, 1.0e3
SHAPE_SCALE, CURRENT_SCALE = 0.02, 0.02

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NPHI, NTHETA, VC_DIGITS, MAXITER = 8, 8, 3, 1

DATA = Path(__file__).resolve().parents[1] / "data"
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres").change_resolution(
    mpol=5, ntor=5, ntheta=16, nzeta=16)
am = np.zeros(21); am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1-s)
inp = replace(inp, phiedge=-0.025, pmass_type="power_series", am=am, pres_scale=PRES_SCALE,
              ns_array=np.array([17]), ftol_array=np.array([1e-9]), niter_array=np.array([4000]))

print("Solving the fixed-boundary finite-beta target...")
equilibrium = opt.solve_equilibrium(inp)
surface_data = fbd.surface_field_data_from_state(
    inp, equilibrium.state, runtime=equilibrium.runtime, nphi=NPHI, ntheta=NTHETA)
precision = fbd.plan_vc_precision(surface_data, digits=VC_DIGITS)
interface = fbd.FreeBoundaryDiffProblem.from_surface_data(
    surface_data, digits=VC_DIGITS, precision=precision)
surface = surfacerzfourier_from_boundary(inp.rbc, inp.zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)

# These vacuum coils reproduce the same QA boundary. The pressure-balance term
# modifies their shape and currents to supply the finite-beta external field.
coils0 = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
n_shape = coils0.curves.dofs.size
x0 = np.asarray(coils0.dofs); scales = np.r_[np.full(n_shape, SHAPE_SCALE),
                                             np.full(coils0.dofs_currents.size, CURRENT_SCALE)]
B_reference = jnp.sqrt(jnp.sum(interface.weights * jnp.sum(surface_data.B_total**2, axis=0)))

def coils_from_u(u):
    return coils0.with_dofs(jnp.asarray(x0) + jnp.asarray(scales) * u)

def coil_field(coils):
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)

def objective(u):
    coils = coils_from_u(u); external_field = coil_field(coils)
    normal = jnp.sqrt(interface.weights) * interface.bnormal_residual(external_field) / B_reference
    pressure = jnp.sqrt(interface.weights) * interface.pressure_balance_residual(external_field) / B_reference**2
    lengths = coils.length[:len(coils0.dofs_currents)] - LENGTH_TARGET
    curvature = jnp.maximum(coils.curvature - CURVATURE_LIMIT, 0.0)
    costs = jnp.asarray([
        0.5 * NORMAL_FIELD_WEIGHT * jnp.vdot(normal, normal),
        0.5 * PRESSURE_BALANCE_WEIGHT * jnp.vdot(pressure, pressure),
        0.5 * LENGTH_WEIGHT * jnp.vdot(lengths, lengths),
        0.5 * CURVATURE_WEIGHT * jnp.mean(curvature**2),
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(coils, COIL_DISTANCE_LIMIT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_LIMIT, block_size=32),
    ])
    return jnp.sum(costs), costs

monitor = opt.OptimizationMonitor()
scipy_objective = monitor.wrap_value_and_grad(
    jax.jit(jax.value_and_grad(objective, has_aux=True)),
    ("normal field", "pressure balance", "length", "curvature",
     "coil separation", "coil-surface separation"))
problem = vj.FunctionProblem.from_functions(np.zeros_like(x0), value_and_grad=scipy_objective,
                                            names=coils0.dof_names)

print(f"Optimizing {x0.size} ESSOS shape/current variables with exact reverse-mode gradients")
print(f"dof_names = {problem.dof_names}")
problem.compile_value_and_gradient(report_interval=10.0)
result = minimize(problem.value_and_grad, problem.x0, jac=True, method="L-BFGS-B",
    bounds=[(-3.0, 3.0)] * x0.size, callback=monitor,
    options={"maxiter": MAXITER, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 20})

coils = coils_from_u(result.x); external_field = coil_field(coils)
Bmag = jnp.linalg.norm(surface_data.B_total, axis=0)
Bn_over_B = interface.bnormal_residual(external_field) / Bmag
pressure_error = interface.pressure_balance_residual(external_field) / Bmag**2
print(f"B.n/B RMS = {100 * float(jnp.sqrt(jnp.sum(interface.weights * Bn_over_B**2))):.3f}%, "
      f"max = {100 * float(jnp.max(jnp.abs(Bn_over_B))):.3f}%")
print("Normalized total-pressure jump RMS = "
      f"{float(jnp.sqrt(jnp.sum(interface.weights * pressure_error**2))):.3e}")
print(f"Coil lengths = {np.asarray(coils.length[:len(coils0.dofs_currents)])}")
print(f"Maximum curvature = {float(jnp.max(coils.curvature)):.3f} 1/m")

# Save and plot results
coils.to_json("coils_LandremanPaulQA_finite_beta_optimized.json")
surface.to_vtk("surface_LandremanPaulQA_finite_beta", extra_data={
    "B_dot_n_over_B": np.asarray(Bn_over_B)[None], "B": np.asarray(Bmag)[None],
    "pressure_balance_error": np.asarray(pressure_error)[None]})
coils.to_vtk("coils_LandremanPaulQA_finite_beta_optimized")
monitor.save("finite_beta_coil_objectives.csv")
monitor.plot("finite_beta_coil_objectives.png", title="Finite-beta coil objective terms")
print("Wrote finite-beta coil JSON, VTK, CSV, and objective plot")
