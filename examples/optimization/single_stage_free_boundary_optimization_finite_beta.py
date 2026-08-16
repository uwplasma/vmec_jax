#!/usr/bin/env python
"""True finite-beta free-boundary QA optimization with only coil dofs."""

from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import implicit as im
from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch

from essos.coils import Coils
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation
from essos.surfaces import SurfaceRZFourier, surfacerzfourier_from_boundary

TARGET_BETA = 0.025
SURFACES = np.linspace(0.1, 0.9, 8)
NS, MPOL, NTOR, NITER, FTOL = 31, 5, 5, 20000, 1.0e-10
MAXITER, METHOD, PARAMETER_BOUND = 20, "L-BFGS-B", 1.0
ASPECT_TARGET, IOTA_TARGET = 6.0, 0.42
LENGTH_TARGET, LENGTH_WEIGHT = 3.5, 1.0
CURVATURE_LIMIT, CURVATURE_WEIGHT = 7.0, 10.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.08, 1.0e3
OPTIONS = {"maxiter": MAXITER, "maxls": 10, "ftol": 1.0e-12, "gtol": 1.0e-8}

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.linspace(0.2, 0.8, 4)
    NS, MPOL, NTOR, NITER, FTOL, MAXITER = 12, 2, 2, 5000, 1.0e-7, 0
    OPTIONS = {"maxiter": MAXITER, "maxls": 5, "ftol": 1.0e-8, "gtol": 1.0e-5}

DATA = Path(__file__).resolve().parents[1] / "data"
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_beta2p5_bootstrap")
inp = inp.change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=16)
inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field",
              phiedge=-inp.phiedge, curtor=-inp.curtor,
              ns_array=np.array([NS]), niter_array=np.array([NITER]),
              ftol_array=np.array([FTOL]))
coils0 = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA_beta2p5_bootstrap.json"))
# The stored coils use the input's opposite global field orientation. Reversing
# PHIEDGE and CURTOR together preserves the equilibrium and bootstrap physics.
if ci_smoke:
    coils0.n_segments = 24

x0 = np.asarray(coils0.dofs)
n_curve_dofs = coils0.dofs_curves.size
scales = np.concatenate([np.full(n_curve_dofs, 0.01),
                         0.02 * np.maximum(np.abs(x0[n_curve_dofs:]), 1.0e5)])
dof_names = coils0.dof_names

def coils_from_u(u):
    return coils0.with_dofs(jnp.asarray(x0) + jnp.asarray(scales) * u)

def field_from_u(u):
    return BiotSavart(coils_from_u(u))

params = im.params_from_input(inp)
config = vj.make_free_boundary_config(
    inp, BiotSavart(coils0), ns=NS, ftol=FTOL, max_iterations=NITER,
    adjoint_tol=1.0e-8, field_from_parameters=field_from_u)
runtime = im.runtime_from_params(params, config.implicit)

# ne=n0(1-s^5), Te=Ti=T0(1-s); rescale n0*T0 to the peak pressure in AM.
n0, T0 = 3.0e20 * (TARGET_BETA / 0.05)**(1 / 3), 15.0e3 * (TARGET_BETA / 0.05)**(2 / 3)
profile_scale = inp.pres_scale * inp.am[0] / (2 * ELEMENTARY_CHARGE * n0 * T0)
n0 *= profile_scale**(1 / 3); T0 *= profile_scale**(2 / 3)
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
bootstrap = RedlBootstrapMismatch(
    profiles, helicity_n=0, surfaces=SURFACES, n_lambda=12 if ci_smoke else 32)
tuples = [(qs.residuals_state, 0.0, 1.0), (bootstrap.residuals_state, 0.0, 1.0),
          (opt.aspect_ratio, ASPECT_TARGET, 1.0), (opt.mean_iota, IOTA_TARGET, 10.0),
          (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2)]

def objective(u):
    state, status, _, _ = vj.solve_free_boundary_implicit_status(params, u, config)

    def accepted(_):
        residual = opt.residuals_from_tuples(state, runtime, tuples)
        coils = coils_from_u(u)
        costs = jnp.asarray([
            0.5 * LENGTH_WEIGHT * jnp.sum((coils.length - LENGTH_TARGET)**2),
            0.5 * CURVATURE_WEIGHT * jnp.sum(
                jnp.maximum(coils.curvature - CURVATURE_LIMIT, 0.0)**2),
            0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
                coils, COIL_DISTANCE_LIMIT, block_size=32),
        ])
        return 0.5 * jnp.vdot(residual, residual) + jnp.sum(costs), (residual, costs, status)

    def rejected(_):
        # A failed trial gets an explicit smooth wall; only certified states
        # enter the physics objective and the implicit adjoint.
        residual = jnp.zeros_like(opt.residuals_from_tuples(state, runtime, tuples))
        wall = 1.0e3 * (1.0 + jnp.sqrt(1.0e-12 + jnp.vdot(u, u)))**2
        return wall, (residual, jnp.zeros(3), status)

    return jax.lax.cond(status == 0, accepted, rejected, operand=None)

monitor = opt.OptimizationMonitor()
value_and_grad_jax = jax.value_and_grad(objective, has_aux=True)

def value_and_grad(u):
    (value, (residual, coil_costs, status)), gradient = value_and_grad_jax(jnp.asarray(u))
    rows, n_boot = np.asarray(residual), len(SURFACES)
    n_qs = rows.size - n_boot - 3
    parts = (rows[:n_qs], rows[n_qs:n_qs + n_boot],
             rows[n_qs + n_boot:n_qs + n_boot + 1],
             rows[n_qs + n_boot + 1:n_qs + n_boot + 2], rows[-1:])
    terms = {name: 0.5 * float(part @ part) for name, part in zip(
        ("QA", "bootstrap", "aspect", "mean iota", "beta"), parts)}
    terms.update(zip(("coil length", "coil curvature", "coil separation"),
                     map(float, np.asarray(coil_costs))))
    terms["rejected trial"] = float(value) if int(status) else 0.0
    return monitor.cache_evaluation(u, value, gradient, terms)

print("Running single_stage_free_boundary_optimization_finite_beta.py")
print(f"True finite-beta NESTOR + ESSOS: {x0.size} coil variables; no boundary dofs or mgrid file")
print(f"dof_names = {dof_names}")
print("The first value/gradient solves the free boundary and compiles one implicit adjoint...")
if ci_smoke:
    final_cost, _ = value_and_grad(np.zeros_like(x0)); optimized_u, iterations = np.zeros_like(x0), 0
else:
    result = minimize(value_and_grad, np.zeros_like(x0), jac=True, method=METHOD,
        bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size,
        callback=monitor, options=OPTIONS)
    optimized_u, final_cost, iterations = result.x, result.fun, result.nit

coils_final = coils_from_u(jnp.asarray(optimized_u))
print("Solving the optimized finite-beta free boundary for output...")
free_result = vj.solve_free_boundary_multigrid(
    inp, external_field=BiotSavart(coils_final), verbose=not ci_smoke)
wout = vj.wout_from_state(
    inp=inp, state=free_result.state, fsqr=free_result.fsqr,
    fsqz=free_result.fsqz, fsql=free_result.fsql,
    niter=free_result.iterations, converged=free_result.converged,
    vacuum_output=free_result.vacuum)
equilibrium = opt.Equilibrium(inp, free_result.state, runtime, free_result)

# Print results
print(f"[final] QA = {float(qs.total_state(free_result.state, runtime)):.5e}, "
      f"f_boot = {float(bootstrap.total_state(free_result.state, runtime)):.5e}, "
      f"beta = {float(wout.betatotal):.3%}, "
      f"aspect = {float(opt.aspect_ratio(free_result.state, runtime)):.3f}, "
      f"mean iota = {float(opt.mean_iota(free_result.state, runtime)):.3f}")
print(f"Objective = {float(final_cost):.6e} after {iterations} {METHOD} iterations")
print(f"Coil lengths = {np.asarray(coils_final.length)}")
print(f"Maximum curvature = {float(np.max(np.asarray(coils_final.curvature))):.3f} 1/m")

# Save results
input_path = inp.to_indata("input.single_stage_free_boundary_finite_beta_optimized")
wout_path = vj.write_wout("wout_single_stage_free_boundary_finite_beta_optimized.nc", wout)
coils_final.to_json("coils_single_stage_free_boundary_finite_beta_optimized.json")
surface_initial = surfacerzfourier_from_boundary(inp.rbc, inp.zbs, inp.nfp, nphi=60, ntheta=60)
surface_final = SurfaceRZFourier.from_wout_file(wout_path, nphi=60, ntheta=60)
surface_initial.to_vtk("surface_single_stage_free_boundary_finite_beta_initial")
coils0.to_vtk("coils_single_stage_free_boundary_finite_beta_initial")
surface_final.to_vtk("surface_single_stage_free_boundary_finite_beta_optimized")
coils_final.to_vtk("coils_single_stage_free_boundary_finite_beta_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")

# Plot results
monitor.save("single_stage_free_boundary_finite_beta_objectives.csv")
monitor.plot("single_stage_free_boundary_finite_beta_objectives.png",
             title="Finite-beta free-boundary objective terms")
vj.plot_bootstrap_current("single_stage_free_boundary_finite_beta_bootstrap_current.png",
                          equilibrium, bootstrap)
vj.plot_optimization_objects("single_stage_free_boundary_finite_beta_optimization.png",
    ("Initial", surface_initial, coils0), ("Optimized", surface_final, coils_final))
print("Wrote finite-beta free-boundary optimization, bootstrap, and objective plots")
