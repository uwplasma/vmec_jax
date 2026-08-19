#!/usr/bin/env python
"""True free-boundary QA optimization with ESSOS coils as the only dofs."""

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

from essos.coils import Coils
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation
from essos.surfaces import SurfaceRZFourier, surfacerzfourier_from_boundary

SURFACES = np.linspace(0.1, 1.0, 6)
NS, MPOL, NTOR, NITER, FTOL = 25, 5, 5, 4000, 1.0e-10
MAXITER, METHOD, PARAMETER_BOUND = 20, "L-BFGS-B", 1.0
ASPECT_TARGET, IOTA_FLOOR = 6.0, 0.42
LENGTH_TARGET, LENGTH_WEIGHT = 3.5, 1.0
CURVATURE_LIMIT, CURVATURE_WEIGHT = 7.0, 10.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.08, 1.0e3
OPTIONS = {"maxiter": MAXITER, "maxls": 10, "ftol": 1.0e-12, "gtol": 1.0e-8}

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NS, MPOL, NTOR, NITER, FTOL, MAXITER = 12, 2, 2, 3000, 1.0e-7, 0
    OPTIONS = {"maxiter": MAXITER, "maxls": 5, "ftol": 1.0e-8, "gtol": 1.0e-5}

DATA = Path(__file__).resolve().parents[1] / "data"
inp = vj.VmecInput.from_file(DATA / "input.minimal_seed_nfp2").change_resolution(
    mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=16)
inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", phiedge=-0.025,
              ns_array=np.array([NS]), niter_array=np.array([NITER]),
              ftol_array=np.array([FTOL]))
coils0 = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
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
solver_context = im.runtime_from_params(params, config.implicit)
# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
tuples = [(qs.residuals_state, 0.0, 1.0),
          (opt.aspect_ratio, ASPECT_TARGET, 1.0),
          (iota_floor, 0.0, 10.0)]

def objective(u):
    equilibrium_state, status, _, _ = vj.solve_free_boundary_implicit_status(params, u, config)

    def accepted(_):
        residual = opt.residuals_from_tuples(equilibrium_state, solver_context, tuples)
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
        # A smooth, finite wall lets SciPy backtrack after a failed trial. Its
        # derivative is explicit here; the failed equilibrium contributes zero.
        residual = jnp.zeros_like(opt.residuals_from_tuples(
            equilibrium_state, solver_context, tuples))
        wall = 1.0e3 * (1.0 + jnp.sqrt(1.0e-12 + jnp.vdot(u, u)))**2
        return wall, (residual, jnp.zeros(3), status)

    return jax.lax.cond(status == 0, accepted, rejected, operand=None)

monitor = opt.OptimizationMonitor()
value_and_grad_jax = jax.value_and_grad(objective, has_aux=True)

def value_and_grad(u):
    (value, (residual, coil_costs, status)), gradient = value_and_grad_jax(jnp.asarray(u))
    rows = np.asarray(residual); parts = (rows[:-2], rows[-2:-1], rows[-1:])
    terms = {name: 0.5 * float(part @ part)
             for name, part in zip(("QA", "aspect", "mean iota"), parts)}
    terms.update(zip(("coil length", "coil curvature", "coil separation"),
                     map(float, np.asarray(coil_costs))))
    terms["rejected trial"] = float(value) if int(status) else 0.0
    return monitor.cache_evaluation(u, value, gradient, terms)

print("Running single_stage_free_boundary_optimization.py")
print(f"True NESTOR free boundary + ESSOS: {x0.size} coil variables; "
      "no boundary dofs or mgrid file")
print(f"dof_names = {dof_names}")
free_problem = vj.FunctionProblem.from_functions(
    np.zeros_like(x0), value_and_grad=value_and_grad, names=dof_names)
first = free_problem.compile_value_and_gradient(progress=not ci_smoke, report_interval=10.0)
if ci_smoke:
    final_cost, optimized_u, iterations = first.value, np.zeros_like(x0), 0
else:
    result = minimize(free_problem.value_and_grad, np.zeros_like(x0), jac=True, method=METHOD,
        bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size,
        callback=monitor, options=OPTIONS)
    optimized_u, final_cost, iterations = result.x, result.fun, result.nit

coils_final = coils_from_u(jnp.asarray(optimized_u))
print("Solving the optimized free boundary for output...")
free_result = vj.solve_free_boundary_multigrid(
    inp, external_field=BiotSavart(coils_final), verbose=not ci_smoke)
wout = vj.wout_from_state(
    inp=inp, state=free_result.state, fsqr=free_result.fsqr,
    fsqz=free_result.fsqz, fsql=free_result.fsql,
    niter=free_result.iterations, converged=free_result.converged,
    vacuum_output=free_result.vacuum)

# Print results
print(f"[final] QA = {float(qs.total_state(free_result.state, solver_context)):.5e}, "
      f"aspect = {float(opt.aspect_ratio(free_result.state, solver_context)):.3f}, "
      f"min |iota| = {float(opt.min_abs_iota(free_result.state, solver_context)):.3f}")
print(f"Objective = {float(final_cost):.6e} after {iterations} {METHOD} iterations")
print(f"Coil lengths = {np.asarray(coils_final.length)}")
print(f"Maximum curvature = {float(np.max(np.asarray(coils_final.curvature))):.3f} 1/m")

# Save results
input_path = inp.to_indata("input.single_stage_free_boundary_optimized")
wout_path = vj.write_wout("wout_single_stage_free_boundary_optimized.nc", wout)
coils_final.to_json("coils_single_stage_free_boundary_optimized.json")
surface_initial = surfacerzfourier_from_boundary(
    inp.rbc, inp.zbs, inp.nfp, nphi=60, ntheta=60)
surface_final = SurfaceRZFourier.from_wout_file(wout_path, nphi=60, ntheta=60)
surface_initial.to_vtk("surface_single_stage_free_boundary_initial")
coils0.to_vtk("coils_single_stage_free_boundary_initial")
surface_final.to_vtk("surface_single_stage_free_boundary_optimized")
coils_final.to_vtk("coils_single_stage_free_boundary_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")

# Plot results
monitor.save("single_stage_free_boundary_objectives.csv")
monitor.plot("single_stage_free_boundary_objectives.png", title="Free-boundary objective terms")
vj.plot_optimization_objects("single_stage_free_boundary_optimization.png",
    ("Initial", surface_initial, coils0), ("Optimized", surface_final, coils_final))
print("Wrote single_stage_free_boundary_optimization.png and objective history")
