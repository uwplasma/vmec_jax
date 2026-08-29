#!/usr/bin/env python
"""Fast-start quasi-axisymmetric optimization with a magnetic well.

The scalar objective uses one reverse implicit solve per gradient, avoiding
the large pointwise residual Jacobian that made the former staged
least-squares example spend minutes compiling before its first optimizer
iteration.  The objective value and gradient are exactly the scalarization of
the same weighted residual tuples.
"""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt

nfp = 2  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODE, MAXITER = 3, 30
MAGNETIC_WELL_TARGET = 0.01
ASPECT_TARGET = 5.0
# For a larger design space use MAX_MODE, MAXITER = 9, 60.
# MAGNETIC_WELL_TARGET = 0.07
# ASPECT_TARGET = 3.5
IOTA_FLOOR = 0.42
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
ESS_ALPHA = 1.2  # smaller values let high Fourier modes move more
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODE, MAXITER = 1, 4

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# The exactly circular torus has zero first-order iota sensitivity. This
# explicit rotating-ellipse perturbation gives the local optimizer a QA basin.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

# Objective function terms
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
         (qs, 0.0, 1.0),
         (opt.aspect_ratio, ASPECT_TARGET, 1.0),
         (iota_floor, 0.0, 10.0),
         (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
         ]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)

# Scalarize the exact least-squares rows before differentiating. Reverse-mode
# implicit differentiation then needs one adjoint regardless of boundary dof
# count, instead of materializing the 6723 x 48 pointwise Jacobian.
def loss(equilibrium_state, solver_context):
    rows = opt.residuals_from_tuples(
        equilibrium_state, solver_context, objective_function_terms)
    return 0.5 * jnp.vdot(rows, rows)


mpol = max(MAX_MODE + 2, MINIMUM_MPOL)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
problem = opt.VmecProblem.from_loss(
    inp, loss, max_mode=MAX_MODE, vary_major_radius=VARY_MAJOR_RADIUS,
    use_ess=True, ess_alpha=ESS_ALPHA)
print(f"dof_names = {problem.dof_names}")
monitor.problem = problem
problem.compile_value_and_gradient()

x0 = problem.x0
step = PARAMETER_STEP * problem.scales

def x_from_y(y):
    return x0 + step * y


def value_and_gradient(y):
    value, gradient = problem.value_and_grad(x_from_y(y))
    return value, step * gradient


def monitor_y(intermediate_result):
    x = x_from_y(intermediate_result.x)
    monitor({"x": x, "fun": intermediate_result.fun,
             "jac": value_and_gradient(intermediate_result.x)[1]})


result = minimize(
    value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
    bounds=[(-MAX_PARAMETER_CHANGE, MAX_PARAMETER_CHANGE)] * x0.size,
    callback=monitor_y,
    options={"maxiter": MAXITER, "gtol": 1.0e-6, "ftol": 1.0e-12,
             "maxls": 20, "maxcor": 20})
result.x = x_from_y(result.x)
inp = problem.input_from_x(result.x)
equilibrium = problem.equilibrium_from_x(result.x)
report(f"mode {MAX_MODE}", equilibrium)

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")

vacuum_name = "QA_optimized"
vacuum_input_path = final_input.to_indata(f"input.{vacuum_name}")
vacuum_wout_path = vj.write_wout(f"wout_{vacuum_name}.nc", final_equilibrium.wout)
print(f"wrote {vacuum_input_path}\nwrote {vacuum_wout_path}")

# Plot results
monitor.save("QA_optimization_objectives.csv")
monitor.plot("QA_optimization_objectives.png")
for path in vj.plot_wout(vacuum_wout_path, ".").values():
    print(f"wrote {path}")
