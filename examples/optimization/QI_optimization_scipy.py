#!/usr/bin/env python
"""QI boundary optimization with a SciPy scalar-gradient method."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import OptimizationMonitor
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual

nfp = 2  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 6)
MAX_MODES = [1, 2, 3, 4]
MAXITER = 50
METHOD = "L-BFGS-B"  # or "BFGS"
PARAMETER_BOUND = 3.0
BOUNDARY_STEP = 0.05  # typical change represented by one scaled variable
ASPECT_TARGET = 5.0
IOTA_FLOOR = 0.26
MIRROR_LIMIT = 0.21
ELONGATION_LIMIT = 8.0
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

qi_options = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
validation_options = dict(mboz=14, nboz=14, nphi=101, nalpha=29, n_bounce=31)

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    qi_options = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    validation_options = qi_options
    MAX_MODES, MAXITER = [2], 1

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

# Objective function terms
qi = ConstructedQIResidual(SURFACES, **qi_options)

def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        0.3 - jnp.abs(opt.mean_iota(equilibrium_state, solver_context)), 0.0)

def elongation_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - 8.0, 0.0)

def mirror_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context) - 0.25, 0.0)

objective_function_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0),
    (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0),
]
qi_terms = [(qi, 0.0, 10.0), *objective_function_terms]

report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("mirror", opt.mirror_ratio, ".4f"),
    ("elongation", opt.max_elongation, ".4f"))

def x_from_y(y):
    return x0 + scales * y

def cost(y):
    return problem.fun(x_from_y(y))

def gradient(y):
    return scales * problem.grad(x_from_y(y))

options = {"maxiter": MAXITER, "gtol": 1.0e-6}
if METHOD == "L-BFGS-B":
    options.update(maxls=20, ftol=1.0e-12, maxcor=20)

# If a RuntimeWarning reports uncertified Jacobian columns, it is expected
# once the optimizer leaves the seed and needs no action: the shipped
# jacobian_adjoint_tol=1e-4 and jacobian_adjoint_maxiter=10 are the measured
# optimum, since ten times that budget moved the Jacobian by 2e-8 and
# certified no extra column. Both are from_tuples arguments; pass
# evaluation_progress=False to drop the per-evaluation timing lines.
for max_mode in MAX_MODES:
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(
        inp, qi_terms, max_mode=max_mode, use_ess=True, progress=not ci_smoke,
        evaluation_progress=not ci_smoke,
        vary_major_radius=VARY_MAJOR_RADIUS,
    )
    print(f"dof_names = {problem.dof_names}")
    if not ci_smoke:
        problem.compile_value_and_gradient()
    x0, scales = problem.x0, BOUNDARY_STEP * problem.scales
    monitor = OptimizationMonitor(problem)

    def monitor_y(intermediate_result):
        monitor({"x": x_from_y(intermediate_result.x), "fun": intermediate_result.fun,
                 "jac": gradient(intermediate_result.x)})

    result = minimize(cost, np.zeros_like(x0), jac=gradient, method=METHOD,
        bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size if METHOD == "L-BFGS-B" else None,
        callback=monitor_y, options=options)
    result.x = x_from_y(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    inp = problem.input_from_x(result.x)

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
qi_final = report("final", final_equilibrium)["constructed QI"]
print(f"\n{METHOD}: final cost = {float(result.fun):.12e}, QI total = {qi_final:.3e}")
qi_validation = ConstructedQIResidual(SURFACES, **validation_options)
print(f"\nQI total {qi_final:.3e}; independent fine-grid validation "
      f"{float(qi_validation.total(final_equilibrium)):.3e}")

# Save results
input_path = final_input.to_indata(f"input.QI_scipy_{METHOD}")
wout_path = vj.write_wout(f"wout_QI_scipy_{METHOD}.nc", final_equilibrium.wout)
print(f"wrote {input_path}")
print(f"wrote {wout_path}")

# Plot results
monitor.save(f"QI_scipy_{METHOD}_objectives.csv")
monitor.plot(f"QI_scipy_{METHOD}_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
