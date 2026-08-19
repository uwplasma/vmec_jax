#!/usr/bin/env python
"""Optimize an explicit VMEX QI problem with an arbitrary Optax transform."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax

import vmex as vj
from vmex import OptimizationMonitor
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual


ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
MAX_MODE = 1 if ci_smoke else 3
STEPS = 1 if ci_smoke else 100
VARY_MAJOR_RADIUS = False     # set True to optimize RBC(0,0) instead of fixing it

inp = VmecInput.from_file(Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial")
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
SURFACES = np.linspace(0.1, 1.0, 6)
qi = ConstructedQIResidual(SURFACES, mboz=8 if ci_smoke else 12,
    nboz=8 if ci_smoke else 12, nphi=31 if ci_smoke else 61,
    nalpha=7 if ci_smoke else 18, n_bounce=7 if ci_smoke else 21)

def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        0.3 - jnp.abs(opt.mean_iota(equilibrium_state, solver_context)), 0.0)

def elongation_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - 8.0, 0.0)

terms = [(qi, 0.0, 10.0), (opt.aspect_ratio, 10.0, 0.005),
         (iota_floor, 0.0, 10.0), (elongation_excess, 0.0, 1.0)]
problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=MAX_MODE,
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, progress=True, evaluation_progress=True)
print(f"dof_names = {problem.dof_names}")
problem.compile_value_and_gradient()
transform = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1.0e-2),
)
x0, scales = jnp.asarray(problem.x0), 0.02 * jnp.asarray(problem.scales)
y = jnp.zeros_like(x0)
state = transform.init(y)
monitor = OptimizationMonitor(problem)

for iteration in range(STEPS):
    x = x0 + scales * y
    value, gradient_x = problem.jax_value_and_grad(x)
    gradient = scales * gradient_x
    updates, state = transform.update(gradient, state, y)
    y = optax.apply_updates(y, updates)
    monitor.record(
        x,
        cost=float(value),
        optimality=float(jnp.linalg.norm(gradient, ord=jnp.inf)),
        iteration=iteration,
    )

x = x0 + scales * y
equilibrium = problem.equilibrium_from_x(x)
final_input = replace(problem.input_from_x(x), ns_array=np.array([31 if ci_smoke else 101]),
                      ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
                                          verbose=not ci_smoke, raise_on_max_iterations=True)
input_path = final_input.to_indata("input.QI_optax_adam")
wout_path = vj.write_wout("wout_QI_optax_adam.nc", final_equilibrium.wout)
print(f"Optax Adam: final cost = {float(problem.jax_fun(x)):.12e}, "
      f"QI total = {float(qi.total(final_equilibrium)):.6e}")
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QI_optax_adam_objectives.csv")
monitor.plot("QI_optax_adam_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
