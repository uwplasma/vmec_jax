#!/usr/bin/env python
"""Optimize an explicit VMEX QI problem with JAXopt LBFGS or LM."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual


MAX_MODE = 1 if os.environ.get("VMEX_EXAMPLES_CI") == "1" else 3
METHOD = "LBFGS"               # or "LM"
ci = os.environ.get("VMEX_EXAMPLES_CI") == "1"
BUDGET = 1 if ci else 20
VARY_MAJOR_RADIUS = False       # set True to optimize RBC(0,0) instead of fixing it

# JAX 0.9 removed this deprecated alias before JAXopt 0.8.3 stopped using it.
# Keep the compatibility local to this external-backend example.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

inp = VmecInput.from_file(Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial")
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
SURFACES = np.linspace(0.1, 1.0, 6)
qi = ConstructedQIResidual(SURFACES, mboz=8 if ci else 12, nboz=8 if ci else 12,
                           nphi=31 if ci else 61, nalpha=7 if ci else 18,
                           n_bounce=7 if ci else 21)

def iota_floor(state, runtime):
    return jnp.maximum(0.3 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

def elongation_excess(state, runtime):
    return jnp.maximum(opt.max_elongation(state, runtime) - 8.0, 0.0)

terms = [(qi, 0.0, 10.0), (opt.aspect_ratio, 10.0, 0.005),
         (iota_floor, 0.0, 10.0), (elongation_excess, 0.0, 1.0)]
problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=MAX_MODE,
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, progress=True)
print(f"dof_names = {problem.dof_names}")
x0, scales = jnp.asarray(problem.x0), 0.02 * jnp.asarray(problem.scales)

def x_from_y(y):
    return x0 + scales * y

def value_and_grad_y(y):
    value, gradient = problem.jax_value_and_grad(x_from_y(y))
    return value, scales * gradient

if METHOD == "LBFGS":
    problem.compile_value_and_gradient()
    result = jaxopt.LBFGS(
        value_and_grad_y,
        value_and_grad=True,
        maxiter=BUDGET,
        maxls=3 if ci else 10,
        stepsize=1.0e-3 if ci else 0.0,
        jit=False,  # equilibrium is a host callback; only its kernels are jitted
    ).run(jnp.zeros_like(x0))
else:
    problem.compile_residual_and_jacobian()

    @jax.custom_jvp
    def residual(y):
        return problem.jax_residual(x_from_y(y))

    @residual.defjvp
    def residual_jvp(primals, tangents):
        y, = primals
        tangent, = tangents
        return residual(y), problem.jax_residual_jac(x_from_y(y)) @ (scales * tangent)

    result = jaxopt.LevenbergMarquardt(
        residual,
        maxiter=BUDGET,
        materialize_jac=True,
        solver="cholesky",
        jit=False,
    ).run(jnp.zeros_like(x0))

x = x_from_y(result.params)
equilibrium = problem.equilibrium_from_x(x)
final_input = replace(problem.input_from_x(x), ns_array=np.array([31 if ci else 101]),
                      ftol_array=np.array([1e-10 if ci else 1e-14]), niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.state,
                                          verbose=not ci, raise_on_max_iterations=True)
input_path = final_input.to_indata(f"input.QI_jaxopt_{METHOD}")
wout_path = vj.write_wout(f"wout_QI_jaxopt_{METHOD}.nc", final_equilibrium.wout)
print(f"JAXopt {METHOD}: final cost = {float(problem.jax_fun(x)):.12e}, "
      f"QI total = {float(qi.total(final_equilibrium)):.6e}")
print(f"wrote {input_path}\nwrote {wout_path}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
