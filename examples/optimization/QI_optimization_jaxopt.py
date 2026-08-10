#!/usr/bin/env python
"""Optimize the shared VMEX QI problem with JAXopt LBFGS or LM."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import jaxopt

from qi_shared_problem import iteration_budget, make_qi_problem


MAX_MODE = 1                   # boundary Fourier modes released to the optimizer
METHOD = "LBFGS"               # or "LM"
BUDGET = iteration_budget(20)  # optimizer iterations (1 under VMEX_EXAMPLES_CI=1)

# JAX 0.9 removed this deprecated alias before JAXopt 0.8.3 stopped using it.
# Keep the compatibility local to this external-backend example.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

problem = make_qi_problem(MAX_MODE)
x0 = jnp.asarray(problem.x0)

if METHOD == "LBFGS":
    problem.compile_value_and_gradient()
    ci = os.environ.get("VMEX_EXAMPLES_CI") == "1"
    result = jaxopt.LBFGS(
        problem.jax_value_and_grad,
        value_and_grad=True,
        maxiter=BUDGET,
        maxls=3 if ci else 10,
        stepsize=1.0e-3 if ci else 0.0,
        jit=False,  # equilibrium is a host callback; only its kernels are jitted
    ).run(x0)
else:
    problem.compile_residual_and_jacobian()

    @jax.custom_jvp
    def residual(x):
        return problem.jax_residual(x)

    @residual.defjvp
    def residual_jvp(primals, tangents):
        x, = primals
        tangent, = tangents
        return residual(x), problem.jax_residual_jac(x) @ tangent

    result = jaxopt.LevenbergMarquardt(
        residual,
        maxiter=BUDGET,
        materialize_jac=True,
        solver="cholesky",
        jit=False,
    ).run(x0)

x = result.params
problem.input_from_x(x).to_indata(f"input.QI_jaxopt_{METHOD}")
print(f"JAXopt {METHOD}: final cost = {float(problem.jax_fun(x)):.12e}")
