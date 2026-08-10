#!/usr/bin/env python
"""Optimize the shared VMEX QI problem with JAXopt LBFGS or LM."""

from __future__ import annotations

import os

import jax.numpy as jnp
import jaxopt

from qi_shared_problem import iteration_budget, make_qi_problem


MAX_MODE = 1                   # boundary Fourier modes released to the optimizer
METHOD = "LBFGS"               # or "LM"
BUDGET = iteration_budget(20)  # optimizer iterations (1 under VMEX_EXAMPLES_CI=1)

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
    result = jaxopt.LevenbergMarquardt(
        problem.jax_residual,
        jac_fun=problem.jax_residual_jac,
        maxiter=BUDGET,
        jit=False,
    ).run(x0)

x = result.params
problem.input_from_x(x).to_indata(f"input.QI_jaxopt_{METHOD}")
print(f"JAXopt {METHOD}: final cost = {float(problem.jax_fun(x)):.12e}")
