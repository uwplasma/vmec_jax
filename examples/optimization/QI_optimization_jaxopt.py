#!/usr/bin/env python
"""Optimize the shared VMEX QI problem with JAXopt LBFGS or LM."""

from __future__ import annotations

import argparse
import os

import jax.numpy as jnp
import jaxopt

from qi_backend_problem import iteration_budget, make_qi_problem


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("LBFGS", "LM"), default="LBFGS")
args = parser.parse_args()

problem = make_qi_problem()
budget = iteration_budget(20)
x0 = jnp.asarray(problem.x0)

if args.method == "LBFGS":
    ci = os.environ.get("VMEX_EXAMPLES_CI") == "1"
    result = jaxopt.LBFGS(
        problem.jax_value_and_grad,
        value_and_grad=True,
        maxiter=budget,
        maxls=3 if ci else 10,
        stepsize=1.0e-3 if ci else 0.0,
        jit=False,  # equilibrium is a host callback; only its kernels are jitted
    ).run(x0)
else:
    result = jaxopt.LevenbergMarquardt(
        problem.jax_residual,
        jac_fun=problem.jax_residual_jac,
        maxiter=budget,
        jit=False,
    ).run(x0)

x = result.params
problem.input_from_x(x).to_indata(f"input.QI_jaxopt_{args.method}")
print(f"JAXopt {args.method}: final cost = {float(problem.jax_fun(x)):.12e}")
