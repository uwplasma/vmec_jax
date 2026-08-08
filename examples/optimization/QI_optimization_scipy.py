#!/usr/bin/env python
"""Optimize one VMEX QI problem with any standard SciPy gradient method."""

from __future__ import annotations

import argparse

import scipy.optimize

from vmex import OptimizationMonitor

from qi_backend_problem import iteration_budget, make_qi_problem


parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    choices=("least_squares", "BFGS", "L-BFGS-B"),
    default="least_squares",
)
args = parser.parse_args()

problem = make_qi_problem()
monitor = OptimizationMonitor(problem)
budget = iteration_budget(20)

if args.method == "least_squares":
    result = scipy.optimize.least_squares(
        problem.residual,
        problem.x0,
        jac=problem.residual_jac,
        x_scale=problem.scales,
        callback=monitor,
        max_nfev=budget,
    )
else:
    result = scipy.optimize.minimize(
        problem.value_and_grad,
        problem.x0,
        jac=True,
        method=args.method,
        bounds=problem.bounds,
        callback=monitor,
        options={"maxiter": budget},
    )

problem.input_from_x(result.x).to_indata(f"input.QI_scipy_{args.method}")
print(f"{args.method}: final cost = {problem.fun(result.x):.12e}")
