#!/usr/bin/env python
"""Optimize one VMEX QI problem with any standard SciPy gradient method."""

from __future__ import annotations

import os

import numpy as np
import scipy.optimize

from vmex import OptimizationMonitor

from qi_shared_problem import iteration_budget, make_qi_problem


MAX_MODE = 1                   # boundary Fourier modes released to the optimizer
METHOD = "least_squares"       # or "BFGS", "L-BFGS-B"
BUDGET = iteration_budget(20)  # optimizer iterations (1 under VMEX_EXAMPLES_CI=1)
ci = os.environ.get("VMEX_EXAMPLES_CI") == "1"

problem = make_qi_problem(MAX_MODE)
monitor = OptimizationMonitor(problem)

if METHOD == "least_squares":
    problem.compile_residual_and_jacobian()
    result = scipy.optimize.least_squares(
        problem.residual,
        problem.x0,
        jac=problem.residual_jac,
        x_scale=problem.scales,
        callback=monitor,
        max_nfev=BUDGET,
    )
else:
    problem.compile_value_and_gradient()
    x0, scales = problem.x0, problem.scales

    def x_from_y(y):
        return x0 + scales * y

    def monitor_y(intermediate):
        return monitor(x_from_y(getattr(intermediate, "x", intermediate)))

    result = scipy.optimize.minimize(
        lambda y: problem.fun(x_from_y(y)),
        np.zeros_like(x0),
        jac=lambda y: scales * problem.grad(x_from_y(y)),
        method=METHOD,
        callback=monitor_y,
        options={"maxiter": BUDGET, "maxls": 3 if ci else 40}
        if METHOD == "L-BFGS-B" else {"maxiter": BUDGET},
    )
    result.x = x_from_y(result.x)

problem.input_from_x(result.x).to_indata(f"input.QI_scipy_{METHOD}")
print(f"{METHOD}: final cost = {problem.fun(result.x):.12e}")
