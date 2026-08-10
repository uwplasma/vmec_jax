#!/usr/bin/env python
"""Optimize one VMEX QI problem with any standard SciPy gradient method."""

from __future__ import annotations

import scipy.optimize

from vmex import OptimizationMonitor

from qi_shared_problem import iteration_budget, make_qi_problem


MAX_MODE = 1                   # boundary Fourier modes released to the optimizer
METHOD = "least_squares"       # or "BFGS", "L-BFGS-B"
BUDGET = iteration_budget(20)  # optimizer iterations (1 under VMEX_EXAMPLES_CI=1)

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
    result = scipy.optimize.minimize(
        problem.value_and_grad,
        problem.x0,
        jac=True,
        method=METHOD,
        bounds=problem.bounds,
        callback=monitor,
        options={"maxiter": BUDGET},
    )

problem.input_from_x(result.x).to_indata(f"input.QI_scipy_{METHOD}")
print(f"{METHOD}: final cost = {problem.fun(result.x):.12e}")
