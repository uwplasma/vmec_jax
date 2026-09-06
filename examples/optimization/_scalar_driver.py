"""Shared scalar stage used by the eight optimization examples.

The scalar lane trades objective progress per evaluation for a cheaper cold
start and lower peak memory: at a matched evaluation budget the least-squares
driver reached roughly a 3x lower objective on the same problem, so it remains
the default for objective progress.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from vmex import optimize as opt


def run_scalar_stage(
    inp,
    equilibrium,
    objective_terms,
    *,
    label: str,
    max_mode: int,
    maxiter: int,
    parameter_step: float,
    max_parameter_change: float,
    minimum_mpol: int,
    vary_major_radius: bool,
    ess_alpha: float,
    monitor,
    compile_first: bool,
    progress: bool = False,
):
    """Minimize ``0.5 * r.T @ r`` with one reverse equilibrium adjoint."""
    print(f"\n===== scalar {label} stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, minimum_mpol)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

    def loss(state, runtime):
        rows = opt.residuals_from_tuples(state, runtime, objective_terms)
        return 0.5 * jnp.vdot(rows, rows)

    problem = opt.VmecProblem.from_loss(
        inp, loss, max_mode=max_mode,
        vary_major_radius=vary_major_radius, use_ess=True,
        ess_alpha=ess_alpha, restart_from=equilibrium,
        progress=progress, evaluation_progress=progress)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if compile_first:
        problem.compile_value_and_gradient()

    x0 = problem.x0
    step = parameter_step * problem.scales

    def x_from_y(y):
        return x0 + step * y

    def value_and_gradient(y):
        value, gradient = problem.value_and_grad(x_from_y(y))
        return value, step * gradient

    def monitor_y(intermediate_result):
        x = x_from_y(intermediate_result.x)
        value, gradient = problem.value_and_grad(x)
        monitor({"x": x, "fun": value, "jac": gradient})

    initial_value = float(value_and_gradient(np.zeros_like(x0))[0])
    result = minimize(
        value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
        bounds=[(-max_parameter_change, max_parameter_change)] * x0.size,
        callback=monitor_y,
        options={"maxiter": maxiter, "gtol": 1e-6, "ftol": 1e-12,
                 "maxls": 20, "maxcor": 20})
    result.x = x_from_y(result.x)
    print(f"scalar cost: {initial_value:.12e} -> {float(result.fun):.12e}")
    return (
        problem.input_from_x(result.x),
        problem.equilibrium_from_x(result.x),
    )
