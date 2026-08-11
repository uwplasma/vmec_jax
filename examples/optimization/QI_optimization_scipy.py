#!/usr/bin/env python
"""Optimize one VMEX QI problem with any standard SciPy gradient method."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.optimize

from vmex import OptimizationMonitor
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual


MAX_MODE = 1                   # boundary Fourier modes released to the optimizer
METHOD = "least_squares"       # or "BFGS", "L-BFGS-B"
ci = os.environ.get("VMEX_EXAMPLES_CI") == "1"
BUDGET = 1 if ci else 20

inp = VmecInput.from_file(Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2")
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
qi = QIResidual(np.linspace(0.2, 1.0, 4), mboz=8, nboz=8, nphi=41, nalpha=9, n_levels=6)

def iota_floor(state, runtime):
    return jnp.maximum(0.3 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

def elongation_excess(state, runtime):
    return jnp.maximum(opt.max_elongation(state, runtime) - 8.0, 0.0)

terms = [(qi, 0.0, 1.0), (opt.aspect_ratio, 6.0, 0.1),
         (iota_floor, 0.0, 10.0), (elongation_excess, 0.0, 1.0)]
problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=MAX_MODE, use_ess=True, progress=True)
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
