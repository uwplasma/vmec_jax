#!/usr/bin/env python
"""Optimize an explicit VMEX QI problem with an arbitrary Optax transform."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax

from vmex import OptimizationMonitor
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual


MAX_MODE = 1                  # boundary Fourier modes released to the optimizer
STEPS = 1 if os.environ.get("VMEX_EXAMPLES_CI") == "1" else 100

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
problem.compile_value_and_gradient()
transform = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1.0e-2),
)
x = jnp.asarray(problem.x0)
state = transform.init(x)
monitor = OptimizationMonitor(problem)

for iteration in range(STEPS):
    value, gradient = problem.jax_value_and_grad(x)
    updates, state = transform.update(gradient, state, x)
    x = optax.apply_updates(x, updates)
    monitor.record(
        x,
        cost=float(value),
        optimality=float(jnp.linalg.norm(gradient, ord=jnp.inf)),
        iteration=iteration,
    )

problem.input_from_x(x).to_indata("input.QI_optax_adam")
print(f"Optax Adam: final cost = {float(problem.jax_fun(x)):.12e}")
