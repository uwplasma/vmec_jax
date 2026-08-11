#!/usr/bin/env python
"""Optimize the shared VMEX QI problem with an arbitrary Optax transform."""

from __future__ import annotations

import jax.numpy as jnp
import optax

from vmex import OptimizationMonitor

from qi_shared_problem import iteration_budget, make_qi_problem


MAX_MODE = 1                  # boundary Fourier modes released to the optimizer
STEPS = iteration_budget(100)  # Adam steps (1 under VMEX_EXAMPLES_CI=1)

problem = make_qi_problem(MAX_MODE)
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
