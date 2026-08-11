#!/usr/bin/env python
"""Optimize an explicit VMEX QI problem with JAXopt LBFGS or LM."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual


MAX_MODE = 1                   # boundary Fourier modes released to the optimizer
METHOD = "LBFGS"               # or "LM"
ci = os.environ.get("VMEX_EXAMPLES_CI") == "1"
BUDGET = 1 if ci else 20

# JAX 0.9 removed this deprecated alias before JAXopt 0.8.3 stopped using it.
# Keep the compatibility local to this external-backend example.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

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
x0 = jnp.asarray(problem.x0)

if METHOD == "LBFGS":
    problem.compile_value_and_gradient()
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
