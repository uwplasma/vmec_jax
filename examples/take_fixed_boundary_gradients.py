#!/usr/bin/env python
"""Differentiate a converged fixed-boundary solve through its own fixed point.

The companion to ``take_free_boundary_gradients.py``, with no coils and no
ESSOS: the boundary Fourier coefficients are the parameters, and VMEX returns
the exact derivative of a converged equilibrium by the implicit function
theorem rather than by re-solving. Certified here against central finite
differences over two independent re-solves.
"""

import dataclasses
from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import vmex as vj
from vmex.core import implicit as im

DATA = Path(__file__).resolve().parent / "data"
NS, MPOL, NTOR, NITER, FTOL = 25, 5, 5, 12000, 1.0e-11
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if CI:
    NS, MPOL, NTOR, NITER, FTOL = 11, 3, 3, 3000, 1.0e-9

inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
inp = inp.change_resolution(mpol=MPOL, ntor=NTOR,
                            ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 4)
inp = replace(inp, ns_array=np.array([NS]), niter_array=np.array([NITER]),
              ftol_array=np.array([FTOL]))

params = im.params_from_input(inp)
config = im.make_config(inp, multigrid=True, ftol=FTOL, max_iterations=NITER,
                        adjoint_tol=1.0e-9)
solver_context = im.runtime_from_params(params, config)

# RBC(1, 0) is the dominant elongation coefficient of this boundary, so its
# derivative is large enough to certify against a finite difference without
# fighting the solver's own convergence floor.
mode = (inp.ntor, 1)


def aspect_from_boundary(value):
    """Aspect ratio of the equilibrium the perturbed boundary converges to."""
    moved = dataclasses.replace(params, rbc=params.rbc.at[mode].set(value))
    equilibrium_state = im.solve_implicit(moved, config)
    return im.aspect_ratio(equilibrium_state, solver_context)


print("Solving the fixed boundary and its implicit adjoint...")
value = jnp.asarray(float(inp.rbc[mode]))
aspect, gradient = jax.value_and_grad(aspect_from_boundary)(value)

step = 1.0e-4 * max(abs(float(value)), 1.0)
finite_difference = (aspect_from_boundary(value + step)
                     - aspect_from_boundary(value - step)) / (2.0 * step)
relative_error = abs(float(gradient - finite_difference)) / abs(
    float(finite_difference))
print(f"aspect = {float(aspect):.6f}")
print(f"d(aspect)/d(RBC{mode}): implicit = {float(gradient):.6e}, "
      f"central FD = {float(finite_difference):.6e}, "
      f"relative error = {relative_error:.2e}")
