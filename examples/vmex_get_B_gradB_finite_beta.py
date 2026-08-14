#!/usr/bin/env python
"""Evaluate finite-beta VMEX fields inside and outside the plasma boundary."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vmex import optimize as opt
from vmex.core.input import VmecInput

import jax.numpy as jnp

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
MAX_DERIVATIVE = 1  # set to 2 or 3 for the more expensive higher spatial derivatives
MAX_VJP_DERIVATIVE = -1  # set to 0, 1, 2, or 3 for increasingly expensive VJPs
VC_NPHI, VC_NTHETA, VC_DIGITS = 12, 12, 4
if ci_smoke:
    VC_NPHI, VC_NTHETA, VC_DIGITS = 8, 8, 3
DATA = Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure"

inp = VmecInput.from_file(DATA)
inp = replace(inp, ns_array=np.array([13]), ftol_array=np.array([1e-12]),
              niter_array=np.array([2000])).change_resolution(
                  mpol=5, ntor=0, ntheta=16, nzeta=4)
problem = opt.VmecProblem.from_tuples(
    inp, [(opt.aspect_ratio, 3.0, 1.0)], max_mode=1, use_ess=True)
result = SimpleNamespace(x=problem.x0)  # replace with any optimizer result
print("Solving the finite-beta equilibrium...")
final_equilibrium = problem.equilibrium_from_x(result.x)

# Inside: nested-surface VMEC field with VJPs in problem.dof_names order.
print("Evaluating the interior field (parameter VJPs are opt-in)...")
final_equilibrium.set_points([[6.5, 0.0, 0.0]])
B = final_equilibrium.B()
absB = final_equilibrium.absB()
gradB = final_equilibrium.gradB()
if MAX_VJP_DERIVATIVE >= 0:
    dBdx = final_equilibrium.B_vjp(jnp.ones_like(B))
if MAX_VJP_DERIVATIVE >= 1:
    dgradBdx = final_equilibrium.gradB_vjp(jnp.ones_like(gradB))
if MAX_DERIVATIVE >= 2:
    gradgradB = final_equilibrium.gradgradB()
    if MAX_VJP_DERIVATIVE >= 2:
        d2Bdx = final_equilibrium.gradgradB_vjp(jnp.ones_like(gradgradB))
if MAX_DERIVATIVE >= 3:
    gradgradgradB = final_equilibrium.gradgradgradB()
    if MAX_VJP_DERIVATIVE >= 3:
        d3Bdx = final_equilibrium.gradgradgradB_vjp(jnp.ones_like(gradgradgradB))
print("inside (s, theta, phi) =", final_equilibrium.field.flux_coordinates())
print("inside B, |B|, gradB shapes =", B.shape, absB.shape, gradB.shape)
if MAX_VJP_DERIVATIVE >= 0:
    print("inside B VJP shape =", dBdx.shape)

def external_field(points):
    x, y, z = points.T
    radius2 = x*x + y*y
    return jnp.stack((-6.0*y / radius2, 6.0*x / radius2, 0.0*z), axis=-1)

# Outside: external vacuum field plus the plasma-current virtual-casing field.
# The live problem graph retains exact VJPs with respect to boundary/current dofs.
outside = final_equilibrium.exterior_field(
    external_field=external_field, nphi=VC_NPHI, ntheta=VC_NTHETA,
    digits=VC_DIGITS).set_points([[8.5, 0.0, 0.0]])
B_out = outside.B()
gradB_out = outside.gradB()
if MAX_VJP_DERIVATIVE >= 0:
    dBdx_out = outside.B_vjp(jnp.ones_like(B_out))
if MAX_VJP_DERIVATIVE >= 1:
    dgradBdx_out = outside.gradB_vjp(jnp.ones_like(gradB_out))
if MAX_DERIVATIVE >= 2:
    gradgradB_out = outside.gradgradB()
    if MAX_VJP_DERIVATIVE >= 2:
        d2Bdx_out = outside.gradgradB_vjp(jnp.ones_like(gradgradB_out))
if MAX_DERIVATIVE >= 3:
    gradgradgradB_out = outside.gradgradgradB()
    if MAX_VJP_DERIVATIVE >= 3:
        d3Bdx_out = outside.gradgradgradB_vjp(jnp.ones_like(gradgradgradB_out))
print("outside uses virtual casing =", outside.uses_virtual_casing)
print("outside B, |B|, gradB shapes =", B_out.shape, outside.absB().shape, gradB_out.shape)
if MAX_VJP_DERIVATIVE >= 0:
    print("outside B VJP shape =", dBdx_out.shape)
print("dof_names =", outside.dof_names)
