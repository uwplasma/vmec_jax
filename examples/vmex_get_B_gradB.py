#!/usr/bin/env python
"""Evaluate a vacuum VMEX field and its exact spatial/parameter derivatives."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vmex import optimize as opt
from vmex.core.input import VmecInput

import jax.numpy as jnp

MAX_DERIVATIVE = 1  # set to 2 or 3 for the more expensive higher spatial derivatives
MAX_VJP_DERIVATIVE = 0  # set to 0, 1, 2, or 3 for increasingly expensive VJPs
DATA = Path(__file__).resolve().parent / "data" / "input.minimal_seed_nfp2"

inp = VmecInput.from_file(DATA)
problem = opt.VmecProblem.from_tuples(
    inp, [(opt.aspect_ratio, 10.0, 1.0)], max_mode=1, use_ess=True)
result = SimpleNamespace(x=problem.x0)  # replace with any optimizer result
print("Solving the vacuum equilibrium...")
final_equilibrium = problem.equilibrium_from_x(result.x)

# Cartesian points inside the last closed flux surface.
print("Evaluating the interior field (parameter VJPs are opt-in)...")
final_equilibrium.set_points([[1.05, 0.0, 0.0]])
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

# Common SIMSOPT-compatible helpers and inverted VMEC flux coordinates.
field = final_equilibrium.field
print("inside (s, theta, phi) =", field.flux_coordinates())
print("inside B, |B|, gradB shapes =", B.shape, absB.shape, gradB.shape)
print("inside B_cyl, GradAbsB, dB_by_dX shapes =",
      field.B_cyl().shape, field.GradAbsB().shape, field.dB_by_dX().shape)
print("boundary dof_names =", field.dof_names)
if MAX_VJP_DERIVATIVE >= 0:
    print("B VJP shape =", dBdx.shape)
if MAX_VJP_DERIVATIVE >= 1:
    print("gradB VJP shape =", dgradBdx.shape)

# A vacuum exterior is the supplied coil/MGRID field; this analytic toroidal
# field keeps the example dependency-free. Replace it with an ESSOS field.B.
def external_field(points):
    x, y, z = points.T
    radius2 = x*x + y*y
    return jnp.stack((-y / radius2, x / radius2, 0.0*z), axis=-1)

outside = final_equilibrium.exterior_field(
    external_field=external_field, plasma="vacuum").set_points([[1.30, 0.0, 0.0]])
gradgradB_out = outside.gradgradB() if MAX_DERIVATIVE >= 2 else None
gradgradgradB_out = outside.gradgradgradB() if MAX_DERIVATIVE >= 3 else None
print("outside B, |B|, gradB shapes =",
      outside.B().shape, outside.absB().shape, outside.gradB().shape)
if gradgradB_out is not None:
    print("outside gradgradB shape =", gradgradB_out.shape)
if gradgradgradB_out is not None:
    print("outside gradgradgradB shape =", gradgradgradB_out.shape)
# The supplied vacuum field has no VMEX boundary parameters; differentiate its
# coil/current parameters directly in ESSOS. Finite-beta exterior VJPs are below.
