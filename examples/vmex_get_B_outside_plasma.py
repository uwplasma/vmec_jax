#!/usr/bin/env python
"""Evaluate the coil plus finite-beta plasma field outside a VMEX boundary."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import vmex as vj
from vmex import optimize as opt

from essos.coils import Coils
from essos.fields import BiotSavart

DATA = Path(__file__).resolve().parent / "data"
INPUT = DATA / "input.LandremanPaul2021_QA_lowres"
COILS = DATA / "ESSOS_biot_savart_LandremanPaulQA_finite_beta.json"

print("Building a finite-beta equilibrium and its boundary-derivative graph...")
inp = vj.VmecInput.from_file(INPUT).change_resolution(mpol=3, ntor=3, ntheta=12, nzeta=12)
am = np.zeros(21); am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1-s)
inp = replace(inp, phiedge=-0.025, pmass_type="power_series", am=am, pres_scale=1400.0,
              ns_array=np.array([9]), ftol_array=np.array([1e-8]), niter_array=np.array([3000]))
problem = opt.VmecProblem.from_input(inp, max_mode=1, use_ess=True, progress=True)
final_equilibrium = problem.equilibrium_from_x(problem.x0)

print("Loading ESSOS coils optimized for this finite-beta equilibrium...")
coils = Coils.from_json(str(COILS))

def coil_field_from_dofs(dofs):
    field = BiotSavart(coils.with_dofs(dofs))
    return lambda points: jax.vmap(field.B)(points)

# The exterior total field is the actual ESSOS coil field plus the plasma-current
# field from virtual casing. The point is placed just outside the VMEC LCFS.
final_equilibrium.set_points_flux([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
axis, edge = final_equilibrium.field.get_points_cart()
xyz = edge + 0.03 * (edge - axis) / jnp.linalg.norm(edge - axis)
print("Building the coil + virtual-casing exterior field...")
outside = final_equilibrium.exterior_field(
    external_parameters=coils.dofs, external_field_from_parameters=coil_field_from_dofs,
    external_dof_names=coils.dof_names, nphi=12, ntheta=12, digits=4).set_points_xyz(xyz[None])

# All returned field components and spatial derivative axes are Cartesian.
# VJPs hold xyz fixed and return boundary modes followed by ESSOS coil modes.
print("Evaluating B and its spatial derivatives...")
B = outside.B()
absB = outside.absB()
gradB = outside.gradB()
gradgradB = outside.gradgradB()
gradgradgradB = outside.gradgradgradB()
print("Evaluating B VJP...")
dBdx = outside.B_vjp(jnp.ones_like(B))
print("Evaluating gradB VJP...")
dgradBdx = outside.gradB_vjp(jnp.ones_like(gradB))
print("Evaluating gradgradB VJP...")
d2Bdx = outside.gradgradB_vjp(jnp.ones_like(gradgradB))
print("Evaluating gradgradgradB VJP (the most expensive derivative)...")
d3Bdx = outside.gradgradgradB_vjp(jnp.ones_like(gradgradgradB))

print("outside Cartesian point (x, y, z) =", xyz)
print("B [T] =", B)
print("|B| [T] =", absB)
print("uses virtual casing =", outside.uses_virtual_casing)
print("gradB, gradgradB, gradgradgradB shapes =",
      gradB.shape, gradgradB.shape, gradgradgradB.shape)
print("dof_names =", outside.dof_names)
print("B, gradB, gradgradB, gradgradgradB VJP shapes =",
      dBdx.shape, dgradBdx.shape, d2Bdx.shape, d3Bdx.shape)
