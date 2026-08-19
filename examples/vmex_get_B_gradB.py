#!/usr/bin/env python
"""Evaluate a finite-beta VMEX field inside the plasma and its exact VJPs."""

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import vmex as vj
from vmex import optimize as opt

DATA = Path(__file__).resolve().parent / "data" / "input.LandremanPaul2021_QA_lowres"

print("Building a finite-beta equilibrium and its boundary-derivative graph...")
inp = vj.VmecInput.from_file(DATA).change_resolution(mpol=3, ntor=3, ntheta=12, nzeta=12)
am = np.zeros(21); am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1-s)
inp = replace(inp, phiedge=-0.025, pmass_type="power_series", am=am, pres_scale=1400.0,
              ns_array=np.array([9]), ftol_array=np.array([1e-8]), niter_array=np.array([3000]))

# A VmecProblem is needed only because the VJPs differentiate with respect to
# its named boundary modes. It solves the same equilibrium as solve_equilibrium;
# no optimization or quasisymmetry objective is involved here.
problem = opt.VmecProblem.from_input(inp, max_mode=1, use_ess=True, progress=True)
final_equilibrium = problem.equilibrium_from_x(problem.x0, newton_iterations=5)

# Points can be supplied as VMEC (s, theta, phi) or Cartesian (x, y, z).
# B and every spatial derivative below use Cartesian components and Cartesian
# derivative axes. VJPs hold xyz fixed and return problem.dof_names ordering.
final_equilibrium.set_points_flux([[0.5, 0.0, 0.0]])
xyz = final_equilibrium.field.get_points_cart()
print("flux point (s, theta, phi) =", final_equilibrium.field.get_points_flux())
print("Cartesian point (x, y, z) =", xyz)
final_equilibrium.set_points_xyz(xyz)

print("Evaluating B and its spatial derivatives...")
B = final_equilibrium.B()
absB = final_equilibrium.absB()
gradB = final_equilibrium.gradB()
gradgradB = final_equilibrium.gradgradB()
gradgradgradB = final_equilibrium.gradgradgradB()
print("Evaluating B VJP...")
dBdx = final_equilibrium.B_vjp(jnp.ones_like(B))
print("Evaluating gradB VJP...")
dgradBdx = final_equilibrium.gradB_vjp(jnp.ones_like(gradB))
print("Evaluating gradgradB VJP...")
d2Bdx = final_equilibrium.gradgradB_vjp(jnp.ones_like(gradgradB))
print("Evaluating gradgradgradB VJP (the most expensive derivative)...")
d3Bdx = final_equilibrium.gradgradgradB_vjp(jnp.ones_like(gradgradgradB))

print("B [T] =", B)
print("|B| [T] =", absB)
print("gradB, gradgradB, gradgradgradB shapes =",
      gradB.shape, gradgradB.shape, gradgradgradB.shape)
print("dof_names =", problem.dof_names)
print("B, gradB, gradgradB, gradgradgradB VJP shapes =",
      dBdx.shape, dgradBdx.shape, d2Bdx.shape, d3Bdx.shape)
print("Cylindrical B and SIMSOPT-order dB/dX shapes =",
      final_equilibrium.field.B_cyl().shape, final_equilibrium.field.dB_by_dX().shape)
