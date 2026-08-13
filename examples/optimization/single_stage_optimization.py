#!/usr/bin/env python
"""Single-stage fixed-boundary plasma and ESSOS coil optimization."""

from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt

from essos.coils import Coils, Curves, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation
from essos.surfaces import SurfaceRZFourier

nfp = 2  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 6)
MAX_MODE = 1
MAXITER = 20
ASPECT_TARGET = 5.0
IOTA_TARGET = 0.42
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.10

N_COILS = 3
COIL_ORDER = 5
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64
STELLSYM = True

NORMAL_FIELD_WEIGHT = 1.0e3
LENGTH_TARGET = 3.3
LENGTH_WEIGHT = 0.1
CURVATURE_LIMIT = 7.0
CURVATURE_WEIGHT = 1.0e-3
COIL_DISTANCE_LIMIT = 0.08
COIL_DISTANCE_WEIGHT = 1.0

NPHI = NTHETA = 16
METHOD = "BFGS"  # also accepts "L-BFGS-B"
OPTIONS = {"maxiter": MAXITER, "gtol": 1.0e-8}
if METHOD == "L-BFGS-B":
    OPTIONS.update(maxls=20, ftol=1e-12, maxcor=20)

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, N_SEGMENTS, NPHI, NTHETA = 1, 24, 8, 8
    COIL_ORDER = 2
    OPTIONS = {"maxiter": MAXITER, "gtol": 1.0e-5}

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = 0.20
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_PERTURBATION, -SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
plasma_terms = [
    (qs.residuals_state, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 10.0),
]
plasma_problem = opt.VmecProblem.from_tuples(
    inp, plasma_terms, max_mode=MAX_MODE, vary_major_radius=VARY_MAJOR_RADIUS,
    use_ess=True, progress=not ci_smoke)

curves0 = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=inp.nfp, stellsym=STELLSYM)
coils0 = Coils(curves0, np.full(N_COILS, COIL_CURRENT))

def normal_field_residual(coils, surface):
    field = BiotSavart(coils)
    magnetic_field = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    normal_field = jnp.sum(magnetic_field * surface.unitnormal, axis=2)
    weights = surface.area_element / surface.area_element.size
    return jnp.sqrt(weights) * normal_field / jnp.linalg.norm(magnetic_field, axis=2)

def coil_lengths(coils, _surface):
    return coils.length[:N_COILS]

def coil_curvature_excess(coils, _surface):
    return jnp.maximum(coils.curvature[:N_COILS] - CURVATURE_LIMIT, 0.0)

def coil_distance_excess(coils, _surface):
    penalty = loss_coil_separation(coils, COIL_DISTANCE_LIMIT, block_size=16)
    return jnp.sqrt(jnp.maximum(penalty, 0.0) + 1.0e-30)

coil_terms = [
    (normal_field_residual, 0.0, NORMAL_FIELD_WEIGHT),
    (coil_lengths, LENGTH_TARGET, LENGTH_WEIGHT),
    (coil_curvature_excess, 0.0, CURVATURE_WEIGHT),
    (coil_distance_excess, 1.0e-15, COIL_DISTANCE_WEIGHT),
]

# The public VMEX problem owns the boundary-mode convention and RBC(0,0) choice.
x_boundary0 = plasma_problem.x0
rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)

curve_shape = curves0.dofs.shape
n_curve_dofs = curves0.dofs.size
x_coils0 = np.asarray(curves0.dofs).ravel()
x0 = np.concatenate([x_boundary0, x_coils0])
fourier_names = ["0"] + [f"{kind}({order})" for order in range(1, COIL_ORDER + 1) for kind in ("s", "c")]
coil_dof_names = tuple(f"coil[{coil}].{axis}{coefficient}" for coil in range(N_COILS)
                       for axis in "xyz" for coefficient in fourier_names)
dof_names = plasma_problem.dof_names + coil_dof_names

# SciPy works in dimensionless increments u, with x = x0 + scales*u.
scales = np.concatenate([
    0.02 * plasma_problem.scales, np.full(n_curve_dofs, 0.05)])


def surface_from_boundary(rbc, zbs, nphi=NPHI, ntheta=NTHETA, range_torus="full torus"):
    rc = jnp.concatenate([rbc[inp.ntor:, 0], rbc[:, 1:].T.ravel()])
    zs = jnp.concatenate([zbs[inp.ntor:, 0], zbs[:, 1:].T.ravel()])
    return SurfaceRZFourier(rc, zs, inp.nfp, inp.mpol - 1, inp.ntor,
                            nphi=nphi, ntheta=ntheta, close=False, range_torus=range_torus)


def objects_from_x(x):
    x_boundary, x_coils = x[:x_boundary0.size], x[x_boundary0.size:]
    rbc, zbs = plasma_problem.boundary_from_x(x_boundary)
    surface = surface_from_boundary(rbc, zbs)

    curve_dofs = x_coils.reshape(curve_shape)
    curves = Curves(curve_dofs, N_SEGMENTS, inp.nfp, STELLSYM)
    coils = Coils(curves, coils0.dofs_currents * coils0.currents_scale,
                  currents_scale=coils0.currents_scale)
    return surface, coils


def objective(u):
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    x_boundary = x[:x_boundary0.size]
    surface, coils = objects_from_x(x)
    coil_residuals = [jnp.sqrt(weight) * (jnp.atleast_1d(function(coils, surface)) - target).ravel()
                      for function, target, weight in coil_terms]
    coil_costs = jnp.stack([0.5 * jnp.vdot(rows, rows) for rows in coil_residuals])
    plasma_rows = plasma_problem.jax_residual(x_boundary)
    return plasma_problem.jax_fun(x_boundary) + jnp.sum(coil_costs), (plasma_rows, coil_costs)


value_and_grad = jax.value_and_grad(objective, has_aux=True)
history = {"total": []}
for name, _start, _stop in plasma_problem.metadata["term_slices"]:
    history[name] = []
for function, _target, _weight in coil_terms:
    history[function.__name__] = []


def scipy_objective(u):
    (value, (plasma_rows, coil_costs)), gradient = value_and_grad(jnp.asarray(u))
    history["total"].append(float(value))
    for name, start, stop in plasma_problem.metadata["term_slices"]:
        rows = plasma_rows[start:stop]; history[name].append(float(0.5 * jnp.vdot(rows, rows)))
    for (function, _target, _weight), term_cost in zip(coil_terms, coil_costs):
        history[function.__name__].append(float(term_cost))
    print(f"{len(history['total']):4d}  J = {float(value):.6e}  |grad J| = {float(jnp.linalg.norm(gradient)):.3e}")
    return float(value), np.asarray(gradient, dtype=float)


print("Running single_stage_optimization.py")
print(f"Fixed-boundary VMEX + ESSOS: {x_boundary0.size} boundary and "
      f"{x_coils0.size} coil variables, exact reverse-mode derivatives")
print(f"dof_names = {dof_names}")
print("Evaluating the initial objective and gradient (the first call compiles JAX)...")
result = minimize(scipy_objective, np.zeros_like(x0), jac=True, method=METHOD, options=OPTIONS)
initial_value = history["total"][0]

x_final = x0 + scales * result.x
_, coils_final = objects_from_x(jnp.asarray(x_final))
equilibrium = plasma_problem.equilibrium_from_x(x_final[:x_boundary0.size])
final_input = plasma_problem.input_from_x(x_final[:x_boundary0.size])
final_input = replace(final_input,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state, verbose=not ci_smoke,
    raise_on_max_iterations=True)

input_path = final_input.to_indata("input.single_stage_optimized")
wout_path = vj.write_wout("wout_single_stage_optimized.nc", final_equilibrium.wout)
coils_final.to_json("coils_single_stage_optimized.json")

surface_initial = surface_from_boundary(rbc0, zbs0, nphi=60, ntheta=60)
surface_final = surface_from_boundary(jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs),
                                      nphi=60, ntheta=60)
figure = plt.figure(figsize=(10, 4))
for panel, surface, coils, title in [
    (1, surface_initial, coils0, "Initial"),
    (2, surface_final, coils_final, "Optimized"),
]:
    axis = figure.add_subplot(1, 2, panel, projection="3d")
    surface.plot(ax=axis, show=False)
    coils.plot(ax=axis, show=False)
    points = np.concatenate([np.asarray(surface.gamma).reshape(-1, 3),
                             np.asarray(coils.curves.gamma).reshape(-1, 3)])
    center = 0.5 * (points.min(axis=0) + points.max(axis=0)); span = np.ptp(points, axis=0).max()
    axis.set_xlim(center[0] - span / 2, center[0] + span / 2)
    axis.set_ylim(center[1] - span / 2, center[1] + span / 2)
    axis.set_zlim(center[2] - span / 2, center[2] + span / 2); axis.set_box_aspect((1, 1, 1))
    axis.set_title(title)
figure.tight_layout()
figure.savefig("single_stage_optimization.png", dpi=200)
plt.close(figure)

figure, axis = plt.subplots(figsize=(6.5, 4.0))
for name, values in history.items():
    axis.semilogy(values, label=name)
axis.set(xlabel="objective evaluation", ylabel="weighted cost", title="Single-stage objective terms")
axis.grid(True, alpha=0.3); axis.legend(fontsize=8, ncol=2); figure.tight_layout()
figure.savefig("single_stage_objectives.png", dpi=200); plt.close(figure)

# ESSOS writes |B| and B.n/B on the surface and the coil filaments for ParaView.
field_final = BiotSavart(coils_final)
surface_final.to_vtk("surface_single_stage_optimized", field=field_final)
coils_final.to_vtk("coils_single_stage_optimized")

final_value = float(result.fun)
qs_final = float(qs.total(final_equilibrium))
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} in {result.nit} {METHOD} iterations")
print(f"QA total = {qs_final:.6e}, aspect = {float(opt.aspect_ratio(final_equilibrium.state, final_equilibrium.runtime)):.4f}, "
      f"mean iota = {float(opt.mean_iota(final_equilibrium.state, final_equilibrium.runtime)):.4f}")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"Wrote {input_path}\nWrote {wout_path}")
print("Wrote coils_single_stage_optimized.json")
print("Wrote single_stage_optimization.png")
print("Wrote single_stage_objectives.png")
print("Wrote surface_single_stage_optimized.vts and coils_single_stage_optimized.vtu")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
