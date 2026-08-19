#!/usr/bin/env python
"""Single-stage fixed-boundary plasma and ESSOS coil optimization.

Use the commented ``Coils.from_simsopt`` line to replace the generated coils
with a SIMSOPT coil JSON while keeping the objective and derivative code.
"""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt

import jax
import jax.numpy as jnp

try:
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    from essos.surfaces import surfacerzfourier_from_boundary
except ImportError:
    raise ImportError(
        "The single-stage optimization example requires ESSOS. "
        "Install with `pip install essos` or `conda install -c conda-forge essos`."
    )

nfp = 2  # number of field periods
MAKE_MOVIE = True  # set True for a compact GIF of accepted iterates
# Surface colors: None, "absB", "B.n/B", or a callable ``(x, objects) -> values``.
MOVIE_SURFACE_COLOR = "absB"

SURFACES = np.linspace(0.05, 1.0, 6)
MAX_MODE = 3
MAXITER = 200
ASPECT_TARGET = 4.0
ASPECT_WEIGHT = 1.0
IOTA_FLOOR = 0.42
IOTA_WEIGHT = 100.0
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.02

N_COILS = 4
COIL_ORDER = 5
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64
STELLSYM = True

NORMAL_FIELD_WEIGHT = 1.0e3
NORMAL_FIELD_LIMIT = 0.01
NORMAL_FIELD_OBJECTIVE_LIMIT = 0.008  # margin for the independent final grid
NORMAL_FIELD_LIMIT_WEIGHT = 2.0e5
LENGTH_TARGET = 3.3
LENGTH_WEIGHT = 1.0
CURVATURE_LIMIT = 7.0
CURVATURE_OBJECTIVE_LIMIT = 6.9  # margin for the independent final grid
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_LIMIT = 0.15
COIL_DISTANCE_WEIGHT = 1.0e3
COIL_SURFACE_DISTANCE_LIMIT = 0.20
COIL_SURFACE_DISTANCE_WEIGHT = 1.0e3

# A toroidal grid commensurate with the coil count can alias narrow B.n/B structure.
NPHI, NTHETA = 37, 32
METHOD = "BFGS"  # also accepts "L-BFGS-B"
PARAMETER_BOUND = 3.0
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
# VmecInput is frozen, so copy its arrays before adding the 3-D perturbation.
# The circular RBC(0,1)=ZBS(0,1)=0.1 seed is already explicit in the input file.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_PERTURBATION, -SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
plasma_terms = [
    (qs.residuals_state, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (iota_floor, 0.0, IOTA_WEIGHT),
]
plasma_problem = opt.VmecProblem.from_tuples(
    inp, plasma_terms, max_mode=MAX_MODE, vary_major_radius=VARY_MAJOR_RADIUS,
    use_ess=True, progress=not ci_smoke)

curves0 = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=inp.nfp, stellsym=STELLSYM)
coils0 = Coils(curves0, np.full(N_COILS, COIL_CURRENT))
# To start from a SIMSOPT coil file instead, use:
# coils0 = Coils.from_simsopt("coils.json", nfp=inp.nfp, stellsym=STELLSYM)
# curves0 = coils0.curves

def normalized_normal_field(coils, surface):
    field = BiotSavart(coils)
    magnetic_field = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    return jnp.sum(magnetic_field * surface.unitnormal, axis=2) / jnp.linalg.norm(magnetic_field, axis=2)

def coil_field(coils):
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)

def normal_field_residual(coils, surface):
    weights = surface.area_element / jnp.sum(surface.area_element)
    values = normalized_normal_field(coils, surface)
    return (jnp.sqrt(weights) * values).ravel()

def normal_field_excess(coils, surface):
    values = jnp.sqrt(normalized_normal_field(coils, surface)**2 + 1.0e-12)
    smooth_maximum = jax.scipy.special.logsumexp(2000.0 * values) / 2000.0
    return jnp.maximum(smooth_maximum - NORMAL_FIELD_OBJECTIVE_LIMIT, 0.0)

def coil_lengths(coils, _surface):
    return coils.length[:N_COILS]

def coil_curvature_excess(coils, _surface):
    return jnp.maximum(coils.curvature[:N_COILS] - CURVATURE_OBJECTIVE_LIMIT, 0.0)

coil_terms = [
    (normal_field_residual, 0.0, NORMAL_FIELD_WEIGHT),
    (normal_field_excess, 0.0, NORMAL_FIELD_LIMIT_WEIGHT),
    (coil_lengths, LENGTH_TARGET, LENGTH_WEIGHT),
    (coil_curvature_excess, 0.0, CURVATURE_WEIGHT),
]
coil_term_names = tuple(function.__name__ for function, _target, _weight in coil_terms) + (
    "coil separation", "coil-surface separation")

# The public VMEX problem owns the boundary-mode convention and RBC(0,0) choice.
x_boundary0 = plasma_problem.x0
rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)

n_curve_dofs = curves0.dofs.size
x_coils0 = np.asarray(curves0.dofs).ravel()
x0 = np.concatenate([x_boundary0, x_coils0])
dof_names = plasma_problem.dof_names + curves0.dof_names

# SciPy works in dimensionless increments u, with x = x0 + scales*u.
scales = np.concatenate([
    0.02 * plasma_problem.scales, np.full(n_curve_dofs, 0.05)])


def objects_from_x(x):
    x_boundary, x_coils = x[:x_boundary0.size], x[x_boundary0.size:]
    rbc, zbs = plasma_problem.boundary_from_x(x_boundary)
    surface = surfacerzfourier_from_boundary(
        rbc, zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)

    coils = coils0.with_dofs(jnp.concatenate((x_coils, coils0.dofs_currents)))
    return surface, coils


def coil_costs(x):
    surface, coils = objects_from_x(x)
    rows = [jnp.sqrt(weight) * (jnp.atleast_1d(function(coils, surface)) - target).ravel()
            for function, target, weight in coil_terms]
    return jnp.concatenate([jnp.stack([0.5 * jnp.vdot(row, row) for row in rows]), jnp.asarray([
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_LIMIT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_LIMIT, block_size=32),
    ])])

def plasma_component(u):
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    x_boundary = x[:x_boundary0.size]
    value, gradient = plasma_problem.jax_value_and_grad(x_boundary)
    residual = plasma_problem.jax_residual(x_boundary)
    scaled_gradient = gradient * jnp.asarray(scales[:x_boundary0.size])
    return (value, residual), jnp.pad(scaled_gradient, (0, x0.size - x_boundary0.size))

def coil_objective(u):
    costs = coil_costs(jnp.asarray(x0) + jnp.asarray(scales) * u)
    return jnp.sum(costs), costs


monitor = opt.OptimizationMonitor()
plasma_value_and_grad = jax.jit(plasma_component)
coil_value_and_grad = jax.jit(jax.value_and_grad(coil_objective, has_aux=True))

# VMEX supplies the exact equilibrium derivative; JAX differentiates the coil
# objective. Their values and gradients add directly for any SciPy optimizer.
def value_and_grad(u):
    (plasma_value, residual), plasma_gradient = plasma_value_and_grad(u)
    (coil_value, coil_cost_values), coil_gradient = coil_value_and_grad(u)
    residual = np.asarray(residual)
    terms = {name: 0.5 * float(residual[start:stop] @ residual[start:stop])
             for name, start, stop in plasma_problem.metadata["term_slices"]}
    terms.update(zip(coil_term_names, map(float, np.asarray(coil_cost_values))))
    return monitor.cache_evaluation(
        u, plasma_value + coil_value, plasma_gradient + coil_gradient, terms)


print("Running single_stage_optimization.py")
print(f"Fixed-boundary VMEX + ESSOS: {x_boundary0.size} boundary and "
      f"{x_coils0.size} coil variables, exact reverse-mode derivatives")
print(f"dof_names = {dof_names}")
joint_problem = vj.FunctionProblem.from_functions(
    np.zeros_like(x0), value_and_grad=value_and_grad)
joint_problem.compile_value_and_gradient(report_interval=10.0)
result = minimize(joint_problem.value_and_grad, joint_problem.x0,
                  jac=True, method=METHOD,
                  bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size if METHOD == "L-BFGS-B" else None,
                  callback=monitor, options=OPTIONS)
initial_value = monitor.records[0].cost

x_final = x0 + scales * result.x
_, coils_final = objects_from_x(jnp.asarray(x_final))
equilibrium = plasma_problem.equilibrium_from_x(x_final[:x_boundary0.size])
final_input = plasma_problem.input_from_x(x_final[:x_boundary0.size])
final_input = replace(final_input,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)

surface_final = surfacerzfourier_from_boundary(
    jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs), inp.nfp,
    nphi=61, ntheta=64)
normal_field = np.asarray(normalized_normal_field(coils_final, surface_final))
area_weights = np.asarray(surface_final.area_element); area_weights = area_weights / area_weights.sum()
normal_field_rms = float(np.sqrt(np.sum(area_weights * normal_field**2)))
normal_field_max = float(np.max(np.abs(normal_field)))
coil_points, surface_points = np.asarray(coils_final.gamma), np.asarray(surface_final.gamma).reshape(-1, 3)
coil_surface_distance = min(float(np.linalg.norm(points[:, None] - surface_points[None], axis=2).min())
                            for points in coil_points)
coil_pairs = [(i, j) for i in range(len(coil_points)) for j in range(i + 1, len(coil_points))]
coil_distance = min(float(np.linalg.norm(coil_points[i][:, None] - coil_points[j][None], axis=2).min())
                    for i, j in coil_pairs)
maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))

# Print results
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"))
final_value = float(result.fun)
report("final", final_equilibrium)
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} in {result.nit} {METHOD} iterations")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"B.n/B: area-weighted RMS = {100 * normal_field_rms:.3f}%, max = {100 * normal_field_max:.3f}% "
      f"(target < {100 * NORMAL_FIELD_LIMIT:.1f}%)")
print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m "
      f"(target >= {COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {COIL_DISTANCE_LIMIT:.4f} m)")
print(f"Maximum curvature = {maximum_curvature:.4f} 1/m (target <= {CURVATURE_LIMIT:.4f} 1/m)")

# Save results
input_path = final_input.to_indata("input.single_stage_optimized")
wout_path = vj.write_wout("wout_single_stage_optimized.nc", final_equilibrium.wout)
coils_final.to_json("coils_single_stage_optimized.json")
# ESSOS writes |B| and B.n/B on the surface and the coil filaments for ParaView.
surface_initial = surfacerzfourier_from_boundary(
    rbc0, zbs0, inp.nfp, nphi=60, ntheta=60)
surface_initial.to_vtk("surface_single_stage_initial", field=BiotSavart(coils0))
coils0.to_vtk("coils_single_stage_initial")
field_final = BiotSavart(coils_final)
surface_final.to_vtk("surface_single_stage_optimized", field=field_final)
coils_final.to_vtk("coils_single_stage_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")
print("Wrote coils_single_stage_optimized.json")
print("Wrote initial and optimized surface/coils VTK files")

# Plot results
print("Plotting results...")
vj.plot_optimization_objects("single_stage_optimization.png",
    ("Initial", surface_initial, coils0), ("Optimized", surface_final, coils_final))
monitor.save("single_stage_objectives.csv")
monitor.plot("single_stage_objectives.png", title="Single-stage objective terms")
print("Wrote single_stage_optimization.png")
print("Wrote single_stage_objectives.csv and single_stage_objectives.png")
if MAKE_MOVIE:
    print("Making movie of accepted iterates...")
    monitor.movie_surface_coils("single_stage_optimization.gif", objects_from_x,
        x0=x0, scales=scales, surface_color=MOVIE_SURFACE_COLOR, plasma_problem=plasma_problem,
        external_field=lambda objects: coil_field(objects[1]), nphi=NPHI, ntheta=NTHETA, cmap="jet")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
