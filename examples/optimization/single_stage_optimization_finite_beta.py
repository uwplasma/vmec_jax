#!/usr/bin/env python
"""Fixed-boundary finite-beta QA, bootstrap-current, and coil optimization.

The boundary is varied but each VMEX evaluation is a fixed-boundary solve.
Virtual casing only separates the converged total field into its plasma and
required external-coil parts; no free-boundary equilibrium is solved here.
Use the commented ``Coils.from_simsopt`` line to load a SIMSOPT coil JSON.
"""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
# This prescribed-interface API separates plasma and coil fields on each trial
# boundary. It does not move the boundary or run a free-boundary equilibrium.
from vmex.core import virtual_casing as vc
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch,
                                 self_consistent_bootstrap)

import jax
import jax.numpy as jnp

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import surfacerzfourier_from_boundary

nfp = 2  # number of field periods
MAKE_MOVIE = True  # set True for a compact GIF of accepted iterates
# Surface colors: None, "absB", "B.n/B", or a callable ``(x, objects) -> values``.
MOVIE_SURFACE_COLOR = "absB"

TARGET_BETA = 0.025
SURFACES = np.linspace(0.1, 0.9, 8)
MAX_MODE, MAXITER = 2, 25
N_CURRENT_SPLINE = 6
ASPECT_TARGET, IOTA_FLOOR = 6.0, 0.42
VARY_MAJOR_RADIUS = False
SEED_PERTURBATION = 0.02

N_COILS, COIL_ORDER = 3, 4
COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS = 1.0, 0.5
COIL_CURRENT, N_SEGMENTS = 2.7e5, 64
NORMAL_FIELD_WEIGHT = 1.0e3
# The pressure-balance residual is (|B_out|^2-|B_in|^2-2 mu0 p_edge)/B_ref^2.
# Its weighted squared cost enforces the ideal-MHD interface condition in
# addition to B.n=0; it is not a penalty on the volume pressure profile.
PRESSURE_BALANCE_WEIGHT = 1.0e2
LENGTH_TARGET, LENGTH_WEIGHT = 3.5, 1.0
CURVATURE_LIMIT, CURVATURE_WEIGHT = 7.0, 10.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.08, 1.0e3
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = 0.20, 1.0e3
# This resolves the mode-2 boundary and order-4 coils; the final diagnostics
# below use an independent finer grid, so an aliased optimum is not accepted.
NPHI, NTHETA, VC_DIGITS = 16, 16, 4
FINAL_NPHI, FINAL_NTHETA = 32, 32
METHOD = "L-BFGS-B"
PARAMETER_BOUND = 3.0
OPTIONS = {"maxiter": MAXITER, "maxls": 10, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 20}

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES, MAX_MODE, MAXITER, N_CURRENT_SPLINE = np.linspace(0.2, 0.8, 4), 1, 0, 4
    COIL_ORDER, N_SEGMENTS, NPHI, NTHETA, VC_DIGITS = 2, 24, 8, 8, 3
    FINAL_NPHI = FINAL_NTHETA = 8
    METHOD, MAKE_MOVIE = "BFGS", False
    OPTIONS = {"maxiter": MAXITER, "gtol": 1e-5}

print("Running single_stage_optimization_finite_beta.py", flush=True)
DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# VmecInput is frozen, so copy its arrays before adding the 3-D perturbation.
# The circular RBC(0,1)=ZBS(0,1)=0.1 seed is already explicit in the input file.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_PERTURBATION, -SEED_PERTURBATION

# ne=n0(1-s^5), Te=Ti=T0(1-s), with p=e ne(Te+Ti) calibrated to TARGET_BETA.
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
# VMEC stores at least 21 power-series coefficients. AM below is
# (1-s)(1-s^5), matching the kinetic pressure shape; AC starts from constant I'(s).
am = np.zeros(21); am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
ac = np.zeros(21); ac[0] = 1.0
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
              pcurr_type="power_series", ac=ac, curtor=0.0).change_resolution(
                  mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
print("Calibrating the finite-beta seed...", flush=True)
# Measure this seed instead of carrying a geometry-dependent fitted constant.
profile_scale = TARGET_BETA / float(opt.solve_equilibrium(inp).wout.betatotal)
n0 *= profile_scale ** (1 / 3); T0 *= profile_scale ** (2 / 3)
inp = replace(inp, pres_scale=inp.pres_scale * profile_scale)
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
print("Computing the self-consistent bootstrap-current seed...", flush=True)
picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=1 if ci_smoke else 8,
    tol=1e-3, degree=N_CURRENT_SPLINE - 1, s_eval=SURFACES, verbose=not ci_smoke)
# The Picard bootstrap solve leaves the prescribed pressure and boundary shape
# unchanged, and mainly updates the current profile / equilibrium state (I'(s), CURTOR)
# to the self-consistent bootstrap response before the optimization starts.
inp, equilibrium = opt.resample_current_profile(picard.input, N_CURRENT_SPLINE), picard.equilibrium

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
bootstrap = RedlBootstrapMismatch(profiles, helicity_n=0, surfaces=SURFACES,
                                  n_lambda=12 if ci_smoke else 32)
plasma_terms = [
    (qs, 0.0, 1.0), (bootstrap, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2),
    # (opt.mercier_stability_residual, 0.0, 1.0e-6),
    # (opt.glasser_stability_residual, 0.0, 1.0e-6),
]
plasma_problem = opt.VmecProblem.from_tuples(inp, plasma_terms, max_mode=MAX_MODE,
    current_dofs=N_CURRENT_SPLINE - 1, vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
    restart_from=equilibrium, progress=not ci_smoke)

curves0 = CreateEquallySpacedCurves(N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=inp.nfp, stellsym=True)
coils0 = Coils(curves0, np.full(N_COILS, COIL_CURRENT))
# To start from a SIMSOPT coil file instead, use:
# coils0 = Coils.from_simsopt("coils.json", nfp=inp.nfp, stellsym=True)
# curves0 = coils0.curves

def coil_lengths(coils, _surface):
    return coils.length[:N_COILS]

def coil_curvature_excess(coils, _surface):
    return jnp.maximum(coils.curvature[:N_COILS] - CURVATURE_LIMIT, 0.0)

def coil_field(coils):
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)

extra_term_names = ("normal field", "pressure balance", "coil length",
                    "coil curvature", "coil separation", "coil-surface separation")

x_plasma0 = plasma_problem.x0
n_curve_dofs = curves0.dofs.size
x_coils0 = np.asarray(coils0.dofs)
x0 = np.concatenate([x_plasma0, x_coils0])
dof_names = plasma_problem.dof_names + coils0.dof_names
plasma_scales = 0.02 * plasma_problem.scales
plasma_scales[-N_CURRENT_SPLINE:] = 0.05  # n-1 spline shapes + CURTOR
scales = np.concatenate([plasma_scales, np.full(n_curve_dofs, 0.05),
                         np.full(coils0.dofs_currents.size, 0.05)])

def objects_from_x(x):
    end_curves = x_plasma0.size + n_curve_dofs
    xp, xc, xi = x[:x_plasma0.size], x[x_plasma0.size:end_curves], x[end_curves:]
    rbc, zbs = plasma_problem.boundary_from_x(xp)
    surface = surfacerzfourier_from_boundary(
        rbc, zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)
    coils = coils0.with_dofs(jnp.concatenate((xc, xi)))
    return surface, coils


print("Preparing virtual casing on the initial surface...", flush=True)
state0, runtime0 = plasma_problem.metadata["jax_state_runtime"](jnp.asarray(x_plasma0))
surface_data0 = vc.surface_field_data_from_state(
    inp, state0, runtime=runtime0, nphi=NPHI, ntheta=NTHETA)
precision = vc.plan_vc_precision(surface_data0, digits=VC_DIGITS)

def interface_costs(x, equilibrium_state, solver_context):
    surface, coils = objects_from_x(x)
    surface_data = vc.surface_field_data_from_state(
        inp, equilibrium_state, runtime=solver_context, nphi=NPHI, ntheta=NTHETA)
    # This object evaluates the fixed trial surface's virtual-casing residuals;
    # VMEX has already solved the volume equilibrium with that boundary prescribed.
    interface = vc.PlasmaVacuumInterface.from_surface_data(
        surface_data, digits=VC_DIGITS, precision=precision)
    B_scale = jnp.sqrt(jnp.sum(interface.weights * jnp.sum(surface_data.B_total**2, axis=0)))
    external_field = coil_field(coils)
    normal_rows = jnp.sqrt(interface.weights).ravel() * (
        interface.bnormal_residual(external_field) / B_scale).ravel()
    # Why this term exists, when the SIMSOPT single stage carries only the
    # normal-field one: at zero beta the plasma exerts no pressure, so a
    # boundary with B.n = 0 is already a force-free interface and matching the
    # normal field is the whole condition. At finite beta the jump condition
    # is total-pressure continuity, p + |B_in|^2/2mu0 = |B_out|^2/2mu0, and
    # B.n = 0 does not imply it: the coils must also supply the field strength
    # that holds the edge pressure. Optimizing the normal field alone yields
    # coils whose surface is the right shape and the wrong pressure, so the
    # free-boundary solve relaxes away from the boundary that was optimized.
    # These rows are the dimensionless RMS of that jump, not a profile error.
    pressure_rows = jnp.sqrt(interface.weights).ravel() * (
        interface.pressure_balance_residual(external_field) / B_scale**2).ravel()
    return jnp.asarray([
        0.5 * NORMAL_FIELD_WEIGHT * jnp.vdot(normal_rows, normal_rows),
        0.5 * PRESSURE_BALANCE_WEIGHT * jnp.vdot(pressure_rows, pressure_rows),
    ])

def geometry_costs(x):
    surface, coils = objects_from_x(x)
    length_rows = coil_lengths(coils, surface) - LENGTH_TARGET
    curvature_rows = coil_curvature_excess(coils, surface)
    return jnp.asarray([
        0.5 * LENGTH_WEIGHT * jnp.vdot(length_rows, length_rows),
        0.5 * CURVATURE_WEIGHT * jnp.vdot(curvature_rows, curvature_rows),
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_LIMIT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_LIMIT, block_size=32),
    ])

def physics_objective(u):
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    return plasma_problem.jax_objective_from_state(
        x[:x_plasma0.size], lambda equilibrium_state, solver_context:
            interface_costs(x, equilibrium_state, solver_context),
        n_extra_terms=2)

def geometry_objective(u):
    costs = geometry_costs(jnp.asarray(x0) + jnp.asarray(scales) * u)
    return jnp.sum(costs), costs

monitor = opt.OptimizationMonitor()
physics_value_and_grad = jax.jit(jax.value_and_grad(physics_objective, has_aux=True))
geometry_value_and_grad = jax.jit(jax.value_and_grad(geometry_objective, has_aux=True))

# JAX differentiates the fixed-boundary VMEX/virtual-casing physics and ESSOS
# geometry graphs separately to limit memory; their scalar gradients simply add.
def value_and_grad(u):
    (physics_value, (residual, interface_values)), physics_gradient = physics_value_and_grad(u)
    (geometry_value, geometry_values), geometry_gradient = geometry_value_and_grad(u)
    residual = np.asarray(residual)
    terms = {name: 0.5 * float(residual[start:stop] @ residual[start:stop])
             for name, start, stop in plasma_problem.metadata["term_slices"]}
    extra_values = np.concatenate((np.asarray(interface_values), np.asarray(geometry_values)))
    terms.update(zip(extra_term_names, map(float, extra_values)))
    return monitor.cache_evaluation(
        u, physics_value + geometry_value, physics_gradient + geometry_gradient, terms)

print(f"Finite-beta VMEX + virtual casing + ESSOS: {x_plasma0.size} plasma, "
      f"{n_curve_dofs} coil-shape, and {coils0.dofs_currents.size} coil-current variables")
print(f"dof_names = {dof_names}")
joint_problem = vj.FunctionProblem.from_functions(
    np.zeros_like(x0), value_and_grad=value_and_grad)
joint_problem.compile_value_and_gradient(report_interval=10.0)
result = minimize(joint_problem.value_and_grad, joint_problem.x0, jac=True, method=METHOD,
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size if METHOD == "L-BFGS-B" else None,
    callback=monitor, options=OPTIONS)

x_final = x0 + scales * result.x
_, coils_final = objects_from_x(jnp.asarray(x_final))
equilibrium = plasma_problem.equilibrium_from_x(x_final[:x_plasma0.size])
final_input = plasma_problem.input_from_x(x_final[:x_plasma0.size])
if ci_smoke:
    final_equilibrium = equilibrium
else:
    final_input = replace(final_input, ns_array=np.array([31, 51, 101]),
        ftol_array=np.array([1e-10, 1e-12, 1e-14]), niter_array=np.full(3, 20000))
    final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
        verbose=True, raise_on_max_iterations=True)

# Print results
report = opt.EquilibriumReporter(
    ("QA", qs.total, ".5e"), ("f_boot", bootstrap.total, ".5e"),
    ("beta", opt.volume_average_beta, ".3%"), ("aspect", opt.aspect_ratio, ".3f"),
    ("iota", opt.mean_iota, ".3f"))
report("final", final_equilibrium)
data_f = vc.surface_field_data_from_state(
    final_input, final_equilibrium.solution, runtime=final_equilibrium.solver_context,
    nphi=FINAL_NPHI, ntheta=FINAL_NTHETA)
final_precision = vc.plan_vc_precision(data_f, digits=VC_DIGITS)
interface_f = vc.PlasmaVacuumInterface.from_surface_data(
    data_f, digits=VC_DIGITS, precision=final_precision)
Bn = np.asarray(interface_f.bnormal_residual(coil_field(coils_final)))
Bmag = np.linalg.norm(np.asarray(data_f.B_total), axis=0)
Bn_over_B = Bn / Bmag
pressure_error = np.asarray(
    interface_f.pressure_balance_residual(coil_field(coils_final))) / Bmag**2
surface_final = surfacerzfourier_from_boundary(
    jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs), inp.nfp,
    nphi=FINAL_NPHI, ntheta=FINAL_NTHETA)
print(f"B.n/B RMS = {100 * np.sqrt(np.sum(np.asarray(interface_f.weights) * Bn_over_B**2)):.3f}%, "
      f"max = {100 * np.max(np.abs(Bn_over_B)):.3f}%")
print("Normalized total-pressure jump RMS = "
      f"{np.sqrt(np.sum(np.asarray(interface_f.weights) * pressure_error**2)):.3e} (target 0)")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"Maximum curvature = {float(np.max(np.asarray(coils_final.curvature))):.3f} 1/m")

# Save results
input_path = final_input.to_indata("input.single_stage_finite_beta_optimized")
wout_path = vj.write_wout("wout_single_stage_finite_beta_optimized.nc", final_equilibrium.wout)
coils_final.to_json("coils_single_stage_finite_beta_optimized.json")
interface0 = vc.PlasmaVacuumInterface.from_surface_data(
    surface_data0, digits=VC_DIGITS, precision=precision)
Bn0 = np.asarray(interface0.bnormal_residual(coil_field(coils0)))
Bmag0 = np.linalg.norm(np.asarray(surface_data0.B_total), axis=0)
pressure_error0 = np.asarray(interface0.pressure_balance_residual(coil_field(coils0))) / Bmag0**2
surface_initial, _ = objects_from_x(jnp.asarray(x0))
surface_initial.to_vtk("surface_single_stage_finite_beta_initial", extra_data={
    "B_dot_n_over_B": (Bn0 / Bmag0)[None].copy(), "B": Bmag0[None].copy(),
    "pressure_balance_error": pressure_error0[None].copy()})
coils0.to_vtk("coils_single_stage_finite_beta_initial")
surface_final.to_vtk("surface_single_stage_finite_beta_optimized", extra_data={
    "B_dot_n_over_B": Bn_over_B[None].copy(), "B": Bmag[None].copy(),
    "pressure_balance_error": pressure_error[None].copy()})
coils_final.to_vtk("coils_single_stage_finite_beta_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")

# Plot results
monitor.save("single_stage_finite_beta_objectives.csv")
monitor.plot("single_stage_finite_beta_objectives.png", title="Finite-beta single-stage terms")
vj.plot_bootstrap_current("single_stage_finite_beta_bootstrap_current.png",
                          final_equilibrium, bootstrap)
vj.plot_optimization_objects("single_stage_finite_beta_optimization.png",
    ("Initial", *objects_from_x(jnp.asarray(x0))), ("Optimized", surface_final, coils_final))
print("Wrote single_stage_finite_beta_optimization.png")
print("Wrote single_stage_finite_beta_objectives.csv and single_stage_finite_beta_objectives.png")
print("Wrote single_stage_finite_beta_bootstrap_current.png")
if MAKE_MOVIE:
    print("Making movie of accepted iterates...")
    monitor.movie_surface_coils("single_stage_finite_beta_optimization.gif", objects_from_x,
        x0=x0, scales=scales, surface_color=MOVIE_SURFACE_COLOR, plasma_problem=plasma_problem,
        external_field=lambda objects: coil_field(objects[1]), nphi=NPHI, ntheta=NTHETA,
        digits=VC_DIGITS, precision=precision, cmap="jet")
if not ci_smoke:
    for path in vj.plot_wout(wout_path, ".").values():
        print(f"Wrote {path}")
