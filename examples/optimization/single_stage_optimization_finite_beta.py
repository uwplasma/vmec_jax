#!/usr/bin/env python
"""Finite-beta QA boundary, bootstrap-current, and ESSOS coil optimization."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from pyevtk.hl import gridToVTK
from scipy.optimize import minimize, OptimizeResult

import vmex as vj
from vmex import optimize as opt
from vmex.core import freeboundary_diff as fbd
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch,
                                 self_consistent_bootstrap)

import jax
import jax.numpy as jnp

from essos.coils import Coils, Curves, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import SurfaceRZFourier

nfp = 2
TARGET_BETA = 0.025
SURFACES = np.linspace(0.1, 0.9, 8)
MAX_MODE, MAXITER = 2, 60
N_CURRENT_SPLINE = 6
ASPECT_TARGET, IOTA_TARGET = 6.0, 0.42
VARY_MAJOR_RADIUS = False
SEED_PERTURBATION = 0.10

N_COILS, COIL_ORDER = 4, 4
COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS = 1.0, 0.5
COIL_CURRENT, N_SEGMENTS = 2.7e5, 64
NORMAL_FIELD_WEIGHT = 1.0e3
PRESSURE_BALANCE_WEIGHT = 1.0e2
LENGTH_TARGET, LENGTH_WEIGHT = 3.5, 1.0
CURVATURE_LIMIT, CURVATURE_WEIGHT = 7.0, 10.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.08, 1.0e3
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = 0.20, 1.0e3
NPHI, NTHETA, VC_DIGITS = 19, 24, 4
METHOD = "L-BFGS-B"
OPTIONS = {"maxiter": MAXITER, "maxls": 10, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 20}
MAKE_MOVIE = False  # set True for a compact GIF of accepted iterates

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES, MAX_MODE, MAXITER, N_CURRENT_SPLINE = np.linspace(0.2, 0.8, 4), 1, 0, 4
    COIL_ORDER, N_SEGMENTS, NPHI, NTHETA, VC_DIGITS = 2, 24, 8, 8, 3
    OPTIONS = {"maxiter": MAXITER, "maxls": 10, "ftol": 1e-8, "gtol": 1e-5}

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = 0.20
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_PERTURBATION, -SEED_PERTURBATION

# ne=n0(1-s^5), Te=Ti=T0(1-s), with p=e ne(Te+Ti) calibrated to TARGET_BETA.
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
am = np.zeros(21); am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
ac = np.zeros(21); ac[0] = 1.0
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
              pcurr_type="power_series", ac=ac, curtor=0.0).change_resolution(
                  mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
seed = opt.solve_equilibrium(inp)
profile_scale = TARGET_BETA / float(seed.wout.betatotal)
n0 *= profile_scale ** (1 / 3); T0 *= profile_scale ** (2 / 3)
inp = replace(inp, pres_scale=inp.pres_scale * profile_scale)
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=2 if ci_smoke else 8,
    tol=1e-3, degree=N_CURRENT_SPLINE - 1, s_eval=SURFACES, verbose=not ci_smoke)
# The Picard bootstrap solve leaves the prescribed pressure and boundary shape
# unchanged, and mainly updates the current profile / equilibrium state (I'(s), CURTOR)
# to the self-consistent bootstrap response before the optimization starts.
inp, equilibrium = opt.resample_current_profile(picard.input, N_CURRENT_SPLINE), picard.equilibrium

qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
bootstrap = RedlBootstrapMismatch(profiles, helicity_n=0, surfaces=SURFACES,
                                  n_lambda=12 if ci_smoke else 32)
plasma_terms = [
    (qs, 0.0, 1.0), (bootstrap, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 10.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2),
    (opt.mercier_stability_residual, 0.0, 1.0e-6),
    (opt.glasser_stability_residual, 0.0, 1.0e-6),
]
plasma_problem = opt.VmecProblem.from_tuples(inp, plasma_terms, max_mode=MAX_MODE,
    current_dofs=N_CURRENT_SPLINE - 1, vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
    restart_from=equilibrium, progress=not ci_smoke)

curves0 = CreateEquallySpacedCurves(N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=inp.nfp, stellsym=True)
coils0 = Coils(curves0, np.full(N_COILS, COIL_CURRENT))

def coil_lengths(coils, _surface):
    return coils.length[:N_COILS]

def coil_curvature_excess(coils, _surface):
    return jnp.maximum(coils.curvature[:N_COILS] - CURVATURE_LIMIT, 0.0)

def coil_distance_excess(coils, _surface):
    first, second = jnp.triu_indices(coils.gamma.shape[0], 1)
    distances = jnp.linalg.norm(
        coils.gamma[first, :, None] - coils.gamma[second, None, :], axis=-1)
    return jnp.maximum(COIL_DISTANCE_LIMIT - jnp.min(distances, axis=(1, 2)), 0.0)

def coil_surface_distance_excess(coils, surface):
    distances = jnp.linalg.norm(
        coils.gamma[:N_COILS, :, None] - surface.gamma.reshape(1, 1, -1, 3), axis=-1)
    return jnp.maximum(COIL_SURFACE_DISTANCE_LIMIT - jnp.min(distances, axis=2), 0.0).ravel()

def coil_field(coils):
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)

coil_terms = [
    (coil_lengths, LENGTH_TARGET, LENGTH_WEIGHT),
    (coil_curvature_excess, 0.0, CURVATURE_WEIGHT),
    (coil_distance_excess, 0.0, COIL_DISTANCE_WEIGHT),
    (coil_surface_distance_excess, 0.0, COIL_SURFACE_DISTANCE_WEIGHT),
]
extra_term_names = ("normal field", "pressure balance")
extra_term_names += tuple(function.__name__ for function, _target, _weight in coil_terms)

x_plasma0 = plasma_problem.x0
curve_shape, x_curves0 = curves0.dofs.shape, np.asarray(curves0.dofs).ravel()
x_currents0 = np.asarray(coils0.dofs_currents).ravel()
x0 = np.concatenate([x_plasma0, x_curves0, x_currents0])
fourier_names = ["0"] + [f"{kind}({order})" for order in range(1, COIL_ORDER + 1)
                          for kind in ("s", "c")]
coil_shape_names = tuple(f"coil[{coil}].{axis}{coefficient}" for coil in range(N_COILS)
                         for axis in "xyz" for coefficient in fourier_names)
dof_names = plasma_problem.dof_names + coil_shape_names + tuple(
    f"coil[{coil}].current" for coil in range(N_COILS))
plasma_scales = 0.02 * plasma_problem.scales
plasma_scales[-N_CURRENT_SPLINE:] = 0.05  # n-1 spline shapes + CURTOR
scales = np.concatenate([plasma_scales, np.full(x_curves0.size, 0.05),
                         np.full(x_currents0.size, 0.05)])

def surface_from_boundary(rbc, zbs, nphi=NPHI, ntheta=NTHETA):
    rc = jnp.concatenate([rbc[inp.ntor:, 0], rbc[:, 1:].T.ravel()])
    zs = jnp.concatenate([zbs[inp.ntor:, 0], zbs[:, 1:].T.ravel()])
    return SurfaceRZFourier(rc, zs, inp.nfp, inp.mpol - 1, inp.ntor,
                            nphi=nphi, ntheta=ntheta, close=False, range_torus="full torus")

def objects_from_x(x):
    end_curves = x_plasma0.size + x_curves0.size
    xp, xc, xi = x[:x_plasma0.size], x[x_plasma0.size:end_curves], x[end_curves:]
    rbc, zbs = plasma_problem.boundary_from_x(xp)
    surface = surface_from_boundary(rbc, zbs)
    coils = Coils(Curves(xc.reshape(curve_shape), N_SEGMENTS, inp.nfp, True),
                  xi * coils0.currents_scale, currents_scale=coils0.currents_scale)
    return surface, coils

state0, runtime0 = plasma_problem.metadata["jax_state_runtime"](jnp.asarray(x_plasma0))
surface_data0 = fbd.surface_field_data_from_state(
    inp, state0, runtime=runtime0, nphi=NPHI, ntheta=NTHETA)
precision = fbd.plan_vc_precision(surface_data0, digits=VC_DIGITS)

def objective(u):
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    xp = x[:x_plasma0.size]
    surface, coils = objects_from_x(x)
    state, runtime, status = plasma_problem.metadata["jax_state_runtime_status"](xp)

    def accepted(_):
        surface_data = fbd.surface_field_data_from_state(
            inp, state, runtime=runtime, nphi=NPHI, ntheta=NTHETA)
        vc = fbd.FreeBoundaryDiffProblem.from_surface_data(
            surface_data, digits=VC_DIGITS, precision=precision)
        B_scale = jnp.sqrt(jnp.sum(vc.weights * jnp.sum(surface_data.B_total**2, axis=0)))
        external_field = coil_field(coils)
        normal_rows = jnp.sqrt(vc.weights).ravel() * (
            vc.bnormal_residual(external_field) / B_scale).ravel()
        pressure_rows = jnp.sqrt(vc.weights).ravel() * (
            vc.pressure_balance_residual(external_field) / B_scale**2).ravel()
        coil_rows = [jnp.sqrt(weight) * (jnp.atleast_1d(function(coils, surface)) - target).ravel()
                     for function, target, weight in coil_terms]
        coil_costs = jnp.stack([0.5 * jnp.vdot(rows, rows) for rows in coil_rows])
        plasma_rows = plasma_problem.metadata["jax_residual_from_state"](state, runtime)
        normal_cost = 0.5 * NORMAL_FIELD_WEIGHT * jnp.vdot(normal_rows, normal_rows)
        pressure_cost = 0.5 * PRESSURE_BALANCE_WEIGHT * jnp.vdot(pressure_rows, pressure_rows)
        plasma_cost = 0.5 * jnp.vdot(plasma_rows, plasma_rows)
        return plasma_cost + normal_cost + pressure_cost + jnp.sum(coil_costs), \
            (plasma_rows, normal_cost, pressure_cost, coil_costs)

    def rejected(_):
        distance = jnp.linalg.norm(u)
        wall = 1.0e4 * (1.0 + distance)**2
        n_rows = plasma_problem.metadata["term_slices"][-1][2]
        return wall, (jnp.zeros(n_rows), wall, jnp.asarray(0.0), jnp.zeros(len(coil_terms)))

    return jax.lax.cond(status == 0, accepted, rejected, operand=None)

monitor = opt.OptimizationMonitor()
jax_value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
scipy_objective = monitor.wrap_value_and_grad(
    lambda u: jax_value_and_grad(jnp.asarray(u)), extra_term_names,
    residual_slices=plasma_problem.metadata["term_slices"])

print("Running single_stage_optimization_finite_beta.py")
print(f"Finite-beta VMEX + virtual casing + ESSOS: {x_plasma0.size} plasma, "
      f"{x_curves0.size} coil-shape, and {x_currents0.size} coil-current variables")
print(f"dof_names = {dof_names}")
joint_problem = vj.FunctionProblem.from_functions(
    np.zeros_like(x0), value_and_grad=scipy_objective)
joint_problem.compile_value_and_gradient(report_interval=10.0)
result = (OptimizeResult(x=joint_problem.x0, fun=monitor.records[-1].cost, nit=0)
          if ci_smoke else minimize(joint_problem.value_and_grad, joint_problem.x0,
              jac=True, method=METHOD, bounds=[(-3.0, 3.0)] * x0.size,
              callback=monitor, options=OPTIONS))

x_final = x0 + scales * result.x
surface_final, coils_final = objects_from_x(jnp.asarray(x_final))
equilibrium = plasma_problem.equilibrium_from_x(x_final[:x_plasma0.size])
final_input = replace(plasma_problem.input_from_x(x_final[:x_plasma0.size]),
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)

# Print results
report = opt.EquilibriumReporter(
    ("QA", qs.total, ".5e"), ("f_boot", bootstrap.total, ".5e"),
    ("beta", opt.volume_average_beta, ".3%"), ("aspect", opt.aspect_ratio, ".3f"),
    ("iota", opt.mean_iota, ".3f"))
report("final", final_equilibrium)
state_f, runtime_f = plasma_problem.metadata["jax_state_runtime"](jnp.asarray(x_final[:x_plasma0.size]))
data_f = fbd.surface_field_data_from_state(inp, state_f, runtime=runtime_f, nphi=NPHI, ntheta=NTHETA)
vc_f = fbd.FreeBoundaryDiffProblem.from_surface_data(data_f, digits=VC_DIGITS, precision=precision)
Bn = np.asarray(vc_f.bnormal_residual(coil_field(coils_final)))
Bmag = np.linalg.norm(np.asarray(data_f.B_total), axis=0)
Bn_over_B = Bn / Bmag
pressure_error = np.asarray(vc_f.pressure_balance_residual(coil_field(coils_final))) / Bmag**2
print(f"B.n/B RMS = {100 * np.sqrt(np.sum(np.asarray(vc_f.weights) * Bn_over_B**2)):.3f}%, "
      f"max = {100 * np.max(np.abs(Bn_over_B)):.3f}%")
print(f"Pressure-balance RMS = {np.sqrt(np.sum(np.asarray(vc_f.weights) * pressure_error**2)):.3e}")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"Maximum curvature = {float(np.max(np.asarray(coils_final.curvature))):.3f} 1/m")

# Save results
input_path = final_input.to_indata("input.single_stage_finite_beta_optimized")
wout_path = vj.write_wout("wout_single_stage_finite_beta_optimized.nc", final_equilibrium.wout)
coils_final.to_json("coils_single_stage_finite_beta_optimized.json")
vc0 = fbd.FreeBoundaryDiffProblem.from_surface_data(
    surface_data0, digits=VC_DIGITS, precision=precision)
Bn0 = np.asarray(vc0.bnormal_residual(coil_field(coils0)))
Bmag0 = np.linalg.norm(np.asarray(surface_data0.B_total), axis=0)
pressure_error0 = np.asarray(vc0.pressure_balance_residual(coil_field(coils0))) / Bmag0**2
gamma0 = np.asarray(surface_data0.gamma)
gridToVTK("surface_single_stage_finite_beta_initial",
    gamma0[0][None].copy(), gamma0[1][None].copy(), gamma0[2][None].copy(),
    pointData={"B_dot_n_over_B": (Bn0 / Bmag0)[None].copy(), "B": Bmag0[None].copy(),
               "pressure_balance_error": pressure_error0[None].copy()})
coils0.to_vtk("coils_single_stage_finite_beta_initial")
gamma = np.asarray(data_f.gamma)
gridToVTK("surface_single_stage_finite_beta_optimized",
    gamma[0][None].copy(), gamma[1][None].copy(), gamma[2][None].copy(),
    pointData={"B_dot_n_over_B": Bn_over_B[None].copy(), "B": Bmag[None].copy(),
               "pressure_balance_error": pressure_error[None].copy()})
coils_final.to_vtk("coils_single_stage_finite_beta_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")

# Plot results
monitor.save("single_stage_finite_beta_objectives.csv")
monitor.plot("single_stage_finite_beta_objectives.png", title="Finite-beta single-stage terms")
vj.plot_optimization_objects("single_stage_finite_beta_optimization.png",
    ("Initial", *objects_from_x(jnp.asarray(x0))), ("Optimized", surface_final, coils_final))
if MAKE_MOVIE:
    monitor.movie("single_stage_finite_beta_optimization.gif",
        lambda u: objects_from_x(jnp.asarray(x0 + scales * u)))
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
