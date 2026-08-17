#!/usr/bin/env python
"""Finite-beta QA optimization with an outer-radius favorable-J continuation.

Exact maximum-J is incompatible with quasisymmetry near the magnetic axis.
Pressure can reverse the trapped-particle precession in the outer volume, so
this example reports the retained QA, bootstrap, and maximum-J fractions.
"""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles,
                                 RedlBootstrapMismatch, self_consistent_bootstrap)
from vmex.core.maxj import MaximumJResidual, common_trapped_pitches_state

nfp = 2
SURFACES = np.array([0.6, 0.7, 0.8, 0.9])
QA_SURFACES = np.linspace(0.1, 0.9, 8)
QA_MAX_MODES, QA_MAX_NFEV = [2, 3, 4], [25, 40, 50]
MAXJ_MAX_MODES, MAXJ_MAX_NFEV = [3, 4], [40, 60]
ASPECT_TARGET, IOTA_TARGET, MAGNETIC_WELL_TARGET = 5.0, 0.42, 0.01
TARGET_BETA, TRAPPING_DEPTHS, MAXJ_TARGET = 0.025, (0.4, 0.8), -0.01
BETA_WEIGHT = 1.0 / TARGET_BETA**2
MAXJ_WEIGHT, QA_WEIGHT, BOOTSTRAP_WEIGHT, WELL_WEIGHT = 100.0, 10.0, 1.0, 10.0
MINIMUM_MPOL, SEED_PERTURBATION = 5, 0.05
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
CURRENT_PARAMETER_STEP, N_CURRENT_SPLINE = 0.05, 8
ACTION_NALPHA, ACTION_POINTS, ACTION_PERIODS = 7, 32, 8
ACTION_MAX_WELLS, ACTION_QUADRATURE = 20, 24

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES, QA_SURFACES, TRAPPING_DEPTHS, MINIMUM_MPOL = (
        np.array([0.6, 0.7, 0.8, 0.9]), np.linspace(0.2, 0.8, 4), (0.4, 0.8), 3)
    QA_MAX_MODES, QA_MAX_NFEV = [1], [2]
    MAXJ_MAX_MODES, MAXJ_MAX_NFEV = [1], [2]
    ACTION_NALPHA, ACTION_POINTS, ACTION_QUADRATURE = 7, 32, 16

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
if ci_smoke:
    inp = replace(inp, ns_array=np.array([11]), ftol_array=np.array([1e-8]),
                  niter_array=np.array([1500]))
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)
equilibrium = opt.solve_equilibrium(inp)

qs = opt.QuasisymmetryRatioResidual(QA_SURFACES, helicity_m=1, helicity_n=0)
shape_terms = [(qs, 0.0, QA_WEIGHT), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 10.0), (opt.magnetic_well, MAGNETIC_WELL_TARGET, WELL_WEIGHT),
]
report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("aspect", opt.aspect_ratio, ".3f"),
    ("iota", opt.mean_iota, ".3f"),
    ("magnetic well", opt.magnetic_well, ".3f"))
monitor = opt.OptimizationMonitor(stream=None)

for max_mode, max_nfev in zip(QA_MAX_MODES, QA_MAX_NFEV):
    print(f"\n===== vacuum QA seed stage, max_mode = {max_mode} =====")
    mpol, ntor = max(inp.mpol, max_mode + 2, MINIMUM_MPOL), max(inp.ntor, max_mode + 2)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6, nzeta=2 * ntor + 4)
    problem = opt.VmecProblem.from_tuples(inp, shape_terms, max_mode=max_mode,
        use_ess=True, restart_from=equilibrium,
        forward_max_iterations=100 if ci_smoke else 3000,
        progress=True)
    problem.compile_residual_and_jacobian()
    monitor.problem = problem
    step = (0.001 if ci_smoke else PARAMETER_STEP) * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp, equilibrium = problem.input_from_x(result.x), problem.equilibrium_from_x(result.x)
    report(f"QA mode {max_mode}", equilibrium)

if ci_smoke:
    # Keep the smoke lane short: minimal-seed QA wiring was exercised above;
    # a bundled self-consistent finite-beta QA state supplies common wells for
    # the maximum-J/current-profile API.
    inp = vj.VmecInput.from_file(
        Path(__file__).resolve().parents[1] / "data"
        / "input.LandremanPaul2021_QA_beta2p5_bootstrap")
    equilibrium = opt.solve_equilibrium(inp)

# Add the Landreman--Buller--Drevlak pressure profiles to the optimized QA
# boundary, then alternate hot-started VMEX solves and Redl current updates.
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
if not ci_smoke:
    am = np.zeros(21); am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
    ac = np.zeros(21); ac[0] = 1.0
    inp = replace(inp, pmass_type="power_series", am=am,
        pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
        pcurr_type="power_series", ac=ac, curtor=0.0)
    calibration = opt.solve_equilibrium(inp, initial_state=equilibrium.solution)
    profile_scale = TARGET_BETA / float(calibration.wout.betatotal)
    n0 *= profile_scale ** (1 / 3); T0 *= profile_scale ** (2 / 3)
    profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                               T0 * np.array([1, -1]), T0 * np.array([1, -1]))
    inp = replace(inp, pres_scale=inp.pres_scale * profile_scale)
    picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=8, tol=1e-3,
        degree=N_CURRENT_SPLINE - 1, s_eval=QA_SURFACES, verbose=True)
    inp, equilibrium = picard.input, picard.equilibrium

bootstrap = RedlBootstrapMismatch(
    profiles, helicity_n=0, surfaces=QA_SURFACES, n_lambda=12 if ci_smoke else 32)
finite_beta_terms = [(qs, 0.0, QA_WEIGHT), (bootstrap, 0.0, BOOTSTRAP_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0), (opt.mean_iota, IOTA_TARGET, 10.0),
    (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, WELL_WEIGHT)]

# One physical lambda must represent the same particles on every radius and
# field-line label. Keep these pitches fixed throughout the maximum-J stages.
pitch = np.asarray(common_trapped_pitches_state(
    equilibrium.solution, equilibrium.solver_context, SURFACES, TRAPPING_DEPTHS,
    nalpha=ACTION_NALPHA, points_per_period=ACTION_POINTS,
    num_periods=ACTION_PERIODS))
maximum_j = MaximumJResidual(SURFACES, pitch, mboz=10, nboz=10,
    nalpha=ACTION_NALPHA, points_per_period=ACTION_POINTS,
    num_periods=ACTION_PERIODS, max_wells=ACTION_MAX_WELLS,
    quadrature_order=ACTION_QUADRATURE, target=MAXJ_TARGET)
for max_mode, max_nfev in zip(MAXJ_MAX_MODES, MAXJ_MAX_NFEV):
    print(f"\n===== QA + maximum-J stage, max_mode = {max_mode} =====")
    mpol, ntor = max(inp.mpol, max_mode + 2, MINIMUM_MPOL), max(inp.ntor, max_mode + 2)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6, nzeta=2 * ntor + 4)
    inp = opt.resample_current_profile(inp, N_CURRENT_SPLINE)
    terms = [(maximum_j, 0.0, MAXJ_WEIGHT), *finite_beta_terms]
    problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=max_mode,
        current_dofs=N_CURRENT_SPLINE - 1, use_ess=True, restart_from=equilibrium,
        forward_max_iterations=100 if ci_smoke else 3000,
        progress=True)
    print(f"dof_names = {problem.dof_names}")
    problem.compile_residual_and_jacobian()
    monitor.problem = problem
    step = (0.001 if ci_smoke else PARAMETER_STEP) * problem.scales
    step[-N_CURRENT_SPLINE:] = CURRENT_PARAMETER_STEP
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp, equilibrium = problem.input_from_x(result.x), problem.equilibrium_from_x(result.x)
    report(f"maximum-J mode {max_mode}", equilibrium)

final_input = replace(inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([20000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)
diagnostics = maximum_j.compute_state(final_equilibrium.solution, final_equilibrium.solver_context)
report("final", final_equilibrium)
print(f"maximum-J residual = {float(diagnostics['total']):.4e}, "
      f"outer-radius maximum-J fraction = {float(diagnostics['maximum_j_fraction']):.1%}")
print(f"beta = {float(final_equilibrium.wout.betatotal):.3%}; "
      "the reported fraction excludes the near-axis region where QA and maximum-J are incompatible.")

input_path = final_input.to_indata("input.QA_maxJ_optimized")
wout_path = vj.write_wout("wout_QA_maxJ_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QA_maxJ_objectives.csv"); monitor.plot("QA_maxJ_objectives.png")
for path in vj.plot_wout(wout_path, ".", j_pitch=float(pitch[0])).values():
    print(f"wrote {path}")
