#!/usr/bin/env python
"""Quasi-helically symmetric boundary optimization with a magnetic well."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 4  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES = [2,3]
MAX_NFEV = [15, 15]
ASPECT_TARGET = 6.0
# IOTA_TARGET = -1.1
# MAGNETIC_WELL_TARGET = 0.01
TRIAL_BETA = 0.025  # pressure proxy: beta=2.5%, p(s) proportional to 1-s
USE_TRIAL_STABILITY = False
STABILITY_COST_PER_SURFACE, EDGE_WEIGHT_FACTOR = 1.0e-2, 10.0
STABILITY_MIN_S, STABILITY_MARGIN = 0.2, 1.0e-3
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.12

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# A circular torus cannot acquire iota to first order. Seed a rotating ellipse
# explicitly so the local optimization starts in the QH basin.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

def trial_dmerc(equilibrium_state, solver_context):
    return opt.trial_pressure_mercier_stability_residual(
        equilibrium_state, solver_context, beta=TRIAL_BETA,
        margin=STABILITY_MARGIN)

def trial_dr(equilibrium_state, solver_context):
    return opt.trial_pressure_glasser_stability_residual(
        equilibrium_state, solver_context, beta=TRIAL_BETA,
        margin=STABILITY_MARGIN)

stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
# Trial stability is added only after a QH basin exists. It omits the singular
# core, rises smoothly toward the difficult edge, and is normalized at that
# QH seed so dimensional Mercier rows cannot overwhelm quasisymmetry.
stability_shape = np.where(stability_s >= STABILITY_MIN_S,
    1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4, 0.0)

# Objective function terms
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
objective_function_terms = [
         (qs, 0.0, 1.0),
         (opt.aspect_ratio, ASPECT_TARGET, 1.0),
        #  (opt.mean_iota, IOTA_TARGET, 10.0),
        #  (opt.magnetic_well, MAGNETIC_WELL_TARGET, 10.0),
         ]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)

# Optimize for QH in stages, adding the optional stability proxy after stage 1.
equilibrium = opt.solve_equilibrium(inp)
for stage, (max_mode, max_nfev) in enumerate(zip(MAX_MODES, MAX_NFEV)):
    print(f"\n===== QH stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    stage_terms = objective_function_terms
    if USE_TRIAL_STABILITY and stage > 0:
        dmerc0 = np.asarray(trial_dmerc(equilibrium.state, equilibrium.runtime))
        dr0 = np.asarray(trial_dr(equilibrium.state, equilibrium.runtime))
        stability_scale = np.maximum.reduce((np.abs(dmerc0), np.abs(dr0), np.ones_like(dmerc0)))
        stability_weights = STABILITY_COST_PER_SURFACE * stability_shape / stability_scale**2
        stage_terms = [*objective_function_terms,
            (trial_dmerc, 0.0, stability_weights), (trial_dr, 0.0, stability_weights)]
        print(f"Adding trial-pressure stability on s >= {STABILITY_MIN_S:.1f}; "
              "weights rise smoothly toward the edge.")
    problem = opt.VmecProblem.from_tuples(inp, stage_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, restart_from=equilibrium)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    step = PARAMETER_STEP * problem.scales
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step), max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    # inp.to_indata(f"input.QH_max_mode_{max_mode:03d}")

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")
if USE_TRIAL_STABILITY:
    final_s = np.linspace(0.0, 1.0, int(final_input.ns_array[-1]))[2:-1]
    keep = final_s >= STABILITY_MIN_S
    final_dmerc = np.asarray(opt.trial_pressure_d_merc_state(
        final_equilibrium.state, final_equilibrium.runtime, beta=TRIAL_BETA))[2:-1]
    final_dr = np.asarray(opt.trial_pressure_glasser_d_r_state(
        final_equilibrium.state, final_equilibrium.runtime, beta=TRIAL_BETA,
        shear_epsilon=1.0e-8))[2:-1]
    print(f"Trial-pressure proxy on s >= {STABILITY_MIN_S:.1f}: "
          f"min DMerc = {final_dmerc[keep].min():.3e}, max DR = {final_dr[keep].max():.3e}")

# Save results
input_path = final_input.to_indata("input.QH_optimized")
wout_path = vj.write_wout("wout_QH_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")

# Plot results
monitor.save("QH_optimization_objectives.csv")
monitor.plot("QH_optimization_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
