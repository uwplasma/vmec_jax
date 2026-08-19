#!/usr/bin/env python
"""Quasi-axisymmetric boundary optimization with a magnetic well."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 2  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [2, 4, 4, 4], [15, 35, 25, 60]
ASPECT_TARGET = 5.0
IOTA_FLOOR = 0.42
MAGNETIC_WELL_TARGET = 0.01
TRIAL_BETA = 0.001  # optimize a 0.1%-beta p(s) proportional to 1-s stability proxy
USE_TRIAL_STABILITY = True
STABILITY_COST_PER_SURFACE, EDGE_WEIGHT_FACTOR = 1.0e-2, 10.0
STABILITY_MIN_S, STABILITY_MARGIN = 0.2, 1.0e-3
CERTIFICATE_RESOLUTIONS, CERTIFICATE_MAX_NFEV = [31, 51, 101], [30, 35, 80]
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
ESS_ALPHA = 1.2  # smaller values let high Fourier modes move more
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# The exactly circular torus has zero first-order iota sensitivity. This
# explicit rotating-ellipse perturbation gives the local optimizer a QA basin.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

# Mercier and Glasser criteria for MHD stability (use a small beta)
def trial_dmerc(equilibrium_state, solver_context):
    return opt.trial_pressure_mercier_stability_residual(
        equilibrium_state, solver_context, beta=TRIAL_BETA,
        margin=STABILITY_MARGIN)

def trial_dr(equilibrium_state, solver_context):
    return opt.trial_pressure_glasser_stability_residual(
        equilibrium_state, solver_context, beta=TRIAL_BETA,
        margin=STABILITY_MARGIN)

stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
# The first QA stage excludes stability. Later stages omit the singular core,
# emphasize the difficult edge smoothly, and normalize each dimensional row
# to its value at the established QA seed; the final finite-beta solve remains
# the physical stability certificate.
stability_shape = np.where(stability_s >= STABILITY_MIN_S,
    1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4, 0.0)

# Objective function terms
# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
         (qs, 0.0, 1.0),
         (opt.aspect_ratio, ASPECT_TARGET, 1.0),
         (iota_floor, 0.0, 10.0),
         (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
         ]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)

# Optimize for QA first, then add the pressure-stability proxy locally.
equilibrium = opt.solve_equilibrium(inp)
for stage, (max_mode, max_nfev) in enumerate(zip(MAX_MODES, MAX_NFEV)):
    print(f"\n===== QA stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    stage_terms = objective_function_terms
    if USE_TRIAL_STABILITY and stage > 0:
        dmerc0 = np.asarray(trial_dmerc(equilibrium.solution, equilibrium.solver_context))
        dr0 = np.asarray(trial_dr(equilibrium.solution, equilibrium.solver_context))
        stability_scale = np.maximum.reduce((np.abs(dmerc0), np.abs(dr0), np.ones_like(dmerc0)))
        stage_cost = STABILITY_COST_PER_SURFACE * 10.0**(stage - 1)
        stability_weights = stage_cost * stability_shape / stability_scale**2
        stage_terms = [*objective_function_terms,
            (trial_dmerc, 0.0, stability_weights),
            (trial_dr, 0.0, stability_weights)
        ]
        print(f"Adding trial-pressure stability on s >= {STABILITY_MIN_S:.1f}; "
              f"weights rise smoothly toward the edge (cost scale {stage_cost:g}).")
    problem = opt.VmecProblem.from_tuples(inp, stage_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA,
        restart_from=equilibrium)
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
    # inp.to_indata(f"input.QA_max_mode_{max_mode:03d}")

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")
if USE_TRIAL_STABILITY:
    final_s = np.linspace(0.0, 1.0, int(final_input.ns_array[-1]))[2:-1]
    keep = final_s >= STABILITY_MIN_S
    final_dmerc = np.asarray(opt.trial_pressure_d_merc_state(
        final_equilibrium.solution, final_equilibrium.solver_context, beta=TRIAL_BETA))[2:-1]
    final_dr = np.asarray(opt.trial_pressure_glasser_d_r_state(
        final_equilibrium.solution, final_equilibrium.solver_context, beta=TRIAL_BETA,
        shear_epsilon=1.0e-8))[2:-1]
    print(f"Trial-pressure proxy on s >= {STABILITY_MIN_S:.1f}: "
          f"min DMerc = {final_dmerc[keep].min():.3e}, max DR = {final_dr[keep].max():.3e}")

# Preserve the optimized vacuum equilibrium separately. When trial-pressure
# stability is enabled, the primary QA output below is the resolved 0.1%-beta
# certificate, so its plotted DMerc and DR have their finite-pressure meaning.
vacuum_name = "QA_optimized_vacuum" if USE_TRIAL_STABILITY and not ci_smoke else "QA_optimized"
vacuum_input_path = final_input.to_indata(f"input.{vacuum_name}")
vacuum_wout_path = vj.write_wout(f"wout_{vacuum_name}.nc", final_equilibrium.wout)
print(f"wrote {vacuum_input_path}\nwrote {vacuum_wout_path}")

# A vacuum DMerc is only the formal zero-pressure limit. Add 0.1% pressure,
# then polish the actual finite-beta DMerc/DR from coarse to resolved radial grids.
certificate_wout_path = None
if not ci_smoke:
    am = np.zeros(21); am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1-s)
    calibration_input = replace(inp, pmass_type="power_series", am=am, pres_scale=10.0,
        ns_array=np.array([31]), ftol_array=np.array([1.0e-12]), niter_array=np.array([8000]))
    certificate = opt.solve_equilibrium(calibration_input, initial_state=equilibrium.solution)
    pressure_scale = calibration_input.pres_scale * TRIAL_BETA / float(certificate.wout.betatotal)
    for fraction in np.linspace(0.25, 1.0, 4):
        continuation_input = replace(calibration_input, pres_scale=fraction * pressure_scale)
        certificate = opt.solve_equilibrium(continuation_input, initial_state=certificate.solution)
    for ns, max_nfev in zip(CERTIFICATE_RESOLUTIONS, CERTIFICATE_MAX_NFEV):
        certificate_input = replace(inp, pmass_type="power_series", am=am,
            pres_scale=pressure_scale, ns_array=np.array([ns]), ftol_array=np.array([1e-12]),
            niter_array=np.array([16000]))
        certificate = opt.solve_equilibrium(certificate_input, initial_state=certificate.solution,
            raise_on_max_iterations=True)
        s = np.linspace(0.0, 1.0, ns)[2:-1]
        dmerc = np.asarray(opt.d_merc_state(certificate.solution, certificate.solver_context))[2:-1]
        dr = np.asarray(opt.glasser_d_r_state(
            certificate.solution, certificate.solver_context, shear_epsilon=1e-8))[2:-1]
        shape = np.where(s >= STABILITY_MIN_S,
            1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * s**4, 0.0)
        scale = np.maximum.reduce((np.abs(dmerc), np.abs(dr), np.full_like(dmerc, 5e-4)))

        def finite_beta_dmerc(equilibrium_state, solver_context):
            return opt.mercier_stability_residual(
                equilibrium_state, solver_context, margin=5e-4, smoothing=1e-5)

        def finite_beta_dr(equilibrium_state, solver_context):
            return opt.glasser_stability_residual(
                equilibrium_state, solver_context, margin=5e-4, smoothing=1e-5)

        print(f"Polishing the physical 0.1%-beta stability certificate at NS={ns}; "
              f"the core s < {STABILITY_MIN_S:.1f} is excluded and edge weights increase smoothly.")
        certificate_terms = [*objective_function_terms,
            (finite_beta_dmerc, 0.0, 5.0 * shape / scale**2),
            (finite_beta_dr, 0.0, 5.0 * shape / scale**2)]
        certificate_problem = opt.VmecProblem.from_tuples(certificate_input, certificate_terms,
            max_mode=MAX_MODES[-1], use_ess=True, ess_alpha=ESS_ALPHA,
            restart_from=certificate, progress=True, evaluation_progress=True)
        step = 0.01 * certificate_problem.scales
        certificate_result = least_squares(certificate_problem.residual, certificate_problem.x0,
            jac=certificate_problem.residual_jac, x_scale=step,
            bounds=(certificate_problem.x0 - 8.0 * step,
                    certificate_problem.x0 + 8.0 * step),
            max_nfev=max_nfev, ftol=1e-7, xtol=1e-10, verbose=2)
        certificate_input = certificate_problem.input_from_x(certificate_result.x)
        certificate = certificate_problem.equilibrium_from_x(certificate_result.x)
    certificate_input = replace(certificate_input, ftol_array=np.array([1e-14]))
    certificate = opt.solve_equilibrium(certificate_input, initial_state=certificate.solution,
        verbose=True, raise_on_max_iterations=True)
    certificate_input_path = certificate_input.to_indata("input.QA_optimized")
    certificate_wout_path = vj.write_wout("wout_QA_optimized.nc", certificate.wout)
    certificate_s = np.linspace(0.0, 1.0, int(certificate.wout.ns))[2:-1]
    keep = certificate_s >= STABILITY_MIN_S
    certificate_dmerc = np.asarray(certificate.wout.DMerc)[2:-1]
    certificate_dr = np.asarray(opt.glasser_d_r_state(
        certificate.solution, certificate.solver_context, shear_epsilon=1e-8))[2:-1]
    print(f"0.1%-beta certificate: beta={certificate.wout.betatotal:.4e}, "
          f"min DMerc={certificate_dmerc[keep].min():.3e}, "
          f"max DR={certificate_dr[keep].max():.3e} on s >= {STABILITY_MIN_S:.1f}")
    print(f"wrote {certificate_input_path}\nwrote {certificate_wout_path}")

# Plot results
monitor.save("QA_optimization_objectives.csv")
monitor.plot("QA_optimization_objectives.png")
for path in vj.plot_wout(vacuum_wout_path, ".").values():
    print(f"wrote {path}")
if certificate_wout_path is not None:
    for path in vj.plot_wout(certificate_wout_path, ".").values():
        print(f"wrote {path}")
