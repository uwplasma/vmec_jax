#!/usr/bin/env python
"""Finite-beta QI optimization with a self-consistent bootstrap current."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch,
                                 self_consistent_bootstrap)
from vmex.core.qi import ConstructedQIResidual

nfp = 2
TARGET_BETA = 0.025
BETA_WEIGHT = 1.0 / TARGET_BETA**2  # beta residual is relative
SURFACES = np.linspace(0.1, 0.9, 8)
MAX_MODES, MAX_NFEV = [2, 3], [15, 30]
N_CURRENT_SPLINE = [6, 8]  # optimized I'(s) spline knots at each stage
ASPECT_TARGET, IOTA_FLOOR = 6.0, 0.51
MIRROR_LIMIT, ELONGATION_LIMIT = 0.21, 8.0
# VMEC's dimensional DMerc/DR values are O(1e2-1e3) for this seed.
STABILITY_WEIGHT, EDGE_WEIGHT_FACTOR, STABILITY_MIN_S = 1.0e-6, 10.0, 0.2
# Characteristic low-order boundary step in meters; ESS reduces higher modes.
# Current dofs are dimensionless here, so they have their own optimizer scale.
PARAMETER_STEP, CURRENT_PARAMETER_STEP = 0.02, 0.05
MAX_PARAMETER_CHANGE = 10.0  # per-stage box guardrail, in scaled step units
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05
qi_options = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES, MAX_MODES, MAX_NFEV, N_CURRENT_SPLINE = np.linspace(0.2, 0.8, 4), [1], [4], [4]
    qi_options = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION

# The Landreman-Buller-Drevlak profiles: ne=n0(1-s^5), Te=Ti=T0(1-s).
# Their product gives p=2 e ne Te; one seed solve calibrates its amplitude to
# the requested VMEC volume-average beta for this magnetic-field scale.
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
am = np.zeros(21); am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
ac = np.zeros(21); ac[0] = 1.0
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
              pcurr_type="power_series", ac=ac, curtor=0.0)
seed = opt.solve_equilibrium(inp)
profile_scale = TARGET_BETA / float(seed.wout.betatotal)
n0 *= profile_scale ** (1 / 3); T0 *= profile_scale ** (2 / 3)
inp = replace(inp, pres_scale=inp.pres_scale * profile_scale)

# These polynomials provide ne(s), Te(s), and Ti(s) to the Redl model.
# The Picard loop alternates hot-restarted VMEC solves with current-profile fits.
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
# helicity_n=0 for QI: a quasi-isodynamic field carries no helical symmetry, so
# the isomorphism shift iota -> iota - nfp*helicity_n is the identity. The Redl
# formula was fitted on quasisymmetric fields, so on a QI boundary it is a
# reasonable analytic estimate rather than a converged kinetic answer;
# QI_optimization_bootstrap_dkx.py in the DKX repository is the drift-kinetic
# variant of this same script.
picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=2 if ci_smoke else 8,
                                   tol=1e-3, degree=N_CURRENT_SPLINE[0] - 1,
                                   s_eval=SURFACES, verbose=not ci_smoke)
# The Picard bootstrap solve leaves the prescribed pressure and boundary shape
# unchanged, and mainly updates the current profile / equilibrium state (I'(s), CURTOR)
# to the self-consistent bootstrap response before the optimization starts.
inp, equilibrium = picard.input, picard.equilibrium
stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
# Mercier coordinates are unreliable near the axis. Zero weight below s=0.2
# and increase it smoothly toward the edge, where stability is hardest.
stability_weights = np.where(stability_s >= STABILITY_MIN_S,
    STABILITY_WEIGHT * (1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4), 0.0)

# Objective function terms
qi = ConstructedQIResidual(SURFACES, **qi_options)
bootstrap = RedlBootstrapMismatch(profiles, helicity_n=0, surfaces=SURFACES,
                                  n_lambda=12 if ci_smoke else 32)

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

def mirror_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)

def elongation_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)

objective_function_terms = [
    (qi, 0.0, 10.0), (bootstrap, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0),
    (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0),
    (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT),
    (opt.mercier_stability_residual, 0.0, stability_weights),
    (opt.glasser_stability_residual, 0.0, stability_weights),
]
def minimum_dmerc(equilibrium_state, solver_context):
    return opt.d_merc_state(equilibrium_state, solver_context)[2:-1].min()

def maximum_dr(equilibrium_state, solver_context):
    return opt.glasser_d_r_state(equilibrium_state, solver_context)[2:-1].max()

report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".4e"), ("f_boot", bootstrap.total, ".4e"),
    ("beta", opt.volume_average_beta, ".3%"), ("aspect", opt.aspect_ratio, ".3f"),
    ("min |iota|", opt.min_abs_iota, ".3f"), ("mirror", opt.mirror_ratio, ".3f"),
    ("min DMerc", minimum_dmerc, ".2e"), ("max DR", maximum_dr, ".2e"))
monitor = opt.OptimizationMonitor()

report("self-consistent seed", equilibrium)
# If a RuntimeWarning reports uncertified Jacobian columns, it is expected
# once the optimizer leaves the seed and needs no action: the shipped
# jacobian_adjoint_tol=1e-4 and jacobian_adjoint_maxiter=10 are the measured
# optimum, since ten times that budget moved the Jacobian by 2e-8 and
# certified no extra column. Both are from_tuples arguments; pass
# evaluation_progress=False to drop the per-evaluation timing lines.
for max_mode, max_nfev, n_spline in zip(MAX_MODES, MAX_NFEV, N_CURRENT_SPLINE):
    print(f"\n===== QI bootstrap stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    inp = opt.resample_current_profile(inp, n_spline)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        current_dofs=n_spline - 1, vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
        restart_from=equilibrium, progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    step[-n_spline:] = CURRENT_PARAMETER_STEP  # n-1 spline shapes + CURTOR
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)

final_input = replace(inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

# Print results
report("final", final_equilibrium)

# Save results
input_path = final_input.to_indata("input.QI_bootstrap_optimized")
wout_path = vj.write_wout("wout_QI_bootstrap_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")

# Plot results
monitor.save("QI_bootstrap_objectives.csv")
monitor.plot("QI_bootstrap_objectives.png")
vj.plot_bootstrap_current("QI_bootstrap_current.png", final_equilibrium, bootstrap)
print("wrote QI_bootstrap_current.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
