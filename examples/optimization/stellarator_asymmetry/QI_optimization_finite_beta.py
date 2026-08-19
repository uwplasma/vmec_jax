#!/usr/bin/env python
"""LASYM finite-beta constructed-QI boundary optimization."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.qi import ConstructedQIResidual

nfp, TARGET_BETA = 2, 0.01
SURFACES = np.linspace(0.1, 1.0, 6)
MAX_MODES, MAX_NFEV = [3, 5], [80, 100]
ASPECT_TARGET, IOTA_FLOOR, MIRROR_LIMIT, ELONGATION_LIMIT = 5.0, 0.51, 0.21, 8.0
MINIMUM_MPOL, SEED_PERTURBATION, ASYMMETRY_PERTURBATION = 5, 0.05, 0.01
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.01, 3.0
ESS_ALPHA = 1.2  # lower only after a low-mode QI basin has converged
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES, MINIMUM_MPOL = np.array([0.25, 0.6, 0.9]), 3
    MAX_MODES, MAX_NFEV = [1], [2]
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)

DATA = Path(__file__).resolve().parents[2] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
if ci_smoke:
    inp = replace(inp, ns_array=np.array([11]), ftol_array=np.array([1e-8]),
                  niter_array=np.array([1500]))
rbc, zbs, rbs, zbc = inp.rbc.copy(), inp.zbs.copy(), inp.rbs.copy(), inp.zbc.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
# Nonzero RBS(1,1)/ZBC(1,1) prevents the local solve from remaining in the
# original stellarator-symmetric subspace.
rbs[inp.ntor + 1, 1], zbc[inp.ntor + 1, 1] = ASYMMETRY_PERTURBATION, -ASYMMETRY_PERTURBATION
am = np.zeros(21); am[:2] = [1.0, -1.0]  # p(s)=PRES_SCALE*(1-s)
inp = replace(inp, lasym=True, rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc,
    pmass_type="power_series", am=am, pres_scale=100.0)
calibration = opt.solve_equilibrium(inp)
inp = replace(inp, pres_scale=inp.pres_scale * TARGET_BETA / float(calibration.wout.betatotal))
equilibrium = opt.solve_equilibrium(inp, initial_state=calibration.solution)

qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

def mirror_excess(equilibrium_state, solver_context):
    return jnp.maximum(opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)

def elongation_excess(equilibrium_state, solver_context):
    return jnp.maximum(opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)

objective_function_terms = [(qi, 0.0, 10.0), (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0), (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2)]
report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".4e"), ("beta", opt.volume_average_beta, ".3%"),
    ("aspect", opt.aspect_ratio, ".3f"), ("iota", opt.mean_iota, ".3f"))
monitor = opt.OptimizationMonitor(stream=None)

# If a RuntimeWarning reports uncertified Jacobian columns, it is expected
# once the optimizer leaves the seed and needs no action: the shipped
# jacobian_adjoint_tol=1e-4 and jacobian_adjoint_maxiter=10 are the measured
# optimum, since ten times that budget moved the Jacobian by 2e-8 and
# certified no extra column. Both are from_tuples arguments; pass
# evaluation_progress=False to drop the per-evaluation timing lines.
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== LASYM finite-beta QI stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        use_ess=True, ess_alpha=ESS_ALPHA, restart_from=equilibrium, progress=True, evaluation_progress=True,
        forward_max_iterations=100 if ci_smoke else 3000)
    print(f"dof_names = {problem.dof_names}")
    problem.compile_residual_and_jacobian()
    monitor.problem = problem
    step = (0.001 if ci_smoke else PARAMETER_STEP) * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp, equilibrium = problem.input_from_x(result.x), problem.equilibrium_from_x(result.x)
    inp.to_indata(f"input.QI_finite_beta_max_mode_{max_mode:03d}")
    report(f"mode {max_mode}", equilibrium)

final_input = replace(inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([20000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)
print(f"asymmetric boundary norm = "
      f"{np.linalg.norm(final_input.rbs) + np.linalg.norm(final_input.zbc):.6e}")
report("final", final_equilibrium)

input_path = final_input.to_indata("input.QI_LASYM_finite_beta_optimized")
wout_path = vj.write_wout("wout_QI_LASYM_finite_beta_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QI_LASYM_finite_beta_objectives.csv")
monitor.plot("QI_LASYM_finite_beta_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
