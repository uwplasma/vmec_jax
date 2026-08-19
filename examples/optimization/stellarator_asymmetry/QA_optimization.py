#!/usr/bin/env python
"""LASYM quasi-axisymmetric boundary optimization in vacuum."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 2
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [2, 4], [20, 45]
ASPECT_TARGET, IOTA_FLOOR, MAGNETIC_WELL_TARGET = 5.0, 0.42, 0.01
MINIMUM_MPOL, SEED_PERTURBATION, ASYMMETRY_PERTURBATION = 5, 0.05, 0.01
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.01, 3.0
ESS_ALPHA = 1.2  # smaller values let high Fourier modes move more

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES, MINIMUM_MPOL = np.array([0.25, 0.6, 0.9]), 3
    MAX_MODES, MAX_NFEV = [1], [2]

DATA = Path(__file__).resolve().parents[2] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
if ci_smoke:
    inp = replace(inp, ns_array=np.array([11]), ftol_array=np.array([1e-8]),
                  niter_array=np.array([1500]))
rbc, zbs, rbs, zbc = inp.rbc.copy(), inp.zbs.copy(), inp.rbs.copy(), inp.zbc.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
# LASYM adds independent sine-R and cosine-Z families; this (m,n)=(1,1)
# perturbation keeps the optimizer away from the symmetric stationary subspace.
rbs[inp.ntor + 1, 1], zbc[inp.ntor + 1, 1] = ASYMMETRY_PERTURBATION, -ASYMMETRY_PERTURBATION
inp = replace(inp, lasym=True, rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc)

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0), (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0)]
report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)

equilibrium = opt.solve_equilibrium(inp)
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== LASYM QA stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        use_ess=True, ess_alpha=ESS_ALPHA, restart_from=equilibrium,
        forward_max_iterations=100 if ci_smoke else 2000, progress=True,
        evaluation_progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    problem.compile_residual_and_jacobian()
    monitor.problem = problem
    step = (0.001 if ci_smoke else PARAMETER_STEP) * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10,
        verbose=2, callback=monitor)
    inp, equilibrium = problem.input_from_x(result.x), problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)

final_input = replace(inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([12000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)
print(f"asymmetric boundary norm = "
      f"{np.linalg.norm(final_input.rbs) + np.linalg.norm(final_input.zbc):.6e}")
report("final", final_equilibrium)

input_path = final_input.to_indata("input.QA_LASYM_optimized")
wout_path = vj.write_wout("wout_QA_LASYM_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QA_LASYM_optimization_objectives.csv")
monitor.plot("QA_LASYM_optimization_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
