#!/usr/bin/env python
"""Quasi-poloidal boundary optimization with an explicit mode ladder."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 2  # number of field periods
SURFACES = np.array([0.5,0.7,0.9])
MAX_MODES, MAX_NFEV = [3], [60]  # mode-ladder alternative: [1,2,3], [20,20,20]
ASPECT_TARGET = 7.0
IOTA_FLOOR = 0.51
MIRROR_LIMIT = 0.35
ELONGATION_LIMIT = 12.0
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
inp.rbc[inp.ntor-1, 1] =-SEED_PERTURBATION
inp.zbs[inp.ntor-1, 1] = SEED_PERTURBATION
inp = replace(inp, delt=0.5,
              niter_array=np.array([300, 8000]),
              ftol_array=np.array([1.0e-11, 1e-12]),
              ns_array=np.array([25, 35]))
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)

def mirror_excess(state, runtime):
    return jnp.maximum(opt.mirror_ratio(state, runtime) - MIRROR_LIMIT, 0.0)

def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

def elongation_excess(state, runtime):
    return jnp.maximum(opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0.0)

def report(label, equilibrium):
    total = float(qs.total(equilibrium))
    print(f"[{label}] QS total = {total:.6e}, "
          f"aspect = {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"elongation = {float(opt.max_elongation(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"mirror = {float(opt.mirror_ratio(equilibrium.state, equilibrium.runtime)):.4f}")
    return total

objective_function_terms = [
         (qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
         (iota_floor, 0.0, 100.0),
         (mirror_excess, 0.0, 10.0),
         (elongation_excess, 0.0, 10.0)
         ]

for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QP stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
                                          vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True)
    print(f"dof_names = {problem.dof_names}")
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    inp.to_indata(f"input.QP_max_mode_{max_mode:03d}")

# Print results
final_total = report("final", equilibrium)
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([35000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)
print(f"\nQS total {final_total:.3e}")

# Save results
input_path = final_input.to_indata("input.QP_optimized")
wout_path = vj.write_wout("wout_QP_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}")
print(f"wrote {wout_path}")

# Plot results
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
