#!/usr/bin/env python
"""Constructed-QI boundary optimization with an explicit mode ladder."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual

nfp = 2  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 6)
MAX_MODES, MAX_NFEV = [3], [250]  # mode-ladder alternative: [1, 2], [20, 20]
ASPECT_TARGET = 5.0
IOTA_FLOOR = 0.51
MIRROR_LIMIT = 0.21
ELONGATION_LIMIT = 8.0
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05
qi_options = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
validation_options = dict(mboz=14, nboz=14, nphi=101, nalpha=29, n_bounce=31)

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    qi_options = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    validation_options = qi_options
    MAX_MODES, MAX_NFEV = [2], [5]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

# Objective function terms
qi = ConstructedQIResidual(SURFACES, **qi_options)

def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

def mirror_excess(state, runtime):
    return jnp.maximum(opt.mirror_ratio(state, runtime) - MIRROR_LIMIT, 0.0)

def elongation_excess(state, runtime):
    return jnp.maximum(opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0.0)

objective_function_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0),
    (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0),
]
qi_terms = [(qi, 0.0, 10.0), *objective_function_terms]

report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("mirror", opt.mirror_ratio, ".4f"),
    ("elongation", opt.max_elongation, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)

# Optimize for QI in stages, increasing the maximum mode number each time.
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol,
        ntheta=2 * mpol + 6,
        nzeta=2 * mpol + 4,
    )
    # Restart SciPy's trust-region model; equal-shape JAX executables are reused.
    problem = opt.VmecProblem.from_tuples(
        inp, qi_terms, max_mode=max_mode, use_ess=True, progress=not ci_smoke,
        vary_major_radius=VARY_MAJOR_RADIUS,
    )
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=max_nfev,
        ftol=1.0e-6, xtol=1.0e-10, verbose=2, callback=monitor
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"QI mode {max_mode}", equilibrium)

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)
qi_final = report("final", final_equilibrium)["constructed QI"]
qi_validation = ConstructedQIResidual(SURFACES, **validation_options)
print(f"\nQI total {qi_final:.3e}; independent fine-grid validation "
      f"{float(qi_validation.total(final_equilibrium)):.3e}")

# Save results
input_path = final_input.to_indata("input.QI_optimized")
wout_path = vj.write_wout("wout_QI_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}")
print(f"wrote {wout_path}")

# Plot results
monitor.save("QI_optimization_objectives.csv")
monitor.plot("QI_optimization_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
