#!/usr/bin/env python
"""Constructed-QI and maximum-J boundary optimization."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.maxj import JInvariantQIAndMaximumJResidual
from vmex.core.qi import ConstructedQIResidual

SURFACES = np.array([0.25, 0.6, 0.9])
MAX_MODES, MAX_NFEV = [1, 2, 3], [10, 20, 30]
ASPECT_TARGET, IOTA_TARGET, MIRROR_LIMIT = 10.0, -0.61, 0.25
TRAPPING_DEPTHS = (0.35, 0.55, 0.75)
MINIMUM_MPOL = 5
BOUNDARY_STEP = 0.002  # local trust region: bounce-well topology must remain fixed
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
qi_options = dict(nphi=61, nalpha=13, n_bounce=21)
action_options = dict(nalpha=7, points_per_period=48, num_periods=3,
                      max_wells=8, quadrature_order=32)

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]
    qi_options = dict(nphi=31, nalpha=7, n_bounce=7)
    action_options = dict(nalpha=5, points_per_period=24, num_periods=2,
                          max_wells=6, quadrature_order=16)

DATA = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"
inp = vj.VmecInput.from_file(DATA)
qi = ConstructedQIResidual(SURFACES, mboz=8 if ci_smoke else 14,
                           nboz=8 if ci_smoke else 14, **qi_options)
equilibrium = opt.solve_equilibrium(inp)
probe = qi.compute_state(equilibrium.state, equilibrium.runtime)
bmin, bmax = float(jnp.min(probe["bmag"])), float(jnp.max(probe["bmag"]))
pitch = np.array([1 / (bmax - depth * (bmax - bmin)) for depth in TRAPPING_DEPTHS])
qi_maxj = JInvariantQIAndMaximumJResidual(SURFACES, pitch,
    mboz=8 if ci_smoke else 14, nboz=8 if ci_smoke else 14,
    qi_options=action_options, maxj_options=action_options)

def mirror_excess(state, runtime):
    return jnp.maximum(opt.mirror_ratio(state, runtime) - MIRROR_LIMIT, 0.0)

objective_function_terms = [
    (qi, 0.0, 1.0), (qi_maxj, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0), (opt.mean_iota, IOTA_TARGET, 1.0),
    (mirror_excess, 0.0, 100.0),
]

def report(label, equilibrium):
    print(f"[{label}] QI = {float(qi.total(equilibrium)):.4e}, "
          f"aspect = {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.3f}, "
          f"iota = {float(opt.mean_iota(equilibrium.state, equilibrium.runtime)):.3f}, "
          f"mirror = {float(opt.mirror_ratio(equilibrium.state, equilibrium.runtime)):.3f}")

report("seed", equilibrium)
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QI + maximum-J stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    step = BOUNDARY_STEP * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - step, problem.x0 + step), max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-10, verbose=2)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)

final_input = replace(inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)
report("final", final_equilibrium)
diagnostics = qi_maxj.compute_state(final_equilibrium.state, final_equilibrium.runtime)
print(f"J-invariance = {float(diagnostics['qi']['total']):.4e}, "
      f"maximum-J = {float(diagnostics['maximum_j']['total']):.4e}")
input_path = final_input.to_indata("input.QI_maxJ_optimized")
wout_path = vj.write_wout("wout_QI_maxJ_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
