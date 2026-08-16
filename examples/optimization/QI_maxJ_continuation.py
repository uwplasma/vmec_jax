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

SURFACES = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 0.90])
QI_SEED_MAX_MODES, QI_SEED_MAX_NFEV = [1, 2, 3], [20, 30, 40]
MAX_MODES, MAX_NFEV = [2, 2, 2, 3], [20, 40, 35, 50]
ASPECT_TARGET, IOTA_TARGET, MIRROR_LIMIT = 6.0, -1.3, 0.30
MAGNETIC_WELL_TARGET = 0.01
MAXIMUM_J_TARGETS = [-0.05, -0.01, -0.01, -0.01]
MAXIMUM_J_WEIGHTS = [50.0, 1.0e4, 1.0e4, 1.0e4]
QI_INVARIANCE_WEIGHTS = [100.0, 1.0e4, 1.0e4, 1.0e4]
CONSTRUCTED_QI_WEIGHTS = [100.0, 1.0e3, 1.0e3, 1.0e3]
MAGNETIC_WELL_WEIGHTS = [10.0, 100.0, 100.0, 100.0]
TRAPPING_DEPTHS = (0.35, 0.55, 0.75)
MINIMUM_MPOL = 5
BOUNDARY_STEP = 0.05  # local trust region: large enough to move, small enough to preserve wells
QI_SEED_BOUNDARY_STEP = 0.10
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.10
qi_options = dict(nphi=61, nalpha=18, n_bounce=21)
coarse_action = dict(nalpha=7, points_per_period=48, num_periods=3,
                     max_wells=8, quadrature_order=32)
# The last stages cover a full poloidal transit and more alpha values. This
# removes the visually apparent alias that a short, coarse action trace misses.
resolved_action = dict(nalpha=9, points_per_period=32, num_periods=10,
                       max_wells=24, quadrature_order=24)
ACTION_MBOZ = [8, 8, 10, 10]
ACTION_OPTIONS = [coarse_action, coarse_action, resolved_action, resolved_action]

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QI_SEED_MAX_MODES, QI_SEED_MAX_NFEV = [1, 2], [12, 20]
    MAX_MODES, MAX_NFEV = [2, 2, 2], [12, 30, 35]
    MAXIMUM_J_TARGETS, MAXIMUM_J_WEIGHTS = [-0.05, -0.01, -0.01], [50.0, 1.0e4, 1.0e4]
    QI_INVARIANCE_WEIGHTS = [100.0, 1.0e4, 1.0e4]
    CONSTRUCTED_QI_WEIGHTS = [100.0, 1.0e3, 1.0e3]
    MAGNETIC_WELL_WEIGHTS = [10.0, 100.0, 100.0]
    SURFACES, TRAPPING_DEPTHS = np.array([0.25, 0.45, 0.65, 0.85]), (0.5,)
    qi_options = dict(nphi=25, nalpha=5, n_bounce=5)
    coarse_action = dict(nalpha=5, points_per_period=24, num_periods=3,
                         max_wells=8, quadrature_order=12)
    ACTION_MBOZ = [8, 8, 10]
    ACTION_OPTIONS = [coarse_action, coarse_action, resolved_action]

# Start from the same transparent vacuum seed used by the other optimization
# examples. A rotating ellipse gives iota; the QI-only first stage then creates
# the common trapped-well topology required by the physical-pitch J objective.
DATA = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp4"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)
qi = ConstructedQIResidual(SURFACES, mboz=8 if ci_smoke else 14,
                           nboz=8 if ci_smoke else 14, **qi_options)

def mirror_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)

shape_terms = [
    (qi, 0.0, CONSTRUCTED_QI_WEIGHTS[0]),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 1.0),
    (mirror_excess, 0.0, 100.0),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, MAGNETIC_WELL_WEIGHTS[0]),
]

report = opt.EquilibriumReporter(
    ("QI", qi.total, ".4e"), ("aspect", opt.aspect_ratio, ".3f"),
    ("iota", opt.mean_iota, ".3f"), ("mirror", opt.mirror_ratio, ".3f"),
    ("magnetic well", opt.magnetic_well, ".3f"))
monitor = opt.OptimizationMonitor(stream=None)

# First form a vacuum QI basin from the minimal seed. A common physical pitch
# generally does not exist on the circular seed, so evaluating dJ/ds earlier
# would compare different trapped-particle populations on adjacent surfaces.
equilibrium = opt.solve_equilibrium(inp)
report("seed", equilibrium)
for max_mode, max_nfev in zip(QI_SEED_MAX_MODES, QI_SEED_MAX_NFEV):
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    print(f"\n===== QI seed stage, max_mode = {max_mode} =====")
    problem = opt.VmecProblem.from_tuples(inp, shape_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, restart_from=equilibrium,
        progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    step = QI_SEED_BOUNDARY_STEP * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, bounds=(problem.x0 - step, problem.x0 + step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x); equilibrium = problem.equilibrium_from_x(result.x)
    report(f"QI seed mode {max_mode}", equilibrium)

# ``bmag`` is indexed by (surface, Boozer toroidal angle, field-line label).
# Select pitches trapped on every sampled line. The strong stages freeze the
# post-homotopy selection so they optimize the same physical particles.
def common_pitch(equilibrium):
    probe = qi.compute_state(equilibrium.state, equilibrium.runtime)
    bmin = float(jnp.max(jnp.min(probe["bmag"], axis=1)))
    bmax = float(jnp.min(jnp.max(probe["bmag"], axis=1)))
    if bmin >= bmax:
        raise RuntimeError("the QI seed has no common trapped-particle pitch; increase the QI seed stages")
    return np.array([1 / (bmax - depth * (bmax - bmin)) for depth in TRAPPING_DEPTHS])

pitch = common_pitch(equilibrium)
for stage, (max_mode, max_nfev, maxj_target, maxj_weight, qi_weight,
            constructed_weight, well_weight, action_mboz, action_options) in enumerate(zip(
        MAX_MODES, MAX_NFEV, MAXIMUM_J_TARGETS, MAXIMUM_J_WEIGHTS,
        QI_INVARIANCE_WEIGHTS, CONSTRUCTED_QI_WEIGHTS, MAGNETIC_WELL_WEIGHTS,
        ACTION_MBOZ, ACTION_OPTIONS)):
    print(f"\n===== QI + maximum-J stage, max_mode = {max_mode}, "
          f"target = {maxj_target:g}, weight = {maxj_weight:g} =====")
    # The weak first stage creates the maximum-J basin. Select the physical
    # pitches once more after it, then keep those same particles for every
    # strong and sampling-fidelity stage and for the final polar plot.
    if stage == 1:
        pitch = common_pitch(equilibrium)
    qi_maxj = JInvariantQIAndMaximumJResidual(SURFACES, pitch,
        mboz=action_mboz, nboz=action_mboz,
        qi_options=action_options, qi_weight=qi_weight,
        maxj_weight=maxj_weight,
        maxj_options={**action_options, "target": maxj_target})
    if not bool(jnp.all(qi_maxj.compute_state(
            equilibrium.state, equilibrium.runtime)["maximum_j"]["valid_pitch_pair"])):
        raise RuntimeError(
            "the QI seed does not retain matched wells between adjacent surfaces; "
            "increase the preceding continuation stage")
    stage_shape_terms = [
        (qi, 0.0, constructed_weight),
        (opt.aspect_ratio, ASPECT_TARGET, 1.0),
        (opt.mean_iota, IOTA_TARGET, 1.0),
        (mirror_excess, 0.0, 100.0),
        (opt.magnetic_well, MAGNETIC_WELL_TARGET, well_weight),
    ]
    objective_function_terms = [(qi_maxj, 0.0, 1.0), *stage_shape_terms]
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, restart_from=equilibrium,
        progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    step = BOUNDARY_STEP * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, bounds=(problem.x0 - step, problem.x0 + step), max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    print(f"normalized boundary displacement = "
          f"{np.linalg.norm((result.x - problem.x0) / problem.scales):.3e}")
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
      f"maximum-J = {float(diagnostics['maximum_j']['total']):.4e}, "
      f"maximum-J fraction = {float(diagnostics['maximum_j']['maximum_j_fraction']):.1%}, "
      f"target-margin fraction = {float(diagnostics['maximum_j']['target_fraction']):.1%}")
input_path = final_input.to_indata("input.QI_maxJ_optimized")
wout_path = vj.write_wout("wout_QI_maxJ_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QI_maxJ_objectives.csv")
monitor.plot("QI_maxJ_objectives.png")
for path in vj.plot_wout(wout_path, ".", j_pitch=float(pitch[0])).values():
    print(f"wrote {path}")
