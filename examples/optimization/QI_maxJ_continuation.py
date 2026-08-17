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
from vmex.core.maxj import (
    JInvariantQIAndMaximumJResidual, common_trapped_pitches_state,
)
from vmex.core.qi import ConstructedQIResidual

nfp = 3 # number of field periods
SURFACES = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 0.90])
QI_SEED_MAX_MODES, QI_SEED_MAX_NFEV = [1, 2, 3, 4], [20, 30, 50, 60]
QI_SEED_ESS_ALPHA = [1.2, 1.2, 0.9, 0.7]
MAX_MODES, MAX_NFEV = [2, 3, 4], [8, 12, 30]
ASPECT_TARGET, IOTA_FLOOR, MIRROR_LIMIT = 4.0, 1.03, 0.35
MAGNETIC_WELL_TARGET = 0.01
QI_SEED_WEIGHT = 1.0e3
MAXIMUM_J_TARGETS = [0.0, -0.002, -0.005]
MAXIMUM_J_WEIGHTS = [500.0, 2.0e3, 5.0e3]
QI_INVARIANCE_WEIGHTS = [1.0e3, 5.0e3, 1.0e4]
CONSTRUCTED_QI_WEIGHTS = [1.0e4, 1.0e4, 1.0e4]
MAGNETIC_WELL_WEIGHTS = [100.0, 1.0e3, 1.0e3]
TRAPPING_DEPTHS = (0.35, 0.55, 0.75)
MINIMUM_MPOL = 5
BOUNDARY_STEP = 0.05  # local trust region: large enough to move, small enough to preserve wells
QI_SEED_BOUNDARY_STEP = 0.10
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.08
qi_options = dict(nphi=61, nalpha=18, n_bounce=21)
coarse_action = dict(nalpha=5, points_per_period=24, num_periods=6,
                     max_wells=16, quadrature_order=16)
# The last stages cover a full poloidal transit and more alpha values. This
# removes the visually apparent alias that a short, coarse action trace misses.
resolved_action = dict(nalpha=9, points_per_period=32, num_periods=10,
                       max_wells=24, quadrature_order=24)
ACTION_MBOZ = [8, 8, 10, 10]
ACTION_OPTIONS = [coarse_action, coarse_action, resolved_action, resolved_action]

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QI_SEED_MAX_MODES, QI_SEED_MAX_NFEV = [1], [2]
    QI_SEED_ESS_ALPHA = [1.2]
    MAX_MODES, MAX_NFEV = [1], [2]
    MAXIMUM_J_TARGETS, MAXIMUM_J_WEIGHTS = [0.0], [500.0]
    QI_INVARIANCE_WEIGHTS = [1.0e3]
    CONSTRUCTED_QI_WEIGHTS = [1.0e4]
    MAGNETIC_WELL_WEIGHTS = [100.0]
    SURFACES, TRAPPING_DEPTHS = np.array([0.25, 0.45, 0.65, 0.85]), (0.5,)
    qi_options = dict(nphi=25, nalpha=5, n_bounce=5)
    coarse_action = dict(nalpha=5, points_per_period=24, num_periods=6,
                         max_wells=16, quadrature_order=16)
    ACTION_MBOZ = [8]
    ACTION_OPTIONS = [coarse_action]

# Start from the same transparent vacuum seed used by the other optimization
# examples. A rotating ellipse gives iota; the QI-only first stage then creates
# the common trapped-well topology required by the physical-pitch J objective.
DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)
qi = ConstructedQIResidual(SURFACES, mboz=8 if ci_smoke else 14,
                           nboz=8 if ci_smoke else 14, **qi_options)

def mirror_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)

def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - jnp.abs(opt.mean_iota(equilibrium_state, solver_context)), 0.0)

def magnetic_well_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        MAGNETIC_WELL_TARGET - opt.magnetic_well(equilibrium_state, solver_context), 0.0)

shape_terms = [
    (qi, 0.0, QI_SEED_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (mirror_excess, 0.0, 100.0),
    (magnetic_well_floor, 0.0, MAGNETIC_WELL_WEIGHTS[0]),
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
for max_mode, max_nfev, ess_alpha in zip(
        QI_SEED_MAX_MODES, QI_SEED_MAX_NFEV, QI_SEED_ESS_ALPHA):
    mpol, ntor = max(inp.mpol, max_mode + 2, MINIMUM_MPOL), max(inp.ntor, max_mode + 2)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6, nzeta=2 * ntor + 4)
    print(f"\n===== QI seed stage, max_mode = {max_mode} =====")
    problem = opt.VmecProblem.from_tuples(inp, shape_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ess_alpha,
        restart_from=equilibrium, forward_ftol=1e-6,
        forward_max_iterations=500, progress=not ci_smoke)
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

if ci_smoke:
    # The smoke lane already exercised minimal-seed continuation above. Use a
    # bundled QI state for the separate maximum-J wiring check so nightly CI
    # does not spend minutes forming matched wells before testing one AD step.
    inp = vj.VmecInput.from_file(
        Path(__file__).resolve().parents[1] / "data" / "input.nfp3_QI_fixed_resolution_final")
    inp = replace(inp, ns_array=np.array([11]), ftol_array=np.array([1e-8]),
                  niter_array=np.array([2000]))
    equilibrium = opt.solve_equilibrium(inp)

# Select field strengths that trap the same particles on every sampled line.
# After the weak first maximum-J stage, keep those pitches fixed so every
# later stage differentiates the same particle population.
pitch = np.asarray(common_trapped_pitches_state(
    equilibrium.solution, equilibrium.solver_context, SURFACES, TRAPPING_DEPTHS))
for stage, (max_mode, max_nfev, maxj_target, maxj_weight, qi_weight,
            constructed_weight, well_weight, action_mboz, action_options) in enumerate(zip(
        MAX_MODES, MAX_NFEV, MAXIMUM_J_TARGETS, MAXIMUM_J_WEIGHTS,
        QI_INVARIANCE_WEIGHTS, CONSTRUCTED_QI_WEIGHTS, MAGNETIC_WELL_WEIGHTS,
        ACTION_MBOZ, ACTION_OPTIONS)):
    print(f"\n===== QI + maximum-J stage, max_mode = {max_mode}, "
          f"target = {maxj_target:g}, weight = {maxj_weight:g} =====")
    # Re-select the common physical pitches once after the weak first stage,
    # then keep the same trapped particles in every stronger stage.
    if stage == 1:
        pitch = np.asarray(common_trapped_pitches_state(
            equilibrium.solution, equilibrium.solver_context, SURFACES, TRAPPING_DEPTHS))
    qi_maxj = JInvariantQIAndMaximumJResidual(SURFACES, pitch,
        mboz=action_mboz, nboz=action_mboz,
        qi_options=action_options, qi_weight=qi_weight,
        maxj_weight=maxj_weight,
        maxj_options={**action_options, "target": maxj_target})
    maxj_diagnostics = qi_maxj.compute_state(
        equilibrium.solution, equilibrium.solver_context)["maximum_j"]
    if not bool(jnp.all(maxj_diagnostics["valid_pitch_pair"])):
        raise RuntimeError(
            "the QI seed does not retain usable trapped wells across the sampled surfaces; "
            "increase the preceding continuation stage")
    stage_shape_terms = [
        (qi, 0.0, constructed_weight),
        (opt.aspect_ratio, ASPECT_TARGET, 1.0),
        (iota_floor, 0.0, 10.0),
        (mirror_excess, 0.0, 100.0),
        (magnetic_well_floor, 0.0, well_weight),
    ]
    objective_function_terms = [(qi_maxj, 0.0, 1.0), *stage_shape_terms]
    mpol, ntor = max(inp.mpol, max_mode + 2, MINIMUM_MPOL), max(inp.ntor, max_mode + 2)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6, nzeta=2 * ntor + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=0.7,
        restart_from=equilibrium, forward_ftol=1e-7,
        forward_max_iterations=800, progress=not ci_smoke)
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
    stage_maxj = qi_maxj.compute_state(
        equilibrium.solution, equilibrium.solver_context)["maximum_j"]
    print(f"actual-field maximum-J fraction = "
          f"{float(stage_maxj['maximum_j_fraction']):.1%}")

final_input = replace(inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]), niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
report("final", final_equilibrium)
# The final report repeats the actual-field J-invariance and matched-well dJ/ds
# at the most resolved action quadrature used by the continuation.
qi_maxj_certificate = JInvariantQIAndMaximumJResidual(SURFACES, pitch,
    mboz=ACTION_MBOZ[-1], nboz=ACTION_MBOZ[-1],
    qi_options=ACTION_OPTIONS[-1], qi_weight=1.0, maxj_weight=1.0,
    maxj_options={**ACTION_OPTIONS[-1], "target": MAXIMUM_J_TARGETS[-1]})
diagnostics = qi_maxj_certificate.compute_state(
    final_equilibrium.solution, final_equilibrium.solver_context)
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
