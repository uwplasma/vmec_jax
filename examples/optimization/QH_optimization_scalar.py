#!/usr/bin/env python
"""Quasi-helical optimization with one adjoint per scalar gradient.

The scalar lane trades objective progress per evaluation for a cheaper cold
start and lower peak memory: at a matched evaluation budget the least-squares
driver reached roughly a 3x lower objective on the same problem, so it remains
the default for objective progress.
"""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np

import vmex as vj
from vmex import optimize as opt
from _scalar_driver import run_scalar_stage

nfp = 4
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAXITER = [2, 3], [15, 15]
ASPECT_TARGET = 6.0
TRIAL_BETA, USE_TRIAL_STABILITY = 0.025, False
STABILITY_COST_PER_SURFACE, EDGE_WEIGHT_FACTOR = 1e-2, 10.0
STABILITY_MIN_S, STABILITY_MARGIN = 0.2, 1e-3
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
ESS_ALPHA, MINIMUM_MPOL = 1.2, 5
VARY_MAJOR_RADIUS = False
SEED_PERTURBATION = 0.12
POLISH_FORCE_BALANCE = False  # Set True to polish only the final saved state.

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAXITER = [1], [4]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)


def trial_dmerc(state, runtime):
    return opt.trial_pressure_mercier_stability_residual(
        state, runtime, beta=TRIAL_BETA, margin=STABILITY_MARGIN)


def trial_dr(state, runtime):
    return opt.trial_pressure_glasser_stability_residual(
        state, runtime, beta=TRIAL_BETA, margin=STABILITY_MARGIN)


stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_shape = np.where(
    stability_s >= STABILITY_MIN_S,
    1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4, 0.0)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
objective_terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0)]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"),
    ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)
equilibrium = opt.solve_equilibrium(inp)

for stage, (max_mode, maxiter) in enumerate(zip(MAX_MODES, MAXITER)):
    stage_terms = objective_terms
    if USE_TRIAL_STABILITY and stage > 0:
        dmerc0 = np.asarray(trial_dmerc(equilibrium.solution, equilibrium.solver_context))
        dr0 = np.asarray(trial_dr(equilibrium.solution, equilibrium.solver_context))
        scale = np.maximum.reduce((np.abs(dmerc0), np.abs(dr0), np.ones_like(dmerc0)))
        weights = STABILITY_COST_PER_SURFACE * stability_shape / scale**2
        stage_terms = [*objective_terms,
                       (trial_dmerc, 0.0, weights), (trial_dr, 0.0, weights)]

    inp, equilibrium = run_scalar_stage(
        inp, equilibrium, stage_terms, label="QH",
        max_mode=max_mode, maxiter=maxiter,
        parameter_step=PARAMETER_STEP,
        max_parameter_change=MAX_PARAMETER_CHANGE,
        minimum_mpol=MINIMUM_MPOL,
        vary_major_radius=VARY_MAJOR_RADIUS,
        ess_alpha=ESS_ALPHA, monitor=monitor, compile_first=not ci_smoke)
    report(f"mode {max_mode}", equilibrium)

final_input = replace(
    inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True,
    polish_force_balance=POLISH_FORCE_BALANCE)
report("final", final_equilibrium)

input_path = final_input.to_indata("input.QH_scalar_optimized")
wout_path = vj.write_wout("wout_QH_scalar_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QH_scalar_optimization_objectives.csv")
monitor.plot("QH_scalar_optimization_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
