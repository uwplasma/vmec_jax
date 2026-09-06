#!/usr/bin/env python
"""Finite-beta quasi-axisymmetric optimization with one scalar adjoint.

The scalar lane trades objective progress per evaluation for a cheaper cold
start and lower peak memory: at a matched evaluation budget the least-squares
driver reached roughly a 3x lower objective on the same problem, so it remains
the default for objective progress.
"""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import vmex as vj
from vmex import optimize as opt
from _scalar_driver import run_scalar_stage

nfp, TARGET_BETA = 2, 0.01
SURFACES = np.linspace(0.1, 0.9, 8)
MAX_MODES, MAXITER = [2, 4], [20, 45]
ASPECT_TARGET, IOTA_FLOOR, MAGNETIC_WELL_TARGET = 5.0, 0.42, 0.01
STABILITY_MIN_S, STABILITY_WEIGHT, EDGE_WEIGHT_FACTOR = 0.2, 1e-6, 10.0
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.01, 3.0
ESS_ALPHA, MINIMUM_MPOL = 1.2, 5
VARY_MAJOR_RADIUS = False
SEED_PERTURBATION = 0.05
POLISH_FORCE_BALANCE = False  # Set True to polish only the final saved state.

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.array([0.25, 0.6, 0.9])
    MAX_MODES, MAXITER, MINIMUM_MPOL = [1], [2], 3

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
am = np.zeros(21)
am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1-s)
inp = replace(inp, rbc=rbc, zbs=zbs, pmass_type="power_series", am=am,
              pres_scale=100.0)
calibration = opt.solve_equilibrium(inp)
inp = replace(
    inp, pres_scale=inp.pres_scale * TARGET_BETA / float(calibration.wout.betatotal))
equilibrium = opt.solve_equilibrium(inp, initial_state=calibration.solution)

stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_weights = np.where(
    stability_s >= STABILITY_MIN_S,
    STABILITY_WEIGHT * (1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4), 0.0)


def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_terms = [
    (qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2),
    (opt.mercier_stability_residual, 0.0, stability_weights),
    (opt.glasser_stability_residual, 0.0, stability_weights)]


report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("beta", opt.volume_average_beta, ".3%"),
    ("aspect", opt.aspect_ratio, ".3f"),
    ("min |iota|", opt.min_abs_iota, ".3f"))
monitor = opt.OptimizationMonitor(stream=None)

for max_mode, maxiter in zip(MAX_MODES, MAXITER):
    inp, equilibrium = run_scalar_stage(
        inp, equilibrium, objective_terms, label="finite-beta QA",
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
    niter_array=np.array([20000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True,
    polish_force_balance=POLISH_FORCE_BALANCE)
report("final", final_equilibrium)

input_path = final_input.to_indata("input.QA_finite_beta_scalar_optimized")
wout_path = vj.write_wout(
    "wout_QA_finite_beta_scalar_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QA_finite_beta_scalar_objectives.csv")
monitor.plot("QA_finite_beta_scalar_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
