#!/usr/bin/env python
"""Constructed-QI optimization with one adjoint per scalar gradient.

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
from vmex.core.qi import ConstructedQIResidual
from _scalar_driver import run_scalar_stage

nfp = 2
SURFACES = np.linspace(0.1, 1.0, 6)
MAX_MODES, MAXITER = [3], [250]
ASPECT_TARGET, IOTA_FLOOR = 5.0, 0.51
MIRROR_LIMIT, ELONGATION_LIMIT = 0.21, 8.0
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
ESS_ALPHA, MINIMUM_MPOL = 1.2, 5
VARY_MAJOR_RADIUS = False
SEED_PERTURBATION = 0.05
POLISH_FORCE_BALANCE = False  # Set True to polish only the final saved state.
qi_options = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
validation_options = dict(mboz=14, nboz=14, nphi=101, nalpha=29, n_bounce=31)

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    qi_options = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    validation_options = qi_options
    MAX_MODES, MAXITER = [2], [5]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)


def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0.0)


def mirror_excess(state, runtime):
    return jnp.maximum(opt.mirror_ratio(state, runtime) - MIRROR_LIMIT, 0.0)


def elongation_excess(state, runtime):
    return jnp.maximum(opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0.0)


qi = ConstructedQIResidual(SURFACES, **qi_options)
objective_terms = [
    (qi, 0.0, 10.0), (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0), (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0)]


report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".6e"),
    ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"),
    ("mirror", opt.mirror_ratio, ".4f"),
    ("elongation", opt.max_elongation, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)
equilibrium = opt.solve_equilibrium(inp)

for max_mode, maxiter in zip(MAX_MODES, MAXITER):
    inp, equilibrium = run_scalar_stage(
        inp, equilibrium, objective_terms, label="QI",
        max_mode=max_mode, maxiter=maxiter,
        parameter_step=PARAMETER_STEP,
        max_parameter_change=MAX_PARAMETER_CHANGE,
        minimum_mpol=MINIMUM_MPOL,
        vary_major_radius=VARY_MAJOR_RADIUS,
        ess_alpha=ESS_ALPHA, monitor=monitor, compile_first=not ci_smoke,
        progress=not ci_smoke)
    report(f"QI mode {max_mode}", equilibrium)

final_input = replace(
    inp, ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1e-10 if ci_smoke else 1e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True,
    polish_force_balance=POLISH_FORCE_BALANCE)
qi_final = report("final", final_equilibrium)["constructed QI"]
qi_validation = ConstructedQIResidual(SURFACES, **validation_options)
print(f"QI total {qi_final:.3e}; independent fine-grid validation "
      f"{float(qi_validation.total(final_equilibrium)):.3e}")

input_path = final_input.to_indata("input.QI_scalar_optimized")
wout_path = vj.write_wout("wout_QI_scalar_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
monitor.save("QI_scalar_optimization_objectives.csv")
monitor.plot("QI_scalar_optimization_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
