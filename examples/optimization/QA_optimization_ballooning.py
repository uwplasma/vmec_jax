#!/usr/bin/env python
"""Quasi-axisymmetric optimization against the infinite-n ballooning growth rate.

The seed (``input.nfp2_QA_finite_beta``, beta = 2.7 %) is Mercier-STABLE and
ballooning-UNSTABLE, which is the case a ballooning objective is for: the
interchange criteria see nothing to fix.  The objective is the smooth upper
bound ``ballooning_growth_rate``, so driving it below zero is a sufficient
condition for every sampled field line to be stable.
"""

from dataclasses import replace
from functools import partial
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.stability import ballooning_growth_rate, ballooning_lambda

SURFACES = np.linspace(0.1, 1.0, 6)
MAX_MODES, MAX_NFEV = [1, 2], [12, 20]
ASPECT_TARGET = 5.0
# lambda is least stable at a configuration-dependent zeta0 (Gaur et al.,
# J. Plasma Phys. 89 (2023), footnote 2).  On this seed the single-point
# default misses 26 % of it: max lambda is 3.27e-3 at zeta0 = 0 and 4.42e-3
# over the scan below, so a zeta0 = 0 objective would optimize a bound that
# is not the bound.
ZETA0S = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5)
LINES = dict(npoints=97, nturns=3.0, zeta0s=ZETA0S)
# The softmax bound sits above max lambda by at most temperature * log(nlines)
# (here 60 lines, so 8.2e-3): it is the quantity being targeted, and the hard
# max is what gets reported.
TEMPERATURE, BALLOONING_TARGET, BALLOONING_WEIGHT = 0.002, 0.0, 200.0
STAGE_NS, FINAL_NS = 25, 45
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 4.0
ESS_ALPHA = 1.2  # smaller values let high Fourier modes move more

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV, STAGE_NS, FINAL_NS = [1], [3], 15, 15

DATA = Path(__file__).resolve().parents[1] / "data" / "input.nfp2_QA_finite_beta"
inp = replace(vj.VmecInput.from_file(DATA),
              ns_array=np.array([STAGE_NS]), ftol_array=np.array([1.0e-11]),
              niter_array=np.array([4000]))

# The optimizable: a smooth upper bound on max lambda over all sampled lines.
ballooning = partial(ballooning_growth_rate, temperature=TEMPERATURE, **LINES)


def worst_lambda(equilibrium):
    """Hard max lambda over the sampled lines -- the number worth quoting."""
    lam = ballooning_lambda(equilibrium.solution, equilibrium.solver_context, **LINES)
    return float(np.max(np.asarray(lam)))


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (ballooning, BALLOONING_TARGET, BALLOONING_WEIGHT),
]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"),
    ("ballooning bound", ballooning, ".4e"))
monitor = opt.OptimizationMonitor(stream=None)

equilibrium = opt.solve_equilibrium(inp, verbose=not ci_smoke)
seed_lambda = worst_lambda(equilibrium)
seed_dmerc = float(np.min(np.asarray(equilibrium.wout.DMerc)[2:-1]))
print(f"\nseed: beta = {float(equilibrium.wout.betatotal):.3%}, "
      f"max lambda = {seed_lambda:+.4e} (unstable), "
      f"min DMerc = {seed_dmerc:+.3e} (Mercier-stable)")

for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QA + ballooning stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, 5)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(
        inp, objective_function_terms, max_mode=max_mode, use_ess=True,
        ess_alpha=ESS_ALPHA, restart_from=equilibrium)
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac, x_scale=step,
        bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    print(f"max lambda = {worst_lambda(equilibrium):+.4e}")

# The physics certificate is the resolved solve, not the optimizer's grid.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
                      ftol_array=np.array([1.0e-11 if ci_smoke else 1.0e-13]),
                      niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)
final_lambda = worst_lambda(final_equilibrium)
final_dmerc = float(np.min(np.asarray(final_equilibrium.wout.DMerc)[2:-1]))
report("final", final_equilibrium)
print(f"\nmax lambda {seed_lambda:+.4e} -> {final_lambda:+.4e} "
      f"({'stable' if final_lambda < 0.0 else 'still unstable'}) at NS = {FINAL_NS}\n"
      f"min DMerc {seed_dmerc:+.3e} -> {final_dmerc:+.3e}, "
      f"beta {float(final_equilibrium.wout.betatotal):.3%}")

input_path = final_input.to_indata("input.QA_ballooning_optimized")
wout_path = vj.write_wout("wout_QA_ballooning_optimized.nc", final_equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")

monitor.save("QA_optimization_ballooning_objectives.csv")
monitor.plot("QA_optimization_ballooning_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
