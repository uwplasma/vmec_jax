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


DATA = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [1, 2, 3], 200
ASPECT_TARGET, IOTA_FLOOR, MIRROR_TARGET, MINIMUM_MPOL = 6.0, 0.15, 0.20, 5
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], 4

inp = vj.VmecInput.from_file(DATA)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)


def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, runtime)), 0.0)


def report(label, equilibrium):
    total = float(qs.total(equilibrium))
    print(f"[{label}] QS total = {total:.6e}, "
          f"aspect = {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"mirror = {float(opt.mirror_ratio(equilibrium.state, equilibrium.runtime)):.4f}")
    return total


terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
         (iota_floor, 0.0, 100.0), (opt.mirror_ratio, MIRROR_TARGET, 10.0)]
seed = opt.solve_equilibrium(inp)
seed_total = report("seed", seed)
for max_mode in MAX_MODES:
    print(f"\n===== QP stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=max_mode, use_ess=True)
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=MAX_NFEV, ftol=1e-6, xtol=1e-10, verbose=2)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    inp.to_indata(f"input.QP_max_mode_{max_mode:03d}")

final_total = report("final", equilibrium)
print(f"\nQS total: seed {seed_total:.3e} -> final {final_total:.3e}")
input_path = inp.to_indata("input.QP_optimized")
wout_path = vj.write_wout("wout_QP_optimized.nc", equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
