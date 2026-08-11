#!/usr/bin/env python
"""Quasi-helically symmetric boundary optimization from a circular nfp=4 seed."""

import os

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt


NFP, R0, A_MINOR = 4, 1.0, 0.125
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [1, 2, 3, 4, 5], 200
ASPECT_TARGET, MINIMUM_MPOL = 8.0, 5
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], 4

mpol = ntor = max(MAX_MODES) + 2
rbc, zbs = np.zeros((2 * ntor + 1, mpol)), np.zeros((2 * ntor + 1, mpol))
rbc[ntor, 0], rbc[ntor, 1], zbs[ntor, 1] = R0, A_MINOR, A_MINOR
inp = vj.VmecInput(
    nfp=NFP, mpol=mpol, ntor=ntor, rbc=rbc, zbs=zbs,
    phiedge=np.pi * A_MINOR**2, lasym=False, lfreeb=False, mgrid_file="NONE",
    ncurr=1, curtor=0.0, pres_scale=0.0,
    ns_array=[35], ftol_array=[1e-13], niter_array=[3000], delt=0.5)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0)]


def report(label, equilibrium):
    total = float(qs.total(equilibrium))
    print(f"[{label}] QS total = {total:.6e}, "
          f"aspect = {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(equilibrium.state, equilibrium.runtime)):.4f}")
    return total


seed = opt.solve_equilibrium(inp)
seed_total = report("seed", seed)
for max_mode in MAX_MODES:
    print(f"\n===== QH stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = inp.change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=max_mode, use_ess=True)
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=MAX_NFEV, ftol=1e-6, xtol=1e-10, verbose=2)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    inp.to_indata(f"input.QH_max_mode_{max_mode:03d}")

final_total = report("final", equilibrium)
print(f"\nQS total: seed {seed_total:.3e} -> final {final_total:.3e}")
input_path = inp.to_indata("input.QH_optimized")
wout_path = vj.write_wout("wout_QH_optimized.nc", equilibrium.wout)
print(f"wrote {input_path}\nwrote {wout_path}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"wrote {path}")
