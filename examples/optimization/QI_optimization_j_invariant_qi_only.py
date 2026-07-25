#!/usr/bin/env python
"""Pure J-invariant QI optimization from the staged example seed.

This script keeps the same circular-torus seed and practical constraints as
``QI_optimization_j_invariant.py`` but runs only the J-invariant QI objective.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity_j import JInvariantQIResidual

# --------------------------- parameters ------------------------------------
NFP = 1
MPOL = NTOR = 7
R0, A_MINOR = 1.0, 0.2
PHIEDGE = 0.083
OUT_DIR = Path("output_QI_optimization_j_invariant_qi_only")
SURFACES = np.linspace(0.1, 1.0, 6)
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.15
MIRROR_TARGET = 0.20
MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5, 6)
QI_NFEV = 1000
FTOL = 1e-6

# --------------------------- seed equilibrium -------------------------------
rbc = np.zeros((2 * NTOR + 1, MPOL))
zbs = np.zeros((2 * NTOR + 1, MPOL))
rbc[NTOR, 0] = R0
rbc[NTOR, 1] = A_MINOR
zbs[NTOR, 1] = A_MINOR
inp = vj.VmecInput(
    nfp=NFP, mpol=MPOL, ntor=NTOR, rbc=rbc, zbs=zbs, phiedge=PHIEDGE,
    lasym=False, lfreeb=False, mgrid_file="NONE",
    ncurr=1, curtor=0.0, pres_scale=1.0,
    ns_array=[35], ftol_array=[1e-13], niter_array=[1500], delt=0.9,
)
eq = opt.solve_equilibrium(inp)
qp = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)
qi = JInvariantQIResidual(SURFACES)


def iota_shortfall(state, rt):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, rt)), 0.0)


def report(tag, eq):
    print(
        f"[{tag}] QI total = {float(qi.total(eq)):.6e}, "
        f"QP total = {float(qp.total(eq)):.6e}, "
        f"aspect = {float(opt.aspect_ratio(eq.state, eq.runtime)):.4f}, "
        f"mean iota = {float(opt.mean_iota(eq.state, eq.runtime)):.4f}"
    )
    return float(qi.total(eq))


qi_seed = report("seed", eq)

practical_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, 0.25),
    (iota_shortfall, 0.0, 100.0),
    (opt.mirror_ratio, MIRROR_TARGET, 10.0),
]
qi_terms = [(qi, 0.0, 10.0)] + practical_terms

for max_mode in MAX_MODE_SCHEDULE:
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    result = opt.least_squares(
        qi_terms, inp, max_mode=max_mode, jac="implicit",
        use_ess=True, verbose=1, max_nfev=QI_NFEV, ftol=FTOL, xtol=1e-10,
    )
    inp = result.input
    if result.equilibrium is not None:
        report(f"QI stage {max_mode}", result.equilibrium)

eq = result.equilibrium or opt.solve_equilibrium(inp)
qi_final = report("final", eq)
print(f"\nQI total: seed {qi_seed:.3e} -> final {qi_final:.3e}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUT_DIR / "input.QI_j_invariant_only_optimized")
wout_path = vj.write_wout(OUT_DIR / "wout_QI_j_invariant_only_optimized.nc", eq.wout)
print(f"wrote {OUT_DIR / 'input.QI_j_invariant_only_optimized'}\nwrote {wout_path}")
for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
    print(f"wrote {path}")
