#!/usr/bin/env python
"""Quasi-isodynamic (QI) optimization from a circular torus, nfp=1.

This is the staged ``QI_optimization.py`` example with the QI refinement
objective fixed to the J-based implementation
(:class:`vmex.core.omnigenity_j.JInvariantQIResidual`) instead of the
Goodman-style residual.  No environment variables are needed.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity_j import JInvariantQIResidual

# --------------------------- parameters ------------------------------------
NFP = 1
MPOL = NTOR = 7                            # one harmonic above max_mode 6
R0, A_MINOR = 1.0, 0.2                     # circular-torus seed (minimal_seed_nfp1)
PHIEDGE = 0.083
OUT_DIR = Path("output_QI_optimization_j_invariant")
SURFACES = np.linspace(0.1, 1.0, 6)        # QP and QI surfaces
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.15
MIRROR_TARGET = 0.20
MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5, 6)     # full boundary-harmonic ladder
QP_SCHEDULE = MAX_MODE_SCHEDULE[:3]        # stage 1: QP basin (implicit)
QI_SCHEDULE = MAX_MODE_SCHEDULE[2:]        # stage 2: QI refinement (implicit)
QP_NFEV, QI_NFEV = 2000, 1000              # trial budgets per stage
FTOL = 1e-6                                # per-stage convergence tolerance
QI_OBJECTIVE = "j_invariant"

# --------------------------- seed equilibrium -------------------------------
rbc = np.zeros((2 * NTOR + 1, MPOL))       # dense INDATA layout [n + NTOR, m]
zbs = np.zeros((2 * NTOR + 1, MPOL))
rbc[NTOR, 0] = R0                          # RBC(0,0): major radius
rbc[NTOR, 1] = A_MINOR                     # RBC(0,1) = ZBS(0,1): circular
zbs[NTOR, 1] = A_MINOR                     # cross-section of radius a
inp = vj.VmecInput(
    nfp=NFP, mpol=MPOL, ntor=NTOR, rbc=rbc, zbs=zbs, phiedge=PHIEDGE,
    lasym=False, lfreeb=False, mgrid_file="NONE",
    ncurr=1, curtor=0.0, pres_scale=1.0,   # AM defaults to 0: vacuum, no net current
    ns_array=[35], ftol_array=[1e-13], niter_array=[1500], delt=0.9,
)
eq = opt.solve_equilibrium(inp)
qp = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)
qi = JInvariantQIResidual(SURFACES)


def iota_shortfall(state, rt):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, rt)), 0.0)


def report(tag, eq):
    qi_total = float(qi.total(eq))
    print(f"[{tag}] objective[{QI_OBJECTIVE}] = {qi_total:.6e}, QP total = {float(qp.total(eq)):.6e}, "
          f"aspect = {float(opt.aspect_ratio(eq.state, eq.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(eq.state, eq.runtime)):.4f}")
    return qi_total


qi_seed = report("seed", eq)

# ------------------- objectives: one terms-list swap ------------------------
practical_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, 0.25),
    (iota_shortfall, 0.0, 100.0),
    (opt.mirror_ratio, MIRROR_TARGET, 10.0),
]
qp_terms = [(qp, 0.0, 1.0)] + practical_terms             # stage 1
qi_terms = [(qi, 0.0, 10.0)] + practical_terms            # stage 2

# --------------------------- stage 1: QP basin ------------------------------
for max_mode in QP_SCHEDULE:
    print(f"\n===== QP stage, max_mode = {max_mode} =====")
    result = opt.least_squares(
        qp_terms, inp, max_mode=max_mode, jac="implicit",
        use_ess=True, verbose=1, max_nfev=QP_NFEV, ftol=FTOL, xtol=1e-10,
    )
    inp = result.input
    if result.equilibrium is not None:
        report(f"QP stage {max_mode}", result.equilibrium)

# --------------------------- stage 2: QI refinement -------------------------
for max_mode in QI_SCHEDULE:
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    result = opt.least_squares(
        qi_terms, inp, max_mode=max_mode, jac="implicit",
        use_ess=True, verbose=1, max_nfev=QI_NFEV, ftol=FTOL, xtol=1e-10,
    )
    inp = result.input
    if result.equilibrium is not None:
        report(f"QI stage {max_mode}", result.equilibrium)

# --------------------------- final results ---------------------------------
eq = result.equilibrium or opt.solve_equilibrium(inp)
qi_final = report("final", eq)
print(f"\nQI total: seed {qi_seed:.3e} -> final {qi_final:.3e}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUT_DIR / "input.QI_j_invariant_optimized")
wout_path = vj.write_wout(OUT_DIR / "wout_QI_j_invariant_optimized.nc", eq.wout)
print(f"wrote {OUT_DIR / 'input.QI_j_invariant_optimized'}\nwrote {wout_path}")
for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
    print(f"wrote {path}")
