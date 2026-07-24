#!/usr/bin/env python
"""QI-only optimization from the bundled ``input.QI_nfp2_initial`` seed.

This variant keeps the J-based QI objective but removes the QP pre-stage.
It runs a pure QI continuation ladder with additional soft targets on:

- aspect ratio = 10.0
- mean iota = -0.61
- mirror ratio = 0.29
"""

from pathlib import Path

import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity_j import JInvariantQIResidual

# --------------------------- parameters ------------------------------------
SEED_INPUT = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"
OUT_DIR = Path("output_QI_optimization_j_invariant_targeted")
# QI surfaces / resolution in the same style as the older "snorms" setup.
SURFACES = np.asarray(
    [1 / 51, 5 / 51, 10 / 51, 15 / 51, 20 / 51, 25 / 51, 30 / 51, 35 / 51, 40 / 51, 45 / 51, 50 / 51],
    dtype=float,
)
QI_NPHI = 141
QI_NALPHA = 27
QI_NBOUNCE = 51
QI_MBOZ = 18
QI_NBOZ = 18
ASPECT_TARGET = 10.0
IOTA_TARGET = -0.61
MIRROR_TARGET = 0.25
QI_WEIGHT = 1.0
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
MIRROR_WEIGHT = 1.0e2
MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5, 6)
QI_NFEV = 50 
FTOL = 1e-6
QI_OBJECTIVE = "j_invariant"

# --------------------------- seed equilibrium -------------------------------
inp = vj.VmecInput.from_file(SEED_INPUT)
eq = opt.solve_equilibrium(inp)
qi = JInvariantQIResidual(
    SURFACES,
    nphi=QI_NPHI,
    nalpha=QI_NALPHA,
    n_bounce=QI_NBOUNCE,
    mboz=QI_MBOZ,
    nboz=QI_NBOZ,
)


def report(tag, eq):
    qi_total = float(qi.total(eq))
    aspect = float(opt.aspect_ratio(eq.state, eq.runtime))
    mean_iota = float(opt.mean_iota(eq.state, eq.runtime))
    mirror = float(opt.mirror_ratio(eq.state, eq.runtime))
    print(
        f"[{tag}] objective[{QI_OBJECTIVE}] = {qi_total:.6e}, "
        f"aspect = {aspect:.4f}, mean iota = {mean_iota:.4f}, mirror = {mirror:.4f}"
    )
    return qi_total


qi_seed = report("seed", eq)

# --------------------------- QI-only objective ------------------------------
qi_terms = [
    (qi, 0.0, QI_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.mean_iota, IOTA_TARGET, IOTA_WEIGHT),
    (opt.mirror_ratio, MIRROR_TARGET, MIRROR_WEIGHT),
]

# --------------------------- continuation ladder ----------------------------
for max_mode in MAX_MODE_SCHEDULE:
    print(f"\n===== QI-only stage, max_mode = {max_mode} =====")
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
inp.to_indata(OUT_DIR / SEED_INPUT.name)
inp.to_indata(OUT_DIR / "input.QI_j_invariant_targeted_optimized")
wout_path = vj.write_wout(OUT_DIR / "wout_QI_j_invariant_targeted_optimized.nc", eq.wout)
print(
    f"wrote {OUT_DIR / SEED_INPUT.name}\n"
    f"wrote {OUT_DIR / 'input.QI_j_invariant_targeted_optimized'}\n"
    f"wrote {wout_path}"
)
for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
    print(f"wrote {path}")
