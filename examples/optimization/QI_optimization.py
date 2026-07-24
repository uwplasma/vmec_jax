#!/usr/bin/env python
"""Quasi-isodynamic (QI) optimization from a circular torus, nfp=1.

Two-stage campaign, one terms-list swap — the "QP first, then QI" route,
now with exact implicit gradients in *both* stages (R26h.h2):

1. **QP basin** (implicit gradients): drive the quasisymmetry ratio residual
   with helicity (m, n) = (0, 1) plus aspect / iota-floor / mirror targets.
   This forms poloidally closed ``|B|`` contours — the topological
   prerequisite of omnigenity — from the crude circular seed.
2. **QI refinement** (implicit gradients): swap the QP term for the
   *traceable* omnigenity residual (:class:`vmex.core.omnigenity.
   QIResidual` — Goodman constructed-QI-target distance on a fully
   differentiable in-state Boozer ``|B|`` transform: bounce-distance
   uniformity, extremum-contour closure, single-well monotonicity;
   Goodman et al., J. Plasma Phys. 89, 905890504 (2023), arXiv:2211.09829).
   Unlike the earlier wout/booz_xform residual this stage runs with
   ``jac="implicit"`` too — one exact Jacobian per trust-region step instead
   of one finite-difference equilibrium solve per boundary dof — so the
   continuation ladder extends to max_mode 6 (168 boundary dofs).

The full boundary-harmonic ladder is ``MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5,
6)``: the QP stages climb 1 -> 3 (the basin is insensitive to the deep
harmonics), the QI stages refine at 3 and continue 4 -> 6.  The seed is
built from scratch (mpol = ntor = 7, one harmonic above the deepest stage)
with the same circular-torus coefficients as ``input.minimal_seed_nfp1``.

Honest history: the previous revision of this script (git history) used the
distilled 4-term wout-engine QI residual with finite differences for stage
2 (max_mode 3 only) and reached QI total 2.139e-02 from a 2.430 seed on the
office 36-core CPU — a > 2-order improvement but *not* precise QI, with the
residual plateauing near the FD noise floor.  This revision replaces the
formulation and the gradient path (the two suspects for that plateau);
full-budget achieved numbers for the new pipeline are not yet recorded —
expect a multi-hour CPU run at the default budget.  Stage 1 remains
basin-sensitive: different CPU runs can land in different QP basins, and
the implicit path is required (finite differences land in a much worse
basin; cf. ``QP_optimization.py``).

Validation of the objective itself lives in ``tests/test_omnigenity.py``:
the traceable Boozer spectrum matches booz_xform_jax mode-by-mode, the
residual is exactly zero on an analytic QI field, far lower on the bundled
QI deck than on tokamak/QA states (same ordering as the wout-engine QI
total), and composes through ``jac="implicit"``.
"""

import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity import QIResidual
from vmex.core.omnigenity_j import JInvariantQIAndMaxJResidual, JInvariantQIResidual

# --------------------------- parameters ------------------------------------
NFP = 1
MPOL = NTOR = 7                            # one harmonic above max_mode 6
R0, A_MINOR = 1.0, 0.2                     # circular-torus seed (minimal_seed_nfp1)
PHIEDGE = 0.083
OUT_DIR = Path("output_QI_optimization")
SURFACES = np.linspace(0.1, 1.0, 6)        # QP and QI surfaces
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.15
MIRROR_TARGET = 0.20
MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5, 6)     # full boundary-harmonic ladder
QP_SCHEDULE = MAX_MODE_SCHEDULE[:3]        # stage 1: QP basin (implicit)
QI_SCHEDULE = MAX_MODE_SCHEDULE[2:]        # stage 2: QI refinement (implicit)
QP_NFEV, QI_NFEV = 2000, 1000              # trial budgets per stage
FTOL = 1e-6                                # per-stage convergence tolerance
QI_OBJECTIVE = os.environ.get("VMEX_QI_OBJECTIVE", "goodman").strip().lower()
if os.environ.get("VMEX_EXAMPLES_CI") == "1":  # smoke-test budget
    QP_SCHEDULE, QI_SCHEDULE, QP_NFEV, QI_NFEV = (1,), (1,), 6, 3
    FTOL = 1e-4

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
if QI_OBJECTIVE == "goodman":
    qi = QIResidual(SURFACES)              # traceable omnigenity residual
elif QI_OBJECTIVE == "j_invariant":
    qi = JInvariantQIResidual(SURFACES)
elif QI_OBJECTIVE == "j_invariant_maxj":
    qi = JInvariantQIAndMaxJResidual(SURFACES)
else:
    raise ValueError(
        "VMEX_QI_OBJECTIVE must be 'goodman', 'j_invariant', or 'j_invariant_maxj'."
    )


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
# The traceable omnigenity residual differentiates end-to-end, so the QI
# stages use the same implicit-gradient path as the QP basin stages.
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
inp.to_indata(OUT_DIR / "input.QI_optimized")
wout_path = vj.write_wout(OUT_DIR / "wout_QI_optimized.nc", eq.wout)
print(f"wrote {OUT_DIR / 'input.QI_optimized'}\nwrote {wout_path}")
for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
    print(f"wrote {path}")
