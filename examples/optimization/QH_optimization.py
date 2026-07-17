#!/usr/bin/env python
"""Precise quasi-helical symmetry (QH) from a circular torus, nfp=4.

Same recipe as ``QA_optimization.py`` with helicity (m, n) = (1, -1): the
quasisymmetry ratio residual now demands ``|B| = |B|(s, theta + nfp*phi)``,
which an axisymmetric torus cannot satisfy — so unlike the QA case no seed
kick is needed (compare simsopt's ``QH_fixed_resolution.py``).  Note this
relies on JAC="implicit": the QS residual is *even* in the symmetry-breaking
harmonics, so its gradient vanishes at the exact-axisymmetric seed — a
saddle where finite differences stall (measured: QS unchanged, one Jacobian
evaluation), but the exact implicit gradient (internal-grid, plus the tiny
asymmetry of the host solve) escapes it and the QS term pulls the boundary
into a helically symmetric shape.

This script also demonstrates building a :class:`vmex.VmecInput` from
scratch instead of reading a file: the circular-torus seed (R0 = 1 m,
a = 1/8 m, ~1 T) is assembled directly from its Fourier coefficients.

Runtime per continuation stage is dominated by the one-time implicit-
Jacobian XLA compile (the warm forward solve is ~0.9 s).  Achieved
2026-07-11 with JAC="implicit" + ESS on the office 36-core CPU (the
implicit Jacobian is CPU-pinned by default; see below) -- the full
max_mode 1 -> 5 continuation reaches *precise* QH:

    QS total 6.908e-01 (circular seed)
        -> 1.401e-01 (max_mode 1, iota -0.917)
        -> 2.788e-03 (max_mode 2, iota -1.148)
        -> 2.407e-04 (max_mode 3, iota -1.207)
        -> 1.647e-04 (max_mode 4, iota -1.218)
        -> 5.831e-05 (max_mode 5, iota -1.218), aspect 8.000

i.e. > 4 orders of magnitude below the axisymmetric seed (the
Landreman-Paul precise-QH literature value is fQS ~ 2e-3,
arXiv:2311.16386, so this is comfortably precise).  The deep stages are
tractable on CPU because the per-dof implicit Jacobian is launch-bound
(one preconditioned GMRES per boundary dof): a max_mode-2 (24-dof)
Jacobian evaluates in ~101 s on CPU versus > 37 min hung in a single
kernel-launch-bound GMRES on the GPU before the CPU pin (R1).
The whole 1->5 campaign is a multi-hour CPU run.
"""

import os
from pathlib import Path

import numpy as np

import vmex as vj
from vmex import optimize as opt

# --------------------------- parameters ------------------------------------
NFP = 4
MPOL = NTOR = 6                            # one harmonic above max_mode 5
R0, A_MINOR = 1.0, 0.125                   # circular-torus seed, aspect 8
PHIEDGE = np.pi * A_MINOR**2               # ~1 T mean field
OUT_DIR = Path("output_QH_optimization")
QS_SURFACES = np.linspace(0.1, 1.0, 10)
HELICITY_M, HELICITY_N = 1, -1             # QH: |B| = |B|(s, theta + nfp*phi)
ASPECT_TARGET = 8.0
MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5)
MAX_NFEV = 2000                            # trial budget per stage
FTOL = 1e-6                                # per-stage convergence tolerance
JAC = "implicit"
if os.environ.get("VMEX_EXAMPLES_CI") == "1":  # smoke-test budget
    MAX_MODE_SCHEDULE, MAX_NFEV, FTOL = (1,), 4, 1e-4

# --------------------------- seed input, built from scratch -----------------
rbc = np.zeros((2 * NTOR + 1, MPOL))       # dense INDATA layout [n + NTOR, m]
zbs = np.zeros((2 * NTOR + 1, MPOL))
rbc[NTOR, 0] = R0                          # RBC(0,0): major radius
rbc[NTOR, 1] = A_MINOR                     # RBC(0,1) = ZBS(0,1): circular
zbs[NTOR, 1] = A_MINOR                     # cross-section of radius a
inp = vj.VmecInput(
    nfp=NFP, mpol=MPOL, ntor=NTOR, rbc=rbc, zbs=zbs, phiedge=PHIEDGE,
    lasym=False, lfreeb=False, mgrid_file="NONE",
    ncurr=1, curtor=0.0, pres_scale=0.0,   # vacuum, zero net current
    ns_array=[35], ftol_array=[1e-13], niter_array=[3000], delt=0.9,
)
eq = opt.solve_equilibrium(inp)
qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, HELICITY_M, HELICITY_N)


def report(tag, eq):
    total = float(qs.total(eq))
    aspect = float(opt.aspect_ratio(eq.state, eq.runtime))
    iota = float(opt.mean_iota(eq.state, eq.runtime))
    print(f"[{tag}] QS total = {total:.6e}, aspect = {aspect:.4f}, "
          f"mean iota = {iota:.4f}")
    return total


qs_seed = report("seed", eq)

# --------------------------- objective (user-authored) ----------------------
objective_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    # CI-tested extras (see QA_optimization.py for the jac caveats):
    # (opt.magnetic_well, 0.05, 1.0),
    # (lambda eq: np.minimum(opt.d_merc(eq)[2:-1], 0.0), 0.0, 100.0),
    # (lambda eq: max(1.0 / opt.l_grad_b(eq) - 1.0 / 0.35, 0.0), 0.0, 1.0),
]

# --------------------------- staged optimization ----------------------------
for max_mode in MAX_MODE_SCHEDULE:
    ndofs = len(opt.boundary_dof_names(inp, max_mode))
    print(f"\n===== stage max_mode = {max_mode} ({ndofs} boundary dofs) =====")
    result = opt.least_squares(
        objective_terms, inp, max_mode=max_mode, jac=JAC,
        use_ess=True, verbose=1, max_nfev=MAX_NFEV, ftol=FTOL, xtol=1e-10,
    )
    inp = result.input
    if result.equilibrium is not None:
        report(f"stage {max_mode}", result.equilibrium)

# --------------------------- final results ---------------------------------
eq = result.equilibrium or opt.solve_equilibrium(inp)
qs_final = report("final", eq)
print(f"\nQS total: seed {qs_seed:.3e} -> final {qs_final:.3e}")
names = opt.boundary_dof_names(inp, MAX_MODE_SCHEDULE[-1])
values = opt.pack_boundary(inp, MAX_MODE_SCHEDULE[-1])
print("optimized boundary (largest coefficients):")
for k in np.argsort(-np.abs(values))[:8]:
    print(f"  {names[k]:>10s} = {values[k]:+.6f}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUT_DIR / "input.QH_optimized")
wout_path = vj.write_wout(OUT_DIR / "wout_QH_optimized.nc", eq.wout)
print(f"wrote {OUT_DIR / 'input.QH_optimized'}\nwrote {wout_path}")
for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
    print(f"wrote {path}")
