#!/usr/bin/env python
"""Precise quasi-axisymmetry (QA) from a circular torus, nfp=2.

Mirrors simsopt's ``QH_fixed_resolution.py`` example: build an equilibrium,
write the objective yourself as ``(function, target, weight)`` terms —
here the Landreman-Paul QA recipe, the quasisymmetry ratio residual on a
set of surfaces plus aspect-ratio and mean-iota targets — and hand it to one
least-squares call per continuation stage.  The decision variables are the
boundary Fourier coefficients RBC/ZBS up to ``max_mode``, staged 1 -> 5,
with Exponential Spectral Scaling (ESS) of the trust region and implicit
(adjoint) gradients: no finite differences, no MPI.

The seed (``input.minimal_seed_nfp2``) is a circular torus, R0 = 1 m,
a = 0.2 m — exactly axisymmetric, so the QS term starts at ~0 and the iota
target pulls the boundary into three dimensions.

Runtime is dominated by the one-time implicit-Jacobian XLA compile per
continuation stage (the warm forward solve itself is ~0.9 s); each stage
then stops early at ftol.  Achieved 2026-07-10 on an office RTX A4000 GPU
at this script's default staged budget with JAC="implicit" + ESS: the QS
ratio residual total falls 2.043e-01 (circular seed) -> 9.82e-03
(max_mode=1) -> 1.70e-04 (max_mode=2) -- >3 orders of magnitude, i.e.
*precise* QA -- with aspect 6.000 and mean iota 0.420 both on target
(max_mode 3-5 polish further).  Per-stage wall was ~13 min (max_mode=1,
compile-bound) and ~21 min (max_mode=2).  A CPU run reaches the same
optimum: the cold forward solve is ~2x faster than the GPU (13 vs 27 s)
because this small solve is a host callback that the GPU does not help.
"""

import dataclasses
import os
from pathlib import Path

import numpy as np

import vmex as vj
from vmex import optimize as opt

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"
OUT_DIR = Path("output_QA_optimization")
QS_SURFACES = np.linspace(0.1, 1.0, 10)   # surfaces for the QS ratio residual
HELICITY_M, HELICITY_N = 1, 0             # QA: |B| = |B|(s, theta)
ASPECT_TARGET = 6.0
IOTA_TARGET = 0.42                        # mean rotational transform
SEED_PERTURBATION = 0.01                  # helical kick, see below
MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5)       # boundary-harmonic continuation
MAX_NFEV = 2000                           # trial-boundary budget per stage
FTOL = 1e-6                               # per-stage convergence tolerance
JAC = "implicit"                          # adjoint gradients; None = finite diff
if os.environ.get("VMEX_EXAMPLES_CI") == "1":  # smoke-test budget
    MAX_MODE_SCHEDULE, MAX_NFEV, FTOL = (1,), 4, 1e-4

# --------------------------- seed equilibrium ------------------------------
inp = vj.VmecInput.from_file(INPUT_FILE)   # plain &INDATA parsing
# The exact circular torus is a saddle point: with zero current the
# rotational transform is produced by 3D shaping at *second* order, so its
# gradient vanishes there.  A small RBC/ZBS(n=1, m=1) kick breaks the tie
# (this replaces the seed-preconditioner machinery of the old examples).
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor + 1, 1] += SEED_PERTURBATION
zbs[inp.ntor + 1, 1] += SEED_PERTURBATION
inp = dataclasses.replace(inp, rbc=rbc, zbs=zbs)
eq = opt.solve_equilibrium(inp)
qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, HELICITY_M, HELICITY_N)


def report(tag, eq):
    """User-side progress metric: the wout-engine QS total + scalar targets."""
    total = float(qs.total(eq))
    aspect = float(opt.aspect_ratio(eq.state, eq.runtime))
    iota = float(opt.mean_iota(eq.state, eq.runtime))
    print(f"[{tag}] QS total = {total:.6e}, aspect = {aspect:.4f}, "
          f"mean iota = {iota:.4f}")
    return total


qs_seed = report("seed", eq)

# --------------------------- objective (user-authored) ----------------------
objective_terms = [
    (qs, 0.0, 1.0),                       # quasisymmetry ratio residual
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 10.0),
    # Extra physics terms, CI-tested (tests/test_examples.py runs
    # them uncommented).  magnetic_well works with JAC="implicit"; d_merc and
    # l_grad_b are wout-engine (host) objectives -> set JAC = None for those.
    # (opt.magnetic_well, 0.05, 1.0),
    # (lambda eq: np.minimum(opt.d_merc(eq)[2:-1], 0.0), 0.0, 100.0),
    # (lambda eq: max(1.0 / opt.l_grad_b(eq) - 1.0 / 0.35, 0.0), 0.0, 1.0),
    # Turbulence proxies (plan R26h.h4; optional dep: pip install spectraxgk;
    # gates in tests/test_turbulence.py).  SPECTRAX-GK linear ITG growth rate
    # (traceable -> works with JAC="implicit" or JAC=None) and quasilinear
    # heat-flux proxy (eigenvector-weighted -> JAC=None only) on a core flux
    # tube:
    #   from vmex.core import turbulence as turb
    # (turb.turbulent_growth_rate, 0.0, 1.0),
    # (turb.quasilinear_flux_proxy, 0.0, 0.1),
]

# --------------------------- staged optimization ----------------------------
for max_mode in MAX_MODE_SCHEDULE:
    ndofs = len(opt.boundary_dof_names(inp, max_mode))
    print(f"\n===== stage max_mode = {max_mode} ({ndofs} boundary dofs) =====")
    result = opt.least_squares(
        objective_terms, inp, max_mode=max_mode, jac=JAC,
        use_ess=True, verbose=1, max_nfev=MAX_NFEV, ftol=FTOL, xtol=1e-10,
    )
    inp = result.input                     # warm-start the next stage
    if result.equilibrium is not None:
        report(f"stage {max_mode}", result.equilibrium)

# --------------------------- final results ---------------------------------
eq = result.equilibrium or opt.solve_equilibrium(inp)
qs_final = report("final", eq)
print(f"\nQS total: seed {qs_seed:.3e} -> final {qs_final:.3e}")
print("optimized boundary (largest coefficients):")
names = opt.boundary_dof_names(inp, MAX_MODE_SCHEDULE[-1])
values = opt.pack_boundary(inp, MAX_MODE_SCHEDULE[-1])
for k in np.argsort(-np.abs(values))[:8]:
    print(f"  {names[k]:>10s} = {values[k]:+.6f}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUT_DIR / "input.QA_optimized")               # optimized deck
wout_path = vj.write_wout(OUT_DIR / "wout_QA_optimized.nc", eq.wout)
print(f"wrote {OUT_DIR / 'input.QA_optimized'}\nwrote {wout_path}")
for key, path in vj.plot_wout(wout_path, OUT_DIR).items():  # figures
    print(f"wrote {path}")
