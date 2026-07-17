#!/usr/bin/env python
"""QI in ONE least-squares call: no mode ladder, no QP pre-stage — just ESS.

The staged QI example (``QI_optimization.py``) walks a quasi-poloidal basin
stage and then a QI refinement up the ``max_mode`` ladder.  This script is the
single-call alternative: release *all* the max_mode-6 boundary harmonics at
once and let **Exponential Spectral Scaling** (``use_ess=True``) impose the
coarse-to-fine ordering through the trust region — each dof's radius scales as
``exp(-alpha * max(|m|, |n|))``, so high harmonics move on exponentially
shorter leashes and no continuation loop is needed.

The objective combines the traceable Goodman constructed-QI omnigenity
residual (:class:`vmex.core.omnigenity.QIResidual`, implicit-adjoint
differentiable) with a weak quasi-poloidal term that plays the basin-guiding
role of the staged version's QP stage, plus the same aspect / iota-floor /
mirror practical targets.

Physics: nfp=1 vacuum quasi-isodynamic shaping from a circular torus,
mpol=ntor=7.  All gradients via ``jac="implicit"`` (adjoint + block-
tridiagonal Jacobian + perturbation warm start).  Measured 2026-07-12 on the
office 36-core CPU:

    seed QI 4.515e-01 -> final QI 1.812e-02 (25x) in ONE call, 1037 s
    (17.3 min), 168 dofs; iota 0.137, mirror 0.28 held (aspect relaxed to
    10.8 under its weak 0.25 weight — retighten it if a compact device is
    the goal).
"""

import os
from pathlib import Path

import numpy as np

import jax.numpy as jnp

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity import QIResidual

# --------------------------- parameters ------------------------------------
NFP = 1
MPOL = NTOR = 7                            # one harmonic above max_mode 6
R0, A_MINOR = 1.0, 1.0 / 6.0               # circular-torus seed, aspect 6
PHIEDGE = np.pi * A_MINOR**2               # ~1 T mean field
OUT_DIR = Path("output_QI_optimization_ess")
SURFACES = np.linspace(0.15, 0.95, 6)
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.12
MIRROR_TARGET = 0.20
MAX_MODE = 6                               # ALL harmonics at once — no ladder
ESS_ALPHA = 0.7
MAX_NFEV = 4000
FTOL = 1e-8
if os.environ.get("VMEX_EXAMPLES_CI") == "1":  # smoke-test budget
    MAX_MODE, MAX_NFEV, FTOL = 2, 4, 1e-4
    SURFACES = np.linspace(0.25, 0.75, 3)

# --------------------------- seed input, built from scratch -----------------
rbc = np.zeros((2 * NTOR + 1, MPOL))
zbs = np.zeros((2 * NTOR + 1, MPOL))
rbc[NTOR, 0] = R0
rbc[NTOR, 1] = A_MINOR
zbs[NTOR, 1] = A_MINOR
rbc[NTOR + 1, 1] += 0.01                   # helical kick off the axisymmetric saddle
zbs[NTOR + 1, 1] += 0.01
inp = vj.VmecInput(
    nfp=NFP, mpol=MPOL, ntor=NTOR, rbc=rbc, zbs=zbs, phiedge=PHIEDGE,
    lasym=False, lfreeb=False, mgrid_file="NONE",
    ncurr=1, curtor=0.0, pres_scale=0.0,
    ns_array=[35], ftol_array=[1e-13], niter_array=[1500], delt=0.9,
)
qi = QIResidual(SURFACES)
qp = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)


def iota_shortfall(state, rt):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, rt)), 0.0)


def report(tag, eq):
    total = float(qi.total(eq))
    print(f"[{tag}] QI total = {total:.6e}, "
          f"aspect = {float(opt.aspect_ratio(eq.state, eq.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(eq.state, eq.runtime)):.4f}, "
          f"mirror = {float(opt.mirror_ratio(eq.state, eq.runtime)):.4f}")
    return total


qi_seed = report("seed", opt.solve_equilibrium(inp))

# --------------------------- objective (user-authored) ---------------------
objective_terms = [
    (qi, 0.0, 10.0),                       # the omnigenity target
    (qp, 0.0, 0.3),                        # weak QP guide (replaces the staged QP basin)
    (opt.aspect_ratio, ASPECT_TARGET, 0.25),
    (iota_shortfall, 0.0, 100.0),
    (opt.mirror_ratio, MIRROR_TARGET, 10.0),
]

# --------------------------- ONE least-squares call ------------------------
ndofs = len(opt.boundary_dof_names(inp, MAX_MODE))
print(f"\nsingle stage: max_mode = {MAX_MODE} ({ndofs} boundary dofs), "
      f"ESS alpha = {ESS_ALPHA}")
result = opt.least_squares(
    objective_terms, inp, max_mode=MAX_MODE, jac="implicit",
    use_ess=True, ess_alpha=ESS_ALPHA,
    verbose=1, max_nfev=MAX_NFEV, ftol=FTOL, xtol=1e-10,
)
inp = result.input

# --------------------------- final results ---------------------------------
qi_final = qi_seed
if result.equilibrium is not None:
    qi_final = report("final", result.equilibrium)
print(f"\nQI total: seed {qi_seed:.3e} -> final {qi_final:.3e} "
      f"(one call, no max_mode ladder)")
OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUT_DIR / "input.QI_ess_optimized")
if result.equilibrium is not None:
    wout_path = vj.write_wout(OUT_DIR / "wout_QI_ess_optimized.nc",
                              result.equilibrium.wout)
    print(f"wrote {wout_path}")
