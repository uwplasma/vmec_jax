#!/usr/bin/env python
"""Quasi-poloidal symmetry (QP) from a circular torus, nfp=2.

Helicity (m, n) = (0, 1): the quasisymmetry ratio residual demands
``|B| = |B|(s, phi)`` — poloidally closed ``|B|`` contours, the symmetry of
configurations like Wistell-B/QPS and the natural precursor of a
quasi-isodynamic field (see ``QI_optimization.py``).  The objective adds the
practical targets the legacy QP runs used: aspect ratio, a *floor* on
``|mean iota|`` (authored inline below — any traceable function of
``(state, runtime)`` can be a term), and a mirror-ratio target that sets the
toroidal ``|B|`` modulation depth.

The default continuation stops at ``max_mode = 3``.  The QP basin caveat
recorded for the legacy runs ("releasing high harmonics early trades QS
against irremovable ripple") was re-confirmed on the office 36-core CPU
(2026-07-11) with this script extended to ``(1, 2, 3, 4, 5)``: the
max_mode-5 run lands at QS(0,1) total 9.415e-02 (aspect 6.02, mean iota
-0.15, mirror 0.20) — the *same* basin as max_mode 1, so stages > 3
neither hurt nor help (each extra stage stops at ``ftol`` within a few
trials and only adds compile time).  QP from a crude circular seed is
basin-sensitive: most of the reduction comes from stages 1-2, and CPU/GPU
rounding alone can land in mirror-image basins (iota of either sign).

The gradient method selects the basin: with JAC="implicit" (below) this
reaches QS(0,1) total 4.458e-01 -> 9.4e-02 already at max_mode=1 (aspect
6.02, |mean iota| 0.15, mirror 0.20), and stays at that ~9.4e-2 ceiling
through max_mode 5; the same objective with finite differences stalls far
worse at 2.3e-01.  QP from a crude circular seed is basin-limited: this is
*not* precise QS (near-axis theory forbids exact QP), and the docstring
above reflects the documented, empirically confirmed ceiling.

Expected runtime is dominated by the one-time implicit-Jacobian XLA
compile per stage (measured max_mode=1 wall ~18 min on an RTX A4000;
the warm forward solve is ~0.9 s).
"""

import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import vmex as vj
from vmex import optimize as opt

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"
OUT_DIR = Path("output_QP_optimization")
QS_SURFACES = np.linspace(0.1, 1.0, 10)
HELICITY_M, HELICITY_N = 0, 1              # QP: |B| = |B|(s, phi)
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.15                          # penalize |mean iota| below this
MIRROR_TARGET = 0.20                       # (Bmax-Bmin)/(Bmax+Bmin) at the edge
MAX_MODE_SCHEDULE = (1, 2, 3)              # stages > 3 stall — see docstring
MAX_NFEV = 2000                            # trial budget per stage
FTOL = 1e-6                                # per-stage convergence tolerance
JAC = "implicit"
if os.environ.get("VMEX_EXAMPLES_CI") == "1":  # smoke-test budget
    MAX_MODE_SCHEDULE, MAX_NFEV, FTOL = (1,), 4, 1e-4

# --------------------------- seed equilibrium -------------------------------
inp = vj.VmecInput.from_file(INPUT_FILE)
eq = opt.solve_equilibrium(inp)
qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, HELICITY_M, HELICITY_N)


def iota_shortfall(state, rt):
    """max(IOTA_FLOOR - |mean iota|, 0) — a user-authored traceable term."""
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, rt)), 0.0)


def report(tag, eq):
    total = float(qs.total(eq))
    aspect = float(opt.aspect_ratio(eq.state, eq.runtime))
    iota = float(opt.mean_iota(eq.state, eq.runtime))
    mirror = float(opt.mirror_ratio(eq.state, eq.runtime))
    print(f"[{tag}] QS total = {total:.6e}, aspect = {aspect:.4f}, "
          f"mean iota = {iota:.4f}, mirror = {mirror:.4f}")
    return total


qs_seed = report("seed", eq)

# --------------------------- objective (user-authored) ----------------------
objective_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_shortfall, 0.0, 100.0),
    (opt.mirror_ratio, MIRROR_TARGET, 10.0),
    # CI-tested extras (see QA_optimization.py for the jac caveats):
    # (opt.magnetic_well, 0.05, 1.0),
    # (opt.mercier_stability_residual, 0.0, 100.0),
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
inp.to_indata(OUT_DIR / "input.QP_optimized")
wout_path = vj.write_wout(OUT_DIR / "wout_QP_optimized.nc", eq.wout)
print(f"wrote {OUT_DIR / 'input.QP_optimized'}\nwrote {wout_path}")
for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
    print(f"wrote {path}")
