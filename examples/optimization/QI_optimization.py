#!/usr/bin/env python
"""QP basin selection followed by constructed-QI optimization, nfp=2.

The driver keeps the numerical choices visible: VMEC resolution is changed in
the script, objective tuples are ordinary Python, and SciPy receives VMEX's
residual and exact implicit Jacobian directly.  A short QP stage selects the
poloidally closed-|B| basin; the fuller Goodman squash-and-shuffle residual
then refines mode 5 in three short trust-region stages.  The final equilibrium
is hot-started at NS=101.
"""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual


DATA = Path(__file__).resolve().parents[1] / "data" / "input.nfp2_QI_seed"
OUT_DIR = Path("output_QI_optimization")
SURFACES = np.linspace(0.1, 1.0, 6)
ASPECT_TARGET = 5.0
IOTA_FLOOR = 0.33
MIRROR_LIMIT = 0.21
ELONGATION_LIMIT = 8.0
MINIMUM_MPOL = 5
QI_MODES = [5, 5, 5]
QI_BUDGETS = [30, 20, 20]
QP_BUDGET = 25
FULL_QI_BUDGET = 10

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QI_MODES, QI_BUDGETS, QP_BUDGET, FULL_QI_BUDGET = [1], [3], 3, 2

inp = VmecInput.from_file(DATA)
qp = opt.QuasisymmetryRatioResidual(
    SURFACES, helicity_m=0, helicity_n=1
)
qi_options = dict(mboz=12, nboz=12, nphi=61, nalpha=13, n_bounce=15)
if ci_smoke:
    qi_options = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
qi = ConstructedQIResidual(SURFACES, **qi_options)
qi_report = ConstructedQIResidual(SURFACES)


def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, runtime)), 0.0)


def mirror_excess(state, runtime):
    return jnp.maximum(opt.mirror_ratio(state, runtime) - MIRROR_LIMIT, 0.0)


def elongation_excess(state, runtime):
    return jnp.maximum(
        opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0.0
    )


def report(label, equilibrium):
    total = float(qi.total(equilibrium))
    print(
        f"[{label}] constructed QI = {total:.6e}, "
        f"aspect = {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.4f}, "
        f"mean iota = {float(opt.mean_iota(equilibrium.state, equilibrium.runtime)):.4f}, "
        f"mirror = {float(opt.mirror_ratio(equilibrium.state, equilibrium.runtime)):.4f}, "
        f"elongation = {float(opt.max_elongation(equilibrium.state, equilibrium.runtime)):.4f}"
    )
    return total


practical_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0),
    (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0),
]
qp_terms = [(qp, 0.0, 10.0), *practical_terms]
qi_terms = [(qi, 0.0, 10.0), *practical_terms]

print("\n===== QP basin stage, max_mode = 1 =====")
mpol = max(1 + 2, MINIMUM_MPOL)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol,
    ntor=mpol,
    ntheta=2 * mpol + 6,
    nzeta=2 * mpol + 4,
)
problem = opt.VmecProblem.from_tuples(
    inp, qp_terms, max_mode=1, use_ess=True, progress=not ci_smoke
)
seed_eq = problem.equilibrium_from_x(problem.x0)
qi_seed = report("seed", seed_eq)
result = least_squares(
    problem.residual,
    problem.x0,
    jac=problem.residual_jac,
    x_scale=problem.scales,
    verbose=2,
    max_nfev=QP_BUDGET,
    ftol=1.0e-6,
    xtol=1.0e-10,
)
inp = problem.input_from_x(result.x)
equilibrium = problem.equilibrium_from_x(result.x)
report("QP basin", equilibrium)

for stage, (max_mode, max_nfev) in enumerate(zip(QI_MODES, QI_BUDGETS), 1):
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol,
        ntor=mpol,
        ntheta=2 * mpol + 6,
        nzeta=2 * mpol + 4,
    )
    # Restart SciPy's trust-region model; equal-shape JAX executables are reused.
    problem = opt.VmecProblem.from_tuples(
        inp, qi_terms, max_mode=max_mode, use_ess=True, progress=not ci_smoke,
    )
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    result = least_squares(
        problem.residual,
        problem.x0,
        jac=problem.residual_jac,
        x_scale=problem.scales,
        verbose=2,
        max_nfev=max_nfev,
        ftol=1.0e-6,
        xtol=1.0e-10,
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"QI mode {max_mode}", equilibrium)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inp.to_indata(OUT_DIR / f"input.QI_max_mode_{max_mode:03d}_stage_{stage:02d}")

print(f"\n===== Full constructed-QI polish, max_mode = {QI_MODES[-1]} =====")
problem = opt.VmecProblem.from_tuples(
    inp, [(qi_report, 0.0, 10.0), *practical_terms], max_mode=QI_MODES[-1],
    use_ess=True, progress=not ci_smoke,
)
if not ci_smoke:
    problem.compile_residual_and_jacobian()
result = least_squares(
    problem.residual, problem.x0, jac=problem.residual_jac,
    x_scale=problem.scales, verbose=2, max_nfev=FULL_QI_BUDGET,
    ftol=1.0e-6, xtol=1.0e-10,
)
inp = problem.input_from_x(result.x)
equilibrium = problem.equilibrium_from_x(result.x)
resolved_qi = float(qi_report.total(equilibrium))
print(f"Full-resolution constructed QI = {resolved_qi:.6e}")

final_input = replace(
    inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]),
)
final_equilibrium = opt.solve_equilibrium(
    final_input,
    initial_state=equilibrium.state,
    verbose=not ci_smoke,
    raise_on_max_iterations=True,
)
qi_final = report("final", final_equilibrium)
print(
    "Final full-resolution constructed QI = "
    f"{float(qi_report.total(final_equilibrium)):.6e}"
)
print(f"\nQI total: seed {qi_seed:.3e} -> final {qi_final:.3e}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
input_path = final_input.to_indata(OUT_DIR / "input.QI_optimized")
wout_path = vj.write_wout(
    OUT_DIR / "wout_QI_optimized.nc", final_equilibrium.wout
)
print(f"wrote {input_path}")
print(f"wrote {wout_path}")
for path in vj.plot_wout(wout_path, OUT_DIR).values():
    print(f"wrote {path}")
