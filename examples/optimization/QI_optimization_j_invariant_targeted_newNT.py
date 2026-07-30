#!/usr/bin/env python
"""QI-only optimization from the bundled ``input.QI_nfp2_initial`` seed.

This variant keeps the J-based QI objective but removes the QP pre-stage.
It runs a pure QI continuation ladder with additional soft targets on:

- aspect ratio = 10.0
- mean iota = -0.61
- mirror ratio = 0.29
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity_j import JInvariantQIResidual
from vmex.core.omnigenity import QIResidual
from vmex.core.omnigenity_j import JInvariantQIAndMaxJResidual

# --------------------------- parameters ------------------------------------
SEED_INPUT = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"
OUT_DIR = Path("output_QI_optimization_j_invariant_newNT")
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
IOTA_FLOOR = 0.15
MIRROR_TARGET = 0.25
QI_WEIGHT = 1.0
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 100.
MIRROR_WEIGHT = 100.0
MAX_MODE_SCHEDULE = (1,2)#, 3, 4, 5, 6)
QI_NFEV = 10 
FTOL = 1e-6
QI_OBJECTIVE = "j_invariant"
MAXJ_PAIRING = "soft_local"  # "all_to_all", "same_alpha", or "soft_local"
MAXJ_SIGMA_ALPHA = 2.0 * np.pi / QI_NALPHA

# --------------------------- seed equilibrium -------------------------------
inp = vj.VmecInput.from_file(SEED_INPUT)
eq = opt.solve_equilibrium(inp)

qi_maxj_0p5 = JInvariantQIAndMaxJResidual(
    SURFACES,
    nphi=QI_NPHI,
    nalpha=QI_NALPHA,
    n_bounce=QI_NBOUNCE,
    mboz=QI_MBOZ,
    nboz=QI_NBOZ,
    qi_weight=1.0,
    maxj_weight=1.0,
    target_maxj=-0.06,
    include_qi=False,
    include_maxj=True,
    p_j=0.5,
    maxj_pairing=MAXJ_PAIRING,
    maxj_sigma_alpha=MAXJ_SIGMA_ALPHA,
)

qi_maxj_1 = JInvariantQIAndMaxJResidual(
    SURFACES,
    nphi=QI_NPHI,
    nalpha=QI_NALPHA,
    n_bounce=QI_NBOUNCE,
    mboz=QI_MBOZ,
    nboz=QI_NBOZ,
    qi_weight=1.0,
    maxj_weight=1.0,
    target_maxj=-0.06,
    include_qi=False,
    include_maxj=True,
    p_j=1.0,
    maxj_pairing=MAXJ_PAIRING,
    maxj_sigma_alpha=MAXJ_SIGMA_ALPHA,
)

qi_maxj_2 = JInvariantQIAndMaxJResidual(
    SURFACES,
    nphi=QI_NPHI,
    nalpha=QI_NALPHA,
    n_bounce=QI_NBOUNCE,
    mboz=QI_MBOZ,
    nboz=QI_NBOZ,
    qi_weight=1.0,
    maxj_weight=1.0,
    target_maxj=-0.06,
    include_qi=False,
    include_maxj=True,
    p_j=2.0,
    maxj_pairing=MAXJ_PAIRING,
    maxj_sigma_alpha=MAXJ_SIGMA_ALPHA,
)

qi = JInvariantQIResidual(
    SURFACES,
    nphi=QI_NPHI,
    nalpha=QI_NALPHA,
    n_bounce=QI_NBOUNCE,
    mboz=QI_MBOZ,
    nboz=QI_NBOZ,
)


qi_original = QIResidual(
    SURFACES,
    nphi=QI_NPHI,
    nalpha=QI_NALPHA,
    n_levels=QI_NBOUNCE,
    mboz=QI_MBOZ,
    nboz=QI_NBOZ,
)


def plot_j_polar_contours(eq, objective, out_dir, *, lambda_samples=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Write polar J_I / J_C contours at fixed lambda with radius = flux surface."""

    out = objective.compute_state(eq.state, eq.runtime)
    alpha = np.asarray(out["alpha"], dtype=float)
    surfaces = np.asarray(out["surfaces"], dtype=float)
    ji = np.asarray(out["ji"], dtype=float)
    jc = np.asarray(out["jc"], dtype=float)
    lambda_grid = np.power(
        np.arange(objective.n_bounce, dtype=float) / max(objective.n_bounce - 1, 1),
        objective.p_lambda,
    )

    theta = np.concatenate([alpha, alpha[:1] + 2.0 * np.pi])
    radius = np.asarray(surfaces, dtype=float)
    theta_grid, radius_grid = np.meshgrid(theta, radius, indexing="xy")
    sample_idx = sorted({
        int(np.clip(round(lam * (objective.n_bounce - 1)), 0, objective.n_bounce - 1))
        for lam in lambda_samples
    })

    for name, data in (("ji", ji), ("jc", jc)):
        for idx in sample_idx:
            values = data[:, :, idx]
            values_periodic = np.concatenate([values, values[:, :1]], axis=1)

            fig = plt.figure(figsize=(12, 5))
            ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
            contour = ax_polar.contourf(theta_grid, radius_grid, values_periodic, levels=32, cmap="viridis")
            lam = lambda_grid[idx]
            ax_polar.set_title(f"{name.upper()} polar contour at lambda={lam:.2f}")
            ax_polar.set_ylim(float(radius.min()), float(radius.max()))
            fig.colorbar(contour, ax=ax_polar, pad=0.12, label=name.upper())

            ax_lines = fig.add_subplot(1, 2, 2)
            for isurf, surface in enumerate(surfaces):
                ax_lines.plot(alpha, data[isurf, :, idx], label=f"s={surface:.2f}")
            ax_lines.set_title(f"{name.upper()} vs alpha across surfaces")
            ax_lines.set_xlabel("alpha")
            ax_lines.set_ylabel(name.upper())
            ax_lines.grid(True, alpha=0.3)
            ax_lines.legend(loc="best", ncol=2, fontsize=8)

            fig.tight_layout()
            path = out_dir / f"{name}_polar_lambda_{idx:02d}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")


def iota_shortfall(state, rt):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, rt)), 0.0)

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
    #(qi_maxj_1, 0.0, 0.1), 
    #(qi_maxj_2, 0.0, 0.1),  
    #(qi_maxj_0p5, 0.0, 0.1),      
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    #(iota_shortfall, 0.0, IOTA_WEIGHT),    
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
plot_j_polar_contours(eq, qi, OUT_DIR)
