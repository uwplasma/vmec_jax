#!/usr/bin/env python
"""QI + maximum-J continuation from the bundled ``input.QI_nfp2_initial`` seed.

Single-stage-type campaign on the nfp=2 simsopt QI seed: a boundary
mode-schedule continuation ladder (``max_mode = 1..N``) drives the traceable
constructed-QI residual (:class:`vmex.core.qi.ConstructedQIResidual` — the
Goodman squash-and-shuffle distance on the in-state Boozer transform)
together with a matched-well maximum-J penalty
(:class:`vmex.core.maxj.MaximumJResidual`) and soft targets on:

- aspect ratio = 10.0
- mean iota = -0.61
- mirror ratio = 0.25

Both bounce-action objectives work at *physical* pitch ``1/B`` (units 1/T).
The pitch grid is derived once from the seed's Boozer ``|B|`` range at fixed
trapping depths, so the same particles are followed through the whole
ladder.  Every stage uses exact implicit gradients (``jac="implicit"``): one
adjoint solve per trust-region step instead of one finite-difference
equilibrium solve per boundary dof.

After the ladder the script reports the bounce-action diagnostics through
:func:`vmex.core.maxj.qi_and_maximum_j_from_boozer` — a *single* Boozer
transform feeding both the J-invariance residual
(:class:`vmex.core.qi.JInvariantQIResidual` semantics) and the maximum-J
residual — and draws polar J(alpha, s) contour maps at each pitch from the
returned bounce actions.

The default settings below are the tiny smoke campaign (max_mode 1, few
trials, coarse ns/Boozer sampling; a few minutes on a laptop CPU) exercised
by the test-suite smoke lane.  The commented values beside them are the
production campaign (max_mode 1..6, ns 24/51, the 141x27x51 QI sampling of
the original workflow); expect a multi-hour CPU run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core.maxj import MaximumJResidual, qi_and_maximum_j_from_boozer
from vmex.core.qi import ConstructedQIResidual, JInvariantQIResidual

SEED_INPUT = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"
OUT_DIR = Path("output_QI_maxJ_continuation")
ASPECT_TARGET = 10.0
IOTA_TARGET = -0.61
MIRROR_TARGET = 0.25
QI_WEIGHT = 1.0
MAXJ_WEIGHT = 1.0
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
MIRROR_WEIGHT = 1.0e2
TRAPPING_DEPTHS = (0.35, 0.55, 0.75)   # 1/pitch = Bmax - depth*(Bmax - Bmin)

# Campaign toggles.
USE_ESS = True               # equilibrium-subspace-saving in the driver
INCLUDE_MAXJ = True          # keep the maximum-J penalty in the ladder terms
INCLUDE_J_INVARIANT = False  # add the J-invariance residual as a ladder term
                             # (NaN-guarded semantics: an invalidated pitch
                             # block fails loudly, not with a plausible zero)
MAKE_PLOTS = True            # write the matplotlib figures

# Sampling/budget knobs: tiny defaults, production values in comments.
SURFACES = np.asarray([0.25, 0.6, 0.9])
# SURFACES = np.asarray([(1 + 5 * k) / 51 for k in range(10)] + [50 / 51])
MBOZ = NBOZ = 6              # full: 18
NPHI = 61                    # full: 141
NALPHA = 9                   # full: 27
N_BOUNCE = 13                # full: 51
ACTION_NALPHA = 7            # full: 27
POINTS_PER_PERIOD = 48       # full: 128
NUM_PERIODS = 3              # full: 4
MAX_WELLS = 8
QUADRATURE_ORDER = 32        # full: 64
MODE_SCHEDULE = (1,)         # full: (1, 2, 3, 4, 5, 6)
MAX_NFEV = 5                 # full: 50
FTOL = 1e-4                  # full: 1e-6
NS_ARRAY = [13]              # full: None (keep the seed deck's 24/51 ladder)


def pitch_from_field_range(bmag, depths=TRAPPING_DEPTHS) -> np.ndarray:
    """Shared physical pitch grid: ``1/pitch`` at fixed trapping depths."""
    bmin = float(np.min(bmag))
    bmax = float(np.max(bmag))
    return np.asarray(
        [1.0 / (bmax - d * (bmax - bmin)) for d in depths], dtype=float)


def plot_j_polar_contours(shared, surfaces, out_dir: Path) -> None:
    """Polar J(alpha, s) contours at each physical pitch.

    ``shared`` is the :func:`qi_and_maximum_j_from_boozer` output.  The mean
    action of its complete, usable wells gives one diagnostic value per field
    line; the objective itself continues to compare every well separately.
    Following Rodríguez, Helander & Goodman (JPP 90, 905900212, 2024), the
    disk coordinates are ``x=s*cos(alpha)``, ``y=s*sin(alpha)``: concentric
    contours indicate alpha-independent J and hence omnigenity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = shared["qi"]
    action = np.asarray(out["action"])          # (nsurf, nalpha, npitch, nwell)
    usable = np.asarray(out["usable_mask"])
    alpha = np.asarray(out["alpha"])
    pitch = np.asarray(out["pitch"])
    count = np.sum(usable, axis=-1)
    j_line = np.where(
        count > 0, np.sum(np.where(usable, action, 0.0), axis=-1) / np.maximum(count, 1), np.nan)

    theta = np.concatenate([alpha, alpha[:1] + 2.0 * np.pi])
    radius = np.asarray(surfaces, dtype=float)
    theta_grid, radius_grid = np.meshgrid(theta, radius, indexing="xy")
    x, y = radius_grid * np.cos(theta_grid), radius_grid * np.sin(theta_grid)

    for ip, pitch_value in enumerate(pitch):
        values = j_line[:, :, ip]
        values_periodic = np.concatenate([values, values[:, :1]], axis=1)

        fig = plt.figure(figsize=(12, 5))
        ax_polar = fig.add_subplot(1, 2, 1)
        contour = ax_polar.contourf(x, y, values_periodic, levels=24, cmap="viridis")
        ax_polar.contour(x, y, values_periodic, levels=24, colors="0.25", linewidths=0.35)
        ax_polar.axhline(0.0, color="white", lw=0.6); ax_polar.axvline(0.0, color="white", lw=0.6)
        ax_polar.set_title(f"J(alpha, s) at 1/pitch = {1.0 / pitch_value:.2f} T")
        ax_polar.set_aspect("equal", adjustable="box"); ax_polar.set_xlabel("s cos(alpha)")
        ax_polar.set_ylabel("s sin(alpha)"); fig.colorbar(contour, ax=ax_polar, pad=0.03, label="J")

        ax_lines = fig.add_subplot(1, 2, 2)
        for isurf, surface in enumerate(radius):
            ax_lines.plot(alpha, values[isurf], label=f"s={surface:.2f}")
        ax_lines.set_title("J vs alpha across surfaces (flat = omnigenous)")
        ax_lines.set_xlabel("alpha")
        ax_lines.set_ylabel("J")
        ax_lines.grid(True, alpha=0.3)
        ax_lines.legend(loc="best", ncol=2, fontsize=8)

        fig.tight_layout()
        path = out_dir / f"j_polar_pitch_{ip:02d}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


# --------------------------- seed equilibrium -------------------------------
inp = vj.VmecInput.from_file(SEED_INPUT)
if NS_ARRAY is not None:  # tiny mode: coarse radial ladder
    inp = dataclasses.replace(
        inp, ns_array=NS_ARRAY, ftol_array=[1e-9], niter_array=[600])
eq = opt.solve_equilibrium(inp)

qi = ConstructedQIResidual(
    SURFACES, mboz=MBOZ, nboz=NBOZ,
    nphi=NPHI, nalpha=NALPHA, n_bounce=N_BOUNCE)

# Shared physical pitch from the seed's Boozer |B| range: the same
# trapped-particle classes are followed through the whole ladder.
probe = qi.compute_state(eq.state, eq.runtime)
pitch = pitch_from_field_range(np.asarray(probe["bmag"]))
print("physical pitch grid (1/T):", np.array2string(pitch, precision=4))

action_options = dict(
    nalpha=ACTION_NALPHA, points_per_period=POINTS_PER_PERIOD,
    num_periods=NUM_PERIODS, max_wells=MAX_WELLS,
    quadrature_order=QUADRATURE_ORDER)
maxj = MaximumJResidual(
    SURFACES, pitch, mboz=MBOZ, nboz=NBOZ, **action_options)
j_invariant = JInvariantQIResidual(
    SURFACES, pitch, mboz=MBOZ, nboz=NBOZ, **action_options)


def report(tag, eq):
    qi_total = float(qi.total(eq))
    print(f"[{tag}] constructed QI = {qi_total:.6e}, "
          f"aspect = {float(opt.aspect_ratio(eq.state, eq.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(eq.state, eq.runtime)):.4f}, "
          f"mirror = {float(opt.mirror_ratio(eq.state, eq.runtime)):.4f}")
    return qi_total


qi_seed = report("seed", eq)

# --------------------------- objective terms --------------------------------
terms = [
    (qi, 0.0, QI_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.mean_iota, IOTA_TARGET, IOTA_WEIGHT),
    (opt.mirror_ratio, MIRROR_TARGET, MIRROR_WEIGHT),
]
if INCLUDE_MAXJ:
    terms.insert(1, (maxj, 0.0, MAXJ_WEIGHT))
if INCLUDE_J_INVARIANT:
    terms.insert(1, (j_invariant, 0.0, QI_WEIGHT))

# --------------------------- continuation ladder ----------------------------
result = None
for max_mode in MODE_SCHEDULE:
    print(f"\n===== QI+maxJ stage, max_mode = {max_mode} =====")
    result = opt.least_squares(
        terms, inp, max_mode=max_mode, jac="implicit",
        use_ess=USE_ESS, verbose=1, max_nfev=MAX_NFEV,
        ftol=FTOL, xtol=1e-10,
    )
    inp = result.input
    if result.equilibrium is not None:
        report(f"stage {max_mode}", result.equilibrium)

# --------------------------- final results ----------------------------------
eq = result.equilibrium or opt.solve_equilibrium(inp)
qi_final = report("final", eq)
print(f"\nQI total: seed {qi_seed:.3e} -> final {qi_final:.3e}")

# One Boozer transform feeds both bounce-action diagnostics.
shared = qi_and_maximum_j_from_boozer(
    eq.state, eq.runtime, surfaces=SURFACES, pitch=pitch,
    mboz=MBOZ, nboz=NBOZ,
    qi_options=action_options, maxj_options=action_options)
print(f"J-invariance total = {float(shared['qi']['total']):.6e} "
      f"(NaN = invalidated pitch block)")
print(f"maximum-J total = {float(shared['maximum_j']['total']):.6e}, "
      f"maximum-J fraction = "
      f"{float(shared['maximum_j']['maximum_j_fraction']):.3f}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUT_DIR / "input.QI_maxJ_continuation_optimized")
wout_path = vj.write_wout(
    OUT_DIR / "wout_QI_maxJ_continuation_optimized.nc", eq.wout)
print(f"wrote {OUT_DIR / 'input.QI_maxJ_continuation_optimized'}\n"
      f"wrote {wout_path}")
if MAKE_PLOTS:
    for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
        print(f"wrote {path}")
    plot_j_polar_contours(shared, SURFACES, OUT_DIR)
