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

Run modes (the default finishes in a few minutes on a laptop CPU):

- default        tiny ladder (max_mode 1, few trials, coarse ns/Boozer) —
                 exercised by the test-suite smoke lane
- ``--full``     the production campaign (max_mode 1..6, ns 24/51, the
                 141x27x51 QI sampling of the original workflow); expect a
                 multi-hour CPU run
- ``--no-ess``   disable equilibrium-subspace-saving in the driver
- ``--no-maxj``  drop the maximum-J penalty from the ladder terms
- ``--j-invariant``  add the J-invariance residual as a ladder term as well
                 (NaN-guarded semantics: an invalidated pitch block fails
                 loudly instead of returning a plausible zero)
- ``--no-plot``  skip the matplotlib figures

``VMEX_EXAMPLES_CI=1`` forces the tiny budget regardless of flags.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core.maxj import MaximumJResidual, qi_and_maximum_j_from_boozer
from vmex.core.qi import ConstructedQIResidual, JInvariantQIResidual

SEED_INPUT = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"
ASPECT_TARGET = 10.0
IOTA_TARGET = -0.61
MIRROR_TARGET = 0.25
QI_WEIGHT = 1.0
MAXJ_WEIGHT = 1.0
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
MIRROR_WEIGHT = 1.0e2
TRAPPING_DEPTHS = (0.35, 0.55, 0.75)   # 1/pitch = Bmax - depth*(Bmax - Bmin)


def build_settings(full: bool) -> dict:
    """Sampling/budget knobs for the tiny (default) and ``--full`` campaigns."""
    if full:
        return dict(
            surfaces=np.asarray([(1 + 5 * k) / 51 for k in range(10)] + [50 / 51]),
            mboz=18, nboz=18, nphi=141, nalpha=27, n_bounce=51,
            action_nalpha=27, points_per_period=128, num_periods=4,
            max_wells=8, quadrature_order=64,
            mode_schedule=(1, 2, 3, 4, 5, 6), max_nfev=50, ftol=1e-6,
            ns_array=None,             # keep the seed deck's 24/51 ladder
        )
    return dict(
        surfaces=np.asarray([0.25, 0.6, 0.9]),
        mboz=6, nboz=6, nphi=61, nalpha=9, n_bounce=13,
        action_nalpha=7, points_per_period=48, num_periods=3,
        max_wells=8, quadrature_order=32,
        mode_schedule=(1,), max_nfev=5, ftol=1e-4,
        ns_array=[13],
    )


def pitch_from_field_range(bmag, depths=TRAPPING_DEPTHS) -> np.ndarray:
    """Shared physical pitch grid: ``1/pitch`` at fixed trapping depths."""
    bmin = float(np.min(bmag))
    bmax = float(np.max(bmag))
    return np.asarray(
        [1.0 / (bmax - d * (bmax - bmin)) for d in depths], dtype=float)


def plot_j_polar_contours(shared, surfaces, out_dir: Path) -> None:
    """Polar J(alpha, s) contours at each pitch from the bounce actions.

    ``shared`` is the :func:`qi_and_maximum_j_from_boozer` output: the total
    usable bounce action per field line, ``J(s, alpha, pitch)``, is the
    quantity whose alpha-independence the J-invariance residual enforces.
    Radius is the flux label ``s``, angle the field-line label ``alpha``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = shared["qi"]
    action = np.asarray(out["action"])          # (nsurf, nalpha, npitch, nwell)
    usable = np.asarray(out["usable_mask"])
    alpha = np.asarray(out["alpha"])
    pitch = np.asarray(out["pitch"])
    j_line = np.sum(np.where(usable, action, 0.0), axis=-1)  # (nsurf, nalpha, npitch)

    theta = np.concatenate([alpha, alpha[:1] + 2.0 * np.pi])
    radius = np.asarray(surfaces, dtype=float)
    theta_grid, radius_grid = np.meshgrid(theta, radius, indexing="xy")

    for ip, pitch_value in enumerate(pitch):
        values = j_line[:, :, ip]
        values_periodic = np.concatenate([values, values[:, :1]], axis=1)

        fig = plt.figure(figsize=(12, 5))
        ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
        contour = ax_polar.contourf(
            theta_grid, radius_grid, values_periodic, levels=32, cmap="viridis")
        ax_polar.set_title(f"J(alpha, s) at 1/pitch = {1.0 / pitch_value:.2f} T")
        ax_polar.set_ylim(0.0, float(radius.max()))
        fig.colorbar(contour, ax=ax_polar, pad=0.12, label="J")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--full", action="store_true",
                        help="production budget (max_mode 1..6; multi-hour CPU)")
    parser.add_argument("--no-ess", action="store_true",
                        help="disable equilibrium-subspace-saving")
    parser.add_argument("--no-maxj", action="store_true",
                        help="drop the maximum-J penalty from the ladder")
    parser.add_argument("--j-invariant", action="store_true",
                        help="add the J-invariance residual as a ladder term")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip matplotlib figures")
    parser.add_argument("--out", type=Path,
                        default=Path("output_QI_maxJ_continuation"))
    args = parser.parse_args()

    full = args.full and os.environ.get("VMEX_EXAMPLES_CI") != "1"
    cfg = build_settings(full)
    out_dir = args.out

    # --------------------------- seed equilibrium ---------------------------
    inp = vj.VmecInput.from_file(SEED_INPUT)
    if cfg["ns_array"] is not None:  # tiny mode: coarse radial ladder
        inp = dataclasses.replace(
            inp, ns_array=cfg["ns_array"], ftol_array=[1e-9],
            niter_array=[600])
    eq = opt.solve_equilibrium(inp)

    qi = ConstructedQIResidual(
        cfg["surfaces"], mboz=cfg["mboz"], nboz=cfg["nboz"],
        nphi=cfg["nphi"], nalpha=cfg["nalpha"], n_bounce=cfg["n_bounce"])

    # Shared physical pitch from the seed's Boozer |B| range: the same
    # trapped-particle classes are followed through the whole ladder.
    probe = qi.compute_state(eq.state, eq.runtime)
    pitch = pitch_from_field_range(np.asarray(probe["bmag"]))
    print("physical pitch grid (1/T):", np.array2string(pitch, precision=4))

    action_options = dict(
        nalpha=cfg["action_nalpha"], points_per_period=cfg["points_per_period"],
        num_periods=cfg["num_periods"], max_wells=cfg["max_wells"],
        quadrature_order=cfg["quadrature_order"])
    maxj = MaximumJResidual(
        cfg["surfaces"], pitch, mboz=cfg["mboz"], nboz=cfg["nboz"],
        **action_options)
    j_invariant = JInvariantQIResidual(
        cfg["surfaces"], pitch, mboz=cfg["mboz"], nboz=cfg["nboz"],
        **action_options)

    def report(tag, eq):
        qi_total = float(qi.total(eq))
        print(f"[{tag}] constructed QI = {qi_total:.6e}, "
              f"aspect = {float(opt.aspect_ratio(eq.state, eq.runtime)):.4f}, "
              f"mean iota = {float(opt.mean_iota(eq.state, eq.runtime)):.4f}, "
              f"mirror = {float(opt.mirror_ratio(eq.state, eq.runtime)):.4f}")
        return qi_total

    qi_seed = report("seed", eq)

    # --------------------------- objective terms ----------------------------
    terms = [
        (qi, 0.0, QI_WEIGHT),
        (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
        (opt.mean_iota, IOTA_TARGET, IOTA_WEIGHT),
        (opt.mirror_ratio, MIRROR_TARGET, MIRROR_WEIGHT),
    ]
    if not args.no_maxj:
        terms.insert(1, (maxj, 0.0, MAXJ_WEIGHT))
    if args.j_invariant:
        terms.insert(1, (j_invariant, 0.0, QI_WEIGHT))

    # --------------------------- continuation ladder ------------------------
    result = None
    for max_mode in cfg["mode_schedule"]:
        print(f"\n===== QI+maxJ stage, max_mode = {max_mode} =====")
        result = opt.least_squares(
            terms, inp, max_mode=max_mode, jac="implicit",
            use_ess=not args.no_ess, verbose=1, max_nfev=cfg["max_nfev"],
            ftol=cfg["ftol"], xtol=1e-10,
        )
        inp = result.input
        if result.equilibrium is not None:
            report(f"stage {max_mode}", result.equilibrium)

    # --------------------------- final results ------------------------------
    eq = result.equilibrium or opt.solve_equilibrium(inp)
    qi_final = report("final", eq)
    print(f"\nQI total: seed {qi_seed:.3e} -> final {qi_final:.3e}")

    # One Boozer transform feeds both bounce-action diagnostics.
    shared = qi_and_maximum_j_from_boozer(
        eq.state, eq.runtime, surfaces=cfg["surfaces"], pitch=pitch,
        mboz=cfg["mboz"], nboz=cfg["nboz"],
        qi_options=action_options, maxj_options=action_options)
    print(f"J-invariance total = {float(shared['qi']['total']):.6e} "
          f"(NaN = invalidated pitch block)")
    print(f"maximum-J total = {float(shared['maximum_j']['total']):.6e}, "
          f"maximum-J fraction = "
          f"{float(shared['maximum_j']['maximum_j_fraction']):.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    inp.to_indata(out_dir / "input.QI_maxJ_continuation_optimized")
    wout_path = vj.write_wout(
        out_dir / "wout_QI_maxJ_continuation_optimized.nc", eq.wout)
    print(f"wrote {out_dir / 'input.QI_maxJ_continuation_optimized'}\n"
          f"wrote {wout_path}")
    if not args.no_plot:
        for key, path in vj.plot_wout(wout_path, out_dir).items():
            print(f"wrote {path}")
        plot_j_polar_contours(shared, cfg["surfaces"], out_dir)


if __name__ == "__main__":
    main()
