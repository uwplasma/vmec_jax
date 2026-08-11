#!/usr/bin/env python
"""Scan and plot low-mode objective landscapes used in the README."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual


REPO = Path(__file__).resolve().parents[1]
SURFACES = np.linspace(0.1, 1.0, 6)
DATA_PATH = REPO / "benchmarks" / "optimization_landscapes.json"
FIGURE_PATH = REPO / "docs" / "_static" / "figures" / "readme_optimization_landscapes.png"


def make_problem(kind: str):
    """Build one visible tuple objective at a common NS/angular resolution."""
    decks = {
        "QI": "input.nfp2_QI",
        "QA": "input.LandremanPaul2021_QA_lowres",
        "QH": "input.LandremanPaul2021_QH_reactorScale_lowres",
    }
    inp = VmecInput.from_file(REPO / "examples" / "data" / decks[kind])
    inp = replace(inp, ns_array=np.array([31]), ftol_array=np.array([1e-12]), niter_array=np.array([5500]), delt=0.5)
    inp = inp.change_resolution(mpol=5, ntor=5, ntheta=16, nzeta=14)

    if kind == "QI":
        primary = ConstructedQIResidual(SURFACES, mboz=12, nboz=12, nphi=61, nalpha=13, n_bounce=15)

        def iota_floor(state, rt):
            return jnp.maximum(0.33 - jnp.abs(opt.mean_iota(state, rt)), 0)

        def mirror_excess(state, rt):
            return jnp.maximum(opt.mirror_ratio(state, rt) - 0.21, 0)

        def elongation_excess(state, rt):
            return jnp.maximum(opt.max_elongation(state, rt) - 8.0, 0)

        terms = [
            (primary, 0, 10),
            (opt.aspect_ratio, 5.0, 0.005),
            (iota_floor, 0, 10),
            (mirror_excess, 0, 10),
            (elongation_excess, 0, 10),
        ]
    elif kind == "QA":
        primary = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
        terms = [(primary, 0, 1), (opt.mean_iota, 0.41, 10), (opt.aspect_ratio, 6.0, 1)]
    else:
        primary = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
        terms = [(primary, 0, 1), (opt.aspect_ratio, 8.0, 1)]
    return inp, decks[kind], opt.VmecProblem.from_tuples(inp, terms, max_mode=1, use_ess=True)


def scan(points: int, span_percent: float) -> dict:
    """Evaluate converged-equilibrium costs on a serpentine two-mode grid."""
    grid = np.linspace(-span_percent, span_percent, points)
    payload = {"points": points, "span_percent_r00": span_percent, "cases": {}}
    for kind in ("QI", "QA", "QH"):
        print(f"Scanning {kind} ({points} x {points})...")
        inp, deck, problem = make_problem(kind)
        ix, iy = problem.names.index("RBC(1,1)"), problem.names.index("ZBS(1,1)")
        r00 = float(inp.rbc[inp.ntor, 0])
        costs = np.full((points, points), np.nan)
        for row, dy in enumerate(grid):
            columns = range(points) if row % 2 == 0 else range(points - 1, -1, -1)
            for column in columns:
                x = problem.x0.copy()
                x[ix] += 0.01 * grid[column] * r00
                x[iy] += 0.01 * dy * r00
                costs[row, column] = float(problem.fun(x))
                print(f"  {row * points + column + 1:3d}/{points**2}: cost={costs[row, column]:.5e}")
        payload["cases"][kind] = {
            "deck": deck,
            "nfp": inp.nfp,
            "r00": r00,
            "rbc11": float(problem.x0[ix]),
            "zbs11": float(problem.x0[iy]),
            "cost": costs.tolist(),
        }
    return payload


def plot(payload: dict, output: Path) -> None:
    """Render the three landscapes with common geometry and direct labels."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "axes.linewidth": 0.8,
        }
    )
    span = float(payload["span_percent_r00"])
    grid = np.linspace(-span, span, int(payload["points"]))
    labels = {
        "QI": "QI + mirror + elongation + iota + aspect",
        "QA": "QA + iota 0.41 + aspect 6",
        "QH": "QH + aspect 8",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    for ax, kind in zip(axes, ("QI", "QA", "QH")):
        values = np.asarray(payload["cases"][kind]["cost"])
        shown = np.log10(np.maximum(values, np.nanmin(values[values > 0]) * 0.999))
        image = ax.contourf(grid, grid, shown, levels=18, cmap="viridis")
        ax.contour(grid, grid, shown, levels=9, colors="white", linewidths=0.45, alpha=0.55)
        minimum = np.unravel_index(np.nanargmin(values), values.shape)
        ax.plot(
            grid[minimum[1]],
            grid[minimum[0]],
            marker="*",
            ms=9,
            color="#f4b942",
            mec="#171717",
            mew=0.6,
            label="grid minimum",
        )
        ax.plot(0, 0, marker="o", ms=12, mfc="none", mec="white", mew=1.6, label="reference")
        ax.set_title(labels[kind], fontsize=11, pad=10)
        ax.set_xlabel(r"$\Delta$ RBC(1,1) / R00 [%]")
        ax.set_ylabel(r"$\Delta$ ZBS(1,1) / R00 [%]")
        ax.set_aspect("equal")
        cbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.02)
        cbar.set_label(r"$\log_{10}$ least-squares cost", fontsize=9)
        ax.text(
            0.03,
            0.97,
            f"NFP = {payload['cases'][kind]['nfp']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#171717", "ec": "none", "alpha": 0.72},
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    order = [legend_labels.index("reference"), legend_labels.index("grid minimum")]
    axes[0].legend(
        [handles[i] for i in order], [legend_labels[i] for i in order], loc="lower left", fontsize=8, frameon=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument("--span-percent", type=float, default=3.0)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=FIGURE_PATH)
    args = parser.parse_args()
    payload = json.loads(args.data.read_text()) if args.plot_only else scan(args.points, args.span_percent)
    if not args.plot_only:
        args.data.write_text(json.dumps(payload, indent=2) + "\n")
    plot(payload, args.output)
    print(f"Wrote {args.data}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
