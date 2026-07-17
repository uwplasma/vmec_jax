#!/usr/bin/env python
"""Generate the two optimization-docs figures (docs/optimization.rst).

Outputs (both < 100 KB, light-theme PNGs matching the README figures):

- ``docs/_static/figures/ess_x_scale.png`` — the Exponential Spectral
  Scaling trust-region weights ``x_scale = exp(-alpha*max(|m|,|n|)) /
  exp(-alpha)`` per harmonic level, for the example alpha = 0.7 and the
  default alpha = 1.2 (pure formula, no solves).
- ``docs/_static/figures/gradient_stack_speedup.png`` — measured
  before/after of the R25 gradient stack (block-tridiagonal Jacobian,
  perturbation warm start, single-call ESS campaign), numbers from the
  R25/R26k measurements quoted in docs/optimization.rst.

Run from the repo root::

    python docs/_static/figures/sources/make_optimization_docs_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]

BLUE = "#2a78d6"       # categorical slot 1 (after / vmex)
BLUE_LIGHT = "#9ec5f4"  # same hue, light step (before)
AQUA = "#1baf7a"        # categorical slot 2
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3e0"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def ess_figure() -> None:
    levels = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=150)
    for alpha, color, label_y in ((0.7, BLUE, None), (1.2, AQUA, None)):
        scale = np.exp(-alpha * levels) / np.exp(-alpha)
        ax.plot(levels, scale, "-o", color=color, lw=2, ms=5,
                markeredgecolor="white", markeredgewidth=1, zorder=3)
        ax.annotate(rf"$\alpha = {alpha}$", (levels[-1], scale[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    va="center", color=color, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlim(0.7, 7.0)
    ax.set_xticks(levels)
    ax.set_xlabel(r"harmonic level  $\max(|m|,\, |n|)$")
    ax.set_ylabel("trust-region scale (log)")
    ax.set_title("Exponential Spectral Scaling: per-dof trust radius",
                 fontsize=10, loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("level-1 dofs keep the full step", (1.55, 0.93),
                color=INK2, fontsize=8, va="top")
    ax.annotate("max_mode-6 dofs:\n~1/400 of the step\n"
                r"($\alpha = 1.2$)",
                (6, np.exp(-1.2 * 6) / np.exp(-1.2)),
                textcoords="offset points", xytext=(-86, 6),
                color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ess_x_scale.png")
    plt.close(fig)


def stack_figure() -> None:
    panels = [
        {
            "title": "implicit Jacobian, warm\n(nfp2 seed, max_mode 2)",
            "unit": "s",
            "rows": [("per-dof GMRES", 20.35), ("block-tridiagonal", 0.61)],
            "ratio": "33x",
        },
        {
            "title": "forward-solve iterations\n(20 trust-region trials)",
            "unit": "iters",
            "rows": [("plain hot restart", 23685),
                     ("perturbation seed", 6364)],
            "ratio": "3.7x",
        },
        {
            "title": "QA campaign to precise QS\n(36-core CPU)",
            "unit": "min",
            "rows": [("max_mode ladder 1→5", 25.5),
                     ("single-call ESS", 14.5)],
            "ratio": "1.8x",
        },
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.5), dpi=150)
    fig.subplots_adjust(wspace=0.9)
    for ax, panel in zip(axes, panels):
        labels = [r[0] for r in panel["rows"]]
        values = [r[1] for r in panel["rows"]]
        y = [1, 0]  # before on top
        ax.barh(y, values, height=0.62, color=[BLUE_LIGHT, BLUE], zorder=3)
        # before-bar label inside the bar; after-bar label just outside it
        ax.annotate(f"{values[0]:,g} {panel['unit']}", (values[0], 1),
                    textcoords="offset points", xytext=(-4, 0),
                    ha="right", va="center", fontsize=8, color=INK)
        ax.annotate(f"{values[1]:,g} {panel['unit']}", (values[1], 0),
                    textcoords="offset points", xytext=(4, 6),
                    va="center", fontsize=8, color=INK)
        ax.annotate(panel["ratio"] + " less", (values[1], 0),
                    textcoords="offset points", xytext=(4, -7),
                    va="center", fontsize=9, fontweight="bold", color=BLUE)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_title(panel["title"], fontsize=8.5, loc="left", color=INK)
        ax.set_xlim(0, max(values) * 1.06)
        ax.set_ylim(-0.55, 1.55)
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.suptitle("The measured gradient stack (2026-07-12, CPU)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "gradient_stack_speedup.png")
    plt.close(fig)


if __name__ == "__main__":
    ess_figure()
    stack_figure()
    for name in ("ess_x_scale.png", "gradient_stack_speedup.png"):
        print(name, (OUT / name).stat().st_size, "bytes")
