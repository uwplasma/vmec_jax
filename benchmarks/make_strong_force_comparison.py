#!/usr/bin/env python
"""Render the README strong-force comparison from committed clean artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _provenance import file_sha256

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO / "benchmarks"
DEFAULT_FIGURE = REPO / "docs" / "_static" / "figures" / "readme_strong_force_comparison.webp"
DEFAULT_METADATA = BENCHMARKS / "strong_force_comparison_m4.json"

SURFACE = "#fcfcfb"
INK = "#161616"
MUTED = "#66645f"
GRID = "#dddcd6"
COLORS = {
    "VMEX legacy": "#8b8a84",
    "VMEC2000": "#333333",
    "VMEC++": "#e49c00",
    "DESC": "#7357b8",
    "VMEX polished": "#1674d1",
}
FILES = {
    "VMEX legacy": "strong_certificate_solovev_vmex_m4.json",
    "VMEC2000": "strong_certificate_solovev_vmec2000_m4.json",
    "VMEC++": "strong_certificate_solovev_vmecpp_m4.json",
    "DESC": "strong_certificate_solovev_desc_m4.json",
    "VMEX polished": "strong_polish_solovev_solvax_d3_m5_m4.json",
}


def _load() -> dict[str, dict]:
    artifacts = {
        name: json.loads((BENCHMARKS / filename).read_text())
        for name, filename in FILES.items()
    }
    for name, artifact in artifacts.items():
        if artifact["measurement_dirty"]:
            raise RuntimeError(f"{name} artifact was measured from dirty source")
    polished = artifacts["VMEX polished"]
    if not polished["polish_report"]["converged"]:
        raise RuntimeError("VMEX polished artifact is not independently certified")
    if polished["final_certificate"]["normalized_l2"] > polished[
        "validation_tolerance"
    ]:
        raise RuntimeError("VMEX polished force exceeds its frozen validation gate")
    return artifacts


def _profile(name: str, artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    profile = (
        artifact["final_certificate"]["radial_profile"]
        if name == "VMEX polished"
        else artifact["radial_profile"]
    )
    return (
        np.asarray(profile["rho"], dtype=float),
        np.asarray(profile["flux_surface_average_force_density"], dtype=float),
    )


def _normalized(name: str, artifact: dict) -> float:
    return float(
        artifact["final_certificate"]["normalized_l2"]
        if name == "VMEX polished"
        else artifact["metrics"]["normalized_l2"]
    )


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
        }
    )


def render(artifacts: dict[str, dict], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 5.25),
        gridspec_kw={"width_ratios": (1.55, 1.0, 1.0)},
        dpi=180,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.28, top=0.76, wspace=0.34)
    fig.suptitle(
        "Solov'ev force balance — one independent oracle, clean source states",
        x=0.065,
        y=0.955,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.895,
        "Lower is better. VMEX polishing clears the DESC accuracy gate; the measured cold pipeline is not yet faster.",
        ha="left",
        color=MUTED,
        fontsize=10,
    )

    line_styles = {
        "VMEX legacy": (0, (2, 2)),
        "VMEC2000": (0, (5, 2)),
        "VMEC++": (0, (1, 1)),
        "DESC": "-.",
        "VMEX polished": "-",
    }
    ax = axes[0]
    for name, artifact in artifacts.items():
        rho, force = _profile(name, artifact)
        ax.plot(
            rho,
            np.maximum(force, 1.0e-30),
            color=COLORS[name],
            linestyle=line_styles[name],
            linewidth=2.5 if name == "VMEX polished" else 1.7,
            alpha=1.0 if name in ("DESC", "VMEX polished") else 0.88,
            label=name,
        )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"normalized radius $\rho$")
    ax.set_ylabel(r"$\langle|\mathbf{J}\!\times\!\mathbf{B}-\nabla p|\rangle$  [N m$^{-3}$]")
    ax.grid(True, which="both", color=GRID, linewidth=0.65)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, -0.30), ncols=3)
    ax.set_title("(a) Radial force profile", loc="left", fontsize=11, fontweight="bold")

    ax = axes[1]
    names = list(artifacts)
    values = [_normalized(name, artifacts[name]) for name in names]
    positions = np.arange(len(names))
    bars = ax.bar(
        positions,
        values,
        color=[COLORS[name] for name in names],
        width=0.72,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_ylim(1.0e-3, 3.2e-1)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["VMEX\nlegacy", "VMEC\n2000", "VMEC++", "DESC", "VMEX\npolished"]
    )
    ax.set_ylabel("normalized L2 force")
    ax.grid(True, axis="y", which="both", color=GRID, linewidth=0.65)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.10,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ratio = values[names.index("DESC")] / values[names.index("VMEX polished")]
    ax.text(
        3.5,
        3.0e-2,
        f"DESC / polished = {ratio:.1f}×",
        ha="center",
        va="center",
        color=COLORS["VMEX polished"],
        fontsize=9,
        fontweight="bold",
    )
    ax.set_title("(b) Common force certificate", loc="left", fontsize=11, fontweight="bold")

    desc = artifacts["DESC"]
    polished = artifacts["VMEX polished"]
    timing_names = ("DESC", "VMEX polished")
    solve = (
        float(desc["external_source"]["solve_seconds"]),
        float(polished["polish_report"]["solve_seconds"]),
    )
    total = (
        float(desc["external_source"]["total_seconds"]),
        float(polished["total_seconds"]),
    )
    memory = (
        float(desc["external_source"]["peak_rss_mib"]) / 1024.0,
        float(polished["total_peak_rss_increase_mib"]) / 1024.0,
    )
    ax = axes[2]
    x = np.arange(2)
    width = 0.34
    bars_solve = ax.bar(
        x - width / 2,
        solve,
        width,
        color=[COLORS[name] for name in timing_names],
        alpha=0.55,
        label="reported solve",
    )
    bars_total = ax.bar(
        x + width / 2,
        total,
        width,
        color=[COLORS[name] for name in timing_names],
        label="end to end",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(timing_names)
    ax.set_ylabel("wall time [s]")
    ax.set_ylim(0.0, max(total) * 1.34)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65)
    for bars_group in (bars_solve, bars_total):
        for bar in bars_group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.9,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.legend(loc="upper left")
    ax.text(
        0.5,
        0.73,
        f"peak RSS\n{memory[0]:.2f} / {memory[1]:.2f} GiB",
        transform=ax.transAxes,
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    ax.set_title("(c) Measured cold pipeline", loc="left", fontsize=11, fontweight="bold")
    ax.text(
        0.0,
        -0.35,
        "Same Apple host; disclosed pipeline boundaries.\nNot a warm-JIT speed claim.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="webp", dpi=180, pil_kwargs={"lossless": True})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    artifacts = _load()
    render(artifacts, args.output)
    metadata = {
        "schema": "vmex.strong-force-readme-figure/1",
        "case": "solovev_analytical",
        "figure": args.output.relative_to(REPO).as_posix(),
        "figure_sha256": file_sha256(args.output),
        "sources": {
            name: {
                "path": f"benchmarks/{filename}",
                "sha256": file_sha256(BENCHMARKS / filename),
                "normalized_l2": _normalized(name, artifacts[name]),
            }
            for name, filename in FILES.items()
        },
        "timing_note": (
            "Cold measured pipelines on one Apple host; boundaries differ by "
            "implementation and are not a warm-JIT speed claim."
        ),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(args.output.relative_to(REPO))


if __name__ == "__main__":
    main()
