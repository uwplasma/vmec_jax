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
    "VMEX": "#1674d1",
    "VMEX legacy": "#8b8a84",
    "VMEC2000": "#333333",
    "VMEC++": "#e49c00",
    "DESC": "#7357b8",
    "VMEX polished": "#1674d1",
}
SOLOVEV_FILES = {
    "VMEX legacy": "strong_certificate_solovev_vmex_m4.json",
    "VMEC2000": "strong_certificate_solovev_vmec2000_m4.json",
    "VMEC++": "strong_certificate_solovev_vmecpp_m4.json",
    "DESC": "strong_certificate_solovev_desc_m4.json",
    "VMEX polished": "strong_polish_solovev_solvax_d3_m5_m4.json",
}
STELLARATOR_FILES = {
    "VMEX": "strong_certificate_nfp2_QA_finite_beta_vmex_m4.json",
    "DESC": "strong_certificate_nfp2_QA_finite_beta_desc_m4.json",
}


def _load(files: dict[str, str]) -> dict[str, dict]:
    artifacts = {name: json.loads((BENCHMARKS / filename).read_text()) for name, filename in files.items()}
    for name, artifact in artifacts.items():
        if artifact["measurement_dirty"]:
            raise RuntimeError(f"{name} artifact was measured from dirty source")
    if "VMEX polished" in artifacts:
        polished = artifacts["VMEX polished"]
        if not polished["polish_report"]["converged"]:
            raise RuntimeError("VMEX polished artifact is not independently certified")
        if polished["final_certificate"]["normalized_l2"] > polished["validation_tolerance"]:
            raise RuntimeError("VMEX polished force exceeds its validation gate")
    return artifacts


def _profile(name: str, artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    profile = (
        artifact["final_certificate"]["radial_profile"]
        if "final_certificate" in artifact
        else artifact["radial_profile"]
    )
    return (
        np.asarray(profile["rho"], dtype=float),
        np.asarray(profile["flux_surface_normalized_l2"], dtype=float),
    )


def _normalized(name: str, artifact: dict) -> float:
    return float(
        artifact["final_certificate"]["normalized_l2"]
        if "final_certificate" in artifact
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


def _timing(artifact: dict) -> tuple[float, float]:
    if "polish_report" in artifact:
        return float(artifact["polish_report"]["solve_seconds"]), float(artifact["total_seconds"])
    external = artifact.get("external_source")
    if external is None:
        return float(artifact["solve_seconds"]), float(artifact["total_seconds"])
    if "timing_seconds" in external:
        return (
            float(external["timing_seconds"]["solve"]),
            float(external["timing_seconds"]["total"]),
        )
    return float(external["solve_seconds"]), float(external["total_seconds"])


def _render_row(
    axes,
    artifacts: dict[str, dict],
    *,
    case_label: str,
    letters: tuple[str, str, str],
    timing_names: tuple[str, str],
) -> None:
    line_styles = {
        "VMEX": "-",
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
    ax.set_ylabel(
        "relative force error\n"
        r"$\epsilon_F=2|\mathbf{J}\!\times\!\mathbf{B}-\nabla p|/"
        r"(|\mathbf{J}\!\times\!\mathbf{B}|+|\nabla p|)$"
    )
    ax.grid(True, which="both", color=GRID, linewidth=0.65)
    ax.legend(loc="best", ncols=2)
    ax.set_title(
        f"({letters[0]}) {case_label}: radial profile",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

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
    ax.set_ylim(min(values) * 0.5, max(values) * 2.5)
    ax.set_xticks(positions)
    labels = {
        "VMEX legacy": "VMEX\nlegacy",
        "VMEC2000": "VMEC\n2000",
        "VMEC++": "VMEC++",
        "VMEX polished": "VMEX\npolished",
    }
    ax.set_xticklabels([labels.get(name, name) for name in names])
    ax.set_ylabel("relative force error, volume L2")
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
    ax.set_title(
        f"({letters[1]}) Volume L2 error",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    timings = [_timing(artifacts[name]) for name in timing_names]
    solve = tuple(item[0] for item in timings)
    total = tuple(item[1] for item in timings)
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
    ax.set_yscale("log")
    ax.set_ylim(max(0.5, min(solve) * 0.5), max(total) * 2.0)
    ax.grid(True, axis="y", which="both", color=GRID, linewidth=0.65)
    for bars_group in (bars_solve, bars_total):
        for bar in bars_group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.08,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.legend(loc="upper left")
    ax.set_title(
        f"({letters[2]}) Cold runtime",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )


def render(
    solovev: dict[str, dict],
    stellarator: dict[str, dict],
    output: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.2, 9.2),
        gridspec_kw={"width_ratios": (1.55, 1.0, 1.0)},
        dpi=180,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.08,
        top=0.91,
        wspace=0.38,
        hspace=0.42,
    )
    fig.suptitle(
        "Force-balance comparison",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    _render_row(
        axes[0],
        solovev,
        case_label="analytical Solov'ev",
        letters=("a", "b", "c"),
        timing_names=("DESC", "VMEX polished"),
    )
    _render_row(
        axes[1],
        stellarator,
        case_label="finite-beta QA stellarator",
        letters=("d", "e", "f"),
        timing_names=("VMEX", "DESC"),
    )
    fig.text(
        0.985,
        0.022,
        "Same Apple host; solve and end-to-end pipeline boundaries are reported separately.",
        ha="right",
        va="bottom",
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
    solovev = _load(SOLOVEV_FILES)
    stellarator = _load(STELLARATOR_FILES)
    render(solovev, stellarator, args.output)
    cases = {
        "solovev_analytical": (SOLOVEV_FILES, solovev),
        "nfp2_QA_finite_beta": (STELLARATOR_FILES, stellarator),
    }
    metadata = {
        "schema": "vmex.strong-force-readme-figure/3",
        "cases": {
            case: {
                "sources": {
                    name: {
                        "path": f"benchmarks/{filename}",
                        "sha256": file_sha256(BENCHMARKS / filename),
                        "normalized_l2": _normalized(name, artifacts[name]),
                    }
                    for name, filename in files.items()
                }
            }
            for case, (files, artifacts) in cases.items()
        },
        "figure": args.output.relative_to(REPO).as_posix(),
        "figure_sha256": file_sha256(args.output),
        "timing_note": (
            "Cold measured pipelines on one Apple host; boundaries differ by "
            "implementation and are not a warm-JIT speed claim."
        ),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(args.output.relative_to(REPO))


if __name__ == "__main__":
    main()
