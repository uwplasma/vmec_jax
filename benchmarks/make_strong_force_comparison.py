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
DEFAULT_ARTIFACT = BENCHMARKS / "strong_force_cases_m4.json"
DEFAULT_SUMMARY_FIGURE = (
    REPO / "docs" / "_static" / "figures" / "readme_polish_summary.webp"
)

SURFACE = "#fcfcfb"
INK = "#161616"
MUTED = "#66645f"
EDGE = "#d5d3ce"
COLORS = {
    "VMEX": "#1674d1",
    "VMEC2000": "#333333",
    "VMEC++": "#e49c00",
    "DESC": "#7357b8",
}


def _load(path: Path) -> dict[str, dict[str, dict]]:
    bundle = json.loads(path.read_text())
    if bundle["schema"] != "vmex.strong-force-comparison-cases/1":
        raise RuntimeError("unexpected strong-force comparison artifact schema")
    cases = bundle["cases"]
    artifacts = {
        name: artifact
        for case in cases.values()
        for name, artifact in case["sources"].items()
    }
    for name, artifact in artifacts.items():
        if artifact["measurement_dirty"]:
            raise RuntimeError(f"{name} artifact was measured from dirty source")
        external = artifact.get("external_source")
        if external is not None and not external["success"]:
            raise RuntimeError(f"{name} external solve did not succeed")
    for case in cases.values():
        polished = case["sources"].get("VMEX")
        if polished is None or "polish_report" not in polished:
            continue
        if not polished["polish_report"]["converged"]:
            raise RuntimeError("VMEX artifact is not independently certified")
        if polished["final_certificate"]["normalized_l2"] > polished["validation_tolerance"]:
            raise RuntimeError("VMEX polished force exceeds its validation gate")
    return cases


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
            "axes.edgecolor": EDGE,
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


def _timing(artifact: dict) -> float:
    external = artifact.get("external_source")
    if external is None:
        return float(artifact["solve_seconds"])
    if "timing_seconds" in external:
        return float(external["timing_seconds"]["total"])
    return float(external["total_seconds"])


def _render_row(
    axes,
    artifacts: dict[str, dict],
    *,
    case_label: str,
    letters: tuple[str, str, str],
    timing_names: tuple[str, ...],
) -> None:
    line_styles = {
        "VMEX": "-",
        "VMEC2000": (0, (5, 2)),
        "VMEC++": (0, (1, 1)),
        "DESC": "-.",
    }
    ax = axes[0]
    for name, artifact in artifacts.items():
        rho, force = _profile(name, artifact)
        ax.plot(
            rho,
            np.maximum(force, 1.0e-30),
            color=COLORS[name],
            linestyle=line_styles[name],
            linewidth=2.5 if name == "VMEX" else 1.7,
            alpha=1.0 if name in ("VMEX", "DESC") else 0.88,
            label=name,
        )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(
        r"normalized radius $\rho=\sqrt{s}$,  $s=\psi/\psi_B$"
    )
    ax.set_ylabel(
        "relative force error\n"
        r"$\epsilon_F=2|\mathbf{J}\!\times\!\mathbf{B}-\nabla p|/"
        r"(|\mathbf{J}\!\times\!\mathbf{B}|+|\nabla p|)$"
    )
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
        "VMEC2000": "VMEC\n2000",
        "VMEC++": "VMEC++",
    }
    ax.set_xticklabels([labels.get(name, name) for name in names])
    ax.set_ylabel("relative force error, volume L2")
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
    ax = axes[2]
    x = np.arange(len(timing_names))
    bars = ax.bar(
        x,
        timings,
        0.72,
        color=[COLORS[name] for name in timing_names],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(name, name) for name in timing_names])
    ax.set_ylabel("cold wall time [s]")
    ax.set_yscale("log")
    ax.set_ylim(max(0.2, min(timings) * 0.5), max(timings) * 2.0)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.08,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(
        f"({letters[2]}) Cold runtime",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )


def render(
    tokamak: dict[str, dict],
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
        "Force balance across equilibrium solvers",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    _render_row(
        axes[0],
        tokamak,
        case_label="finite-pressure tokamak",
        letters=("a", "b", "c"),
        timing_names=("VMEX", "VMEC2000", "VMEC++", "DESC"),
    )
    _render_row(
        axes[1],
        stellarator,
        case_label="finite-beta QA stellarator",
        letters=("d", "e", "f"),
        timing_names=("VMEX", "VMEC2000", "VMEC++", "DESC"),
    )
    fig.text(
        0.985,
        0.022,
        "Cold CPU runs on the same Apple host; each bar includes the code's load, solve, and export path.",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="webp", dpi=180, pil_kwargs={"lossless": True})
    plt.close(fig)


def render_summary_pair(before: Path, after: Path, output: Path) -> None:
    """Place the standard ``--plot`` summaries side by side without restyling."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.9), dpi=170)
    for ax, path, title in zip(
        axes,
        (before, after),
        ("Before polishing", "After polishing"),
        strict=True,
    ):
        ax.imshow(plt.imread(path))
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.93, wspace=0.015)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="webp", dpi=170, pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--before-summary", type=Path)
    parser.add_argument("--after-summary", type=Path)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_FIGURE)
    args = parser.parse_args()
    if (args.before_summary is None) != (args.after_summary is None):
        parser.error("pass both --before-summary and --after-summary")
    cases = _load(args.artifact)
    render(
        cases["shaped_tokamak_pressure"]["sources"],
        cases["nfp2_QA_finite_beta"]["sources"],
        args.output,
    )
    if args.before_summary is not None:
        render_summary_pair(
            args.before_summary,
            args.after_summary,
            args.summary_output,
        )
    try:
        figure_path = args.output.relative_to(REPO).as_posix()
    except ValueError:
        figure_path = str(args.output)
    metadata = {
        "schema": "vmex.strong-force-readme-figure/4",
        "cases": {
            case: {
                "sources": {
                    name: {
                        "path": f"benchmarks/{args.artifact.name}",
                        "sha256": file_sha256(args.artifact),
                        "normalized_l2": _normalized(name, artifact),
                    }
                    for name, artifact in contents["sources"].items()
                }
            }
            for case, contents in cases.items()
        },
        "figure": figure_path,
        "figure_sha256": file_sha256(args.output),
        "summary_figure": (
            None
            if args.before_summary is None
            else args.summary_output.relative_to(REPO).as_posix()
        ),
        "summary_figure_sha256": (
            None
            if args.before_summary is None
            else file_sha256(args.summary_output)
        ),
        "timing_note": (
            "Cold load, solve, and export paths on one Apple host; no warm-JIT claim."
        ),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(figure_path)


if __name__ == "__main__":
    main()
