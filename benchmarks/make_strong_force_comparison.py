#!/usr/bin/env python
"""Render the README strong-force comparison from committed clean artifacts.

The optional before/after polishing pair renders from two clean
``benchmarks/strong_certificate.py --wout <export>`` certificates (the
unpolished and the certified polished WOUT exports of the bundled shaped
tokamak) plus the polished wout itself for the cross-sections.
"""

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
    for case_name, case in cases.items():
        for name, artifact in case["sources"].items():
            if artifact["measurement_dirty"]:
                raise RuntimeError(
                    f"{case_name}/{name} was measured from dirty source"
                )
            external = artifact.get("external_source")
            if external is not None and not external["success"]:
                raise RuntimeError(f"{case_name}/{name} external solve failed")
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


def _render_row(
    axes,
    artifacts: dict[str, dict],
    *,
    case_label: str,
    letters: tuple[str, str, str],
) -> None:
    artifacts = {name: artifacts[name] for name in COLORS}
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
        r"(|\mathbf{J}\!\times\!\mathbf{B}|+|\nabla p|+F_{\rm floor})$"
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

def render(
    tokamak: dict[str, dict],
    output: Path,
) -> None:
    # Accuracy only, and only rows where certified polishing demonstrably
    # beats the legacy codes, per the plan's README figure policy. The
    # stellarator row returns when the production polish is tractable in 3D
    # (the captured-constants elimination now in progress); its bundle case
    # stays recorded for that day.
    _style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.9),
        gridspec_kw={"width_ratios": (1.7, 1.0)},
        dpi=180,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.86,
        wspace=0.38,
    )
    fig.suptitle(
        "Force balance across equilibrium solvers",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    _render_row(
        axes,
        tokamak,
        case_label="shaped tokamak, finite pressure",
        letters=("a", "b"),
    )
    fig.text(
        0.985,
        0.022,
        "One shared independent force-balance oracle on each code's exported equilibrium; VMEX is the certified polished result. Case: input.shaped_tokamak_pressure_polished.",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="webp", dpi=180, pil_kwargs={"lossless": True})
    plt.close(fig)


def _load_certificate(path: Path) -> dict:
    cert = json.loads(path.read_text())
    if cert["schema"] != "vmex.strong-certificate-benchmark/1":
        raise RuntimeError("unexpected strong-force certificate schema")
    if cert["measurement_dirty"]:
        raise RuntimeError(f"{path.name} was measured from dirty source")
    return cert


def _cross_sections(wout_path: Path, ax) -> None:
    """Draw phi=0 flux-surface cross-sections of a stellarator-symmetric wout."""
    import netCDF4

    with netCDF4.Dataset(wout_path) as ds:
        xm = np.asarray(ds.variables["xm"][:], dtype=float)
        rmnc = np.asarray(ds.variables["rmnc"][:], dtype=float)
        zmns = np.asarray(ds.variables["zmns"][:], dtype=float)
        ns = int(ds.variables["ns"][:])
    theta = np.linspace(0.0, 2.0 * np.pi, 241)
    cos = np.cos(np.outer(xm, theta))
    sin = np.sin(np.outer(xm, theta))
    # Interior surfaces at uniform rho (uniform s crowds the edge).
    interior = sorted(
        {int(round((ns - 1) * rho * rho)) for rho in np.linspace(0.28, 0.9, 7)}
    )
    for j in interior:
        ax.plot(rmnc[j] @ cos, zmns[j] @ sin, color=EDGE, linewidth=1.0)
    ax.plot(
        rmnc[ns - 1] @ cos,
        zmns[ns - 1] @ sin,
        color=COLORS["VMEX"],
        linewidth=2.2,
        label="prescribed boundary",
    )
    ax.plot(rmnc[0] @ cos[:, :1], zmns[0] @ sin[:, :1], "+", color=INK, markersize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$R$ [m]")
    ax.set_ylabel(r"$Z$ [m]")
    ax.legend(loc="upper right")


def render_summary_pair(
    before_cert: Path,
    after_cert: Path,
    wout: Path,
    output: Path,
) -> tuple[float, float]:
    """Render the polish before/after evidence for the README.

    Left: flux-surface cross-sections of the certified polished export — the
    boundary and profiles are prescribed, so they are identical before and
    after polishing.  Right: the same independent force oracle as the
    solver-comparison figure, applied to the unpolished and the certified
    polished WOUT exports of the one bundled case.  Returns the two volume-L2
    values read from the certificates.
    """
    before = _load_certificate(before_cert)
    after = _load_certificate(after_cert)
    values = (
        float(before["metrics"]["normalized_l2"]),
        float(after["metrics"]["normalized_l2"]),
    )
    _style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.9),
        gridspec_kw={"width_ratios": (1.0, 1.7)},
        dpi=180,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.86, wspace=0.3)
    fig.suptitle(
        "Force balance before and after polishing",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

    _cross_sections(wout, axes[0])
    axes[0].set_title(
        "(a) shaped tokamak: flux surfaces",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[1]
    styles = (
        ("unpolished export", "VMEC2000", (0, (5, 2)), 1.7, before),
        ("certified polished export", "VMEX", "-", 2.5, after),
    )
    for label, color_key, linestyle, width, cert in styles:
        profile = cert["radial_profile"]
        rho = np.asarray(profile["rho"], dtype=float)
        force = np.asarray(profile["flux_surface_normalized_l2"], dtype=float)
        value = float(cert["metrics"]["normalized_l2"])
        ax.plot(
            rho,
            np.maximum(force, 1.0e-30),
            color=COLORS[color_key],
            linestyle=linestyle,
            linewidth=width,
            label=f"{label}  (volume L2 {value:.3g})",
        )
        peak = int(np.argmax(force))
        ax.annotate(
            f"max {force[peak]:.2g}",
            xy=(rho[peak], force[peak]),
            xytext=(12, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=COLORS[color_key],
        )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"normalized radius $\rho=\sqrt{s}$,  $s=\psi/\psi_B$")
    ax.set_ylabel(
        "relative force error\n"
        r"$\epsilon_F=2|\mathbf{J}\!\times\!\mathbf{B}-\nabla p|/"
        r"(|\mathbf{J}\!\times\!\mathbf{B}|+|\nabla p|+F_{\rm floor})$"
    )
    ax.legend(loc="upper right")
    ax.set_title(
        "(b) independent force error, exported equilibria",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.022,
        "One shared independent force-balance oracle on the unpolished and certified polished WOUT exports; "
        "boundary and profiles are identical. Case: input.shaped_tokamak_pressure_polished.",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="webp", dpi=180, pil_kwargs={"lossless": True})
    plt.close(fig)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--summary-before-cert", type=Path)
    parser.add_argument("--summary-after-cert", type=Path)
    parser.add_argument("--summary-wout", type=Path)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_FIGURE)
    args = parser.parse_args()
    summary_inputs = (
        args.summary_before_cert, args.summary_after_cert, args.summary_wout,
    )
    if any(path is None for path in summary_inputs) and any(
        path is not None for path in summary_inputs
    ):
        parser.error(
            "pass --summary-before-cert, --summary-after-cert, and "
            "--summary-wout together (benchmarks/strong_certificate.py "
            "--wout <export> writes the certificates)"
        )
    cases = _load(args.artifact)
    render(
        cases["shaped_tokamak_pressure"]["sources"],
        args.output,
    )
    summary_values = None
    if args.summary_before_cert is not None:
        summary_values = render_summary_pair(
            args.summary_before_cert,
            args.summary_after_cert,
            args.summary_wout,
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
            if summary_values is None
            else args.summary_output.relative_to(REPO).as_posix()
        ),
        "summary_figure_sha256": (
            None
            if summary_values is None
            else file_sha256(args.summary_output)
        ),
        "summary_independent_l2": (
            None
            if summary_values is None
            else {
                "before": round(summary_values[0], 6),
                "after": round(summary_values[1], 6),
            }
        ),
        "timing_note": (
            "Accuracy only, and only rows where certified polishing "
            "demonstrably beats the legacy codes, per the plan's README "
            "figure policy; runtime evidence lives in benchmarks/baselines. "
            "All errors from one shared independent oracle on each code's "
            "exported equilibrium (VMEX: the certified polished state). The "
            "stellarator row returns when the 3-D production polish is "
            "tractable."
        ),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(figure_path)


if __name__ == "__main__":
    main()
