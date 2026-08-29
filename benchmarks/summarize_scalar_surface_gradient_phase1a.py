#!/usr/bin/env python3
"""Consolidate the run-71 Phase-1A diagnostic artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TOLERANCES = (
    ("current 1e-10", "tol1e-10"),
    ("intermediate 1e-8", "tol1e-8"),
    ("disabled", "disabled"),
)
METHODS = ("residual", "jacobian", "scalar")


def _load(root: Path, name: str):
    return json.loads((root / name).read_text())


def _calls(report, names):
    return [call for call in report["calls"] if call["point"] in names]


def _median(values):
    return float(np.median(values))


def _relative_l2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()

    timing = {}
    for label, suffix in TOLERANCES:
        timing[label] = {}
        for method in METHODS:
            report = _load(root, f"{method}_nearby_{suffix}.json")
            fresh = _calls(report, ("plus", "minus"))
            repeats = _calls(report, ("plus_repeat", "minus_repeat"))
            timing[label][method] = {
                "fresh_seconds_median": _median([c["seconds"] for c in fresh]),
                "repeat_seconds_median": _median([c["seconds"] for c in repeats]),
                "fresh_host_solve_calls": [
                    c["components"].get(
                        "actual_host_solve_calls",
                        c["components"].get("actual_host_solves", 0),
                    ) for c in fresh
                ],
                "fresh_host_iterations": [
                    c["components"]["host_solver_iterations"] for c in fresh
                ],
                "fresh_refinement_steps": [
                    c["components"]["refinement_steps"] for c in fresh
                ],
                "fresh_refinement_seconds_total": float(sum(
                    c["components"]["refinement_seconds"] for c in fresh
                )),
                "peak_rss_bytes": int(report["peak_rss_bytes"]),
            }

    exact = {}
    for method in METHODS:
        report = _load(root, f"{method}_exact_tol1e-10.json")
        repeats = _calls(report, ("seed_repeat_1", "seed_repeat_2"))
        exact[method] = {
            "repeat_seconds_median": _median([c["seconds"] for c in repeats]),
            "host_solve_calls": [
                c["components"].get(
                    "actual_host_solve_calls",
                    c["components"].get("actual_host_solves", 0),
                ) for c in repeats
            ],
            "refinement_invocations": [
                c["components"]["refinement_invocations"] for c in repeats
            ],
        }

    parity = {}
    for label, suffix in TOLERANCES:
        scalar = _load(root, f"scalar_nearby_{suffix}.json")["calls"][0]
        jacobian = _load(root, f"jacobian_nearby_{suffix}.json")["calls"][0]
        parity[label] = {
            "seed_objective_absolute_difference": abs(
                scalar["objective"] - jacobian["objective"]),
            "seed_gradient_relative_l2": _relative_l2(
                scalar["gradient"], jacobian["gradient"]),
        }

    current = _load(root, "jacobian_nearby_tol1e-10.json")["calls"][0]
    accuracy = {}
    for label, suffix in TOLERANCES[1:]:
        other = _load(root, f"jacobian_nearby_{suffix}.json")["calls"][0]
        accuracy[label] = {
            "seed_objective_relative_change_vs_current": abs(
                other["objective"] - current["objective"]
            ) / abs(current["objective"]),
            "seed_gradient_relative_l2_change_vs_current": _relative_l2(
                other["gradient"], current["gradient"]),
        }

    directional_fd = {}
    for profile in ("aspect", "iota"):
        directional_fd[profile] = {}
        for label, suffix in TOLERANCES:
            directional_fd[profile][label] = {
                method: _load(
                    root, f"{profile}_{method}_nearby_{suffix}.json"
                )["central_directional_fd"]["relative_disagreement"]
                for method in ("jacobian", "scalar")
            }
    directional_fd["run71_qi_dominated"] = {
        label: {
            method: _load(root, f"{method}_nearby_{suffix}.json")[
                "central_directional_fd"
            ]["relative_disagreement"]
            for method in ("jacobian", "scalar")
        }
        for label, suffix in TOLERANCES
    }

    parent = _load(root, "parent_b8c04320_residual.json")
    consolidated = {
        "benchmark": "scalar_surface_gradient_phase1a_summary",
        "timing_and_work": timing,
        "exact_repeat_cache_behavior": exact,
        "scalar_jacobian_seed_parity": parity,
        "refinement_accuracy_curve": accuracy,
        "directional_finite_difference": directional_fd,
        "pre_refinement_parent": {
            "revision": parent["environment"]["vmex_git"]["revision"],
            "warm_residual_seconds": parent["timing"]["warm_execution_seconds"],
            "warm_residual_median_seconds": parent["timing"][
                "warm_execution_median_seconds"
            ],
            "seed_objective": parent["calls"][0]["objective"],
        },
    }
    (root / "PHASE1A_SUMMARY.json").write_text(
        json.dumps(consolidated, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Phase 1A Summary",
        "",
        "Run-71-scale diagnostic results. No production default or public API "
        "was changed.",
        "",
        "## Fresh nearby-point cost",
        "",
        "| refinement | residual | explicit Jacobian | scalar adjoint |",
        "|---|---:|---:|---:|",
    ]
    for label, _ in TOLERANCES:
        lines.append(
            f"| {label} | {timing[label]['residual']['fresh_seconds_median']:.2f} s "
            f"| {timing[label]['jacobian']['fresh_seconds_median']:.2f} s "
            f"| {timing[label]['scalar']['fresh_seconds_median']:.2f} s |"
        )
    lines.extend([
        "",
        "Every fresh nearby point used exactly one host solve. Current "
        "refinement used three Newton/GCROT steps; `1e-8` used two; disabled "
        "used none. Exact repeated points used no new solve or refinement.",
        "",
        "## Exact-repeat derivative cost",
        "",
        "| residual rows | explicit Jacobian | scalar adjoint |",
        "|---:|---:|---:|",
        f"| {exact['residual']['repeat_seconds_median']:.3f} s "
        f"| {exact['jacobian']['repeat_seconds_median']:.3f} s "
        f"| {exact['scalar']['repeat_seconds_median']:.3f} s |",
        "",
        "This isolates objective-row, Jacobian, and scalar-pullback work after "
        "the equilibrium and refinement memos hit.",
        "",
        "## Accuracy observations",
        "",
        f"- At `1e-8`, the seed objective changes by "
        f"{accuracy['intermediate 1e-8']['seed_objective_relative_change_vs_current']:.3e} "
        f"relative and the gradient by "
        f"{accuracy['intermediate 1e-8']['seed_gradient_relative_l2_change_vs_current']:.3e} "
        "relative-L2 versus current `1e-10`.",
        f"- Disabling refinement changes the seed objective by "
        f"{accuracy['disabled']['seed_objective_relative_change_vs_current']:.3e} "
        f"relative and the gradient by "
        f"{accuracy['disabled']['seed_gradient_relative_l2_change_vs_current']:.3e} "
        "relative-L2.",
        "- The smooth aspect-ratio directional-FD gate remains tight for all "
        "three configurations. QI-dominated and solver-sensitive re-solve FDs "
        "retain their documented path/noise sensitivity and are recorded but "
        "are not sole acceptance gates.",
        "",
        "## Diagnosis",
        "",
        "The regression is not a broken exact-point memo and is not duplicate "
        "host/refinement work. A new nearby point performs a very short warm "
        "host solve followed by two or three expensive refinement solves. The "
        "immediate pre-refinement parent has a 2.08 s warm residual median, "
        "matching the current controlled `refine_tol=inf` result. Phase 1B "
        "should therefore target accuracy-preserving refinement acceleration, "
        "not a cache-key repair or an unconditional tolerance/default change.",
        "",
        "Peak RSS varies substantially between fresh processes and does not "
        "show a stable method- or tolerance-dependent trend on this matrix.",
    ])
    (root / "PHASE1A_SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
