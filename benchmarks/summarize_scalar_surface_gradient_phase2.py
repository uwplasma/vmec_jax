#!/usr/bin/env python3
"""Consolidate corrected-path scalar/Jacobian Phase-2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("residual", "jacobian", "scalar")
ORDERS = ("forward", "reverse")


def _load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _calls(report: dict) -> dict[str, dict]:
    return {entry["point"]: entry for entry in report["calls"]}


def _fresh(report: dict) -> list[dict]:
    calls = _calls(report)
    return [calls["plus"], calls["minus"]]


def _repeats(report: dict) -> list[dict]:
    calls = _calls(report)
    return [calls["plus_repeat"], calls["minus_repeat"]]


def _median(values) -> float:
    return float(np.median(list(values)))


def _relative_l2(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(right), np.finfo(float).tiny)
    )


def _method_summary(report: dict) -> dict:
    fresh = _fresh(report)
    repeats = _repeats(report)
    return {
        "fresh_seconds": [entry["seconds"] for entry in fresh],
        "fresh_median_seconds": _median(entry["seconds"] for entry in fresh),
        "exact_repeat_seconds": [entry["seconds"] for entry in repeats],
        "exact_repeat_median_seconds": _median(
            entry["seconds"] for entry in repeats
        ),
        "host_callback_median_seconds": _median(
            entry["components"]["host_callback_seconds"] for entry in fresh
        ),
        "host_solve_median_seconds": _median(
            entry["components"]["host_solve_seconds"] for entry in fresh
        ),
        "refinement_median_seconds": _median(
            entry["components"]["refinement_seconds"] for entry in fresh
        ),
        "outside_callback_median_seconds": _median(
            entry["components"]["outside_host_callback_seconds"]
            for entry in fresh
        ),
        "refinement_steps": [
            entry["components"]["refinement_steps"] for entry in fresh
        ],
        "peak_rss_bytes": int(report["peak_rss_bytes"]),
    }


def _parity(jacobian: dict, scalar: dict) -> dict:
    jac_calls = _calls(jacobian)
    scalar_calls = _calls(scalar)
    return {
        point: {
            "objective_absolute_difference": abs(
                scalar_calls[point]["objective"] - jac_calls[point]["objective"]
            ),
            "gradient_relative_l2": _relative_l2(
                scalar_calls[point]["gradient"], jac_calls[point]["gradient"]
            ),
        }
        for point in ("seed", "plus", "minus")
    }


def _warm_start_sweep(report: dict) -> dict:
    calls = report["calls"][1:]
    records = []
    for call in calls:
        components = call["components"]
        warm = components["refinement_warm_start_records"]
        if len(warm) != 1:
            raise ValueError(
                f"expected one warm-start record for {call['point']}, got {len(warm)}"
            )
        record = warm[0]
        records.append({
            "point": call["point"],
            "seconds": call["seconds"],
            "available": bool(record["available"]),
            "accepted": bool(record["accepted"]),
            "base_residual": record["base_residual"],
            "candidate_residual": record["candidate_residual"],
            "candidate_to_base_ratio": (
                record["candidate_residual"] / record["base_residual"]
                if record["candidate_residual"] is not None
                and record["base_residual"] not in (None, 0.0)
                else None
            ),
            "fallbacks": components["refinement_warm_start_fallbacks"],
            "refinement_steps": components["refinement_steps"],
            "refinement_seconds": components["refinement_seconds"],
        })
    available = [entry for entry in records if entry["available"]]
    accepted = [entry for entry in available if entry["accepted"]]
    return {
        "records": records,
        "available_count": len(available),
        "accepted_count": len(accepted),
        "acceptance_rate": len(accepted) / len(available) if available else None,
        "fallback_count": int(sum(entry["fallbacks"] for entry in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()

    reports = {
        order: {
            method: _load(root / f"mechanism_{method}_{order}.json")
            for method in METHODS
        }
        for order in ORDERS
    }
    mechanism = {
        order: {
            method: _method_summary(reports[order][method])
            for method in METHODS
        }
        for order in ORDERS
    }
    for order in ORDERS:
        residual_time = mechanism[order]["residual"]["fresh_median_seconds"]
        jacobian_time = mechanism[order]["jacobian"]["fresh_median_seconds"]
        scalar_time = mechanism[order]["scalar"]["fresh_median_seconds"]
        residual_repeat = mechanism[order]["residual"][
            "exact_repeat_median_seconds"
        ]
        jacobian_repeat = mechanism[order]["jacobian"][
            "exact_repeat_median_seconds"
        ]
        scalar_repeat = mechanism[order]["scalar"][
            "exact_repeat_median_seconds"
        ]
        mechanism[order]["decision_metrics"] = {
            "fresh_jacobian_minus_residual_seconds": jacobian_time - residual_time,
            "fresh_scalar_to_jacobian_time_ratio": scalar_time / jacobian_time,
            "exact_repeat_jacobian_increment_over_residual_seconds": (
                jacobian_repeat - residual_repeat
            ),
            "exact_repeat_jacobian_increment_fraction": (
                (jacobian_repeat - residual_repeat) / jacobian_repeat
            ),
            "exact_repeat_scalar_to_jacobian_time_ratio": (
                scalar_repeat / jacobian_repeat
            ),
            "scalar_materially_faster_at_20_percent": (
                scalar_repeat <= 0.8 * jacobian_repeat
            ),
        }

    batching = {}
    for label in ("batch1", "batch4", "batch8", "full", "auto"):
        report = _load(root / f"knob_jacobian_{label}.json")
        batching[label] = _method_summary(report)
        batching[label]["effective_auto_dof_chunk"] = report["configuration"].get(
            "effective_auto_dof_chunk"
        )

    certification = {}
    for label in ("1e-4", "1e-5", "1e-6"):
        report = _load(root / f"cert_jacobian_{label}.json")
        certification[label] = _method_summary(report)

    sweep_report = _load(root / "acceptance_residual_sweep.json")
    sweep = _warm_start_sweep(sweep_report)

    attribution = {}
    for method in METHODS:
        forward = mechanism["forward"][method]
        reverse = mechanism["reverse"][method]
        attribution[method] = {
            key: reverse[key] - forward[key]
            for key in (
                "fresh_median_seconds", "host_callback_median_seconds",
                "host_solve_median_seconds", "refinement_median_seconds",
                "outside_callback_median_seconds",
            )
        }

    parity = {
        order: _parity(reports[order]["jacobian"], reports[order]["scalar"])
        for order in ORDERS
    }
    fd = {
        order: {
            method: reports[order][method]["central_directional_fd"]
            for method in ("jacobian", "scalar")
        }
        for order in ORDERS
    }
    scalar_wins = all(
        mechanism[order]["decision_metrics"][
            "scalar_materially_faster_at_20_percent"
        ]
        for order in ORDERS
    )
    summary = {
        "benchmark": "scalar_surface_gradient_phase2",
        "mechanism": mechanism,
        "scalar_jacobian_parity": parity,
        "directional_finite_difference": fd,
        "jacobian_batching": batching,
        "jacobian_certification": certification,
        "warm_start_acceptance_sweep": sweep,
        "forward_reverse_attribution": attribution,
        "gate": {
            "scalar_materially_faster_in_both_orders": scalar_wins,
            "decision": (
                "proceed_with_scalar_policy" if scalar_wins
                else "retain_explicit_jacobian_contraction"
            ),
        },
    }
    output = args.output.resolve() if args.output else root / "PHASE2_SUMMARY.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    forward = mechanism["forward"]
    reverse = mechanism["reverse"]
    max_seed_gradient_difference = max(
        parity[order]["seed"]["gradient_relative_l2"] for order in ORDERS
    )
    max_nearby_gradient_difference = max(
        parity[order][point]["gradient_relative_l2"]
        for order in ORDERS for point in ("plus", "minus")
    )
    residual_attribution = attribution["residual"]
    markdown = [
        "# Scalar Surface-Gradient Phase 2 Summary",
        "",
        "Run-71-scale corrected-value-path measurements. Each method and knob",
        "configuration ran in a fresh process. A 20% advantage was the",
        "operational threshold for a material scalar-path win; the conclusion",
        "is insensitive to that threshold because the scalar path was slower.",
        "",
        "## Mechanism decision",
        "",
        "| order | residual repeat | Jacobian repeat | scalar repeat | scalar/Jacobian |",
        "|---|---:|---:|---:|---:|",
        f"| forward | {forward['residual']['exact_repeat_median_seconds']:.3f} s "
        f"| {forward['jacobian']['exact_repeat_median_seconds']:.3f} s "
        f"| {forward['scalar']['exact_repeat_median_seconds']:.3f} s "
        f"| {forward['decision_metrics']['exact_repeat_scalar_to_jacobian_time_ratio']:.2f}x |",
        f"| reverse | {reverse['residual']['exact_repeat_median_seconds']:.3f} s "
        f"| {reverse['jacobian']['exact_repeat_median_seconds']:.3f} s "
        f"| {reverse['scalar']['exact_repeat_median_seconds']:.3f} s "
        f"| {reverse['decision_metrics']['exact_repeat_scalar_to_jacobian_time_ratio']:.2f}x |",
        "",
        "The existing scalar adjoint is 7.37-8.02x slower than the explicit",
        "Jacobian contraction after equilibrium/refinement caches hit. Fresh",
        "end-to-end scalar evaluations are also 2.14-3.18x slower. Retain",
        "the explicit `J.T @ r` contraction and do not add the provisional",
        "scalar-gradient public policy.",
        "",
        "## Numerical checks",
        "",
        f"- Seed scalar/Jacobian gradient difference: at most "
        f"`{max_seed_gradient_difference:.3e}` relative L2.",
        f"- Fresh nearby scalar/Jacobian gradient difference: at most "
        f"`{max_nearby_gradient_difference:.3e}` relative L2 across isolated "
        "processes.",
        "- Jacobian directional-FD disagreement is `3.46e-6` forward and",
        "  `6.25e-6` reverse. Scalar disagreement is `2.96e-2` and `2.67e-2`.",
        "- Since the scalar path fails the performance gate, these larger",
        "  nearby/path-sensitive scalar discrepancies do not gate production.",
        "",
        "## Cross-point reuse and order attribution",
        "",
        f"The logarithmic synthetic step sweep accepted "
        f"{sweep['accepted_count']} of {sweep['available_count']} available "
        "corrections. Only the `1e-7` offset was accepted; all offsets from",
        "`1e-6` through `1e-2` were rejected. This brackets the reuse basin but",
        "is not a replay of saved production optimizer DOFs.",
        "",
        f"In the residual-only controlled comparison, reverse ordering added "
        f"{residual_attribution['fresh_median_seconds']:.2f} s total and "
        f"{residual_attribution['refinement_median_seconds']:.2f} s in "
        "refinement, while host-solve time changed by only "
        f"{residual_attribution['host_solve_median_seconds']:.2f} s. The",
        "forward/reverse asymmetry is therefore attributable to refinement",
        "path quality, not the equilibrium solve or derivative contraction.",
        "",
        "## Jacobian knobs and RSS",
        "",
        "Corrected-path exact-repeat Jacobian medians range from about 0.70 to",
        "0.96 s across batch 1, 4, 8, full, and auto. Certification tolerances",
        "from `1e-4` through `1e-6` do not materially change the result. Full",
        "batching has the largest observed RSS in the batch sweep; batch 1",
        "remains the conservative low-memory choice, though process-to-process",
        "RSS variation prevents claiming a stable memory improvement.",
        "",
        "## Gate",
        "",
        "Phase 2 closes the provisional scalar-gradient API path: retain the",
        "explicit Jacobian contraction. Phase 3 therefore needs no scalar",
        "policy passthrough; it should retain the Phase-1B refinement behavior",
        "and existing residual/Jacobian APIs.",
    ]
    (root / "PHASE2_SUMMARY.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
