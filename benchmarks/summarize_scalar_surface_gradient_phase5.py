#!/usr/bin/env python3
"""Consolidate Phase-5 isolated-cache evaluator and production replays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _median(items) -> float:
    return float(statistics.median(float(item) for item in items))


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm((value - reference).ravel())
        / max(float(np.linalg.norm(reference.ravel())), 1.0e-300)
    )


def _calls(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["point"]: entry for entry in report["calls"]}


def _evaluation_record(path: Path) -> dict[str, Any]:
    report = _read(path)
    calls = _calls(report)
    fresh = [calls[name]["seconds"] for name in ("plus", "minus")]
    repeats = [calls[name]["seconds"] for name in ("plus_repeat", "minus_repeat")]
    all_events = report["all_events"]
    names = [event["event"] for event in all_events]
    steps = [event for event in all_events if event["event"] == "refine_step"]
    warm = [event for event in all_events if event["event"] == "refine_warm_start"]
    return {
        "path": str(path),
        "build_seconds": float(report["timing"]["build_seconds"]),
        "seed_seconds": float(calls["seed"]["seconds"]),
        "fresh_nearby_median_seconds": _median(fresh),
        "exact_repeat_median_seconds": _median(repeats),
        "peak_rss_bytes": int(report["peak_rss_bytes"]),
        "seed_objective": float(calls["seed"]["objective"]),
        "seed_gradient": np.asarray(calls["seed"]["gradient"], dtype=float),
        "central_fd_relative_disagreement": float(
            report["central_directional_fd"]["relative_disagreement"]
        ),
        "host_solve_attempts": names.count("host_solve_attempt"),
        "refinement_invocations": names.count("refine_start"),
        "refinement_steps": len(steps),
        "refinement_krylov_iterations": int(
            sum(int(event["krylov_iterations"]) for event in steps)
        ),
        "warm_starts_available": sum(bool(event["available"]) for event in warm),
        "warm_starts_accepted": sum(bool(event["accepted"]) for event in warm),
        "warm_start_fallbacks": names.count("refine_warm_start_fallback"),
    }


def _read_accepted(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        parsed = {}
        for key in ("J", "failed", "physical_step_norm"):
            value = row.get(key)
            if value in (None, ""):
                continue
            parsed[key] = float(value == "True") if key == "failed" else float(value)
        result.append(parsed)
    return result


def _path_difference(candidate: list[dict[str, float]], reference: list[dict[str, float]]) -> dict[str, Any]:
    common = min(len(candidate), len(reference))
    result = {"candidate_rows": len(candidate), "reference_rows": len(reference), "compared_rows": common}
    for key in ("J", "failed", "physical_step_norm"):
        left = np.asarray([reference[index][key] for index in range(common)])
        right = np.asarray([candidate[index][key] for index in range(common)])
        result[key] = {
            "max_absolute_difference": float(np.max(np.abs(right - left))) if common else float("inf"),
            "relative_l2_difference": _relative(right, left) if common else float("inf"),
        }
    return result


def _production_record(path: Path) -> dict[str, Any]:
    replay = _read(path / "phase4_replay_summary.json")
    final = _read(path / "final_summary.json")
    stage = _read(path / "single_stage_summary_max_mode_3.json")
    events = replay["event_summary"]
    refinement_results = events["refinement_results"]
    surface = final["surface"]
    coils = final["coils"]
    return {
        "path": str(path),
        "wall_seconds": float(replay["wall_seconds"]),
        "peak_rss_bytes": int(replay["peak_rss_bytes"]),
        "actual_host_solve_calls": int(events["actual_host_solve_calls"]),
        "failed_host_solves": int(events["failed_host_solves"]),
        "host_solver_iterations": int(events["host_solver_iterations"]),
        "refinement_invocations": int(events["refinement_invocations"]),
        "refinement_steps": int(events["refinement_steps"]),
        "refinement_seconds": float(events["refinement_seconds"]),
        "refinement_krylov_iterations": int(events["refinement_krylov_iterations"]),
        "warm_starts_available": int(events["refinement_warm_starts_available"]),
        "warm_starts_accepted": int(events["refinement_warm_starts_accepted"]),
        "warm_start_fallbacks": int(events["refinement_warm_start_fallbacks"]),
        "all_refinements_certified": all(
            bool(item["met_tolerance"]) for item in refinement_results
        ),
        "maximum_refined_residual": max(
            (float(item["best_residual"]) for item in refinement_results),
            default=0.0,
        ),
        "optimizer": stage["optimizer_result"],
        "surface_objective": float(surface["surface_objective"]),
        "qi_legacy_squared_norm": float(
            surface["qi_residuals"]["legacy_four_block"]["squared_norm"]
        ),
        "qi_new_squared_norm": float(
            surface["qi_residuals"]["omnigenity_three_term"]["squared_norm"]
        ),
        "normalized_squared_flux": float(coils["Jf"]),
        "mean_abs_bdotn": float(coils["mean_abs_BdotN"]),
        "max_abs_bdotn": float(coils["max_abs_BdotN"]),
        "total_length": float(coils["total_length"]),
        "cc_distance": float(coils["cc_distance"]),
        "cs_distance": float(coils["cs_distance"]),
        "max_curvature": float(coils["max_curvature"]),
        "max_msc": float(coils["max_msc"]),
        "constraint_penalties": {
            key: float(coils[key])
            for key in ("J_length", "J_CC", "J_CS", "J_CURVATURE", "J_MSC")
        },
        "accepted_path": _read_accepted(
            path / "preconditioning_diagnostics_max_mode_3" / "accepted_iterations.csv"
        ),
    }


def _finite_record(path: Path) -> dict[str, Any]:
    candidates = list(path.glob("finite_beta_construction_*.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one finite-beta JSON in {path}")
    report = _read(candidates[0])
    arrays_path = candidates[0].with_suffix(".npz")
    with np.load(arrays_path) as arrays:
        target = np.asarray(arrays["virtual_casing_target"], dtype=float)
        gradient = np.asarray(arrays["squared_flux_surface_gradient"], dtype=float)
    refinement = [
        event for event in report["event_log"] if event["event"] == "refine_complete"
    ]
    return {
        "path": str(path),
        "total_seconds": float(report["total_seconds"]),
        "peak_rss_bytes": int(report["peak_rss_bytes"]),
        "objective": float(report["objective_value"]),
        "target": target,
        "gradient": gradient,
        "timings": report["timings"],
        "event_summary": report["event_summary"],
        "all_refinements_certified": all(
            bool(item["met_tolerance"]) for item in refinement
        ),
        "maximum_refined_residual": max(
            (float(item["best_residual"]) for item in refinement), default=0.0
        ),
    }


def _without_arrays(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in record.items()
    }


def _aggregate(records: list[dict[str, Any]], fields: tuple[str, ...]) -> dict:
    return {f"median_{field}": _median(record[field] for record in records) for field in fields}


def _policy_groups(manifest: dict, component: str, loader) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for entry in manifest["runs"]:
        if entry["component"] != component or entry["status"] != "complete":
            continue
        groups.setdefault(entry["policy"], []).append(loader(Path(entry["result"])))
    return groups


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Scalar Surface Gradient Phase 5",
        "",
        f"Gate: **{'PASS' if summary['gate']['passed'] else 'FAIL'}**.",
        f"Provisional disposition: **{summary['default_decision']}**.",
        "",
    ]
    for component in ("evaluation", "production", "finite_beta"):
        lines.extend([f"## {component.replace('_', ' ').title()}", ""])
        for policy, aggregate in summary[component]["aggregate"].items():
            metrics = "; ".join(
                f"{name.replace('median_', '')}={value:.6g}"
                for name, value in aggregate.items()
            )
            lines.append(f"- `{policy}`: {metrics}.")
        lines.append("")
    lines.extend(["## Gate Details", ""])
    for name, value in summary["gate"].items():
        if name != "passed":
            lines.append(f"- `{name}`: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    manifest = _read(root / "phase5_manifest.json")
    if any(entry["status"] != "complete" for entry in manifest["runs"]):
        raise RuntimeError("Phase-5 manifest contains incomplete runs")

    evaluations = _policy_groups(manifest, "evaluation", _evaluation_record)
    production = _policy_groups(manifest, "production", _production_record)
    finite = _policy_groups(manifest, "finite_beta", _finite_record)
    required = {"strict_no_reuse", "strict_reuse", "relaxed_reuse"}
    for name, groups in (("evaluation", evaluations), ("production", production), ("finite_beta", finite)):
        if set(groups) != required or any(len(items) < 2 for items in groups.values()):
            raise RuntimeError(f"{name} does not contain two runs for every policy")

    eval_reference = evaluations["strict_no_reuse"][0]
    finite_reference = finite["strict_no_reuse"][0]
    evaluation_deltas = {}
    for policy, records in evaluations.items():
        evaluation_deltas[policy] = {
            "max_seed_objective_absolute_difference": max(
                abs(record["seed_objective"] - eval_reference["seed_objective"])
                for record in records
            ),
            "max_seed_gradient_relative_difference": max(
                _relative(record["seed_gradient"], eval_reference["seed_gradient"])
                for record in records
            ),
            "max_central_fd_relative_disagreement": max(
                record["central_fd_relative_disagreement"] for record in records
            ),
        }
    finite_deltas = {}
    for policy, records in finite.items():
        finite_deltas[policy] = {
            "max_objective_absolute_difference": max(
                abs(record["objective"] - finite_reference["objective"])
                for record in records
            ),
            "max_target_relative_difference": max(
                _relative(record["target"], finite_reference["target"])
                for record in records
            ),
            "max_gradient_relative_difference": max(
                _relative(record["gradient"], finite_reference["gradient"])
                for record in records
            ),
        }

    eval_aggregate = {
        policy: _aggregate(records, (
            "build_seconds", "seed_seconds", "fresh_nearby_median_seconds",
            "exact_repeat_median_seconds", "peak_rss_bytes", "refinement_steps",
            "refinement_krylov_iterations",
        ))
        for policy, records in evaluations.items()
    }
    production_aggregate = {
        policy: _aggregate(records, (
            "wall_seconds", "peak_rss_bytes", "refinement_seconds",
            "refinement_steps", "refinement_krylov_iterations",
            "normalized_squared_flux", "qi_legacy_squared_norm",
        ))
        for policy, records in production.items()
    }
    finite_aggregate = {
        policy: _aggregate(records, ("total_seconds", "peak_rss_bytes", "objective"))
        for policy, records in finite.items()
    }

    strict_wall_ratio = (
        production_aggregate["strict_reuse"]["median_wall_seconds"]
        / production_aggregate["strict_no_reuse"]["median_wall_seconds"]
    )
    relaxed_wall_ratio = (
        production_aggregate["relaxed_reuse"]["median_wall_seconds"]
        / production_aggregate["strict_no_reuse"]["median_wall_seconds"]
    )
    relaxed_rss_ratio = (
        production_aggregate["relaxed_reuse"]["median_peak_rss_bytes"]
        / production_aggregate["strict_no_reuse"]["median_peak_rss_bytes"]
    )
    production_reference = production["strict_no_reuse"][0]
    production_relaxed = production["relaxed_reuse"]
    production_path_deltas = {
        policy: [
            _path_difference(record["accepted_path"], production_reference["accepted_path"])
            for record in records
        ]
        for policy, records in production.items()
    }
    max_flux_relative = max(
        abs(record["normalized_squared_flux"] - production_reference["normalized_squared_flux"])
        / max(abs(production_reference["normalized_squared_flux"]), 1.0e-300)
        for record in production_relaxed
    )
    max_qi_relative = max(
        abs(record["qi_legacy_squared_norm"] - production_reference["qi_legacy_squared_norm"])
        / max(abs(production_reference["qi_legacy_squared_norm"]), 1.0e-300)
        for record in production_relaxed
    )
    relaxed_failures = sum(
        record["failed_host_solves"] + record["warm_start_fallbacks"]
        for record in production_relaxed
    )
    max_relaxed_path_objective = max(
        item["J"]["relative_l2_difference"]
        for item in production_path_deltas["relaxed_reuse"]
    )
    max_relaxed_path_step = max(
        item["physical_step_norm"]["relative_l2_difference"]
        for item in production_path_deltas["relaxed_reuse"]
    )
    relaxed_failure_path_equal = all(
        item["failed"]["max_absolute_difference"] == 0.0
        for item in production_path_deltas["relaxed_reuse"]
    )
    relaxed_refinements_certified = all(
        record["all_refinements_certified"] for record in production_relaxed
    )
    finite_relaxed_certified = all(
        record["all_refinements_certified"] for record in finite["relaxed_reuse"]
    )
    reference_penalties = np.asarray(
        list(production_reference["constraint_penalties"].values()), dtype=float
    )
    max_relaxed_constraint_penalty_relative = max(
        _relative(
            np.asarray(list(record["constraint_penalties"].values()), dtype=float),
            reference_penalties,
        )
        for record in production_relaxed
    )
    optimizer_counts_equal = all(
        all(
            record["optimizer"].get(key) == production_reference["optimizer"].get(key)
            for key in ("status", "nit", "nfev", "njev")
        )
        for record in production_relaxed
    )
    gate = {
        "strict_evaluation_gradient_relative_below_1e_7": evaluation_deltas["strict_reuse"]["max_seed_gradient_relative_difference"] < 1.0e-7,
        "relaxed_evaluation_gradient_relative_below_1e_4": evaluation_deltas["relaxed_reuse"]["max_seed_gradient_relative_difference"] < 1.0e-4,
        "finite_beta_target_relative_below_1e_4": finite_deltas["relaxed_reuse"]["max_target_relative_difference"] < 1.0e-4,
        "finite_beta_gradient_relative_below_1e_4": finite_deltas["relaxed_reuse"]["max_gradient_relative_difference"] < 1.0e-4,
        "relaxed_accepted_objective_path_relative_below_1e_3": max_relaxed_path_objective < 1.0e-3,
        "relaxed_accepted_step_path_relative_below_1e_2": max_relaxed_path_step < 1.0e-2,
        "relaxed_failure_path_equal": relaxed_failure_path_equal,
        "relaxed_refinements_certified": relaxed_refinements_certified,
        "finite_beta_relaxed_refinements_certified": finite_relaxed_certified,
        "relaxed_optimizer_counts_equal": optimizer_counts_equal,
        "relaxed_constraint_penalty_relative_below_1e_3": max_relaxed_constraint_penalty_relative < 1.0e-3,
        "relaxed_final_flux_relative_below_1e_2": max_flux_relative < 1.0e-2,
        "relaxed_final_qi_relative_below_1e_2": max_qi_relative < 1.0e-2,
        "relaxed_no_refinement_fallbacks": relaxed_failures == sum(record["failed_host_solves"] for record in production_relaxed),
        "relaxed_peak_rss_ratio_below_1p1": relaxed_rss_ratio < 1.1,
        "strict_production_wall_ratio": strict_wall_ratio,
        "relaxed_production_wall_ratio": relaxed_wall_ratio,
        "material_wall_reduction": relaxed_wall_ratio < 0.8,
    }
    gate["passed"] = all(
        bool(value) for name, value in gate.items()
        if name not in {"strict_production_wall_ratio", "relaxed_production_wall_ratio"}
    )
    if gate["passed"]:
        default_decision = (
            "pending reviewer agreement; candidate guarded reuse with "
            "refine_tol=1e-8"
        )
    else:
        default_decision = (
            "pending reviewer agreement; recommend retaining guarded reuse "
            "with strict refine_tol=1e-10 and not adopting relaxed refinement"
        )

    summary = {
        "protocol": {
            "sequence": manifest["sequence"],
            "cache_protocol": manifest["cache_protocol"],
            "fresh_processes_per_policy_per_component": 2,
            "provenance": manifest.get("provenance", {}),
        },
        "evaluation": {
            "aggregate": eval_aggregate,
            "deltas": evaluation_deltas,
            "runs": {policy: [_without_arrays(record) for record in records] for policy, records in evaluations.items()},
        },
        "production": {
            "aggregate": production_aggregate,
            "runs": production,
            "relaxed_max_final_flux_relative_difference": max_flux_relative,
            "relaxed_max_final_qi_relative_difference": max_qi_relative,
            "relaxed_max_constraint_penalty_relative_difference": max_relaxed_constraint_penalty_relative,
            "accepted_path_deltas": production_path_deltas,
        },
        "finite_beta": {
            "aggregate": finite_aggregate,
            "deltas": finite_deltas,
            "runs": {policy: [_without_arrays(record) for record in records] for policy, records in finite.items()},
        },
        "gate": gate,
        "default_decision": default_decision,
    }
    (root / "phase5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "phase5_summary.md").write_text(_markdown(summary))


if __name__ == "__main__":
    main()
