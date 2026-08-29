#!/usr/bin/env python3
"""Consolidate the reduced Phase 4 production-driver replays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = ROOT / "SVD" / "single_stage_vacuum_jax" / "output"
DEFAULT_OUTPUT = ROOT / "external" / "SIMSOPT_VMEX_SCALAR_GRADIENT_PHASE4_ARTIFACTS"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _relative_norm(delta: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(delta.ravel())
        / max(float(np.linalg.norm(reference.ravel())), 1.0e-300)
    )


def _compare_arrays(legacy: Path, new: Path, label: str) -> dict[str, Any]:
    return _compare_npz(
        legacy / "phase4_validation" / f"{label}.npz",
        new / "phase4_validation" / f"{label}.npz",
        legacy / "phase4_validation" / f"{label}.json",
        new / "phase4_validation" / f"{label}.json",
    )


def _compare_npz(
    legacy_arrays: Path,
    new_arrays: Path,
    legacy_metadata: Path | None = None,
    new_metadata: Path | None = None,
) -> dict[str, Any]:
    with np.load(legacy_arrays) as left, np.load(new_arrays) as right:
        result = {}
        for name in sorted(set(left.files) & set(right.files)):
            reference = np.asarray(left[name], dtype=float)
            candidate = np.asarray(right[name], dtype=float)
            delta = candidate - reference
            result[name] = {
                "shape": list(reference.shape),
                "max_absolute_difference": float(np.max(np.abs(delta))),
                "relative_l2_difference": _relative_norm(delta, reference),
            }
    if legacy_metadata is not None and new_metadata is not None:
        left_meta = _read_json(legacy_metadata)
        right_meta = _read_json(new_metadata)
        result["objective"] = {
            "legacy": float(left_meta["adapter_objective"]),
            "new": float(right_meta["adapter_objective"]),
            "absolute_difference": abs(
                float(right_meta["adapter_objective"])
                - float(left_meta["adapter_objective"])
            ),
        }
    return result


def _read_numeric_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        numeric = {}
        for key, value in row.items():
            if value in (None, ""):
                continue
            if value in ("True", "False"):
                numeric[key] = float(value == "True")
                continue
            try:
                numeric[key] = float(value)
            except ValueError:
                continue
        result.append(numeric)
    return result


def _compare_csv(legacy: Path, new: Path) -> dict[str, Any]:
    left = _read_numeric_csv(legacy)
    right = _read_numeric_csv(new)
    common_rows = min(len(left), len(right))
    keys = sorted(
        set.intersection(*(set(row) for row in left[:common_rows] + right[:common_rows]))
    ) if common_rows else []
    comparisons = {}
    for key in keys:
        reference = np.asarray([left[index][key] for index in range(common_rows)])
        candidate = np.asarray([right[index][key] for index in range(common_rows)])
        delta = candidate - reference
        comparisons[key] = {
            "max_absolute_difference": float(np.max(np.abs(delta))),
            "relative_l2_difference": _relative_norm(delta, reference),
        }
    return {
        "legacy_rows": len(left),
        "new_rows": len(right),
        "compared_rows": common_rows,
        "columns": comparisons,
    }


def _run_summary(path: Path) -> dict[str, Any]:
    data = _read_json(path / "phase4_replay_summary.json")
    events = data["event_summary"]
    return {
        "wall_seconds": float(data["wall_seconds"]),
        "peak_rss_bytes": int(data["peak_rss_bytes"]),
        "actual_host_solve_calls": int(events["actual_host_solve_calls"]),
        "successful_host_solves": int(events["successful_host_solves"]),
        "failed_host_solves": int(events["failed_host_solves"]),
        "host_solve_seconds": float(events["host_solve_seconds"]),
        "host_solver_iterations": int(events["host_solver_iterations"]),
        "refinement_invocations": int(events["refinement_invocations"]),
        "refinement_steps": int(events["refinement_steps"]),
        "refinement_seconds": float(events["refinement_seconds"]),
        "refinement_krylov_iterations": int(events["refinement_krylov_iterations"]),
        "warm_starts_available": int(events["refinement_warm_starts_available"]),
        "warm_starts_accepted": int(events["refinement_warm_starts_accepted"]),
        "warm_start_fallbacks": int(events["refinement_warm_start_fallbacks"]),
    }


def _pair(run_root: Path, workflow: str) -> dict[str, Any]:
    legacy = run_root / f"phase4_scalar_gradient_{workflow}_legacy"
    new = run_root / f"phase4_scalar_gradient_{workflow}_new"
    label = (
        "stage1_max_mode_1_initial"
        if workflow == "stage1"
        else "single_stage_max_mode_3_initial"
    )
    legacy_summary = _run_summary(legacy)
    new_summary = _run_summary(new)
    path_file = (
        Path("phase4_validation/stage1_accepted_steps.csv")
        if workflow == "stage1"
        else Path("preconditioning_diagnostics_max_mode_3/accepted_iterations.csv")
    )
    result = {
        "legacy": legacy_summary,
        "new": new_summary,
        "new_over_legacy": {
            "wall_time_ratio": new_summary["wall_seconds"] / legacy_summary["wall_seconds"],
            "peak_rss_ratio": new_summary["peak_rss_bytes"] / legacy_summary["peak_rss_bytes"],
            "refinement_time_ratio": (
                new_summary["refinement_seconds"] / legacy_summary["refinement_seconds"]
            ),
        },
        "initial_linearization": _compare_arrays(legacy, new, label),
        "accepted_path": _compare_csv(legacy / path_file, new / path_file),
    }
    if workflow == "single_stage":
        result["evaluation_path"] = _compare_csv(
            legacy / "loss_history_max_mode_3.csv",
            new / "loss_history_max_mode_3.csv",
        )
    return result


def _finite_beta(artifact_root: Path) -> dict[str, Any]:
    legacy_json = artifact_root / "finite_beta_construction_legacy.json"
    new_json = artifact_root / "finite_beta_construction_new.json"
    legacy = _read_json(legacy_json)
    new = _read_json(new_json)
    return {
        "legacy": {
            "total_seconds": float(legacy["total_seconds"]),
            "peak_rss_bytes": int(legacy["peak_rss_bytes"]),
            "objective_value": float(legacy["objective_value"]),
            "surface_gradient_norm": float(legacy["surface_gradient_norm"]),
            "timings": legacy["timings"],
            "event_summary": legacy["event_summary"],
        },
        "new": {
            "total_seconds": float(new["total_seconds"]),
            "peak_rss_bytes": int(new["peak_rss_bytes"]),
            "objective_value": float(new["objective_value"]),
            "surface_gradient_norm": float(new["surface_gradient_norm"]),
            "timings": new["timings"],
            "event_summary": new["event_summary"],
        },
        "new_over_legacy": {
            "total_time_ratio": float(new["total_seconds"]) / float(legacy["total_seconds"]),
            "peak_rss_ratio": int(new["peak_rss_bytes"]) / int(legacy["peak_rss_bytes"]),
        },
        "arrays": _compare_npz(
            artifact_root / "finite_beta_construction_legacy.npz",
            artifact_root / "finite_beta_construction_new.npz",
        ),
        "objective_absolute_difference": abs(
            float(new["objective_value"]) - float(legacy["objective_value"])
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Scalar Surface Gradient Phase 4 Replay",
        "",
        f"Gate: **{'PASS' if summary['gate']['passed'] else 'FAIL'}**.",
        "",
    ]
    for workflow in ("stage1", "single_stage"):
        item = summary[workflow]
        old = item["legacy"]
        new = item["new"]
        linear = item["initial_linearization"]
        path = item["accepted_path"]
        lines.extend([
            f"## {workflow.replace('_', ' ').title()}",
            "",
            f"- Wall time: legacy {old['wall_seconds']:.3f} s; new {new['wall_seconds']:.3f} s; ratio {item['new_over_legacy']['wall_time_ratio']:.3f}.",
            f"- Peak RSS: legacy {old['peak_rss_bytes'] / 2**30:.3f} GiB; new {new['peak_rss_bytes'] / 2**30:.3f} GiB; ratio {item['new_over_legacy']['peak_rss_ratio']:.3f}.",
            f"- Refinement: legacy {old['refinement_steps']} steps / {old['refinement_krylov_iterations']} Krylov iterations / {old['refinement_seconds']:.3f} s; new {new['refinement_steps']} / {new['refinement_krylov_iterations']} / {new['refinement_seconds']:.3f} s.",
            f"- New-mode warm starts: {new['warm_starts_available']} available, {new['warm_starts_accepted']} accepted, {new['warm_start_fallbacks']} fallbacks.",
            f"- Initial residual relative L2 delta: {linear['residual']['relative_l2_difference']:.3e}.",
            f"- Initial Jacobian relative Frobenius delta: {linear['jacobian']['relative_l2_difference']:.3e}.",
            f"- Initial contracted-gradient relative L2 delta: {linear['contracted_gradient']['relative_l2_difference']:.3e}.",
            f"- Accepted path rows: legacy {path['legacy_rows']}, new {path['new_rows']}, compared {path['compared_rows']}.",
            "",
        ])
    finite = summary.get("finite_beta")
    if finite is not None:
        old = finite["legacy"]
        new = finite["new"]
        arrays = finite["arrays"]
        lines.extend([
            "## Finite Beta Construction",
            "",
            f"- Total time: legacy {old['total_seconds']:.3f} s; new {new['total_seconds']:.3f} s; ratio {finite['new_over_legacy']['total_time_ratio']:.3f}.",
            f"- Peak RSS: legacy {old['peak_rss_bytes'] / 2**30:.3f} GiB; new {new['peak_rss_bytes'] / 2**30:.3f} GiB; ratio {finite['new_over_legacy']['peak_rss_ratio']:.3f}.",
            f"- Virtual Casing provider construction: legacy {old['timings']['virtual_casing_provider_construction_seconds']:.3f} s; new {new['timings']['virtual_casing_provider_construction_seconds']:.3f} s.",
            f"- Squared-flux surface gradient: legacy {old['timings']['squared_flux_surface_gradient_seconds']:.3f} s; new {new['timings']['squared_flux_surface_gradient_seconds']:.3f} s.",
            f"- Virtual Casing target relative L2 delta: {arrays['virtual_casing_target']['relative_l2_difference']:.3e}.",
            f"- Squared-flux surface-gradient relative L2 delta: {arrays['squared_flux_surface_gradient']['relative_l2_difference']:.3e}.",
            f"- Objective absolute delta: {finite['objective_absolute_difference']:.3e}.",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stage1 = _pair(args.run_root, "stage1")
    single_stage = _pair(args.run_root, "single_stage")
    finite_beta = _finite_beta(args.output)
    gate = {
        "initial_linearizations_exact": all(
            pair["initial_linearization"][name]["relative_l2_difference"] == 0.0
            for pair in (stage1, single_stage)
            for name in ("residual", "jacobian", "contracted_gradient")
        ),
        "stage1_steps_exact": (
            stage1["accepted_path"]["columns"]["physical_step_norm"][
                "relative_l2_difference"
            ] == 0.0
        ),
        "single_stage_step_relative_l2_below_1e_7": (
            single_stage["accepted_path"]["columns"]["physical_step_norm"][
                "relative_l2_difference"
            ] < 1.0e-7
        ),
        "single_stage_objective_relative_l2_below_1e_8": (
            single_stage["accepted_path"]["columns"]["J"][
                "relative_l2_difference"
            ] < 1.0e-8
        ),
        "single_stage_failure_path_equal": (
            single_stage["evaluation_path"]["columns"]["failed"][
                "max_absolute_difference"
            ] == 0.0
            and single_stage["legacy"]["failed_host_solves"]
            == single_stage["new"]["failed_host_solves"]
        ),
        "finite_beta_target_and_gradient_exact": all(
            finite_beta["arrays"][name]["relative_l2_difference"] == 0.0
            for name in ("virtual_casing_target", "squared_flux_surface_gradient")
        ),
    }
    gate["passed"] = all(gate.values())
    summary = {
        "comparison": "legacy cross-point warm-start disabled versus Phase-1B guarded reuse enabled",
        "provenance": {
            "python": sys.executable,
            "vmex_git_head": _git_head(Path(__file__).resolve().parents[1]),
            "simsopt_git_head": _git_head(ROOT / "external" / "simsopt_latest_vmex"),
            "vacuum_driver_git_head": _git_head(
                ROOT / "SVD" / "single_stage_vacuum_jax"
            ),
            "run_root": str(args.run_root.resolve()),
        },
        "stage1": stage1,
        "single_stage": single_stage,
        "finite_beta": finite_beta,
        "gate": gate,
    }
    (args.output / "phase4_replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output / "phase4_replay_summary.md").write_text(_markdown(summary) + "\n")


if __name__ == "__main__":
    main()
