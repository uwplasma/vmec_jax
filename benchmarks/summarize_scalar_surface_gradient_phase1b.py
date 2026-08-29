"""Summarize the run-71 Phase-1B refinement benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _calls(report: dict) -> dict[str, dict]:
    return {entry["point"]: entry for entry in report["calls"]}


def _fresh_summary(report: dict) -> dict:
    calls = _calls(report)
    selected = [calls[name] for name in ("plus", "minus")]
    return {
        "seconds": [entry["seconds"] for entry in selected],
        "median_seconds": float(np.median(
            [entry["seconds"] for entry in selected]
        )),
        "refinement_seconds": [
            entry["components"]["refinement_seconds"] for entry in selected
        ],
        "median_refinement_seconds": float(np.median([
            entry["components"]["refinement_seconds"] for entry in selected
        ])),
        "refinement_steps": [
            entry["components"]["refinement_steps"] for entry in selected
        ],
        "peak_rss_bytes": report["peak_rss_bytes"],
    }


def _relative_l2(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(np.linalg.norm(right_array), np.finfo(float).tiny)
    )


def _final_refinement_residual(entry: dict) -> float | None:
    results = entry["components"]["refinement_results"]
    return results[-1]["best_residual"] if results else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a-jacobian", type=Path, required=True)
    parser.add_argument("--legacy-residual", type=Path, required=True)
    parser.add_argument("--selected-residual", type=Path, required=True)
    parser.add_argument("--reverse-residual", type=Path, required=True)
    parser.add_argument("--selected-jacobian", type=Path, required=True)
    parser.add_argument("--reverse-jacobian", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "phase1a_jacobian": args.phase1a_jacobian.resolve(),
        "legacy_residual": args.legacy_residual.resolve(),
        "selected_residual": args.selected_residual.resolve(),
        "reverse_residual": args.reverse_residual.resolve(),
        "selected_jacobian": args.selected_jacobian.resolve(),
        "reverse_jacobian": args.reverse_jacobian.resolve(),
    }
    reports = {name: _load(path) for name, path in paths.items()}

    residual = {
        name: _fresh_summary(reports[name])
        for name in ("legacy_residual", "selected_residual", "reverse_residual")
    }
    legacy_median = residual["legacy_residual"]["median_seconds"]
    for name in ("selected_residual", "reverse_residual"):
        residual[name]["speedup_vs_legacy"] = (
            legacy_median / residual[name]["median_seconds"]
        )

    legacy_calls = _calls(reports["phase1a_jacobian"])
    selected_calls = _calls(reports["selected_jacobian"])
    reverse_calls = _calls(reports["reverse_jacobian"])
    parity = {}
    for point in ("seed", "plus", "minus"):
        legacy = legacy_calls[point]
        selected = selected_calls[point]
        reverse = reverse_calls[point]
        parity[point] = {
            "selected_vs_phase1a": {
                "objective_abs": abs(selected["objective"] - legacy["objective"]),
                "gradient_relative_l2": _relative_l2(
                    selected["gradient"], legacy["gradient"]
                ),
                "final_refinement_residual": _final_refinement_residual(selected),
            },
            "reverse_vs_phase1a": {
                "objective_abs": abs(reverse["objective"] - legacy["objective"]),
                "gradient_relative_l2": _relative_l2(
                    reverse["gradient"], legacy["gradient"]
                ),
                "final_refinement_residual": _final_refinement_residual(reverse),
            },
            "selected_vs_reverse": {
                "objective_abs": abs(selected["objective"] - reverse["objective"]),
                "gradient_relative_l2": _relative_l2(
                    selected["gradient"], reverse["gradient"]
                ),
                "residual_norm_abs": abs(
                    selected["residual_norm"] - reverse["residual_norm"]
                ),
            },
        }

    report = {
        "benchmark": "scalar_surface_gradient_phase1b",
        "artifacts": {name: str(path) for name, path in paths.items()},
        "residual_performance": residual,
        "jacobian_parity": parity,
        "directional_fd_relative_disagreement": {
            "phase1a": reports["phase1a_jacobian"][
                "central_directional_fd"
            ]["relative_disagreement"],
            "selected": reports["selected_jacobian"][
                "central_directional_fd"
            ]["relative_disagreement"],
            "reverse": reports["reverse_jacobian"][
                "central_directional_fd"
            ]["relative_disagreement"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
