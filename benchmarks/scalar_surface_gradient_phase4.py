#!/usr/bin/env python3
"""Run a reduced saved-state production-driver replay for Phase 4.

This diagnostic harness is the only code that touches VMEX's private
instrumentation and legacy/new refinement toggle. The production driver sees
ordinary public VMEX/Simsopt APIs and records its normal configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DRIVER_DIR = ROOT / "SVD" / "single_stage_vacuum_jax"
WRAPPER = DRIVER_DIR / "single_stage_optimization_vacuum_QI_joint.py"
COMMON_DRIVER = DRIVER_DIR / "vacuum_single_stage_common_joint.py"
RUN71 = (
    DRIVER_DIR
    / "output"
    / "QI_jax_JOINTsvd_Stage123_Lengthbound5.5_ncoils8_nfp1_71"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    names = [entry["event"] for entry in events]
    attempts = [
        entry for entry in events if entry["event"] == "host_solve_attempt"
    ]
    successful_attempts = [entry for entry in attempts if entry["succeeded"]]
    callbacks = [
        entry
        for entry in events
        if entry["event"] == "host_callback_complete" and entry["succeeded"]
    ]
    refinement = [
        entry for entry in events if entry["event"] == "refine_complete"
    ]
    steps = [entry for entry in events if entry["event"] == "refine_step"]
    warm_starts = [
        entry for entry in events if entry["event"] == "refine_warm_start"
    ]
    warm_fallbacks = [
        entry
        for entry in events
        if entry["event"] == "refine_warm_start_fallback"
    ]
    return {
        "event_counts": {name: names.count(name) for name in sorted(set(names))},
        "actual_host_solve_calls": len(attempts),
        "successful_host_solves": len(successful_attempts),
        "failed_host_solves": len(attempts) - len(successful_attempts),
        "host_solve_seconds": float(
            sum(float(entry["seconds"]) for entry in successful_attempts)
        ),
        "host_solver_iterations": int(
            sum(int(entry["iterations"]) for entry in successful_attempts)
        ),
        "host_callback_seconds": float(
            sum(float(entry["seconds"]) for entry in callbacks)
        ),
        "refinement_invocations": names.count("refine_start"),
        "refinement_steps": len(steps),
        "refinement_seconds": float(
            sum(float(entry["seconds"]) for entry in refinement)
        ),
        "refinement_krylov_iterations": int(
            sum(int(entry["krylov_iterations"]) for entry in steps)
        ),
        "refinement_warm_starts_available": sum(
            bool(entry["available"]) for entry in warm_starts
        ),
        "refinement_warm_starts_accepted": sum(
            bool(entry["accepted"]) for entry in warm_starts
        ),
        "refinement_warm_start_fallbacks": len(warm_fallbacks),
        "refinement_warm_start_records": warm_starts,
        "refinement_results": refinement,
    }


def _load_wrapper():
    if str(DRIVER_DIR) not in sys.path:
        sys.path.insert(0, str(DRIVER_DIR))
    spec = importlib.util.spec_from_file_location("phase4_vacuum_wrapper", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_output(workflow: str, output_name: str) -> tuple[Path, Path | None]:
    output_dir = DRIVER_DIR / "output" / output_name
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to replace existing Phase 4 output directory: {output_dir}"
        )
    (output_dir / "coils").mkdir(parents=True)
    if workflow == "stage1":
        source_input = DRIVER_DIR / "input" / "input.nfp1_QI"
        source_coils = None
    else:
        source_input = RUN71 / "input.stage1"
        source_coils = RUN71 / "coils" / "biot_savart_inner_loop_max_mode_3.json"
    shutil.copy2(source_input, output_dir / "input.final")
    if source_coils is not None:
        shutil.copy2(source_coils, output_dir / "coils" / "phase4_seed.json")
    return output_dir, source_coils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=("stage1", "single_stage"), required=True)
    parser.add_argument("--mode", choices=("legacy", "new"), required=True)
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument("--output-name")
    args = parser.parse_args()
    if args.maxiter < 1:
        raise ValueError("--maxiter must be positive")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-cache")

    import vmex.core.implicit as imp

    output_name = args.output_name or f"phase4_{args.workflow}_{args.mode}"
    output_dir, source_coils = _prepare_output(args.workflow, output_name)
    events: list[dict[str, Any]] = []
    started = time.perf_counter()

    def diagnostic_hook(event: str, payload: dict[str, Any]) -> None:
        events.append({
            "event": event,
            "elapsed_seconds": time.perf_counter() - started,
            **_json_safe(payload),
        })

    previous_hook = imp._DIAGNOSTIC_HOOK
    previous_cross_point = imp._REFINE_CROSS_POINT_WARM_START
    coil_input = DRIVER_DIR / "coil_inputs" / "biot_savart_nfp1_QI_ncoils8.json"
    coil_backup = coil_input.read_bytes() if coil_input.exists() else None
    try:
        imp._DIAGNOSTIC_HOOK = diagnostic_hook
        imp._REFINE_CROSS_POINT_WARM_START = args.mode == "new"
        if source_coils is not None:
            shutil.copy2(source_coils, coil_input)

        wrapper = _load_wrapper()
        wrapper.OUTPUT_NAME = output_name
        wrapper.RUN_STAGE_1 = args.workflow == "stage1"
        wrapper.RUN_INITIAL_STAGE_2 = False
        wrapper.RUN_SINGLE_STAGE = args.workflow == "single_stage"
        wrapper.RUN_FINAL_STAGE_2_CLEANUP = False
        wrapper.MAX_MODES = [3]
        wrapper.STAGE_1_MAX_MODE = 1
        wrapper.STAGE_1_MAXITER = args.maxiter
        wrapper.SINGLE_STAGE_MAXITER = args.maxiter
        wrapper.PHASE4_VALIDATION_DIAGNOSTICS = True
        wrapper.SAVE_PRECONDITIONING_DIAGNOSTICS = args.workflow == "single_stage"
        wrapper.DIAGNOSTICS_ACCEPTED_INTERVAL = 1
        wrapper.DIAGNOSTICS_COMPONENT_INTERVAL = 0
        wrapper.DIAGNOSTICS_SAVE_FULL_VECTORS = True
        wrapper.SAVE_L_GRAD_B_HISTORY = False
        wrapper.USE_INITIAL_COILS_IF_AVAILABLE = source_coils is not None
        wrapper.CURRENT_INITIALIZATION = (
            "preoptimized" if source_coils is not None else "wrapper"
        )
        wrapper.apply_parameters()

        from src import inputs

        inputs.remove_previous_results = False
        inputs.output_interval = max(args.maxiter + 1, 50)
        inputs.phase4_replay = {
            "workflow": args.workflow,
            "mode": args.mode,
            "cross_point_warm_start": args.mode == "new",
            "maxiter": args.maxiter,
            "source_input": str(
                RUN71 / "input.stage1"
                if args.workflow == "single_stage"
                else DRIVER_DIR / "input" / "input.nfp1_QI"
            ),
            "source_coils": None if source_coils is None else str(source_coils),
        }
        runpy.run_path(str(COMMON_DRIVER), run_name="__main__")
    finally:
        imp._DIAGNOSTIC_HOOK = previous_hook
        imp._REFINE_CROSS_POINT_WARM_START = previous_cross_point
        if coil_backup is None:
            coil_input.unlink(missing_ok=True)
        else:
            coil_input.write_bytes(coil_backup)

    wall_seconds = time.perf_counter() - started
    _write_json(output_dir / "phase4_vmex_events.json", events)
    _write_json(
        output_dir / "phase4_replay_summary.json",
        {
            "workflow": args.workflow,
            "mode": args.mode,
            "cross_point_warm_start": args.mode == "new",
            "maxiter": args.maxiter,
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "event_summary": _event_summary(events),
        },
    )


if __name__ == "__main__":
    main()
