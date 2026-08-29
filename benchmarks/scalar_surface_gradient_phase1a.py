#!/usr/bin/env python3
"""Diagnose VMEX host-solve and fixed-point-refinement costs.

This Phase-1A harness is diagnostic only. It runs one method/configuration in
a fresh process, records private VMEX diagnostic events, and leaves production
defaults and public APIs unchanged. Use exact repeats to expose memo behavior
and nearby repeats to represent optimizer trial points.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
from pathlib import Path
import time
from typing import Any

import jax
import numpy as np

from vmex import optimize as opt
from vmex.core import implicit as imp
from vmex.core.input import VmecInput

from scalar_surface_gradient_phase0 import (
    DEFAULT_INPUT,
    REPO,
    SIMSOPT_REPO,
    _array_sha256,
    _git_revision,
    _peak_rss_bytes,
    _result_summary,
    _scalar_from_terms,
    _sha256,
    _terms,
    _timed,
    _version,
)


def _parse_batch(value: str) -> int | str | None:
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    if lowered in ("none", "full"):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return parsed


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _points(x0: np.ndarray, relative_step: float, pattern: str,
            nearby_order: str = "plus-minus"):
    if pattern == "exact":
        return [("seed", x0.copy()), ("seed_repeat_1", x0.copy()),
                ("seed_repeat_2", x0.copy())]
    index = np.arange(x0.size, dtype=float)
    direction = np.where((index.astype(int) % 2) == 0, 1.0, -1.0)
    direction /= np.linalg.norm(direction)
    delta = relative_step * np.maximum(np.abs(x0), 1.0e-2) * direction
    points = [
        ("seed", x0.copy()),
        ("plus", x0 + delta),
        ("plus_repeat", x0 + delta),
        ("minus", x0 - delta),
        ("minus_repeat", x0 - delta),
    ]
    if nearby_order == "minus-plus":
        points = [points[0], points[3], points[4], points[1], points[2]]
    return points


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    names = [entry["event"] for entry in events]
    successful_attempts = [
        entry for entry in events
        if entry["event"] == "host_solve_attempt" and entry["succeeded"]
    ]
    attempts = [
        entry for entry in events if entry["event"] == "host_solve_attempt"
    ]
    refinement = [
        entry for entry in events if entry["event"] == "refine_complete"
    ]
    callbacks = [
        entry for entry in events
        if entry["event"] == "host_callback_complete" and entry["succeeded"]
    ]
    steps = [entry for entry in events if entry["event"] == "refine_step"]
    warm_starts = [
        entry for entry in events if entry["event"] == "refine_warm_start"
    ]
    return {
        "event_counts": {
            name: names.count(name) for name in sorted(set(names))
        },
        "actual_host_solve_calls": len(attempts),
        "successful_host_solves": len(successful_attempts),
        "failed_host_solves": len(attempts) - len(successful_attempts),
        "host_solve_seconds": float(sum(
            float(entry["seconds"]) for entry in successful_attempts
        )),
        "host_solver_iterations": int(sum(
            int(entry["iterations"]) for entry in successful_attempts
        )),
        "host_callback_seconds": float(sum(
            float(entry["seconds"]) for entry in callbacks
        )),
        "refinement_invocations": names.count("refine_start"),
        "refinement_steps": len(steps),
        "refinement_seconds": float(sum(
            float(entry["seconds"]) for entry in refinement
        )),
        "refinement_krylov_iterations": int(sum(
            int(entry["krylov_iterations"]) for entry in steps
        )),
        "refinement_warm_starts_available": sum(
            bool(entry["available"]) for entry in warm_starts
        ),
        "refinement_warm_starts_accepted": sum(
            bool(entry["accepted"]) for entry in warm_starts
        ),
        "refinement_results": refinement,
    }


def _clear_problem_cache(problem) -> None:
    # Expose the VMEX solve/refinement memos rather than stopping at the
    # FunctionProblem exact-key cache.
    problem._vg_cache = None
    problem._rj_cache = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("residual", "jacobian", "scalar"), required=True)
    parser.add_argument("--pattern", choices=("exact", "nearby"), required=True)
    parser.add_argument(
        "--objective-profile", choices=("run71", "aspect", "iota"),
        default="run71",
        help="run71 includes QI; aspect is the smooth FD gate; iota probes a "
             "solver-sensitive scalar",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--refine-tol", type=float, default=1.0e-10)
    parser.add_argument("--refine-cross-point-warm-start", action="store_true")
    parser.add_argument(
        "--nearby-order", choices=("plus-minus", "minus-plus"),
        default="plus-minus",
    )
    parser.add_argument("--batch-size", type=_parse_batch, default="auto")
    parser.add_argument("--jacobian-adjoint-tol", type=float, default=1.0e-4)
    parser.add_argument("--jacobian-adjoint-maxiter", type=int, default=10)
    parser.add_argument("--adjoint-tol", type=float, default=1.0e-11)
    parser.add_argument("--adjoint-maxiter", type=int, default=300)
    parser.add_argument("--relative-step", type=float, default=1.0e-7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    if args.objective_profile == "run71":
        terms = _terms()
    elif args.objective_profile == "aspect":
        terms = _terms()[:1]
    else:
        terms = [(imp.iota_edge, 0.0, 1.0)]
    current_stage = ["build"]
    events: list[dict[str, Any]] = []

    def diagnostic_hook(event: str, payload: dict[str, Any]) -> None:
        events.append({
            "stage": current_stage[0],
            "event": event,
            **{key: _json_safe(value) for key, value in payload.items()},
        })

    imp._DIAGNOSTIC_HOOK = diagnostic_hook
    imp._REFINE_CROSS_POINT_WARM_START = args.refine_cross_point_warm_start

    original_make_config = imp.make_config

    def configured_make_config(*call_args, **call_kwargs):
        cfg = original_make_config(*call_args, **call_kwargs)
        return dataclasses.replace(cfg, refine_tol=float(args.refine_tol))

    imp.make_config = configured_make_config

    adjoint_dispatch = []
    original_adjoint = imp._adjoint_solve_gcrot

    def counted_adjoint(*call_args, **call_kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        started = time.perf_counter()
        result = original_adjoint(*call_args, **call_kwargs)
        adjoint_dispatch.append({
            "stage": current_stage[0],
            "caller": caller,
            "python_dispatch_seconds": time.perf_counter() - started,
        })
        return result

    imp._adjoint_solve_gcrot = counted_adjoint

    inp = VmecInput.from_file(input_path)
    kwargs = dict(
        max_mode=3,
        vary_major_radius=False,
        weight_semantics="residual",
        jacobian_batch_size=args.batch_size,
        implicit_jacobian_method="block_tridiagonal",
        jacobian_adjoint_tol=args.jacobian_adjoint_tol,
        jacobian_adjoint_maxiter=args.jacobian_adjoint_maxiter,
        adjoint_tol=args.adjoint_tol,
        adjoint_maxiter=args.adjoint_maxiter,
        warm_start="perturbation",
        use_ess=True,
        solve_kwargs={"mode": "cli", "lconm1": True, "use_fft": False},
        device="auto",
    )
    started = time.perf_counter()
    problem = opt.make_problem(
        inp,
        loss=_scalar_from_terms(terms) if args.method == "scalar" else None,
        objective_terms=None if args.method == "scalar" else terms,
        **kwargs,
    )
    build_seconds = time.perf_counter() - started
    cfg = problem.metadata["config"]
    if float(cfg.refine_tol) != float(args.refine_tol):
        raise RuntimeError("diagnostic refinement tolerance was not applied")

    calls = []
    for label, point in _points(
        np.asarray(problem.x0, dtype=float), args.relative_step, args.pattern,
        args.nearby_order,
    ):
        current_stage[0] = label
        _clear_problem_cache(problem)
        event_start = len(events)
        dispatch_start = len(adjoint_dispatch)
        if args.method == "residual":
            result, seconds = _timed(lambda p=point: problem.residual(p))
        elif args.method == "jacobian":
            result, seconds = _timed(
                lambda p=point: problem.residual_and_jac(p))
        else:
            result, seconds = _timed(lambda p=point: problem.value_and_grad(p))
        call_events = events[event_start:]
        components = _event_summary(call_events)
        components["outside_host_callback_seconds"] = max(
            0.0, seconds - components["host_callback_seconds"])
        calls.append({
            "point": label,
            "seconds": seconds,
            **_result_summary(args.method, result),
            "components": components,
            "adjoint_dispatch": adjoint_dispatch[dispatch_start:],
        })

    central_fd = None
    if args.pattern == "nearby" and args.method in ("jacobian", "scalar"):
        by_name = {entry["point"]: entry for entry in calls}
        nearby = dict(_points(
            np.asarray(problem.x0, dtype=float), args.relative_step, "nearby",
            args.nearby_order,
        ))
        delta = nearby["plus"] - np.asarray(problem.x0, dtype=float)
        gradient = np.asarray(by_name["seed"]["gradient"], dtype=float)
        fd_delta = 0.5 * (
            by_name["plus"]["objective"] - by_name["minus"]["objective"])
        predicted_delta = float(gradient @ delta)
        scale = max(abs(fd_delta), abs(predicted_delta), 1.0e-30)
        central_fd = {
            "half_difference": fd_delta,
            "gradient_dot_delta": predicted_delta,
            "relative_disagreement": abs(fd_delta - predicted_delta) / scale,
            "delta_norm": float(np.linalg.norm(delta)),
            "delta_sha256": _array_sha256(delta),
        }

    report = {
        "benchmark": "scalar_surface_gradient_phase1a",
        "method": args.method,
        "pattern": args.pattern,
        "objective_profile": args.objective_profile,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "configuration": {
            "refine_tol": _json_safe(float(args.refine_tol)),
            "refine_cross_point_warm_start": args.refine_cross_point_warm_start,
            "nearby_order": args.nearby_order,
            "requested_batch_size": args.batch_size,
            "effective_auto_dof_chunk": (
                int(opt._auto_jac_chunk(problem.x0.size))
                if args.batch_size == "auto" else None
            ),
            "jacobian_adjoint_tol": args.jacobian_adjoint_tol,
            "jacobian_adjoint_maxiter": args.jacobian_adjoint_maxiter,
            "adjoint_tol": args.adjoint_tol,
            "adjoint_maxiter": args.adjoint_maxiter,
            "relative_step": args.relative_step,
        },
        "dimensions": {
            "active_dofs": int(problem.x0.size),
            "residual_count": int(calls[0].get("residual_count", len(terms))),
        },
        "timing": {"build_seconds": build_seconds},
        "build_events": _event_summary([
            entry for entry in events if entry["stage"] == "build"
        ]),
        "calls": calls,
        "central_directional_fd": central_fd,
        "all_events": events,
        "peak_rss_bytes": _peak_rss_bytes(),
        "environment": {
            "jax": _version("jax"),
            "jaxlib": _version("jaxlib"),
            "vmex": _version("vmex"),
            "simsopt": _version("simsopt"),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "vmex_git": _git_revision(REPO),
            "simsopt_git": _git_revision(SIMSOPT_REPO),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
