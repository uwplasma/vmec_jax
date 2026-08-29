#!/usr/bin/env python3
"""Run the reduced Phase 4 finite-beta construction smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SIMSOPT = ROOT / "external" / "simsopt_latest_vmex"
TEST_INPUT = SIMSOPT / "tests" / "test_files" / "input.li383_low_res"
DEFAULT_OUTPUT = ROOT / "external" / "SIMSOPT_VMEX_SCALAR_GRADIENT_PHASE4_ARTIFACTS"


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


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    names = [entry["event"] for entry in events]
    attempts = [entry for entry in events if entry["event"] == "host_solve_attempt"]
    successful = [entry for entry in attempts if entry["succeeded"]]
    refinement = [entry for entry in events if entry["event"] == "refine_complete"]
    warm = [entry for entry in events if entry["event"] == "refine_warm_start"]
    fallback = [
        entry for entry in events if entry["event"] == "refine_warm_start_fallback"
    ]
    return {
        "event_counts": {name: names.count(name) for name in sorted(set(names))},
        "actual_host_solve_calls": len(attempts),
        "successful_host_solves": len(successful),
        "failed_host_solves": len(attempts) - len(successful),
        "host_solve_seconds": float(sum(float(item["seconds"]) for item in successful)),
        "host_solver_iterations": int(sum(int(item["iterations"]) for item in successful)),
        "refinement_invocations": names.count("refine_start"),
        "refinement_steps": names.count("refine_step"),
        "refinement_seconds": float(sum(float(item["seconds"]) for item in refinement)),
        "warm_starts_available": sum(bool(item["available"]) for item in warm),
        "warm_starts_accepted": sum(bool(item["accepted"]) for item in warm),
        "warm_start_fallbacks": len(fallback),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "new"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-cache")

    import vmex
    import vmex.core.implicit as implicit
    import virtual_casing_jax
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.mhd import Vmex, VmexSurfaceProblem, VmexVirtualCasingTarget
    from simsopt.objectives import SurfaceSquaredFlux

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"finite_beta_construction_{args.mode}"
    json_path = args.output / f"{stem}.json"
    arrays_path = args.output / f"{stem}.npz"
    if json_path.exists() or arrays_path.exists():
        raise FileExistsError(f"refusing to replace existing Phase 4 artifact {stem}")

    events: list[dict[str, Any]] = []
    started = time.perf_counter()

    def diagnostic_hook(event: str, payload: dict[str, Any]) -> None:
        events.append({
            "event": event,
            "elapsed_seconds": time.perf_counter() - started,
            **_json_safe(payload),
        })

    previous_hook = implicit._DIAGNOSTIC_HOOK
    previous_cross_point = implicit._REFINE_CROSS_POINT_WARM_START
    timings: dict[str, float] = {}
    try:
        implicit._DIAGNOSTIC_HOOK = diagnostic_hook
        implicit._REFINE_CROSS_POINT_WARM_START = args.mode == "new"

        tick = time.perf_counter()
        equilibrium = Vmex(
            str(TEST_INPUT),
            verbose=False,
            nphi=8,
            ntheta=8,
            range_surface="full torus",
        )
        equilibrium.boundary.fix_all()
        equilibrium.boundary.unfix("rc(1,0)")
        equilibrium.boundary.unfix("zs(1,0)")
        timings["equilibrium_adapter_construction_seconds"] = time.perf_counter() - tick

        tick = time.perf_counter()
        problem = VmexSurfaceProblem(
            equilibrium,
            [(vmex.optimize.aspect_ratio, 5.0, 1.0)],
            max_mode=1,
            jac_chunk_size=1,
            jac_solver="reverse",
            warm_start="state",
            solve_kwargs={"mode": "cli", "lconm1": True, "use_fft": False},
            adjoint_tol=1.0e-9,
        )
        timings["surface_problem_construction_seconds"] = time.perf_counter() - tick

        x0 = np.asarray(problem.x0, dtype=float)
        tick = time.perf_counter()
        residual = np.asarray(problem.residuals(x0), dtype=float)
        timings["initial_vmex_residual_seconds"] = time.perf_counter() - tick

        tick = time.perf_counter()
        provider = VmexVirtualCasingTarget(
            problem,
            source_nphi=8,
            source_ntheta=8,
            digits=2,
            chunk_size=64,
        )
        timings["virtual_casing_provider_construction_seconds"] = time.perf_counter() - tick

        tick = time.perf_counter()
        curves = create_equally_spaced_curves(
            2,
            equilibrium.boundary.nfp,
            stellsym=equilibrium.boundary.stellsym,
            R0=1.4,
            R1=0.7,
            order=3,
        )
        currents = [Current(1.0e5) for _ in curves]
        for curve in curves:
            curve.fix_all()
        for current in currents:
            current.fix_all()
        field = BiotSavart(
            coils_via_symmetries(
                curves,
                currents,
                equilibrium.boundary.nfp,
                equilibrium.boundary.stellsym,
            )
        )
        objective = SurfaceSquaredFlux(
            equilibrium.boundary,
            field,
            target_provider=provider,
            definition="normalized",
        )
        timings["field_and_objective_construction_seconds"] = time.perf_counter() - tick

        tick = time.perf_counter()
        objective_value = float(objective.J())
        timings["squared_flux_value_seconds"] = time.perf_counter() - tick

        tick = time.perf_counter()
        derivative = objective.dJ(partials=True)
        surface_gradient = np.asarray(
            derivative(equilibrium.boundary), dtype=float
        ).reshape(-1)
        timings["squared_flux_surface_gradient_seconds"] = time.perf_counter() - tick

        target = np.asarray(provider.target(x0), dtype=float)
        np.savez_compressed(
            arrays_path,
            surface_x=x0,
            surface_residual=residual,
            virtual_casing_target=target,
            squared_flux_surface_gradient=surface_gradient,
        )
    finally:
        implicit._DIAGNOSTIC_HOOK = previous_hook
        implicit._REFINE_CROSS_POINT_WARM_START = previous_cross_point

    payload = {
        "mode": args.mode,
        "cross_point_warm_start": args.mode == "new",
        "input": str(TEST_INPUT),
        "versions": {
            "vmex": getattr(vmex, "__version__", "unknown"),
            "virtual_casing_jax": getattr(virtual_casing_jax, "__version__", "unknown"),
        },
        "resolution": {
            "target_nphi": 8,
            "target_ntheta": 8,
            "source_nphi": 8,
            "source_ntheta": 8,
            "digits": 2,
        },
        "timings": timings,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "objective_value": objective_value,
        "surface_gradient_norm": float(np.linalg.norm(surface_gradient)),
        "provider_diagnostics": provider.diagnostics,
        "objective_derivative_components": objective.last_derivative_components,
        "event_summary": _event_summary(events),
        "event_log": events,
        "arrays_file": str(arrays_path),
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")


if __name__ == "__main__":
    main()
