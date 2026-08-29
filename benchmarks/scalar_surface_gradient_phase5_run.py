#!/usr/bin/env python3
"""Run one isolated Phase-5 production or finite-beta replay.

This wrapper reuses the reviewed Phase-4 runners while controlling the VMEX
fixed-point refinement tolerance in the fresh process. It is benchmark-only:
no production default or public API is changed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parents[1]
DRIVER_OUTPUT = ROOT / "SVD" / "single_stage_vacuum_jax" / "output"
PHASE4_PRODUCTION = REPO / "benchmarks" / "scalar_surface_gradient_phase4.py"
PHASE4_FINITE_BETA = (
    REPO / "benchmarks" / "scalar_surface_gradient_phase4_finite_beta.py"
)

POLICIES = {
    "strict_no_reuse": {"mode": "legacy", "refine_tol": 1.0e-10},
    "strict_reuse": {"mode": "new", "refine_tol": 1.0e-10},
    "relaxed_reuse": {"mode": "new", "refine_tol": 1.0e-8},
}


def _rewrite_json(path: Path, additions: dict) -> None:
    payload = json.loads(path.read_text())
    payload.update(additions)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("production", "finite_beta"), required=True)
    parser.add_argument("--policy", choices=tuple(POLICIES), required=True)
    parser.add_argument("--output-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maxiter", type=int, default=3)
    args = parser.parse_args()

    policy = POLICIES[args.policy]
    mode = policy["mode"]
    refine_tol = float(policy["refine_tol"])
    if args.kind == "production" and not args.output_name:
        parser.error("--output-name is required for production")
    if args.kind == "finite_beta" and args.output is None:
        parser.error("--output is required for finite_beta")

    import vmex.core.implicit as imp

    original_make_config = imp.make_config
    original_argv = sys.argv[:]

    def configured_make_config(*call_args, **call_kwargs):
        cfg = original_make_config(*call_args, **call_kwargs)
        return dataclasses.replace(cfg, refine_tol=refine_tol)

    imp.make_config = configured_make_config
    try:
        if args.kind == "production":
            sys.argv = [
                str(PHASE4_PRODUCTION),
                "--workflow", "single_stage",
                "--mode", mode,
                "--maxiter", str(args.maxiter),
                "--output-name", str(args.output_name),
            ]
            runpy.run_path(str(PHASE4_PRODUCTION), run_name="__main__")
            summary = DRIVER_OUTPUT / args.output_name / "phase4_replay_summary.json"
            _rewrite_json(
                summary,
                {
                    "phase5_policy": args.policy,
                    "refine_tol": refine_tol,
                    "jax_compilation_cache_dir": os.environ.get(
                        "JAX_COMPILATION_CACHE_DIR"
                    ),
                },
            )
        else:
            output = args.output.resolve()
            output.mkdir(parents=True, exist_ok=True)
            sys.argv = [
                str(PHASE4_FINITE_BETA),
                "--mode", mode,
                "--output", str(output),
            ]
            runpy.run_path(str(PHASE4_FINITE_BETA), run_name="__main__")
            summary = output / f"finite_beta_construction_{mode}.json"
            _rewrite_json(
                summary,
                {
                    "phase5_policy": args.policy,
                    "refine_tol": refine_tol,
                    "jax_compilation_cache_dir": os.environ.get(
                        "JAX_COMPILATION_CACHE_DIR"
                    ),
                },
            )
    finally:
        imp.make_config = original_make_config
        sys.argv = original_argv


if __name__ == "__main__":
    main()
