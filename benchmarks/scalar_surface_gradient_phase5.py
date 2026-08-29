#!/usr/bin/env python3
"""Orchestrate the isolated-cache Phase-5 production study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parents[1]
DRIVER_OUTPUT = ROOT / "SVD" / "single_stage_vacuum_jax" / "output"
DEFAULT_OUTPUT = ROOT / "external" / "SIMSOPT_VMEX_SCALAR_GRADIENT_PHASE5_ARTIFACTS"
EVALUATOR = REPO / "benchmarks" / "scalar_surface_gradient_phase1a.py"
RUNNER = REPO / "benchmarks" / "scalar_surface_gradient_phase5_run.py"
SUMMARY = REPO / "benchmarks" / "summarize_scalar_surface_gradient_phase5.py"

DEFAULT_SEQUENCE = (
    "strict_no_reuse",
    "strict_reuse",
    "relaxed_reuse",
    "relaxed_reuse",
    "strict_reuse",
    "strict_no_reuse",
)
POLICIES = {
    "strict_no_reuse": {"reuse": False, "refine_tol": 1.0e-10},
    "strict_reuse": {"reuse": True, "refine_tol": 1.0e-10},
    "relaxed_reuse": {"reuse": True, "refine_tol": 1.0e-8},
}


def _parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _run(command: list[str], env: dict[str, str], log_path: Path) -> float:
    started = time.perf_counter()
    with log_path.open("w") as stream:
        subprocess.run(
            command,
            check=True,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument(
        "--components",
        type=_parse_csv,
        default=("evaluation", "production", "finite_beta"),
    )
    parser.add_argument("--sequence", type=_parse_csv, default=DEFAULT_SEQUENCE)
    args = parser.parse_args()
    if args.maxiter < 1:
        raise ValueError("--maxiter must be positive")
    unknown_components = set(args.components) - {
        "evaluation", "production", "finite_beta"
    }
    unknown_policies = set(args.sequence) - set(POLICIES)
    if unknown_components:
        raise ValueError(f"unknown components: {sorted(unknown_components)}")
    if unknown_policies:
        raise ValueError(f"unknown policies: {sorted(unknown_policies)}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = output / "isolated_caches"
    cache_root.mkdir(exist_ok=True)
    manifest_path = output / "phase5_manifest.json"
    manifest = {
        "python": sys.executable,
        "provenance": {
            "vmex_git_head": _git_head(REPO),
            "simsopt_git_head": _git_head(ROOT / "external" / "simsopt_latest_vmex"),
            "vacuum_driver_git_head": _git_head(
                ROOT / "SVD" / "single_stage_vacuum_jax"
            ),
        },
        "sequence": list(args.sequence),
        "components": list(args.components),
        "maxiter": args.maxiter,
        "cache_protocol": "fresh isolated JAX and XDG cache per process",
        "runs": [],
    }
    _write_manifest(manifest_path, manifest)

    for component in args.components:
        for position, policy_name in enumerate(args.sequence, start=1):
            policy = POLICIES[policy_name]
            run_id = f"{component}_{position:02d}_{policy_name}"
            cache_dir = cache_root / run_id
            if cache_dir.exists() and any(cache_dir.iterdir()):
                raise FileExistsError(f"isolated cache is not empty: {cache_dir}")
            cache_dir.mkdir(exist_ok=True)
            xdg_cache = cache_dir / "xdg"
            jax_cache = cache_dir / "jax"
            mpl_cache = cache_dir / "mpl"
            env = os.environ.copy()
            env.update(
                JAX_COMPILATION_CACHE_DIR=str(jax_cache),
                XDG_CACHE_HOME=str(xdg_cache),
                MPLCONFIGDIR=str(mpl_cache),
                PYTHONHASHSEED="0",
            )
            log_path = output / f"{run_id}.log"

            if component == "evaluation":
                result_path = output / "evaluation" / f"{run_id}.json"
                result_path.parent.mkdir(exist_ok=True)
                command = [
                    sys.executable,
                    str(EVALUATOR),
                    "--method", "scalar",
                    "--pattern", "nearby",
                    "--objective-profile", "run71",
                    "--batch-size", "auto",
                    "--refine-tol", str(policy["refine_tol"]),
                    "--output", str(result_path),
                ]
                if policy["reuse"]:
                    command.append("--refine-cross-point-warm-start")
            elif component == "production":
                output_name = f"phase5_scalar_gradient_{position:02d}_{policy_name}"
                result_path = DRIVER_OUTPUT / output_name
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--kind", "production",
                    "--policy", policy_name,
                    "--output-name", output_name,
                    "--maxiter", str(args.maxiter),
                ]
            else:
                result_path = output / "finite_beta" / run_id
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--kind", "finite_beta",
                    "--policy", policy_name,
                    "--output", str(result_path),
                ]

            record = {
                "component": component,
                "position": position,
                "policy": policy_name,
                "refine_tol": policy["refine_tol"],
                "cross_point_warm_start": policy["reuse"],
                "cache_dir": str(cache_dir),
                "result": str(result_path),
                "log": str(log_path),
                "command": command,
                "status": "running",
            }
            manifest["runs"].append(record)
            _write_manifest(manifest_path, manifest)
            try:
                record["orchestrator_wall_seconds"] = _run(command, env, log_path)
                record["status"] = "complete"
            except subprocess.CalledProcessError as exc:
                record["status"] = "failed"
                record["returncode"] = exc.returncode
                _write_manifest(manifest_path, manifest)
                raise
            _write_manifest(manifest_path, manifest)

    subprocess.run(
        [sys.executable, str(SUMMARY), "--input", str(output)], check=True
    )


if __name__ == "__main__":
    main()
