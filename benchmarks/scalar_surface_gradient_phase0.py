#!/usr/bin/env python3
"""Profile run-71-scale scalar and residual-Jacobian surface derivatives.

Run each configuration in a fresh process. The scalar loss and explicit
residual problem are generated from one shared objective-term list so their
mathematics cannot drift. Timings distinguish construction, the first
compile-and-execute derivative call, and two warm executions at deterministic
nearby surfaces. Process peak RSS is recorded by ``resource.getrusage``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
from importlib import metadata
import json
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from vmex import optimize as opt
from vmex.core import implicit as imp
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual


REPO = Path(__file__).resolve().parents[1]
SIMSOPT_REPO = REPO.parent / "simsopt_latest_vmex"
DEFAULT_INPUT = REPO / "benchmarks" / "input.run71_maxmode3"


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_revision(path: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {"revision": revision, "branch": branch, "dirty": dirty}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if platform.system() == "Darwin" else rss * 1024


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


def _terms() -> list[tuple[Callable, float, float]]:
    qi = ConstructedQIResidual(
        surfaces=np.asarray([0.0625, 0.3125, 0.5625, 0.8125]),
        mboz=18,
        nboz=18,
        oversample=2,
        nphi=141,
        nalpha=27,
        n_bounce=51,
        include_bounce_endpoints=True,
        softness=0.02,
        width_weight=1.0,
        branch_width_weight=0.5,
        branch_width_softness=0.02,
        profile_weight=0.1,
        shuffle_profile_weight=1.0,
        shuffle_profile_softness=0.02,
    )

    def mirror_excess(state, runtime):
        return jnp.atleast_1d(jnp.maximum(
            opt.mirror_ratio(state, runtime) - 0.21, 0.0))

    def elongation_excess(state, runtime):
        return jnp.atleast_1d(jnp.maximum(
            opt.max_elongation(state, runtime) - 6.0, 0.0))

    # These are residual multipliers. They exactly match the run-71 wrapper,
    # which passed sqrt(objective_weight) with weight_semantics="residual".
    return [
        (opt.aspect_ratio, 7.0, 1.0),
        (qi.residuals_state, 0.0, 1.0),
        (mirror_excess, 0.0, 10.0),
        (elongation_excess, 0.0, 10.0),
    ]


def _scalar_from_terms(terms):
    """Return 0.5 ||r||^2 using the exact shared residual definitions."""
    frozen = tuple(terms)

    def scalar(state, runtime):
        rows = jnp.concatenate([
            jnp.atleast_1d(
                weight * (jnp.asarray(function(state, runtime)) - target)
            ).ravel()
            for function, target, weight in frozen
        ])
        return 0.5 * jnp.vdot(rows, rows)

    return scalar


def _timed(call):
    started = time.perf_counter()
    result = call()
    for leaf in jax.tree.leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result, time.perf_counter() - started


def _points(x0: np.ndarray, relative_step: float) -> list[np.ndarray]:
    index = np.arange(x0.size, dtype=float)
    direction = np.where((index.astype(int) % 2) == 0, 1.0, -1.0)
    direction /= np.linalg.norm(direction)
    delta = relative_step * np.maximum(np.abs(x0), 1.0e-2) * direction
    return [x0.copy(), x0 + delta, x0 - delta]


def _result_summary(method: str, result: Any) -> dict[str, Any]:
    if method == "residual":
        residual = np.asarray(result, dtype=float).ravel()
        return {
            "residual_count": int(residual.size),
            "residual_norm": float(np.linalg.norm(residual)),
            "objective": 0.5 * float(residual @ residual),
            "residual_sha256": _array_sha256(residual),
        }
    if method == "jacobian":
        residual, jacobian = map(np.asarray, result)
        residual = np.asarray(residual, dtype=float).ravel()
        jacobian = np.asarray(jacobian, dtype=float)
        gradient = jacobian.T @ residual
        return {
            "residual_count": int(residual.size),
            "residual_norm": float(np.linalg.norm(residual)),
            "residual_sha256": _array_sha256(residual),
            "jacobian_shape": list(jacobian.shape),
            "jacobian_norm": float(np.linalg.norm(jacobian)),
            "jacobian_sha256": _array_sha256(jacobian),
            "objective": 0.5 * float(residual @ residual),
            "gradient": gradient.tolist(),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "gradient_sha256": _array_sha256(gradient),
        }
    value, gradient = result
    gradient = np.asarray(gradient, dtype=float)
    return {
        "objective": float(value),
        "gradient": gradient.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "gradient_sha256": _array_sha256(gradient),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("residual", "jacobian", "scalar"), required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=_parse_batch, default="auto")
    parser.add_argument("--jacobian-adjoint-tol", type=float, default=1.0e-4)
    parser.add_argument("--jacobian-adjoint-maxiter", type=int, default=10)
    parser.add_argument("--adjoint-tol", type=float, default=1.0e-10)
    parser.add_argument("--adjoint-maxiter", type=int, default=300)
    parser.add_argument("--relative-step", type=float, default=1.0e-7)
    parser.add_argument("--point-count", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    terms = _terms()
    counters = {
        "host_solve_callbacks": 0,
        "adjoint_gcrot_calls": 0,
        "adjoint_gcrot_seconds": 0.0,
        "adjoint_iterations": [],
        "adjoint_gcrot_by_caller": {},
    }
    original_host = imp._host_solve_and_mask_impl
    original_adjoint = imp._adjoint_solve_gcrot

    def counted_host(*call_args, **call_kwargs):
        counters["host_solve_callbacks"] += 1
        return original_host(*call_args, **call_kwargs)

    def counted_adjoint(*call_args, **call_kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        started = time.perf_counter()
        output = original_adjoint(*call_args, **call_kwargs)
        elapsed = time.perf_counter() - started
        counters["adjoint_gcrot_calls"] += 1
        counters["adjoint_gcrot_seconds"] += elapsed
        report = output[1]
        try:
            iterations = int(np.max(np.asarray(report.iterations)))
        except (AttributeError, TypeError, ValueError):
            iterations = None
        counters["adjoint_iterations"].append(iterations)
        entry = counters["adjoint_gcrot_by_caller"].setdefault(
            caller, {"calls": 0, "seconds": 0.0, "iterations": []})
        entry["calls"] += 1
        entry["seconds"] += elapsed
        entry["iterations"].append(iterations)
        return output

    imp._host_solve_and_mask_impl = counted_host
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

    x_points = _points(
        np.asarray(problem.x0, dtype=float), args.relative_step
    )[:args.point_count]
    calls = []
    seed_summary = None
    for index, point in enumerate(x_points):
        if args.method == "residual":
            result, seconds = _timed(lambda p=point: problem.residual(p))
        elif args.method == "jacobian":
            result, seconds = _timed(lambda p=point: problem.residual_and_jac(p))
        else:
            result, seconds = _timed(lambda p=point: problem.value_and_grad(p))
        summary = _result_summary(args.method, result)
        calls.append({"point": ("seed", "plus", "minus")[index], "seconds": seconds,
                      **summary})
        if index == 0:
            seed_summary = summary

    holder = dict(problem.metadata.get("holder", {}))
    warm_seconds = [call["seconds"] for call in calls[1:]]
    safe_holder = {
        key: value for key, value in holder.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    report = {
        "benchmark": "scalar_surface_gradient_phase0",
        "method": args.method,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "objective": {
            "definition": "run71_legacy_four_block",
            "max_mode": 3,
            "term_count": len(terms),
            "weight_semantics": "residual",
        },
        "configuration": {
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
            "expected_qi_residual_count": 41472,
            "expected_total_residual_count": 41475,
            "residual_count": (
                int(seed_summary["residual_count"])
                if seed_summary is not None and "residual_count" in seed_summary
                else 41475
            ),
        },
        "timing": {
            "build_seconds": build_seconds,
            "cold_compile_and_first_execution_seconds": calls[0]["seconds"],
            "warm_execution_seconds": warm_seconds,
            "warm_execution_median_seconds": (
                float(np.median(warm_seconds)) if warm_seconds else None
            ),
        },
        "calls": calls,
        "counters": counters,
        "problem_holder": safe_holder,
        "peak_rss_bytes": _peak_rss_bytes(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
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
