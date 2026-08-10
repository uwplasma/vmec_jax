#!/usr/bin/env python
"""Compare exact and parallel-FD VMEX derivatives under one physics contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vmex import optimize as opt
from vmex.core.input import VmecInput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/data/input.solovev",
    )
    parser.add_argument("--max-mode", type=int, default=1)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    inp = VmecInput.from_file(args.input)
    terms = [(opt.aspect_ratio, 4.0, 1.0)]
    exact = opt.VmecProblem.from_tuples(inp, terms, max_mode=args.max_mode)
    finite = opt.VmecProblem.from_tuples(
        inp,
        terms,
        max_mode=args.max_mode,
        derivative_method="finite_difference",
        workers=args.workers,
    )

    # Compile and populate each lane before timing.  Mixing a cold exact JAX
    # trace with already-compiled FD forward solves is not a fair comparison.
    exact.residual_jac(exact.x0)
    finite.residual_jac(finite.x0)

    started = time.perf_counter()
    exact_jacobian = exact.residual_jac(exact.x0)
    exact_seconds = time.perf_counter() - started
    started = time.perf_counter()
    finite_jacobian = finite.residual_jac(finite.x0)
    finite_seconds = time.perf_counter() - started
    error = np.linalg.norm(exact_jacobian - finite_jacobian) / max(
        np.linalg.norm(exact_jacobian), 1.0
    )
    print(json.dumps({
        "input": str(args.input.resolve()),
        "max_mode": args.max_mode,
        "dofs": int(exact.x0.size),
        "finite_difference_probes": int(2 * exact.x0.size),
        "cache_state": "warm",
        "exact_seconds": exact_seconds,
        "finite_difference_seconds": finite_seconds,
        "relative_jacobian_error": float(error),
        "workers": args.workers if args.workers is not None else "auto",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
