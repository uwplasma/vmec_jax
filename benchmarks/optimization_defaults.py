#!/usr/bin/env python
"""Measure one cold QI/QS Jacobian-policy case.

Run each case in a fresh process with ``VMEX_COMPILATION_CACHE=disabled`` so
the reported construction and first-derivative times do not inherit compiled
executables from another row.  This is a benchmark, not a wall-clock CI test.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np
import scipy.optimize

from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "examples/data/input.minimal_seed_nfp2",
    )
    parser.add_argument("--case", choices=("qi", "qs", "mixed"), required=True)
    parser.add_argument("--max-mode", type=int, choices=(1, 3, 5), required=True)
    parser.add_argument("--batch", choices=("1", "auto"), required=True)
    parser.add_argument("--nfev", type=int, default=2)
    args = parser.parse_args()

    inp = VmecInput.from_file(args.input)
    mpol = max(args.max_mode + 2, 5)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol,
        ntor=mpol,
        ntheta=2 * mpol + 6,
        nzeta=2 * mpol + 4,
    )

    def iota_floor(state, runtime):
        return jnp.maximum(
            0.33 - jnp.abs(opt.mean_iota(state, runtime)), 0.0
        )

    common = [
        (opt.aspect_ratio, 4.0, 0.005),
        (iota_floor, 0.0, 10.0),
        (opt.mirror_ratio, 0.21, 1.0),
    ]
    if args.case == "qi":
        terms = [(QIResidual(np.linspace(0.1, 1.0, 6)), 0.0, 1.0), *common]
    elif args.case == "qs":
        qs = opt.QuasisymmetryRatioResidual(
            np.linspace(0.1, 1.0, 6), helicity_m=1, helicity_n=0
        )
        terms = [(qs, 0.0, 1.0), *common]
    else:
        terms = [
            (opt.aspect_ratio, 4.0, 1.0),
            (opt.mirror_ratio, 0.21, 1.0),
            (opt.magnetic_well, 0.05, 1.0),
            (iota_floor, 0.0, 10.0),
        ]

    started = time.perf_counter()
    problem = opt.VmecProblem.from_tuples(
        inp,
        terms,
        max_mode=args.max_mode,
        jacobian_batch_size=1 if args.batch == "1" else "auto",
        use_ess=True,
    )
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    compiled = problem.compile_residual_and_jacobian(progress=False)
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = scipy.optimize.least_squares(
        problem.residual,
        problem.x0,
        jac=problem.residual_jac,
        x_scale=problem.scales,
        max_nfev=args.nfev,
        ftol=1.0e-6,
        xtol=1.0e-10,
    )
    optimize_seconds = time.perf_counter() - started
    print(json.dumps({
        "case": args.case,
        "max_mode": args.max_mode,
        "batch": args.batch,
        "dofs": int(problem.x0.size),
        "residuals": int(compiled.residual.size),
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "optimize_seconds": optimize_seconds,
        "initial_cost": float(compiled.value),
        "final_cost": float(result.cost),
        "nfev": int(result.nfev),
        "failed_trials": int(problem.metadata["holder"]["failed_trials"]),
        "derivative_fallbacks": int(
            problem.metadata["holder"]["derivative_fallbacks"]
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
