#!/usr/bin/env python
"""Profile one optimizer-neutral VMEX problem in a fresh process.

The selectable QI, QA, QH, QP, and scalar cases exercise the public SciPy and
JAX contracts without committing machine-specific timings to the repository.
Use separate processes when comparing cold compilation policies.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib import metadata
import json
import platform
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.optimize

import vmex
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual

from _provenance import assert_repo_vmex, file_sha256, git_state


REPO = Path(__file__).resolve().parents[1]


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def terms(case: str):
    surfaces = np.linspace(0.1, 1.0, 6)

    def iota_floor(state, runtime):
        return jnp.maximum(0.33 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

    constraints = [(opt.aspect_ratio, 6.0, 0.01), (iota_floor, 0.0, 10.0)]
    if case == "qi":
        return [(QIResidual(surfaces), 0.0, 1.0), *constraints]
    if case in ("qa", "qh", "qp"):
        helicity = {"qa": (1, 0), "qh": (1, -1), "qp": (0, 1)}[case]
        return [(opt.QuasisymmetryRatioResidual(surfaces, *helicity), 0.0, 1.0), *constraints]
    return [
        (opt.aspect_ratio, 6.0, 1.0),
        (opt.mirror_ratio, 0.2, 1.0),
        (opt.magnetic_well, 0.05, 1.0),
        (iota_floor, 0.0, 10.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("qi", "qa", "qh", "qp", "scalar"), default="qi")
    parser.add_argument("--nfp", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--max-mode", type=int, default=1)
    parser.add_argument("--derivatives", choices=("implicit", "finite_difference"), default="implicit")
    parser.add_argument("--optimizer", choices=("none", "least_squares", "BFGS", "L-BFGS-B"), default="none")
    parser.add_argument("--nfev", type=int, default=2)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--forward-ftol", type=float)
    parser.add_argument("--forward-max-iterations", type=int)
    parser.add_argument("--max-fsq-ratio", type=float, default=1.0e6)
    args = parser.parse_args()

    path = args.input or REPO / f"examples/data/input.minimal_seed_nfp{min(args.nfp, 4)}"
    inp = VmecInput.from_file(path)
    if args.input is None and inp.nfp != args.nfp:
        inp = replace(inp, nfp=args.nfp)
    mpol = max(args.max_mode + 2, 5)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4,
    )
    started = time.perf_counter()
    problem = opt.VmecProblem.from_tuples(
        inp, terms(args.case), max_mode=args.max_mode,
        derivative_method=args.derivatives, workers=args.workers,
        forward_ftol=args.forward_ftol,
        forward_max_iterations=args.forward_max_iterations,
        max_fsq_ratio=args.max_fsq_ratio, use_ess=True,
    )
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    value, gradient = problem.value_and_grad(problem.x0)
    derivative_seconds = time.perf_counter() - started
    contract = {"host_value": value, "host_gradient_norm": float(np.linalg.norm(gradient))}
    if args.derivatives == "implicit":
        started = time.perf_counter()
        jax_value, jax_gradient = jax.device_get(
            problem.jax_value_and_grad(jnp.asarray(problem.x0))
        )
        contract.update(
            jax_seconds=time.perf_counter() - started,
            value_relative_error=float(abs(jax_value - value) / max(abs(value), 1.0)),
            gradient_relative_error=float(
                np.linalg.norm(jax_gradient - gradient) / max(np.linalg.norm(gradient), 1.0)
            ),
        )
        started = time.perf_counter()
        graph_value, graph_gradient = jax.device_get(
            jax.value_and_grad(problem.jax_fun)(jnp.asarray(problem.x0))
        )
        contract.update(
            differentiated_graph_seconds=time.perf_counter() - started,
            differentiated_graph_value_relative_error=float(
                abs(graph_value - value) / max(abs(value), 1.0)
            ),
            differentiated_graph_gradient_relative_error=float(
                np.linalg.norm(graph_gradient - gradient) / max(np.linalg.norm(gradient), 1.0)
            ),
        )

    x = problem.x0
    started = time.perf_counter()
    if args.optimizer == "least_squares":
        result = scipy.optimize.least_squares(
            problem.residual, x, jac=problem.residual_jac, x_scale=problem.scales,
            max_nfev=args.nfev,
        )
        x = result.x
    elif args.optimizer != "none":
        result = scipy.optimize.minimize(
            problem.value_and_grad, x, jac=True, method=args.optimizer,
            options={"maxiter": args.nfev},
        )
        x = result.x
    optimize_seconds = time.perf_counter() - started
    evaluation = problem.evaluate(x, derivatives=False)

    report = {
        "case": args.case,
        "nfp": int(inp.nfp),
        "max_mode": args.max_mode,
        "dofs": int(problem.x0.size),
        "derivatives": args.derivatives,
        "optimizer": args.optimizer,
        "forward_ftol": problem.metadata["forward_ftol"],
        "forward_max_iterations": problem.metadata["forward_max_iterations"],
        "build_seconds": build_seconds,
        "derivative_seconds": derivative_seconds,
        "optimize_seconds": optimize_seconds,
        "initial_cost": value,
        "final_cost": float(problem.fun(x)),
        "contract": contract,
        "diagnostics": dict(evaluation.diagnostics),
        "input_sha256": file_sha256(path),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(), "vmex": vmex.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "jax": jax.__version__, "jaxopt": version("jaxopt"), "optax": version("optax"),
        },
        **git_state(REPO),
        "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
