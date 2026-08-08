#!/usr/bin/env python
"""Reproducible alex_qi derivative and external-optimizer acceptance run."""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy
import scipy.optimize

from vmex import OptimizationMonitor, __version__
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual


def installed_version(name: str) -> str | None:
    """Return an installed distribution version without importing the package."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_problem(path: Path, max_mode: int) -> opt.VmecProblem:
    """Reproduce the objective tuple from alex_qi/QI_opt_vmex.py."""
    inp = VmecInput.from_file(path)
    qi = QIResidual(np.linspace(0.1, 1.0, 6))

    def iota_floor(state, runtime):
        return jnp.maximum(0.33 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

    return opt.VmecProblem.from_tuples(
        inp,
        [
            (opt.aspect_ratio, 4.0, 0.005),
            (qi, 0.0, 1.0),
            (iota_floor, 0.0, 10.0),
            (opt.mirror_ratio, 0.21, 1.0),
        ],
        max_mode=max_mode,
        derivative_method="implicit",
        weight_semantics="cost",
        use_ess=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/rogeriojorge/local/alex_qi/input.nfp2_circular"),
    )
    parser.add_argument("--max-mode", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        choices=("none", "least_squares", "BFGS", "L-BFGS-B", "jaxopt", "optax"),
        default="none",
    )
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()

    started = time.perf_counter()
    problem = build_problem(args.input, args.max_mode)
    value, gradient = problem.value_and_grad(problem.x0)
    derivative_seconds = time.perf_counter() - started
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("initial QI value or gradient is non-finite")

    x = problem.x0
    monitor = OptimizationMonitor(problem)
    optimize_started = time.perf_counter()
    if args.optimizer == "least_squares":
        result = scipy.optimize.least_squares(
            problem.residual,
            x,
            jac=problem.residual_jac,
            x_scale=problem.scales,
            callback=monitor,
            max_nfev=args.iterations,
        )
        x = result.x
    elif args.optimizer in ("BFGS", "L-BFGS-B"):
        result = scipy.optimize.minimize(
            problem.value_and_grad,
            x,
            jac=True,
            method=args.optimizer,
            callback=monitor,
            options={"maxiter": args.iterations},
        )
        x = result.x
    elif args.optimizer == "jaxopt":
        import jaxopt

        x = jaxopt.LBFGS(
            problem.jax_value_and_grad,
            value_and_grad=True,
            maxiter=args.iterations,
            maxls=10,
            jit=False,
        ).run(jnp.asarray(x)).params
    elif args.optimizer == "optax":
        import optax

        transform = optax.adam(1.0e-2)
        x = jnp.asarray(x)
        state = transform.init(x)
        for iteration in range(args.iterations):
            cost, grad = problem.jax_value_and_grad(x)
            updates, state = transform.update(grad, state, x)
            x = optax.apply_updates(x, updates)
            monitor.record(x, cost=float(cost), iteration=iteration)

    report = {
        "input": str(args.input.resolve()),
        "max_mode": args.max_mode,
        "dofs": int(problem.x0.size),
        "optimizer": args.optimizer,
        "initial_cost": float(value),
        "final_cost": float(problem.fun(x)),
        "finite_gradient": True,
        "derivative_seconds": derivative_seconds,
        "optimization_seconds": time.perf_counter() - optimize_started,
        "failed_trials": int(problem.metadata["holder"]["failed_trials"]),
        "derivative_fallbacks": int(
            problem.metadata["holder"]["derivative_fallbacks"]
        ),
        "versions": {
            "python": platform.python_version(),
            "vmex": __version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "jaxopt": installed_version("jaxopt"),
            "optax": installed_version("optax"),
            "simsopt": installed_version("simsopt"),
            "desc-opt": installed_version("desc-opt"),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
