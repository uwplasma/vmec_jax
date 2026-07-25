#!/usr/bin/env python
"""Fresh-process resource profiles for implicit AD and mirror scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _peak_rss_gib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
    return peak_bytes / 2**30


def _implicit(args) -> dict:
    from vmex.core import optimize as opt
    from vmex.core.input import VmecInput

    if args.input is None:
        raise ValueError("--input is required for the implicit case")
    inp = VmecInput.from_file(args.input)
    terms = ([(opt.aspect_ratio, args.aspect_target, 1.0)]
             if args.implicit_objective == "aspect" else
             [(opt.mercier_stability_residual, 0.0, 1.0),
              (opt.glasser_stability_residual, 0.0, 1.0),
              (opt.jdotb_residual, 0.0, 1.0e-6)])
    if args.optimizer == "minimize":
        x0 = opt.pack_boundary(inp, args.max_mode)
        result = opt.minimize(
            terms, inp, max_mode=args.max_mode, device=args.device,
            solve_kwargs={
                "force_backend": args.force_backend, "threads": args.threads
            },
            bounds=list(zip(x0, x0)))
    else:
        result = opt.least_squares(
            terms, inp, max_mode=args.max_mode, jac="implicit",
            jac_solver=args.jac_solver, jac_chunk_size=args.jac_chunk,
            max_nfev=1, ftol=1e-9, xtol=1e-10, device=args.device,
            solve_kwargs={
                "force_backend": args.force_backend, "threads": args.threads
            })
    jac = np.asarray(result.jac, dtype=np.float64)
    return {
        "optimizer": args.optimizer,
        "objective": args.implicit_objective,
        "jac_solver": args.jac_solver,
        "force_backend": args.force_backend,
        "threads": args.threads,
        "ns_array": np.asarray(inp.ns_array).tolist(),
        "mpol": int(inp.mpol),
        "ntor": int(inp.ntor),
        "jac_shape": list(jac.shape),
        "jac_finite": bool(np.all(np.isfinite(jac))),
        "jac_norm": float(np.linalg.norm(jac)),
        "jac_sha256": hashlib.sha256(jac.tobytes()).hexdigest(),
        "solve_stats": result.solve_stats,
    }


def _external_mirror_field(points):
    points = jnp.asarray(points)
    x, y, z = jnp.moveaxis(points, -1, 0)
    curvature = 0.02
    return jnp.stack(
        (-curvature * x * z,
         -curvature * y * z,
         0.08 + curvature * (z**2 - 0.5 * (x**2 + y**2))),
        axis=-1,
    )


def _mirror(args) -> dict:
    from vmex.mirror import (
        MirrorBoundary,
        MirrorConfig,
        MirrorResolution,
        SplineMirrorDiscretization,
        solve_beta_scan,
    )

    config = MirrorConfig(
        resolution=MirrorResolution(ns=args.ns, mpol=0, nxi=args.nxi),
        z_min=-0.8,
        z_max=0.8,
        ftol=1e-12,
        max_iterations=args.max_iterations,
    )
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build_cgl(
        config, elements=args.elements)
    grid = discretization.grid
    on_axis = 0.08 + 0.02 * jnp.asarray(grid.z) ** 2
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = discretization.fit_boundary(
        MirrorBoundary.from_axis_field(flux, on_axis, grid), source_grid)
    betas = jnp.asarray([float(value) for value in args.betas.split(",")])
    results = solve_beta_scan(
        boundary,
        discretization,
        config,
        _external_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        exterior_ntheta=args.exterior_ntheta,
        exterior_order=8,
        device=args.device,
    )
    return {
        "resolution": {
            "ns": args.ns,
            "nxi": args.nxi,
            "elements": args.elements,
            "exterior_ntheta": args.exterior_ntheta,
        },
        "betas": np.asarray(betas).tolist(),
        "converged": [bool(result.converged) for result in results],
        "iterations": [int(result.iterations) for result in results],
        "variational_max": [
            float(result.variational_max) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("implicit", "mirror"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out")
    parser.add_argument("--input")
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--aspect-target", type=float, default=8.0)
    parser.add_argument(
        "--implicit-objective", choices=("aspect", "stability"),
        default="aspect",
    )
    parser.add_argument(
        "--optimizer", choices=("least-squares", "minimize"),
        default="least-squares",
    )
    parser.add_argument(
        "--jac-solver",
        choices=("auto", "block", "gmres", "reverse"),
        default="auto",
    )
    parser.add_argument("--jac-chunk", default="auto")
    parser.add_argument("--force-backend", choices=("jax", "native"), default="jax")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ns", type=int, default=5)
    parser.add_argument("--nxi", type=int, default=7)
    parser.add_argument("--elements", type=int, default=4)
    parser.add_argument("--exterior-ntheta", type=int, default=8)
    parser.add_argument("--betas", default="0,0.1")
    parser.add_argument("--max-iterations", type=int, default=500)
    args = parser.parse_args()
    if args.device == "none":
        args.device = None
    if args.jac_chunk != "auto":
        args.jac_chunk = int(args.jac_chunk)

    started = time.perf_counter()
    payload = (_implicit(args) if args.case == "implicit" else _mirror(args))
    record = {
        "case": args.case,
        "device": None if args.device is None else str(args.device),
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "wall_s": time.perf_counter() - started,
        "peak_rss_gib": _peak_rss_gib(),
        **payload,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
