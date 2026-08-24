#!/usr/bin/env python3
"""Minimal QH residual-Jacobian comparison for VMEX.

This deliberately uses the public ``least_squares`` interface and the
packaged ``input.nfp4_QH_warm_start`` deck.  ``jac=None`` is the independent
full-resolve central-FD oracle; ``jac='implicit'`` is the production path.
The two lanes use the same traceable objective and strict solver ladder.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from vmex.core import optimize as opt
from vmex.core.input import VmecInput


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fd", "implicit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=1)
    parser.add_argument("--max-mode", type=int, default=1)
    parser.add_argument(
        "--jac-solver", choices=("auto", "block", "gmres", "reverse"),
        default="auto",
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "vmex" / "resources" / "input.nfp4_QH_warm_start",
    )
    args = parser.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    inp = VmecInput.from_file(args.input)
    surfaces = np.arange(0.0, 1.01, 0.1)
    qh = opt.QuasisymmetryRatioResidual(
        surfaces, helicity_m=1, helicity_n=-1, ntheta=63, nphi=64,
    )

    def qh_state(state, runtime):
        return qh.residuals_state(state, runtime)

    started = time.perf_counter()
    result = opt.least_squares(
        [(qh_state, 0.0, 1.0), (opt.aspect_ratio, 7.0, 1.0)],
        inp,
        max_mode=args.max_mode,
        jac=None if args.mode == "fd" else "implicit",
        jac_solver=args.jac_solver,
        jac_chunk_size=1 if args.mode == "fd" else "auto",
        hot_restart=False,
        warm_start=None,
        use_ess=False,
        solve_kwargs={
            "ns_array": [16, 50],
            "ftol_array": [1e-16, 1e-11],
            "niter_array": [600, 2000],
            "device": "cpu",
        },
        max_nfev=args.max_nfev,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    jac = np.asarray(result.jac, dtype=float)
    residual = np.asarray(result.fun, dtype=float)
    payload = {
        "schema": 1,
        "mode": args.mode,
        "max_mode": args.max_mode,
        "wall_seconds": time.perf_counter() - started,
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        },
        "parameters": np.asarray(result.x).tolist(),
        "residual": residual.tolist(),
        "jacobian": jac.tolist(),
        "residual_count": int(residual.size),
        "parameter_count": int(jac.shape[1]),
        "provenance": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "vmex": "0.6.0-source",
            "input": str(args.input),
        },
        "config": {
            "surfaces": [round(0.1 * i, 1) for i in range(11)],
            "ntheta": 63,
            "nphi": 64,
            "helicity_m": 1,
            "helicity_n": -1,
            "residual_definition": (
                "QuasisymmetryRatioResidual.residuals_state(state,runtime) "
                "+ aspect_ratio(state,runtime)-7"
            ),
            "ns": [16, 50],
            "ftol": [1e-16, 1e-11],
            "niter": [600, 2000],
            "device": "cpu",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps({
        "mode": args.mode,
        "wall_seconds": payload["wall_seconds"],
        "residual_count": payload["residual_count"],
        "parameter_count": payload["parameter_count"],
        "cost": payload["optimizer"]["cost"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
