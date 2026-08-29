#!/usr/bin/env python
"""Measure one common strong-force certificate for VMEX or a compatible wout."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import platform
from pathlib import Path
import resource
import shlex
import sys
import time

import jax
import numpy as np

import vmex
from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import (
    certify_strong_force,
    high_order_state_from_wout,
    lift_high_order_state,
)

from _provenance import assert_repo_vmex, git_state

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "examples" / "data" / "input.solovev_analytical"


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def _portable_argument(value: str) -> str:
    """Remove host-specific prefixes from recorded reproduction commands."""

    path = Path(value)
    return path.name if path.is_absolute() else value


def _portable_path(path: Path) -> str:
    """Return a repository-relative path or a basename for external data."""

    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON after provenance is captured instead of using stdout",
    )
    parser.add_argument(
        "--wout",
        type=Path,
        help="optional VMEX/VMEC2000/VMEC++/DESC-exported compatible wout",
    )
    parser.add_argument(
        "--source-provenance",
        type=Path,
        help="optional JSON metadata emitted by the external equilibrium run",
    )
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument(
        "--radial-spans",
        type=int,
        help="explicit uniform spline spans; omit to use the stable default lift",
    )
    parser.add_argument("--angular-multiplier", type=int, default=2)
    parser.add_argument("--radial-order-increment", type=int, default=2)
    parser.add_argument("--ftol", type=float, default=1.0e-12)
    parser.add_argument("--max-iterations", type=int, default=2000)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.wout is not None and not args.wout.is_file():
        parser.error(f"wout does not exist: {args.wout}")
    if args.source_provenance is not None and not args.source_provenance.is_file():
        parser.error(f"source provenance does not exist: {args.source_provenance}")
    if args.radial_spans is not None and args.radial_spans < 1:
        parser.error("radial-spans must be positive")
    if args.angular_multiplier < 1 or args.radial_order_increment < 0:
        parser.error("certificate refinement controls are invalid")

    inp = VmecInput.from_file(str(args.input))
    basis = (
        None
        if args.radial_spans is None
        else BSplineBasis.clamped(
            np.linspace(0.0, 1.0, args.radial_spans + 1),
            degree=args.degree,
            quadrature_order=args.degree + 3,
        )
    )
    started = time.perf_counter()
    rss_before = _peak_rss_mib()
    solve_seconds = None
    if args.wout is None:
        solve_started = time.perf_counter()
        config = implicit.make_config(
            inp,
            ftol=args.ftol,
            max_iterations=args.max_iterations,
        )
        params = implicit.params_from_input(inp)
        state, _ = implicit.solve_implicit_with_aux(params, config)
        runtime = implicit.runtime_from_params(params, config)
        solve_seconds = time.perf_counter() - solve_started
        continuous = lift_high_order_state(
            state,
            runtime,
            radial_basis=basis,
            degree=args.degree,
        )
        source = "VMEX"
        ns = int(runtime.resolution.ns)
    else:
        continuous = high_order_state_from_wout(
            args.wout,
            inp=inp,
            radial_basis=basis,
            degree=args.degree,
        )
        source = args.wout.name
        ns = None
    lift_seconds = time.perf_counter() - started - (solve_seconds or 0.0)
    certificate_started = time.perf_counter()
    report = certify_strong_force(
        continuous,
        angular_multiplier=args.angular_multiplier,
        radial_order_increment=args.radial_order_increment,
    )
    jax.block_until_ready(report)
    certificate_seconds = time.perf_counter() - certificate_started
    scalar_fields = (
        "absolute_l2",
        "absolute_p99",
        "absolute_linf",
        "normalized_l2",
        "normalized_p99",
        "normalized_linf",
        "radial_normalized_l2",
        "helical_normalized_l2",
        "near_axis_l2",
        "bulk_l2",
        "edge_l2",
        "angular_spectral_tail",
        "radial_refinement_difference",
        "minimum_signed_jacobian",
        "nestedness_margin",
        "boundary_residual",
        "gauge_residual",
    )
    result = {
        "schema": "vmex.strong-certificate-benchmark/1",
        "command": " ".join(
            shlex.quote(_portable_argument(value)) for value in sys.argv
        ),
        "case": args.input.name.removeprefix("input."),
        "source": source,
        "external_source": (
            None
            if args.source_provenance is None
            else json.loads(args.source_provenance.read_text())
        ),
        "input": _portable_path(args.input),
        "ns": ns,
        "mpol": int(inp.mpol),
        "ntor": int(inp.ntor),
        "degree": int(continuous.radial_basis.degree),
        "radial_spans": int(continuous.radial_basis.breakpoints.size - 1),
        "radial_coefficients": int(continuous.radial_basis.size),
        "angular_multiplier": args.angular_multiplier,
        "radial_order_increment": args.radial_order_increment,
        "solve_seconds": solve_seconds,
        "lift_seconds": lift_seconds,
        "certificate_seconds": certificate_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_increase_mib": _peak_rss_mib() - rss_before,
        "normalization": report.normalization,
        "coordinate_convention": report.coordinate_convention,
        "metrics": {name: float(np.asarray(getattr(report, name))) for name in scalar_fields},
        "radial_profile": {
            "rho": np.asarray(report.radial_nodes).tolist(),
            "flux_surface_average_force_density": np.asarray(
                report.flux_surface_average
            ).tolist(),
            "units": "N m^-3",
        },
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "vmex": vmex.__version__,
            "jax": jax.__version__,
            "jaxlib": _version("jaxlib"),
            "numpy": np.__version__,
        },
        **git_state(REPO),
        "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
