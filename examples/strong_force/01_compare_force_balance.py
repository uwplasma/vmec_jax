"""Compare independent strong-force certificates from VMEC-compatible wouts.

DESC equilibria can be exported first with ``desc.vmec.VMECIO.save``.  VMEX,
VMEC2000, VMEC++, and that DESC export then enter this script through the same
continuous reconstruction and validation grid.

Example
-------
python examples/strong_force/01_compare_force_balance.py \
  --input examples/data/input.solovev_analytical \
  --wout VMEX=output/wout_vmex.nc VMEC2000=reference/wout_vmec.nc \
  --output force_balance_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from vmex.core.input import VmecInput
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import (
    certify_strong_force,
    high_order_state_from_wout,
    plot_strong_force_report,
)

jax.config.update("jax_enable_x64", True)


def _labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("wout must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not label.strip() or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid or missing wout: {value}")
    return label.strip(), path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="source VMEC INDATA deck")
    parser.add_argument(
        "--wout",
        required=True,
        type=_labeled_path,
        nargs="+",
        help="one or more LABEL=PATH VMEC-compatible outputs",
    )
    parser.add_argument("--output", type=Path, default=Path("strong_force_comparison.png"))
    parser.add_argument("--degree", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument(
        "--radial-spans",
        type=int,
        help="explicit uniform reconstruction spans; use the same value for every code",
    )
    parser.add_argument("--angular-multiplier", type=int, default=2)
    args = parser.parse_args()
    if args.radial_spans is not None and args.radial_spans < 1:
        parser.error("radial-spans must be positive")

    inp = VmecInput.from_file(str(args.input))
    radial_basis = (
        None
        if args.radial_spans is None
        else BSplineBasis.clamped(
            np.linspace(0.0, 1.0, args.radial_spans + 1),
            degree=args.degree,
            quadrature_order=args.degree + 3,
        )
    )
    reports = {}
    summary = {}
    for label, path in args.wout:
        continuous = high_order_state_from_wout(
            path,
            inp=inp,
            radial_basis=radial_basis,
            degree=args.degree,
        )
        report = certify_strong_force(continuous, angular_multiplier=args.angular_multiplier)
        reports[label] = report
        summary[label] = {
            "radial_degree": int(continuous.radial_basis.degree),
            "radial_spans": int(continuous.radial_basis.breakpoints.size - 1),
            "radial_coefficients": int(continuous.radial_basis.size),
            "normalization": report.normalization,
            "coordinate_convention": report.coordinate_convention,
        }
        summary[label].update(
            {
                name: float(np.asarray(getattr(report, name)))
                for name in (
                    "absolute_l2",
                    "absolute_p99",
                    "absolute_linf",
                    "normalized_l2",
                    "normalized_p99",
                    "normalized_linf",
                    "near_axis_l2",
                    "bulk_l2",
                    "edge_l2",
                    "angular_spectral_tail",
                    "radial_refinement_difference",
                    "minimum_signed_jacobian",
                    "boundary_residual",
                    "gauge_residual",
                )
            }
        )

    figure, _ = plot_strong_force_report(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output} and {json_path}")


if __name__ == "__main__":
    main()
