#!/usr/bin/env python3
"""Render the public capability contract from its benchmark artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "capabilities.json"
DOC = ROOT / "docs" / "reference" / "capabilities.rst"
FIELDS = ("cpu", "gpu", "forward", "jvp", "vjp", "optimization")
MARK = {"validated": "validated", "limited": "limited", "not-available": "—"}


def render(data: dict) -> str:
    """Return the complete reStructuredText capability page."""
    lines = [
        "Capability contract",
        "===================",
        "",
        "This table is the public support contract, generated from",
        "``benchmarks/capabilities.json``. ``validated`` means that committed",
        "evidence exercises the path; ``limited`` means that only the scope",
        "stated in the row is validated; ``—`` means no public path.",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 12 15 8 9 8 7 7 7 7 7 7 12 24",
        "",
        "   * - topology",
        "     - configuration",
        "     - boundary",
        "     - symmetry",
        "     - pressure",
        "     - CPU",
        "     - GPU",
        "     - forward",
        "     - JVP",
        "     - VJP",
        "     - optimize",
        "     - status",
        "     - scope and evidence",
    ]
    for row in data["rows"]:
        evidence = ", ".join(
            f"`{Path(path).name} <https://github.com/uwplasma/VMEX/blob/main/{path}>`__"
            for path in row["evidence"]
        )
        cells = [
            row["topology"], row["configuration"], row["boundary"],
            row["symmetry"], row["pressure"],
            *(MARK[row[field]] for field in FIELDS),
            row["status"], f'{row["scope"]} Evidence: {evidence}.',
        ]
        lines.append(f"   * - {cells[0]}")
        lines.extend(f"     - {cell}" for cell in cells[1:])
    lines += [
        "",
        "Free-boundary differentiation",
        "-----------------------------",
        "",
        "A supported forward free-boundary solve does not imply differentiation",
        "through the reconverged plasma-vacuum equilibrium. The toroidal",
        "specified-boundary virtual-casing residual is differentiable, but the",
        "fully coupled NESTOR root is not yet a public AD path.",
        "",
        "Mirror beta labels",
        "------------------",
        "",
        "The axisymmetric open-mirror free-boundary lane is supported through",
        "10% requested beta. The 25%, 50%, and 80% cases remain extended validation:",
        "the 80% example passes its force gate, but refined-grid promotion above",
        "the current 50% campaign is incomplete.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render(json.loads(SOURCE.read_text()))
    if args.check:
        if not DOC.exists() or DOC.read_text() != text:
            print("capability table is stale; run python tools/render_capabilities.py")
            return 1
        print("capability table is current")
        return 0
    DOC.write_text(text)
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
