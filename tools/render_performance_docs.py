#!/usr/bin/env python3
"""Render the docs/performance.rst baseline table from benchmarks/baseline.json.

The table lives between the ``.. begin generated-baseline-table`` and
``.. end generated-baseline-table`` markers and is REWRITTEN by this script —
the narrative around it never carries numbers of its own, so the docs cannot
drift from the committed benchmark artifact.

Usage::

    python tools/render_performance_docs.py           # rewrite in place
    python tools/render_performance_docs.py --check   # exit 1 when stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "benchmarks" / "baseline.json"
DOC = REPO / "docs" / "performance.rst"
BEGIN = ".. begin generated-baseline-table (tools/render_performance_docs.py)"
END = ".. end generated-baseline-table"


def _cell(entry: dict | None) -> tuple[str, float | None]:
    """Table cell text and the comparable wall value (None when absent)."""
    if not entry:
        return "n/a", None
    wall = entry.get("wall_s", entry.get("warm_s"))
    if not entry.get("ok"):
        # a run that finished with a nonzero exit but produced a full
        # equal-iteration trajectory is still comparable work; anything
        # without a wall time is a genuine failure
        return ("failed" if wall is None else f"{wall:.3g}*"), wall
    return f"{wall:.3g}", wall


def render(baseline: dict) -> str:
    rows = []
    for key, row in baseline.items():
        if key.startswith("_"):
            continue
        case, grid = key[:-1].split("[")
        label = case + (" (multigrid)" if grid == "multigrid" else "")
        v2k_txt, v2k = _cell(row.get("vmec2000"))
        cold_txt, _ = _cell(row.get("vmex_cold"))
        warm_txt, warm = _cell(row.get("vmex_warm"))
        ref_txt, _ = _cell(row.get("vmecpp"))
        if warm is not None and v2k is not None and warm < v2k:
            warm_txt = f"**{warm_txt}**"
        rows.append((v2k if v2k is not None else float("inf"), label, v2k_txt, cold_txt, warm_txt, ref_txt))
    rows.sort()

    wins = sum(1 for r in rows if r[4].startswith("**"))
    lines = [
        BEGIN,
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 34 14 14 14 14",
        "",
        "   * - case",
        "     - VMEC2000",
        "     - vmex cold",
        "     - vmex warm",
        "     - reference C++",
    ]
    for _, label, v2k_txt, cold_txt, warm_txt, ref_txt in rows:
        lines += [
            f"   * - {label}",
            f"     - {v2k_txt}",
            f"     - {cold_txt}",
            f"     - {warm_txt}",
            f"     - {ref_txt}",
        ]
    lines += [
        "",
        f"Bold marks vmex warm beating VMEC2000 ({wins} of {len(rows)} rows).",
        "``*`` marks an equal-iteration-budget run whose CLI exit was nonzero",
        "(the deliberately NITER-bounded LASYM stress row: both codes exhaust",
        "the same budget, so the wall times compare equal work); ``failed``",
        "marks an aborted run and ``n/a`` an unsupported configuration.",
        "",
        END,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 when the doc is stale instead of rewriting")
    args = ap.parse_args()

    baseline = json.loads(BASELINE.read_text())
    text = DOC.read_text()
    try:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
    except ValueError:
        print(f"markers not found in {DOC}", file=sys.stderr)
        return 2
    new = head + render(baseline) + tail
    if args.check:
        if new != text:
            print(
                "docs/performance.rst baseline table is stale; run python tools/render_performance_docs.py",
                file=sys.stderr,
            )
            return 1
        print("performance table is current")
        return 0
    if new != text:
        DOC.write_text(new)
        print(f"rewrote {DOC}")
    else:
        print("performance table already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
