"""Print Step-10 fsq parity against bundled reference wouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import vmec_jax.api as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--solve-metric", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    vj.step10_getfsq_parity_cases(root=root, solve_metric=args.solve_metric, include_all=args.all)


if __name__ == "__main__":
    main()

