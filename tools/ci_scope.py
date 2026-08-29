#!/usr/bin/env python3
"""Classify whether a pull request needs numerical tests and coverage."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable


DOCUMENTATION_SUFFIXES = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".md",
    ".pdf",
    ".png",
    ".rst",
    ".svg",
    ".webp",
}


def classify(paths: Iterable[str], *, force_all: bool = False) -> tuple[bool, bool]:
    """Return ``(run_tests, run_coverage)`` for changed repository paths.

    Only documentation and rendered media bypass the numerical matrix. Unknown
    files remain conservative and run it. Changed-line coverage is necessary
    only when executable package code changed; test, workflow, and tool changes
    still run the matrix but do not spend another job combining coverage.
    """

    if force_all:
        return True, True
    changed = tuple(path.strip("/") for path in paths if path.strip("/"))
    if not changed:
        return True, True
    documentation_only = all(
        "/." not in path
        and any(path.lower().endswith(suffix) for suffix in DOCUMENTATION_SUFFIXES)
        for path in changed
    )
    run_tests = not documentation_only
    run_coverage = any(
        path.startswith("vmex/") and path.lower().endswith(".py")
        for path in changed
    )
    return run_tests, run_coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="select the full main-branch gate")
    parser.add_argument("--null", action="store_true", help="read NUL-delimited paths")
    args = parser.parse_args(argv)
    payload = sys.stdin.buffer.read()
    separator = b"\0" if args.null else b"\n"
    paths = [
        item.decode("utf-8", errors="surrogateescape")
        for item in payload.split(separator)
    ]
    run_tests, run_coverage = classify(paths, force_all=args.all)
    print(f"run_tests={str(run_tests).lower()}")
    print(f"run_coverage={str(run_coverage).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
