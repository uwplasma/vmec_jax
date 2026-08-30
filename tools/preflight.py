#!/usr/bin/env python3
"""Run CI's cheap gates and the diff-affected tests before pushing.

One CI attempt costs ~25-45 minutes of wall clock, so a failure that a
local run would have caught in minutes costs a full extra attempt. This
tool runs, against the diff to ``origin/main`` (or ``--base``):

1. the static gates (ruff, mypy, docs prose) — the ``quality`` job's
   cheap half;
2. the guard tests — fast meta-tests that police the manifest, committed
   reports (personal-path privacy), docs navigation, and packaging;
3. the affected tests — changed test files plus every test module that
   imports a changed vmex module — under coverage, followed by the same
   changed-line bar CI enforces (``diff-cover --fail-under=95``) when
   diff-cover is installed.

The import scan is a local approximation (it overshoots harmlessly and
cannot see dynamic imports), and the mirror-package coverage floor needs
the full matrix, so CI remains the authority — preflight exists to make
the first CI attempt the only one.

Usage::

    python tools/preflight.py            # gates + guards + affected tests
    python tools/preflight.py --static   # gates + guards only (seconds)
    python tools/preflight.py --docs     # also build sphinx -W
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GUARD_TESTS = (
    "tests/test_test_manifest.py",
    "tests/test_performance_docs.py",
    "tests/test_docs_nav.py",
    "tests/test_capability_docs.py",
    "tests/test_packaging_metadata.py",
    "tests/test_coverage_margin.py",
)


def _run(title: str, command: list[str], **kwargs) -> bool:
    print(f"\n== {title}: {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, **kwargs).returncode == 0


def _changed_files(base: str) -> list[str]:
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--name-only", merge_base], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.split()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout
    dirty = [line[3:].split(" -> ")[-1] for line in status.splitlines()]
    return sorted(set(diff) | set(dirty))


def _affected_tests(changed: list[str]) -> list[str]:
    tests = {f for f in changed
             if f.startswith("tests/") and f.endswith(".py")
             and Path(ROOT, f).exists() and "/__" not in f}
    modules = [f for f in changed
               if f.startswith("vmex/") and f.endswith(".py")]
    names = {Path(m).stem for m in modules} - {"__init__"}
    if names:
        pattern = re.compile(
            r"^\s*(?:from\s+vmex[.\w]*\s+import\b.*\b({0})\b"
            r"|import\s+vmex[.\w]*\b({0})\b"
            r"|from\s+vmex[.\w]*\b({0})\s+import\b)".format(
                "|".join(map(re.escape, sorted(names)))),
            re.MULTILINE)
        for path in sorted(ROOT.glob("tests/**/test_*.py")):
            if pattern.search(path.read_text(errors="replace")):
                tests.add(str(path.relative_to(ROOT)))
    return sorted(tests)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--static", action="store_true",
                        help="static gates and guard tests only")
    parser.add_argument("--docs", action="store_true",
                        help="also build the docs (sphinx -W)")
    args = parser.parse_args(argv)

    changed = _changed_files(args.base)
    print("changed files vs", args.base)
    for f in changed:
        print("  ", f)

    ok = True
    ok &= _run("ruff", ["ruff", "check", "vmex", "tests", "examples",
                        "benchmarks", "tools"])
    ok &= _run("mypy", ["mypy", "vmex"])
    ok &= _run("docs prose", [sys.executable, "tools/check_docs_prose.py"])
    if args.docs:
        ok &= _run("sphinx", [sys.executable, "-m", "sphinx", "-W", "-j",
                              "auto", "-b", "html", "docs",
                              "docs/_build/html"])

    guards = [t for t in GUARD_TESTS if Path(ROOT, t).exists()]
    ok &= _run("guard tests", [sys.executable, "-m", "pytest", "-q",
                               "-m", "not full and not weekly", *guards])

    if not args.static:
        affected = _affected_tests(changed)
        if affected:
            print("\naffected test modules:")
            for t in affected:
                print("  ", t)
            ok &= _run(
                "affected tests", [
                    sys.executable, "-m", "pytest", "-q", "-n", "auto",
                    "-m", "not full and not weekly",
                    "--cov=vmex", "--cov-report=xml", *affected])
            if shutil.which("diff-cover"):
                ok &= _run("changed-line coverage", [
                    "diff-cover", "coverage.xml",
                    f"--compare-branch={args.base}", "--fail-under=95"])
            else:
                print("\n== changed-line coverage: diff-cover not installed "
                      "(pip install diff-cover); CI will enforce it")
        else:
            print("\nno affected test modules for this diff")

    print("\npreflight:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
