"""Guard: every test file is claimed by exactly one CI shard.

The parity workflow lists test files explicitly per shard instead of using a
catch-all, which keeps each shard's runtime bounded and its coverage disjoint.
The cost of that design is that a *new* test file is not picked up
automatically: unless someone adds it to a shard it is collected by nobody, and
CI stays green while the tests never run.

That failure is silent and has already happened twice while merging parallel
branches, so it gets a test of its own.  A file is considered accounted for if
it is either named in ``.github/workflows/ci.yml`` or deliberately excluded
because a dedicated job owns it (the gradient/example jobs) or because a marker
routes it to the nightly / external-binary lanes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TESTS = ROOT / "tests"

#: Files intentionally outside the ordinary parity shards, with the reason.
DEDICATED_LANES = {
    "tests/test_implicit_grad.py": "own sharded implicit-gradient job",
    "tests/test_examples.py": "own example-smoke job",
    "tests/test_lasym_free_case.py": "helper module, defines no tests",
    "tests/test_qi_free_boundary_case.py": "helper module, defines no tests",
    "tests/test_lasym_free_convergence.py": "pytest.mark.full -> nightly lane",
    "tests/test_vmec2000_live.py": "pytest.mark.vmec2000_live -> external binary",
}


def test_every_test_file_is_claimed_by_a_shard() -> None:
    """A new test file must be routed somewhere, or CI silently skips it."""
    listed = set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", WORKFLOW.read_text()))
    on_disk = {f"tests/{path.name}" for path in TESTS.glob("test_*.py")}

    unclaimed = sorted(on_disk - listed - set(DEDICATED_LANES))
    assert not unclaimed, (
        "these test files are in no CI shard, so they never run:\n  "
        + "\n  ".join(unclaimed)
        + "\nAdd each to a shard list in .github/workflows/ci.yml, or record it "
          "in DEDICATED_LANES with the job/marker that owns it."
    )


def test_dedicated_lane_entries_still_exist() -> None:
    """Keep the exemption list honest as files are renamed or removed."""
    missing = sorted(name for name in DEDICATED_LANES if not (ROOT / name).exists())
    assert not missing, (
        "DEDICATED_LANES names files that no longer exist: " + ", ".join(missing)
    )


def test_workflow_referenced_test_files_exist() -> None:
    """Every test path named in ANY workflow must exist on disk.

    The claim check above only covers top-level ``tests/test_*.py``; a stale
    reference to a renamed or deleted file (including ``tests/mirror/...`` and
    the GPU workflow) fails the job at collection time on CI while looking like
    an infrastructure problem.  Guard the reverse direction here.
    """
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows, "no workflow files found"
    missing: list[str] = []
    for wf in workflows:
        for ref in re.findall(r"tests/[A-Za-z0-9_/]+\.py", wf.read_text()):
            if not (ROOT / ref).exists():
                missing.append(f"{wf.name}: {ref}")
    assert not missing, (
        "workflows reference test files that do not exist:\n  "
        + "\n  ".join(missing)
    )


def _full_shard_sets() -> dict[str, set[str]]:
    """Simulate the full-matrix selection rules against the real test tree.

    Parses each ``FILES="..."`` assignment in the full job's case block and
    applies its ``--ignore`` / ``--ignore-glob`` semantics, returning the
    top-level test files each core shard would collect.
    """
    import fnmatch

    text = WORKFLOW.read_text()
    # several jobs use a matrix.shard case statement; take the
    # full-physics one -- the block defining the core-* letter ranges
    case_block = None
    start = 0
    while True:
        i = text.find('case "${{ matrix.shard }}" in', start)
        if i < 0:
            break
        candidate = text[i:text.index("esac", i)]
        if "core-a-c)" in candidate:
            case_block = candidate
            break
        start = i + 1
    assert case_block is not None, "full-physics case block not found"
    on_disk = sorted(f"tests/{p.name}" for p in TESTS.glob("test_*.py"))

    shards: dict[str, set[str]] = {}
    for m in re.finditer(
            r"(\S+)\)\s*(?:#[^\n]*\n\s*)*FILES=\"((?:[^\"\\]|\\.|\\\n)*)\"",
            case_block):
        name, spec = m.group(1), m.group(2).replace("\\\n", " ")
        tokens = spec.split()
        if tokens and tokens[0] != "tests":
            # explicit-file shard (e.g. high-mode-fb, opt-*): the listed files
            shards[name] = {t.split("::")[0] for t in tokens
                            if t.startswith("tests/test_")}
            continue
        ignored_globs = [t.split("=", 1)[1] for t in tokens
                         if t.startswith("--ignore-glob=")]
        ignored = {t.split("=", 1)[1] for t in tokens
                   if t.startswith("--ignore=") and not t.startswith("--ignore-glob=")}
        selected = set()
        for f in on_disk:
            if f in ignored:
                continue
            if any(fnmatch.fnmatch(f, g) for g in ignored_globs):
                continue
            selected.add(f)
        shards[name] = selected
    return shards


def test_full_matrix_core_shards_partition_the_suite() -> None:
    """Core shards must be DISJOINT and jointly COMPLETE (simulated rules).

    Review finding: the earlier guard only checked that each filename appears
    somewhere in the workflow text — it could not catch a file collected by
    two shards (duplicate coverage skews timings) or dropped by every letter
    range (silent skip).  This simulates the actual --ignore/--ignore-glob
    selection against the on-disk tree.
    """
    shards = _full_shard_sets()
    core = {k: v for k, v in shards.items() if k.startswith("core-")}
    assert core, "no core-* shard definitions parsed from the workflow"

    # disjoint among core shards and against the dedicated-file shards
    dedicated = set().union(*(v for k, v in shards.items()
                              if not k.startswith("core-")))
    seen: dict[str, str] = {}
    for name, files in core.items():
        for f in files - dedicated:
            assert f not in seen, (
                f"{f} collected by BOTH {seen[f]} and {name}")
            seen[f] = name

    # complete: every top-level test file lands in exactly one full shard
    on_disk = {f"tests/{p.name}" for p in TESTS.glob("test_*.py")}
    covered = set(seen) | dedicated | {"tests/test_examples.py"}
    missing = sorted(on_disk - covered - set(DEDICATED_LANES))
    assert not missing, (
        "full-matrix selection drops these files from every shard:\n  "
        + "\n  ".join(missing))
