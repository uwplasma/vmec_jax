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
