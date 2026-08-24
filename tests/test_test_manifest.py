"""Ownership and reporting gates for the test manifest."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import test_manifest  # noqa: E402


def test_collected_suite_has_exact_manifest_ownership() -> None:
    nodes = test_manifest.collect()
    assert nodes
    assert not test_manifest.validate(nodes)


def test_manifest_routes_the_previously_nightly_only_mirror_module() -> None:
    selected = test_manifest.select("pr-mirror-spline")
    assert "tests/mirror/test_qi_hybrid.py" in selected


def test_manifest_routes_short_pr_and_weekly_selectors() -> None:
    assert "tests/mirror/test_implicit.py" in test_manifest.select(
        "pr-mirror-equilibrium"
    )
    assert "tests/mirror/test_free_boundary.py" in test_manifest.select(
        "pr-mirror-field"
    )
    assert "tests/mirror/test_splines.py" in test_manifest.select(
        "pr-physics-mirror-spline"
    )
    assert test_manifest.select("pr-physics-mirror-output") == [
        "tests/mirror/test_output.py"
    ]
    assert len(test_manifest.select("weekly-hmfb-fixed")) == 2
    assert len(test_manifest.select("weekly-hmfb-free")) == 2
    weekly = test_manifest.select("weekly-mirror")
    assert weekly == [
        "tests/mirror/test_free_boundary.py::"
        "test_unbounded_exterior_free_boundary_beta_scan_converges",
        "tests/mirror/test_free_boundary.py::"
        "test_unbounded_exterior_beta_observables_converge_with_resolution"
    ]


def test_manifest_report_lists_timings_and_every_skip(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop("RUN_FULL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_capability_docs.py",
            "tests/test_lasym_free_convergence.py",
            f"--vmex-report={report}",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["schema"] == "vmex.test-report/1"
    assert data["collected"] == 5
    assert data["slowest"]
    assert len(data["skips"]) == 1
    assert data["skips"][0]["nodeid"].startswith(
        "tests/test_lasym_free_convergence.py::"
    )
    for record in data["slowest"] + data["skips"]:
        assert {
            "owner", "primary", "duration", "device", "asset", "oracle"
        } <= record.keys()


def _invoked_lanes() -> set[str]:
    """Lane names the workflows actually hand to ``test_manifest.py select``.

    Both spellings: a literal ``select <lane>`` and a matrix ``selector:``
    value interpolated into one.
    """
    text = "\n".join(
        path.read_text() for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")))
    invoked: set[str] = set()
    for value in re.findall(r"^\s*selector:\s*(.+?)\s*$", text, re.M):
        invoked.update(value.split())          # a matrix value may list several
    invoked |= set(re.findall(r"test_manifest\.py select ([a-z][A-Za-z0-9-]*)", text))
    return invoked


def test_every_primary_pr_lane_is_invoked_by_a_workflow() -> None:
    """A primary lane no job runs is a module no job runs.

    ``validate`` already makes every record declare exactly one primary PR
    lane, so the union of those lanes is the whole pull-request suite.  What
    was missing is the other half of the contract: the workflows have to
    invoke all of them.  Twelve of fourteen once did not, which took 94 of
    100 modules out of pull-request CI and made the changed-line coverage
    gate fail on code that was in fact tested.
    """
    _, records = test_manifest.load()
    primary = {
        lane
        for record in records
        for lane in record["lanes"]
        if lane != "pr-fast"
        and any(lane.startswith(prefix) for prefix in test_manifest.PRIMARY_LANES)
    }
    assert primary
    missing = sorted(primary - _invoked_lanes())
    assert not missing, (
        "primary PR lanes that no workflow invokes, so their modules never "
        f"run in CI: {missing}. Add a matrix entry in .github/workflows/ci.yml "
        "or move the modules to a lane that has one."
    )


def test_workflow_selects_manifest_lanes() -> None:
    workflows = {
        path.name: path.read_text()
        for path in (ROOT / ".github" / "workflows").glob("*.yml")
    }
    for name in ("ci.yml", "gpu.yml", "nightly.yml", "weekly.yml"):
        assert "tools/test_manifest.py select" in workflows[name]
    assert "name: PR gate" in workflows["ci.yml"]
    nightly = workflows["nightly.yml"]
    assert "campaign: [opt-qi, opt-qa, opt-qh, opt-qp]" in nightly
    assert "tools/test_manifest.py select optional" in nightly
    assert "--cov=vmex" not in nightly
    assert "timeout-minutes: 45" in nightly
    for stale in ("A1_FILES=", "C2_FILES=", "core-a-c)"):
        assert stale not in "".join(workflows.values())
