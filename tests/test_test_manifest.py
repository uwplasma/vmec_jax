"""Ownership and reporting gates for the test manifest."""

from __future__ import annotations

import json
import os
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
        "pr-mirror-spline"
    )
    weekly = test_manifest.select("weekly-mirror")
    assert weekly == [
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


def test_workflow_selects_manifest_lanes() -> None:
    workflows = {
        path.name: path.read_text()
        for path in (ROOT / ".github" / "workflows").glob("*.yml")
    }
    for name in ("ci.yml", "gpu.yml", "nightly.yml", "weekly.yml"):
        assert "tools/test_manifest.py select" in workflows[name]
    assert "name: PR gate" in workflows["ci.yml"]
    for stale in ("A1_FILES=", "C2_FILES=", "core-a-c)"):
        assert stale not in "".join(workflows.values())
