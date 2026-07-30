"""Ownership and reporting gates for the test manifest."""

from __future__ import annotations

import json
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


def test_manifest_report_lists_timings_and_every_skip(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "2",
            "tests/test_capability_docs.py",
            "tests/test_lasym_free_convergence.py",
            f"--vmex-report={report}",
        ],
        cwd=ROOT,
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
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "tools/test_manifest.py select" in workflow
    for stale in ("A1_FILES=", "C2_FILES=", "core-a-c)"):
        assert stale not in workflow
