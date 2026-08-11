"""Guards for the generated public capability contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import render_capabilities as capabilities  # noqa: E402


def _data() -> dict:
    return json.loads(capabilities.SOURCE.read_text())


def test_capability_table_is_current_and_evidence_exists() -> None:
    data = _data()
    assert data["schema"] == "vmex.capabilities/1"
    assert capabilities.DOC.read_text() == capabilities.render(data)
    ids = [row["id"] for row in data["rows"]]
    assert len(ids) == len(set(ids))
    for row in data["rows"]:
        assert row["status"] and row["scope"] and row["evidence"]
        assert all(row[field] in capabilities.MARK for field in capabilities.FIELDS)
        assert all((ROOT / path).is_file() for path in row["evidence"])


def test_contract_does_not_overclaim_free_boundary_ad() -> None:
    free_toroidal = [
        row for row in _data()["rows"]
        if row["topology"] == "toroidal" and row["boundary"] == "free"
    ]
    assert free_toroidal
    assert not any(row["coupled_ad"] for row in free_toroidal)


def test_supported_mirror_beta_matches_benchmark() -> None:
    data = _data()
    row = next(
        row for row in data["rows"]
        if row["id"] == "mirror-open-free-axisymmetric-supported"
    )
    benchmark = json.loads(
        (ROOT / "benchmarks" / "mirror_free_boundary_axisymmetric.json").read_text()
    )
    assert row["beta_max"] == benchmark["case"]["supported_beta_max"]


def test_readme_links_the_capability_contract() -> None:
    readme = (ROOT / "README.md").read_text()
    url = "https://vmex.readthedocs.io/en/latest/reference/capabilities.html"
    assert readme.count(url) == 1
