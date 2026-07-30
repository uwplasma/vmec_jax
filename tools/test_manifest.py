#!/usr/bin/env python3
"""Validate and query the VMEX test manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "manifest.json"
PRIMARY = {"unit", "integration", "oracle", "ad", "gpu", "campaign"}
DURATIONS = {"fast", "medium", "slow", "campaign"}
DEVICES = {"cpu", "cpu-gpu", "gpu"}
ASSETS = {"none", "generated", "golden", "reference-nc", "external"}
ORACLES = {"none", "analytic", "vmec2000", "golden", "fd", "external"}
PRIMARY_LANES = (
    "pr-parity-", "pr-mirror-", "pr-gradient", "pr-examples",
    "pr-full-only", "pr-external",
)


def load() -> tuple[dict, list[dict]]:
    """Load compact rows as named records."""
    data = json.loads(MANIFEST.read_text())
    if data.get("schema") != "vmex.test-manifest/1":
        raise ValueError("unsupported test-manifest schema")
    fields = data["fields"]
    records = []
    for row in data["records"]:
        if len(row) != len(fields):
            raise ValueError(f"manifest row has {len(row)} fields, expected {len(fields)}")
        records.append(dict(zip(fields, row, strict=True)))
    return data, records


def collect() -> list[str]:
    """Return every pytest node ID without executing tests."""
    env = os.environ.copy()
    env.pop("RUN_FULL", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return [
        line.strip() for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    ]


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def validate(nodes: list[str]) -> list[str]:
    """Return manifest errors for the collected suite."""
    data, records = load()
    errors: list[str] = []
    paths = [record["path"] for record in records]
    excluded = set(data["excluded"])
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
    }
    collected_files = {node.split("::", 1)[0] for node in nodes}

    if duplicates := _duplicates(paths):
        errors.append(f"duplicate module ownership: {sorted(duplicates)}")
    if missing := set(paths) - on_disk:
        errors.append(f"manifest paths do not exist: {sorted(missing)}")
    if stale := excluded - on_disk:
        errors.append(f"excluded helpers do not exist: {sorted(stale)}")
    if unowned := collected_files - set(paths):
        errors.append(f"collected modules lack ownership: {sorted(unowned)}")
    if uncollected := set(paths) - collected_files:
        errors.append(f"manifest modules collect no tests: {sorted(uncollected)}")
    if unknown := on_disk - set(paths) - excluded:
        errors.append(f"test modules are absent from manifest: {sorted(unknown)}")

    allowed = {
        "primary": PRIMARY, "duration": DURATIONS, "device": DEVICES,
        "asset": ASSETS, "oracle": ORACLES,
    }
    for record in records:
        for field, values in allowed.items():
            if record[field] not in values:
                errors.append(f"{record['path']}: invalid {field}={record[field]!r}")
        primary_lanes = [
            lane for lane in record["lanes"]
            if lane != "pr-fast" and any(lane.startswith(prefix) for prefix in PRIMARY_LANES)
        ]
        if len(primary_lanes) != 1:
            errors.append(
                f"{record['path']}: expected one primary PR lane, got {primary_lanes}"
            )

    node_set = set(nodes)
    full_members: list[str] = []
    by_file: dict[str, list[str]] = {}
    for node in nodes:
        by_file.setdefault(node.split("::", 1)[0], []).append(node)
    for record in records:
        for lane in record["lanes"]:
            if lane.startswith("full-"):
                full_members.extend(by_file.get(record["path"], ()))
    for lane, selectors in data["campaigns"].items():
        if not lane.startswith("full-"):
            errors.append(f"campaign lane is not full-* metadata: {lane}")
        missing = set(selectors) - node_set
        if missing:
            errors.append(f"{lane}: selectors do not collect: {sorted(missing)}")
        full_members.extend(selectors)
    if duplicates := _duplicates(full_members):
        errors.append(f"full-suite duplicate ownership: {sorted(duplicates)}")
    if missing := node_set - set(full_members):
        errors.append(f"full-suite nodes lack ownership: {sorted(missing)}")

    random_patterns = (
        r"default_rng\(\s*\)",
        r"np\.random\.(?:rand|randn|random|random_sample|standard_normal)\(",
    )
    for path in paths:
        text = (ROOT / path).read_text()
        if any(re.search(pattern, text) for pattern in random_patterns):
            errors.append(f"{path}: random generator has no explicit seed")
    return errors


def select(lane: str) -> list[str]:
    """Return module or node selectors owned by a CI lane."""
    data, records = load()
    selected = [record["path"] for record in records if lane in record["lanes"]]
    selected.extend(data["campaigns"].get(lane, ()))
    if not selected:
        raise ValueError(f"unknown or empty manifest lane: {lane}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="collect tests and validate all ownership")
    choose = sub.add_parser("select", help="print pytest selectors for a lane")
    choose.add_argument("lane")
    args = parser.parse_args()

    if args.command == "select":
        print(" ".join(select(args.lane)))
        return 0
    nodes = collect()
    errors = validate(nodes)
    if errors:
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    data, records = load()
    print(
        f"test manifest is current: {len(records)} modules, "
        f"{len(nodes)} collected tests, {len(data['campaigns'])} campaigns"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
