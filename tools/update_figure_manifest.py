#!/usr/bin/env python3
"""Refresh the mechanical fields of ``docs/_static/figures/figures.json``.

The manifest records, for every committed figure, where it came from: the
generator, the inputs, the date, the hardware, and whether one command
reproduces it.  Those fields are authored by hand.  Three fields are not:
``sha256``, ``bytes`` and ``used_in`` are derived from the tree, and this
script rewrites them so a regenerated figure cannot silently drift from its
provenance row.

Usage::

    python tools/update_figure_manifest.py           # rewrite in place
    python tools/update_figure_manifest.py --check   # exit 1 when stale

``tests/test_figure_provenance.py`` runs the same comparison, so a figure
that changes without its row fails CI rather than shipping unexplained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs" / "_static" / "figures" / "figures.json"
FIGURE_DIR = REPO / "docs" / "_static" / "figures"
SCHEMA = "vmex.figure-provenance/1"

#: Sources scanned for figure references, relative to the repository root.
SEARCH_ROOTS = ("README.md", "docs")
SEARCH_SUFFIXES = (".md", ".rst")
#: Never scan the built site or the manifest's own prose.
SKIP_PARTS = ("_build", "_static")


def figure_paths() -> list[Path]:
    """Every committed image under ``docs/_static/figures``."""
    return sorted(FIGURE_DIR.glob("*.webp"))


def _sources() -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        path = REPO / root
        if path.is_file():
            found.append(path)
            continue
        for candidate in path.rglob("*"):
            if candidate.suffix not in SEARCH_SUFFIXES:
                continue
            if any(part in SKIP_PARTS for part in candidate.parts):
                continue
            found.append(candidate)
    return sorted(found)


def used_in(figure: Path) -> list[str]:
    """Repository-relative pages that reference this figure."""
    name = figure.name
    pages = []
    for source in _sources():
        if name in source.read_text(encoding="utf-8"):
            pages.append(source.relative_to(REPO).as_posix())
    return pages


def refreshed(manifest: dict) -> dict:
    """The manifest with its derived fields recomputed."""
    updated = json.loads(json.dumps(manifest))
    for row in updated["figures"]:
        path = REPO / row["path"]
        if not path.is_file():
            continue
        blob = path.read_bytes()
        row["sha256"] = hashlib.sha256(blob).hexdigest()
        row["bytes"] = len(blob)
        row["used_in"] = used_in(path)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when the manifest is stale"
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    if manifest["schema"] != SCHEMA:
        print(f"unexpected schema: {manifest['schema']}", file=sys.stderr)
        return 1

    listed = {row["path"] for row in manifest["figures"]}
    present = {p.relative_to(REPO).as_posix() for p in figure_paths()}
    problems = [f"figure without a manifest row: {p}" for p in sorted(present - listed)]
    problems += [f"manifest row without a figure: {p}" for p in sorted(listed - present)]
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    updated = refreshed(manifest)
    if args.check:
        if updated != manifest:
            print(
                "docs/_static/figures/figures.json is stale; run "
                "python tools/update_figure_manifest.py",
                file=sys.stderr,
            )
            return 1
        return 0

    MANIFEST.write_text(json.dumps(updated, indent=2) + "\n")
    print(f"updated {MANIFEST.relative_to(REPO)} ({len(updated['figures'])} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
