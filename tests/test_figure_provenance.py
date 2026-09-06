"""Every committed figure must say where it came from (plan 31.5 item 4).

Before this guard, 2 of 19 figures under ``docs/_static/figures`` were hashed
anywhere in the tree; the rest could be replaced, regenerated on a different
machine, or quietly drift from the numbers they plot, and nothing would fail.
``docs/_static/figures/figures.json`` now carries one row per figure — path,
sha256, byte count, generator, inputs, generation date, hardware, which pages
embed it, what it takes to rebuild it, and whether the committed bytes were
actually re-derived — and these tests fail when a figure and its row
disagree.

The manifest's mechanical fields are rewritten by
``python tools/update_figure_manifest.py``; the provenance fields are authored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "_static" / "figures"
MANIFEST = FIGURE_DIR / "figures.json"

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REPRODUCIBLE = {"command", "manual", "external-data"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_manifest_covers_every_figure_and_nothing_else(manifest: dict) -> None:
    """A figure may not be added or removed without its provenance row."""
    assert manifest["schema"] == "vmex.figure-provenance/1"
    listed = [row["path"] for row in manifest["figures"]]
    assert len(listed) == len(set(listed)), "duplicate manifest rows"
    present = sorted(
        p.relative_to(ROOT).as_posix() for p in FIGURE_DIR.glob("*.webp")
    )
    assert sorted(listed) == present


def test_every_figure_matches_its_recorded_hash(manifest: dict) -> None:
    """The guard the plan asked for: a changed figure fails until its row moves."""
    import hashlib

    stale = []
    for row in manifest["figures"]:
        blob = (ROOT / row["path"]).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != row["sha256"] or len(blob) != row["bytes"]:
            stale.append(
                f"{row['path']}: file is {digest[:12]}/{len(blob)}B, "
                f"manifest says {row['sha256'][:12]}/{row['bytes']}B"
            )
    assert not stale, (
        "figure changed without its manifest row; run "
        "python tools/update_figure_manifest.py\n  " + "\n  ".join(stale)
    )


def test_provenance_fields_are_present_and_point_at_real_files(
    manifest: dict,
) -> None:
    """A generator, input, or record that does not exist is not provenance."""
    problems = []
    for row in manifest["figures"]:
        where = row["path"]
        if not ISO_DATE.match(row["generated"]):
            problems.append(f"{where}: generated={row['generated']!r} is not ISO")
        if row["reproducible"] not in REPRODUCIBLE:
            problems.append(f"{where}: reproducible={row['reproducible']!r}")
        if not isinstance(row["bytes_verified"], bool):
            problems.append(f"{where}: bytes_verified is not a boolean")
        if row["bytes_verified"] and row["reproducible"] != "command":
            problems.append(
                f"{where}: bytes_verified without a reproducing command"
            )
        if not row.get("note"):
            problems.append(f"{where}: empty note")
        if not row.get("command"):
            problems.append(f"{where}: no command recorded")
        for key in ("generator", "record"):
            target = row[key]
            if target is not None and not (ROOT / target).exists():
                problems.append(f"{where}: {key} {target} does not exist")
        if row["generator"] is None:
            problems.append(f"{where}: no generator; generate the figure or delete it")
        for source in row["inputs"]:
            if not (ROOT / source).exists():
                problems.append(f"{where}: input {source} does not exist")
    assert not problems, "\n  " + "\n  ".join(problems)


def test_recorded_usage_matches_the_docs_tree(manifest: dict) -> None:
    """``used_in`` is derived, so it cannot rot into a wrong citation."""
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    import update_figure_manifest as tool

    wrong = []
    for row in manifest["figures"]:
        actual = tool.used_in(ROOT / row["path"])
        if actual != row["used_in"]:
            wrong.append(f"{row['path']}: recorded {row['used_in']}, found {actual}")
    assert not wrong, (
        "run python tools/update_figure_manifest.py\n  " + "\n  ".join(wrong)
    )


def test_every_figure_the_docs_embed_is_in_the_manifest(manifest: dict) -> None:
    """No page may embed an image that has no provenance row."""
    listed = {Path(row["path"]).name for row in manifest["figures"]}
    embedded: set[str] = set()
    pages = [ROOT / "README.md"]
    pages += [
        p
        for p in (ROOT / "docs").rglob("*")
        if p.suffix in (".md", ".rst") and "_build" not in p.parts
    ]
    for page in pages:
        for name in re.findall(r"[\w./-]*?([\w-]+\.webp)", page.read_text()):
            embedded.add(name)
    assert embedded <= listed, f"figures embedded but unrecorded: {embedded - listed}"


def test_preconditioner_table_matches_its_measurement(manifest: dict) -> None:
    """The 2D-preconditioner figure and prose quote the committed run.

    The bar heights and the ``performance.rst`` table used to be literals
    typed in two places, sourced only from a commit message. Both now read
    from ``benchmarks/preconditioner_2d_stiff_cases.json``, and this pins
    them together so the page cannot drift from the artifact again.
    """
    record = json.loads(
        (ROOT / "benchmarks" / "preconditioner_2d_stiff_cases.json").read_text()
    )
    assert record["schema"] == "vmex.preconditioner-2d-stiff-cases/2"

    provenance = record["provenance"]
    assert provenance["measurement_dirty"] is False
    for key in ("measurement_commit", "measurement_date", "host", "protocol"):
        assert provenance[key], f"provenance.{key} is empty"
    assert provenance["versions"]["jax"]

    page = (ROOT / "docs" / "reference" / "performance.rst").read_text()
    for case in record["cases"]:
        assert case["iterations_2d"] < case["iterations_1d"]
        # both paths land on the same equilibrium, so wb is the parity witness
        assert case["wb_relative_difference"] < 1e-5
        if not case["in_readme_figure"]:
            continue
        for value in (case["iterations_1d"], case["iterations_2d"]):
            assert f"     - {value}\n" in page, (
                f"performance.rst does not quote {value} for {case['key']}; "
                "the table drifted from benchmarks/preconditioner_2d_stiff_cases.json"
            )

    row = next(
        r for r in manifest["figures"] if r["path"].endswith("readme_precond.webp")
    )
    assert row["record"] == "benchmarks/preconditioner_2d_stiff_cases.json"


def test_diagnostics_figure_matches_its_recorded_solve(manifest: dict) -> None:
    """The regenerated README diagnostics panel names the deck it plots."""
    record = json.loads((FIGURE_DIR / "readme_diagnostics.json").read_text())
    assert record["schema"] == "vmex.readme-diagnostics-figures/1"
    assert record["provenance"]["measurement_dirty"] is False

    import hashlib

    for case in record["cases"]:
        figure = ROOT / case["figure"]
        assert hashlib.sha256(figure.read_bytes()).hexdigest() == case["figure_sha256"]
        deck = ROOT / case["deck"]
        assert deck.is_file()
        assert hashlib.sha256(deck.read_bytes()).hexdigest() == case["deck_sha256"]
        # the README quotes the beta this solve reached
        beta = case["scalars"]["betatotal"]
        readme = (ROOT / "README.md").read_text()
        assert f"{100 * beta:.2f}" in readme, (
            f"README does not quote the measured beta {100 * beta:.2f}% "
            f"for {case['key']}"
        )
