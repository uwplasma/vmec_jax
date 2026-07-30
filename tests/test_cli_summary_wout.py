"""Print-vs-wout consistency sweep for the CLI equilibrium summary.

For each deck the CLI is run in-process, the wout it wrote is read back with
netCDF4, and every printed summary quantity must equal the wout value at
print precision — i.e. re-formatting the netCDF value with the exact format
the summary uses must reproduce the printed field byte-for-byte.  This pins
the summary against the file forever: any drift in either the printing or
the wout path (including the E-notation iota lines, which fixed-point %f
used to flatten to ``-0.000000``) breaks the sweep.

Kept cheap for the ordinary lane: solovev runs its stock 500-iteration
budget (~215 to converge) and nfp2_QA runs at a reduced budget
(``--ftol 1e-11 --max-iter 500``; converges by ~200 iterations).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

netCDF4 = pytest.importorskip("netCDF4")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from vmex.core import cli

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

#: deck name -> extra CLI args (reduced budgets keep the ordinary lane cheap).
DECKS = {
    "solovev": [],
    "nfp2_QA": ["--ftol", "1e-11", "--max-iter", "500"],
}

#: summary label -> (wout value from the open netCDF dataset, print format).
SUMMARY_CHECKS = [
    ("Aspect Ratio", lambda ds: float(ds["aspect"][()]), "14.6f"),
    ("Plasma Volume", lambda ds: float(ds["volume_p"][()]), "14.6f"),
    ("Major Radius", lambda ds: float(ds["Rmajor_p"][()]), "14.6f"),
    ("Minor Radius", lambda ds: float(ds["Aminor_p"][()]), "14.6f"),
    ("Volume Average B", lambda ds: float(ds["volavgB"][()]), "14.6f"),
    ("Iota on Axis", lambda ds: float(ds["iotaf"][0]), "14.6E"),
    ("Iota at Edge", lambda ds: float(ds["iotaf"][-1]), "14.6E"),
    ("|B| on Axis (b0)", lambda ds: float(ds["b0"][()]), "14.6f"),
    ("<|B|> at Edge (half)", lambda ds: float(ds["bmnc"][-1, 0]), "14.6f"),
    ("beta total", lambda ds: float(ds["betatotal"][()]), "14.6E"),
    (
        "MHD Energy (wb + wp)",
        lambda ds: float(ds["wb"][()]) + float(ds["wp"][()]),
        "14.6E",
    ),
]


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Run ``cli.main`` in-process, capturing stdout."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = cli.main(argv)
    return int(rc), buffer.getvalue()


def _printed_field(stdout: str, label: str) -> str:
    """The value text of one summary line (units bracket stripped)."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith(f" {label} ")]
    assert len(lines) == 1, f"summary line {label!r} not found exactly once"
    return lines[0].split("=", 1)[1].split("[")[0].strip()


@pytest.fixture(scope="module", params=sorted(DECKS))
def deck_run(request, tmp_path_factory) -> tuple[str, str, Path]:
    """One CLI solve per deck: (case, stdout, wout_path); rc must be 0."""
    case = request.param
    outdir = tmp_path_factory.mktemp(f"summary_{case}")
    rc, stdout = _run_cli(
        [str(DATA_DIR / f"input.{case}"), "--outdir", str(outdir)] + DECKS[case]
    )
    assert rc == 0, f"CLI solve of input.{case} failed (rc={rc}):\n{stdout}"
    wout_path = outdir / f"wout_{case}.nc"
    assert wout_path.exists()
    return case, stdout, wout_path


def test_summary_matches_wout_at_print_precision(deck_run):
    """Every printed summary quantity == the wout value at print precision."""
    case, stdout, wout_path = deck_run
    with netCDF4.Dataset(str(wout_path)) as ds:
        for label, wout_value, fmt in SUMMARY_CHECKS:
            printed = _printed_field(stdout, label)
            expected = format(wout_value(ds), fmt).strip()
            assert printed == expected, (
                f"input.{case}: summary {label!r} prints {printed!r} but the "
                f"wout value formats to {expected!r}"
            )


def test_tiny_iota_is_not_flattened_to_zero(deck_run):
    """E-notation keeps ~1e-10 iota visible (the -0.000000 regression)."""
    case, stdout, wout_path = deck_run
    with netCDF4.Dataset(str(wout_path)) as ds:
        iotaf0 = float(ds["iotaf"][0])
    if iotaf0 == 0.0:  # pragma: no cover - neither bundled deck hits this
        pytest.skip("deck has identically-zero axis iota")
    printed = _printed_field(stdout, "Iota on Axis")
    assert float(printed) != 0.0
    assert "E" in printed


def test_stdout_hygiene(deck_run):
    """No stacked blank lines and no stale preconditioned legend."""
    _, stdout, _ = deck_run
    assert "\n\n\n" not in stdout
    assert "Preconditioned" not in stdout
