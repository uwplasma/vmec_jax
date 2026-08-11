"""End-to-end tests for the new-core ``vmec`` CLI (``vmex.core.cli``):
``vmec input.solovev`` writes a readable wout with ``wb`` at golden parity
(1e-8); stdout structure matches the golden ``xvmec2000`` capture at print
precision; ``--plot``/``--booz`` smoke; JSON input solves to the same
``wb``; ``--test`` smoke; and zero-crash exit codes (unreadable input ->
``ier_flag = 5`` with the werror INPUT message; NITER exhaustion ->
``ier_flag = 2`` with the WOUT written — fileout.f semantics).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import re
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from vmex.core import cli
from vmex.core.errors import INPUT_ERROR_FLAG, MORE_ITER_FLAG, WERROR_MESSAGES
from vmex.core.input import VmecInput
from vmex.core.wout import read_wout

from tests.conftest import resolve_golden_dir

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
SOLOVEV_DECK = DATA_DIR / "input.solovev"
GOLDEN_DIR = resolve_golden_dir()

_ITER_ROW = re.compile(r"^\d\.\d{2}E[+-]\d{2}$")


@pytest.fixture(autouse=True)
def _enable_jit():
    """Full solves need JIT (the repo conftest disables it for unit tests)."""
    jax.config.update("jax_disable_jit", False)
    yield


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Run ``cli.main`` in-process, capturing stdout."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = cli.main(argv)
    return int(rc), buffer.getvalue()


def _iteration_rows(text: str) -> list[tuple[int, list[float]]]:
    """Parse VMEC2000 screen-format iteration rows from console output."""
    rows = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 6 and tokens[0].isdigit() and _ITER_ROW.match(tokens[1]):
            rows.append((int(tokens[0]), [float(tok) for tok in tokens[1:]]))
    return rows


def _line_containing(text: str, pattern: str) -> str | None:
    return next((ln for ln in text.splitlines() if pattern in ln), None)


@pytest.fixture(scope="module")
def solovev_cli(tmp_path_factory) -> tuple[int, str, Path]:
    """One CLI solve of the solovev deck, shared by the checks below."""
    jax.config.update("jax_disable_jit", False)
    outdir = tmp_path_factory.mktemp("solovev_cli")
    rc, stdout = _run_cli([str(SOLOVEV_DECK), "--outdir", str(outdir)])
    return rc, stdout, outdir / "wout_solovev.nc"


# ---------------------------------------------------------------------------
# solve -> wout
# ---------------------------------------------------------------------------


def test_solovev_run_writes_readable_wout(solovev_cli):
    rc, _, wout_path = solovev_cli
    assert rc == 0
    assert wout_path.exists()
    wout = read_wout(wout_path)
    assert int(wout.ns) == 11
    assert int(wout.mnmax) == 6
    assert int(wout.nfp) == 1
    assert int(wout.ier_flag) == 0
    assert np.isfinite(np.asarray(wout.rmnc)).all()


@pytest.mark.skipif(GOLDEN_DIR is None, reason="golden VMEC2000 fixtures unavailable")
def test_solovev_wb_matches_golden(solovev_cli):
    _, _, wout_path = solovev_cli
    wout = read_wout(wout_path)
    with netCDF4.Dataset(str(GOLDEN_DIR / "solovev" / "wout_solovev.nc")) as gd:
        wb_gold = float(gd["wb"][()])
    assert abs(float(wout.wb) / wb_gold - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# stdout structure vs the golden xvmec2000 capture
# ---------------------------------------------------------------------------


@pytest.mark.skipif(GOLDEN_DIR is None, reason="golden VMEC2000 fixtures unavailable")
def test_solovev_stdout_structure_matches_golden(solovev_cli):
    _, stdout, _ = solovev_cli
    golden = (GOLDEN_DIR / "solovev" / "stdout.txt").read_text()

    for pattern in (
        "- - - -",
        "SEQ =",
        "PROCESSING INPUT.solovev",
        "THIS IS",
        "NS = ",
        "EXECUTION TERMINATED NORMALLY",
        "FILE : solovev",
        "NUMBER OF JACOBIAN RESETS",
        "TOTAL COMPUTATIONAL TIME",
    ):
        assert _line_containing(stdout, pattern) is not None, f"missing banner: {pattern!r}"
        assert _line_containing(golden, pattern) is not None

    # NS-stage banner and iteration header are byte-identical to VMEC2000.
    assert _line_containing(stdout, "NS = ") == _line_containing(golden, "NS = ")
    header = _line_containing(golden, "ITER    FSQR")
    assert _line_containing(stdout, "ITER    FSQR") == header

    ours, gold = _iteration_rows(stdout), _iteration_rows(golden)
    assert ours and gold
    # First iteration row: same column count, values equal at print precision.
    assert ours[0][0] == gold[0][0] == 1
    assert len(ours[0][1]) == len(gold[0][1])
    np.testing.assert_allclose(ours[0][1], gold[0][1], rtol=5e-2)
    # Final row: same column count, iteration count near golden (215 +- 20%),
    # residual columns at/below the printed golden values' magnitude.
    assert len(ours[-1][1]) == len(gold[-1][1])
    assert 0.8 * gold[-1][0] <= ours[-1][0] <= 1.2 * gold[-1][0]
    for k in range(3):  # FSQR, FSQZ, FSQL all converged below ftol
        assert ours[-1][1][k] <= 1.1e-14
    np.testing.assert_allclose(ours[-1][1][-1], gold[-1][1][-1], rtol=1e-3)  # WMHD


def test_summary_reports_iota_and_modb(solovev_cli):
    """The equilibrium summary carries the iota and |B| axis/edge lines."""
    _, stdout, _ = solovev_cli
    for pattern in (
        " Iota on Axis          = ",
        " Iota at Edge          = ",
        " |B| on Axis (b0)      = ",
        " <|B|> at Edge (half)  = ",
    ):
        line = _line_containing(stdout, pattern)
        assert line is not None, f"missing summary line: {pattern!r}"
        assert np.isfinite(float(line.split("=")[1].split("[")[0]))
    # iota can be legitimately tiny (near-axisymmetric decks: ~1e-10), so the
    # summary prints it in E-notation; fixed-point %f would show -0.000000.
    for pattern in (" Iota on Axis ", " Iota at Edge "):
        value_text = _line_containing(stdout, pattern).split("=")[1].strip()
        assert "E" in value_text, f"iota not in E-notation: {value_text!r}"


def test_stdout_has_no_consecutive_blank_lines(solovev_cli):
    """At most one blank line between blocks anywhere in the CLI output."""
    _, stdout, _ = solovev_cli
    assert "\n\n\n" not in stdout


def test_no_stale_preconditioned_legend(solovev_cli):
    """The screen path never prints lowercase preconditioned rows, so the
    legend must not announce them (threed1-file-only, printout.f FORMAT 40)."""
    _, stdout, _ = solovev_cli
    assert "Preconditioned" not in stdout
    assert _line_containing(stdout, "FSQR, FSQZ = Normalized") is not None


# ---------------------------------------------------------------------------
# --plot / --booz on the produced wout
# ---------------------------------------------------------------------------


def test_plot_wout_smoke(solovev_cli, tmp_path):
    _, _, wout_path = solovev_cli
    rc, _ = _run_cli(["--plot", str(wout_path), "--outdir", str(tmp_path), "--quiet"])
    assert rc == 0
    pngs = sorted(p.name for p in tmp_path.glob("*.png"))
    assert pngs == [
        "solovev_boundary3d.png", "solovev_modB.png", "solovev_profiles.png",
        "solovev_stability.png", "solovev_summary.png", "solovev_surfaces.png",
    ]
    assert not list(tmp_path.glob("boozmn_*.nc"))  # in-process Boozer panels need no --booz


def test_booz_and_plot_boozmn_smoke(solovev_cli, tmp_path):
    _, _, wout_path = solovev_cli
    rc, _ = _run_cli(
        [
            str(wout_path), "--booz", "--mbooz", "8", "--nbooz", "8",
            "--booz-surfaces", "0.5", "--outdir", str(tmp_path), "--quiet",
        ]
    )
    assert rc == 0
    boozmn = tmp_path / "boozmn_solovev.nc"
    assert boozmn.exists()
    rc, _ = _run_cli(["--plot", str(boozmn), "--outdir", str(tmp_path), "--quiet"])
    assert rc == 0
    assert (tmp_path / "boozmn_solovev_modB.png").exists()


# ---------------------------------------------------------------------------
# VMEC++-style JSON input
# ---------------------------------------------------------------------------


def test_json_input_solves_to_same_wb(solovev_cli, tmp_path):
    _, _, wout_path = solovev_cli
    json_deck = VmecInput.from_file(SOLOVEV_DECK).to_json(tmp_path / "solovev.json")
    rc, _ = _run_cli([str(json_deck), "--outdir", str(tmp_path), "--quiet"])
    assert rc == 0
    wb_json = float(read_wout(tmp_path / "wout_solovev.nc").wb)
    wb_indata = float(read_wout(wout_path).wb)
    assert abs(wb_json / wb_indata - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# --test bundled smoke (reduced tolerance for CI speed)
# ---------------------------------------------------------------------------


def test_bundled_test_smoke(tmp_path):
    rc, _ = _run_cli(["--test", "--outdir", str(tmp_path), "--ftol", "1e-8", "--quiet"])
    assert rc == 0
    wout = read_wout(tmp_path / "wout_nfp4_QH_warm_start.nc")
    assert int(wout.ns) == 35
    figures = sorted(p.name for p in (tmp_path / "figures").glob("*.png"))
    assert len(figures) == 5, figures


# ---------------------------------------------------------------------------
# zero-crash exit codes
# ---------------------------------------------------------------------------


def test_unreadable_input_exits_with_input_error(tmp_path):
    bad = tmp_path / "input.bad"
    bad.write_text("this is not a namelist\n")
    rc, stdout = _run_cli([str(bad)])
    assert rc == INPUT_ERROR_FLAG
    assert WERROR_MESSAGES[INPUT_ERROR_FLAG] in stdout


def test_missing_input_exits_with_input_error(tmp_path):
    rc, stdout = _run_cli([str(tmp_path / "input.does_not_exist")])
    assert rc == INPUT_ERROR_FLAG
    assert WERROR_MESSAGES[INPUT_ERROR_FLAG] in stdout


def test_iteration_exhaustion_without_lfull3d1out_does_not_write_wout(tmp_path):
    """VMEC2000: ordinary NITER exhaustion returns ier=2 before fileout."""
    rc, stdout = _run_cli(
        [str(SOLOVEV_DECK), "--outdir", str(tmp_path), "--max-iter", "20"]
    )
    assert rc == MORE_ITER_FLAG
    assert WERROR_MESSAGES[MORE_ITER_FLAG] in stdout
    assert "Wrote WOUT file:" not in stdout
    assert not (tmp_path / "wout_solovev.nc").exists()


def test_lfull3d1out_writes_wout_on_iteration_exhaustion(tmp_path):
    """LFULL3D1OUT=T forces VMEC2000's full-output path for ier=2."""
    inp = dataclasses.replace(
        VmecInput.from_file(SOLOVEV_DECK),
        lfull3d1out=True,
    )
    deck = inp.to_indata(tmp_path / "input.lfull3d1out")
    rc, stdout = _run_cli(
        [str(deck), "--outdir", str(tmp_path), "--max-iter", "20"]
    )
    assert rc == MORE_ITER_FLAG
    for pattern in (
        "Aspect Ratio", "Volume Average B", "Iota on Axis", "Iota at Edge",
        "|B| on Axis (b0)", "<|B|> at Edge (half)", "MHD Energy (wb + wp)",
        "NUMBER OF JACOBIAN RESETS", "TOTAL COMPUTATIONAL TIME",
        "Wrote WOUT file:", "HINT : increase NITER or loosen FTOL",
    ):
        assert _line_containing(stdout, pattern) is not None, pattern
    wout = read_wout(tmp_path / "wout_lfull3d1out.nc")
    assert int(wout.ier_flag) == MORE_ITER_FLAG


def test_iteration_exhaustion_quiet_without_lfull3d1out_has_no_wout(tmp_path):
    rc, stdout = _run_cli(
        [str(SOLOVEV_DECK), "--outdir", str(tmp_path), "--max-iter", "20", "--quiet"]
    )
    assert rc == MORE_ITER_FLAG
    # Typed termination messages remain visible even under --quiet.
    assert WERROR_MESSAGES[MORE_ITER_FLAG] in stdout
    assert not (tmp_path / "wout_solovev.nc").exists()


def test_lforbal_iteration_exhaustion_writes_wout(tmp_path):
    """LFORBAL=T is solved, not ignored, and keeps VMEC's NITER WOUT policy."""
    inp = dataclasses.replace(
        VmecInput.from_file(SOLOVEV_DECK),
        lforbal=True,
        lfull3d1out=True,
        lmove_axis=False,
        nstep=1,
        niter_array=np.asarray([3]),
        ftol_array=np.asarray([1.0e-30]),
    )
    deck = inp.to_indata(tmp_path / "input.lforbal")
    rc, stdout = _run_cli([str(deck), "--outdir", str(tmp_path)])
    assert rc == MORE_ITER_FLAG
    rows = _iteration_rows(stdout)
    assert [row[1][:3] for row in rows] == [
        [8.33e-2, 4.94e-4, 3.21e-2],
        [6.82e-3, 1.41e-3, 4.37e-3],
        [1.52e-2, 9.15e-4, 6.98e-3],
    ]
    wout_path = tmp_path / "wout_lforbal.nc"
    assert wout_path.exists()
    wout = read_wout(wout_path)
    assert int(wout.ier_flag) == MORE_ITER_FLAG
    assert bool(wout.lmove_axis) is False
