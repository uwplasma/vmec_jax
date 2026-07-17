"""CLI free-boundary routing tests (``vmex.core.cli`` -> ``core.freeboundary``).

Covered:

1. the golden free-boundary deck (``input.cth_like_free_bdy_lasym_small``)
   through the CLI with a reduced iteration cap (``--max-iter 80`` — enough
   to activate the vacuum solve, golden turn-on iteration 53): the
   ``In VACUUM`` block and ``VACUUM PRESSURE TURNED ON`` banner appear, the
   wout is written and readable with ``lfreeb = True`` and the mgrid's
   ``nextcur``/``extcur``, and the exit code is 2 (MORE ITERATIONS
   REQUIRED — the capped run cannot converge but VMEC2000 still writes the
   wout);
2. missing mgrid file: warning + fixed-boundary fallback (VMEC2000 policy);
3. direct-coil conventions: ``MGRID_FILE = 'DIRECT_COILS'`` without
   ``--coils`` and ``--coils`` on a fixed-boundary deck are typed input
   errors (exit code 5).
"""

from __future__ import annotations

import contextlib
import io
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from vmex.core import cli
from vmex.core.errors import INPUT_ERROR_FLAG, MORE_ITER_FLAG
from vmex.core.mgrid import read_mgrid
from vmex.core.wout import read_wout

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
DECK = DATA_DIR / "input.cth_like_free_bdy_lasym_small"
MGRID = DATA_DIR / "mgrid_cth_like_lasym_small.nc"
CASE = "cth_like_free_bdy_lasym_small"
SOLOVEV_DECK = DATA_DIR / "input.solovev"

#: EXTCUR of the golden deck (HF, TVF).
DECK_EXTCUR = (-12.0, -2.55)


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


@pytest.fixture(scope="module")
def freeb_cli(tmp_path_factory) -> tuple[int, str, Path]:
    """One capped CLI free-boundary run of the golden deck (shared)."""
    jax.config.update("jax_disable_jit", False)
    workdir = tmp_path_factory.mktemp("cli_freeb")
    deck = workdir / DECK.name
    shutil.copyfile(DECK, deck)
    shutil.copyfile(MGRID, workdir / MGRID.name)
    outdir = workdir / "out"
    rc, stdout = _run_cli([str(deck), "--max-iter", "80", "--outdir", str(outdir)])
    return rc, stdout, outdir / f"wout_{CASE}.nc"


# ---------------------------------------------------------------------------
# free-boundary solve through the CLI
# ---------------------------------------------------------------------------


def test_vacuum_banners_printed(freeb_cli):
    _, stdout, _ = freeb_cli
    assert "In VACUUM" in stdout
    match = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)\s+ITERATIONS", stdout)
    assert match is not None, "missing VACUUM PRESSURE TURNED ON banner"
    # golden xvmec2000 turn-on is iteration 53; allow a small drift (the
    # trajectory is chaotic past activation, see test_freeboundary_ab).
    assert abs(int(match.group(1)) - 53) <= 5
    # free-boundary screen header carries the DEL-BSQ column.
    assert "DEL-BSQ" in stdout


def test_exit_code_reflects_more_iter(freeb_cli):
    rc, stdout, _ = freeb_cli
    assert rc == MORE_ITER_FLAG
    assert "MORE ITERATIONS REQUIRED" in stdout


def test_wout_written_with_free_boundary_fields(freeb_cli):
    _, _, wout_path = freeb_cli
    assert wout_path.exists(), "capped free-boundary run must still write the wout"
    # wrout.f dimensions extcur by the mgrid's nextcur (the bundled synthetic
    # mgrid holds a single summed coil group), truncating the deck's EXTCUR.
    mgrid = read_mgrid(MGRID)
    nextcur = int(mgrid.nextcur)
    extcur_expected = np.asarray(DECK_EXTCUR[:nextcur], dtype=float)
    wout = read_wout(wout_path)
    assert bool(wout.lfreeb) is True
    assert int(wout.ier_flag) == MORE_ITER_FLAG
    assert int(wout.nextcur) == nextcur
    np.testing.assert_allclose(np.asarray(wout.extcur), extcur_expected)
    assert wout.curlabel == tuple(mgrid.coil_groups)
    assert str(wout.mgrid_mode) == str(mgrid.mgrid_mode)
    assert MGRID.name in str(wout.mgrid_file)
    assert np.isfinite(np.asarray(wout.rmnc)).all()
    with netCDF4.Dataset(str(wout_path)) as ds:
        assert int(ds["lfreeb__logical__"][()]) == 1
        assert int(ds["nextcur"][()]) == nextcur
        np.testing.assert_allclose(np.asarray(ds["extcur"][:]), extcur_expected)
        # potvac is a documented gap: variables exist (netCDF fill) since
        # solve_free_boundary does not return the NESTOR potential yet.
        assert "potsin" in ds.variables


# ---------------------------------------------------------------------------
# missing mgrid -> fixed-boundary fallback (VMEC2000 policy)
# ---------------------------------------------------------------------------


def test_missing_mgrid_falls_back_to_fixed_boundary(tmp_path):
    deck = tmp_path / DECK.name
    shutil.copyfile(DECK, deck)  # mgrid deliberately not copied
    rc, stdout = _run_cli([str(deck), "--max-iter", "5", "--outdir", str(tmp_path)])
    assert "WARNING: mgrid file not found" in stdout
    assert "FIXED-BOUNDARY" in stdout
    assert "VACUUM PRESSURE TURNED ON" not in stdout
    assert "In VACUUM" not in stdout
    # the capped fixed-boundary fallback exhausts NITER -> exit code 2.
    assert rc == MORE_ITER_FLAG


# ---------------------------------------------------------------------------
# direct-coil conventions (typed input errors, no solve)
# ---------------------------------------------------------------------------


def test_direct_coils_deck_requires_coils_flag(tmp_path):
    text = DECK.read_text()
    text, count = re.subn(
        r"(?im)^\s*MGRID_FILE\s*=.*$", "  MGRID_FILE = 'DIRECT_COILS',", text, count=1
    )
    assert count == 1
    deck = tmp_path / DECK.name
    deck.write_text(text)
    rc, stdout = _run_cli([str(deck), "--outdir", str(tmp_path)])
    assert rc == INPUT_ERROR_FLAG
    assert "--coils" in stdout


def test_coils_flag_rejected_on_fixed_boundary_deck(tmp_path):
    rc, stdout = _run_cli([
        str(SOLOVEV_DECK), "--coils", str(tmp_path / "coils.json"),
        "--outdir", str(tmp_path),
    ])
    assert rc == INPUT_ERROR_FLAG
    assert "LFREEB" in stdout
