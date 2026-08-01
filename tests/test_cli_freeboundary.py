"""CLI free-boundary routing tests (``vmex.core.cli`` ->
``core.freeboundary``): the golden lasym deck at ``--max-iter 80`` (past
the golden turn-on iteration 53) shows the ``In VACUUM`` block and
activation banner, uses ``LFULL3D1OUT=T`` to write a readable
``lfreeb = True`` wout with the mgrid's ``nextcur``/``extcur``, and exits
2; a
missing mgrid warns and falls back to fixed boundary (VMEC2000 policy);
direct-coil misuse is a typed input error (exit code 5).
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
from vmex.core.errors import (
    INPUT_ERROR_FLAG,
    MORE_ITER_FLAG,
    VmecConvergenceError,
    WERROR_MESSAGES,
)
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
    """One capped ``LFULL3D1OUT=T`` free-boundary run (shared)."""
    jax.config.update("jax_disable_jit", False)
    workdir = tmp_path_factory.mktemp("cli_freeb")
    deck = workdir / DECK.name
    text, count = re.subn(
        r"(?m)^\s*/\s*$", "  LFULL3D1OUT = T,\n/", DECK.read_text(), count=1
    )
    assert count == 1
    deck.write_text(text)
    shutil.copyfile(MGRID, workdir / MGRID.name)
    outdir = workdir / "out"
    rc, stdout = _run_cli([str(deck), "--max-iter", "80", "--outdir", str(outdir)])
    return rc, stdout, outdir / f"wout_{CASE}.nc"


# ---------------------------------------------------------------------------
# free-boundary solve through the CLI
# ---------------------------------------------------------------------------


def test_redirected_stdout_streams_banner_and_compile_notice(tmp_path):
    """Cluster-log contract: with stdout redirected to a FILE, the ``NS =``
    stage banner and a free-lane ``compiling NS = ...`` notice must be
    readable in that file WHILE the solve is still running.

    This pins two independent fixes: the CLI sink flushes every line
    (otherwise block buffering hides hours of output on a batch node), and
    every free-lane compile prints its attribution BEFORE the compile
    starts (otherwise a large-grid run sits silently inside XLA).  The
    subprocess is killed as soon as both lines are observed live.
    """
    import os
    import subprocess
    import sys
    import time

    deck = tmp_path / DECK.name
    deck.write_text(DECK.read_text())
    shutil.copyfile(MGRID, tmp_path / MGRID.name)
    log = tmp_path / "run.log"

    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(Path(__file__).resolve().parents[1]),
                    env.get("PYTHONPATH")) if p
    )
    with open(log, "wb") as sink:
        proc = subprocess.Popen(
            [sys.executable, "-m", "vmex.core.cli", str(deck),
             "--max-iter", "40", "--outdir", str(tmp_path / "out")],
            stdout=sink, stderr=subprocess.DEVNULL, cwd=str(tmp_path),
            env=env,
        )
        try:
            seen_live = False
            deadline = time.monotonic() + 600.0
            while proc.poll() is None and time.monotonic() < deadline:
                text = log.read_text(errors="replace")
                if "NS = " in text and " compiling NS =" in text:
                    seen_live = True
                    break
                time.sleep(0.05)
        finally:
            proc.kill()
            proc.wait()
    assert seen_live, (
        "NS banner + compile notice never reached the redirected log while "
        "the run was still executing; log contents:\n"
        + log.read_text(errors="replace")[:4000]
    )
    # the observed notice is a tagged free-lane one, not a stray fragment
    assert re.search(r" compiling NS = *\d+ .*executable\.\.\.",
                     log.read_text(errors="replace"))


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
    assert wout_path.exists(), (
        "LFULL3D1OUT=T must retain the capped free-boundary state"
    )
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
        label_count_dim = f"dim_{nextcur:05d}"
        assert ds["curlabel"].dimensions == (label_count_dim, "current_label")
        assert ds["curlabel"].shape == (nextcur, 30)
        for name in (
            "potsin",
            "potcos",
            "bsubumnc_sur",
            "bsubvmnc_sur",
            "bsupumnc_sur",
            "bsupvmnc_sur",
            "bsubumns_sur",
            "bsubvmns_sur",
            "bsupumns_sur",
            "bsupvmns_sur",
        ):
            values = ds[name][:]
            assert not np.ma.getmaskarray(values).any(), name
            assert np.isfinite(np.asarray(values)).all(), name


# ---------------------------------------------------------------------------
# missing mgrid -> fixed-boundary fallback (VMEC2000 policy)
# ---------------------------------------------------------------------------


def test_missing_mgrid_falls_back_to_fixed_boundary_without_wout(tmp_path):
    deck = tmp_path / DECK.name
    shutil.copyfile(DECK, deck)  # mgrid deliberately not copied
    rc, stdout = _run_cli([str(deck), "--max-iter", "5", "--outdir", str(tmp_path)])
    assert "WARNING: mgrid file not found" in stdout
    assert "FIXED-BOUNDARY" in stdout
    assert "VACUUM PRESSURE TURNED ON" not in stdout
    assert "In VACUUM" not in stdout
    # The capped fixed-boundary fallback exhausts NITER.  With the deck's
    # default LFULL3D1OUT=F, VMEC2000 returns ier=2 before fileout.
    assert rc == MORE_ITER_FLAG
    assert not (tmp_path / f"wout_{CASE}.nc").exists()


def test_free_boundary_default_raises_before_wout(monkeypatch, tmp_path):
    """The free solver receives the same LFULL3D1OUT gate as fixed boundary."""
    deck = tmp_path / DECK.name
    shutil.copyfile(DECK, deck)
    shutil.copyfile(MGRID, tmp_path / MGRID.name)
    seen = {}

    def fake_solve(_inp, **kwargs):
        seen["raise_on_max_iterations"] = kwargs["raise_on_max_iterations"]
        raise VmecConvergenceError(
            WERROR_MESSAGES[MORE_ITER_FLAG],
            hint="increase NITER or loosen FTOL",
            iteration=1,
            fsq=(1.0, 1.0, 1.0),
            ftol=1.0e-12,
        )

    import vmex.core.multigrid as multigrid

    monkeypatch.setattr(multigrid, "solve_free_boundary_multigrid", fake_solve)
    rc, stdout = _run_cli([str(deck), "--outdir", str(tmp_path)])
    assert seen == {"raise_on_max_iterations": True}
    assert rc == MORE_ITER_FLAG
    assert WERROR_MESSAGES[MORE_ITER_FLAG] in stdout
    assert not (tmp_path / f"wout_{CASE}.nc").exists()


def test_missing_mgrid_forced_wout_has_effective_fixed_metadata(tmp_path):
    """LFULL3D1OUT retains the fallback state but must not label it free."""
    deck = tmp_path / DECK.name
    text, count = re.subn(
        r"(?m)^\s*/\s*$", "  LFULL3D1OUT = T,\n/", DECK.read_text(), count=1
    )
    assert count == 1
    deck.write_text(text)  # mgrid deliberately absent
    rc, stdout = _run_cli(
        [str(deck), "--max-iter", "5", "--outdir", str(tmp_path)]
    )
    assert rc == MORE_ITER_FLAG
    assert "FIXED-BOUNDARY" in stdout
    wout = read_wout(tmp_path / f"wout_{CASE}.nc")
    assert bool(wout.lfreeb) is False
    with netCDF4.Dataset(str(tmp_path / f"wout_{CASE}.nc")) as ds:
        assert int(ds["lfreeb__logical__"][()]) == 0


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
