"""Opt-in end-to-end comparison against a locally installed VMEC2000."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vmex.core.wout import read_wout
from vmex.core.mgrid import write_mgrid

from tests.test_lasym_free_case import (
    lasym_free_input,
    lasym_free_mgrid_data,
)

pytestmark = [
    pytest.mark.vmec2000_live,
    pytest.mark.usefixtures("_module_jit_enabled"),
]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"


def _executable(pytestconfig) -> Path:
    configured = str(pytestconfig.getoption("--vmec2000-executable")).strip()
    candidates = [Path(configured) if configured else None]
    discovered = shutil.which("xvmec2000")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    pytest.fail(
        "--run-vmec2000 requested but xvmec2000 was not found; pass "
        "--vmec2000-executable PATH"
    )


def _run(command: list[str], *, cwd: Path, timeout: int = 300) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{' '.join(command)} failed with {completed.returncode}\n"
        f"stdout:\n{completed.stdout[-4000:]}\n"
        f"stderr:\n{completed.stderr[-4000:]}"
    )


@pytest.mark.parametrize(
    "deck_name", ["input.solovev", "input.li383_low_res"]
)
def test_live_vmec2000_fixed_boundary_parity(
    pytestconfig, tmp_path, deck_name
):
    """Both public CLIs solve the same finite-beta deck and agree."""
    vmec2000_dir = tmp_path / "vmec2000"
    vmex_dir = tmp_path / "vmex"
    vmec2000_dir.mkdir()
    vmex_dir.mkdir()
    deck = DATA / deck_name
    vmec2000_deck = vmec2000_dir / deck.name
    vmex_deck = vmex_dir / deck.name
    shutil.copy2(deck, vmec2000_deck)
    shutil.copy2(deck, vmex_deck)

    _run([str(_executable(pytestconfig)), vmec2000_deck.name], cwd=vmec2000_dir)
    _run(
        [
            sys.executable,
            "-m",
            "vmex.core.cli",
            str(vmex_deck),
            "--outdir",
            str(vmex_dir),
        ],
        cwd=ROOT,
    )

    suffix = deck.name.removeprefix("input.")
    reference = read_wout(vmec2000_dir / f"wout_{suffix}.nc")
    actual = read_wout(vmex_dir / f"wout_{suffix}.nc")
    assert int(actual.ier_flag) == int(reference.ier_flag) == 0
    assert int(actual.ns) == int(reference.ns)
    for name in ("wb", "wp", "volume_p", "aspect"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(reference, name),
            rtol=1e-8,
            err_msg=name,
        )
    for name in ("iotaf", "rmnc", "zmns", "lmns"):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, name)),
            np.asarray(getattr(reference, name)),
            rtol=1e-6,
            atol=1e-10,
            err_msg=name,
        )
    for name, rtol in (
        # Low-resolution bsubv filtering makes this derivative-based profile
        # noisier than the equilibrium and bdotb fields.
        ("jdotb", 1.0e-3),
        ("bdotb", 1.0e-5),
        ("DMerc", 0.05),
        ("DShear", 0.05),
        ("DCurr", 0.05),
        ("DWell", 1.0e-8),
        # Solovev's ns=11 first validated DGeod points differ by up to 7.8%;
        # the profile amplitude and stability sign remain matched.
        ("DGeod", 0.1),
    ):
        expected = np.asarray(getattr(reference, name))[2:-1]
        scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
        np.testing.assert_allclose(
            np.asarray(getattr(actual, name))[2:-1],
            expected,
            rtol=rtol,
            atol=1e-10 * scale,
            err_msg=name,
        )
    assert np.array_equal(
        np.sign(np.asarray(actual.DMerc)[2:-1]),
        np.sign(np.asarray(reference.DMerc)[2:-1]),
    )


def test_live_vmec2000_converged_lasym_free_boundary(pytestconfig, tmp_path):
    """Converged LASYM geometry, vacuum potential, and surface fields agree."""
    vmec2000_dir = tmp_path / "vmec2000_lasym"
    vmex_dir = tmp_path / "vmex_lasym"
    vmec2000_dir.mkdir()
    vmex_dir.mkdir()
    inp = lasym_free_input(DATA)
    for directory in (vmec2000_dir, vmex_dir):
        inp.to_indata(directory / "input.diii_lasym")
        write_mgrid(
            directory / inp.mgrid_file,
            lasym_free_mgrid_data(),
        )

    _run(
        [str(_executable(pytestconfig)), "input.diii_lasym"],
        cwd=vmec2000_dir,
        timeout=600,
    )
    _run(
        [
            sys.executable,
            "-m",
            "vmex.core.cli",
            str(vmex_dir / "input.diii_lasym"),
            "--outdir",
            str(vmex_dir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        timeout=600,
    )

    reference = read_wout(next(vmec2000_dir.glob("wout*.nc")))
    actual = read_wout(next(vmex_dir.glob("wout*.nc")))
    assert int(actual.ier_flag) == int(reference.ier_flag) == 0
    assert bool(actual.lasym) and bool(reference.lasym)
    geometry_limits = {
        "rmnc": 2.0e-6,
        "zmns": 2.0e-6,
        "rmns": 4.0e-3,
        "zmnc": 4.0e-3,
    }
    for name, limit in geometry_limits.items():
        expected = np.asarray(getattr(reference, name))
        scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
        error = float(np.max(np.abs(
            np.asarray(getattr(actual, name)) - expected
        ))) / scale
        assert error < limit, (name, error)
    for name in ("xmpot", "xnpot"):
        np.testing.assert_array_equal(
            getattr(actual, name), getattr(reference, name)
        )
    assert (
        int(actual.mnmaxpot)
        == int(reference.mnmaxpot)
        == len(np.asarray(actual.xmpot))
    )
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
        expected = np.asarray(getattr(reference, name))
        scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
        error = float(np.max(np.abs(
            np.asarray(getattr(actual, name)) - expected
        ))) / scale
        assert error < 5.0e-3, (name, error)
