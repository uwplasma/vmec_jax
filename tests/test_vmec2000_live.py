"""Opt-in end-to-end comparison against a locally installed VMEC2000.

The pinned arrays here and in ``test_stability.py`` were measured against
STELLOPT ``v6.5.0-42-g9177f58`` (gfortran 13.4.0, ``-O2 -march=native``).
LASYM sine harmonics move by ~1e-2 between VMEC2000 builds years apart, well
above these tolerances, so record the build whenever a golden is re-pinned.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmex.core.input import VmecInput
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


def _updown_beta_input():
    """Finite-beta up-down-asymmetric tokamak; one definition, shared."""
    from tests.test_stability import _lasym_finite_beta_input

    return _lasym_finite_beta_input()


def _li383_lasym_input():
    """3-D lasym variant of li383: small stellarator-asymmetric boundary."""
    inp = VmecInput.from_file(DATA / "input.li383_low_res")
    rbs = np.zeros_like(np.asarray(inp.rbc, dtype=float))
    zbc = np.zeros_like(rbs)
    ntor = int(inp.ntor)
    for tab, n, m, val in (
        (rbs, 0, 1, 0.008), (zbc, 0, 1, 0.006),
        (rbs, 0, 2, 0.0015), (zbc, 0, 2, -0.0012),
        (rbs, 1, 1, 0.0010), (zbc, 1, 1, -0.0008),
    ):
        tab[n + ntor, m] = val
    return replace(
        inp,
        lasym=True,
        rbs=rbs,
        zbc=zbc,
        ns_array=np.array([16]),
        ftol_array=np.array([1e-11]),
        niter_array=np.array([10000]),
    )


@pytest.mark.parametrize(
    "name, build",
    [("updown_beta", _updown_beta_input), ("li383_lasym", _li383_lasym_input)],
)
def test_live_vmec2000_lasym_mercier_parity(pytestconfig, tmp_path, name, build):
    """LASYM Mercier profiles agree per-term with live VMEC2000.

    VMEC2000's lasym Mercier output is the anchor (mercier.f integrates
    full-theta-grid real-space fields with the uniform lasym wint, and its
    jxbforce.f inputs carry both parity channels), and the lasym lane is
    held to the symmetric parity suite's tolerance class — measured
    agreement is 1.6e-4 (updown) / 2.0e-3 (li383) scale-relative on DMerc.
    """
    vmec2000_dir = tmp_path / "vmec2000"
    vmex_dir = tmp_path / "vmex"
    vmec2000_dir.mkdir()
    vmex_dir.mkdir()
    inp = build()
    for directory in (vmec2000_dir, vmex_dir):
        inp.to_indata(directory / f"input.{name}")

    _run([str(_executable(pytestconfig)), f"input.{name}"], cwd=vmec2000_dir)
    _run(
        [
            sys.executable,
            "-m",
            "vmex.core.cli",
            str(vmex_dir / f"input.{name}"),
            "--outdir",
            str(vmex_dir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )

    reference = read_wout(vmec2000_dir / f"wout_{name}.nc")
    actual = read_wout(vmex_dir / f"wout_{name}.nc")
    assert int(actual.ier_flag) == int(reference.ier_flag) == 0
    assert bool(actual.lasym) and bool(reference.lasym)
    assert int(actual.ns) == int(reference.ns)
    # Scale-relative interior bounds, >= 10x the measured agreement; DShear
    # is profile-only (no parity content), the current-carrying DCurr/DGeod
    # are the terms the lasym filter feeds.
    limits = {
        "jdotb": 2.0e-3,
        "DMerc": 2.0e-2,
        "DShear": 1.0e-3,
        "DCurr": 2.0e-2,
        "DWell": 5.0e-3,
        "DGeod": 2.0e-2,
    }
    for field, limit in limits.items():
        expected = np.asarray(getattr(reference, field))[2:-1]
        got = np.asarray(getattr(actual, field))[2:-1]
        scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
        error = float(np.max(np.abs(got - expected))) / scale
        assert error < limit, (field, error)
    assert np.array_equal(
        np.sign(np.asarray(actual.DMerc)[2:-1]),
        np.sign(np.asarray(reference.DMerc)[2:-1]),
    )


def test_live_vmec2000_near_degenerate_vacuum(pytestconfig, tmp_path):
    """The LFORBAL vacuum remedy converges to the same equilibrium."""
    vmec2000_dir = tmp_path / "vmec2000_vacuum"
    vmex_dir = tmp_path / "vmex_vacuum"
    vmec2000_dir.mkdir()
    vmex_dir.mkdir()
    deck = DATA / "input.near_degenerate_vacuum_nfp3"
    for directory in (vmec2000_dir, vmex_dir):
        shutil.copy2(deck, directory / deck.name)

    _run([str(_executable(pytestconfig)), deck.name], cwd=vmec2000_dir)
    _run(
        [
            sys.executable,
            "-m",
            "vmex.core.cli",
            str(vmex_dir / deck.name),
            "--outdir",
            str(vmex_dir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )

    suffix = deck.name.removeprefix("input.")
    reference = read_wout(vmec2000_dir / f"wout_{suffix}.nc")
    actual = read_wout(vmex_dir / f"wout_{suffix}.nc")
    assert int(actual.ier_flag) == int(reference.ier_flag) == 0
    assert (int(actual.niter), int(reference.niter)) == (941, 942)
    for name in ("volume_p", "Rmajor_p", "Aminor_p", "aspect", "b0", "wb"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(reference, name), rtol=5e-10
        )
    for name in ("rmnc", "zmns", "bmnc", "iotaf"):
        expected = np.asarray(getattr(reference, name))
        error = np.linalg.norm(np.asarray(getattr(actual, name)) - expected)
        assert error / np.linalg.norm(expected) < 1e-8, (name, error)


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


def test_live_vmec2000_exact_jvp_gmres_robustness(pytestconfig, tmp_path):
    """Exact-JVP GMRES converges where VMEC2000's block GMRES stalls."""
    inp = replace(
        VmecInput.from_file(DATA / "input.circular_tokamak_aspect_100"),
        ns_array=np.asarray([51]),
        niter_array=np.asarray([20000]),
        ftol_array=np.asarray([1.0e-11]),
        precon_type="GMRES",
        prec2d_threshold=1.0e-6,
    )
    legacy_dir = tmp_path / "legacy_gmres"
    reference_dir = tmp_path / "legacy_1d"
    vmex_dir = tmp_path / "vmex_gmres"
    for directory in (legacy_dir, reference_dir, vmex_dir):
        directory.mkdir()
    inp.to_indata(legacy_dir / "input.aspect100")
    inp.to_indata(vmex_dir / "input.aspect100")
    replace(inp, precon_type="NONE").to_indata(
        reference_dir / "input.aspect100"
    )

    legacy = subprocess.run(
        [str(_executable(pytestconfig)), "input.aspect100"],
        cwd=legacy_dir,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert legacy.returncode == 0
    assert "Beginning GMRES iterations" in legacy.stdout
    assert "Try increasing NITER or PRE_NITER" in legacy.stdout
    rows = []
    for line in legacy.stdout.splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0].isdigit():
            try:
                rows.append((int(fields[0]), *(float(x) for x in fields[1:4])))
            except ValueError:
                pass
    assert rows[-1][0] == 110
    assert 1.0e-10 < max(rows[-1][1:]) < 1.0e-8

    _run(
        [str(_executable(pytestconfig)), "input.aspect100"],
        cwd=reference_dir,
    )
    _run(
        [
            sys.executable,
            "-m",
            "vmex.core.cli",
            str(vmex_dir / "input.aspect100"),
            "--outdir",
            str(vmex_dir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    reference = read_wout(reference_dir / "wout_aspect100.nc")
    actual = read_wout(vmex_dir / "wout_aspect100.nc")
    assert int(actual.ier_flag) == int(reference.ier_flag) == 0
    assert max(float(actual.fsqr), float(actual.fsqz), float(actual.fsql)) < 1e-11
    assert int(actual.niter) < int(reference.niter)
    assert abs(float(actual.wb) - float(reference.wb)) / abs(reference.wb) < 1e-8
    for name in ("rmnc", "zmns"):
        got = np.asarray(getattr(actual, name))
        expected = np.asarray(getattr(reference, name))
        assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-5
    np.testing.assert_allclose(actual.iotaf, reference.iotaf, atol=2e-14)


