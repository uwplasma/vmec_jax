"""APHI toroidal-flux normalization semantics (VMEC2000 ``profil1d.f`` parity).

VMEC2000 ground truth being pinned here:

- ``Sources/Initialization_Cleanup/magnetic_fluxes.f`` (``torflux_deriv``,
  lines 22-25): ``torflux_deriv = x*torflux_deriv + i*tf(i)`` accumulated from
  the highest index down, i.e. ``Phi'(x) = sum_i i*aphi(i)*x**(i-1)`` — the
  APHI coefficients parameterize the flux ``Phi(x) = sum_i aphi(i)*x**i``
  itself, not its derivative; and (``torflux``, lines 70-77) ``torflux(x)`` is
  the fixed 101-point trapezoid integral of ``torflux_deriv`` on ``[0, x]``.
- ``Sources/Initialization_Cleanup/profil1d.f`` (lines 289-293):
  ``torflux_edge = signgs*phiedge/twopi`` then ``torflux_edge =
  torflux_edge/torflux(one)`` when ``torflux(one) /= 0``.  The radial flux
  profile is therefore NORMALIZED: PHIEDGE is always the physical edge flux
  no matter the overall scale of the APHI coefficients, and ``phips(i) =
  torflux_edge*torflux_deriv(si)`` (line 311) never inherits that scale.
- ``profil1d.f`` (lines 307, 313-314): the *profile argument* ``tf =
  MIN(one, torflux(si))`` is the UNnormalized flux, clamped at 1 — iota,
  current, and mass are evaluated at the clamped raw polynomial, so an
  unnormalized APHI legitimately reshapes those profiles while leaving the
  flux arrays untouched.

An unnormalized coefficient set (``torflux(1) != 1``) must therefore leave
``phips/chips/phipf/chipf/lamscale`` bit-identical when it only rescales the
identity map, and must never rescale the equilibrium energy: ``wb`` scales as
flux squared, so any missed normalization shows up as a ``torflux(1)**2``
blow-up of the row-1 WMHD.  These tests pin that contract at three levels:
the polynomial/trapezoid functions, the ``flux_profiles`` arrays, and full
solver row-1 goldens for a public deck (plus an opt-in live comparison
against a local ``xvmec2000``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from vmex.core.input import VmecInput
from vmex.core.setup import _torflux_functions, flux_profiles, radial_grids

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
FIXED_DECK = DATA_DIR / "input.cth_like_fixed_bdy"

#: Scale-only set: ``Phi(x) = 8.578*x`` (``torflux(1) = 8.578``).
APHI_SCALE = (8.578,)
#: Multi-term unnormalized set: ``Phi(x) = 2x - 1.5x^2 + x^3``
#: (``torflux(1) = 1.5`` up to trapezoid quadrature).
APHI_MULTI = (2.0, -1.5, 1.0)

_ROW_FIELD = re.compile(r"^\d\.\d{2}E[+-]\d{2}$")


# ---------------------------------------------------------------------------
# Reference transliterations of magnetic_fluxes.f
# ---------------------------------------------------------------------------


def _fortran_torflux_deriv(aphi, x: float) -> float:
    """magnetic_fluxes.f lines 22-25 (Horner loop, highest index first)."""
    y = 0.0
    for i in range(len(aphi), 0, -1):
        y = x * y + i * aphi[i - 1]
    return y


def _fortran_torflux(aphi, x: float) -> float:
    """magnetic_fluxes.f lines 70-77 (101-point trapezoid on [0, x])."""
    h = 1.0e-2 * x
    total = sum(_fortran_torflux_deriv(aphi, (i - 1) * h) for i in range(1, 102))
    total -= 0.5 * (_fortran_torflux_deriv(aphi, 0.0) + _fortran_torflux_deriv(aphi, x))
    return h * total


def _with_aphi(inp: VmecInput, aphi) -> VmecInput:
    padded = np.zeros(20)
    padded[: len(aphi)] = aphi
    return dataclasses.replace(inp, aphi=padded)


# ---------------------------------------------------------------------------
# (a) unit level: flux polynomial and trapezoid vs the Fortran formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aphi",
    [(1.0,), APHI_SCALE, APHI_MULTI, (0.5, 0.25, 0.0, 1.25)],
    ids=["identity", "scale", "multi", "quartic"],
)
def test_torflux_functions_match_fortran_formula(aphi) -> None:
    """``_torflux_functions`` reproduces magnetic_fluxes.f for any scale."""
    torflux, torflux_deriv = _torflux_functions(np.asarray(aphi))
    for x in (0.0, 0.13, 0.5, 1.0):
        np.testing.assert_allclose(
            float(torflux_deriv(x)),
            _fortran_torflux_deriv(aphi, x),
            rtol=1e-14,
            err_msg=f"torflux_deriv({x})",
        )
        np.testing.assert_allclose(
            float(torflux(x)),
            _fortran_torflux(aphi, x),
            rtol=1e-13,
            atol=1e-300,
            err_msg=f"torflux({x})",
        )


def test_flux_profiles_normalize_torflux_edge_multi_term() -> None:
    """profil1d.f lines 289-293/311: phips = signgs*phiedge/(2pi*Phi(1))*Phi'."""
    inp = _with_aphi(VmecInput.from_file(FIXED_DECK), APHI_MULTI)
    grids = radial_grids(15)
    prof = flux_profiles(inp, grids, r00=np.asarray(0.78), signgs=-1)
    edge = -1 * inp.phiedge / (2.0 * np.pi) / _fortran_torflux(APHI_MULTI, 1.0)
    s_half = np.asarray(grids.s_half)
    s_full = np.asarray(grids.s_full)
    expected_half = edge * np.asarray(
        [_fortran_torflux_deriv(APHI_MULTI, s) for s in s_half]
    )
    expected_half[0] = 0.0
    expected_full = edge * np.asarray(
        [_fortran_torflux_deriv(APHI_MULTI, s) for s in s_full]
    )
    np.testing.assert_allclose(np.asarray(prof["phips"]), expected_half, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(prof["phipf"]), expected_full, rtol=1e-13)


def test_profile_argument_is_unnormalized_clamped_torflux() -> None:
    """profil1d.f lines 307/313: tf = MIN(1, torflux(s)) stays unnormalized."""
    text = """&INDATA
    NFP = 3, MPOL = 3, NTOR = 0, NS_ARRAY = 11, PHIEDGE = 1.7,
    NCURR = 0,
    AI = 1.0, 1.0,
    APHI = 8.578,
    RBC(0,0) = 3.0, RBC(0,1) = 1.0, ZBS(0,1) = 1.0,
    /
    """
    inp = VmecInput.from_indata_text(text)
    grids = radial_grids(11)
    prof = flux_profiles(inp, grids, r00=np.asarray(3.0), signgs=-1)
    torflux, _ = _torflux_functions(inp.aphi)
    expected = 1.0 + np.minimum(np.asarray(torflux(np.asarray(grids.s_half))), 1.0)
    expected[0] = 0.0  # half-mesh axis slot is zeroed (profil1d.f DO i = 2, ns)
    np.testing.assert_array_equal(np.asarray(prof["iotas"]), expected)


# ---------------------------------------------------------------------------
# (c) bit-identity: normalized decks and pure-scale coefficient sets
# ---------------------------------------------------------------------------


def _profiles_for(aphi=None) -> dict[str, np.ndarray]:
    inp = VmecInput.from_file(FIXED_DECK)
    if aphi is not None:
        inp = _with_aphi(inp, aphi)
    grids = radial_grids(15)
    out = flux_profiles(inp, grids, r00=np.asarray(0.780906309727434), signgs=-1)
    return {k: np.asarray(v) for k, v in out.items()}


def test_default_and_explicit_identity_aphi_are_bit_identical() -> None:
    """APHI absent and APHI = 1.0 (torflux(1) = 1) share every last bit."""
    base = _profiles_for()
    explicit = _profiles_for((1.0,))
    for key, value in base.items():
        np.testing.assert_array_equal(value, explicit[key], err_msg=key)


def test_pure_scale_aphi_leaves_flux_arrays_bit_identical() -> None:
    """torflux(1) = 8.578 with identity shape must not touch the flux arrays.

    profil1d.f divides torflux_edge by torflux(1), so ``Phi(x) = 8.578*x``
    yields exactly the same phips/chips/phipf/chipf/lamscale as the default
    identity map — while mass/icurv legitimately move because their argument
    ``tf = MIN(1, torflux(s))`` saturates (the unnormalized clamp above).
    A missed normalization would scale phips by 8.578 and the row-1 energy
    by 8.578**2 = 73.58.
    """
    base = _profiles_for()
    scaled = _profiles_for(APHI_SCALE)
    for key in ("phips", "chips", "phipf", "chipf", "lamscale"):
        np.testing.assert_array_equal(base[key], scaled[key], err_msg=key)
    # The clamp reshapes the pressure/current inputs; equality here would mean
    # the clamp was silently normalized away.
    assert not np.array_equal(base["mass"], scaled["mass"])
    assert not np.array_equal(base["icurv"], scaled["icurv"])


# ---------------------------------------------------------------------------
# (b) row-1 goldens for the public deck with unnormalized APHI
# ---------------------------------------------------------------------------


def _deck_with_aphi(tmp_path: Path, aphi_line: str | None, niter: int) -> Path:
    text = FIXED_DECK.read_text()
    text = text.replace("NITER_array = 25000,", f"NITER_array = {niter},")
    if aphi_line is not None:
        text = text.replace("PHIEDGE = -0.035,", f"PHIEDGE = -0.035,\n{aphi_line}")
    deck = tmp_path / "input.aphinorm"
    deck.write_text(text)
    return deck


def _first_iteration_row(stdout: str) -> list[float]:
    for line in stdout.splitlines():
        tokens = line.split()
        if (
            len(tokens) >= 6
            and tokens[0] == "1"
            and _ROW_FIELD.match(tokens[1])
        ):
            return [float(tok) for tok in tokens[1:]]
    raise AssertionError(f"no iteration-1 row found in output:\n{stdout}")


def _run_vmex_cli(deck: Path, outdir: Path) -> str:
    from vmex.core import cli

    jax.config.update("jax_disable_jit", False)
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli.main([str(deck), "--outdir", str(outdir)])
    finally:
        jax.config.update("jax_disable_jit", True)
    return buffer.getvalue()


def test_row1_goldens_with_unnormalized_aphi(tmp_path: Path) -> None:
    """Row-1 FSQR/WMHD stay at the VMEC2000-parity goldens, not 73.6x off.

    Golden row 1 for the public cth_like_fixed_bdy deck (both codes print
    identical rows; xvmec2000 capture 2026-08):

    - APHI absent:      FSQR 3.76E-02, FSQZ 9.68E-04, WMHD 4.5096E-02
    - APHI = 8.578:     FSQR 4.02E-02, FSQZ 6.18E-03, WMHD 4.6403E-02
    """
    base_out = tmp_path / "base"
    unnorm_out = tmp_path / "unnorm"
    base_out.mkdir()
    unnorm_out.mkdir()
    base_row = _first_iteration_row(
        _run_vmex_cli(_deck_with_aphi(tmp_path / "base", None, 25), base_out)
    )
    unnorm_row = _first_iteration_row(
        _run_vmex_cli(
            _deck_with_aphi(tmp_path / "unnorm", "APHI = 8.578,", 25), unnorm_out
        )
    )
    fsqr, fsqz, _, _, _, wmhd = unnorm_row[:6]
    np.testing.assert_allclose(fsqr, 4.02e-2, rtol=5e-3)
    np.testing.assert_allclose(fsqz, 6.18e-3, rtol=5e-3)
    np.testing.assert_allclose(wmhd, 4.6403e-2, rtol=2e-4)
    # The direct regression guard: an unnormalized identity-shaped APHI must
    # not rescale the starting energy (a missed torflux(1) division would
    # multiply WMHD by 8.578**2 = 73.58 and blow FSQR up by orders).
    wmhd_base = base_row[5]
    np.testing.assert_allclose(wmhd_base, 4.5096e-2, rtol=2e-4)
    assert 0.9 < wmhd / wmhd_base < 1.1
    assert unnorm_row[0] / base_row[0] < 10.0


# ---------------------------------------------------------------------------
# live xvmec2000 comparison (opt-in, --run-vmec2000)
# ---------------------------------------------------------------------------


@pytest.mark.vmec2000_live
@pytest.mark.parametrize(
    "aphi_line",
    ["APHI = 8.578,", "APHI = 2.0, -1.5, 1.0,"],
    ids=["scale", "multi"],
)
def test_row1_parity_live_vmec2000(pytestconfig, tmp_path: Path, aphi_line) -> None:
    """Both codes print the identical row 1 for unnormalized-APHI decks."""
    configured = str(pytestconfig.getoption("--vmec2000-executable")).strip()
    candidates = [Path(configured)] if configured else []
    discovered = shutil.which("xvmec2000")
    if discovered:
        candidates.append(Path(discovered))
    executable = next((c for c in candidates if c.is_file()), None)
    if executable is None:
        pytest.fail(
            "--run-vmec2000 requested but xvmec2000 was not found; pass "
            "--vmec2000-executable PATH"
        )

    ref_dir = tmp_path / "vmec2000"
    ref_dir.mkdir()
    deck = _deck_with_aphi(ref_dir, aphi_line, 60)
    completed = subprocess.run(
        [str(executable), deck.name],
        cwd=ref_dir,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    reference_row = _first_iteration_row(completed.stdout)

    vmex_dir = tmp_path / "vmex"
    vmex_dir.mkdir()
    actual_row = _first_iteration_row(
        _run_vmex_cli(_deck_with_aphi(vmex_dir, aphi_line, 60), vmex_dir)
    )
    # Print-precision fields (3 significant digits): identical rows parse to
    # identical floats.
    np.testing.assert_array_equal(
        np.asarray(actual_row[:6]), np.asarray(reference_row[:6])
    )
