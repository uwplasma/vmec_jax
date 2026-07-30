"""The self-consistent public QI free-boundary gate (sheet-current field).

Public replacement for the confidential high-mode QI failure class: a QI
boundary, a confining field, the private-style radial ladder, every
reported input feature, and a recorded VMEC2000 comparison — built
deterministically in-session by :mod:`tools.build_qi_sheet_mgrid`.

VMEC2000 goldens on the byte-equivalent gate deck (2026-07-28,
DELT = 0.50; five rungs 21→34→55→89→144, 238 modes, LFORBAL=T,
PRECON_TYPE='NONE', PREC2D_THRESHOLD=1e-30, APHI, no supplied axis):
vacuum on at 45, ``wb = 2.2735640332e-3``, ``sum raxis_cc = 0.93039119``,
``iotaf(edge) = -0.4147612``, ``aspect = 8.0935``.  VMEX measured:
arm64-macos ``wb = 2.2735814e-3`` (7.7e-6 rel), x86-linux
``wb = 2.2735892e-3`` (1.1e-5 rel), vacuum on at 45 on both.

DELT rationale: a sweep mapped a platform-sensitive stability edge (0.55 /
0.60 once non-finite on x86-linux), root-caused to NESTOR consuming a
sign-changed transient and fixed at base commit "Free boundary: never feed
NESTOR a sign-changed state"; post-fix all four swept DELT values converge
on both platforms.  0.50 stays the stable default gate;
:func:`test_qi_sheet_gate_ladder_delt_055_regression` pins the 0.55 case
against its own VMEC2000 goldens (2026-07-28): ``wb = 2.2738091757e-3``,
``sum raxis_cc = 0.93032287``, ``iotaf(edge) = -0.4160441``,
``aspect = 8.0965``, activation 45.

Calibration disclosure: the sheet amplitude and PHIEDGE derive from VMEX
fixed-boundary outputs, so the free comparison is self-consistent; the
independent leg is VMEC2000 solving the SAME deck + mgrid byte-for-byte.

ftol rationale: the sheet fit nulls ``B.n`` to 2.5e-4 of ``|B|`` and the
64x64x36 trilinear mgrid floors the free residual near 1e-6 (measured
6.7e-7 … 2.7e-6 per rung), so the free gate demands ``ftol = 1e-5``; the
FIXED ladder has no floor and must converge at 1e-8.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

# jit required: unjitted, the fixture alone takes ~24 min and its
# Biot-Savart loops exhaust memory.
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.multigrid import (  # noqa: E402
    solve_free_boundary_multigrid,
    solve_multigrid,
)

@pytest.fixture(autouse=True)
def _release_jax_caches():
    """Free compiled executables between the heavy 238-mode lanes; keeps
    peak memory at the largest single lane (hosted runners were evicted)."""
    yield
    jax.clear_caches()
    import gc

    gc.collect()


GOLDEN_WB = 2.2735640332039e-3
GOLDEN_R00 = 0.9303911858840
GOLDEN_IOTA_EDGE = -0.4147611657
GOLDEN_ASPECT = 8.0935
GOLDEN_ACTIVATION = 45

# VMEC2000 on the SAME gate deck rewritten to DELT = 0.55 (recorded
# 2026-07-28) — the historically non-finite case the vacuum-source fix
# restored; see test_qi_sheet_gate_ladder_delt_055_regression.
GOLDEN_WB_055 = 2.2738091756761185e-3
GOLDEN_R00_055 = 0.9303228697937338
GOLDEN_IOTA_EDGE_055 = -0.416044061525186

# VMEC2000 on the FIXED-boundary 238-mode ladder (2026-07-28, ftol 1e-8,
# fsqr 9.32e-9); VMEX parity measured at the 1e-10 class on this deck.
FIXED_GOLDEN_WB = 2.2762082139521e-3
FIXED_GOLDEN_R00 = 0.9293551107718
FIXED_GOLDEN_IOTA_EDGE = -0.4410936869
FIXED_GOLDEN_ASPECT = 8.0019

# initialize_radial.f FORMAT 1000 stage banner and the printout.f screen
# line (the terminating iteration of a stage is always printed).
_NS_BANNER = re.compile(
    r"NS = *(\d+) NO\. FOURIER MODES = *(\d+) FTOLV = *([0-9.E+-]+)"
    r" NITER = *(\d+)")
_ITER_LINE = re.compile(
    r"^ *(\d+) +(\d[\d.]*E[+-]\d+) +(\d[\d.]*E[+-]\d+) +(\d[\d.]*E[+-]\d+)",
    re.M)


def _assert_five_rungs_crossed_ftol(output: str) -> None:
    """Every NS_ARRAY rung terminates by crossing its own ftol: exactly one
    ``NS = `` banner per rung (a Jacobian-recovery retry would re-banner and
    fail the count), and each rung's final FSQR/FSQZ/FSQL at/below its FTOLV
    in strictly fewer than NITER iterations — no iteration exhaustion."""
    banners = list(_NS_BANNER.finditer(output))
    assert len(banners) == 5, f"expected 5 rung banners, found {len(banners)}"
    assert [int(b.group(1)) for b in banners] == [21, 34, 55, 89, 144]
    assert all(int(b.group(2)) == 238 for b in banners)
    for i, banner in enumerate(banners):
        ns = int(banner.group(1))
        ftol = float(banner.group(3))
        niter = int(banner.group(4))
        end = banners[i + 1].start() if i + 1 < len(banners) else len(output)
        rows = _ITER_LINE.findall(output[banner.start():end])
        assert rows, f"rung NS={ns} printed no iteration rows"
        last_it, fsqr, fsqz, fsql = rows[-1]
        assert int(last_it) < niter, (
            f"rung NS={ns} exhausted NITER={niter} without crossing ftol")
        worst = max(float(fsqr), float(fsqz), float(fsql))
        # screen lines round to 3 significant digits; 1.5% absorbs the
        # worst-case print rounding of a residual just under ftol.
        assert worst <= 1.015 * ftol, (
            f"rung NS={ns} final residuals {rows[-1]} above ftol={ftol:g}")


def _wout_parity_aspect(inp: VmecInput, result) -> float:
    """wout ``aspect`` of the final state (aspectratio.f quadrature, the
    exact wout-writer convention), evaluated on a fresh runtime — no
    re-solve."""
    from vmex.core.solver import prepare_runtime, resolution_from_input
    from vmex.core.statephysics import aspect_ratio

    ns = int(result.state.R_cos.shape[0])
    rt = prepare_runtime(inp, resolution_from_input(inp, ns=ns))
    return float(aspect_ratio(result.state, rt))


@pytest.fixture(scope="module")
def sheet_field(tmp_path_factory):
    """Build the public QI sheet field once (~3 min, deterministic)."""
    import build_qi_sheet_mgrid as builder

    outdir = tmp_path_factory.mktemp("qi_sheet")
    meta = builder.build(outdir)
    return outdir, meta


def _indexed_m0(deck: str) -> str:
    """Rewrite the m=0 boundary rows as one Fortran ``lo:hi`` section each."""
    rvals, zvals, n_list = {}, {}, []
    for m_ in re.finditer(r"RBC\( *(-?\d+), *0\) *= *([0-9.eE+-]+)", deck):
        n_list.append(int(m_.group(1)))
        rvals[int(m_.group(1))] = m_.group(2)
    for m_ in re.finditer(r"ZBS\( *(-?\d+), *0\) *= *([0-9.eE+-]+)", deck):
        zvals[int(m_.group(1))] = m_.group(2)
    if not n_list:
        return deck
    lo, hi = min(n_list), max(n_list)
    r_sec = " ".join(rvals.get(n, "0.0") for n in range(lo, hi + 1))
    z_sec = " ".join(zvals.get(n, "0.0") for n in range(lo, hi + 1))
    deck = re.sub(r" *RBC\( *-?\d+, *0\) *= *[0-9.eE+-]+,?\n", "", deck)
    deck = re.sub(r" *ZBS\( *-?\d+, *0\) *= *[0-9.eE+-]+,?\n", "", deck)
    return deck.replace(
        "/", f"  RBC({lo}:{hi},0) = {r_sec}\n  ZBS({lo}:{hi},0) = {z_sec}\n/", 1)


def _gate_deck(base_deck: str) -> str:
    deck = re.sub(r"MPOL *= *\d+", "MPOL = 13", base_deck)
    deck = re.sub(r"NTOR *= *\d+", "NTOR = 9", deck)
    deck = re.sub(r"NS_ARRAY *= *[\d, ]+", "NS_ARRAY = 21, 34, 55, 89, 144", deck)
    deck = re.sub(r"FTOL_ARRAY *= *[0-9.eE+\-, ]+",
                  "FTOL_ARRAY = 1.0E-5, 1.0E-5, 1.0E-5, 1.0E-5, 1.0E-5", deck)
    deck = deck.replace(
        "&INDATA", "&INDATA\n  NITER_ARRAY = 1500, 1500, 1500, 1500, 1500", 1)
    deck = deck.replace(
        "  LFREEB = T",
        "  LFREEB = T\n  LFORBAL = T\n  PRECON_TYPE = 'NONE'\n"
        "  PREC2D_THRESHOLD = 1.0E-30\n  APHI = 1.0, 0.0, 0.0")
    return _indexed_m0(deck)


@pytest.mark.full  # sheet-field build (~90 s jitted, GB-scale temporaries)
# memory-evicted the shared parity shard; the gate runs in the full matrix.
def test_sheet_field_confines(sheet_field):
    """The deterministic fit reaches the confining Bn + alignment thresholds.

    ``alignment`` is the mean boundary cosine between the sheet field and
    the equilibrium boundary field (measured 0.9999883 on this build; the
    gate demands > 0.999985, ~20% margin on the 1.17e-5 defect from 1).
    """
    _, meta = sheet_field
    assert meta["fit_metric"] < 1.0e-3, meta
    assert meta["alignment"] > 0.999985, meta
    assert meta["phiedge"] == pytest.approx(0.0307113284, rel=1e-3)


@pytest.mark.full  # ~4 min: single-stage free solve on the built field
def test_qi_sheet_free_boundary_converges(sheet_field, tmp_path):
    outdir, _ = sheet_field
    deck = (outdir / "input.qi_sheet_free").read_text()
    path = tmp_path / "input.qi_free"
    path.write_text(deck)
    inp = VmecInput.from_file(str(path))
    from vmex.core.freeboundary import solve_free_boundary

    lines: list[str] = []
    result = solve_free_boundary(
        inp, mgrid_path=str(outdir / "mgrid_qi_sheet.nc"),
        ftol=1.0e-5, max_iterations=600, verbose=True,
        emit=lambda t="", end="\n": lines.append(str(t)),
        error_on_no_convergence=False)
    assert any("VACUUM PRESSURE TURNED ON" in ln for ln in lines)
    assert bool(result.converged), f"fsqr={float(result.fsqr):.2e}"
    assert float(result.r00) == pytest.approx(GOLDEN_R00, rel=5e-3)


@pytest.mark.full  # ~20 min: the FULL private-style gate ladder vs VMEC2000
def test_qi_sheet_gate_ladder_matches_vmec2000(sheet_field, tmp_path):
    """All reported ingredients on a convergent QI free-boundary ladder.

    238 modes, 21→34→55→89→144, LFORBAL + PRECON NONE + APHI + indexed
    sections + no supplied axis, vacuum activation on rung 1 carried
    through four radial transitions, convergence at every rung, and the
    recorded VMEC2000 equilibrium (module docstring).
    """
    outdir, _ = sheet_field
    deck = _gate_deck((outdir / "input.qi_sheet_free").read_text())
    path = tmp_path / "input.qi_gate"
    path.write_text(deck)
    inp = VmecInput.from_file(str(path))
    assert not np.any(np.asarray(inp.raxis_c))  # no supplied axis
    assert bool(inp.lforbal)

    lines: list[str] = []

    def collect(t="", end="\n"):
        lines.append(str(t))

    # release_stage_cache: retaining every rung's executables peaks at
    # 12.4 GB RSS (measured); per-rung release changes no numerics.
    result = solve_free_boundary_multigrid(
        inp, mgrid_path=str(outdir / "mgrid_qi_sheet.nc"), verbose=True,
        emit=collect, raise_on_max_iterations=False,
        release_stage_cache=True)
    output = "\n".join(lines)

    m = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)", output)
    assert m is not None, "vacuum never activated"
    assert abs(int(m.group(1)) - GOLDEN_ACTIVATION) <= 4, m.group(0)
    assert output.rfind("NS = ") > output.find("VACUUM PRESSURE"), (
        "activation did not precede the radial transitions")
    _assert_five_rungs_crossed_ftol(output)
    assert bool(result.converged), f"fsqr={float(result.fsqr):.2e}"
    assert float(result.wb) == pytest.approx(GOLDEN_WB, rel=5e-4)
    assert float(result.r00) == pytest.approx(GOLDEN_R00, rel=3e-3)
    iota_edge = float(np.asarray(result.iotaf)[-1])
    assert iota_edge == pytest.approx(GOLDEN_IOTA_EDGE, rel=2e-3)
    assert _wout_parity_aspect(inp, result) == pytest.approx(
        GOLDEN_ASPECT, rel=3e-3)


@pytest.mark.full  # ~20 min: the SAME gate ladder through the FFT kernel
def test_qi_sheet_gate_ladder_fft_matches_dense(sheet_field, tmp_path):
    """The gate ladder forced through the separable-FFT synthesis
    (``use_fft=True``) must land inside the same recorded VMEC2000 bands as
    the dense lane (the FFT kernel is the same math to roundoff at this
    238-mode/NZETA=36 table).  Regression context: the FFT roundoff
    realization once sat on the wrong side of the DELT stability edge and a
    sign-changed transient poisoned NESTOR (``bsqvac`` NaN); fixed by the
    funct3d.f Jacobian-first ordering (``freeboundary._jacobian_ok``)."""
    outdir, _ = sheet_field
    deck = _gate_deck((outdir / "input.qi_sheet_free").read_text())
    path = tmp_path / "input.qi_gate_fft"
    path.write_text(deck)
    inp = VmecInput.from_file(str(path))

    lines: list[str] = []
    result = solve_free_boundary_multigrid(
        inp, mgrid_path=str(outdir / "mgrid_qi_sheet.nc"), verbose=True,
        emit=lambda t="", end="\n": lines.append(str(t)),
        raise_on_max_iterations=False, release_stage_cache=True,
        use_fft=True)
    output = "\n".join(lines)

    m = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)", output)
    assert m is not None, "vacuum never activated under the FFT kernel"
    assert abs(int(m.group(1)) - GOLDEN_ACTIVATION) <= 4, m.group(0)
    _assert_five_rungs_crossed_ftol(output)
    assert bool(result.converged), f"fsqr={float(result.fsqr):.2e}"
    assert float(result.wb) == pytest.approx(GOLDEN_WB, rel=5e-4)
    assert float(result.r00) == pytest.approx(GOLDEN_R00, rel=3e-3)
    iota_edge = float(np.asarray(result.iotaf)[-1])
    assert iota_edge == pytest.approx(GOLDEN_IOTA_EDGE, rel=2e-3)


@pytest.mark.full  # ~12 min: the restored-DELT free ladder vs VMEC2000
def test_qi_sheet_gate_ladder_delt_055_regression(sheet_field, tmp_path):
    """DELT = 0.55 free gate ladder: the vacuum-source fix stays fixed.

    This exact ladder once raised NON-FINITE FORCE EVALUATION (sign-changed
    transient reaching NESTOR, poisoned ``bsqvac``); the fix restores the
    funct3d.f Jacobian-first ordering (``freeboundary._jacobian_ok``), and
    this pin keeps the repaired behavior from rotting.  Goldens (VMEC2000,
    2026-07-28, DELT = 0.55): ``wb = 2.2738091757e-3``,
    ``sum raxis_cc = 0.93032287``, ``iotaf(edge) = -0.4160441``,
    ``aspect = 8.0965``, vacuum on at 45.  VMEX measured inside all bands
    on both platforms (worst rel: wb 1.8e-4, r00 6.3e-4, iota 8.1e-4; x86
    FFT agrees with x86 dense to 3e-9), activation 45 everywhere.
    """
    outdir, _ = sheet_field
    deck = _gate_deck((outdir / "input.qi_sheet_free").read_text())
    deck = re.sub(r"DELT *= *[0-9.eE+-]+", "DELT = 0.55", deck)
    path = tmp_path / "input.qi_gate_055"
    path.write_text(deck)
    inp = VmecInput.from_file(str(path))
    assert float(inp.delt) == pytest.approx(0.55)

    lines: list[str] = []
    result = solve_free_boundary_multigrid(
        inp, mgrid_path=str(outdir / "mgrid_qi_sheet.nc"), verbose=True,
        emit=lambda t="", end="\n": lines.append(str(t)),
        raise_on_max_iterations=False, release_stage_cache=True)
    output = "\n".join(lines)

    m = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)", output)
    assert m is not None, "vacuum never activated at DELT = 0.55"
    assert abs(int(m.group(1)) - GOLDEN_ACTIVATION) <= 4, m.group(0)
    assert bool(result.converged), f"fsqr={float(result.fsqr):.2e}"
    assert float(result.wb) == pytest.approx(GOLDEN_WB_055, rel=5e-4)
    assert float(result.r00) == pytest.approx(GOLDEN_R00_055, rel=3e-3)
    iota_edge = float(np.asarray(result.iotaf)[-1])
    assert iota_edge == pytest.approx(GOLDEN_IOTA_EDGE_055, rel=2e-3)


@pytest.mark.full  # ~25 min: fixed-boundary ladder at full 1e-8 (no floor)
def test_qi_fixed_238_ladder_converges(tmp_path):
    """FIXED 238-mode ladder converges at 1e-8 AND matches VMEC2000 wout.

    VMEC2000 goldens (2026-07-28, MPOL=13/NTOR=9, NS 21→34→55→89→144, ftol
    1e-8, fsqr 9.32e-9): ``wb = 2.2762082139521e-3``,
    ``r00 = 0.9293551107718``, ``iotaf(edge) = -0.4410936869``,
    ``aspect = 8.0019``.  Local VMEX parity is at the 1e-10 class; the 1e-6
    bands leave platform margin while staying ~3 orders tighter than the
    free gate (no mgrid-interpolation floor in the fixed problem).
    """
    deck = (ROOT / "examples" / "data" / "input.nfp2_QI").read_text()
    deck = re.sub(r"MPOL *= *\d+", "MPOL = 13", deck)
    deck = re.sub(r"NTOR *= *\d+", "NTOR = 9", deck)
    deck = re.sub(r"NS_ARRAY *=[^\n]*", "NS_ARRAY = 21, 34, 55, 89, 144", deck)
    deck = re.sub(r"FTOL_ARRAY *=[^\n]*",
                  "FTOL_ARRAY = 1.0E-8, 1.0E-8, 1.0E-8, 1.0E-8, 1.0E-8", deck)
    deck = deck.replace(
        "&INDATA",
        "&INDATA\n  NITER_ARRAY = 2000, 2000, 2000, 2000, 2000\n"
        "  LFORBAL = T\n  PRECON_TYPE = 'NONE'\n"
        "  PREC2D_THRESHOLD = 1.0E-30\n  APHI = 1.0, 0.0, 0.0", 1)
    path = tmp_path / "input.qi_fixed_gate"
    path.write_text(_indexed_m0(deck))
    inp = VmecInput.from_file(str(path))
    result = solve_multigrid(inp, verbose=False, raise_on_max_iterations=False,
                             release_stage_cache=True)
    assert bool(result.converged), f"fsqr={float(result.fsqr):.2e}"
    assert float(result.wb) == pytest.approx(FIXED_GOLDEN_WB, rel=1e-6)
    assert float(result.r00) == pytest.approx(FIXED_GOLDEN_R00, rel=1e-6)
    iota_edge = float(np.asarray(result.iotaf)[-1])
    assert iota_edge == pytest.approx(FIXED_GOLDEN_IOTA_EDGE, rel=1e-6)
    assert _wout_parity_aspect(inp, result) == pytest.approx(
        FIXED_GOLDEN_ASPECT, rel=1e-3)
