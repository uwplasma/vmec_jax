"""The self-consistent public QI free-boundary gate (sheet-current field).

This is the public replacement for the confidential high-mode QI failure
class the reviews required: a QI boundary, a field that actually confines
it, the private-style radial ladder, every reported input feature, and a
recorded VMEC2000 comparison — all from public data, built deterministically
in-session by :mod:`tools.build_qi_sheet_mgrid` (no fetched asset).

Recorded fresh local xvmec2000/PARVMEC goldens on the byte-equivalent gate
deck (2026-07-28): all five rungs (21→34→55→89→144, 238 modes, LFORBAL=T,
PRECON_TYPE='NONE', PREC2D_THRESHOLD=1e-30, APHI, no supplied axis) —
vacuum on at iteration 45, EXECUTION TERMINATED NORMALLY, wout written:
``wb = 2.2738091757e-3``, ``sum raxis_cc = 0.93032287``,
``iotaf(edge) = -0.4160441``, ``aspect = 8.0965``.  VMEX on the identical
ladder: vacuum on at 45, ``wb = 2.2740006e-3`` (8.4e-5 relative agreement).

ftol rationale: the sheet fit nulls ``B.n`` to 2.5e-4 of ``|B|`` and the
64x64x36 trilinear mgrid adds interpolation error, which floors the
free-boundary residual near 1e-6 (measured per rung: 6.7e-7 … 2.7e-6).
The gate therefore demands convergence at ``ftol = 1e-5`` — decisively
crossed on every rung by both codes — rather than pretending to 1e-8 the
field cannot support.  The FIXED-boundary ladder has no such floor and
must converge at 1e-8.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

# The suite-wide conftest disables jit by default; the sheet-current build
# and the 238-mode ladders are jit-dependent (unjitted, the fixture alone
# takes ~24 minutes and its Biot-Savart loops exhaust memory).
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
    """Free compiled executables between the heavy gate lanes.

    Four 238-mode lanes in one process accumulate JAX executable caches
    (the same syndrome that evicted hosted CI runners); releasing after
    each test keeps peak memory at the largest single lane.
    """
    yield
    jax.clear_caches()
    import gc

    gc.collect()


GOLDEN_WB = 2.2738091756761185e-3
GOLDEN_R00 = 0.9303228697937338
GOLDEN_IOTA_EDGE = -0.416044061525186
GOLDEN_ACTIVATION = 45


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


def test_sheet_field_confines(sheet_field):
    """The deterministic fit reaches the confining Bn threshold."""
    _, meta = sheet_field
    assert meta["fit_metric"] < 1.0e-3, meta
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

    result = solve_free_boundary_multigrid(
        inp, mgrid_path=str(outdir / "mgrid_qi_sheet.nc"), verbose=True,
        emit=collect, raise_on_max_iterations=False)
    output = "\n".join(lines)

    m = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)", output)
    assert m is not None, "vacuum never activated"
    assert abs(int(m.group(1)) - GOLDEN_ACTIVATION) <= 4, m.group(0)
    assert output.rfind("NS = ") > output.find("VACUUM PRESSURE"), (
        "activation did not precede the radial transitions")
    assert bool(result.converged), f"fsqr={float(result.fsqr):.2e}"
    assert float(result.wb) == pytest.approx(GOLDEN_WB, rel=5e-4)
    assert float(result.r00) == pytest.approx(GOLDEN_R00, rel=3e-3)
    iota_edge = float(np.asarray(result.iotaf)[-1])
    assert iota_edge == pytest.approx(GOLDEN_IOTA_EDGE, rel=2e-3)


@pytest.mark.full  # ~25 min: fixed-boundary ladder at full 1e-8 (no floor)
def test_qi_fixed_238_ladder_converges(tmp_path):
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
    result = solve_multigrid(inp, verbose=False, raise_on_max_iterations=False)
    assert bool(result.converged), f"fsqr={float(result.fsqr):.2e}"
