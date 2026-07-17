"""Input parsing and writer round-trip tests for ``vmex.core.input``.

For every bundled input deck, :class:`vmex.core.input.VmecInput` is
round-tripped through both writers (JSON and INDATA) and must reproduce every
field exactly.  A VMEC++ JSON example (``data/solovev.json``, copied verbatim
from the vmecpp repository) validates JSON-schema compatibility.  (Field-level
parity with the historical parser stack was proven by the A/B suite that
accompanied the port and retired with the legacy tree.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmex.core.input import VmecInput

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
DECKS = sorted(DATA.glob("input.*"))
FIXTURES = Path(__file__).parent / "data"

assert DECKS, f"no input decks found under {DATA}"


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_json_round_trip(deck: Path, tmp_path: Path) -> None:
    """VmecInput -> to_json -> from_file reproduces every field exactly."""
    original = VmecInput.from_file(deck)
    reread = VmecInput.from_file(original.to_json(tmp_path / "roundtrip.json"))
    assert reread == original


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_indata_round_trip(deck: Path, tmp_path: Path) -> None:
    """VmecInput -> to_indata -> from_file reproduces every field exactly."""
    original = VmecInput.from_file(deck)
    reread = VmecInput.from_file(original.to_indata(tmp_path / "input.roundtrip"))
    assert reread == original


def test_vmecpp_json_example() -> None:
    """The VMEC++ solovev.json example parses with VMEC++ schema semantics."""
    new = VmecInput.from_file(FIXTURES / "solovev.json")
    assert new.mpol == 6
    assert new.ntor == 0
    assert new.nfp == 1  # default
    assert new.lasym is False  # default
    assert new.ncurr == 0
    assert new.nstep == 250
    assert new.delt == 0.9
    np.testing.assert_array_equal(new.ns_array, [5, 11, 55])
    np.testing.assert_array_equal(new.niter_array, [1000, 2000, 2000])
    np.testing.assert_array_equal(new.ftol_array, [1e-16, 1e-16, 1e-16])
    np.testing.assert_array_equal(new.am[:3], [0.125, -0.125, 0.0])
    assert not np.any(new.am[3:])
    np.testing.assert_array_equal(new.ai[:2], [1.0, 0.0])
    np.testing.assert_array_equal(new.raxis_c, [4.0])
    np.testing.assert_array_equal(new.zaxis_s, [0.0])
    # Sparse boundary lists land at [n + ntor, m] = [0, m] for ntor = 0.
    np.testing.assert_array_equal(new.rbc[0, :3], [3.999, 1.026, -0.068])
    np.testing.assert_array_equal(new.zbs[0, :3], [0.0, 1.58, 0.01])
    assert not np.any(new.rbs) and not np.any(new.zbc)
    assert new.lfreeb is False  # VMEC++ default
    assert new.gamma == 0.0  # default (JSON alias: adiabatic_index)
    assert new.mgrid_file == "NONE"

    # And it survives both writers.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        assert VmecInput.from_file(new.to_json(Path(tmp) / "x.json")) == new
        assert VmecInput.from_file(new.to_indata(Path(tmp) / "input.x")) == new
