from __future__ import annotations

from pathlib import Path

from vmec_jax.namelist import read_indata


def test_indata_basic(tmp_path: Path):
    txt = """
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  LASYM = F
  RBC(0,0) = 6.0
  RBC(0,1) = 2.0
  ZBS(0,1) = 2.0
/
"""
    p = tmp_path / "input.test"
    p.write_text(txt)

    indata = read_indata(p)
    assert indata.get_int("NFP") == 1
    assert indata.get_int("MPOL") == 5
    assert indata.get_bool("LASYM") is False

    assert indata.indexed["RBC"][(0,0)] == 6.0
    assert indata.indexed["RBC"][(0,1)] == 2.0
    assert indata.indexed["ZBS"][(0,1)] == 2.0


def test_fortran_repeat_syntax_is_expanded(tmp_path: Path):
    # VMEC inputs commonly use Fortran repeat counts like `11*0.0`.
    txt = """
&INDATA
  AI = 11*0.0, 0.8
/
"""
    p = tmp_path / "input.repeat"
    p.write_text(txt)

    indata = read_indata(p)
    ai = indata.get("AI")
    assert isinstance(ai, list)
    assert len(ai) == 12
    assert all(not isinstance(v, str) for v in ai)
    assert ai[:11] == [0.0] * 11
    assert ai[11] == 0.8
