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
