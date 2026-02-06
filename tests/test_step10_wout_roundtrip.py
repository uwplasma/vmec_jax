from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.wout import read_wout, write_wout


_CASES = [
    "examples/data/wout_circular_tokamak_reference.nc",
    "examples/data/wout_li383_low_res_reference.nc",
]


@pytest.mark.parametrize("wout_rel", _CASES)
def test_step10_wout_roundtrip_read_write_read(tmp_path: Path, wout_rel: str):
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    src = root / wout_rel
    assert src.exists()

    w0 = read_wout(src)
    out = tmp_path / Path(wout_rel).name
    write_wout(out, w0, overwrite=True)
    w1 = read_wout(out)

    # Compare all fields that we read/write.
    assert w0.ns == w1.ns
    assert w0.mpol == w1.mpol
    assert w0.ntor == w1.ntor
    assert w0.nfp == w1.nfp
    assert w0.lasym == w1.lasym
    assert w0.signgs == w1.signgs

    for name in [
        "xm",
        "xn",
        "xm_nyq",
        "xn_nyq",
    ]:
        assert np.array_equal(getattr(w0, name), getattr(w1, name))

    for name in [
        "rmnc",
        "rmns",
        "zmnc",
        "zmns",
        "lmnc",
        "lmns",
        "phipf",
        "chipf",
        "phips",
        "gmnc",
        "gmns",
        "bsupumnc",
        "bsupumns",
        "bsupvmnc",
        "bsupvmns",
        "bsubumnc",
        "bsubumns",
        "bsubvmnc",
        "bsubvmns",
        "bmnc",
        "bmns",
        "vp",
        "pres",
        "presf",
        "fsqt",
    ]:
        a = np.asarray(getattr(w0, name))
        b = np.asarray(getattr(w1, name))
        assert a.shape == b.shape
        assert np.allclose(a, b, rtol=0.0, atol=0.0)

    for name in ["wb", "volume_p", "gamma", "wp", "fsqr", "fsqz", "fsql"]:
        assert float(getattr(w0, name)) == float(getattr(w1, name))
