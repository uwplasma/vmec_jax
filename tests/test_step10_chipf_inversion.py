from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax._compat import has_jax
from vmec_jax.field import chips_from_chipf
from vmec_jax.wout import read_wout


@pytest.mark.parametrize(
    "wout_rel",
    [
        "examples/data/wout_circular_tokamak_reference.nc",
        "examples/data/wout_up_down_asymmetric_tokamak_reference.nc",
        "examples/data/wout_li383_low_res_reference.nc",
        "examples/data/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc",
    ],
)
def test_step10_chips_from_chipf_matches_iotas_phips_when_ncurr0(wout_rel: str):
    """In VMEC (ncurr=0), chips(js) = iotas(js)*phips(js) and chipf is a radial average.

    `wout_*.nc` stores chipf with a `2π*signgs` factor, and vmec_jax uses that
    same convention for Step-10 parity work.
    """
    pytest.importorskip("netCDF4")
    if not has_jax():
        pytest.skip("chips_from_chipf requires JAX")

    root = Path(__file__).resolve().parents[1]
    wout_path = root / wout_rel
    assert wout_path.exists()

    wout = read_wout(wout_path)

    chips = np.asarray(chips_from_chipf(wout.chipf))
    chips_expected = (2.0 * np.pi * float(wout.signgs)) * (np.asarray(wout.iotas) * np.asarray(wout.phips))
    np.testing.assert_allclose(chips, chips_expected, rtol=0.0, atol=1e-10)
