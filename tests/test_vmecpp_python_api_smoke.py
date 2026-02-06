from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.vmecpp
def test_vmecpp_python_api_produces_reference_wout_for_circular_tokamak(tmp_path: Path):
    """Optional integration test: run VMEC++ and compare wout to bundled reference."""
    if os.environ.get("VMECPP_INTEGRATION", "0") != "1":
        pytest.skip("Set VMECPP_INTEGRATION=1 to run VMEC++ integration tests")

    pytest.importorskip("netCDF4")

    try:
        import vmecpp  # type: ignore  # noqa: PLC0415
    except Exception as e:
        pytest.skip(f"vmecpp import failed: {e!r}")

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples/data/input.circular_tokamak"
    ref_wout_path = repo_root / "examples/data/wout_circular_tokamak_reference.nc"
    assert input_path.exists()
    assert ref_wout_path.exists()

    vmec_input = vmecpp.VmecInput.from_file(input_path)
    output = vmecpp.run(vmec_input, max_threads=1, verbose=False)

    wout_path = tmp_path / "wout_circular_tokamak_vmecpp.nc"
    output.wout.save(str(wout_path))

    from netCDF4 import Dataset  # noqa: PLC0415

    def _arr(ds, name: str):
        return np.asarray(ds.variables[name][:])

    with Dataset(wout_path) as ds_new, Dataset(ref_wout_path) as ds_ref:
        for field in ["iotaf", "rmnc", "zmns", "lmns", "bmnc"]:
            np.testing.assert_allclose(_arr(ds_new, field), _arr(ds_ref, field), atol=1e-10, rtol=1e-6)
