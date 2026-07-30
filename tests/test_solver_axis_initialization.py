"""Production-axis initialization parity with VMEC2000."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from vmex.core.input import VmecInput
from vmex.core.setup import run_setup
from vmex.core.solver import prepare_runtime, resolution_from_input


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_production_runtime_does_not_preinfer_missing_axis() -> None:
    """eqsolve, not setup, owns VMEC2000's one allowed guess_axis transfer."""
    inp = VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
    zeros = np.zeros(inp.ntor + 1)
    inp = dataclasses.replace(
        inp,
        raxis_c=zeros,
        raxis_s=zeros,
        zaxis_c=zeros,
        zaxis_s=zeros,
    )
    resolution = resolution_from_input(inp)

    runtime = prepare_runtime(inp, resolution)
    np.testing.assert_array_equal(runtime.setup.raxis_c, zeros)
    np.testing.assert_array_equal(runtime.setup.raxis_s, zeros)
    np.testing.assert_array_equal(runtime.setup.zaxis_c, zeros)
    np.testing.assert_array_equal(runtime.setup.zaxis_s, zeros)

    inferred = run_setup(
        inp, resolution, infer_axis_if_missing=True,
    )
    assert np.any(np.asarray(inferred.raxis_c) != 0.0)

    # Geometry/force-kernel callers can opt into the setup convenience
    # explicitly without changing the production default.
    inferred_runtime = prepare_runtime(inp, resolution, setup=inferred)
    np.testing.assert_array_equal(
        inferred_runtime.setup.raxis_c, inferred.raxis_c,
    )
