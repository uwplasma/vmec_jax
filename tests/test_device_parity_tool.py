"""Fast contract tests for the CPU/GPU parity audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "device_parity.py"
SPEC = importlib.util.spec_from_file_location("device_parity", PATH)
device_parity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(device_parity)


def test_requested_devices_skips_an_unavailable_gpu():
    selected, skipped = device_parity._requested_devices("cpu,gpu", {"cpu": object()})
    assert selected == ["cpu"]
    assert skipped == {"gpu": "no GPU JAX device is available"}


@pytest.mark.parametrize(
    "name",
    [
        "dmerc_interior_mean",
        "jdotb_interior_mean",
        "glasser_d_r_interior_mean",
    ],
)
def test_metric_selection_includes_traceable_stability_profiles(name):
    assert name in device_parity.METRIC_NAMES
    assert name in device_parity._metrics(quick=True)


def test_compare_lanes_reports_forward_and_gradient_parity():
    cpu = {"state": np.array([1.0, 2.0]), "metrics": {"mhd_energy": (3.0, 4.0)}}
    gpu = {"state": np.array([1.0, 2.0 + 1e-10]), "metrics": {"mhd_energy": (3.0, 4.0)}}
    comparison = device_parity._compare_lanes(cpu, gpu, rtol=1e-7)
    assert comparison["status"] == "passed"
    assert comparison["forward"]["state_relative_l2"] == pytest.approx(1e-10 / np.sqrt(5.0))
    assert comparison["metrics"]["mhd_energy"]["gradient_relative_difference"] == 0.0
