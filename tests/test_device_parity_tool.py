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

CACHE_PATH = PATH.with_name("device_cache_reload.py")
CACHE_SPEC = importlib.util.spec_from_file_location("device_cache_reload", CACHE_PATH)
device_cache_reload = importlib.util.module_from_spec(CACHE_SPEC)
assert CACHE_SPEC.loader is not None
CACHE_SPEC.loader.exec_module(device_cache_reload)


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


def test_cache_reload_device_parser_and_recursive_stats(tmp_path):
    assert device_cache_reload._parse_devices("gpu,cpu,gpu") == ("gpu", "cpu")
    with pytest.raises(ValueError, match="subset of cpu,gpu"):
        device_cache_reload._parse_devices("tpu")
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "first").write_bytes(b"abc")
    (nested / "second").write_bytes(b"12345")
    assert device_cache_reload._cache_stats(tmp_path) == {
        "file_count": 2,
        "bytes": 8,
    }


def _cache_process(*, process_wall, forward_first, forward_warm, gradient_first, gradient_warm):
    return {
        "subprocess_wall_s": process_wall,
        "child": {
            "lanes": {
                "gpu": {
                    "forward": {
                        "cold_wall_s": forward_first,
                        "warm_wall_s": forward_warm,
                    },
                    "gradients": {
                        "mhd_energy": {
                            "cold_wall_s": gradient_first,
                            "warm_wall_s": gradient_warm,
                        }
                    },
                }
            }
        },
    }


def test_cache_reload_speedups_keep_process_compile_and_warm_lanes_separate():
    cold = _cache_process(
        process_wall=30.0,
        forward_first=12.0,
        forward_warm=2.0,
        gradient_first=10.0,
        gradient_warm=1.0,
    )
    reload = _cache_process(
        process_wall=12.0,
        forward_first=4.0,
        forward_warm=2.0,
        gradient_first=3.0,
        gradient_warm=1.0,
    )
    speedups = device_cache_reload._speedups(cold, reload, "gpu")
    assert speedups == {
        "process_cache_reload_speedup": 2.5,
        "forward_cache_reload_speedup": 3.0,
        "gradient_cache_reload_speedup": pytest.approx(10.0 / 3.0),
        "reload_forward_warm_speedup": 2.0,
        "reload_gradient_warm_speedup": 3.0,
    }
    assert device_cache_reload._speedups({}, reload, "gpu") is None
