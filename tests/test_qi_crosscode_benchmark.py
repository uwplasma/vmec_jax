"""Fast integrity gates for the committed cross-code QI/QA benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "optimization_crosscode"
INPUT = ROOT / "examples" / "data" / "input.nfp2_QI_seed"


def _rows() -> list[dict]:
    return [
        row
        for objective in ("qi", "qa")
        for row in json.loads(
            (RESULTS / f"{objective}_results.json").read_text()
        )["cases"].values()
    ]


def test_crosscode_matrix_is_complete_comparable_and_monotone() -> None:
    rows = _rows()
    expected = {
        (objective, backend, schedule, ess)
        for objective, schedules in (
            ("qi", [(mode,) for mode in range(1, 9)] + [(1, 2, 3, 4, 5)]),
            ("qa", [(1, 2, 3, 4, 5), (2,), (5,)]),
        )
        for backend in ("simsopt", "vmex")
        for schedule in schedules
        for ess in (False, True)
    }
    assert len(rows) == len(expected) == 48
    by_case = {
        (row["objective"], row["backend"], tuple(row["schedule"]), row["ess"]): row
        for row in rows
    }
    assert set(by_case) == expected
    input_digest = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    for row in rows:
        mode = row["max_mode"]
        assert row["max_nfev"] == 15
        assert row["ns"] == 31
        assert row["dofs"] == 4 * mode * (mode + 1)
        assert row["resolution"] == {
            "mpol": max(mode + 2, 5),
            "ntor": max(mode + 2, 5),
            "ntheta": 2 * max(mode + 2, 5) + 6,
            "nzeta": 2 * max(mode + 2, 5) + 4,
        }
        assert row["provenance"]["input_sha256"] == input_digest
        assert row["total_seconds"] > 0.0
        if row["status"] == "timed_out":
            assert row["censored"] is True
            assert row["total_seconds"] == row["time_limit_seconds"]
        else:
            assert row["status"] == "complete"
        costs = np.asarray(row["accepted_costs"])
        stages = np.asarray(row["accepted_cost_stages"])
        assert stages.shape == costs.shape
        if not costs.size:
            assert row["status"] == "timed_out"
            continue
        if row["status"] == "complete":
            assert costs.size >= 2
        for stage in np.unique(stages):
            stage_costs = costs[stages == stage]
            assert np.all(np.diff(stage_costs) <= 1.0e-12)
        if row["backend"] == "simsopt" and row["status"] == "timed_out":
            continue
        assert costs[0] == row["initial_cost"]
        assert np.isclose(costs[-1], row["final_cost"], rtol=0.0, atol=1.0e-14)
        if row["backend"] == "vmex" and row["status"] == "complete":
            relative = abs(row["initial_cost"] - row["wout_initial_cost"]) / row["initial_cost"]
            # QA's independently reconstructed wout cost is the limiting
            # lane at 1.97e-8; this still detects an objective mismatch while
            # allowing output-rounding differences.
            assert relative < 1.0e-7
            assert row["compilation_cache"] == "disabled"
        elif row["backend"] == "simsopt" and row["status"] == "complete":
            assert row["workers"] == 14
            limits = row["worker_thread_limits"]
            assert all(
                limits[name] == "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            )
            assert limits["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false"
    for objective, schedules in (
        ("qi", [(mode,) for mode in range(1, 9)] + [(1, 2, 3, 4, 5)]),
        ("qa", [(1, 2, 3, 4, 5), (2,), (5,)]),
    ):
        for schedule in schedules:
            costs = [
                by_case[(objective, backend, schedule, ess)]["initial_cost"]
                for backend in ("simsopt", "vmex")
                for ess in (False, True)
                if "initial_cost" in by_case[(objective, backend, schedule, ess)]
            ]
            if len(costs) >= 2:
                # Same tolerance as the independently measured state/wout
                # parity above; the observed maximum is 1.97e-8 for QA.
                assert np.ptp(costs) / np.mean(costs) < 1.0e-7


def test_crosscode_helpers_encode_the_shared_resolution_and_ess_policy() -> None:
    path = ROOT / "benchmarks" / "qi_simsopt_vmex.py"
    spec = importlib.util.spec_from_file_location("qi_simsopt_vmex", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._input_resolution(4) == (6, 6, 18, 16)
    assert module._input_resolution(8) == (10, 10, 26, 24)
    assert module._schedule("1,2,3,4,5") == (1, 2, 3, 4, 5)
    assert module._schedule_label((1, 2, 3, 4, 5)) == "ladder1-5"
    mode1 = module._ess_scale(["RBC(1,0)", "ZBS(-1,1)"])
    assert np.array_equal(mode1, np.ones(2))
    mode2 = module._ess_scale(["RBC(1,0)", "RBC(-2,1)"])
    assert mode2[0] == 1.0
    assert np.isclose(mode2[1], np.exp(-1.2), rtol=1.0e-15, atol=0.0)
