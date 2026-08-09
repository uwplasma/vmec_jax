"""Fast integrity gates for the committed cross-code QI benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "qi_crosscode_macos"
INPUT = ROOT / "benchmarks" / "data" / "input.alex_qi_nfp2"


def _rows() -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(RESULTS.glob("*.json"))]


def test_crosscode_matrix_is_complete_comparable_and_monotone() -> None:
    rows = _rows()
    assert len(rows) == 16
    by_case = {(row["backend"], row["max_mode"], row["ess"]): row for row in rows}
    assert set(by_case) == {
        (backend, mode, ess)
        for backend in ("simsopt", "vmex")
        for mode in range(1, 5)
        for ess in (False, True)
    }
    input_digest = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    expected_dofs = {1: 8, 2: 24, 3: 48, 4: 80}
    for row in rows:
        mode = row["max_mode"]
        assert row["max_nfev"] == 15
        assert row["ns"] == 25
        assert row["dofs"] == expected_dofs[mode]
        assert row["resolution"] == {
            "mpol": max(mode + 2, 5),
            "ntor": max(mode + 2, 5),
            "ntheta": 2 * max(mode + 2, 5) + 6,
            "nzeta": 2 * max(mode + 2, 5) + 4,
        }
        assert row["provenance"]["input_sha256"] == input_digest
        assert row["total_seconds"] > 0.0
        costs = np.asarray(row["accepted_costs"])
        assert costs.size >= 2
        assert np.all(np.diff(costs) <= 1.0e-12)
        assert costs[0] == row["initial_cost"]
        assert np.isclose(costs[-1], row["final_cost"], rtol=0.0, atol=1.0e-14)
        if row["backend"] == "vmex":
            relative = abs(row["initial_cost"] - row["wout_initial_cost"]) / row["initial_cost"]
            assert relative < 1.0e-10
        else:
            assert row["workers"] == 14
    for mode in range(1, 5):
        costs = [
            by_case[(backend, mode, ess)]["initial_cost"]
            for backend in ("simsopt", "vmex")
            for ess in (False, True)
        ]
        assert np.ptp(costs) / np.mean(costs) < 1.0e-12


def test_crosscode_helpers_encode_the_shared_resolution_and_ess_policy() -> None:
    path = ROOT / "benchmarks" / "qi_simsopt_vmex.py"
    spec = importlib.util.spec_from_file_location("qi_simsopt_vmex", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._input_resolution(4) == (6, 6, 18, 16)
    mode1 = module._ess_scale(["RBC(1,0)", "ZBS(-1,1)"])
    assert np.array_equal(mode1, np.ones(2))
    mode2 = module._ess_scale(["RBC(1,0)", "RBC(-2,1)"])
    assert mode2[0] == 1.0
    assert np.isclose(mode2[1], np.exp(-1.2), rtol=1.0e-15, atol=0.0)
