"""Staged force-oracle tests (``tools/force_oracle.py``): the oracle
replays the PRODUCTION iteration body and records staged funct3d.f-chain
quantities at iterations 1, 2, 25, 26 and after an in-loop Jacobian retry.
Pinned: replayed rows are identical (to reassociation noise) to the
production ``solve_multigrid`` trajectory; the staged chain's recomputed
``fsql`` equals the production one; staged internals match the recorded
goldens under ``tests/data/`` (interior stages are VMEX-regression-only —
VMEC2000 prints only iteration rows); the default output is VALUES-FREE
and stops at the first differing stage; opt-in ``--run-vmec2000`` checks
the cross-code row stages ``R01_FSQR..R05_WMHD``.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import force_oracle
from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_multigrid

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
GOLDEN = ROOT / "tests" / "data" / "force_oracle_solovev.json"
GOLDEN_RETRY = ROOT / "tests" / "data" / "force_oracle_solovev_retry.json"

#: Replay configuration shared with the committed goldens (regenerate with
#: ``python tools/force_oracle.py record examples/data/input.solovev
#: --ns 11 --niter 28 --out tests/data/force_oracle_solovev.json`` and
#: ``... --niter 12 --time-step 3.0 --iterations 1,2 --out
#: tests/data/force_oracle_solovev_retry.json``).
NS, NITER = 11, 28
RETRY_NITER, RETRY_DELT = 12, 3.0


@pytest.fixture(scope="module")
def solovev_replay() -> dict:
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    return force_oracle.replay(inp, ns=NS, niter=NITER)


@pytest.fixture(scope="module")
def retry_replay() -> dict:
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    return force_oracle.replay(
        inp, ns=NS, niter=RETRY_NITER, time_step=RETRY_DELT,
        iterations=(1, 2),
    )


def test_replay_rows_match_the_production_loop(solovev_replay: dict) -> None:
    """The oracle replay cannot drift from the solver it instruments."""
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    reference = solve_multigrid(
        inp, ns_array=[NS], niter_array=[NITER], ftol_array=[1e-14],
        raise_on_max_iterations=False, device="cpu",
    )
    rows = solovev_replay["rows"]
    assert len(rows) >= NITER - 1
    for key, row in rows.items():
        it = int(key)
        np.testing.assert_allclose(
            row[:3], reference.fsq_history[it - 1, :3],
            rtol=1e-9, atol=1e-300,
            err_msg=f"replayed row {it} differs from the production loop",
        )
    assert solovev_replay["termination"] == "ITERATION_BUDGET"


def test_staged_chain_is_self_consistent(solovev_replay: dict) -> None:
    """The recomputed stage chain lands on the production residuals.

    ``S08_FSQL`` records both the staged-chain ``fsql`` and the production
    evaluation's ``fsql`` for the same state; a mismatch means the oracle's
    stage recomputation drifted from ``solver._force_pipeline``.
    """
    for key, stages in solovev_replay["iterations"].items():
        fsql = stages["S08_FSQL"]["fsql"]
        fsql_production = stages["S08_FSQL"]["fsql_production"]
        assert fsql == pytest.approx(fsql_production, rel=1e-10), key
        # the chain is complete and ordered
        assert list(stages) == list(force_oracle.STAGE_ORDER)
        # normalization identity: fsql = fnormL * gcl2
        assert stages["S08_FSQL"]["fsql"] == pytest.approx(
            stages["S07_FNORML"]["fnormL"] * stages["S08_FSQL"]["gcl2"],
            rel=1e-12,
        )


def test_recorded_goldens_pin_the_staged_internals(
    solovev_replay: dict,
) -> None:
    """VMEX-regression stages S01..S10 match the committed goldens."""
    golden = json.loads(GOLDEN.read_text())
    lines: list[str] = []
    rc = force_oracle.compare_to_golden(
        solovev_replay, golden, rtol=5e-6, details=True,
        emit=lambda text="": lines.append(str(text)),
    )
    assert rc == 0, "\n".join(lines)
    assert lines[-1] == "assessment: ORACLE_PASS"
    # both sides of the ns4 = 25 preconditioner refresh are pinned
    assert {"1", "2", "25", "26"} <= set(golden["iterations"])


def test_post_jacobian_retry_stage_is_recorded_and_pinned(
    retry_replay: dict,
) -> None:
    """DELT=3 on solovev produces an in-loop Jacobian retry; the first
    iteration after it is recorded and matches the committed golden."""
    golden = json.loads(GOLDEN_RETRY.read_text())
    assert retry_replay["post_retry_iteration"] == golden[
        "post_retry_iteration"] == 4
    assert "4" in retry_replay["iterations"]
    lines: list[str] = []
    rc = force_oracle.compare_to_golden(
        retry_replay, golden, rtol=5e-6, details=True,
        emit=lambda text="": lines.append(str(text)),
    )
    assert rc == 0, "\n".join(lines)


def test_values_free_output_fails_at_first_differing_stage(
    solovev_replay: dict,
) -> None:
    """Privacy contract: stage codes + PASS/FAIL only, stop at first FAIL.

    Perturbing an early stage (S03) AND a later stage (S05) of the golden
    must report exactly the S03 failure — the later stage is never reached,
    so a confidential report identifies the FIRST diverging chain stage.
    """
    golden = json.loads(GOLDEN.read_text())
    perturbed = copy.deepcopy(golden)
    perturbed["iterations"]["1"]["S03_BCOVAR_FIELDS"]["wb"] *= 1.5
    perturbed["iterations"]["1"]["S05_LAMBDA_SPECTRAL"]["flmn_norm"] *= 1.5
    lines: list[str] = []
    rc = force_oracle.compare_to_golden(
        solovev_replay, perturbed, rtol=5e-6, details=False,
        emit=lambda text="": lines.append(str(text)),
    )
    assert rc == 1
    assert lines[-1] == "assessment: ORACLE_FAIL"
    fails = [ln for ln in lines if "FAIL" in ln and "assessment" not in ln]
    assert fails == ["S03_BCOVAR_FIELDS@iter1: FAIL (wb)"]
    assert not any("S05_LAMBDA_SPECTRAL@iter1" in ln for ln in lines), (
        "comparison did not stop at the first differing stage")
    # values-free: no floating-point magnitudes leak into the default output
    for line in lines:
        assert not re.search(r"\d\.\d+[eE][+-]\d", line), line


@pytest.mark.vmec2000_live
def test_cross_code_row_stages_match_vmec2000(pytestconfig, tmp_path) -> None:
    """Cross-code stages R01..R05 pass against a local xvmec2000.

    Only the printed iteration rows are cross-code (the binary exposes
    nothing else); the interior stages remain VMEX-regression-only.
    """
    configured = str(pytestconfig.getoption("--vmec2000-executable")).strip()
    executable = configured or shutil.which("xvmec2000")
    if not executable or not Path(executable).is_file():
        pytest.fail("--run-vmec2000 requested but xvmec2000 was not found")
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "tools" / "force_oracle.py"), "cross",
            str(DATA / "input.solovev"), "--ns", str(NS),
            "--niter", str(NITER), "--xvmec2000", str(executable),
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "assessment: CROSS_CODE_PASS" in completed.stdout
    for stage in force_oracle.ROW_STAGE_ORDER:
        assert f"{stage}@iter1: PASS" in completed.stdout
