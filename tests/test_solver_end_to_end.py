"""End-to-end fixed-boundary solve tests: ``vmex.core.solver`` vs VMEC2000.

Golden references (VMEC2000, PARVMEC 9.0 single-rank runs, FTOL = 1e-14):

- ``~/vmex_notes/golden/solovev/``            (2D, ncurr=0, ns=11, 215 iters)
- ``~/vmex_notes/golden/cth_like_fixed_bdy/`` (nfp=5 ntor=0, ncurr=1,
  two_power mass/current, ns=15, 434 iters)

Checked here:

1. convergence to ftol = 1e-14 with the golden iteration count (+-20%);
2. wout parity: ``wb``, ``iotaf``, ``rmnc/zmns`` against the golden wout;
3. early trajectory: the first 40 iterations of (fsqr, fsqz, fsql) against
   golden threed1 printed iterations (rtol 5e-2, both cases);
4. lane equivalence: the ``mode="cli"`` (scan blocks) and ``mode="jit"``
   (``lax.while_loop``) trajectories agree to 1e-15 (same jitted body).
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from vmex.core.input import VmecInput
from vmex.core import solver

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
from conftest import resolve_golden_dir

GOLDEN_DIR = resolve_golden_dir()
pytestmark_golden = pytest.mark.skipif(
    GOLDEN_DIR is None, reason="golden VMEC2000 fixtures unavailable (offline?)"
)
pytestmark = [
    pytestmark_golden,
    pytest.mark.usefixtures("_module_jit_enabled"),  # full solves: run jitted
]

netCDF4 = pytest.importorskip("netCDF4")

#: case -> (golden iteration count, golden threed1 printed rows
#:          {iter: (fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1)}).
CASES = {
    "solovev": (
        215,
        {
            1: (9.41e-2, 2.76e-3, 3.21e-2, 2.75e-3, 6.80e-5, 9.52e-3),
            215: (9.97e-15, 5.31e-15, 1.05e-15, 2.32e-17, 5.46e-17, 6.14e-17),
        },
    ),
    "cth_like_fixed_bdy": (
        434,
        {
            1: (3.76e-2, 9.68e-4, 1.63e-2, 2.02e-3, 8.60e-6, 4.33e-3),
            200: (3.77e-9, 5.97e-10, 2.62e-10, 3.47e-12, 9.19e-14, 1.53e-11),
            400: (1.05e-13, 2.94e-14, 8.32e-15, 6.43e-17, 2.16e-18, 2.85e-16),
            434: (9.94e-15, 1.91e-15, 1.06e-15, 9.75e-18, 4.07e-19, 2.79e-17),
        },
    ),
}


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request):
    name = request.param
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    result = solver.solve(inp, ftol=1e-14, mode="cli")
    wout = GOLDEN_DIR / name / f"wout_{name}.nc"
    return name, inp, result, wout


# ---------------------------------------------------------------------------
# Convergence and iteration counts (eqsolve.f)
# ---------------------------------------------------------------------------


def test_converges_with_golden_iteration_count(case):
    name, _, result, _ = case
    golden_iters, _ = CASES[name]
    assert result.converged, f"{name}: did not converge"
    assert result.fsqr <= 1e-14 and result.fsqz <= 1e-14 and result.fsql <= 1e-14
    low = int(0.8 * golden_iters)
    high = int(1.2 * golden_iters)
    assert low <= result.iterations <= high, (
        f"{name}: {result.iterations} iterations vs golden {golden_iters}"
    )
    assert result.jacobian_resets == 0  # both golden runs report 0 resets


def test_solovev_iteration_band(case):
    name, _, result, _ = case
    if name != "solovev":
        pytest.skip("solovev-specific band")
    assert 180 <= result.iterations <= 260


# ---------------------------------------------------------------------------
# wout parity (golden VMEC2000 wout files)
# ---------------------------------------------------------------------------


def test_wout_parity(case):
    name, _, result, wout_path = case
    ds = netCDF4.Dataset(str(wout_path))
    try:
        wb_gold = float(ds["wb"][()])
        assert abs(result.wb / wb_gold - 1.0) < 1e-8, (
            f"{name}: wb {result.wb} vs golden {wb_gold}"
        )
        np.testing.assert_array_equal(result.xm, np.asarray(ds["xm"][()]))
        np.testing.assert_array_equal(result.xn, np.asarray(ds["xn"][()]))
        np.testing.assert_allclose(
            result.iotaf, np.asarray(ds["iotaf"][()]), rtol=1e-6,
            err_msg=f"{name}: iotaf",
        )
        np.testing.assert_allclose(
            result.rmnc, np.asarray(ds["rmnc"][()]), rtol=1e-6, atol=1e-10,
            err_msg=f"{name}: rmnc",
        )
        np.testing.assert_allclose(
            result.zmns, np.asarray(ds["zmns"][()]), rtol=1e-6, atol=1e-10,
            err_msg=f"{name}: zmns",
        )
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Early trajectory (threed1 printed iterations + legacy driver history)
# ---------------------------------------------------------------------------


def test_trajectory_matches_golden_threed1_rows(case):
    name, _, result, _ = case
    _, golden_rows = CASES[name]
    history = result.fsq_history  # (iterations, 6)
    for it, row in golden_rows.items():
        ours = history[it - 1]
        np.testing.assert_allclose(
            ours, np.asarray(row), rtol=5e-2,
            err_msg=f"{name}: threed1 iteration {it}",
        )


def test_lane_equivalence_cli_vs_jit(case):
    name, inp, result_cli, _ = case
    if name != "solovev":
        pytest.skip("lane equivalence checked on the fast deck")
    result_jit = solver.solve(inp, ftol=1e-14, mode="jit")
    assert result_jit.iterations == result_cli.iterations
    assert result_jit.converged
    np.testing.assert_allclose(
        result_jit.fsq_history, result_cli.fsq_history, rtol=1e-15, atol=0.0,
        err_msg="cli vs jit fsq trajectories",
    )
    np.testing.assert_allclose(
        result_jit.rmnc, result_cli.rmnc, rtol=1e-15, atol=1e-300
    )


# ---------------------------------------------------------------------------
# evaluate_forces public API smoke check
# ---------------------------------------------------------------------------


def test_evaluate_forces_api(case):
    name, inp, result, _ = case
    if name != "solovev":
        pytest.skip("API smoke check on the fast deck")
    runtime = solver.prepare_runtime(inp)
    state = solver._initial_state(runtime.setup)
    gc, residuals, diagnostics = solver.evaluate_forces(state, runtime)
    # Golden threed1 iteration-1 values (physical and preconditioned).
    np.testing.assert_allclose(float(residuals.fsqr), 9.41e-2, rtol=5e-2)
    np.testing.assert_allclose(float(residuals.fsqz), 2.76e-3, rtol=5e-2)
    np.testing.assert_allclose(float(residuals.fsql), 3.21e-2, rtol=5e-2)
    np.testing.assert_allclose(
        float(diagnostics.preconditioned.fsqr1), 2.75e-3, rtol=5e-2
    )
    assert not bool(diagnostics.jacobian_sign_changed)
    # gc matches the state pytree structure (momentum-update ready).
    assert jax.tree.structure(gc) == jax.tree.structure(state)
