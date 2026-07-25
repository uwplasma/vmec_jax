"""Drift-aware GCROT recycling at the implicit-solver integration point.

The R25.3 recycled-Jacobian lane threads a GCROT pair through solves whose
operator drifts between accepted trust-region iterates. solvax >= 0.8.7
reports ``recycle_drift`` on every warm start; these tests pin the vmex-side
contract that the drift-gated carry in ``optimize.jacobian_rows_recycled``
relies on: zero on a cold start, machine-floor for an unchanged operator, and
growing with the operator step.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmex.core.implicit as imp

jax.config.update("jax_enable_x64", True)


@pytest.mark.skipif(
    not imp._GCROT_REPORTS_DRIFT,
    reason="installed solvax predates recycle_drift (needs >= 0.8.7)",
)
def test_recycled_solve_surfaces_drift() -> None:
    n = 80
    rng = np.random.default_rng(0)
    base = jnp.asarray(np.eye(n) + 0.4 * rng.standard_normal((n, n)) / np.sqrt(n))
    step = jnp.asarray(rng.standard_normal((n, n)) / np.sqrt(n))
    b = {"f": jnp.asarray(rng.standard_normal(n - 10)),
         "g": jnp.asarray(rng.standard_normal(10))}
    cfg = SimpleNamespace(adjoint_tol=1e-10, adjoint_restart=30, adjoint_maxiter=50)

    def operator(matrix):
        def apply(v):
            flat = jnp.concatenate([v["f"], v["g"]])
            out = matrix @ flat
            return {"f": out[: n - 10], "g": out[n - 10:]}
        return apply

    cold_pair = (jnp.zeros((n, imp._RECYCLE_K)), jnp.zeros((n, imp._RECYCLE_K)))
    _, sol0 = imp._recycled_solve(operator(base), b, cfg, cold_pair)
    assert bool(sol0.converged)
    assert float(sol0.recycle_drift) == 0.0  # cold start

    _, sol_same = imp._recycled_solve(operator(base), b, cfg, sol0.recycle)
    assert float(sol_same.recycle_drift) < 1e-10  # unchanged operator

    drifts = []
    for eps in (1e-4, 1e-3, 1e-2):
        _, sol_moved = imp._recycled_solve(
            operator(base + eps * step), b, cfg, sol0.recycle
        )
        drifts.append(float(sol_moved.recycle_drift))
    assert drifts[0] < drifts[1] < drifts[2]  # grows with the operator step
    assert drifts[2] < 0.5  # well below the carry gate at these steps
