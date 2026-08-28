"""High/low transfer and stored raw-block preconditioner tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.polish import (
    HighOrderCorrection,
    PreconditionerRefreshPolicy,
    PreconditionerSnapshot,
    build_low_order_preconditioner,
    make_high_low_transfer,
    preconditioner_quality,
    preconditioner_refresh_decision,
)
from vmex.core.strong_force import lift_high_order_state

jax.config.update("jax_enable_x64", True)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _tree_dot(left, right):
    return sum(
        jnp.vdot(a, b).real
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _tree_norm(value) -> float:
    return float(jnp.sqrt(_tree_dot(value, value)))


def _random_like(value, seed: int):
    leaves, structure = jax.tree.flatten(value)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree.unflatten(
        structure,
        [jax.random.normal(key, leaf.shape, leaf.dtype) for key, leaf in zip(keys, leaves)],
    )


@pytest.fixture(scope="module")
def small_adapter():
    inp = VmecInput.from_file(DATA / "input.solovev").change_resolution(
        mpol=3,
        ntor=0,
        ntheta=12,
        nzeta=4,
    )
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )
    config = implicit.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    params = implicit.params_from_input(inp)
    state, mask = implicit.solve_implicit_with_aux(params, config)
    runtime = implicit.runtime_from_params(params, config)
    native = lift_high_order_state(state, runtime, degree=3)
    adapter = build_low_order_preconditioner(
        native,
        params,
        config,
        state,
        mask,
        probe_chunk_size=4,
    )
    return native, runtime, state, mask, adapter


def test_transfer_preserves_constraints_and_roundtrips_range(small_adapter):
    native, _, _, _, adapter = small_adapter
    transfer = adapter.transfer
    high = _random_like(transfer.zeros_high(jnp.float64), 1)
    projected = transfer.project_high(high)
    low = jax.jit(transfer.restrict)(high)
    roundtrip = transfer.restrict(transfer.prolong(low))

    assert native.radial_basis.size == transfer.ns
    for name in ("R_cos", "R_sin", "Z_cos", "Z_sin"):
        np.testing.assert_array_equal(np.asarray(getattr(projected, name)[:, -1]), 0.0)
        np.testing.assert_array_equal(np.asarray(getattr(low, name)[-1]), 0.0)
    for name in ("R_sin", "Z_cos", "L_cos"):
        np.testing.assert_array_equal(np.asarray(getattr(projected, name)), 0.0)
    gauge = (transfer.m == 0) & (transfer.n == 0)
    np.testing.assert_array_equal(np.asarray(projected.L_sin[gauge]), 0.0)
    difference = jax.tree.map(jnp.subtract, roundtrip, low)
    assert _tree_norm(difference) <= 2.0e-12 * max(_tree_norm(low), 1.0)


def test_transfer_forward_and_transpose_are_exact_duals(small_adapter):
    *_, adapter = small_adapter
    transfer = adapter.transfer
    high = _random_like(transfer.zeros_high(jnp.float64), 2)
    high_bar = _random_like(transfer.zeros_high(jnp.float64), 3)
    low = _random_like(transfer.restrict(high), 4)
    low_bar = _random_like(transfer.restrict(high), 5)

    lhs_restrict = _tree_dot(transfer.restrict(high), low_bar)
    rhs_restrict = _tree_dot(high, transfer.restrict_transpose(low_bar))
    lhs_prolong = _tree_dot(transfer.prolong(low), high_bar)
    rhs_prolong = _tree_dot(low, transfer.prolong_transpose(high_bar))
    np.testing.assert_allclose(lhs_restrict, rhs_restrict, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(lhs_prolong, rhs_prolong, rtol=2.0e-13, atol=2.0e-13)


def test_stored_block_preconditioner_reuses_factors_and_transposes(small_adapter):
    *_, adapter = small_adapter
    transfer = adapter.transfer
    left = _random_like(transfer.zeros_high(jnp.float64), 6)
    right = _random_like(transfer.zeros_high(jnp.float64), 7)
    applied = adapter.apply(left)
    applied_again = adapter.apply(left)
    transpose = adapter.apply_transpose(right)

    assert adapter.factor_build_seconds > 0.0
    for first, second in zip(
        jax.tree.leaves(applied), jax.tree.leaves(applied_again), strict=True
    ):
        np.testing.assert_array_equal(first, second)
        assert np.all(np.isfinite(np.asarray(first)))
    lhs = _tree_dot(applied, right)
    rhs = _tree_dot(left, transpose)
    np.testing.assert_allclose(lhs, rhs, rtol=2.0e-10, atol=2.0e-10)


def test_transfer_validation_and_quality_metric(small_adapter):
    native, runtime, *_ = small_adapter
    invalid = dataclasses.replace(native, m=np.asarray(native.m) + 1)
    with pytest.raises(ValueError, match="mode tables"):
        make_high_low_transfer(invalid, runtime)

    transfer = small_adapter[-1].transfer
    malformed = dataclasses.replace(
        transfer.zeros_high(),
        R_cos=jnp.zeros((transfer.mnmax, transfer.nbasis + 1)),
    )
    with pytest.raises(ValueError, match="R_cos has shape"):
        transfer.project_high(malformed)

    one = transfer.zeros_high(jnp.float64)
    one = HighOrderCorrection(
        *(jnp.ones_like(leaf) for leaf in jax.tree.leaves(one))
    )
    probes = jax.tree.map(lambda value: jnp.stack((value, 2.0 * value)), one)
    quality = preconditioner_quality(lambda value: value, lambda value: value, probes)
    np.testing.assert_array_equal(quality.relative_residual, 0.0)
    assert float(quality.maximum) == 0.0
    assert float(quality.rms) == 0.0


def test_factor_refresh_policy_reports_every_trigger():
    previous = PreconditionerSnapshot(
        alpha=0.1,
        radial_degree=3,
        radial_size=5,
        krylov_iterations=10,
        relative_residual=0.1,
        jacobian_margin=2.0,
    )
    stable = dataclasses.replace(previous, alpha=0.2)
    assert preconditioner_refresh_decision(previous, stable) == (False, ())

    degraded = PreconditionerSnapshot(
        alpha=0.5,
        radial_degree=5,
        radial_size=9,
        krylov_iterations=81,
        relative_residual=0.6,
        jacobian_margin=1.0,
        parameter_distance=0.2,
        transpose_converged=False,
    )
    decision = preconditioner_refresh_decision(previous, degraded)
    assert decision.refresh
    assert decision.reasons == (
        "continuation-step",
        "radial-grid",
        "krylov-work",
        "linear-quality",
        "jacobian-margin",
        "parameter-distance",
        "transpose-certificate",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_alpha_change", 0.0, "max_alpha_change"),
        ("max_krylov_iterations", 0, "max_krylov_iterations"),
        ("max_relative_residual", 0.0, "max_relative_residual"),
        ("min_jacobian_margin_ratio", 0.0, "min_jacobian_margin_ratio"),
        ("max_parameter_distance", 0.0, "max_parameter_distance"),
    ],
)
def test_factor_refresh_policy_rejects_invalid_thresholds(field, value, message):
    with pytest.raises(ValueError, match=message):
        PreconditionerRefreshPolicy(**{field: value})
