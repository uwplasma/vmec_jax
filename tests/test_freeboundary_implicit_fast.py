"""Fast contracts for projected free-boundary tangents and adjoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

import vmex as vj  # noqa: E402
from vmex.core import freeboundary as FB  # noqa: E402
from vmex.core import freeboundary_implicit as FBI  # noqa: E402
from vmex.core.solver import SpectralState  # noqa: E402


def _state(shape=(1, 2), dtype=jnp.float64):
    leaves = [
        jnp.full(shape, float(index + 1), dtype=dtype)
        for index in range(6)
    ]
    return SpectralState(*leaves)


@dataclass(frozen=True)
class _CurrentField:
    extcur: object
    label: str = "unchanged"


class _ResidualOnly(NamedTuple):
    residual: SpectralState


def test_public_exports_are_lazy_aliases():
    names = (
        "FreeBoundaryAdjointResult",
        "FreeBoundaryStatePullbackResult",
        "FreeBoundaryTangentConfig",
        "FreeBoundaryTangentResult",
        "free_boundary_dof_mask",
        "make_projected_free_boundary_residual",
        "one_current_adjoint",
        "one_current_tangent",
        "scalar_parameter_state_pullback",
        "scalar_parameter_tangent",
        "scalar_state_objective_adjoint",
    )
    assert all(getattr(vj, name) is getattr(FBI, name) for name in names)
    assert vj.freeboundary_implicit is FBI


def test_one_current_tangent_builds_scaled_index_field_and_delegates(monkeypatch):
    state = _state()
    evaluator = object()
    mask = object()
    config = FBI.FreeBoundaryTangentConfig()
    field = _CurrentField(extcur=jnp.asarray([10.0, 20.0, 30.0]))
    sentinel = object()
    captured = {}

    def fake_tangent(actual_evaluator, actual_state, field_from_alpha, **kwargs):
        captured.update(
            evaluator=actual_evaluator,
            state=actual_state,
            field=field_from_alpha(jnp.asarray(2.0)),
            kwargs=kwargs,
        )
        return sentinel

    monkeypatch.setattr(FBI, "scalar_parameter_tangent", fake_tangent)
    actual = FBI.one_current_tangent(
        evaluator,
        state,
        field,
        1,
        current_scale=3.5,
        dof_mask=mask,
        config=config,
    )

    assert actual is sentinel
    assert captured["evaluator"] is evaluator
    assert captured["state"] is state
    assert captured["field"].label == field.label
    np.testing.assert_allclose(
        np.asarray(captured["field"].extcur),
        [10.0, 27.0, 30.0],
    )
    np.testing.assert_allclose(np.asarray(field.extcur), [10.0, 20.0, 30.0])
    assert captured["kwargs"] == {"dof_mask": mask, "config": config}


def test_one_current_adjoint_builds_scaled_index_field_and_delegates(monkeypatch):
    state = _state()
    evaluator = object()
    mask = object()
    config = FBI.FreeBoundaryTangentConfig()
    field = _CurrentField(extcur=jnp.asarray([5.0, 6.0, 7.0]))

    def objective(value):
        return jnp.sum(value.R_cos)

    sentinel = object()
    captured = {}

    def fake_adjoint(
        actual_evaluator,
        actual_state,
        field_from_alpha,
        actual_objective,
        **kwargs,
    ):
        captured.update(
            evaluator=actual_evaluator,
            state=actual_state,
            field=field_from_alpha(jnp.asarray(-0.5)),
            objective=actual_objective,
            kwargs=kwargs,
        )
        return sentinel

    monkeypatch.setattr(FBI, "scalar_state_objective_adjoint", fake_adjoint)
    actual = FBI.one_current_adjoint(
        evaluator,
        state,
        field,
        2,
        objective,
        current_scale=4.0,
        dof_mask=mask,
        config=config,
    )

    assert actual is sentinel
    assert captured["evaluator"] is evaluator
    assert captured["state"] is state
    assert captured["objective"] is objective
    assert captured["field"].label == field.label
    np.testing.assert_allclose(
        np.asarray(captured["field"].extcur),
        [5.0, 6.0, 5.0],
    )
    assert captured["kwargs"] == {"dof_mask": mask, "config": config}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rtol", np.nan, "rtol must be finite"),
        ("rtol", np.inf, "rtol must be finite"),
        ("atol", np.nan, "atol must be finite"),
        ("atol", np.inf, "atol must be finite"),
        ("base_residual_atol", np.nan, "base_residual_atol must be finite"),
        ("base_residual_atol", np.inf, "base_residual_atol must be finite"),
    ],
)
def test_tangent_config_rejects_nonfinite_tolerances(field, value, message):
    config = replace(FBI.FreeBoundaryTangentConfig(), **{field: value})
    with pytest.raises(ValueError, match=message):
        FBI._validated_config(config)


def test_tangent_config_rejects_nonpositive_adjoint_batch_size():
    config = replace(FBI.FreeBoundaryTangentConfig(), adjoint_batch_size=0)
    with pytest.raises(ValueError, match="adjoint_batch_size must be"):
        FBI._validated_config(config)


def test_tangent_config_rejects_unknown_adjoint_backend():
    config = replace(FBI.FreeBoundaryTangentConfig(), adjoint_backend="symmetric")
    with pytest.raises(ValueError, match="adjoint_backend must be"):
        FBI._validated_config(config)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_residual_diagnostic_guard_rejects_every_nonfinite_value(value):
    with pytest.raises(ValueError, match="non-finite residual diagnostic"):
        FBI._require_finite_residual_diagnostics(
            {"projected": value},
            context="test root",
        )


def test_linearized_root_rejects_nonfinite_projected_norm(monkeypatch):
    state = _state()
    nonfinite = replace(state, R_cos=state.R_cos.at[0, 0].set(jnp.nan))

    def fake_projected_residual(*_args):
        def residual(_state, _field):
            return nonfinite

        return residual, state, lambda tree: tree

    monkeypatch.setattr(
        FBI,
        "make_projected_free_boundary_residual",
        fake_projected_residual,
    )

    with pytest.raises(ValueError, match="non-finite residual diagnostic"):
        FBI._linearized_projected_root(
            object(),
            state,
            object(),
            state,
            FBI.FreeBoundaryTangentConfig(),
        )


@pytest.mark.parametrize(
    ("alpha0", "message"),
    [
        (jnp.zeros((2,)), "alpha0 must be scalar"),
        (np.nan, "alpha0 must be finite"),
        (np.inf, "alpha0 must be finite"),
        (-np.inf, "alpha0 must be finite"),
    ],
)
@pytest.mark.parametrize("api", ["tangent", "pullback"])
def test_scalar_apis_reject_invalid_alpha_before_building_field(
    alpha0,
    message,
    api,
):
    state = _state()

    def field_from_alpha(_alpha):
        raise AssertionError("invalid alpha0 reached the field builder")

    with pytest.raises(ValueError, match=message):
        if api == "tangent":
            FBI.scalar_parameter_tangent(
                object(),
                state,
                field_from_alpha,
                alpha0=alpha0,
            )
        else:
            FBI.scalar_parameter_state_pullback(
                object(),
                state,
                field_from_alpha,
                state,
                alpha0=alpha0,
            )


def test_free_boundary_projector_keeps_edge_removes_gauges_and_pairs_m1(
    monkeypatch,
):
    resolution = SimpleNamespace(
        ns=3,
        mpol=2,
        ntor=1,
        ntheta=4,
        nzeta=3,
        nfp=1,
        lasym=False,
    )
    modes = SimpleNamespace(
        m=np.asarray([0, 1, 1]),
        n=np.asarray([0, -1, 1]),
    )
    runtime = SimpleNamespace(
        resolution=resolution,
        modes=modes,
        setup=SimpleNamespace(lconm1=True),
    )

    def evaluate(state, _runtime, _field):
        return _ResidualOnly(state)

    evaluator = FB.FreeBoundaryResidualEvaluator(
        runtime=runtime,
        basis=None,
        _evaluate=evaluate,
    )
    monkeypatch.setattr(
        FBI._fixed_implicit,
        "_m1_pair_columns",
        lambda _spec: (np.asarray([2]), np.asarray([1])),
    )

    state = _state(shape=(3, 3))
    mask = FBI.free_boundary_dof_mask(evaluator)
    assert np.all(np.asarray(mask.R_cos)[-1] == 1.0)
    assert np.all(np.asarray(mask.Z_sin)[-1, 1:] == 1.0)
    assert np.all(np.asarray(mask.L_sin)[-1, 1:] == 1.0)
    assert np.all(np.asarray(mask.R_cos)[0, 1:] == 0.0)
    assert np.all(np.asarray(mask.Z_sin)[:, 0] == 0.0)
    assert np.all(np.asarray(mask.L_sin)[0] == 0.0)
    for name in ("R_sin", "Z_cos", "L_cos"):
        assert np.count_nonzero(np.asarray(getattr(mask, name))) == 0

    residual, root, projector = FBI.make_projected_free_boundary_residual(
        evaluator,
        state,
        mask,
    )
    projected = projector(state)
    projected_twice = projector(projected)
    for once, twice in zip(
        jax.tree.leaves(projected),
        jax.tree.leaves(projected_twice),
    ):
        np.testing.assert_array_equal(np.asarray(once), np.asarray(twice))
    np.testing.assert_array_equal(
        np.asarray(projected.Z_sin[:, 1]),
        np.asarray(projected.Z_sin[:, 2]),
    )
    for actual, expected in zip(
        jax.tree.leaves(residual(root, jnp.asarray(0.0))),
        jax.tree.leaves(projected),
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    bad_mask = replace(mask, R_cos=mask.R_cos.at[-1, 0].set(0.5))
    with pytest.raises(ValueError, match="binary"):
        FBI.make_projected_free_boundary_residual(evaluator, state, bad_mask)


def test_projected_executable_reuses_resolution_static_wrapper(monkeypatch):
    state = _state()
    resolution = object()

    def shared_evaluate(*_args):
        return None

    first = FB.FreeBoundaryResidualEvaluator(
        runtime=SimpleNamespace(
            resolution=resolution,
            setup=SimpleNamespace(lconm1=True),
        ),
        basis=None,
        _evaluate=shared_evaluate,
        _runtime_argument_reusable=True,
    )
    second = FB.FreeBoundaryResidualEvaluator(
        runtime=SimpleNamespace(
            resolution=resolution,
            setup=SimpleNamespace(lconm1=True),
        ),
        basis=None,
        _evaluate=shared_evaluate,
        _runtime_argument_reusable=True,
    )
    monkeypatch.setattr(FBI, "_PROJECTED_RESIDUAL_EXECUTABLE_CACHE", {})
    monkeypatch.setattr(FBI, "_projector", lambda *_args: lambda tree: tree)

    first_executable = FBI._projected_residual_executable(first, state)
    second_executable = FBI._projected_residual_executable(second, state)

    assert first_executable is not None
    assert second_executable is first_executable
    assert first_executable.evaluate is shared_evaluate
    assert first.runtime is not second.runtime


@pytest.mark.usefixtures("_module_jit_enabled")
@pytest.mark.parametrize(
    "adjoint_backend",
    ["forward_dense", "forward_dense_jax", "reverse_gmres"],
)
def test_scalar_tangent_and_adjoint_match_nonsymmetric_analytic_root(
    monkeypatch,
    adjoint_backend,
):
    state = jax.tree.map(jnp.zeros_like, _state())
    flat_state, unravel = ravel_pytree(state)
    size = int(flat_state.size)
    dof_mask = jax.tree.map(jnp.ones_like, state)
    evaluator = SimpleNamespace(
        runtime=SimpleNamespace(
            setup=SimpleNamespace(lconm1=False),
            resolution=SimpleNamespace(ntor=0),
        )
    )

    diagonal = np.linspace(1.0, 2.0, size)
    state_matrix = np.diag(diagonal)
    state_matrix += 0.03 * np.diag(np.ones(size - 1), k=1)
    state_matrix -= 0.01 * np.diag(np.ones(size - 2), k=-2)
    parameter_column = np.sin(0.17 * np.arange(1, size + 1)) + 0.1
    state_matrix_jax = jnp.asarray(state_matrix)
    parameter_column_jax = jnp.asarray(parameter_column)
    alpha0 = jnp.asarray(0.2)

    def fake_projected_residual(_evaluator, frozen_state, actual_mask):
        assert frozen_state is state
        assert actual_mask is dof_mask

        def residual(value, alpha):
            flat_value = ravel_pytree(value)[0]
            forcing = (
                state_matrix_jax @ flat_value
                + parameter_column_jax * (alpha - alpha0)
            )
            return unravel(forcing)

        return residual, state, lambda tree: tree

    monkeypatch.setattr(
        FBI,
        "make_projected_free_boundary_residual",
        fake_projected_residual,
    )
    config = FBI.FreeBoundaryTangentConfig(
        rtol=1.0e-11,
        restart=size,
        max_restarts=5,
        adjoint_backend=adjoint_backend,
        adjoint_batch_size=4,
    )
    field_from_alpha = lambda alpha: alpha  # noqa: E731
    tangent = FBI.scalar_parameter_tangent(
        evaluator,
        state,
        field_from_alpha,
        alpha0=alpha0,
        dof_mask=dof_mask,
        config=config,
    )

    state_cotangent_flat = np.cos(0.13 * np.arange(size)) + 0.2
    state_cotangent = unravel(jnp.asarray(state_cotangent_flat))
    pullback = FBI.scalar_parameter_state_pullback(
        evaluator,
        state,
        field_from_alpha,
        state_cotangent,
        alpha0=alpha0,
        dof_mask=dof_mask,
        config=config,
    )

    expected_tangent = -np.linalg.solve(state_matrix, parameter_column)
    expected_adjoint = np.linalg.solve(state_matrix.T, state_cotangent_flat)
    expected_derivative = -expected_adjoint @ parameter_column
    np.testing.assert_allclose(
        np.asarray(ravel_pytree(tangent.state_tangent)[0]),
        expected_tangent,
        rtol=1.0e-9,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(ravel_pytree(pullback.adjoint)[0]),
        expected_adjoint,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        float(pullback.parameter_cotangent),
        expected_derivative,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert bool(tangent.converged)
    assert bool(tangent.krylov_converged)
    assert bool(pullback.converged)
    assert bool(pullback.linear_solver_converged)
    assert pullback.backend == adjoint_backend
    assert float(tangent.base_residual_norm) == pytest.approx(0.0)
    assert float(pullback.base_residual_norm) == pytest.approx(0.0)
    assert float(tangent.relative_linear_residual) < 1.0e-9
    assert float(pullback.relative_adjoint_residual) < 1.0e-11
    np.testing.assert_allclose(
        float(FBI._tree_dot(state_cotangent, tangent.state_tangent)),
        float(pullback.parameter_cotangent),
        rtol=1.0e-9,
        atol=1.0e-10,
    )

    def objective(value):
        return FBI._tree_dot(state_cotangent, value)

    objective_adjoint = FBI.scalar_state_objective_adjoint(
        evaluator,
        state,
        field_from_alpha,
        objective,
        alpha0=alpha0,
        dof_mask=dof_mask,
        config=config,
    )
    assert float(objective_adjoint.objective_value) == pytest.approx(0.0)
    np.testing.assert_allclose(
        float(objective_adjoint.derivative),
        expected_derivative,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
