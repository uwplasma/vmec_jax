"""Fast contracts for projected free-boundary tangents and adjoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
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
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.solver import SpectralState  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy"


def _state(shape=(1, 2), dtype=jnp.float64):
    leaves = [
        jnp.full(shape, float(index + 1), dtype=dtype)
        for index in range(6)
    ]
    return SpectralState(*leaves)


def _minimal_implicit_config(
    *,
    state=None,
    evaluator=None,
    dof_mask=None,
    anchor_result=None,
    branch_inp=None,
    field_from_alpha=lambda alpha: alpha,
    continuation_step=0.1,
    max_continuation_steps=4,
    preserve_m1_constraint_slice=False,
    include_edge_in_convergence=False,
    edge_force_tolerance=None,
    linear_config=None,
):
    state = _state() if state is None else state
    dof_mask = jax.tree.map(jnp.ones_like, state) if dof_mask is None else dof_mask
    anchor_result = (
        SimpleNamespace(state=state)
        if anchor_result is None
        else anchor_result
    )
    return FBI.FreeBoundaryImplicitConfig(
        inp=None,
        field_from_alpha=field_from_alpha,
        alpha_anchor=0.0,
        resolution=SimpleNamespace(ns=1, mnmax=1, nznt=1),
        ftol=1.0e-10,
        max_iterations=20,
        continuation_step=continuation_step,
        max_continuation_steps=max_continuation_steps,
        anchor_result=anchor_result,
        anchor_state=state,
        anchor_iterations=1,
        anchor_residual_norm=0.0,
        anchor_raw_residual_norm=0.0,
        anchor_projected_residual_max_abs=0.0,
        anchor_volume=1.0,
        branch_inp=branch_inp,
        evaluator=evaluator,
        dof_mask=dof_mask,
        linear_config=(
            FBI.FreeBoundaryTangentConfig()
            if linear_config is None
            else linear_config
        ),
        preserve_m1_constraint_slice=preserve_m1_constraint_slice,
        include_edge_in_convergence=include_edge_in_convergence,
        edge_force_tolerance=edge_force_tolerance,
    )


def _register_implicit_config(cfg, *, anchor_solve=False):
    FBI._config_lock(cfg)
    FBI._FREEB_IMPLICIT_SOLVES[cfg] = {
        FBI._alpha_key(cfg.alpha_anchor): cfg.anchor_result,
    }
    FBI._FREEB_IMPLICIT_ROOTS[cfg] = {
        FBI._alpha_key(cfg.alpha_anchor): {
            "raw_residual_norm": cfg.anchor_raw_residual_norm,
            "projected_residual_norm": cfg.anchor_residual_norm,
            "projected_residual_max_abs": (
                cfg.anchor_projected_residual_max_abs
            ),
            "constraint_slice_defect_norm": 0.0,
            "constraint_slice_atol": 1.0e-10,
        },
    }
    FBI._FREEB_IMPLICIT_STATS[cfg] = FBI._new_implicit_stats(
        anchor_solve=anchor_solve,
    )


@dataclass(frozen=True)
class _CurrentField:
    extcur: object
    label: str = "unchanged"


class _ResidualOnly(NamedTuple):
    residual: SpectralState


def test_public_exports_are_lazy_aliases():
    names = (
        "FreeBoundaryAdjointResult",
        "FreeBoundaryImplicitConfig",
        "FreeBoundaryStatePullbackResult",
        "FreeBoundaryTangentConfig",
        "FreeBoundaryTangentResult",
        "free_boundary_dof_mask",
        "free_boundary_implicit_result",
        "free_boundary_implicit_stats",
        "make_free_boundary_implicit_config",
        "make_projected_free_boundary_residual",
        "one_current_adjoint",
        "one_current_tangent",
        "reset_free_boundary_implicit_stats",
        "scalar_parameter_state_pullback",
        "scalar_parameter_tangent",
        "scalar_state_objective_adjoint",
        "solve_free_boundary_implicit",
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


def test_implicit_factory_gates_projected_root_and_uses_auto_device(
    monkeypatch,
):
    inp = VmecInput.from_file(DECK)
    state = _state()
    residual = state
    mask = jax.tree.map(jnp.ones_like, state)
    result = SimpleNamespace(
        converged=True,
        state=state,
        iterations=7,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        fedge=0.0,
    )
    captured = {}

    class FakeEvaluator:
        runtime = object()

        def __call__(self, _state, _field):
            return _ResidualOnly(residual)

    evaluator = FakeEvaluator()

    def fake_solve(*_args, **kwargs):
        captured.update(kwargs)
        return result

    def projector(tree):
        return replace(
            tree,
            R_cos=tree.R_cos.at[0, 0].set(0.0),
        )

    monkeypatch.setattr(FBI, "solve_free_boundary", fake_solve)
    monkeypatch.setattr(FBI, "_branch_input_from_result", lambda *_args: inp)
    monkeypatch.setattr(
        FBI,
        "make_free_boundary_residual_evaluator",
        lambda *_args, **_kwargs: evaluator,
    )
    monkeypatch.setattr(FBI, "free_boundary_dof_mask", lambda _evaluator: mask)
    monkeypatch.setattr(FBI, "_projector", lambda *_args: projector)
    monkeypatch.setattr(
        FBI._fixed_implicit,
        "plasma_volume",
        lambda *_args: 1.25,
    )

    config = FBI.make_free_boundary_implicit_config(
        inp,
        lambda alpha: alpha,
        linear_config=FBI.FreeBoundaryTangentConfig(
            base_residual_atol=100.0,
        ),
    )
    projected = projector(residual)

    assert config.anchor_raw_residual_norm == pytest.approx(
        float(FBI._tree_norm(residual))
    )
    assert config.anchor_residual_norm == pytest.approx(
        float(FBI._tree_norm(projected))
    )
    assert config.anchor_projected_residual_max_abs == pytest.approx(
        max(
            float(jnp.max(jnp.abs(leaf)))
            for leaf in jax.tree.leaves(projected)
        )
    )
    assert config.anchor_residual_norm < config.anchor_raw_residual_norm
    assert captured["device"] == "auto"
    assert FBI.free_boundary_implicit_stats(config)["anchor_host_solves"] == 1


def test_implicit_projected_root_gate_fails_closed():
    state = _state()
    mask = jax.tree.map(jnp.ones_like, state)

    class FakeEvaluator:
        runtime = SimpleNamespace(
            setup=SimpleNamespace(lconm1=False),
            resolution=SimpleNamespace(ntor=0, lasym=False),
        )

        def __call__(self, actual_state, _field):
            return _ResidualOnly(actual_state)

    with pytest.raises(RuntimeError, match=r"not a root.*\|\|P\(F\)\|\|"):
        FBI._implicit_projected_root_diagnostics(
            FakeEvaluator(),
            mask,
            state,
            object(),
            base_residual_atol=1.0e-12,
            context="test continuation point",
        )


def test_implicit_root_gate_rejects_common_chart_defect():
    reference = jax.tree.map(jnp.zeros_like, _state())
    state = replace(
        reference,
        R_sin=reference.R_sin.at[0, 0].set(1.0e-3),
    )
    mask = jax.tree.map(jnp.ones_like, state)
    mask = replace(mask, R_sin=jnp.zeros_like(mask.R_sin))
    zero_residual = jax.tree.map(jnp.zeros_like, state)

    class FakeEvaluator:
        runtime = SimpleNamespace(
            setup=SimpleNamespace(lconm1=False),
            resolution=SimpleNamespace(ntor=0, lasym=False),
        )

        def __call__(self, _state, _field):
            return _ResidualOnly(zero_residual)

    with pytest.raises(RuntimeError, match="common implicit constraint slice"):
        FBI._implicit_projected_root_diagnostics(
            FakeEvaluator(),
            mask,
            state,
            object(),
            base_residual_atol=1.0e-12,
            context="test continuation point",
            reference_state=reference,
        )


def test_scalar_continuation_restarts_each_target_from_common_anchor(
    monkeypatch,
):
    anchor_state = jax.tree.map(jnp.zeros_like, _state())
    anchor_result = SimpleNamespace(state=anchor_state)
    branch_inp = object()
    config = _minimal_implicit_config(
        state=anchor_state,
        anchor_result=anchor_result,
        branch_inp=branch_inp,
        preserve_m1_constraint_slice=True,
        include_edge_in_convergence=True,
        edge_force_tolerance=3.0e-10,
    )
    _register_implicit_config(config)
    calls = []

    def fake_solve(source, *, external_field, initial_state, **kwargs):
        accepted_state = jax.tree.map(
            lambda leaf: leaf + float(len(calls) + 1),
            anchor_state,
        )
        result = SimpleNamespace(
            converged=True,
            iterations=len(calls) + 1,
            fsqr=0.0,
            fsqz=0.0,
            fsql=0.0,
            fedge=0.0,
            state=accepted_state,
        )
        calls.append(
            {
                "source": source,
                "alpha": float(np.asarray(external_field)),
                "initial_state": initial_state,
                "kwargs": kwargs,
                "result": result,
            }
        )
        return result

    monkeypatch.setattr(FBI, "solve_free_boundary", fake_solve)
    monkeypatch.setattr(
        FBI,
        "_implicit_projected_root_diagnostics",
        lambda *_args, **_kwargs: {
            "raw_residual_norm": 0.0,
            "projected_residual_norm": 0.0,
            "projected_residual_max_abs": 0.0,
            "constraint_slice_defect_norm": 0.0,
            "constraint_slice_atol": 1.0e-10,
        },
    )
    monkeypatch.setattr(
        FBI,
        "_branch_input_from_result",
        lambda *_args: pytest.fail("preserved m=1 path must not rebind input"),
    )

    first = FBI._solve_alpha_from_anchor(config, 0.15)
    second = FBI._solve_alpha_from_anchor(config, -0.05)

    assert first is calls[1]["result"]
    assert second is calls[2]["result"]
    assert [call["source"] for call in calls] == [branch_inp] * 3
    np.testing.assert_allclose(
        [call["alpha"] for call in calls],
        [0.075, 0.15, -0.05],
    )
    assert calls[0]["initial_state"] is anchor_state
    assert calls[1]["initial_state"] is calls[0]["result"].state
    assert calls[2]["initial_state"] is anchor_state
    for call in calls:
        assert call["kwargs"]["preserve_m1_constraint_slice"] is True
        assert call["kwargs"]["include_edge_in_convergence"] is True
        assert call["kwargs"]["edge_force_tolerance"] == 3.0e-10
        assert call["kwargs"]["device"] == "auto"
    stats = FBI.free_boundary_implicit_stats(config)
    assert stats["forward_host_solves"] == 3
    assert stats["forward_iterations"] == 6
    assert stats["memo_entries"] == 3
    assert stats["last_forward_projected_residual_norm"] == 0.0


def test_scalar_cold_rebind_path_restarts_from_original_branch_input(
    monkeypatch,
):
    anchor_state = jax.tree.map(jnp.zeros_like, _state())
    branch_inp = object()
    config = _minimal_implicit_config(
        state=anchor_state,
        branch_inp=branch_inp,
        preserve_m1_constraint_slice=False,
    )
    _register_implicit_config(config)
    calls = []
    rebounds = []

    def fake_solve(source, *, external_field, initial_state, **_kwargs):
        result = SimpleNamespace(
            converged=True,
            iterations=1,
            fsqr=0.0,
            fsqz=0.0,
            fsql=0.0,
            state=anchor_state,
        )
        calls.append(
            (source, float(np.asarray(external_field)), initial_state, result)
        )
        return result

    def fake_rebind(source, result):
        rebound = object()
        rebounds.append((source, result, rebound))
        return rebound

    monkeypatch.setattr(FBI, "solve_free_boundary", fake_solve)
    monkeypatch.setattr(FBI, "_branch_input_from_result", fake_rebind)
    monkeypatch.setattr(
        FBI,
        "_implicit_projected_root_diagnostics",
        lambda *_args, **_kwargs: {
            "raw_residual_norm": 0.0,
            "projected_residual_norm": 0.0,
            "projected_residual_max_abs": 0.0,
            "constraint_slice_defect_norm": 0.0,
            "constraint_slice_atol": 1.0e-10,
        },
    )

    FBI._solve_alpha_from_anchor(config, 0.15)
    FBI._solve_alpha_from_anchor(config, -0.05)

    assert calls[0][0] is branch_inp
    assert calls[1][0] is rebounds[0][2]
    assert calls[2][0] is branch_inp
    assert all(call[2] is None for call in calls)
    np.testing.assert_allclose(
        [call[1] for call in calls],
        [0.075, 0.15, -0.05],
    )


def test_scalar_continuation_rejects_intermediate_root_and_stops(
    monkeypatch,
):
    anchor_state = jax.tree.map(jnp.zeros_like, _state())
    config = _minimal_implicit_config(
        state=anchor_state,
        branch_inp=object(),
        preserve_m1_constraint_slice=True,
    )
    _register_implicit_config(config)
    solve_points = []
    gate_points = []

    def fake_solve(_source, *, external_field, **_kwargs):
        solve_points.append(float(np.asarray(external_field)))
        return SimpleNamespace(
            converged=True,
            iterations=1,
            fsqr=0.0,
            fsqz=0.0,
            fsql=0.0,
            state=anchor_state,
        )

    def fake_gate(_evaluator, _mask, _state, field, **_kwargs):
        gate_points.append(float(np.asarray(field)))
        if len(gate_points) == 2:
            raise RuntimeError("intermediate point is not a projected root")
        return {
            "raw_residual_norm": 0.0,
            "projected_residual_norm": 0.0,
            "projected_residual_max_abs": 0.0,
            "constraint_slice_defect_norm": 0.0,
            "constraint_slice_atol": 1.0e-10,
        }

    monkeypatch.setattr(FBI, "solve_free_boundary", fake_solve)
    monkeypatch.setattr(FBI, "_implicit_projected_root_diagnostics", fake_gate)

    target = 0.25
    with pytest.raises(RuntimeError, match="intermediate point"):
        FBI._solve_alpha_from_anchor(config, target)

    np.testing.assert_allclose(solve_points, [1.0 / 12.0, 1.0 / 6.0])
    np.testing.assert_allclose(gate_points, solve_points)
    target_key = FBI._alpha_key(target)
    assert target_key not in FBI._FREEB_IMPLICIT_SOLVES[config]
    assert target_key not in FBI._FREEB_IMPLICIT_ROOTS[config]
    stats = FBI.free_boundary_implicit_stats(config)
    assert stats["forward_host_solves"] == 2
    assert stats["forward_failures"] == 1
    assert stats["last_forward_projected_residual_norm"] is None


def test_forward_diagnostics_follow_failure_and_exact_memo_recovery():
    config = _minimal_implicit_config(
        continuation_step=0.1,
        max_continuation_steps=2,
    )
    _register_implicit_config(config)

    assert FBI._solve_alpha_from_anchor(config, 0.0) is config.anchor_result
    anchored = FBI.free_boundary_implicit_stats(config)
    assert anchored["last_forward_alpha"] == 0.0
    assert anchored["last_forward_projected_residual_norm"] == 0.0

    with pytest.raises(RuntimeError, match="target requires 3"):
        FBI._solve_alpha_from_anchor(config, 0.25)
    failed = FBI.free_boundary_implicit_stats(config)
    assert failed["last_forward_alpha"] == 0.25
    assert failed["last_forward_projected_residual_norm"] is None
    assert failed["last_forward_error"] is not None

    assert FBI._solve_alpha_from_anchor(config, 0.0) is config.anchor_result
    recovered = FBI.free_boundary_implicit_stats(config)
    assert recovered["last_forward_alpha"] == 0.0
    assert recovered["last_forward_projected_residual_norm"] == 0.0
    assert recovered["last_forward_error"] is None


@pytest.mark.usefixtures("_module_jit_enabled")
def test_scalar_custom_vjp_matches_analytic_pullback_and_direct_term(
    monkeypatch,
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
    state_matrix = np.diag(np.linspace(1.0, 2.0, size))
    state_matrix += 0.04 * np.diag(np.ones(size - 1), k=1)
    parameter_column = np.sin(0.19 * np.arange(1, size + 1)) + 0.2
    state_cotangent_flat = np.cos(0.11 * np.arange(size)) + 0.3
    state_matrix_jax = jnp.asarray(state_matrix)
    parameter_column_jax = jnp.asarray(parameter_column)
    state_cotangent = unravel(jnp.asarray(state_cotangent_flat))

    def fake_projected_residual(_evaluator, frozen_state, actual_mask):
        assert actual_mask is dof_mask

        def residual(value, alpha):
            flat_value = ravel_pytree(value)[0]
            return unravel(
                state_matrix_jax @ flat_value
                + parameter_column_jax * alpha
            )

        return residual, frozen_state, lambda tree: tree

    monkeypatch.setattr(
        FBI,
        "make_projected_free_boundary_residual",
        fake_projected_residual,
    )
    config = _minimal_implicit_config(
        state=state,
        evaluator=evaluator,
        dof_mask=dof_mask,
        field_from_alpha=lambda alpha: alpha,
        linear_config=FBI.FreeBoundaryTangentConfig(
            rtol=1.0e-11,
            restart=size,
            max_restarts=5,
            adjoint_backend="forward_dense",
            adjoint_batch_size=4,
        ),
    )
    _register_implicit_config(config)
    monkeypatch.setattr(
        FBI,
        "solve_free_boundary",
        lambda *_args, **_kwargs: pytest.fail(
            "custom-VJP anchor hit or backward launched a nonlinear solve"
        ),
    )

    direct_slope = 0.375

    def objective(alpha):
        solved_state = FBI.solve_free_boundary_implicit(alpha, config)
        return (
            FBI._tree_dot(state_cotangent, solved_state)
            + direct_slope * alpha
        )

    value, derivative = jax.jit(jax.value_and_grad(objective))(
        jnp.asarray(0.0, dtype=config.dtype)
    )
    jax.block_until_ready((value, derivative))

    expected_adjoint = np.linalg.solve(state_matrix.T, state_cotangent_flat)
    expected_implicit = -expected_adjoint @ parameter_column
    assert float(value) == pytest.approx(0.0)
    np.testing.assert_allclose(
        float(derivative),
        expected_implicit + direct_slope,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    stats = FBI.free_boundary_implicit_stats(config)
    assert stats["forward_callbacks"] == 1
    assert stats["forward_host_solves"] == 0
    assert stats["forward_memo_hits"] == 1
    assert stats["backward_callbacks"] == 1
    assert stats["backward_linear_solves"] == 1
    assert stats["backward_failures"] == 0
    assert stats["last_backward"]["converged"] is True

    with pytest.raises(ValueError, match="alpha must be scalar"):
        FBI.solve_free_boundary_implicit(jnp.zeros((2,)), config)
    assert FBI.free_boundary_implicit_result(0.0, config) is config.anchor_result
    FBI.reset_free_boundary_implicit_stats(config, clear_memo=True)
    reset_stats = FBI.free_boundary_implicit_stats(config)
    assert reset_stats["memo_entries"] == 1
    assert reset_stats["forward_host_solves"] == 0
