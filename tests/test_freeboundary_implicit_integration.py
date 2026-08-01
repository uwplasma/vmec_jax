"""Numerical B2--B4 gates against branch-local free-boundary solves."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from vmex.core import freeboundary as FB  # noqa: E402
from vmex.core import freeboundary_implicit as FBI  # noqa: E402
from vmex.core import implicit as IM  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.mgrid import MgridField, read_mgrid  # noqa: E402

pytestmark = [
    pytest.mark.full,
    pytest.mark.usefixtures("_module_jit_enabled"),
]

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy"
MGRID = REPO / "examples" / "data" / "mgrid_cth_like.nc"


@pytest.fixture(scope="module")
def converged_residual_case():
    """Return one low-resolution finite-beta free-boundary fixed point."""
    if not MGRID.exists():
        pytest.skip("real mgrid_cth_like.nc unavailable")

    inp = VmecInput.from_file(DECK)
    data = read_mgrid(MGRID)
    field = MgridField.from_mgrid_data(
        data,
        extcur=np.asarray(inp.extcur, dtype=float)[: data.nextcur],
    )
    solution = FB.solve_free_boundary(
        inp,
        external_field=field,
        ftol=1.0e-14,
        max_iterations=2500,
        error_on_no_convergence=False,
        preserve_m1_constraint_slice=True,
        include_edge_in_convergence=True,
        edge_force_tolerance=1.0e-14,
    )
    assert solution.converged
    assert solution.preserve_m1_constraint_slice is True
    assert solution.fedge <= solution.edge_force_tolerance

    evaluator = FB.make_free_boundary_residual_evaluator(inp)
    residual = evaluator(solution.state, field)
    return inp, field, solution, evaluator, residual


def _flat_residual(evaluator, state, field):
    return ravel_pytree(evaluator(state, field).residual)[0]


def _relative_l2(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return np.linalg.norm(actual - expected) / max(
        np.linalg.norm(expected),
        1.0e-30,
    )


def _traceable_metrics(state, runtime):
    axis_radius = jnp.mean(FB._vacuum_scalars(state, runtime)[2])
    return jnp.stack(
        [
            axis_radius,
            IM.plasma_volume(state, runtime),
            IM.iota_edge(state, runtime),
        ]
    )


def test_state_directional_jvp_matches_central_fd(converged_residual_case):
    """B2: the coupled residual state JVP agrees with central FD."""
    _inp, field, solution, evaluator, _evaluated = converged_residual_case
    rng = np.random.default_rng(2)
    direction = jax.tree.map(
        lambda a: jnp.asarray(rng.standard_normal(a.shape), dtype=a.dtype),
        solution.state,
    )
    direction_norm = np.sqrt(
        sum(float(jnp.vdot(a, a)) for a in jax.tree.leaves(direction))
    )
    direction = jax.tree.map(lambda a: a / direction_norm, direction)

    def fun(state):
        return _flat_residual(evaluator, state, field)

    _value, tangent = jax.jvp(fun, (solution.state,), (direction,))

    h = 1.0e-5
    plus = jax.tree.map(lambda x, v: x + h * v, solution.state, direction)
    minus = jax.tree.map(lambda x, v: x - h * v, solution.state, direction)
    finite_difference = (fun(plus) - fun(minus)) / (2.0 * h)

    relative_error = _relative_l2(finite_difference, tangent)
    assert relative_error < 5.0e-4, relative_error


def test_one_current_directional_jvp_matches_central_fd(
    converged_residual_case,
):
    """B2: a coupled residual current JVP agrees with central FD."""
    _inp, field, solution, evaluator, _evaluated = converged_residual_case

    def fun(fractional_change):
        varied = replace(
            field,
            extcur=field.extcur.at[0].set(
                field.extcur[0] * (1.0 + fractional_change)
            ),
        )
        return _flat_residual(evaluator, solution.state, varied)

    zero = jnp.asarray(0.0, dtype=field.extcur.dtype)
    _value, tangent = jax.jvp(fun, (zero,), (jnp.ones_like(zero),))

    h = jnp.asarray(1.0e-4, dtype=zero.dtype)
    finite_difference = (fun(h) - fun(-h)) / (2.0 * h)
    relative_error = _relative_l2(finite_difference, tangent)
    assert relative_error < 1.0e-8, relative_error


def test_free_boundary_projector_includes_edge_and_excludes_gauges(
    converged_residual_case,
):
    """The B3 state space is square, edge-evolving, and gauge-free."""
    _inp, _field, solution, evaluator, _evaluated = converged_residual_case
    mask = FBI.free_boundary_dof_mask(evaluator)
    modes = evaluator.runtime.modes
    m = np.asarray(modes.m)
    n = np.asarray(modes.n)
    m0n0 = (m == 0) & (n == 0)

    assert np.all(np.asarray(mask.R_cos)[-1] == 1.0)
    assert np.all(np.asarray(mask.Z_sin)[-1, ~m0n0] == 1.0)
    assert np.all(np.asarray(mask.L_sin)[-1, ~m0n0] == 1.0)
    assert np.all(np.asarray(mask.R_cos)[0, m > 0] == 0.0)
    assert np.all(np.asarray(mask.Z_sin)[:, m0n0] == 0.0)
    assert np.all(np.asarray(mask.L_sin)[0] == 0.0)
    for name in ("R_sin", "Z_cos", "L_cos"):
        assert np.count_nonzero(np.asarray(getattr(mask, name))) == 0

    _F, _z_star, projector = FBI.make_projected_free_boundary_residual(evaluator, solution.state, mask)
    projected = projector(solution.state)
    projected_twice = projector(projected)
    for once, twice in zip(jax.tree.leaves(projected), jax.tree.leaves(projected_twice)):
        np.testing.assert_array_equal(np.asarray(once), np.asarray(twice))

    index = {(int(mm), int(nn)): k for k, (mm, nn) in enumerate(zip(m, n))}
    for toroidal_mode in range(1, int(evaluator.runtime.resolution.ntor) + 1):
        pos = index[(1, toroidal_mode)]
        neg = index[(1, -toroidal_mode)]
        np.testing.assert_array_equal(
            np.asarray(projected.Z_sin[:, pos]),
            np.asarray(projected.Z_sin[:, neg]),
        )

    bad_mask = replace(mask, R_cos=mask.R_cos.at[-1, 0].set(0.5))
    with pytest.raises(ValueError, match="binary"):
        FBI.make_projected_free_boundary_residual(evaluator, solution.state, bad_mask)


@pytest.fixture(scope="module")
def equilibrium_tangent_oracle(
    converged_residual_case,
):
    """Cache one B3 tangent and two tight, same-chart hot re-solves."""
    inp, field, solution, evaluator, _evaluated = converged_residual_case
    mask = FBI.free_boundary_dof_mask(evaluator)

    def field_from_fraction(fractional_change):
        return replace(
            field,
            extcur=field.extcur.at[0].set(field.extcur[0] * (1.0 + fractional_change)),
        )

    config = FBI.FreeBoundaryTangentConfig(
        rtol=1.0e-8,
        restart=30,
        max_restarts=50,
        base_residual_atol=2.0e-8,
    )
    tangent = FBI.scalar_parameter_tangent(
        evaluator,
        solution.state,
        field_from_fraction,
        dof_mask=mask,
        config=config,
    )
    base_metrics, metric_tangent = jax.jvp(
        lambda state: _traceable_metrics(state, evaluator.runtime),
        (solution.state,),
        (tangent.state_tangent,),
    )

    h = 1.0e-3
    projected_residual, _z_star, projector = (
        FBI.make_projected_free_boundary_residual(
            evaluator,
            solution.state,
            mask,
        )
    )
    modes = evaluator.runtime.modes
    mode_index = {
        (int(m), int(n)): index
        for index, (m, n) in enumerate(
            zip(np.asarray(modes.m), np.asarray(modes.n))
        )
    }
    branch_metrics = {}
    for sign in (+1.0, -1.0):
        varied_field = field_from_fraction(sign * h)
        solved = FB.solve_free_boundary(
            inp,
            external_field=varied_field,
            ftol=1.0e-14,
            max_iterations=2500,
            error_on_no_convergence=False,
            initial_state=solution.state,
            preserve_m1_constraint_slice=True,
            include_edge_in_convergence=True,
            edge_force_tolerance=1.0e-14,
        )
        assert solved.converged
        assert solved.preserve_m1_constraint_slice is True
        assert solved.fedge <= solved.edge_force_tolerance
        assert max(float(solved.fsqr), float(solved.fsqz), float(solved.fsql)) < 1.1e-14

        # A hot start rebinds rcon0/zcon0 before the active vacuum lane damps
        # them toward zero. Tight host gates plus a fresh projected-residual
        # gate ensure this endpoint solves the asymptotic equation B3 uses.
        delta = jax.tree.map(
            lambda branch, anchor: branch - anchor,
            solved.state,
            solution.state,
        )
        slice_defect = jax.tree.map(
            lambda actual, projected: actual - projected,
            delta,
            projector(delta),
        )
        assert float(FBI._tree_norm(slice_defect)) < 1.0e-10
        for toroidal_mode in range(
            1,
            int(evaluator.runtime.resolution.ntor) + 1,
        ):
            pos = mode_index[(1, toroidal_mode)]
            neg = mode_index[(1, -toroidal_mode)]
            np.testing.assert_allclose(
                np.asarray(
                    solved.state.Z_sin[:, neg]
                    - solved.state.Z_sin[:, pos]
                ),
                np.asarray(
                    solution.state.Z_sin[:, neg]
                    - solution.state.Z_sin[:, pos]
                ),
                rtol=0.0,
                atol=1.0e-14,
            )
        branch_residual = projected_residual(
            projector(solved.state),
            varied_field,
        )
        assert float(FBI._tree_norm(branch_residual)) < config.base_residual_atol
        branch_metrics[sign] = np.asarray(
            _traceable_metrics(solved.state, evaluator.runtime)
        )

    finite_difference = (branch_metrics[+1.0] - branch_metrics[-1.0]) / (2.0 * h)
    return {
        "field": field,
        "solution": solution,
        "evaluator": evaluator,
        "mask": mask,
        "field_from_fraction": field_from_fraction,
        "config": config,
        "tangent": tangent,
        "base_metrics": base_metrics,
        "metric_tangent": metric_tangent,
        "finite_difference": finite_difference,
    }


def test_one_current_equilibrium_tangent_matches_branch_fd(
    equilibrium_tangent_oracle,
):
    """B3: the tangent metrics match full branch-local re-solves."""
    oracle = equilibrium_tangent_oracle
    tangent = oracle["tangent"]
    config = oracle["config"]
    solution = oracle["solution"]
    evaluator = oracle["evaluator"]
    mask = oracle["mask"]
    field_from_fraction = oracle["field_from_fraction"]
    metric_tangent = oracle["metric_tangent"]
    finite_difference = oracle["finite_difference"]

    assert bool(tangent.converged)
    assert bool(tangent.krylov_converged)
    assert int(tangent.iterations) <= config.restart * config.max_restarts
    assert float(tangent.base_residual_norm) < config.base_residual_atol
    assert float(tangent.relative_linear_residual) <= 1.01 * config.rtol
    np.testing.assert_allclose(
        np.asarray(metric_tangent),
        finite_difference,
        rtol=5.0e-3,
        atol=2.0e-5,
    )

    # A visibly perturbed state is not an equilibrium root and must be
    # rejected before any Krylov solve is attempted.
    nonroot = replace(
        solution.state,
        R_cos=solution.state.R_cos.at[-1, 0].add(1.0e-3),
    )
    with pytest.raises(ValueError, match="converged residual root"):
        FBI.scalar_parameter_tangent(
            evaluator,
            nonroot,
            field_from_fraction,
            dof_mask=mask,
            config=config,
        )


def test_volume_adjoint_matches_tangent_and_shared_branch_fd(
    equilibrium_tangent_oracle,
):
    """B4: one scalar adjoint matches B3 and the shared full-solve oracle."""
    oracle = equilibrium_tangent_oracle
    solution = oracle["solution"]
    evaluator = oracle["evaluator"]
    mask = oracle["mask"]
    field_from_fraction = oracle["field_from_fraction"]
    config = oracle["config"]
    base_metrics = oracle["base_metrics"]
    metric_tangent = oracle["metric_tangent"]
    finite_difference = oracle["finite_difference"]

    volume_objective = lambda state: IM.plasma_volume(  # noqa: E731
        state, evaluator.runtime
    )
    adjoint = FBI.scalar_state_objective_adjoint(
        evaluator,
        solution.state,
        field_from_fraction,
        volume_objective,
        dof_mask=mask,
        config=config,
    )
    pullback = adjoint.state_pullback
    assert pullback.backend == "forward_dense"
    assert bool(pullback.converged)
    assert bool(pullback.linear_solver_converged)
    assert int(pullback.iterations) <= config.restart * config.max_restarts
    assert int(pullback.active_dimension) == 1647
    assert float(pullback.base_residual_norm) < config.base_residual_atol
    assert float(pullback.relative_adjoint_residual) <= 1.01 * config.rtol
    assert np.isfinite(float(adjoint.derivative))
    assert abs(float(adjoint.derivative)) > 1.0e-8
    np.testing.assert_allclose(
        np.asarray(adjoint.objective_value),
        np.asarray(base_metrics[1]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(adjoint.derivative),
        np.asarray(metric_tangent[1]),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(adjoint.derivative),
        finite_difference[1],
        rtol=5.0e-3,
        atol=2.0e-5,
    )
