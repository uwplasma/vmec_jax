"""High/low transfer and stored raw-block preconditioner tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import implicit
from vmex.core import solver
from vmex.core.errors import (
    StrongForceCertificationError,
    StrongForceContinuationError,
    StrongForceLinearSolveError,
)
from vmex.core.input import VmecInput
from vmex.core.omnigenity import boozer_spectrum_high_order
from vmex.core.polish import (
    HighOrderCorrection,
    PreconditionerRefreshPolicy,
    PreconditionerSnapshot,
    apply_high_order_correction,
    build_low_order_preconditioner,
    build_strong_physical_block_preconditioner,
    build_strong_mode_block_preconditioner,
    make_high_low_transfer,
    make_strong_physical_chart,
    make_strong_structured_chart,
    make_strong_root_layout,
    make_strong_root_runtime,
    preconditioner_quality,
    preconditioner_refresh_decision,
    sample_high_order_state,
    _strong_residual_unscaled,
    _streaming_ruiz_scales,
    _physical_coordinate_blocks,
    _physical_equation_chart,
    strong_collocation_residual,
    strong_projection_diagnostics,
    strong_physical_residual,
    strong_root_rank,
    strong_root_residual,
    strong_root_residual_at_native,
)
from vmex.core.polish_driver import (
    PolishConfig,
    PolishContext,
    _IdentityPreconditioner,
    _arclength_to_target,
    _bordered_preconditioner,
    _branch_tangent,
    _build_mode_block_preconditioner,
    _continuation_precondition,
    _low_inverse,
    _normalized_low_residual_norm,
    _solve_low_inverse,
    _ptc_config,
    _residual_evaluations,
    _supports_keyword,
    polish_collocation_least_squares,
    polish_strong_root,
    polished_wout_ns,
)
from vmex.core.polish_implicit import (
    PolishLinearConfig,
    PolishLinearReport,
    _checked_solution,
    _linear_report,
    _solve_linear,
    _collocation_stationarity,
    _tree_norm as _implicit_tree_norm,
    collocation_polish_adjoint,
    collocation_polish_tangent,
    implicit_collocation_polished_state,
)
from vmex.core.strong_force import lift_high_order_state

jax.config.update("jax_enable_x64", True)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"rtol": 0.0}, "rtol"),
        ({"atol": -1.0}, "atol"),
        ({"restart": 0}, "restart"),
        ({"max_restarts": 0}, "max_restarts"),
        ({"fail_policy": "ignore"}, "fail_policy"),
    ],
)
def test_polish_linear_config_validation(updates, message):
    with pytest.raises(ValueError, match=message):
        PolishLinearConfig(**updates)


def test_polish_linear_failure_policy_is_explicit():
    value = jnp.ones((2,))
    report = PolishLinearReport(
        residual_norm=jnp.asarray(2.0),
        tolerance=jnp.asarray(1.0),
        iterations=jnp.asarray(3),
        converged=jnp.asarray(False),
    )
    with pytest.raises(StrongForceLinearSolveError, match="did not converge"):
        _checked_solution(value, report, PolishLinearConfig(), "test")
    result = _checked_solution(
        value,
        report,
        PolishLinearConfig(fail_policy="nan"),
        "test",
    )
    assert np.isnan(result).all()
    assert _implicit_tree_norm((jnp.asarray([3.0, 4.0]),)) == pytest.approx(5.0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("rhs", "value", "applied", "flag", "accepted"),
    [
        (1.0, 0.0, 0.0, True, False),
        (1.0, 0.0, 0.0, False, False),
        (1.0, 1.0, 1.0, False, True),
        (1.0, np.nan, 1.0, True, False),
        (1.0, np.inf, 1.0, True, False),
        (np.nan, 1.0, 1.0, True, False),
        (np.inf, 1.0, 1.0, True, False),
        (1.0, 1.0, np.nan, True, False),
        (1.0, 1.0, np.inf, True, False),
        (1.0e200, 1.0e200, 1.0e200, True, False),
        (1.0, 1.0e200, 1.0e200, True, False),
        (0.0, 0.0, 0.0, False, True),
        (0.0, 1.0e-12, 1.0e-12, False, True),
        (0.0, 1.0e-9, 1.0e-9, True, False),
    ],
)
def test_polish_linear_true_certificate(compiled, rhs, value, applied, flag, accepted):
    def report_for(rhs, value, applied):
        return _linear_report(
            lambda _: applied,
            rhs,
            SimpleNamespace(x=value, converged=flag, iterations=jnp.asarray(30)),
            PolishLinearConfig(),
        )

    with jax.disable_jit(False):
        report = (jax.jit(report_for) if compiled else report_for)(
            *[jnp.asarray([item, item]) for item in (rhs, value, applied)]
        )
    assert bool(report.converged) == accepted
    assert int(report.iterations) == 30


@pytest.mark.parametrize("policy", ["raise", "nan"])
def test_polish_linear_compiled_failure_returns_nan_and_status(policy):
    config = PolishLinearConfig(fail_policy=policy)

    def checked(value):
        report = _linear_report(
            lambda x: x, jnp.ones(1),
            SimpleNamespace(x=value, converged=True, iterations=jnp.asarray(1)),
            config,
        )
        return _checked_solution(value, report, config, "tangent"), report

    with jax.disable_jit(False):
        value, report = jax.jit(checked)(jnp.zeros(1))
    assert not bool(report.converged)
    assert bool(jnp.all(jnp.isnan(value)))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_polish_linear_krylov_true_residual(transpose, compiled):
    matrix = jnp.asarray([[4.0, 1.0], [-2.0, 3.0]])
    if transpose:
        matrix = matrix.T
    rhs = jnp.asarray([2.0, -1.0])

    def solve(rhs):
        return _solve_linear(
            lambda x: matrix @ x, rhs, lambda x: x / 4.0,
            PolishLinearConfig(), "adjoint" if transpose else "tangent",
        )

    with jax.disable_jit(False):
        value, report = (jax.jit(solve) if compiled else solve)(rhs)
    assert bool(report.converged)
    np.testing.assert_allclose(value, np.linalg.solve(matrix, rhs), atol=1.0e-11)
    assert float(jnp.linalg.norm(rhs - matrix @ value)) <= float(report.tolerance)


@pytest.mark.parametrize("compiled", [False, True])
def test_polish_linear_iteration_exhaustion(compiled):
    matrix = jnp.asarray([[4.0, 1.0], [-2.0, 3.0]])
    rhs = jnp.asarray([2.0, -1.0])

    def solve(rhs):
        return _solve_linear(
            lambda x: matrix @ x, rhs, lambda x: x,
            PolishLinearConfig(restart=1, max_restarts=1, fail_policy="nan"),
            "tangent",
        )

    with jax.disable_jit(False):
        value, report = (jax.jit(solve) if compiled else solve)(rhs)
    assert not bool(report.converged)
    assert float(report.residual_norm) > float(report.tolerance)
    assert bool(jnp.all(jnp.isnan(value)))


def test_collocation_certification_error_retains_both_failure_gates():
    error = StrongForceCertificationError(
        "not certified",
        solver_converged=True,
        normalized_l2=0.2,
        tolerance=0.1,
        radial_refinement=0.03,
        radial_refinement_tolerance=0.01,
    )

    assert error.solver_converged
    assert error.normalized_l2 > error.tolerance
    assert error.radial_refinement > error.radial_refinement_tolerance


def test_solvax_continuation_api_compatibility_helpers():
    def legacy_preconditioner(state, rhs, dtau):
        del state, dtau
        return rhs

    def parameterized_preconditioner(state, rhs, dtau, parameter):
        del state, dtau, parameter
        return rhs

    assert not _supports_keyword(legacy_preconditioner, "parameter")
    assert _supports_keyword(parameterized_preconditioner, "parameter")
    np.testing.assert_array_equal(
        legacy_preconditioner(None, jnp.ones((2,)), None), jnp.ones((2,))
    )
    np.testing.assert_array_equal(
        parameterized_preconditioner(None, jnp.ones((2,)), None, None),
        jnp.ones((2,)),
    )
    assert _residual_evaluations(
        SimpleNamespace(nonlinear_steps=3, residual_evaluations=9)
    ) == 9
    assert _residual_evaluations(SimpleNamespace(nonlinear_steps=3)) == 4
    assert _residual_evaluations(SimpleNamespace(steps=2)) == 3
    assert not _supports_keyword(1, "parameter")


def test_parameterized_continuation_preconditioner_switches_at_half(monkeypatch):
    rhs = jnp.asarray([1.0, -2.0])
    monkeypatch.setattr(
        "vmex.core.polish_driver._low_inverse", lambda value, runtime: 2.0 * value
    )
    block = SimpleNamespace(apply=lambda value, alpha, dtau: 3.0 * value)
    np.testing.assert_array_equal(
        _continuation_precondition(rhs, 0.25, 1.0, SimpleNamespace(), block),
        2.0 * rhs,
    )
    np.testing.assert_array_equal(
        _continuation_precondition(rhs, 0.75, 1.0, SimpleNamespace(), block),
        3.0 * rhs,
    )
    np.testing.assert_array_equal(
        _continuation_precondition(
            rhs, 0.25, 1.0, SimpleNamespace(), block, SimpleNamespace()
        ),
        3.0 * rhs,
    )
    identity = _IdentityPreconditioner()
    np.testing.assert_array_equal(identity.apply(rhs), rhs)
    np.testing.assert_array_equal(
        _continuation_precondition(
            rhs,
            0.25,
            1.0,
            SimpleNamespace(),
            identity,
        ),
        rhs,
    )


def test_arclength_crossing_runs_target_correction_and_counts_work(monkeypatch):
    zero = jnp.zeros((2,))
    target_vector = jnp.asarray([0.25, -0.5])

    def corrector(
        residual,
        initial,
        *,
        tangent,
        predictor,
        config,
        admissible,
        parameterized_precond,
    ):
        del residual, initial, config
        assert bool(admissible(*predictor))
        np.testing.assert_array_equal(
            parameterized_precond(predictor, predictor, 1.0, tangent, predictor)[0],
            predictor[0],
        )
        return SimpleNamespace(
            x=predictor,
            steps=2,
            linear_iterations=3,
            residual_evaluations=4,
            converged=True,
            linear_converged=True,
        )

    def target(residual, initial, *, precond, admissible, config):
        del residual, initial, config
        assert bool(admissible(target_vector))
        np.testing.assert_array_equal(precond(target_vector, target_vector, 1.0), target_vector)
        return SimpleNamespace(
            x=target_vector,
            steps=5,
            linear_iterations=6,
            residual_evaluations=7,
            converged=True,
            linear_converged=True,
        )

    monkeypatch.setattr(
        "vmex.core.polish_driver._solvax_continuation_api",
        lambda: (None, None, None, corrector, target),
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._ptc_config", lambda config, **kwargs: object()
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._branch_tangent",
        lambda *args, **kwargs: (jnp.zeros_like(zero), jnp.asarray(1.0)),
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._apply_bordered_preconditioner",
        lambda state, rhs, dtau, tangent, runtime, block, chart=None: rhs,
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._low_inverse", lambda rhs, runtime: rhs
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver.strong_root_residual",
        lambda vector, runtime, alpha: vector + alpha,
    )
    result = _arclength_to_target(
        zero,
        0.95,
        SimpleNamespace(layout=SimpleNamespace(size=2), operator_balance=1.0),
        PolishConfig(max_arclength_steps=1, arclength_step=0.1),
        lambda vector, alpha: jnp.all(jnp.isfinite(vector)) & jnp.isfinite(alpha),
        None,
        None,
    )
    np.testing.assert_array_equal(result[0], target_vector)
    assert result[1:] == (1.0, 1, 7, 9, 11)


def test_bordered_tangent_uses_previous_orientation(monkeypatch):
    zero = jnp.zeros((2,))
    previous = (jnp.asarray([-1.0, -1.0]), jnp.asarray(-1.0))

    def fake_gmres(operator, rhs, *, precond, **kwargs):
        del kwargs
        physical, normalization = operator(rhs)
        assert physical.shape == zero.shape
        assert np.isfinite(float(normalization))
        for actual, expected in zip(
            jax.tree.leaves(precond(rhs)), jax.tree.leaves(rhs), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        return SimpleNamespace(
            x=(jnp.asarray([0.5, 0.25]), jnp.asarray(0.5)),
            converged=True,
            residual_norm=jnp.asarray(0.0),
            iterations=1,
        )

    monkeypatch.setattr("vmex.core.polish_driver.gmres", fake_gmres)
    monkeypatch.setattr(
        "vmex.core.polish_driver._bordered_preconditioner",
        lambda *args, **kwargs: lambda state, rhs, dtau: rhs,
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver.strong_root_residual",
        lambda vector, runtime, alpha: vector + alpha * jnp.ones_like(vector),
    )
    tangent = _branch_tangent(
        zero,
        0.5,
        SimpleNamespace(),
        PolishConfig(),
        previous,
        None,
    )
    np.testing.assert_allclose(
        jnp.vdot(tangent[0], tangent[0]).real + tangent[1] ** 2,
        1.0,
        rtol=2.0e-13,
    )
    assert float(jnp.vdot(tangent[0], previous[0]) + tangent[1] * previous[1]) > 0.0


def _tree_dot(left, right):
    return sum(
        jnp.vdot(a, b).real
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _tree_norm(value) -> float:
    return float(jnp.sqrt(_tree_dot(value, value)))


def _matrix_residual(matrix):
    """Module-level builder: the lane's static argument must be identity-stable."""

    def residual(vector):
        return matrix @ vector

    return residual


def test_streaming_equilibration_improves_conditioning_without_dropping_dofs():
    matrix = jnp.asarray([[1.0e-8, 0.0], [0.0, 2.0]])
    rows, columns = _streaming_ruiz_scales(
        _matrix_residual,
        matrix,
        jnp.zeros((2,)),
    )
    balanced = rows[:, None] * np.asarray(matrix) * columns[None, :]
    assert np.all(rows > 0.0)
    assert np.all(columns > 0.0)
    assert np.linalg.matrix_rank(balanced) == 2
    assert np.linalg.cond(balanced) < 1.01


def test_streaming_equilibration_is_deterministic_and_validates_controls():
    matrix = jnp.asarray([[2.0, -1.0], [3.0, 4.0]])

    first = _streaming_ruiz_scales(_matrix_residual, matrix, jnp.zeros((2,)), probes=2)
    second = _streaming_ruiz_scales(_matrix_residual, matrix, jnp.zeros((2,)), probes=2)
    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert np.all(actual > 0.0)
    with pytest.raises(ValueError, match="iterations"):
        _streaming_ruiz_scales(_matrix_residual, matrix, jnp.zeros((2,)), iterations=0)
    with pytest.raises(ValueError, match="probes"):
        _streaming_ruiz_scales(_matrix_residual, matrix, jnp.zeros((2,)), probes=0)


def _random_like(value, seed: int):
    leaves, structure = jax.tree.flatten(value)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree.unflatten(
        structure,
        [jax.random.normal(key, leaf.shape, leaf.dtype) for key, leaf in zip(keys, leaves)],
    )


def _small_solovev_input():
    inp = VmecInput.from_file(DATA / "input.solovev").change_resolution(
        mpol=3,
        ntor=0,
        ntheta=12,
        nzeta=4,
    )
    return dataclasses.replace(
        inp,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )


@pytest.fixture(scope="module")
def small_adapter():
    inp = _small_solovev_input()
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


@pytest.fixture(scope="module")
def small_strong_root(small_adapter):
    native, _, _, mask, adapter = small_adapter
    return make_strong_root_runtime(native, adapter, mask)


def test_transfer_preserves_constraints_and_roundtrips_range(small_adapter):
    native, _, _, _, adapter = small_adapter
    transfer = adapter.transfer
    high = _random_like(transfer.zeros_high(jnp.float64), 1)
    projected = transfer.project_high(high)
    low = jax.jit(transfer.restrict)(high)
    roundtrip = transfer.restrict(transfer.prolong(low))

    assert native.radial_basis.size < transfer.ns
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


def test_three_dimensional_m1_projector_transposes_without_scatter_failure():
    inp = VmecInput.from_file(DATA / "input.solovev").change_resolution(
        mpol=3,
        ntor=1,
        ntheta=12,
        nzeta=4,
    )
    inp = dataclasses.replace(inp, ns_array=np.asarray([5]))
    config = implicit.make_config(inp, ftol=1.0e-8, max_iterations=1)
    params = implicit.params_from_input(inp)
    runtime = implicit.runtime_from_params(params, config)
    state = solver._initial_state(runtime.setup)
    one = jnp.ones_like(state.R_cos)
    zero = jnp.zeros_like(one)
    edge_free = one.at[-1].set(0.0)
    lambda_free = one.at[0].set(0.0)
    mask = solver.SpectralState(
        R_cos=edge_free,
        R_sin=zero,
        Z_cos=zero,
        Z_sin=edge_free,
        L_cos=zero,
        L_sin=lambda_free,
    )
    native = lift_high_order_state(state, runtime, degree=3)
    transfer = make_high_low_transfer(
        native,
        runtime,
        project_config=config,
        project_mask=mask,
    )
    high = _random_like(transfer.zeros_high(jnp.float64), 31)
    low_bar = _random_like(transfer.restrict(high), 32)
    lhs = _tree_dot(transfer.restrict(high), low_bar)
    rhs = _tree_dot(high, transfer.restrict_transpose(low_bar))
    np.testing.assert_allclose(lhs, rhs, rtol=3.0e-13, atol=3.0e-13)

    layout = make_strong_root_layout(
        mask, native, transfer=transfer, lconm1=True
    )
    # Every active constrained +/-n pair contributes one, not two, Z dofs.
    active_z = int(np.count_nonzero(np.asarray(mask.Z_sin)))
    active_l = int(np.count_nonzero(np.asarray(mask.L_sin)))
    assert layout.size < (
        int(np.count_nonzero(np.asarray(mask.R_cos))) + active_z + active_l
    )
    vector = jax.random.normal(jax.random.PRNGKey(35), (layout.size,))
    tangent = layout.unpack(vector)
    np.testing.assert_allclose(layout.pack(tangent), vector, rtol=2.0e-15, atol=2.0e-15)

    low = _random_like(low_bar, 33)
    high_bar = _random_like(high, 34)
    lhs = _tree_dot(transfer.prolong(low), high_bar)
    rhs = _tree_dot(low, transfer.prolong_transpose(high_bar))
    np.testing.assert_allclose(lhs, rhs, rtol=3.0e-13, atol=3.0e-13)


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


def test_square_strong_root_endpoint_jvp_boundary_and_rank(small_strong_root):
    runtime = small_strong_root
    zero = jnp.zeros((runtime.layout.size,), dtype=jnp.float64)
    radial_matrix = runtime.native.radial_basis.basis_matrix(runtime.radial_nodes**2)
    assert runtime.radial_nodes.size > runtime.native.radial_basis.size
    assert runtime.theta.size >= 4 * int(np.max(np.abs(runtime.native.m))) + 5
    assert runtime.zeta.size == 1
    for mode, mode_m in enumerate(np.asarray(runtime.native.m)):
        regularized_matrix = (
            runtime.radial_nodes[:, None] ** abs(int(mode_m)) * radial_matrix
        )
        np.testing.assert_allclose(
            runtime.radial_fit[mode] @ regularized_matrix,
            np.eye(runtime.native.radial_basis.size),
            rtol=5.0e-10,
            atol=5.0e-10,
        )
    low_endpoint = strong_root_residual(zero, runtime, 0.0)
    strong_endpoint = strong_root_residual(zero, runtime, 1.0)
    # The alpha = 0 endpoint is legacy_residual(x0) - legacy_defect: two
    # evaluations of one nonlinear function in two separately compiled
    # programs.  XLA does not promise bit-identical fusion across programs or
    # platforms, so the cancellation bottoms out at round-off (measured
    # 2.8e-14 on the Linux CI runner, exact zero on arm64).  The bound is the
    # cancellation floor of the row-scaled O(1) residual, not a physics
    # tolerance.
    np.testing.assert_allclose(low_endpoint, 0.0, atol=1.0e-12)
    assert strong_endpoint.shape == zero.shape
    assert np.all(np.isfinite(np.asarray(strong_endpoint)))
    # The initial force RMS is divided by the measured low-inverse stiffness.
    np.testing.assert_allclose(
        jnp.linalg.norm(strong_endpoint),
        np.sqrt(runtime.layout.size) / runtime.operator_balance,
        rtol=3.0e-13,
    )
    assert float(runtime.operator_balance) >= 1.0
    assert runtime.coordinate_scale.shape == zero.shape
    assert runtime.equation_scale.shape == zero.shape
    assert np.all(np.asarray(runtime.coordinate_scale) > 0.0)
    assert np.all(np.asarray(runtime.equation_scale) > 0.0)
    assert runtime.strong_block_sign.shape == (3,)
    np.testing.assert_array_equal(jnp.abs(runtime.strong_block_sign), 1.0)

    probe = jnp.linspace(-0.01, 0.015, runtime.layout.size)
    low_probe = strong_root_residual(probe, runtime, 0.0)
    strong_probe = strong_root_residual(probe, runtime, 1.0)
    alpha = 0.37
    np.testing.assert_allclose(
        strong_root_residual(probe, runtime, alpha),
        low_probe + alpha * (strong_probe - low_probe),
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    direction = jnp.linspace(-0.2, 0.3, runtime.layout.size)
    _, tangent = jax.jvp(
        lambda value: strong_root_residual(value, runtime, 1.0),
        (zero,),
        (direction,),
    )
    step = 2.0e-5
    finite_difference = (
        strong_root_residual(step * direction, runtime, 1.0)
        - strong_root_residual(-step * direction, runtime, 1.0)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-6, atol=2.0e-7)

    correction = runtime.layout.unpack(0.01 * direction)
    corrected = apply_high_order_correction(runtime.native, correction)
    for name in ("R_cos", "R_sin", "Z_cos", "Z_sin"):
        np.testing.assert_array_equal(
            np.asarray(getattr(corrected, name)[:, -1]),
            np.asarray(getattr(runtime.native, name)[:, -1]),
        )
    assert corrected.source.endswith("strong-root correction")

    rank, singular_values = strong_root_rank(runtime, relative_tolerance=1.0e-8)
    assert rank == runtime.layout.size
    assert float(singular_values[-1]) > 0.0


def test_physical_chart_eliminates_only_the_linear_coordinate_gauge(
    small_strong_root,
):
    runtime = small_strong_root
    chart = make_strong_physical_chart(runtime)
    assert chart.full_size == runtime.layout.size
    assert chart.size + chart.gauge_rank == chart.full_size
    assert chart.gauge_rank > 0
    assert chart.build_seconds > 0.0
    np.testing.assert_allclose(
        np.asarray(chart.coordinate_basis.T @ chart.coordinate_basis),
        np.eye(chart.size),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(chart.equation_basis.T @ chart.equation_basis),
        np.eye(chart.size),
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    zero = jnp.zeros((chart.size,), dtype=jnp.float64)
    # Same two-program cancellation floor as the endpoint test above.
    np.testing.assert_allclose(
        strong_physical_residual(zero, runtime, chart, 0.0), 0.0, atol=1.0e-12
    )
    probe = jnp.linspace(-0.01, 0.015, chart.size)
    full_probe = chart.lift(probe)
    low_probe = chart.project(strong_root_residual(full_probe, runtime, 0.0))
    strong_probe = chart.project(
        _strong_residual_unscaled(
            full_probe,
            runtime,
            include_coordinate_gauge=False,
        )
        / runtime.strong_scale
    )
    alpha = 0.37
    np.testing.assert_allclose(
        strong_physical_residual(probe, runtime, chart, alpha),
        low_probe + alpha * (strong_probe - low_probe),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    direction = jnp.linspace(-0.2, 0.3, chart.size)
    _, tangent = jax.jvp(
        lambda value: strong_physical_residual(value, runtime, chart, 1.0),
        (zero,),
        (direction,),
    )
    step = 2.0e-5
    finite_difference = (
        strong_physical_residual(step * direction, runtime, chart, 1.0)
        - strong_physical_residual(-step * direction, runtime, chart, 1.0)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-6, atol=2.0e-7)
    jacobian = jax.jacfwd(
        lambda value: strong_physical_residual(value, runtime, chart, 1.0)
    )(zero)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    rank = int(jnp.sum(singular_values > 1.0e-8 * singular_values[0]))
    assert rank == chart.size

    with pytest.raises(ValueError, match="relative_tolerance"):
        make_strong_physical_chart(runtime, relative_tolerance=0.0)
    with pytest.raises(ValueError, match="radial_quadrature_order"):
        make_strong_root_runtime(
            runtime.native,
            runtime.low_preconditioner,
            runtime.transfer.zeros_low(),
            radial_quadrature_order=1,
        )
    with pytest.raises(ValueError, match="physical vector"):
        chart.lift(jnp.zeros((chart.size + 1,)))
    with pytest.raises(ValueError, match="full residual"):
        chart.project(jnp.zeros((chart.full_size + 1,)))


def test_structured_chart_uses_only_physical_layout_channels(small_strong_root):
    runtime = small_strong_root
    chart = make_strong_structured_chart(runtime)
    assert chart.full_size == runtime.layout.size
    assert chart.size + chart.gauge_rank == chart.full_size
    assert chart.gauge_rank > 0
    np.testing.assert_allclose(
        np.asarray(chart.coordinate_basis),
        np.asarray(chart.equation_basis),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(chart.coordinate_basis.T @ chart.coordinate_basis),
        np.eye(chart.size),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    zero = jnp.zeros((chart.size,), dtype=jnp.float64)
    jacobian = jax.jacfwd(
        lambda value: strong_physical_residual(value, runtime, chart, 1.0)
    )(zero)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    rank = int(jnp.sum(singular_values > 1.0e-8 * singular_values[0]))
    assert rank == chart.size

    diagnostics = strong_projection_diagnostics(zero, runtime, chart)
    collocation = strong_collocation_residual(zero, runtime, chart)
    values = np.asarray(tuple(diagnostics))
    assert np.all(np.isfinite(values))
    assert diagnostics.sampled_rms > 0.0
    assert diagnostics.unresolved_rms >= 0.0
    assert diagnostics.unresolved_fraction >= 0.0
    assert diagnostics.angular_unresolved_fraction >= 0.0
    assert diagnostics.radial_fit_unresolved_fraction >= 0.0
    assert diagnostics.radial_unresolved_fraction >= 0.0
    assert diagnostics.helical_unresolved_fraction >= 0.0
    assert diagnostics.equation_discarded_fraction < 1.0e-12
    point_count = (
        runtime.radial_nodes.size * runtime.theta.size * runtime.zeta.size
    )
    assert collocation.shape == (2 * point_count,)
    np.testing.assert_allclose(
        jnp.linalg.norm(collocation) / np.sqrt(float(point_count)),
        diagnostics.sampled_rms,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_structured_chart_mode_blocks_recover_local_jacobian(small_strong_root):
    runtime = small_strong_root
    chart = make_strong_structured_chart(runtime)
    preconditioner = build_strong_physical_block_preconditioner(
        runtime,
        chart,
        poloidal_bandwidth=64,
    )
    zero = jnp.zeros((chart.size,), dtype=jnp.float64)
    direction = jnp.linspace(-0.15, 0.25, chart.size)
    _, response = jax.jvp(
        lambda value: strong_physical_residual(value, runtime, chart, 1.0),
        (zero,),
        (direction,),
    )
    np.testing.assert_allclose(
        preconditioner.apply(response, 1.0),
        direction,
        rtol=5.0e-8,
        atol=5.0e-8,
    )
    with pytest.raises(ValueError, match="poloidal_bandwidth"):
        build_strong_physical_block_preconditioner(
            runtime, chart, poloidal_bandwidth=0
        )
    with pytest.raises(ValueError, match="physical block linearization"):
        build_strong_physical_block_preconditioner(
            runtime,
            chart,
            jnp.zeros((chart.size + 1,)),
        )
    dense_chart = make_strong_physical_chart(runtime)
    with pytest.raises(ValueError, match="local structured chart"):
        build_strong_physical_block_preconditioner(runtime, dense_chart)


def test_strong_root_validation_branches(small_adapter, small_strong_root):
    native, _, _, mask, adapter = small_adapter
    layout = small_strong_root.layout
    with pytest.raises(ValueError, match="free vector"):
        layout.unpack(jnp.zeros((layout.size + 1,)))
    with pytest.raises(ValueError, match="force_floor"):
        make_strong_root_runtime(native, adapter, mask, force_floor=0.0)
    with pytest.raises(ValueError, match="balance_iterations"):
        make_strong_root_runtime(native, adapter, mask, balance_iterations=0)
    zero_mask = jax.tree.map(jnp.zeros_like, mask)
    with pytest.raises(ValueError, match="no free physical displacement"):
        make_strong_root_runtime(native, adapter, zero_mask)
    with pytest.raises(ValueError, match="poloidal_bandwidth"):
        build_strong_mode_block_preconditioner(
            small_strong_root, poloidal_bandwidth=0
        )
    with pytest.raises(ValueError, match="block linearization"):
        build_strong_mode_block_preconditioner(
            small_strong_root,
            jnp.zeros((small_strong_root.layout.size + 1,)),
        )
    mismatched = dataclasses.replace(mask, Z_sin=mask.Z_sin[:, :-1])
    with pytest.raises(ValueError, match="layout must match"):
        make_strong_root_layout(mismatched, native)
    with pytest.raises(ValueError, match="high/low transfer"):
        make_strong_root_layout(mask, native)
    with pytest.raises(ValueError, match="relative_tolerance"):
        strong_root_rank(small_strong_root, relative_tolerance=0.0)
    rank, values = strong_root_rank(
        small_strong_root,
        jnp.zeros((layout.size,)),
        relative_tolerance=1.0e-8,
    )
    assert rank == layout.size
    assert values.shape == (layout.size,)

    unbalanced = make_strong_root_runtime(
        native, adapter, mask, balance_full_root=False
    )
    assert unbalanced.layout.size == layout.size


def test_strong_runtime_and_chart_pytree_roundtrip(small_strong_root):
    """JIT reconstruction preserves the numeric runtime and physical chart."""

    chart = make_strong_structured_chart(small_strong_root)
    for original in (small_strong_root, chart):
        leaves, structure = jax.tree.flatten(original)
        rebuilt = jax.tree.unflatten(structure, leaves)
        assert type(rebuilt) is type(original)
        np.testing.assert_allclose(rebuilt.coordinate_scale, original.coordinate_scale)
    # The layout is a child pytree now (its basis arrays are traced leaves,
    # not baked metadata), so a round trip reconstructs an equivalent layout
    # rather than preserving object identity.
    rebuilt_layout = jax.tree.unflatten(
        jax.tree.structure(small_strong_root),
        jax.tree.leaves(small_strong_root),
    ).layout
    original_layout = small_strong_root.layout
    assert rebuilt_layout.mnmax == original_layout.mnmax
    assert rebuilt_layout.nbasis == original_layout.nbasis
    assert len(rebuilt_layout.groups) == len(original_layout.groups)
    for rebuilt_group, group in zip(rebuilt_layout.groups,
                                    original_layout.groups):
        np.testing.assert_array_equal(rebuilt_group.high_indices,
                                      group.high_indices)
        np.testing.assert_allclose(rebuilt_group.basis, group.basis)
        assert (rebuilt_group.start, rebuilt_group.stop) == (
            group.start, group.stop)
    # What compile reuse actually needs: two flattens of one runtime share a
    # treedef even though the layout and preconditioner are rebuilt objects.
    assert jax.tree.structure(small_strong_root) == jax.tree.structure(
        jax.tree.unflatten(
            jax.tree.structure(small_strong_root),
            jax.tree.leaves(small_strong_root)))
    assert jax.tree.unflatten(
        jax.tree.structure(chart), jax.tree.leaves(chart)
    ).gauge_rank == chart.gauge_rank


def test_physical_chart_adapters_and_validation(small_strong_root, monkeypatch):
    chart = make_strong_structured_chart(small_strong_root)
    rhs = jnp.linspace(-0.2, 0.3, chart.size)
    solved = _solve_low_inverse(rhs, small_strong_root, chart)
    assert solved.shape == rhs.shape
    residual = strong_root_residual_at_native(
        jnp.zeros((small_strong_root.layout.size,)),
        small_strong_root.native,
        small_strong_root,
    )
    assert residual.shape == (small_strong_root.layout.size,)
    sentinel = object()
    monkeypatch.setattr(
        "vmex.core.polish_driver.build_strong_physical_block_preconditioner",
        lambda runtime, physical_chart: sentinel,
    )
    assert _build_mode_block_preconditioner(small_strong_root, chart) is sentinel
    with pytest.raises(ValueError, match="poloidal_bandwidth"):
        _physical_coordinate_blocks(small_strong_root, chart, 0)
    with pytest.raises(ValueError, match="no physical force-output"):
        _physical_equation_chart(dataclasses.replace(small_strong_root.layout, groups=()))
    empty_chart = dataclasses.replace(
        chart, coordinate_basis=jnp.zeros_like(chart.coordinate_basis)
    )
    with pytest.raises(ValueError, match="local structured chart"):
        _physical_coordinate_blocks(small_strong_root, empty_chart, 1)
    first = small_strong_root.layout.groups[0]
    zero_group = dataclasses.replace(first, basis=jnp.zeros_like(first.basis))
    _physical_equation_chart(
        dataclasses.replace(
            small_strong_root.layout,
            groups=(zero_group, *small_strong_root.layout.groups[1:]),
        )
    )
    asymmetric = dataclasses.replace(
        small_strong_root,
        transfer=dataclasses.replace(small_strong_root.transfer, lasym=True),
    )
    with pytest.raises(ValueError, match="stellarator symmetry"):
        make_strong_structured_chart(asymmetric)


def test_implicit_polish_rejects_mismatched_inputs(small_strong_root):
    chart = make_strong_structured_chart(small_strong_root)
    good = PolishContext(
        small_strong_root,
        chart,
        jnp.zeros((chart.size,)),
        jnp.ones((chart.size,)),
    )
    bad = good._replace(correction=jnp.zeros((chart.size + 1,)))
    with pytest.raises(ValueError, match="correction has shape"):
        collocation_polish_tangent(bad, small_strong_root.native)
    with pytest.raises(ValueError, match="correction has shape"):
        collocation_polish_adjoint(bad, small_strong_root.native)
    with pytest.raises(ValueError, match="native_tangent"):
        collocation_polish_tangent(good, jnp.asarray(0.0))
    with pytest.raises(ValueError, match="polished_cotangent"):
        collocation_polish_adjoint(good, jnp.asarray(0.0))
    with pytest.raises(ValueError, match="native must have"):
        implicit_collocation_polished_state(jnp.asarray(0.0), good)


def test_low_vector_preconditioner_is_finite_on_native_coordinates(
    small_strong_root,
):
    runtime = small_strong_root
    zero = jnp.zeros((runtime.layout.size,), dtype=jnp.float64)
    direction = jnp.linspace(-0.1, 0.2, runtime.layout.size)
    _, response = jax.jvp(
        lambda value: strong_root_residual(value, runtime, 0.0),
        (zero,),
        (direction,),
    )
    recovered = _low_inverse(response, runtime)
    assert np.all(np.isfinite(np.asarray(recovered)))
    assert float(jnp.linalg.norm(recovered)) > 0.0
    assert float(jnp.linalg.norm(recovered)) < 10.0 * float(
        jnp.linalg.norm(direction)
    )


def test_scaled_low_inverse_and_transpose_are_exact_duals(small_strong_root):
    runtime = small_strong_root
    left = runtime.transfer.restrict(
        runtime.layout.unpack(jnp.linspace(-0.2, 0.1, runtime.layout.size))
    )
    right = runtime.transfer.restrict(
        runtime.layout.unpack(jnp.linspace(0.3, -0.15, runtime.layout.size))
    )
    forward = runtime.low_preconditioner.solve_scaled(left)
    transpose = runtime.low_preconditioner.solve_scaled_transpose(right)
    np.testing.assert_allclose(
        _tree_dot(forward, right),
        _tree_dot(left, transpose),
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_arclength_tangent_and_bordered_preconditioner_are_finite(
    small_strong_root,
):
    runtime = small_strong_root
    zero = jnp.zeros((runtime.layout.size,), dtype=jnp.float64)
    block_preconditioner = _build_mode_block_preconditioner(runtime)
    direction = jnp.linspace(-0.15, 0.25, runtime.layout.size)
    _, response = jax.jvp(
        lambda value: strong_root_residual(value, runtime, 1.0),
        (zero,),
        (direction,),
    )
    recovered = block_preconditioner.apply(response, 1.0)
    np.testing.assert_allclose(recovered, direction, rtol=3.0e-8, atol=3.0e-8)
    _, pullback = jax.vjp(
        lambda value: strong_root_residual(value, runtime, 1.0), zero
    )
    transpose_response = pullback(direction)[0]
    transpose_recovered = block_preconditioner.apply_transpose(
        transpose_response, 1.0
    )
    np.testing.assert_allclose(
        transpose_recovered, direction, rtol=3.0e-8, atol=3.0e-8
    )
    tangent = _branch_tangent(
        zero,
        0.0,
        runtime,
        PolishConfig(),
        None,
        block_preconditioner,
    )
    np.testing.assert_allclose(
        jnp.vdot(tangent[0], tangent[0]).real + tangent[1] ** 2,
        1.0,
        rtol=2.0e-13,
    )
    assert float(tangent[1]) > 0.0
    rhs = (jnp.linspace(-0.2, 0.3, runtime.layout.size), jnp.asarray(0.4))
    corrected = _bordered_preconditioner(
        runtime, tangent, block_preconditioner
    )((zero, 0.0), rhs, 1.0e6)
    assert corrected[0].shape == zero.shape
    assert np.all(np.isfinite(np.asarray(corrected[0])))
    assert np.isfinite(float(corrected[1]))


def test_polish_driver_records_bounded_unpolished_return(
    small_strong_root, monkeypatch
):
    class InitialCertificate:
        normalized_l2 = jnp.asarray(2.0)

    config = PolishConfig(
        max_continuation_stages=1,
        alpha_initial_step=1.0e-3,
        alpha_min_step=1.0e-3,
        alpha_max_step=1.0e-3,
        max_nonlinear_iterations=12,
        preconditioner="legacy",
        use_pseudo_arclength=True,
        fail_policy="return_unpolished",
    )

    def fail_tangent(*args, **kwargs):
        del args, kwargs
        raise StrongForceContinuationError("test tangent failure")

    def endpoint(residual, initial, **kwargs):
        del residual, kwargs
        return SimpleNamespace(
            x=initial,
            steps=2,
            linear_iterations=3,
            residual_evaluations=4,
            converged=True,
            linear_converged=True,
        )

    def continuation(residual, initial, *, accept_stage, **kwargs):
        del residual, kwargs
        alpha = 1.0e-3
        accept_stage(initial, alpha, None)
        stage = SimpleNamespace(
            nonlinear_steps=5,
            linear_iterations=6,
            residual_evaluations=7,
            accepted=True,
        )
        return SimpleNamespace(
            x=initial,
            alpha=alpha,
            steps=(stage,),
            converged=False,
        )

    monkeypatch.setattr(
        "vmex.core.polish_driver._solvax_continuation_api",
        lambda: (lambda **kwargs: object(), None, continuation, None, endpoint),
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._ptc_config", lambda config, **kwargs: object()
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._arclength_to_target", fail_tangent
    )
    chart = make_strong_structured_chart(small_strong_root)
    result = polish_strong_root(
        small_strong_root,
        config=config,
        initial_certificate=InitialCertificate(),
        chart=chart,
    )
    report = result.polish_report
    assert not report.converged
    assert report.termination_reason == "pseudo-arclength-tangent-failed"
    assert report.final_alpha == pytest.approx(1.0e-3)
    assert report.continuation_accepted == 1
    assert report.continuation_rejected == 0
    assert report.nonlinear_iterations > 0
    assert report.linear_iterations > 0
    assert report.minimum_signed_jacobian > 0.0
    np.testing.assert_array_equal(result.correction, 0.0)
    assert result.native_equilibrium is small_strong_root.native


def test_polish_driver_skips_an_already_certified_state(small_strong_root):
    class InitialCertificate:
        normalized_l2 = jnp.asarray(1.0e-9)
        minimum_signed_jacobian = jnp.asarray(0.5)

    result = polish_strong_root(
        small_strong_root,
        config=PolishConfig(validation_tolerance=1.0e-8),
        initial_certificate=InitialCertificate(),
    )
    report = result.polish_report
    assert report.converged
    assert report.termination_reason == "already-certified"
    assert report.nonlinear_iterations == 0
    assert report.linear_iterations == 0
    assert report.residual_evaluations == 0
    np.testing.assert_array_equal(result.correction, 0.0)


def test_legacy_polish_announces_refinement_and_certificate_phases():
    """The legacy driver's setup phases each emit a notice before starting.

    Small solovev case; the raw lift never certifies at the default bar, so
    the run passes through every phase (refinement, initial certificate,
    preconditioner/root-runtime build) into one bounded Gauss-Newton step.
    """
    from vmex import VmecInput
    from vmex.core.polish_driver import polish_legacy_solution
    from vmex.core.solver import resolution_from_input, solve

    inp = VmecInput.from_file(
        str(DATA / "input.solovev")
    ).change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(
        inp, ns_array=np.asarray([5]), ftol_array=np.asarray([1.0e-9]),
        niter_array=np.asarray([2000]),
    )
    result = solve(inp)
    lines: list[str] = []

    def capture(text="", **kwargs):
        lines.append(str(text))

    polish_legacy_solution(
        inp,
        resolution_from_input(inp, ns=5),
        result.state,
        config=PolishConfig(
            max_nonlinear_iterations=1,
            collocation_scale_probes=2,
            fail_policy="return_unpolished",
        ),
        verbose=True,
        emit=capture,
    )
    text = "\n".join(lines)
    assert "refining the converged state" in text
    assert "evaluating the initial force certificate" in text
    assert "building the polish preconditioner and root runtime" in text


def test_collocation_polish_announces_each_phase(small_strong_root):
    """Every silent setup phase emits a notice before it starts.

    A W7-X-scale user run showed the banner and then nothing for minutes;
    the notices exist so a quiet console is always attributable to a named
    phase.  Chart and certificate are deliberately NOT prebuilt here so the
    driver's own build paths (and their notices) execute.
    """
    lines: list[str] = []

    def capture(text="", **kwargs):
        lines.append(str(text))

    polish_collocation_least_squares(
        small_strong_root,
        config=PolishConfig(
            tolerance=2.0,
            validation_tolerance=10.0,
            radial_refinement_tolerance=10.0,
            collocation_scale_probes=2,
            max_nonlinear_iterations=1,
            fail_policy="return_unpolished",
        ),
        verbose=True,
        emit=capture,
    )
    text = "\n".join(lines)
    assert "building the polish chart" in text
    assert "evaluating the initial force certificate" in text
    assert "collocation:" in text


def test_collocation_polish_primal_and_derivatives(small_strong_root):
    chart = make_strong_structured_chart(
        small_strong_root, balance_iterations=1, balance_probes=2
    )
    result = polish_collocation_least_squares(
        small_strong_root,
        chart=chart,
        config=PolishConfig(
            tolerance=2.0,
            validation_tolerance=10.0,
            radial_refinement_tolerance=10.0,
            collocation_scale_probes=2,
            max_nonlinear_iterations=1,
            fail_policy="return_unpolished",
        ),
    )
    assert result.correction.shape == (small_strong_root.layout.size,)
    assert result.polish_report.least_squares_success is not None
    assert result.polish_report.variable_scale_probes == 2
    assert result.context is not None
    # Acceptance is the independent certificate, not the solver's internal
    # tolerance: with the certificate bars satisfied the result is accepted
    # even when the Gauss-Newton step budget expires first.
    assert result.polish_report.converged
    assert result.polish_report.termination_reason == "independently-certified"

    native_tangent = _random_like(small_strong_root.native, 51)
    output_cotangent = _random_like(small_strong_root.native, 52)
    linear_config = PolishLinearConfig(
        rtol=2.0e-10,
        atol=2.0e-11,
        restart=chart.size,
        max_restarts=5,
    )
    tangent = collocation_polish_tangent(
        result.context, native_tangent, config=linear_config
    )
    adjoint = collocation_polish_adjoint(
        result.context, output_cotangent, config=linear_config
    )
    assert bool(tangent.report.converged)
    assert bool(adjoint.report.converged)
    np.testing.assert_allclose(
        _tree_dot(output_cotangent, tangent.native_tangent),
        _tree_dot(adjoint.native_cotangent, native_tangent),
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    def objective(native):
        polished = implicit_collocation_polished_state(
            native, result.context, linear_config
        )
        return _tree_dot(polished, output_cotangent)

    custom_gradient = jax.grad(objective)(small_strong_root.native)
    difference = jax.tree.map(
        jnp.subtract, custom_gradient, adjoint.native_cotangent
    )
    assert _tree_norm(difference) <= 2.0e-5 * max(
        _tree_norm(adjoint.native_cotangent), 1.0
    )

    polished = implicit_collocation_polished_state(
        small_strong_root.native,
        result.context,
        linear_config,
    )

    def boozer_objective(native):
        spectrum = boozer_spectrum_high_order(
            native,
            surfaces=[0.49],
            mboz=4,
            nboz=2,
            ntheta=12,
            nzeta=8,
        )
        return jnp.sum(spectrum["bmnc_b"][:, 1:] ** 2)

    boozer_cotangent = jax.grad(boozer_objective)(polished)
    boozer_adjoint = collocation_polish_adjoint(
        result.context,
        boozer_cotangent,
        config=linear_config,
    )
    boozer_gradient = jax.grad(
        lambda native: boozer_objective(
            implicit_collocation_polished_state(
                native,
                result.context,
                linear_config,
            )
        )
    )(small_strong_root.native)
    boozer_difference = jax.tree.map(
        jnp.subtract,
        boozer_gradient,
        boozer_adjoint.native_cotangent,
    )
    assert _tree_norm(boozer_difference) <= 2.0e-5 * max(
        _tree_norm(boozer_adjoint.native_cotangent),
        1.0,
    )

    base_stationarity = _collocation_stationarity(
        result.context.correction,
        small_strong_root.native,
        result.context.runtime,
        result.context.chart,
    )

    def stationarity_remainder(step):
        perturbed_native = jax.tree.map(
            lambda value, direction: value + step * direction,
            small_strong_root.native,
            native_tangent,
        )
        perturbed_correction = (
            result.context.correction + step * tangent.correction_tangent
        )
        return jnp.linalg.norm(
            _collocation_stationarity(
                perturbed_correction,
                perturbed_native,
                result.context.runtime,
                result.context.chart,
            )
            - base_stationarity
        )

    coarse = stationarity_remainder(2.0e-5)
    fine = stationarity_remainder(1.0e-5)
    assert fine < 0.35 * coarse


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"tolerance": 0.0}, "tolerances"),
        ({"validation_tolerance": 0.0}, "tolerances"),
        ({"radial_degree": 4}, "radial_degree"),
        ({"radial_spans": 0}, "radial_spans"),
        ({"radial_quadrature_order": 1}, "radial_quadrature_order"),
        ({"radial_refinement_tolerance": 0.0}, "radial_refinement_tolerance"),
        ({"collocation_scale_probes": -1}, "collocation_scale_probes"),
        ({"least_squares_initial_damping": 0.0}, "least_squares_initial_damping"),
        ({"alpha_min_step": 0.1}, "alpha_min_step"),
        ({"ptc_initial_dtau": 0.0}, "ptc_initial_dtau"),
        ({"max_continuation_stages": 0}, "iteration limits"),
        ({"linear_restart": 0}, "linear/backtracking"),
        ({"preconditioner": "bad"}, "preconditioner"),
        ({"minimum_jacobian_ratio": 0.0}, "minimum_jacobian_ratio"),
        ({"minimum_jacobian_floor": 0.0}, "minimum_jacobian_floor"),
        ({"arclength_step": 0.0}, "pseudo-arclength"),
        ({"fail_policy": "bad"}, "fail_policy"),
        ({"tolerance": float("nan")}, "finite"),
    ],
)
def test_polish_config_validation(updates, message):
    with pytest.raises(ValueError, match=message):
        PolishConfig(**updates)


def test_polish_ptc_stopping_is_invariant_to_positive_residual_scaling():
    tolerance = 2.0e-7
    config = _ptc_config(PolishConfig(tolerance=tolerance), residual_scale=3.0e-4)
    rescaled = _ptc_config(
        PolishConfig(tolerance=tolerance), residual_scale=7.0 * 3.0e-4
    )
    assert config.rtol == tolerance
    assert config.atol == pytest.approx(tolerance * 3.0e-4)
    assert rescaled.atol == pytest.approx(7.0 * config.atol)


def test_low_endpoint_check_ignores_numerical_row_equilibration():
    residual = jnp.asarray([2.0e-9, -6.0e-9])
    runtime = SimpleNamespace(
        equation_scale=jnp.asarray([2.0, 3.0]),
        layout=SimpleNamespace(size=2),
    )
    rescaled_runtime = SimpleNamespace(
        equation_scale=7.0 * runtime.equation_scale,
        layout=runtime.layout,
    )
    expected = jnp.linalg.norm(residual / runtime.equation_scale) / jnp.sqrt(2.0)
    np.testing.assert_allclose(
        _normalized_low_residual_norm(residual, runtime),
        expected,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        _normalized_low_residual_norm(7.0 * residual, rescaled_runtime),
        expected,
        rtol=2.0e-13,
    )


def test_public_solver_rejects_unknown_polish_mode_before_solving():
    inp = VmecInput.from_file(DATA / "input.solovev")
    with pytest.raises(ValueError, match="False, True, or 'auto'"):
        solver.solve(inp, polish="unknown")


def test_public_solver_resolves_polish_keywords_only():
    """Directives live in run_options; the solver sees only its keywords."""
    inp = VmecInput.from_file(DATA / "input.solovev")
    assert solver._resolve_force_balance_polish(inp, None, None) is False
    assert solver._resolve_force_balance_polish(inp, True, None) is True
    assert solver._resolve_force_balance_polish(inp, True, False) is True
    assert solver._resolve_force_balance_polish(inp, "auto", None) == "auto"
    with pytest.raises(ValueError, match="either polish or polish_force_balance"):
        solver._resolve_force_balance_polish(inp, False, True)


def test_public_solver_auto_attaches_an_already_certified_native_state():
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
    result = solver.solve(
        inp,
        ftol=1.0e-10,
        max_iterations=1000,
        polish="auto",
        polish_config=PolishConfig(
            radial_degree=3,
            validation_tolerance=3.0,
        ),
    )
    assert result.converged
    assert result.native_equilibrium is not None
    assert result.strong_force is not None
    assert result.polish_report.converged
    assert result.polish_report.termination_reason == "already-certified"
    assert result.polished_state is not None
    assert result.state.R_cos.shape == (5, 3)
    assert result.polished_state.R_cos.shape == result.state.R_cos.shape
    assert np.all(np.isfinite(result.polished_state.R_cos))
    assert result.native_equilibrium.R_cos.shape == (3, 4)


def test_sample_high_order_state_inverts_the_lift_on_any_mesh(small_adapter):
    native, _, state, _, _ = small_adapter
    inp = _small_solovev_input()
    for ns in (5, 11):
        runtime = solver.prepare_runtime(
            inp, solver.resolution_from_input(inp, ns=ns)
        )
        sampled = sample_high_order_state(native, runtime)
        assert np.shape(np.asarray(sampled.R_cos)) == (ns, native.m.size)
        relift = lift_high_order_state(
            sampled, runtime, radial_basis=native.radial_basis, degree=3
        )
        for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
            np.testing.assert_allclose(
                np.asarray(getattr(relift, name)),
                np.asarray(getattr(native, name)),
                rtol=0.0,
                atol=1.0e-11,
            )
    # The solve-mesh sample reproduces the fixed boundary row exactly: the
    # lift pinned the edge spline coefficients to the legacy edge values.
    solve_mesh = solver.prepare_runtime(
        inp, solver.resolution_from_input(inp, ns=5)
    )
    sampled = sample_high_order_state(native, solve_mesh)
    np.testing.assert_allclose(
        np.asarray(sampled.R_cos[-1]), np.asarray(state.R_cos[-1]),
        rtol=0.0, atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(sampled.Z_sin[-1]), np.asarray(state.Z_sin[-1]),
        rtol=0.0, atol=1.0e-13,
    )


def test_sample_high_order_state_requires_matching_mode_tables(small_adapter):
    native, *_ = small_adapter
    other = VmecInput.from_file(DATA / "input.solovev").change_resolution(
        mpol=4, ntor=0, ntheta=12, nzeta=4
    )
    runtime = solver.prepare_runtime(
        other, solver.resolution_from_input(other, ns=5)
    )
    with pytest.raises(ValueError, match="mode tables"):
        sample_high_order_state(native, runtime)


def test_polished_wout_ns_covers_reconstruction_and_the_native_basis():
    native = SimpleNamespace(radial_basis=SimpleNamespace(size=17))
    # The stable wout lift caps at 32 spans; four samples per capped span.
    assert polished_wout_ns(native, solve_ns=31) == 129
    # A finer solve mesh is never coarsened.
    assert polished_wout_ns(native, solve_ns=201) == 201
    # A native basis beyond the cap still stays fully determined.
    wide = SimpleNamespace(radial_basis=SimpleNamespace(size=90))
    assert polished_wout_ns(wide, solve_ns=31) == 181


def test_polished_wout_export_certifies_near_the_native_state(tmp_path):
    pytest.importorskip("netCDF4")
    import vmex as vj
    from vmex.core.strong_force import (
        certify_strong_force,
        high_order_state_from_wout,
    )
    from vmex.core.wout import read_wout

    inp = _small_solovev_input()
    physics = inp.to_indata(tmp_path / "input.polished_export")
    source = tmp_path / "input.polished_export_directive"
    source.write_text(
        "!@VMEX POLISH = .TRUE.\n" + physics.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = PolishConfig(radial_degree=3, validation_tolerance=3.0)
    result = vj.solve_file(source, outdir=tmp_path, polish_config=config)
    assert bool(result.polish_report.converged)
    # The in-memory API contract is untouched: polished_state still matches
    # the solve mesh.  Only the exported file gets the certifiable mesh.
    assert result.polished_state.R_cos.shape == result.state.R_cos.shape

    wout_path = tmp_path / "wout_polished_export_directive.nc"
    solve_ns = int(np.shape(np.asarray(result.state.R_cos))[0])
    exported_ns = int(read_wout(wout_path).ns)
    assert exported_ns == polished_wout_ns(
        result.native_equilibrium, solve_ns=solve_ns
    )
    assert exported_ns > solve_ns

    native_l2 = float(np.asarray(result.strong_force.normalized_l2))
    exported = high_order_state_from_wout(wout_path, inp=inp, degree=5)
    exported_l2 = float(np.asarray(certify_strong_force(exported).normalized_l2))
    # The export is the only carrier of the polish gain for downstream wout
    # consumers: the stable default reconstruction of the file must
    # re-certify within 10% of the native continuous certificate.
    assert exported_l2 <= 1.1 * native_l2
