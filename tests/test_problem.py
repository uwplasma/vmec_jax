"""Fast contract tests for optimizer-neutral problem callables."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vmex
from vmex.core.problem import Evaluation, FunctionProblem


def _quadratic_problem(calls=None):
    calls = {"value_and_grad": 0, "residual_and_jac": 0} if calls is None else calls
    target = np.array([1.0, -2.0])

    def value_and_grad(x):
        calls["value_and_grad"] += 1
        residual = x - target
        return 0.5 * residual @ residual, residual

    def residual_and_jac(x):
        calls["residual_and_jac"] += 1
        return x - target, np.eye(2)

    return FunctionProblem(
        [3.0, 1.0],
        value_and_grad=value_and_grad,
        residual_and_jac=residual_and_jac,
        names=("a", "b"),
        bounds=([-5.0, -5.0], [5.0, 5.0]),
        scales=[1.0, 2.0],
    )


def test_problem_contract_and_exact_key_cache():
    calls = {"value_and_grad": 0, "residual_and_jac": 0}
    problem = _quadratic_problem(calls)

    value, grad = problem.value_and_grad(problem.x0)
    np.testing.assert_allclose((value, *grad), (6.5, 2.0, 3.0))
    np.testing.assert_array_equal(problem.grad(problem.x0.copy()), grad)
    assert calls["value_and_grad"] == 1

    residual, jacobian = problem.residual_and_jac(problem.x0)
    np.testing.assert_array_equal(problem.residual(problem.x0.copy()), residual)
    np.testing.assert_array_equal(problem.residual_jac(problem.x0), jacobian)
    assert calls["residual_and_jac"] == 1

    assert problem.J(problem.x0) == value
    np.testing.assert_array_equal(problem.dJ(problem.x0), grad)
    assert problem.names == ("a", "b")
    np.testing.assert_array_equal(problem.scales, [1.0, 2.0])
    assert FunctionProblem.from_functions([1.0], fun=np.sum).fun([2.0]) == 2.0


def test_evaluation_contains_consistent_scalar_and_residual_forms():
    evaluation = _quadratic_problem().evaluate([3.0, 1.0])
    assert isinstance(evaluation, Evaluation) and evaluation.success
    assert evaluation.value == 6.5
    np.testing.assert_array_equal(evaluation.gradient, [2.0, 3.0])
    np.testing.assert_array_equal(evaluation.residual, [2.0, 3.0])
    np.testing.assert_array_equal(evaluation.jacobian, np.eye(2))


def test_warmup_primes_primary_path_and_reports_progress():
    calls = {"value_and_grad": 0, "residual_and_jac": 0}
    problem = _quadratic_problem(calls)
    evaluation = problem.warmup(progress=False)
    assert evaluation.value == 6.5
    assert calls == {"value_and_grad": 0, "residual_and_jac": 1}

    scalar = problem.warmup(evaluation_path="scalar", progress=False)
    assert scalar.value == 6.5
    assert scalar.residual is None
    assert calls == {"value_and_grad": 1, "residual_and_jac": 1}

    with pytest.raises(ValueError, match="evaluation_path"):
        problem.warmup(evaluation_path="gradient", progress=False)
    scalar_only = FunctionProblem([1.0], value_and_grad=lambda x: (x[0] ** 2, 2 * x))
    with pytest.raises(AttributeError, match="does not provide residuals"):
        scalar_only.warmup(evaluation_path="residual", progress=False)

    def slow_residual_and_jac(x):
        time.sleep(0.03)
        return x - 1.0, np.eye(x.size)

    stream = io.StringIO()
    slow = FunctionProblem(
        [2.0, 4.0],
        residual_and_jac=slow_residual_and_jac,
        metadata={"warmup_description": "a synthetic derivative"},
    )
    warmed = slow.warmup(report_interval=0.005, stream=stream)
    output = stream.getvalue()
    assert "Preparing a synthetic derivative" in output
    assert "Still preparing:" in output
    assert "Preparation complete in" in output
    assert "Initial cost:" in output
    assert "Jacobian: 2 x 2" in output
    np.testing.assert_array_equal(warmed.gradient, [1.0, 3.0])
    with pytest.raises(ValueError, match="report_interval"):
        slow.warmup(report_interval=0.0)


def test_prepare_optimization_input_matches_staged_qi_resolution():
    from vmex.core import optimize as opt
    from vmex.core.input import VmecInput

    rbc = np.arange(9.0).reshape(3, 3)
    zbs = rbc + 20.0
    inp = VmecInput(mpol=3, ntor=1, rbc=rbc, zbs=zbs)
    staged = opt.prepare_optimization_input(inp, 4, minimum_mpol=5)
    assert (staged.mpol, staged.ntor) == (6, 6)
    assert (staged.ntheta, staged.nzeta, staged.delt) == (18, 16, 0.5)
    assert (inp.mpol, inp.ntor, inp.ntheta, inp.nzeta, inp.delt) == (3, 1, 0, 0, 1.0)
    for n in range(-1, 2):
        for m in range(3):
            assert staged.rbc[n + staged.ntor, m] == inp.rbc[n + inp.ntor, m]
            assert staged.zbs[n + staged.ntor, m] == inp.zbs[n + inp.ntor, m]
    assert np.count_nonzero(staged.rbc[:, 3:]) == 0

    custom = opt.prepare_optimization_input(
        inp, 1, mpol=4, ntor=2, ntheta=12, nzeta=7, delt=0.25
    )
    assert (custom.mpol, custom.ntor, custom.ntheta, custom.nzeta, custom.delt) == (
        4, 2, 12, 7, 0.25,
    )
    with pytest.raises(ValueError, match="mpol"):
        opt.prepare_optimization_input(inp, 4, mpol=4)
    with pytest.raises(ValueError, match="aliasing"):
        opt.prepare_optimization_input(inp, 1, ntor=2, nzeta=4)


def test_vmec_problem_factory_reports_construction_progress(monkeypatch):
    from vmex.core import optimize as opt
    from vmex.core.input import VmecInput

    def fake_implicit_problem(*args, **kwargs):
        del args, kwargs
        time.sleep(0.02)
        return FunctionProblem(
            [0.0], fun=np.sum, metadata={"derivative_method": "implicit"}
        )

    monkeypatch.setattr(opt, "_least_squares_implicit", fake_implicit_problem)
    stream = io.StringIO()
    problem = opt.make_problem(
        VmecInput(mpol=2, ntor=1),
        objective_terms=[(lambda equilibrium: equilibrium, 0.0, 1.0)],
        progress=True,
        report_interval=0.005,
        progress_stream=stream,
    )
    output = stream.getvalue()
    assert "Preparing the VMEC problem and initial equilibrium" in output
    assert "Still preparing:" in output
    assert "Preparation complete in" in output
    assert problem.metadata["derivative_method"] == "implicit"


def test_residual_only_problem_derives_scalar_value_and_gradient():
    problem = FunctionProblem(
        [2.0, 4.0],
        residual_and_jac=lambda x: (x - 1.0, np.eye(2)),
    )
    residual = np.array([1.0, 3.0])
    assert problem.fun(problem.x0) == 0.5 * residual @ residual
    np.testing.assert_array_equal(problem.grad(problem.x0), residual)
    evaluation = problem.evaluate(problem.x0)
    assert evaluation.value == 5.0
    np.testing.assert_array_equal(evaluation.gradient, residual)
    only_rows = FunctionProblem([2.0, 4.0], residual=lambda x: x - 1.0)
    np.testing.assert_array_equal(only_rows.residual(only_rows.x0), residual)


def test_separate_callable_and_missing_derivative_paths():
    separate = FunctionProblem(
        [2.0, 4.0],
        fun=lambda x: float(x @ x),
        grad=lambda x: 2.0 * x,
        residual=lambda x: x - 1.0,
        residual_jac=lambda x: np.eye(x.size),
    )
    assert separate.fun(separate.x0) == 20.0
    np.testing.assert_array_equal(separate.grad(separate.x0), [4.0, 8.0])
    np.testing.assert_array_equal(separate.residual(separate.x0), [1.0, 3.0])
    np.testing.assert_array_equal(separate.residual_jac(separate.x0), np.eye(2))
    assert separate.evaluate(separate.x0, derivatives=False).gradient is None

    scalar_only = FunctionProblem([1.0], fun=np.sum)
    with pytest.raises(AttributeError, match="scalar gradient"):
        scalar_only.grad(scalar_only.x0)
    with pytest.raises(AttributeError, match="residuals"):
        scalar_only.residual(scalar_only.x0)
    with pytest.raises(AttributeError, match="residual Jacobian"):
        scalar_only.residual_jac(scalar_only.x0)
    for method in (
        scalar_only.jax_fun,
        scalar_only.jax_value_and_grad,
        scalar_only.jax_residual,
        scalar_only.jax_residual_jac,
    ):
        with pytest.raises(AttributeError):
            method(scalar_only.x0)


def test_jax_fallbacks_and_jacobian_shape_validation():
    problem = FunctionProblem(
        [1.0],
        fun=np.sum,
        jax_value_and_grad=lambda x: (np.sum(x), np.ones_like(x)),
        jax_residual=lambda x: x,
        jax_residual_jac=lambda x: np.eye(len(x)),
    )
    assert problem.jax_fun(problem.x0) == 1.0
    np.testing.assert_array_equal(problem.jax_value_and_grad(problem.x0)[1], [1.0])
    np.testing.assert_array_equal(problem.jax_residual(problem.x0), [1.0])
    np.testing.assert_array_equal(problem.jax_residual_jac(problem.x0), [[1.0]])

    malformed = FunctionProblem(
        [1.0, 2.0], residual_and_jac=lambda x: (x, np.ones((1, 2)))
    )
    with pytest.raises(ValueError, match="must have shape"):
        malformed.residual_and_jac(malformed.x0)


def test_problem_rejects_ambiguous_metadata_shapes():
    with pytest.raises(ValueError, match="one entry"):
        FunctionProblem([1.0, 2.0], fun=np.sum, names=["one"])
    with pytest.raises(ValueError, match="same shape"):
        FunctionProblem([1.0, 2.0], fun=np.sum, scales=[1.0])
    with pytest.raises(ValueError, match="positive"):
        FunctionProblem([1.0], fun=np.sum, scales=[0.0])
    with pytest.raises(ValueError, match="provide"):
        FunctionProblem([1.0])


def test_direct_scipy_minimize_and_least_squares_use_same_problem():
    scipy = pytest.importorskip("scipy.optimize")
    problem = _quadratic_problem()
    minimized = scipy.minimize(
        problem.value_and_grad, problem.x0, jac=True, method="BFGS"
    )
    fitted = scipy.least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac
    )
    np.testing.assert_allclose(minimized.x, [1.0, -2.0], atol=1e-8)
    np.testing.assert_allclose(fitted.x, [1.0, -2.0], atol=1e-8)


def test_vmex_jax_logging_default_and_override() -> None:
    """The default removes PjRt chatter without preventing an opt-in."""
    code = "import vmex, jax; print(jax.config.jax_logging_level)"
    env = os.environ.copy()
    env.pop("VMEX_JAX_LOGGING_LEVEL", None)
    env.pop("JAX_LOGGING_LEVEL", None)
    default = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True,
        capture_output=True, text=True,
    )
    assert default.stdout.strip() == "ERROR"
    assert "pjrt_executable.cc" not in default.stderr

    env["VMEX_JAX_LOGGING_LEVEL"] = "warning"
    override = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True,
        capture_output=True, text=True,
    )
    assert override.stdout.strip() == "WARNING"

    env.pop("VMEX_JAX_LOGGING_LEVEL")
    env["JAX_LOGGING_LEVEL"] = "INFO"
    standard_override = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True,
        capture_output=True, text=True,
    )
    assert standard_override.stdout.strip() == "INFO"


def test_old_jax_gets_one_actionable_logging_notice() -> None:
    old_jax = SimpleNamespace(__version__="0.4.35", config=SimpleNamespace())
    with pytest.warns(RuntimeWarning, match="JAX 0.4.35.*Upgrade JAX"):
        vmex._configure_jax_logging(old_jax)


def test_direct_jaxopt_and_optax_contracts():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jaxopt = pytest.importorskip("jaxopt")
    optax = pytest.importorskip("optax")

    target = jnp.asarray([1.0, -2.0])

    def loss(x):
        return 0.5 * jnp.vdot(x - target, x - target)

    jax_vg = jax.value_and_grad(loss)
    problem = FunctionProblem(
        [3.0, 1.0],
        value_and_grad=lambda x: tuple(map(np.asarray, jax_vg(jnp.asarray(x)))),
        jax_fun=loss,
        jax_value_and_grad=jax_vg,
    )

    result = jaxopt.LBFGS(
        problem.jax_value_and_grad, value_and_grad=True,
        maxiter=10, tol=1e-10,
    ).run(jnp.asarray(problem.x0))
    np.testing.assert_allclose(result.params, target, atol=1e-6)

    transform = optax.adam(0.1)
    x = jnp.asarray(problem.x0)
    state = transform.init(x)
    for _ in range(100):
        _, grad = problem.jax_value_and_grad(x)
        updates, state = transform.update(grad, state, x)
        x = optax.apply_updates(x, updates)
    assert float(problem.jax_fun(x)) < float(problem.jax_fun(problem.x0))
