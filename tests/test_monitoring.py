"""Fast contracts for accepted-iteration monitoring."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import OptimizeResult

from vmex.core.monitoring import EquilibriumReporter, OptimizationMonitor
from vmex.core.problem import FunctionProblem


def test_equilibrium_reporter_supports_both_objective_call_styles() -> None:
    stream = io.StringIO()
    equilibrium = SimpleNamespace(state=np.array([2.0]), runtime=3.0)
    reporter = EquilibriumReporter(
        ("host", lambda eq: eq.state[0], ".2f"),
        ("state", lambda state, runtime: state[0] + runtime, ".1f"),
        ("fraction", lambda eq: 0.025, ".1%"), stream=stream)

    values = reporter("final", equilibrium)

    assert values == {"host": 2.0, "state": 5.0, "fraction": 0.025}
    assert stream.getvalue() == "[final] host = 2.00, state = 5.0, fraction = 2.5%\n"
    with np.testing.assert_raises_regex(ValueError, "unique"):
        EquilibriumReporter(("x", lambda eq: 1.0, ".1f"),
                            ("x", lambda eq: 2.0, ".1f"))
    with np.testing.assert_raises_regex(ValueError, "scalar"):
        EquilibriumReporter(("x", lambda eq: [1.0, 2.0], ".1f"), stream=None)(
            "bad", equilibrium)


def test_monitor_records_scipy_and_manual_iterations() -> None:
    stream = io.StringIO()
    monitor = OptimizationMonitor(stream=stream)
    monitor(OptimizeResult(x=np.ones(2), fun=np.array([2.0, 0.0]), nit=1))
    monitor.record(np.zeros(2), cost=0.5, optimality=0.25, iteration=2)

    assert [item.cost for item in monitor.records] == [2.0, 0.5]
    assert monitor.records[1].reduction == 1.5
    output = stream.getvalue()
    assert output.count("cost") == 1
    assert "reduction" in output
    assert "2.500000e-01" in output


def test_monitor_print_every_and_silent_collection() -> None:
    silent = OptimizationMonitor(stream=None)
    silent.record(np.zeros(1), cost=3.0)
    assert len(silent.records) == 1

    stream = io.StringIO()
    monitor = OptimizationMonitor(stream=stream, print_every=2)
    for i, cost in enumerate((3.0, 2.0, 1.0)):
        monitor.record(np.zeros(1), cost=cost, iteration=i)
    assert len(stream.getvalue().splitlines()) == 3  # header + iterations 0 and 2
    with np.testing.assert_raises(ValueError):
        OptimizationMonitor(print_every=0)


def test_monitor_callback_fallbacks_and_problem_counters() -> None:
    problem = FunctionProblem(
        [2.0],
        fun=lambda x: float(x @ x),
        metadata={"holder": {"failed_trials": 3}},
    )
    monitor = OptimizationMonitor(problem, stream=None)
    monitor(SimpleNamespace(x=np.array([2.0]), nit=4, jac=np.array([4.0])))
    assert monitor.records[0].cost == 4.0
    assert monitor.records[0].optimality == 4.0
    assert monitor.records[0].rejected_trials == 3
    assert monitor.records[0].equilibrium_solves is None

    from vmex.core import implicit as imp

    class Config:
        pass

    config = Config()
    problem.metadata["config"] = config
    imp._SOLVE_STATS[config] = {"solves": 7}
    try:
        monitor(SimpleNamespace(x=np.array([2.0]), fun=4.0, nit=5))
    finally:
        imp._SOLVE_STATS.pop(config, None)
    assert monitor.records[-1].equilibrium_solves == 7

    with np.testing.assert_raises(ValueError):
        OptimizationMonitor(stream=None)({"x": np.array([1.0])})

    # Legacy SciPy minimize callbacks pass the plain parameter vector: it is
    # the iterate itself, never probed for an ``x`` attribute (which used to
    # produce a 0-d NaN evaluation point).
    legacy = OptimizationMonitor(problem, stream=None)
    legacy(np.array([3.0]))
    assert legacy.records[0].cost == 9.0


def test_default_scipy_monitor_respects_an_explicit_callback() -> None:
    from vmex.core import optimize as opt

    kwargs = {}
    monitor = opt._configure_scipy_monitor(
        np.zeros(1),
        lambda x: (float(x @ x), 2.0 * x),
        object(),
        {"failed_trials": 0},
        1,
        kwargs,
    )
    assert isinstance(monitor, OptimizationMonitor)
    assert kwargs["callback"] is monitor

    callback = object()
    explicit = {"callback": callback}
    assert (
        opt._configure_scipy_monitor(
            np.zeros(1), lambda x: (0.0, x), object(), {}, 1, explicit
        )
        is None
    )
    assert explicit["callback"] is callback
    assert (
        opt._configure_scipy_monitor(
            np.zeros(1), lambda x: (0.0, x), object(), {}, 0, {}
        )
        is None
    )


def test_compatibility_least_squares_failure_is_silent_and_counted(monkeypatch) -> None:
    """Rejected finite-difference trials update diagnostics without chatter."""
    import scipy.optimize

    from vmex.core import optimize as opt
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(
        Path(__file__).resolve().parents[1] / "examples/data/input.solovev"
    )
    calls = {"solve": 0}

    def fake_solve(_trial, **kwargs):
        del kwargs
        calls["solve"] += 1
        if calls["solve"] > 1:
            raise RuntimeError("synthetic rejected trial")
        return SimpleNamespace(state=np.zeros(1), value=2.0)

    def fake_least_squares(fun, x0, *, jac, verbose, **kwargs):
        del jac, verbose, kwargs
        initial = fun(x0)
        rejected = np.asarray(x0).copy()
        rejected[0] += 0.1
        assert np.all(fun(rejected) == 1.0e6)
        return OptimizeResult(x=np.asarray(x0), fun=initial, cost=0.5)

    monkeypatch.setattr(opt, "solve_equilibrium", fake_solve)
    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    result = opt.least_squares(
        [(lambda equilibrium: np.atleast_1d(equilibrium.value), 0.0, 1.0)],
        inp,
        max_mode=1,
        jac=None,
    )
    assert result.failed_trials == 1
