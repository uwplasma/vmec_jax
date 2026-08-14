"""Fast contracts for accepted-iteration monitoring."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import OptimizeResult

from vmex.core.monitoring import EquilibriumReporter, OptimizationMonitor
from vmex.core.problem import FunctionProblem


def test_equilibrium_reporter_supports_both_objective_call_styles(monkeypatch) -> None:
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
    with np.testing.assert_raises_regex(ValueError, "at least one"):
        EquilibriumReporter()
    from vmex.core import monitoring
    monkeypatch.setattr(monitoring.inspect, "signature", lambda function: (_ for _ in ()).throw(ValueError))
    assert EquilibriumReporter(("opaque", lambda eq: 1.0, ".1f"), stream=None)(
        "final", equilibrium) == {"opaque": 1.0}


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

    # A continuation stage can restart SciPy's ``nit`` at one.  The combined
    # history remains monotonic, and recorded vectors are defensive copies.
    x = np.array([3.0])
    monitor.record(x, cost=0.25, iteration=1)
    x[0] = 99.0
    assert [record.iteration for record in monitor.records] == [1, 2, 3]
    assert monitor.x_history[-1].tolist() == [3.0]
    copied = monitor.x_history[-1]; copied[0] = -1.0
    assert monitor.x_history[-1].tolist() == [3.0]


def test_monitor_wraps_auxiliary_term_costs_and_records_only_accepted_points() -> None:
    monitor = OptimizationMonitor(stream=None)

    def pair(x):
        value = np.asarray(x) @ np.asarray(x)
        return (value, {"shape": 0.25 * value, "field": 0.75 * value}), 2 * np.asarray(x)

    wrapped = monitor.wrap_value_and_grad(pair)
    wrapped(np.array([2.0])); wrapped(np.array([9.0]))  # rejected trial stays cached
    monitor(np.array([2.0]))  # duplicate of the recorded initial point
    wrapped(np.array([1.0])); monitor(np.array([1.0]))

    assert [record.cost for record in monitor.records] == [4.0, 1.0]
    assert monitor.history["shape"].tolist() == [1.0, 0.25]
    vector = OptimizationMonitor(stream=None)
    vector_pair = vector.wrap_value_and_grad(
        lambda x: ((1.0, np.array([0.25, 0.75])), np.ones(1)), ("a", "b"))
    vector_pair([0.0])
    assert vector.history["b"].tolist() == [0.75]
    residual = OptimizationMonitor(stream=None)
    residual_pair = residual.wrap_value_and_grad(
        lambda x: ((2.0, (np.array([1.0, 1.0]), np.array([1.0]))), np.ones(1)),
        ("extra",), residual_slices=(("rows", 0, 2),))
    residual_pair([0.0])
    assert residual.history["rows"].tolist() == [1.0]
    with np.testing.assert_raises_regex(TypeError, "term name"):
        monitor.wrap_value_and_grad(lambda x: ((1.0, np.ones(1)), np.ones(1)))([0.0])
    with np.testing.assert_raises_regex(TypeError, "one name per extra"):
        monitor.wrap_value_and_grad(
            lambda x: ((1.0, (np.ones(1), np.ones(2))), np.ones(1)),
            ("one",), residual_slices=(("rows", 0, 1),))([0.0])


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


def test_monitor_collects_saves_and_plots_objective_terms(tmp_path) -> None:
    problem = FunctionProblem(
        [1.0, 2.0], residual=lambda x: np.asarray([x[0], x[1], 2 * x[1]]),
        metadata={"term_slices": (("shape", 0, 1), ("field", 1, 3))})
    monitor = OptimizationMonitor(problem, stream=None)
    monitor({"x": np.array([1.0, 2.0]), "cost": 10.5, "nit": 0})
    monitor.record(np.zeros(2), cost=1.0, iteration=1,
                   terms={"shape": 0.0, "field": 0.75})

    np.testing.assert_allclose(monitor.history["shape"], [0.5, 0.0])
    np.testing.assert_allclose(monitor.history["field"], [10.0, 0.75])
    csv = monitor.save(tmp_path / "history.csv")
    plot = monitor.plot(tmp_path / "history.png")
    assert csv.is_file() and "iteration,total,shape,field" in csv.read_text().splitlines()[0]
    assert plot.is_file() and plot.stat().st_size > 0


def test_plot_optimization_objects_is_dependency_neutral(tmp_path) -> None:
    import vmex as vj

    class Object:
        gamma = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])

        def plot(self, *, ax, show):
            assert not show
            ax.plot(*self.gamma.reshape(-1, 3).T)

    path = vj.plot_optimization_objects(
        tmp_path / "objects.png", ("Initial", Object()), ("Final", Object()))
    assert path.is_file() and path.stat().st_size > 0
    curves = SimpleNamespace(gamma=Object.gamma)
    coil = SimpleNamespace(curves=curves, plot=Object().plot)
    assert vj.plot_optimization_objects(tmp_path / "coils.png", ("Coils", coil)).is_file()
    with np.testing.assert_raises_regex(ValueError, "at least one"):
        vj.plot_optimization_objects(tmp_path / "bad.png")


def test_bootstrap_plot_and_small_optimization_movie(tmp_path, monkeypatch) -> None:
    import vmex as vj

    class Mismatch:
        def current_profiles(self, _equilibrium):
            return np.linspace(0.1, 0.9, 4), np.arange(4.0), np.arange(4.0) + 0.1

    bootstrap = vj.plot_bootstrap_current(
        tmp_path / "bootstrap.png", object(), Mismatch())
    assert bootstrap.is_file() and bootstrap.stat().st_size > 0

    class Surface:
        area_element = np.ones((4, 5))

        def __init__(self, shift):
            phi, theta = np.meshgrid(
                np.linspace(0, 2 * np.pi, 4, endpoint=False),
                np.linspace(0, 2 * np.pi, 5, endpoint=False), indexing="ij")
            radius = 1.0 + 0.1 * np.cos(theta)
            self.gamma = np.stack((radius * np.cos(phi) + shift,
                                   radius * np.sin(phi), 0.1 * np.sin(theta)), axis=-1)

    class Coils:
        def __init__(self, shift):
            theta = np.linspace(0, 2 * np.pi, 12)
            self.gamma = np.stack((np.cos(theta) + shift, np.sin(theta), 0 * theta), axis=-1)[None]

    class Line:
        def __init__(self, shift):
            self.gamma = np.array([[shift, 0.0, 0.0], [shift, 0.0, 1.0]])

    class CurvesOnly:
        def __init__(self, shift):
            self.curves = SimpleNamespace(gamma=Line(shift).gamma[None])

    monitor = OptimizationMonitor(stream=None)
    monitor.record([0.0], cost=1.0); monitor.record([0.1], cost=0.5)
    movie = monitor.movie(
        tmp_path / "optimization.gif",
        lambda x: (Surface(x[0]), Coils(x[0]), Line(x[0]), CurvesOnly(x[0])),
        fps=2, max_frames=4, dpi=40)
    assert movie.is_file() and movie.stat().st_size > 0
    with np.testing.assert_raises_regex(ValueError, "at least one"):
        vj.plot_optimization_movie(tmp_path / "empty.gif", (), lambda x: x)
    with np.testing.assert_raises_regex(ValueError, "max_frames"):
        vj.plot_optimization_movie(tmp_path / "bad.gif", ([0.0],), Line, fps=0)
    with np.testing.assert_raises_regex(TypeError, "gamma"):
        vj.plot_optimization_movie(tmp_path / "bad.gif", ([0.0],), lambda x: object())
    with np.testing.assert_raises_regex(ValueError, "gif or .mp4"):
        vj.plot_optimization_movie(tmp_path / "bad.txt", ([0.0],), Line)
    import matplotlib.animation
    monkeypatch.setattr(matplotlib.animation.writers, "is_available", lambda name: False)
    with np.testing.assert_raises_regex(RuntimeError, "requires ffmpeg"):
        vj.plot_optimization_movie(
            tmp_path / "bad.mp4", ([0.0],), lambda x: Line(float(x[0])))


def test_monitor_empty_optional_paths_raise_or_return_empty(tmp_path) -> None:
    problem = FunctionProblem([0.0], fun=np.sum, metadata={
        "term_slices": (("missing", 0, 1),)})
    monitor = OptimizationMonitor(problem, stream=None)
    assert monitor._term_costs(problem.x0) == {}
    with np.testing.assert_raises_regex(ValueError, "no optimization records"):
        monitor.plot(tmp_path / "empty.png")


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
