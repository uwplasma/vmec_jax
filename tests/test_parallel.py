"""Concurrent ensemble solves (``vmex.core.parallel``, Item G / issue #41).

The contract of the threaded ensemble helpers is *byte-identical* results:
solving ``N`` independent inputs through a thread pool must reproduce, bit for
bit, the outcome of solving each one alone (the concurrency only overlaps the
GIL-releasing XLA execution windows — it touches no numerics).  These tests
assert exactly that on a varied ensemble, plus the small behavioural surface
(ordering, worker clamping, exception policy).  Strong-scaling *timing* is a
measurement, not a unit test — it lives in ``examples/parallel_ensemble_scan.py``
and ``docs/explanation/parallelization.rst``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import numpy as np
import pytest

from vmex.core import parallel
from vmex.core.input import VmecInput
from vmex.core.solver import solve

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(autouse=True)
def _enable_jit():
    """Full solves need JIT (the repo conftest disables it for unit tests)."""
    was_disabled = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", was_disabled)

# A varied, deliberately SMALL ensemble (single grid, modest ns) so the test is
# CI-cheap: solovev (2D ncurr=0), circular_tokamak (2D, more modes) and
# li383_low_res (3D, ncurr=1 / lconm1).
_DECKS = ("solovev", "circular_tokamak", "li383_low_res")


def _small(name: str) -> VmecInput:
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    ns = 11 if name == "solovev" else 13
    return dataclasses.replace(
        inp, ns_array=[ns], ftol_array=[1e-11], niter_array=[3000]
    )


def _state_max_abs_diff(a, b) -> float:
    return max(
        float(np.max(np.abs(np.asarray(getattr(a, f)) - np.asarray(getattr(b, f)))))
        for f in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    )


@pytest.mark.parametrize(
    "workers", [2, pytest.param(4, marks=pytest.mark.full)]
)
def test_solve_ensemble_bit_identical_to_serial(workers):
    """Threaded ``solve_ensemble`` == serial solve, bit for bit, per member."""
    inputs = [_small(d) for d in _DECKS]

    serial = [solve(inp) for inp in inputs]
    ensemble = parallel.solve_ensemble(inputs, workers=workers, multigrid=False)

    assert len(ensemble) == len(serial)
    for name, a, b in zip(_DECKS, serial, ensemble):
        # iteration count and convergence flag identical
        assert int(a.iterations) == int(b.iterations), name
        assert bool(a.converged) == bool(b.converged), name
        # converged state bit-identical (exact zero difference)
        assert _state_max_abs_diff(a.state, b.state) == 0.0, name


@pytest.mark.full
def test_solve_ensemble_multigrid_matches_serial():
    """The default (multigrid) path is likewise bit-identical to a serial run."""
    inp = _small("li383_low_res")
    # a small phiedge scan — the balanced same-structure ensemble use case
    inputs = [
        dataclasses.replace(inp, phiedge=float(inp.phiedge) * m)
        for m in (0.98, 1.0, 1.02)
    ]
    from vmex.core.multigrid import solve_multigrid

    serial = [solve_multigrid(x) for x in inputs]
    ensemble = parallel.solve_ensemble(inputs, workers=3)  # multigrid=True default

    for a, b in zip(serial, ensemble):
        assert int(a.iterations) == int(b.iterations)
        assert _state_max_abs_diff(a.state, b.state) == 0.0


def test_map_ensemble_preserves_order_and_is_serial_at_one_worker():
    """``map_ensemble`` keeps input order; workers=1 is a clean serial loop."""
    items = list(range(10))
    fn = lambda k: k * k  # noqa: E731
    assert parallel.map_ensemble(fn, items, workers=1) == [k * k for k in items]
    assert parallel.map_ensemble(fn, items, workers=4) == [k * k for k in items]
    assert parallel.map_ensemble(fn, []) == []


def test_default_workers_clamping():
    cpu = parallel.available_cpus()
    assert parallel.default_workers(3) == min(3, cpu)
    assert parallel.default_workers(10_000) == min(10_000, cpu)
    # explicit -> clamped to [1, n_items]
    assert parallel.default_workers(4, 100) == 4
    assert parallel.default_workers(4, 0) == 1
    assert parallel.default_workers(4, -5) == 1
    assert parallel.default_workers(0) == 1


def test_available_cpus_respects_process_limits(monkeypatch):
    monkeypatch.setattr(parallel.os, "process_cpu_count", lambda: 3, raising=False)
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
    assert parallel.available_cpus() == 3


def test_available_cpus_uses_affinity_fallback(monkeypatch):
    monkeypatch.delattr(parallel.os, "process_cpu_count", raising=False)
    monkeypatch.setattr(
        parallel.os, "sched_getaffinity", lambda _pid: {2, 4}, raising=False
    )
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
    assert parallel.available_cpus() == 2


def test_available_cpus_respects_scheduler_allocation(monkeypatch):
    monkeypatch.delattr(parallel.os, "process_cpu_count", raising=False)
    monkeypatch.delattr(parallel.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS", "LSB_DJOB_NUMPROC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert parallel.available_cpus() == 6


def test_available_cpus_ignores_unavailable_hints(monkeypatch):
    monkeypatch.delattr(parallel.os, "process_cpu_count", raising=False)

    def unavailable(_pid):
        raise OSError("affinity unavailable")

    monkeypatch.setattr(parallel.os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 7)
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS", "LSB_DJOB_NUMPROC"):
        monkeypatch.setenv(name, "invalid")
    assert parallel.available_cpus() == 7


def test_map_ensemble_exception_policy():
    def boom(k: int) -> int:
        if k == 2:
            raise ValueError(f"boom {k}")
        return k

    # default: propagate the first failure (serial-loop semantics)
    with pytest.raises(ValueError, match="boom 2"):
        parallel.map_ensemble(boom, [0, 1, 2, 3], workers=2)

    # return_exceptions=True: one bad member does not abort the batch
    out = parallel.map_ensemble(
        boom, [0, 1, 2, 3], workers=2, return_exceptions=True
    )
    assert out[0] == 0 and out[1] == 1 and out[3] == 3
    assert isinstance(out[2], ValueError)


def test_parallel_module_exposed_as_vmex_attribute():
    import vmex

    assert vmex.parallel is parallel


def test_parallel_finite_difference_and_problem_ensemble():
    """Automatic worker utilities are accurate, ordered, and backend-neutral."""
    from vmex.core.problem import FunctionProblem

    matrix = np.array([[2.0, -1.0], [0.5, 3.0]])
    jacobian = parallel.finite_difference_jacobian(
        lambda x: matrix @ x,
        np.array([0.3, -0.7]),
        workers=None,
    )
    np.testing.assert_allclose(jacobian, matrix, rtol=1e-10, atol=1e-10)
    gradient = parallel.finite_difference_gradient(
        lambda x: 0.5 * x @ matrix.T @ matrix @ x,
        np.array([0.3, -0.7]),
        workers=2,
    )
    np.testing.assert_allclose(gradient, matrix.T @ matrix @ [0.3, -0.7], rtol=1e-9)

    problems = [
        FunctionProblem([float(i)], fun=lambda x, i=i: float((x[0] - i) ** 2))
        for i in range(3)
    ]
    evaluations = parallel.evaluate_problems(problems, workers=None, derivatives=False)
    assert [evaluation.value for evaluation in evaluations] == [0.0, 0.0, 0.0]

    forward = parallel.finite_difference_jacobian(
        lambda x: matrix @ x,
        np.array([0.3, -0.7]),
        method="2-point",
        rel_step=1e-6,
        workers=1,
    )
    np.testing.assert_allclose(forward, matrix, rtol=1e-9, atol=1e-9)
    assert parallel.finite_difference_jacobian(
        lambda x: x, np.array([]), workers=1
    ).shape == (0, 0)

    with pytest.raises(ValueError, match="one-dimensional"):
        parallel.finite_difference_jacobian(np.sum, np.ones((2, 2)))
    with pytest.raises(ValueError, match="method"):
        parallel.finite_difference_jacobian(np.sum, np.ones(2), method="bad")
    with pytest.raises(ValueError, match="finite and positive"):
        parallel.finite_difference_jacobian(np.sum, np.ones(2), rel_step=0.0)
    with pytest.raises(ValueError, match="output size changed"):
        parallel.finite_difference_jacobian(
            lambda x: x[:1] if x[0] > 0.0 else x,
            np.zeros(2),
            workers=1,
        )
    with pytest.raises(ValueError, match="equal length"):
        parallel.evaluate_problems(problems, [problems[0].x0])
