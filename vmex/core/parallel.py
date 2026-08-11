"""Concurrent ensembles of independent equilibrium solves on CPU (Item G).

A parameter scan or an ensemble optimization solves ``N`` *independent*
equilibria (different boundaries / ``phiedge`` / profiles).  Each forward
solve runs on the host behind :func:`jax.pure_callback`
(:mod:`vmex.core.implicit`) or directly through
:func:`vmex.core.solver.solve` / :func:`vmex.core.multigrid.solve_multigrid`,
and — crucially — **releases the Python GIL while XLA executes the compiled
iteration lanes**.  A plain :class:`concurrent.futures.ThreadPoolExecutor` over
those independent solves therefore overlaps their XLA execution and gives real
wall-clock speedup, while every result stays *byte-identical* to solving that
input alone (the solves share no mutable state).

Measured strong scaling (10-core host, 8 balanced ``phiedge`` solves): 1.79x
with 2 workers, 3.29x with 8.  The scaling is deliberately sub-linear: XLA
already multithreads *within* one solve, so as the worker count approaches
the core count the per-solve XLA threads contend — the ensemble speedup and
the intra-solve speedup draw from the same cores.  See :doc:`/explanation/parallelization`
for the full mechanism study (why threading beats ``pmap`` across forced host
devices and ``vmap`` over the callback here), the honest limits (Amdahl on
imbalanced heterogeneous ensembles; the launch-bound implicit adjoint
overlaps far less than the forward solve), and the multi-GPU design sketch.

This module is a thin, additive concurrency layer: it changes nothing in the
single-solve path (which stays byte-identical) and imposes no new dependency.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence, TypeVar

__all__ = [
    "default_workers",
    "evaluate_problems",
    "finite_difference_gradient",
    "finite_difference_jacobian",
    "map_ensemble",
    "solve_ensemble",
]

_T = TypeVar("_T")
_R = TypeVar("_R")


def default_workers(n_items: int, workers: int | None = None) -> int:
    """Resolve the worker count for an ``n_items`` ensemble.

    ``None`` (default) picks ``min(n_items, os.cpu_count())`` — enough threads
    to cover the ensemble without oversubscribing the cores that each solve's
    XLA threads already use.  An explicit ``workers`` is honoured but clamped
    to ``[1, n_items]`` (more threads than items cannot help, and 0/negative is
    meaningless).
    """
    n_items = max(int(n_items), 1)
    if workers is None:
        cpu = os.cpu_count() or 1
        return max(1, min(n_items, cpu))
    return max(1, min(int(workers), n_items))


def map_ensemble(
    fn: Callable[[_T], _R],
    items: Iterable[_T],
    *,
    workers: int | None = None,
    return_exceptions: bool = False,
) -> list[_R]:
    """Apply ``fn`` to each of ``items`` concurrently on CPU; keep input order.

    The general primitive behind :func:`solve_ensemble`.  ``fn`` must be an
    *independent* per-item computation (e.g. a full ``vj.solve`` /
    ``implicit.run`` / ``jax.value_and_grad`` over one input) that shares no
    mutable state with the others — which every vmex forward solve is, since
    each builds its own runtime and the compiled-executable cache is
    thread-safe.  Under those conditions the results are byte-identical to a
    serial ``[fn(x) for x in items]`` (the concurrency only overlaps the
    GIL-releasing XLA execution windows).

    ``workers`` — see :func:`default_workers`.  With ``workers=1`` the pool
    runs sequentially (a clean serial baseline for scaling measurements).

    ``return_exceptions=False`` (default) re-raises the first item's exception
    (preserving vmex's typed :class:`~vmex.core.errors.VmecError` taxonomy),
    exactly as a serial loop would.  ``return_exceptions=True`` instead places
    the caught exception object in that slot so one failed ensemble member does
    not abort the batch (useful for optimization ensembles / robustness scans).
    """
    items = list(items)
    if not items:
        return []
    n_workers = default_workers(len(items), workers)

    def _call(x: _T) -> Any:
        if return_exceptions:
            try:
                return fn(x)
            except Exception as exc:  # noqa: BLE001 - deliberately captured
                return exc
        return fn(x)

    if n_workers == 1:
        return [_call(x) for x in items]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        # executor.map preserves input order and propagates exceptions on
        # iteration (when return_exceptions is False, _call already re-raises).
        return list(pool.map(_call, items))


def finite_difference_jacobian(
    fun: Callable[[Any], Any],
    x: Any,
    *,
    method: str = "3-point",
    rel_step: float | None = None,
    workers: int | None = None,
) -> Any:
    """Differentiate an opaque host function with independent parallel probes.

    ``method="3-point"`` (default) is central and second-order accurate;
    ``"2-point"`` is forward and first-order accurate.  ``workers=None`` uses
    :func:`default_workers`, while ``workers=1`` is deterministic serial
    execution.  Every probe receives its own copy of ``x`` and ``fun`` must not
    mutate shared state.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional decision vector")
    if method not in ("2-point", "3-point"):
        raise ValueError("method must be '2-point' or '3-point'")
    if rel_step is None:
        rel_step = np.finfo(float).eps ** (0.5 if method == "2-point" else 1.0 / 3.0)
    if not np.isfinite(rel_step) or rel_step <= 0.0:
        raise ValueError("rel_step must be finite and positive")
    step = float(rel_step) * np.maximum(1.0, np.abs(x))

    points: list[np.ndarray] = []
    if method == "2-point":
        points.append(x.copy())
        for j in range(x.size):
            point = x.copy()
            point[j] += step[j]
            points.append(point)
    else:
        for j in range(x.size):
            minus = x.copy()
            plus = x.copy()
            minus[j] -= step[j]
            plus[j] += step[j]
            points.extend((minus, plus))

    values = [np.asarray(value, dtype=float).ravel() for value in map_ensemble(
        fun, points, workers=workers
    )]
    if not values:
        return np.empty((0, 0), dtype=float)
    size = values[0].size
    if any(value.size != size for value in values):
        raise ValueError("fun output size changed between finite-difference probes")
    jacobian = np.empty((size, x.size), dtype=float)
    if method == "2-point":
        base = values[0]
        for j in range(x.size):
            jacobian[:, j] = (values[j + 1] - base) / step[j]
    else:
        for j in range(x.size):
            jacobian[:, j] = (values[2 * j + 1] - values[2 * j]) / (2.0 * step[j])
    return jacobian


def finite_difference_gradient(
    fun: Callable[[Any], Any], x: Any, **kwargs: Any
) -> Any:
    """Parallel finite-difference gradient of a scalar host function."""
    return finite_difference_jacobian(fun, x, **kwargs).reshape(-1)


def evaluate_problems(
    problems: Sequence[Any],
    xs: Sequence[Any] | None = None,
    *,
    derivatives: bool = True,
    workers: int | None = None,
    return_exceptions: bool = False,
) -> list[Any]:
    """Evaluate independent problem objects concurrently in input order.

    Use one problem per ensemble or multistart member so each equilibrium cache
    remains local.  ``xs`` defaults to each problem's ``x0``.
    """
    if xs is None:
        xs = [problem.x0 for problem in problems]
    if len(problems) != len(xs):
        raise ValueError("problems and xs must have equal length")

    def _one(item: tuple[Any, Any]) -> Any:
        problem, x = item
        return problem.evaluate(x, derivatives=derivatives)

    return map_ensemble(
        _one,
        list(zip(problems, xs)),
        workers=workers,
        return_exceptions=return_exceptions,
    )


def solve_ensemble(
    inputs: Sequence[Any],
    *,
    workers: int | None = None,
    multigrid: bool = True,
    return_exceptions: bool = False,
    **solve_kwargs: Any,
) -> list[Any]:
    """Solve ``N`` independent :class:`~vmex.core.input.VmecInput` concurrently.

    Threads :func:`vmex.core.multigrid.solve_multigrid` (``multigrid=True``,
    default — runs each input's ``NS_ARRAY`` ladder) or
    :func:`vmex.core.solver.solve` (``multigrid=False``, single grid) over the
    ensemble on CPU, returning the list of
    :class:`~vmex.core.solver.SolveResult` in input order.  Each result is
    byte-identical to solving that input by itself (verified in
    ``tests/test_parallel.py``): the helper only overlaps the solves' XLA
    execution — it does not touch the numerics, the convergence path, or the
    default single-solve code path.

    Extra ``**solve_kwargs`` (e.g. ``verbose``, ``ftol``, ``initial_state``)
    are forwarded unchanged to every solve.  ``workers`` and
    ``return_exceptions`` behave as in :func:`map_ensemble`.

    Best speedup comes from a *balanced* ensemble — a parameter scan at fixed
    resolution, where the members share a compiled executable and take a
    similar iteration count.  A heterogeneous ensemble is limited by its
    slowest member (Amdahl); see :doc:`/explanation/parallelization`.
    """
    # Imported lazily so importing ``vmex.parallel`` stays cheap and free of a
    # hard import cycle with the solver modules.
    from .multigrid import solve_multigrid
    from .solver import solve

    runner = solve_multigrid if multigrid else solve

    def _one(inp: Any) -> Any:
        return runner(inp, **solve_kwargs)

    return map_ensemble(
        _one, inputs, workers=workers, return_exceptions=return_exceptions
    )
