"""Accepted-iteration reporting independent of optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, cast, TextIO

import numpy as np

from .problem import FunctionProblem


_DEFAULT_STREAM = object()


@dataclass(frozen=True)
class OptimizationRecord:
    """One optimizer callback, normally one accepted iteration."""

    iteration: int
    cost: float
    reduction: float | None
    optimality: float | None
    equilibrium_solves: int | None
    rejected_trials: int | None


class OptimizationMonitor:
    """Record and optionally print accepted optimizer iterations.

    Pass the instance as a SciPy ``callback``.  SciPy invokes callbacks after
    an iteration, unlike objective functions which are also called for rejected
    line-search or trust-region trials.  JAXopt, Optax, and custom loops can
    call :meth:`record` with values they already computed.

    The monitor never chooses steps or changes an optimizer.  If ``problem``
    is supplied, VMEX solve/failure counters are read without evaluating the
    objective again.
    """

    def __init__(
        self,
        problem: FunctionProblem | None = None,
        *,
        stream: TextIO | None | object = _DEFAULT_STREAM,
        print_every: int = 1,
    ) -> None:
        if print_every < 1:
            raise ValueError("print_every must be at least 1")
        self.problem = problem
        self.stream: TextIO | None = (
            sys.stdout if stream is _DEFAULT_STREAM else cast(TextIO | None, stream)
        )
        self.print_every = int(print_every)
        self.records: list[OptimizationRecord] = []

    @staticmethod
    def _field(result: Any, name: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    def _counters(self) -> tuple[int | None, int | None]:
        if self.problem is None:
            return None, None
        metadata = self.problem.metadata
        holder = metadata.get("holder", {})
        rejected = holder.get("failed_trials")
        cfg = metadata.get("config")
        if cfg is None:
            return None, rejected
        from . import implicit as imp

        stats = imp._SOLVE_STATS.get(cfg)
        solves = None if stats is None else int(stats.get("solves", 0))
        return solves, rejected

    def __call__(self, intermediate_result: Any) -> None:
        """Consume a SciPy ``OptimizeResult`` callback value."""
        x = np.asarray(self._field(intermediate_result, "x"), dtype=float)
        cost = self._field(intermediate_result, "cost")
        raw_fun = self._field(intermediate_result, "fun")
        if cost is None and raw_fun is not None:
            values = np.asarray(raw_fun, dtype=float)
            cost = (float(values) if values.ndim == 0
                    else 0.5 * float(values.ravel() @ values.ravel()))
        if cost is None:
            if self.problem is None:
                raise ValueError("callback did not provide cost or fun")
            cost = self.problem.fun(x)
        iteration = self._field(intermediate_result, "nit", len(self.records))
        optimality = self._field(intermediate_result, "optimality")
        if optimality is None:
            gradient = self._field(intermediate_result, "jac")
            if gradient is not None and np.asarray(gradient).ndim == 1:
                optimality = np.linalg.norm(np.asarray(gradient), ord=np.inf)
        self.record(
            x,
            cost=float(cost),
            optimality=None if optimality is None else float(optimality),
            iteration=int(iteration),
        )

    def record(
        self,
        x: Any,
        *,
        cost: float,
        optimality: float | None = None,
        iteration: int | None = None,
        equilibrium_solves: int | None = None,
        rejected_trials: int | None = None,
    ) -> OptimizationRecord:
        """Append one already-computed accepted iterate and return its record."""
        del x  # accepted for a uniform callback/manual-loop interface
        if iteration is None:
            iteration = len(self.records)
        if equilibrium_solves is None or rejected_trials is None:
            solves, rejected = self._counters()
            if equilibrium_solves is None:
                equilibrium_solves = solves
            if rejected_trials is None:
                rejected_trials = rejected
        reduction = None
        if self.records:
            reduction = self.records[-1].cost - float(cost)
        item = OptimizationRecord(
            iteration=int(iteration),
            cost=float(cost),
            reduction=reduction,
            optimality=optimality,
            equilibrium_solves=equilibrium_solves,
            rejected_trials=rejected_trials,
        )
        self.records.append(item)
        if self.stream is not None and (len(self.records) - 1) % self.print_every == 0:
            self._print(item)
        return item

    @staticmethod
    def _number(value: float | int | None, *, integer: bool = False) -> str:
        if value is None:
            return "-"
        return str(int(value)) if integer else f"{float(value):.6e}"

    def _print(self, item: OptimizationRecord) -> None:
        if len(self.records) == 1:
            print(
                " iter          cost     reduction   optimality  eq solves  rejected",
                file=self.stream,
            )
        print(
            f"{item.iteration:5d}  {item.cost:12.6e}  "
            f"{self._number(item.reduction):>12}  "
            f"{self._number(item.optimality):>11}  "
            f"{self._number(item.equilibrium_solves, integer=True):>9}  "
            f"{self._number(item.rejected_trials, integer=True):>8}",
            file=self.stream,
        )


__all__ = ["OptimizationMonitor", "OptimizationRecord"]
