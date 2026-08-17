"""Accepted-iteration reporting independent of optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
import sys
from typing import Any, Callable, cast, Mapping, Sequence, TextIO

import numpy as np

from .problem import FunctionProblem


_DEFAULT_STREAM = object()


class EquilibriumReporter:
    """Print a compact set of scalar diagnostics for an equilibrium.

    Each quantity is ``(label, callable, format_spec)``.  Callables may use
    either the ``function(equilibrium)`` or ``function(state, runtime)``
    convention used by VMEX objectives.  Calling the reporter prints one line
    and returns the values by label, so scripts can also reuse a final metric.
    """

    def __init__(
        self,
        *quantities: tuple[str, Callable[..., Any], str],
        stream: TextIO | None | object = _DEFAULT_STREAM,
        separator: str = ", ",
    ) -> None:
        if not quantities:
            raise ValueError("at least one equilibrium quantity is required")
        names = [name for name, _function, _format in quantities]
        if len(set(names)) != len(names):
            raise ValueError("equilibrium quantity labels must be unique")
        self.quantities = quantities
        self.stream: TextIO | None = (
            sys.stdout if stream is _DEFAULT_STREAM else cast(TextIO | None, stream)
        )
        self.separator = str(separator)

    @staticmethod
    def _value(function: Callable[..., Any], equilibrium: Any) -> float:
        try:
            parameters = [
                parameter for parameter in inspect.signature(function).parameters.values()
                if parameter.kind in (parameter.POSITIONAL_ONLY,
                                      parameter.POSITIONAL_OR_KEYWORD)
            ]
            state_function = (len(parameters) >= 2 and
                              parameters[1].default is inspect.Parameter.empty)
        except (TypeError, ValueError):
            state_function = False
        value = (function(equilibrium.state, equilibrium.runtime)
                 if state_function else function(equilibrium))
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError("equilibrium report quantities must be scalar")
        return float(array.reshape(()))

    def __call__(self, label: str, equilibrium: Any) -> dict[str, float]:
        """Evaluate, optionally print, and return the configured quantities."""
        values = {name: self._value(function, equilibrium)
                  for name, function, _format in self.quantities}
        if self.stream is not None:
            fields = [f"{name} = {format(values[name], format_spec)}"
                      for name, _function, format_spec in self.quantities]
            print(f"[{label}] {self.separator.join(fields)}", file=self.stream)
        return values


@dataclass(frozen=True)
class OptimizationRecord:
    """One optimizer callback, normally one accepted iteration."""

    iteration: int
    cost: float
    reduction: float | None
    optimality: float | None
    equilibrium_solves: int | None
    rejected_trials: int | None
    terms: Mapping[str, float] = field(default_factory=dict)


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
        self._x_history: list[np.ndarray] = []
        self._evaluations: dict[bytes, tuple[float, float, dict[str, float]]] = {}
        self._last_key: bytes | None = None

    @staticmethod
    def _key(x: Any) -> bytes:
        array = np.ascontiguousarray(np.asarray(x, dtype=float))
        return array.shape.__repr__().encode() + array.tobytes()

    def wrap_value_and_grad(
        self,
        function: Callable | Sequence[Callable],
        term_names: tuple[str, ...] | None = None,
        *,
        residual_slices: tuple[tuple[str, int, int], ...] = (),
    ) -> Callable:
        """Adapt a JAX ``has_aux`` value/gradient pair for SciPy.

        Each ``function(x)`` must return ``((cost, terms), gradient)``. Pass a
        sequence to compile large additive physics components separately;
        their costs and gradients are summed without changing the optimizer
        contract. ``terms``
        may be a mapping of labels to weighted scalar costs, or one compact
        vector paired with ``term_names``. For a large residual graph, pass
        ``residual_slices`` and return ``(residual, extra_costs...)``; costs are
        reduced on the host to keep the compiled output small. The first
        evaluation is recorded as iteration zero; later evaluations are cached.
        """
        functions = (function,) if callable(function) else tuple(function)
        if not functions:
            raise ValueError("provide at least one value-and-gradient function")

        def wrapped(x):
            outputs = [component(x) for component in functions]
            cost_f = sum(float(output[0][0]) for output in outputs)
            gradient_np = np.sum([
                np.asarray(output[1], dtype=float) for output in outputs], axis=0)
            auxiliaries = [output[0][1] for output in outputs]
            terms = auxiliaries[0] if len(auxiliaries) == 1 else auxiliaries
            if isinstance(terms, Mapping):
                term_values = {str(name): float(value) for name, value in terms.items()}
            elif residual_slices:
                first, *extra = terms
                if isinstance(first, (tuple, list)):
                    residual, *first_extra = first
                    extra = [*first_extra, *extra]
                else:
                    residual = first
                residual = np.asarray(residual, dtype=float).ravel()
                term_values = {
                    str(name): 0.5 * float(residual[start:stop] @ residual[start:stop])
                    for name, start, stop in residual_slices}
                values = np.concatenate([np.asarray(value, dtype=float).ravel()
                                         for value in extra])
                if term_names is None or len(term_names) != values.size:
                    raise TypeError(
                        "residual auxiliary data requires one name per extra cost")
                term_values.update(zip(map(str, term_names), map(float, values)))
            else:
                values = np.asarray(terms, dtype=float).ravel()
                if term_names is None or len(term_names) != values.size:
                    raise TypeError(
                        "vector auxiliary data requires one term name per value")
                term_values = dict(zip(map(str, term_names), map(float, values)))
            return self.cache_evaluation(x, cost_f, gradient_np, term_values)

        return wrapped

    def cache_evaluation(
        self,
        x: Any,
        cost: Any,
        gradient: Any,
        terms: Mapping[str, Any] | None = None,
    ) -> tuple[float, np.ndarray]:
        """Cache an already-computed objective pair for a SciPy callback.

        This method does no differentiation and does not alter the objective.
        Driver scripts can show their explicit ``jax.value_and_grad`` calls,
        sum independently compiled physics components themselves, and use this
        one host conversion to avoid recomputing per-term costs when SciPy later
        reports an accepted iterate.
        """
        cost_f = float(cost)
        gradient_np = np.asarray(gradient, dtype=float)
        term_values = ({} if terms is None else
                       {str(name): float(value) for name, value in terms.items()})
        key = self._key(x)
        self._evaluations[key] = (
            cost_f, float(np.linalg.norm(gradient_np)), term_values)
        if not self.records:
            self.record(
                x, cost=cost_f, optimality=self._evaluations[key][1],
                terms=term_values)
        return cost_f, gradient_np

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

    def _term_costs(self, x: np.ndarray) -> dict[str, float]:
        if self.problem is None or not self.problem.metadata.get("term_slices"):
            return {}
        try:
            residual = self.problem.residual(x)
        except AttributeError:
            return {}
        return {
            str(name): 0.5 * float(residual[start:stop] @ residual[start:stop])
            for name, start, stop in self.problem.metadata["term_slices"]
        }

    def __call__(self, intermediate_result: Any) -> None:
        """Consume a SciPy callback value: ``OptimizeResult``, dict, or x.

        Legacy SciPy ``minimize`` callbacks receive the plain parameter
        vector; treat an array argument as that vector instead of probing
        it for an ``x`` attribute (which silently produced a 0-d NaN).
        """
        if isinstance(intermediate_result, np.ndarray):
            intermediate_result = {"x": intermediate_result}
        x = np.asarray(self._field(intermediate_result, "x"), dtype=float)
        key = self._key(x)
        cached = self._evaluations.get(key)
        if key == self._last_key and cached is not None:
            return
        cost = self._field(intermediate_result, "cost")
        raw_fun = self._field(intermediate_result, "fun")
        if cost is None and cached is not None:
            cost = cached[0]
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
        if optimality is None and cached is not None:
            optimality = cached[1]
        if optimality is None:
            gradient = self._field(intermediate_result, "jac")
            if gradient is not None and np.asarray(gradient).ndim == 1:
                optimality = np.linalg.norm(np.asarray(gradient), ord=np.inf)
        self.record(
            x,
            cost=float(cost),
            optimality=None if optimality is None else float(optimality),
            iteration=int(iteration),
            terms=(cached[2] if cached is not None else self._term_costs(x)),
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
        terms: Mapping[str, Any] | None = None,
    ) -> OptimizationRecord:
        """Append one already-computed accepted iterate and return its record."""
        if terms is None:
            terms = self._term_costs(np.asarray(x, dtype=float))
        if iteration is None:
            iteration = len(self.records)
        elif self.records and int(iteration) <= self.records[-1].iteration:
            # Optimizers restart their iteration count at every continuation
            # stage.  Keep one monitor's combined history strictly ordered.
            iteration = self.records[-1].iteration + 1
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
            terms={} if terms is None else {
                str(name): float(value) for name, value in terms.items()},
        )
        self.records.append(item)
        self._x_history.append(np.asarray(x, dtype=float).copy())
        self._last_key = self._key(x)
        if self.stream is not None and (len(self.records) - 1) % self.print_every == 0:
            self._print(item)
        return item

    @property
    def history(self) -> dict[str, np.ndarray]:
        """Return total and per-term cost histories as one mapping."""
        names = dict.fromkeys(
            name for record in self.records for name in record.terms)
        history = {"total": np.asarray([record.cost for record in self.records])}
        history.update({
            name: np.asarray([record.terms.get(name, np.nan)
                              for record in self.records])
            for name in names})
        return history

    @property
    def x_history(self) -> tuple[np.ndarray, ...]:
        """Copies of the accepted decision vectors, including iteration zero."""
        return tuple(x.copy() for x in self._x_history)

    def save(self, path: str | Path) -> Path:
        """Save the recorded iteration and cost columns as CSV."""
        path = Path(path)
        history = self.history
        names = list(history)
        values = np.column_stack([
            np.asarray([record.iteration for record in self.records]),
            *(history[name] for name in names),
        ])
        np.savetxt(path, values, delimiter=",", header="iteration," + ",".join(names),
                   comments="")
        return path

    def plot(self, path: str | Path, *, title: str = "Optimization objective terms") -> Path:
        """Write a compact log-scale total and per-term cost history plot."""
        if not self.records:
            raise ValueError("no optimization records to plot")
        import matplotlib.pyplot as plt

        path = Path(path)
        figure, axis = plt.subplots(figsize=(6.5, 4.0))
        iterations = np.asarray([record.iteration for record in self.records])
        finite_positive = np.concatenate([
            values[np.isfinite(values) & (values > 0.0)]
            for values in self.history.values()
        ])
        display_floor = max(
            1.0e-8,
            float(np.min(finite_positive)) if finite_positive.size else 1.0e-8,
        )
        for name, values in self.history.items():
            positive = np.where(np.isfinite(values), np.maximum(values, display_floor), np.nan)
            axis.semilogy(
                iterations, positive, marker="o", markersize=3.0, label=name)
        axis.set(xlabel="iteration", ylabel="weighted cost", title=title)
        axis.set_ylim(bottom=display_floor)
        axis.grid(True, alpha=0.3); axis.legend(fontsize=8, ncol=2)
        figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)
        return path

    def movie(
        self,
        path: str | Path,
        object_factory: Callable[[np.ndarray], Any],
        **kwargs: Any,
    ) -> Path:
        """Animate accepted surface/coil iterates with one geometry callback."""
        from .plotting import plot_optimization_movie

        return plot_optimization_movie(path, self.x_history, object_factory, **kwargs)

    def movie_surface_coils(
        self,
        path: str | Path,
        object_factory: Callable[[np.ndarray], Any],
        *,
        x0: Any,
        scales: Any,
        surface_color: str | Callable | None = None,
        plasma_problem: Any | None = None,
        external_field: Callable[[Any], Any] | None = None,
        nphi: int = 32,
        ntheta: int = 32,
        digits: int = 4,
        precision: Any | None = None,
        **kwargs: Any,
    ) -> Path:
        """Animate normalized surface/coil iterates with optional field color.

        ``object_factory`` receives physical variables ``x0 + scales*u``.
        ``surface_color`` may be ``None``, ``"absB"``, ``"B.n/B"``, or a
        callable ``(u, objects) -> values``. For the two named field colors,
        provide ``plasma_problem``; ``B.n/B`` additionally needs
        ``external_field(objects)``. These plotting helpers never enter the
        objective or gradient graph.
        """
        x0, scales = np.asarray(x0, dtype=float), np.asarray(scales, dtype=float)
        if x0.shape != scales.shape:
            raise ValueError("x0 and scales must have the same shape")

        def objects_from_u(u):
            return object_factory(x0 + scales * np.asarray(u, dtype=float))

        color_factory = None
        if callable(surface_color):
            color_factory = surface_color
        elif surface_color is not None:
            if surface_color not in ("absB", "B.n/B"):
                raise ValueError('surface_color must be None, "absB", "B.n/B", or callable')
            if plasma_problem is None:
                raise ValueError("named surface colors require plasma_problem")
            nplasma = int(np.asarray(plasma_problem.x0).size)

            def color_factory(u, objects):
                field = None if external_field is None else external_field(objects)
                return plasma_problem.surface_field_values(
                    x0[:nplasma] + scales[:nplasma] * np.asarray(u)[:nplasma],
                    surface_color, external_field=field, nphi=nphi,
                    ntheta=ntheta, digits=digits, precision=precision)

        return self.movie(
            path, objects_from_u, color_factory=color_factory,
            color_label=str(surface_color), **kwargs)

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


__all__ = ["EquilibriumReporter", "OptimizationMonitor", "OptimizationRecord"]
