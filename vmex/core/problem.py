"""Optimizer-neutral objective and derivative callables.

The classes in this module contain no optimization algorithms.  They expose
the small function contracts consumed by SciPy, JAXopt, Optax, and user code.
VMEC-specific construction is imported lazily so this module remains usable in
lightweight tests and does not introduce an import cycle with
``vmex.core.optimize``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
from threading import Event, RLock, Thread
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


Array = Any
HostFun = Callable[[np.ndarray], Any]


def _run_with_progress(
    function: Callable[[], Any],
    *,
    description: str,
    note: str | None = None,
    progress: bool,
    report_interval: float,
    stream: Any = None,
) -> Any:
    """Run one operation with a low-overhead elapsed-time heartbeat."""
    interval = float(report_interval)
    if interval <= 0.0:
        raise ValueError("report_interval must be positive")
    if not progress:
        return function()
    stream = sys.stdout if stream is None else stream
    print(f"Preparing {description}...", file=stream, flush=True)
    if note:
        print(note, file=stream, flush=True)
    started = time.perf_counter()
    finished = Event()

    def heartbeat() -> None:
        while not finished.wait(interval):
            elapsed = time.perf_counter() - started
            print(
                f"  Still preparing: {elapsed:.1f} s elapsed.",
                file=stream,
                flush=True,
            )

    reporter = Thread(target=heartbeat, name="vmex-progress", daemon=True)
    reporter.start()
    try:
        result = function()
    except Exception:
        elapsed = time.perf_counter() - started
        print(f"Preparation failed after {elapsed:.1f} s.", file=stream, flush=True)
        raise
    finally:
        finished.set()
        reporter.join()
    elapsed = time.perf_counter() - started
    print(f"Preparation complete in {elapsed:.1f} s.", file=stream, flush=True)
    return result


@dataclass(frozen=True)
class Evaluation:
    """Values and diagnostics produced at one decision vector.

    Fields that were not requested or are unavailable are ``None``.  ``status``
    is a short machine-readable value such as ``"success"`` or
    ``"failed_solve"``; ``message`` is intended for a human.  Optimizers use
    the ordinary callable methods and do not need to understand this object.
    """

    x: np.ndarray
    value: float | None = None
    gradient: np.ndarray | None = None
    residual: np.ndarray | None = None
    jacobian: np.ndarray | None = None
    status: str = "success"
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the evaluation completed without a recoverable failure."""
        return self.status == "success"


class FunctionProblem:
    """A decision vector plus optimizer-compatible objective callables.

    Parameters are explicit and immutable from the caller's perspective.
    Supplying combined value/gradient or residual/Jacobian functions enables a
    one-entry exact-key cache, so the common SciPy call sequence does not repeat
    expensive work.  The cache is protected by a lock; JAX-native callables do
    not use host state and remain suitable for tracing.

    This class deliberately does not provide ``solve(method=...)``.  Pass its
    methods directly to the optimizer of choice.
    """

    def __init__(
        self,
        x0: Array,
        *,
        fun: HostFun | None = None,
        grad: HostFun | None = None,
        value_and_grad: HostFun | None = None,
        residual: HostFun | None = None,
        residual_jac: HostFun | None = None,
        residual_and_jac: HostFun | None = None,
        jax_fun: Callable[[Array], Array] | None = None,
        jax_value_and_grad: Callable[[Array], tuple[Array, Array]] | None = None,
        jax_residual: Callable[[Array], Array] | None = None,
        jax_residual_jac: Callable[[Array], Array] | None = None,
        names: Sequence[str] | None = None,
        bounds: Any = None,
        scales: Array | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.names = tuple(names or (f"x[{i}]" for i in range(self.x0.size)))
        if len(self.names) != self.x0.size:
            raise ValueError("names must contain one entry per decision variable")
        self.bounds = bounds
        self.scales = np.ones_like(self.x0) if scales is None else np.asarray(scales, dtype=float)
        if self.scales.shape != self.x0.shape:
            raise ValueError("scales must have the same shape as x0")
        if np.any(~np.isfinite(self.scales)) or np.any(self.scales <= 0.0):
            raise ValueError("scales must be finite and positive")
        self.metadata = dict(metadata or {})

        self._fun = fun
        self._grad = grad
        self._value_and_grad = value_and_grad
        self._residual = residual
        self._residual_jac = residual_jac
        self._residual_and_jac = residual_and_jac
        self._jax_fun = jax_fun
        self._jax_value_and_grad = jax_value_and_grad
        self._jax_residual = jax_residual
        self._jax_residual_jac = jax_residual_jac

        if fun is None and value_and_grad is None and residual is None and residual_and_jac is None:
            raise ValueError("provide a scalar objective, residual function, or combined callable")
        self._lock = RLock()
        self._vg_cache: tuple[tuple[Any, ...], tuple[float, np.ndarray]] | None = None
        self._rj_cache: tuple[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None = None

    @classmethod
    def from_functions(cls, x0: Array, **kwargs: Any) -> "FunctionProblem":
        """Build a problem from user-supplied x-level callables."""
        return cls(x0, **kwargs)

    @staticmethod
    def _x(x: Array) -> np.ndarray:
        return np.asarray(x, dtype=float)

    @staticmethod
    def _key(x: np.ndarray) -> tuple[Any, ...]:
        contiguous = np.ascontiguousarray(x)
        return contiguous.shape, contiguous.dtype.str, contiguous.tobytes()

    def value_and_grad(self, x: Array) -> tuple[float, np.ndarray]:
        """Return scalar value and gradient for SciPy ``jac=True``."""
        xh = self._x(x)
        key = self._key(xh)
        with self._lock:
            if self._vg_cache is not None and self._vg_cache[0] == key:
                value, gradient = self._vg_cache[1]
                return value, gradient.copy()
            if self._value_and_grad is not None:
                value, gradient = self._value_and_grad(xh)
            elif self._fun is not None and self._grad is not None:
                value, gradient = self._fun(xh), self._grad(xh)
            elif self._residual_and_jac is not None:
                residual, jacobian = self._residual_and_jac(xh)
                residual = np.asarray(residual, dtype=float).ravel()
                jacobian = np.asarray(jacobian, dtype=float)
                value = 0.5 * float(residual @ residual)
                gradient = jacobian.T @ residual
            else:
                raise AttributeError("this problem does not provide a scalar gradient")
            pair = float(np.asarray(value)), np.asarray(gradient, dtype=float).reshape(self.x0.shape)
            self._vg_cache = key, pair
            return pair[0], pair[1].copy()

    # SciPy commonly calls this form ``fun_and_grad``; JAX uses
    # ``value_and_grad``.  Both names intentionally share one implementation.
    fun_and_grad = value_and_grad

    def fun(self, x: Array) -> float:
        """Return the scalar objective value."""
        if self._fun is not None:
            return float(np.asarray(self._fun(self._x(x))))
        if self._value_and_grad is not None or self._grad is not None:
            return self.value_and_grad(x)[0]
        residual = self.residual(x)
        return 0.5 * float(residual @ residual)

    def grad(self, x: Array) -> np.ndarray:
        """Return the scalar objective gradient."""
        return self.value_and_grad(x)[1]

    def residual_and_jac(self, x: Array) -> tuple[np.ndarray, np.ndarray]:
        """Return the residual vector and its Jacobian."""
        xh = self._x(x)
        key = self._key(xh)
        with self._lock:
            if self._rj_cache is not None and self._rj_cache[0] == key:
                residual, jacobian = self._rj_cache[1]
                return residual.copy(), jacobian.copy()
            if self._residual_and_jac is not None:
                residual, jacobian = self._residual_and_jac(xh)
            elif self._residual is not None and self._residual_jac is not None:
                residual, jacobian = self._residual(xh), self._residual_jac(xh)
            else:
                raise AttributeError("this problem does not provide a residual Jacobian")
            pair = (np.asarray(residual, dtype=float).ravel(),
                    np.asarray(jacobian, dtype=float))
            if pair[1].shape != (pair[0].size, self.x0.size):
                raise ValueError(
                    "residual Jacobian must have shape "
                    f"({pair[0].size}, {self.x0.size}), got {pair[1].shape}"
                )
            self._rj_cache = key, pair
            return pair[0].copy(), pair[1].copy()

    def residual(self, x: Array) -> np.ndarray:
        """Return the residual vector."""
        if self._residual_and_jac is not None or self._residual_jac is not None:
            return self.residual_and_jac(x)[0]
        if self._residual is None:
            raise AttributeError("this problem does not provide residuals")
        return np.asarray(self._residual(self._x(x)), dtype=float).ravel()

    def residual_jac(self, x: Array) -> np.ndarray:
        """Return the residual Jacobian."""
        return self.residual_and_jac(x)[1]

    def J(self, x: Array) -> float:
        """SIMSOPT-style alias for :meth:`fun`."""
        return self.fun(x)

    def dJ(self, x: Array) -> np.ndarray:
        """SIMSOPT-style alias for :meth:`grad`."""
        return self.grad(x)

    def jax_fun(self, x: Array) -> Array:
        """Return the traceable scalar objective."""
        if self._jax_fun is not None:
            return self._jax_fun(x)
        if self._jax_value_and_grad is not None:
            return self._jax_value_and_grad(x)[0]
        raise AttributeError("this problem does not provide a JAX scalar objective")

    def jax_value_and_grad(self, x: Array) -> tuple[Array, Array]:
        """Return the traceable scalar value and gradient."""
        if self._jax_value_and_grad is None:
            raise AttributeError("this problem does not provide a JAX value-and-gradient")
        return self._jax_value_and_grad(x)

    def jax_residual(self, x: Array) -> Array:
        """Return the traceable residual vector."""
        if self._jax_residual is None:
            raise AttributeError("this problem does not provide JAX residuals")
        return self._jax_residual(x)

    def jax_residual_jac(self, x: Array) -> Array:
        """Return the traceable residual Jacobian."""
        if self._jax_residual_jac is None:
            raise AttributeError("this problem does not provide a JAX residual Jacobian")
        return self._jax_residual_jac(x)

    def evaluate(self, x: Array, *, derivatives: bool = True) -> Evaluation:
        """Evaluate available scalar and residual quantities at ``x``."""
        xh = self._x(x).copy()
        value = gradient = residual = jacobian = None
        if self._fun is not None or self._value_and_grad is not None:
            if derivatives and (self._value_and_grad is not None or self._grad is not None):
                value, gradient = self.value_and_grad(xh)
            else:
                value = self.fun(xh)
        if self._residual is not None or self._residual_and_jac is not None:
            if derivatives and (self._residual_and_jac is not None or self._residual_jac is not None):
                residual, jacobian = self.residual_and_jac(xh)
            else:
                residual = self.residual(xh)
            if value is None:
                value = 0.5 * float(residual @ residual)
                if derivatives and jacobian is not None:
                    gradient = jacobian.T @ residual
        return Evaluation(
            x=xh, value=value, gradient=gradient,
            residual=residual, jacobian=jacobian,
        )

    def warmup(
        self,
        x: Array | None = None,
        *,
        evaluation_path: str = "auto",
        derivatives: bool = True,
        progress: bool = True,
        report_interval: float = 10.0,
        stream: Any = None,
    ) -> Evaluation:
        """Evaluate once, populate caches, and report long first-use work.

        ``evaluation_path="residual"`` prepares residuals and their Jacobian
        for nonlinear least squares.  ``"scalar"`` prepares the value and
        gradient used by BFGS, L-BFGS-B, Adam, and similar optimizers.  The
        default ``"auto"`` selects residuals when available and otherwise the
        scalar path.  Selecting the optimizer's actual path avoids compiling
        an unused derivative graph.

        A heartbeat reports elapsed time while the call is running.  VMEX
        does not invent a first-run ETA: compilation time depends strongly on
        the resolution, objective shape, backend, and local compilation cache.

        Set ``progress=False`` for silent library use or reduce
        ``report_interval`` for more frequent updates.  The returned
        :class:`Evaluation` is the same initial value and derivative data that
        the optimizer will consume.
        """
        xh = self.x0.copy() if x is None else self._x(x).copy()
        has_residual = self._residual is not None or self._residual_and_jac is not None
        has_residual_jac = (
            self._residual_jac is not None or self._residual_and_jac is not None
        )
        has_gradient = self._grad is not None or self._value_and_grad is not None
        if evaluation_path not in ("auto", "residual", "scalar"):
            raise ValueError(
                "evaluation_path must be 'auto', 'residual', or 'scalar'; "
                f"got {evaluation_path!r}"
            )
        selected_path = (
            "residual"
            if evaluation_path == "auto" and has_residual
            else "scalar"
            if evaluation_path == "auto"
            else evaluation_path
        )
        if selected_path == "residual" and not has_residual:
            raise AttributeError("this problem does not provide residuals")

        def evaluate_primary() -> Evaluation:
            if selected_path == "residual":
                if derivatives and has_residual_jac:
                    residual, jacobian = self.residual_and_jac(xh)
                    gradient = jacobian.T @ residual
                else:
                    residual, jacobian, gradient = self.residual(xh), None, None
                return Evaluation(
                    x=xh,
                    value=0.5 * float(residual @ residual),
                    gradient=gradient,
                    residual=residual,
                    jacobian=jacobian,
                )
            if derivatives and has_gradient:
                value, gradient = self.value_and_grad(xh)
            else:
                value, gradient = self.fun(xh), None
            return Evaluation(x=xh, value=value, gradient=gradient)

        description = self.metadata.get(
            f"{selected_path}_warmup_description",
            self.metadata.get(
                "warmup_description",
                (
                    "initial residual and Jacobian"
                    if selected_path == "residual"
                    else "initial value and gradient"
                ),
            ),
        )
        note = self.metadata.get("warmup_note")
        evaluation = _run_with_progress(
            evaluate_primary,
            description=description,
            note=None if note is None else str(note),
            progress=progress,
            report_interval=report_interval,
            stream=stream,
        )
        if progress:
            output = sys.stdout if stream is None else stream
            if evaluation.residual is not None:
                jacobian_shape = (
                    "unavailable"
                    if evaluation.jacobian is None
                    else f"{evaluation.jacobian.shape[0]} x {evaluation.jacobian.shape[1]}"
                )
                print(
                    f"Initial cost: {evaluation.value:.6e}; "
                    f"residual rows: {evaluation.residual.size}; "
                    f"Jacobian: {jacobian_shape}.",
                    file=output,
                    flush=True,
                )
            else:
                gradient_size = (
                    "unavailable"
                    if evaluation.gradient is None
                    else str(evaluation.gradient.size)
                )
                print(
                    f"Initial value: {evaluation.value:.6e}; "
                    f"gradient entries: {gradient_size}.",
                    file=output,
                    flush=True,
                )
        return evaluation


class VmecProblem(FunctionProblem):
    """A :class:`FunctionProblem` backed by a VMEX equilibrium solve."""

    def __init__(self, *args: Any, input_from_x: Callable[[Array], Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._input_from_x = input_from_x

    @classmethod
    def from_tuples(
        cls,
        inp: Any,
        objective_terms: Sequence[tuple[Callable[..., Any], Any, float]],
        **kwargs: Any,
    ) -> "VmecProblem":
        """Build a VMEC least-squares problem from weighted objective tuples."""
        from .optimize import make_problem
        return make_problem(inp, objective_terms=objective_terms, problem_class=cls, **kwargs)

    @classmethod
    def from_loss(cls, inp: Any, loss: Callable[..., Any], **kwargs: Any) -> "VmecProblem":
        """Build a VMEC scalar problem from a traceable state/runtime loss."""
        from .optimize import make_problem
        return make_problem(inp, loss=loss, problem_class=cls, **kwargs)

    def input_from_x(self, x: Array) -> Any:
        """Return a new :class:`VmecInput` containing decision vector ``x``."""
        return self._input_from_x(self._x(x))


__all__ = ["Evaluation", "FunctionProblem", "VmecProblem"]
