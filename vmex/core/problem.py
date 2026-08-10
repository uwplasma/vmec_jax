"""Optimizer-neutral objective and derivative callables.

The classes in this module contain no optimization algorithms.  They expose
the small function contracts consumed by SciPy, JAXopt, Optax, and user code.
VMEC-specific construction is imported lazily so this module remains usable in
lightweight tests and does not introduce an import cycle with
``vmex.core.optimize``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    action: str,
    complete: str,
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
    print(f"{action}...", file=stream, flush=True)
    started = time.perf_counter()
    finished = Event()

    def heartbeat() -> None:
        while not finished.wait(interval):
            elapsed = time.perf_counter() - started
            print(f"  {elapsed:.1f} s elapsed.", file=stream, flush=True)

    reporter = Thread(target=heartbeat, name="vmex-progress", daemon=True)
    reporter.start()
    try:
        result = function()
    except Exception:
        elapsed = time.perf_counter() - started
        print(f"Failed after {elapsed:.1f} s.", file=stream, flush=True)
        raise
    finally:
        finished.set()
        reporter.join()
    elapsed = time.perf_counter() - started
    print(f"{complete} in {elapsed:.1f} s.", file=stream, flush=True)
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
        if self._residual_and_jac is not None:
            return self.residual_and_jac(x)[0]
        if self._residual is not None:
            return np.asarray(self._residual(self._x(x)), dtype=float).ravel()
        raise AttributeError("this problem does not provide residuals")

    def residual_jac(self, x: Array) -> np.ndarray:
        """Return the residual Jacobian."""
        if self._residual_and_jac is not None:
            return self.residual_and_jac(x)[1]
        if self._residual_jac is not None:
            jacobian = np.asarray(self._residual_jac(self._x(x)), dtype=float)
            if jacobian.shape[1:] != (self.x0.size,):
                raise ValueError(
                    "residual Jacobian must have one column per decision variable"
                )
            return jacobian
        raise AttributeError("this problem does not provide a residual Jacobian")

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

    def compile_residual_and_jacobian(
        self,
        x: Array | None = None,
        *,
        progress: bool = True,
        report_interval: float = 10.0,
        stream: Any = None,
    ) -> Evaluation:
        """Compile and cache the least-squares residual and Jacobian.

        This call is optional: an optimizer compiles on its first evaluation
        if it is omitted.  Calling it explicitly provides elapsed-time output
        during a potentially long first JAX compilation.  Later calls at the
        same ``x`` use the normal one-entry problem cache.
        """
        xh = self.x0.copy() if x is None else self._x(x).copy()

        def compile_callables() -> Evaluation:
            residual, jacobian = self.residual_and_jac(xh)
            return Evaluation(
                x=xh,
                value=0.5 * float(residual @ residual),
                gradient=jacobian.T @ residual,
                residual=residual,
                jacobian=jacobian,
            )

        return _run_with_progress(
            compile_callables,
            action="Compiling residual and Jacobian (first call may take a minute)",
            complete="Residual and Jacobian ready",
            progress=progress,
            report_interval=report_interval,
            stream=stream,
        )

    def compile_value_and_gradient(
        self,
        x: Array | None = None,
        *,
        progress: bool = True,
        report_interval: float = 10.0,
        stream: Any = None,
    ) -> Evaluation:
        """Compile and cache the scalar value and gradient.

        This optional call makes the first JAX compilation visible before
        BFGS, L-BFGS-B, Adam, or another gradient optimizer starts.
        """
        xh = self.x0.copy() if x is None else self._x(x).copy()

        def compile_callables() -> Evaluation:
            value, gradient = self.value_and_grad(xh)
            return Evaluation(x=xh, value=value, gradient=gradient)

        return _run_with_progress(
            compile_callables,
            action="Compiling value and gradient (first call may take a minute)",
            complete="Value and gradient ready",
            progress=progress,
            report_interval=report_interval,
            stream=stream,
        )


class VmecProblem(FunctionProblem):
    """A :class:`FunctionProblem` backed by a VMEX equilibrium solve."""

    def __init__(
        self,
        *args: Any,
        input_from_x: Callable[[Array], Any],
        x_from_input: Callable[[Any], Array],
        equilibrium_from_x: Callable[[Array], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._input_from_x = input_from_x
        self._x_from_input = x_from_input
        self._equilibrium_from_x = equilibrium_from_x

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

    def x_from_input(self, inp: Any) -> np.ndarray:
        """Return this problem's decision vector for ``inp``.

        This is the inverse of :meth:`input_from_x` for the boundary and any
        optional current degrees of freedom selected when the problem was
        constructed.  It is the normal continuation-stage starting vector.
        """
        x = np.asarray(self._x_from_input(inp), dtype=float)
        if x.shape != self.x0.shape:
            raise ValueError(
                f"input produced decision-vector shape {x.shape}, "
                f"expected {self.x0.shape}"
            )
        return x

    def equilibrium_from_x(self, x: Array) -> Any:
        """Return the converged equilibrium evaluated at ``x``.

        Implicit problems reuse the accepted optimizer state instead of
        cold-solving the optimized boundary again.  This matters for strongly
        shaped boundaries whose cold magnetic-axis guess may have a
        sign-changing initial Jacobian.
        """
        if self._equilibrium_from_x is None:
            raise AttributeError("this problem does not provide equilibria")
        return self._equilibrium_from_x(self._x(x))

    def evaluate(self, x: Array, *, derivatives: bool = True) -> Evaluation:
        """Evaluate and attach VMEC solve/adjoint status diagnostics."""
        evaluation = super().evaluate(x, derivatives=derivatives)
        cfg = self.metadata.get("config")
        if cfg is None:
            return evaluation
        from . import implicit as imp

        # A cached value/Jacobian can be revisited after an unrelated rejected
        # trial.  Confirm that the requested point still maps to the cached
        # converged equilibrium before consulting the process-local last-error
        # slot, otherwise that older failure would incorrectly mark this
        # successful evaluation as failed.
        try:
            self.equilibrium_from_x(evaluation.x)
        except (AttributeError, RuntimeError):
            pass
        else:
            imp._LAST_STATUS_ERROR.pop(cfg, None)

        diagnostics = dict(evaluation.diagnostics)
        stats = imp._SOLVE_STATS.get(cfg)
        if stats is not None:
            diagnostics["solve_stats"] = dict(stats)
        holder = self.metadata.get("holder", {})
        diagnostics["failed_trials"] = int(holder.get("failed_trials", 0))
        diagnostics["derivative_fallbacks"] = int(
            holder.get("derivative_fallbacks", 0)
        )
        error = imp._LAST_STATUS_ERROR.get(cfg)
        if error is None:
            return replace(evaluation, diagnostics=diagnostics)
        diagnostics["exception_type"] = type(error).__name__
        return replace(
            evaluation,
            status="failed_solve",
            message=str(error),
            diagnostics=diagnostics,
        )


__all__ = ["Evaluation", "FunctionProblem", "VmecProblem"]
