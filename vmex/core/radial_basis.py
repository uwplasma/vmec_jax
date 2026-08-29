"""Local B-spline bases and magnetic-axis regularity helpers.

The basis metadata is static NumPy data while coefficient evaluation and
transfer are differentiable JAX operations.  Clamped and periodic bases share
one implementation for the odd degrees used by high-order force balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

Array = Any
SUPPORTED_DEGREES = (3, 5, 7)


def _validate_degree(degree: int) -> int:
    degree = int(degree)
    if degree not in SUPPORTED_DEGREES:
        supported = ", ".join(map(str, SUPPORTED_DEGREES))
        raise ValueError(f"spline degree must be one of {supported}")
    return degree


def _validate_breakpoints(breakpoints: Array) -> np.ndarray:
    values = np.asarray(breakpoints, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("breakpoints must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
        raise ValueError("breakpoints must be finite and strictly increasing")
    return values


def _basis_levels(knots: Array, points: Array, degree: int) -> list[Array]:
    """Return Cox-de Boor basis levels from degree zero through ``degree``."""

    knots = jnp.asarray(knots)
    points = jnp.asarray(points).reshape(-1)
    evaluation_points = jnp.where(points == knots[-1], jnp.nextafter(knots[-1], -jnp.inf), points)
    level = ((evaluation_points[:, None] >= knots[:-1]) & (evaluation_points[:, None] < knots[1:])).astype(points.dtype)
    levels = [level]
    for order in range(1, degree + 1):
        count = knots.size - order - 1
        left_denominator = knots[order : order + count] - knots[:count]
        right_denominator = knots[order + 1 : order + count + 1] - knots[1 : count + 1]
        left = jnp.where(
            left_denominator > 0.0,
            (evaluation_points[:, None] - knots[:count]) / jnp.where(left_denominator > 0.0, left_denominator, 1.0),
            0.0,
        )
        right = jnp.where(
            right_denominator > 0.0,
            (knots[order + 1 : order + count + 1] - evaluation_points[:, None])
            / jnp.where(right_denominator > 0.0, right_denominator, 1.0),
            0.0,
        )
        level = left * level[:, :count] + right * level[:, 1 : count + 1]
        levels.append(level)
    endpoint = points == knots[-1]
    levels[-1] = levels[-1].at[:, -1].set(jnp.where(endpoint, 1.0, levels[-1][:, -1]))
    levels[-1] = levels[-1].at[:, :-1].set(jnp.where(endpoint[:, None], 0.0, levels[-1][:, :-1]))
    return levels


def _basis_matrix(knots: Array, points: Array, degree: int, derivative: int = 0) -> Array:
    levels = _basis_levels(knots, points, degree)
    if derivative == 0:
        return levels[degree]
    if derivative not in (1, 2):
        raise ValueError("only derivatives 0, 1, and 2 are supported")

    knots = jnp.asarray(knots)
    count = knots.size - degree - 1
    lower = levels[degree - 1]
    left_denominator = knots[degree : degree + count] - knots[:count]
    right_denominator = knots[degree + 1 : degree + count + 1] - knots[1 : count + 1]
    left_scale = jnp.where(left_denominator > 0.0, degree / left_denominator, 0.0)
    right_scale = jnp.where(right_denominator > 0.0, degree / right_denominator, 0.0)
    first = left_scale * lower[:, :count] - right_scale * lower[:, 1 : count + 1]
    if derivative == 1:
        return first

    lower_count = count + 1
    lower_degree = degree - 1
    base = levels[degree - 2]
    lower_left_denominator = knots[lower_degree : lower_degree + lower_count] - knots[:lower_count]
    lower_right_denominator = knots[lower_degree + 1 : lower_degree + lower_count + 1] - knots[1 : lower_count + 1]
    lower_first = (
        jnp.where(
            lower_left_denominator > 0.0,
            lower_degree / lower_left_denominator,
            0.0,
        )
        * base[:, :lower_count]
        - jnp.where(
            lower_right_denominator > 0.0,
            lower_degree / lower_right_denominator,
            0.0,
        )
        * base[:, 1 : lower_count + 1]
    )
    return left_scale * lower_first[:, :count] - right_scale * lower_first[:, 1 : count + 1]


def _span_quadrature(breakpoints: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 1:
        raise ValueError("quadrature order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    centers = 0.5 * (breakpoints[:-1] + breakpoints[1:])
    scales = 0.5 * np.diff(breakpoints)
    return (
        (centers[:, None] + scales[:, None] * nodes[None, :]).reshape(-1),
        (scales[:, None] * weights[None, :]).reshape(-1),
    )


def _apply_matrix(matrix: Array, values: Array, axis: int) -> Array:
    values = jnp.asarray(values)
    axis %= values.ndim
    moved = jnp.moveaxis(values, axis, 0)
    result = jnp.tensordot(jnp.asarray(matrix), moved, axes=((1,), (0,)))
    return jnp.moveaxis(result, 0, axis)


@dataclass(frozen=True, eq=False)
class BSplineBasis:
    """Static clamped or periodic local B-spline basis with JAX evaluation."""

    knots: np.ndarray
    breakpoints: np.ndarray
    periodic: bool
    degree: int
    size: int
    collocation_nodes: np.ndarray
    quadrature_nodes: np.ndarray
    quadrature_weights: np.ndarray

    @classmethod
    def clamped(
        cls,
        breakpoints: Array,
        *,
        degree: int = 3,
        quadrature_order: int | None = None,
    ) -> "BSplineBasis":
        """Build an endpoint-interpolating basis on arbitrary breakpoints."""

        degree = _validate_degree(degree)
        breaks = _validate_breakpoints(breakpoints)
        order = degree + 1 if quadrature_order is None else int(quadrature_order)
        knots = np.concatenate(
            (
                np.repeat(breaks[0], degree + 1),
                breaks[1:-1],
                np.repeat(breaks[-1], degree + 1),
            )
        )
        size = knots.size - degree - 1
        collocation = np.asarray([np.mean(knots[index + 1 : index + degree + 1]) for index in range(size)])
        quadrature_nodes, quadrature_weights = _span_quadrature(breaks, order)
        return cls(
            knots,
            breaks,
            False,
            degree,
            size,
            collocation,
            quadrature_nodes,
            quadrature_weights,
        )

    @classmethod
    def periodic_uniform(
        cls,
        size: int,
        domain: tuple[float, float] = (0.0, 2.0 * np.pi),
        *,
        degree: int = 3,
        quadrature_order: int | None = None,
    ) -> "BSplineBasis":
        """Build folded uniform splines on a periodic interval."""

        degree = _validate_degree(degree)
        size = int(size)
        start, stop = map(float, domain)
        if size < degree + 1:
            raise ValueError(f"periodic degree-{degree} basis requires size >= {degree + 1}")
        if stop <= start:
            raise ValueError("periodic domain must have stop > start")
        order = degree + 1 if quadrature_order is None else int(quadrature_order)
        spacing = (stop - start) / size
        knots = start + spacing * np.arange(-degree, size + degree + 1)
        breaks = np.linspace(start, stop, size + 1)
        quadrature_nodes, quadrature_weights = _span_quadrature(breaks, order)
        return cls(
            knots,
            breaks,
            True,
            degree,
            size,
            breaks[:-1],
            quadrature_nodes,
            quadrature_weights,
        )

    @property
    def domain(self) -> tuple[float, float]:
        """Return the open or fundamental periodic interval."""

        return float(self.breakpoints[0]), float(self.breakpoints[-1])

    def basis_matrix(self, points: Array, *, derivative: int = 0) -> Array:
        """Evaluate basis values or derivatives at one-dimensional ``points``."""

        points = jnp.asarray(points)
        original_shape = points.shape
        evaluation_points = points.reshape(-1)
        if self.periodic:
            start, stop = self.domain
            evaluation_points = jnp.mod(evaluation_points - start, stop - start) + start
            raw = _basis_matrix(self.knots, evaluation_points, self.degree, derivative)
            matrix = jnp.zeros((evaluation_points.size, self.size), dtype=raw.dtype)
            for column in range(raw.shape[1]):
                matrix = matrix.at[:, (column - self.degree) % self.size].add(raw[:, column])
        else:
            matrix = _basis_matrix(self.knots, evaluation_points, self.degree, derivative)
        return matrix.reshape(original_shape + (self.size,))

    def evaluate(
        self,
        coefficients: Array,
        points: Array,
        *,
        derivative: int = 0,
        axis: int = -1,
    ) -> Array:
        """Evaluate spline coefficients along ``axis`` at arbitrary points."""

        coefficients = jnp.asarray(coefficients)
        if coefficients.shape[axis] != self.size:
            raise ValueError(f"coefficient axis has size {coefficients.shape[axis]}; expected {self.size}")
        moved = jnp.moveaxis(coefficients, axis, -1)
        values = jnp.tensordot(
            moved,
            self.basis_matrix(points, derivative=derivative),
            axes=((-1,), (-1,)),
        )
        point_axes = tuple(range(values.ndim - jnp.ndim(points), values.ndim))
        target = tuple(range(axis % coefficients.ndim, axis % coefficients.ndim + len(point_axes)))
        return jnp.moveaxis(values, point_axes, target) if point_axes else values

    def fit(self, values: Array, *, nodes: Array | None = None, axis: int = -1) -> Array:
        """Interpolate one independent sample per coefficient."""

        sample_nodes = self.collocation_nodes if nodes is None else jnp.asarray(nodes)
        matrix = self.basis_matrix(sample_nodes)
        if matrix.shape[0] != self.size:
            raise ValueError("fit requires exactly one independent sample per coefficient")
        values = jnp.asarray(values)
        moved = jnp.moveaxis(values, axis, 0)
        coefficients = jnp.linalg.solve(matrix, moved.reshape((self.size, -1)))
        coefficients = coefficients.reshape((self.size,) + moved.shape[1:])
        return jnp.moveaxis(coefficients, 0, axis)

    def integrate(self, coefficients: Array, *, axis: int = -1) -> Array:
        """Integrate a spline using per-span Gauss-Legendre quadrature."""

        values = self.evaluate(coefficients, self.quadrature_nodes, axis=axis)
        return jnp.tensordot(values, jnp.asarray(self.quadrature_weights), axes=((axis,), (0,)))

    def transfer_matrix_to(self, target: "BSplineBasis") -> np.ndarray:
        """Return the collocation transfer from this basis to ``target``."""

        if self.periodic != target.periodic or not np.allclose(self.domain, target.domain, rtol=0.0, atol=1.0e-14):
            raise ValueError("basis transfer requires matching topology and domain")
        nodes = target.collocation_nodes
        source_values = np.asarray(self.basis_matrix(nodes))
        target_values = np.asarray(target.basis_matrix(nodes))
        return np.linalg.solve(target_values, source_values)

    def transfer(self, coefficients: Array, target: "BSplineBasis", *, axis: int = -1) -> Array:
        """Transfer coefficients to ``target`` through its collocation grid."""

        if np.shape(coefficients)[axis] != self.size:
            raise ValueError("source coefficient axis has the wrong size")
        return _apply_matrix(self.transfer_matrix_to(target), coefficients, axis)

    def transfer_transpose(self, cotangent: Array, target: "BSplineBasis", *, axis: int = -1) -> Array:
        """Apply the transpose of :meth:`transfer` to target-space values."""

        if np.shape(cotangent)[axis] != target.size:
            raise ValueError("target cotangent axis has the wrong size")
        return _apply_matrix(self.transfer_matrix_to(target).T, cotangent, axis)

    def insert_knot(self, coefficients: Array, knot: float, *, axis: int = -1) -> tuple["BSplineBasis", Array]:
        """Insert one open knot exactly with the Boehm coefficient update."""

        if self.periodic:
            raise ValueError("use refine_periodic_uniform for a periodic basis")
        knot = float(knot)
        start, stop = self.domain
        if not start < knot < stop:
            raise ValueError("inserted knot must lie strictly inside the domain")
        if np.any(np.isclose(self.breakpoints, knot, rtol=0.0, atol=1.0e-14)):
            raise ValueError("inserted knot must be new; repeated-knot refinement is unsupported")
        span = int(np.searchsorted(self.knots, knot, side="right") - 1)
        multiplicity = int(np.count_nonzero(np.isclose(self.knots, knot, rtol=0.0, atol=1.0e-14)))
        values = jnp.moveaxis(jnp.asarray(coefficients), axis, 0)
        if values.shape[0] != self.size:
            raise ValueError(f"coefficient axis has size {values.shape[0]}; expected {self.size}")
        updated = jnp.zeros((self.size + 1,) + values.shape[1:], dtype=values.dtype)
        updated = updated.at[: span - self.degree + 1].set(values[: span - self.degree + 1])
        updated = updated.at[span - multiplicity + 1 :].set(values[span - multiplicity :])
        for index in range(span - self.degree + 1, span - multiplicity + 1):
            alpha = (knot - self.knots[index]) / (self.knots[index + self.degree] - self.knots[index])
            updated = updated.at[index].set(alpha * values[index] + (1.0 - alpha) * values[index - 1])
        new_breakpoints = np.sort(np.append(self.breakpoints, knot))
        order = self.quadrature_weights.size // (self.breakpoints.size - 1)
        refined = type(self).clamped(new_breakpoints, degree=self.degree, quadrature_order=order)
        return refined, jnp.moveaxis(updated, 0, axis)

    def refine_periodic_uniform(
        self, coefficients: Array, target_size: int, *, axis: int = -1
    ) -> tuple["BSplineBasis", Array]:
        """Exactly refine a periodic uniform basis by dyadic subdivision."""

        if not self.periodic:
            raise ValueError("uniform periodic refinement requires a periodic basis")
        target_size = int(target_size)
        if target_size < self.size or target_size % self.size:
            raise ValueError("target size must be a dyadic multiple of the source size")
        ratio = target_size // self.size
        if ratio & (ratio - 1):
            raise ValueError("target size must be a dyadic multiple of the source size")
        values = jnp.moveaxis(jnp.asarray(coefficients), axis, 0)
        if values.shape[0] != self.size:
            raise ValueError(f"coefficient axis has size {values.shape[0]}; expected {self.size}")
        order = self.quadrature_weights.size // self.size
        refined = type(self).periodic_uniform(
            target_size,
            self.domain,
            degree=self.degree,
            quadrature_order=order,
        )
        if self.degree == 3:
            # Preserve the established mirror-coordinate cubic subdivision
            # exactly, including its coefficient indexing convention.
            size = self.size
            while size < target_size:
                updated = jnp.empty((2 * size,) + values.shape[1:], dtype=values.dtype)
                updated = updated.at[0::2].set(
                    0.125 * jnp.roll(values, 2, axis=0) + 0.75 * jnp.roll(values, 1, axis=0) + 0.125 * values
                )
                updated = updated.at[1::2].set(0.5 * (jnp.roll(values, 1, axis=0) + values))
                values = updated
                size *= 2
            return refined, jnp.moveaxis(values, 0, axis)
        transferred = _apply_matrix(self.transfer_matrix_to(refined), values, axis=0)
        return refined, jnp.moveaxis(transferred, 0, axis)


# Existing mirror imports retain this name and its cubic-by-default behavior.
CubicBSplineBasis = BSplineBasis


def evaluate_regularized_mode(
    basis: BSplineBasis,
    coefficients: Array,
    s: Array,
    mode_m: int,
    *,
    derivative: int = 0,
) -> Array:
    r"""Evaluate ``rho^|m| q(s)`` and its first two ``rho`` derivatives.

    Coefficients occupy the final axis.  Factoring the regularity analytically
    keeps every axis limit finite, including the ``m=0`` second derivative.
    """

    if derivative not in (0, 1, 2):
        raise ValueError("only derivatives 0, 1, and 2 are supported")
    mode_m = abs(int(mode_m))
    s = jnp.asarray(s)
    rho = jnp.sqrt(s)

    def broadcast(factor: Array, values: Array) -> Array:
        return jnp.reshape(factor, (1,) * (values.ndim - factor.ndim) + factor.shape)

    q = basis.evaluate(coefficients, s)
    if derivative == 0:
        return q * broadcast(rho**mode_m, q)
    q_s = basis.evaluate(coefficients, s, derivative=1)
    first = 2.0 * q_s * broadcast(rho ** (mode_m + 1), q_s)
    if mode_m:
        first = first + mode_m * q * broadcast(rho ** (mode_m - 1), q)
    if derivative == 1:
        return first
    q_ss = basis.evaluate(coefficients, s, derivative=2)
    second = 2.0 * (2 * mode_m + 1) * q_s * broadcast(rho**mode_m, q_s)
    second = second + 4.0 * q_ss * broadcast(rho ** (mode_m + 2), q_ss)
    if mode_m >= 2:
        second = second + mode_m * (mode_m - 1) * q * broadcast(rho ** (mode_m - 2), q)
    return second


__all__ = [
    "BSplineBasis",
    "CubicBSplineBasis",
    "SUPPORTED_DEGREES",
    "evaluate_regularized_mode",
]
