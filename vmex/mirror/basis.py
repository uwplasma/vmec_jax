"""Fourier-Chebyshev collocation for open mirror coordinates.

Public Chebyshev-Gauss-Lobatto nodes are always in increasing physical order
``[-1, ..., +1]``.  Node ordering is part of the API because silently mixing
cosine order with physical order corrupts interpolation and end conditions.

References
----------
Trefethen, *Spectral Methods in MATLAB*, ``cheb.m`` and ``clencurt.m``.
Boyd, *Chebyshev and Fourier Spectral Methods*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np

Array = Any
_DEGREE = 3


class AxialBasis(Protocol):
    """Small operator contract shared by Chebyshev and spline evaluation grids."""

    nodes: np.ndarray
    weights: np.ndarray

    @property
    def size(self) -> int: ...

    def differentiate(self, values: Array, *, axis: int = -1) -> Array: ...

    def differentiate_transpose(self, values: Array, *, axis: int = -1) -> Array: ...

    def differentiate_twice(self, values: Array, *, axis: int = -1) -> Array: ...

    def integrate(self, values: Array, *, axis: int = -1) -> Array: ...

    def interpolation_matrix(self, target_nodes: Array) -> np.ndarray: ...


def _require_nxi(nxi: int) -> int:
    nxi = int(nxi)
    if nxi < 2:
        raise ValueError("Chebyshev-Gauss-Lobatto grids require nxi >= 2")
    return nxi


def _cgl_nodes(nxi: int) -> np.ndarray:
    """Increasing CGL nodes with exact endpoints."""

    nxi = _require_nxi(nxi)
    degree = nxi - 1
    nodes = np.cos(np.pi * np.arange(nxi) / degree)[::-1].copy()
    nodes[0], nodes[-1] = -1.0, 1.0
    return nodes


def _cgl_derivative_matrix(nxi: int) -> np.ndarray:
    """Trefethen first-derivative matrix permuted to increasing order."""

    nxi = _require_nxi(nxi)
    degree = nxi - 1
    k = np.arange(nxi)
    nodes_descending = np.cos(np.pi * k / degree)
    endpoint_scale = np.ones(nxi)
    endpoint_scale[[0, -1]] = 2.0
    c = endpoint_scale * (-1.0) ** k
    delta = nodes_descending[:, None] - nodes_descending[None, :]
    derivative = (c[:, None] / c[None, :]) / (delta + np.eye(nxi))
    derivative -= np.diag(np.sum(derivative, axis=1))
    return derivative[::-1, ::-1].copy()


def _clenshaw_curtis_weights(nxi: int) -> np.ndarray:
    """Clenshaw-Curtis weights in increasing node order."""

    nxi = _require_nxi(nxi)
    degree = nxi - 1
    if degree == 1:
        return np.ones(2)
    theta = np.pi * np.arange(nxi) / degree
    weights = np.zeros(nxi)
    interior = np.arange(1, degree)
    values = np.ones(degree - 1)
    if degree % 2 == 0:
        weights[[0, -1]] = 1.0 / (degree * degree - 1.0)
        for mode in range(1, degree // 2):
            values -= 2.0 * np.cos(2.0 * mode * theta[interior]) / (4.0 * mode * mode - 1.0)
        values -= np.cos(degree * theta[interior]) / (degree * degree - 1.0)
    else:
        weights[[0, -1]] = 1.0 / (degree * degree)
        for mode in range(1, (degree + 1) // 2):
            values -= 2.0 * np.cos(2.0 * mode * theta[interior]) / (4.0 * mode * mode - 1.0)
    weights[interior] = 2.0 * values / degree
    return weights[::-1].copy()


def _interpolation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Barycentric interpolation from increasing CGL nodes."""

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("source and target nodes must be one-dimensional")
    weights = (-1.0) ** np.arange(source.size)
    weights[[0, -1]] *= 0.5
    difference = target[:, None] - source[None, :]
    exact = np.isclose(difference, 0.0, rtol=0.0, atol=8.0 * np.finfo(float).eps)
    safe_difference = np.where(exact, 1.0, difference)
    scaled = weights[None, :] / safe_difference
    matrix = scaled / np.sum(scaled, axis=1, keepdims=True)
    for row in np.flatnonzero(np.any(exact, axis=1)):
        matrix[row] = 0.0
        matrix[row, int(np.argmax(exact[row]))] = 1.0
    return matrix


@dataclass(frozen=True, eq=False)
class ChebyshevBasis:
    """Static CGL nodes, quadrature, and derivative operators."""

    nodes: np.ndarray
    weights: np.ndarray
    derivative_matrix: np.ndarray
    second_derivative_matrix: np.ndarray

    @classmethod
    def build(cls, nxi: int) -> "ChebyshevBasis":
        """Construct a float64 basis once per static axial resolution."""

        nodes = _cgl_nodes(nxi)
        derivative = _cgl_derivative_matrix(nxi)
        return cls(
            nodes=nodes,
            weights=_clenshaw_curtis_weights(nxi),
            derivative_matrix=derivative,
            second_derivative_matrix=derivative @ derivative,
        )

    @property
    def size(self) -> int:
        """Number of CGL nodes."""

        return int(self.nodes.size)

    def differentiate(self, values: Array, *, axis: int = -1) -> Array:
        """Differentiate nodal values along ``axis`` with JAX operations."""

        values = jnp.asarray(values)
        moved = jnp.moveaxis(values, axis, 0)
        result = jnp.tensordot(jnp.asarray(self.derivative_matrix), moved, axes=((1,), (0,)))
        return jnp.moveaxis(result, 0, axis)

    def differentiate_transpose(self, values: Array, *, axis: int = -1) -> Array:
        """Apply the transpose first-derivative matrix along ``axis``."""

        values = jnp.asarray(values)
        moved = jnp.moveaxis(values, axis, 0)
        result = jnp.tensordot(jnp.asarray(self.derivative_matrix).T, moved, axes=((1,), (0,)))
        return jnp.moveaxis(result, 0, axis)

    def differentiate_twice(self, values: Array, *, axis: int = -1) -> Array:
        """Apply the CGL second-derivative matrix."""

        values = jnp.asarray(values)
        moved = jnp.moveaxis(values, axis, 0)
        result = jnp.tensordot(jnp.asarray(self.second_derivative_matrix), moved, axes=((1,), (0,)))
        return jnp.moveaxis(result, 0, axis)

    def integrate(self, values: Array, *, axis: int = -1) -> Array:
        """Integrate nodal values over ``xi in [-1,1]``."""

        values = jnp.asarray(values)
        moved = jnp.moveaxis(values, axis, -1)
        return jnp.tensordot(moved, jnp.asarray(self.weights), axes=((-1,), (0,)))

    def interpolation_matrix(self, target_nodes: Array) -> np.ndarray:
        """Return a static barycentric interpolation matrix."""

        return _interpolation_matrix(self.nodes, np.asarray(target_nodes))

    def interpolate(self, values: Array, target_nodes: Array, *, axis: int = -1) -> Array:
        """Interpolate CGL values to arbitrary target nodes."""

        values = jnp.asarray(values)
        matrix = jnp.asarray(self.interpolation_matrix(target_nodes), dtype=values.dtype)
        moved = jnp.moveaxis(values, axis, 0)
        result = jnp.tensordot(matrix, moved, axes=((1,), (0,)))
        return jnp.moveaxis(result, 0, axis)


@dataclass(frozen=True, eq=False)
class ThetaBasis:
    """Uniform periodic theta grid and FFT derivative operators."""

    nodes: np.ndarray
    weights: np.ndarray
    mpol: int

    @classmethod
    def build(cls, ntheta: int, mpol: int) -> "ThetaBasis":
        """Construct a grid that resolves all modes through ``mpol``."""

        ntheta, mpol = int(ntheta), int(mpol)
        if ntheta < 1:
            raise ValueError("ntheta must be >= 1")
        if mpol < 0:
            raise ValueError("mpol must be >= 0")
        required = 2 * mpol + 1
        if ntheta != required:
            raise ValueError(
                f"theta collocation requires ntheta=2*mpol+1; got "
                f"mpol={mpol}, ntheta={ntheta}"
            )
        nodes = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        return cls(nodes=nodes, weights=np.full(ntheta, 2.0 * np.pi / ntheta), mpol=mpol)

    @property
    def size(self) -> int:
        """Number of theta nodes."""

        return int(self.nodes.size)

    def differentiate(self, values: Array, *, axis: int = -1) -> Array:
        """Differentiate periodic values spectrally."""

        values = jnp.asarray(values)
        axis = axis % values.ndim
        if self.size == 1:
            return jnp.zeros_like(values)
        modes = jnp.fft.fftfreq(self.size, d=1.0 / self.size)
        shape = [1] * values.ndim
        shape[axis] = self.size
        transformed = jnp.fft.fft(values, axis=axis)
        derivative = jnp.fft.ifft(1j * modes.reshape(shape) * transformed, axis=axis)
        return derivative if jnp.iscomplexobj(values) else derivative.real

    def differentiate_transpose(self, values: Array, *, axis: int = -1) -> Array:
        """Apply the transpose periodic derivative on the uniform grid."""

        return -self.differentiate(values, axis=axis)

    def integrate(self, values: Array, *, axis: int = -1) -> Array:
        """Integrate periodic nodal values over ``[0,2*pi)``."""

        values = jnp.asarray(values)
        moved = jnp.moveaxis(values, axis, -1)
        return jnp.tensordot(moved, jnp.asarray(self.weights), axes=((-1,), (0,)))

    def interpolate(self, values: Array, target_nodes: Array, *, axis: int = -1) -> Array:
        """Evaluate the resolved trigonometric interpolant at target angles."""

        values = jnp.asarray(values)
        axis %= values.ndim
        moved = jnp.moveaxis(values, axis, 0)
        if moved.shape[0] != self.size:
            raise ValueError(f"interpolation axis has size {moved.shape[0]}, expected {self.size}")
        modes = jnp.fft.fftfreq(self.size, d=1.0 / self.size)
        coefficients = jnp.fft.fft(moved, axis=0) / self.size
        matrix = jnp.exp(1j * jnp.asarray(target_nodes)[:, None] * modes[None])
        result = jnp.tensordot(matrix, coefficients, axes=((1,), (0,)))
        if not jnp.iscomplexobj(values):
            result = result.real
        return jnp.moveaxis(result, 0, axis)


@dataclass(frozen=True, eq=False)
class MirrorGrid:
    """Static radial, theta, and axial collocation data."""

    s: np.ndarray
    s_half: np.ndarray
    radial_weights: np.ndarray
    theta_basis: ThetaBasis
    axial_basis: AxialBasis
    z: np.ndarray
    dz_dxi: float

    @property
    def ns(self) -> int:
        """Return the number of full radial surfaces."""
        return int(self.s.size)

    @property
    def ntheta(self) -> int:
        """Return the number of poloidal collocation nodes."""
        return self.theta_basis.size

    @property
    def nxi(self) -> int:
        """Return the number of axial collocation nodes."""
        return self.axial_basis.size

    @property
    def theta(self) -> np.ndarray:
        """Return the periodic poloidal nodes."""
        return self.theta_basis.nodes

    @property
    def xi(self) -> np.ndarray:
        """Return the normalized Chebyshev axial nodes."""
        return self.axial_basis.nodes

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the ``(radial, poloidal, axial)`` grid shape."""
        return (self.ns, self.ntheta, self.nxi)


def build_mirror_grid(config: "MirrorConfig") -> MirrorGrid:
    """Build the static grid defined by :class:`MirrorConfig`."""

    resolution = config.resolution
    s = np.linspace(0.0, 1.0, resolution.ns)
    ds = 1.0 / (resolution.ns - 1)
    radial_weights = np.full(resolution.ns, ds)
    radial_weights[[0, -1]] *= 0.5
    axial = ChebyshevBasis.build(resolution.nxi)
    z_mid = 0.5 * (config.z_min + config.z_max)
    dz_dxi = 0.5 * (config.z_max - config.z_min)
    return MirrorGrid(
        s=s,
        s_half=0.5 * (s[:-1] + s[1:]),
        radial_weights=radial_weights,
        theta_basis=ThetaBasis.build(resolution.ntheta, resolution.mpol),
        axial_basis=axial,
        z=z_mid + dz_dxi * axial.nodes,
        dz_dxi=dz_dxi,
    )


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
    if degree > 0:
        endpoint = points == knots[-1]
        levels[-1] = levels[-1].at[:, -1].set(jnp.where(endpoint, 1.0, levels[-1][:, -1]))
        levels[-1] = levels[-1].at[:, :-1].set(jnp.where(endpoint[:, None], 0.0, levels[-1][:, :-1]))
    return levels


def _basis_matrix(knots: Array, points: Array, degree: int, derivative: int = 0) -> Array:
    levels = _basis_levels(knots, points, degree)
    if derivative == 0:
        return levels[degree]
    count = jnp.asarray(knots).size - degree - 1
    knots = jnp.asarray(knots)
    lower = levels[degree - 1]
    left_denominator = knots[degree : degree + count] - knots[:count]
    right_denominator = knots[degree + 1 : degree + count + 1] - knots[1 : count + 1]
    left_scale = jnp.where(left_denominator > 0.0, degree / left_denominator, 0.0)
    right_scale = jnp.where(right_denominator > 0.0, degree / right_denominator, 0.0)
    first = left_scale * lower[:, :count] - right_scale * lower[:, 1 : count + 1]
    if derivative == 1:
        return first
    if derivative != 2:
        raise ValueError("only derivatives 0, 1, and 2 are supported")
    lower_count = count + 1
    degree_minus_one = degree - 1
    base = levels[degree - 2]
    lower_left_denominator = knots[degree_minus_one : degree_minus_one + lower_count] - knots[:lower_count]
    lower_right_denominator = (
        knots[degree_minus_one + 1 : degree_minus_one + lower_count + 1] - knots[1 : lower_count + 1]
    )
    lower_first = (
        jnp.where(lower_left_denominator > 0.0, degree_minus_one / lower_left_denominator, 0.0) * base[:, :lower_count]
        - jnp.where(lower_right_denominator > 0.0, degree_minus_one / lower_right_denominator, 0.0)
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


@dataclass(frozen=True, eq=False)
class CubicBSplineBasis:
    """Static clamped or periodic cubic basis with JAX evaluation."""

    knots: np.ndarray
    breakpoints: np.ndarray
    periodic: bool
    size: int
    collocation_nodes: np.ndarray
    quadrature_nodes: np.ndarray
    quadrature_weights: np.ndarray

    @classmethod
    def clamped(cls, breakpoints: Array, *, quadrature_order: int = 4) -> "CubicBSplineBasis":
        """Build an open cubic basis with repeated endpoint knots."""

        breaks = _validate_breakpoints(breakpoints)
        knots = np.concatenate((np.repeat(breaks[0], _DEGREE + 1), breaks[1:-1], np.repeat(breaks[-1], _DEGREE + 1)))
        size = knots.size - _DEGREE - 1
        collocation = np.asarray([np.mean(knots[index + 1 : index + _DEGREE + 1]) for index in range(size)])
        quadrature_nodes, quadrature_weights = _span_quadrature(breaks, quadrature_order)
        return cls(knots, breaks, False, size, collocation, quadrature_nodes, quadrature_weights)

    @classmethod
    def periodic_uniform(
        cls,
        size: int,
        domain: tuple[float, float] = (0.0, 2.0 * np.pi),
        *,
        quadrature_order: int = 4,
    ) -> "CubicBSplineBasis":
        """Build folded uniform cubic splines on a periodic interval."""

        size = int(size)
        start, stop = map(float, domain)
        if size < _DEGREE + 1:
            raise ValueError("periodic cubic basis requires size >= 4")
        if stop <= start:
            raise ValueError("periodic domain must have stop > start")
        spacing = (stop - start) / size
        knots = start + spacing * np.arange(-_DEGREE, size + _DEGREE + 1)
        breaks = np.linspace(start, stop, size + 1)
        quadrature_nodes, quadrature_weights = _span_quadrature(breaks, quadrature_order)
        return cls(
            knots,
            breaks,
            True,
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
            raw = _basis_matrix(self.knots, evaluation_points, _DEGREE, derivative)
            matrix = jnp.zeros((evaluation_points.size, self.size), dtype=raw.dtype)
            for column in range(raw.shape[1]):
                matrix = matrix.at[:, (column - _DEGREE) % self.size].add(raw[:, column])
        else:
            matrix = _basis_matrix(self.knots, evaluation_points, _DEGREE, derivative)
        return matrix.reshape(original_shape + (self.size,))

    def evaluate(self, coefficients: Array, points: Array, *, derivative: int = 0, axis: int = -1) -> Array:
        """Evaluate spline coefficients along ``axis`` at arbitrary points."""

        coefficients = jnp.asarray(coefficients)
        if coefficients.shape[axis] != self.size:
            raise ValueError(f"coefficient axis has size {coefficients.shape[axis]}; expected {self.size}")
        moved = jnp.moveaxis(coefficients, axis, -1)
        values = jnp.tensordot(moved, self.basis_matrix(points, derivative=derivative), axes=((-1,), (-1,)))
        point_axes = tuple(range(values.ndim - jnp.ndim(points), values.ndim))
        target = tuple(range(axis % coefficients.ndim, axis % coefficients.ndim + len(point_axes)))
        return jnp.moveaxis(values, point_axes, target) if point_axes else values

    def fit(self, values: Array, *, nodes: Array | None = None, axis: int = -1) -> Array:
        """Interpolate nodal values to coefficients with a square collocation solve."""

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

    def insert_knot(self, coefficients: Array, knot: float, *, axis: int = -1) -> tuple["CubicBSplineBasis", Array]:
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
        updated = updated.at[: span - _DEGREE + 1].set(values[: span - _DEGREE + 1])
        updated = updated.at[span - multiplicity + 1 :].set(values[span - multiplicity :])
        for index in range(span - _DEGREE + 1, span - multiplicity + 1):
            alpha = (knot - self.knots[index]) / (self.knots[index + _DEGREE] - self.knots[index])
            updated = updated.at[index].set(alpha * values[index] + (1.0 - alpha) * values[index - 1])
        new_breakpoints = np.sort(np.append(self.breakpoints, knot))
        refined = CubicBSplineBasis.clamped(
            new_breakpoints, quadrature_order=self.quadrature_weights.size // (self.breakpoints.size - 1)
        )
        return refined, jnp.moveaxis(updated, 0, axis)

    def refine_periodic_uniform(
        self,
        coefficients: Array,
        target_size: int,
        *,
        axis: int = -1,
    ) -> tuple["CubicBSplineBasis", Array]:
        """Exactly refine a uniform periodic cubic spline by dyadic subdivision."""

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
        size = self.size
        while size < target_size:
            updated = jnp.empty((2 * size,) + values.shape[1:], dtype=values.dtype)
            updated = updated.at[0::2].set(
                0.125 * jnp.roll(values, 2, axis=0)
                + 0.75 * jnp.roll(values, 1, axis=0)
                + 0.125 * values
            )
            updated = updated.at[1::2].set(
                0.5 * (jnp.roll(values, 1, axis=0) + values)
            )
            values = updated
            size *= 2
        order = self.quadrature_weights.size // self.size
        refined = CubicBSplineBasis.periodic_uniform(
            target_size,
            self.domain,
            quadrature_order=order,
        )
        return refined, jnp.moveaxis(values, 0, axis)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MirrorConfig
