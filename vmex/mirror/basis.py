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

from vmex.core.radial_basis import CubicBSplineBasis as CubicBSplineBasis

Array = Any


class AxialBasis(Protocol):
    """Small operator contract shared by Chebyshev and spline evaluation grids."""

    nodes: np.ndarray
    weights: np.ndarray

    @property
    def size(self) -> int:
        """Number of axial nodes."""
        ...

    def differentiate(self, values: Array, *, axis: int = -1) -> Array:
        """First axial derivative of nodal values along ``axis`` (traceable)."""
        ...

    def differentiate_transpose(self, values: Array, *, axis: int = -1) -> Array:
        """Transpose of :meth:`differentiate`, for adjoint assembly."""
        ...

    def differentiate_twice(self, values: Array, *, axis: int = -1) -> Array:
        """Second axial derivative of nodal values along ``axis``."""
        ...

    def integrate(self, values: Array, *, axis: int = -1) -> Array:
        """Quadrature over the axial coordinate; ``axis`` is contracted away."""
        ...

    def interpolation_matrix(self, target_nodes: Array) -> np.ndarray:
        """Static dense map from nodal values to values at ``target_nodes``."""
        ...


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


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MirrorConfig
