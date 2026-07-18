"""Closed-surface quadrature and Laplace solver for the unbounded mirror vacuum.

The plasma side wall alone is not a closed boundary: the two axial cuts must
be filled by disks before a free-space boundary integral is well posed. The
cuts carry magnetic through-flux and are not plasma-vacuum interfaces; the
disks only close the Green integration surface and do not impose ``B.n=0`` or
pressure balance. This module builds that geometric adapter together with the
reduced-Neumann Laplace boundary-integral solver that runs on it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = Any


@dataclass(frozen=True)
class ClosedMirrorSurface:
    """Lateral wall and end-disk quadrature with outward weighted normals.

    ``weighted_normals`` are ``n dA`` including the tensor-product quadrature
    weights.  Keeping this quantity directly avoids unit-normal divisions at
    the disk centers and is the natural source measure for boundary integrals.
    """

    lateral_xyz: Array
    lateral_weighted_normals: Array
    lower_cap_xyz: Array
    lower_cap_weighted_normals: Array
    upper_cap_xyz: Array
    upper_cap_weighted_normals: Array
    collocation_xyz: Array
    collocation_normals: Array
    quadrature_to_collocation: Array
    collocation_to_reduced: Array
    triangles: Array
    n_reduced: int
    reduced_representatives: tuple[int, ...]
    triangle_connectivity: tuple[tuple[int, int, int], ...]

    @property
    def xyz(self) -> Array:
        """All quadrature nodes as a flat ``(n, 3)`` array."""

        return jnp.concatenate(
            [
                self.lateral_xyz.reshape(-1, 3),
                self.lower_cap_xyz.reshape(-1, 3),
                self.upper_cap_xyz.reshape(-1, 3),
            ],
            axis=0,
        )

    @property
    def weighted_normals(self) -> Array:
        """Outward ``n dA`` vectors in the same ordering as :attr:`xyz`."""

        return jnp.concatenate(
            [
                self.lateral_weighted_normals.reshape(-1, 3),
                self.lower_cap_weighted_normals.reshape(-1, 3),
                self.upper_cap_weighted_normals.reshape(-1, 3),
            ],
            axis=0,
        )

    @property
    def quadrature_weights(self) -> Array:
        """Scalar surface-area weights ``dA``."""

        return jnp.linalg.norm(self.weighted_normals, axis=-1)

    @property
    def normals(self) -> Array:
        """Outward unit normals at all quadrature nodes."""

        weights = self.quadrature_weights
        return self.weighted_normals / weights[:, None]

    @property
    def area(self) -> Array:
        """Total closed-surface area."""

        return jnp.sum(self.quadrature_weights)

    @property
    def volume(self) -> Array:
        """Enclosed volume from ``integral(x dot n) / 3``."""

        return jnp.sum(self.xyz * self.weighted_normals) / 3.0

    def expand_collocation_values(self, values: Array) -> Array:
        """Copy unique boundary values onto all quadrature nodes."""

        values = jnp.asarray(values)
        expected = (self.collocation_xyz.shape[0],)
        if values.shape != expected:
            raise ValueError(f"collocation values shape {values.shape} must be {expected}")
        return values[self.quadrature_to_collocation]

    @property
    def reduced_size(self) -> int:
        """Number of independent density values after symmetry reduction."""

        return self.n_reduced

    def expand_reduced_values(self, values: Array) -> Array:
        """Expand reduced density values onto unique collocation nodes."""

        values = jnp.asarray(values)
        expected = (self.reduced_size,)
        if values.shape != expected:
            raise ValueError(f"reduced values shape {values.shape} must be {expected}")
        return values[self.collocation_to_reduced]

    def reduce_collocation_values(self, values: Array) -> Array:
        """Average collocation values over each symmetry orbit."""

        values = jnp.asarray(values)
        expected = (self.collocation_xyz.shape[0],)
        if values.shape != expected:
            raise ValueError(f"collocation values shape {values.shape} must be {expected}")
        indices = jnp.asarray(self.collocation_to_reduced)
        totals = jnp.zeros(self.reduced_size, dtype=values.dtype).at[indices].add(values)
        counts = jnp.zeros(self.reduced_size, dtype=values.dtype).at[indices].add(1.0)
        return totals / counts

    @property
    def triangle_xyz(self) -> Array:
        """Linear-panel vertices with shape ``(ntriangle, 3, 3)``."""

        return self.collocation_xyz[self.triangles]

    @property
    def mesh_area(self) -> Array:
        """Area of the piecewise-linear closed panel mesh."""

        vertices = self.triangle_xyz
        cross = jnp.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        return 0.5 * jnp.sum(jnp.linalg.norm(cross, axis=1))

    @property
    def mesh_volume(self) -> Array:
        """Signed volume of the outward-oriented panel mesh."""

        vertices = self.triangle_xyz
        return jnp.sum(jnp.einsum("ij,ij->i", vertices[:, 0], jnp.cross(vertices[:, 1], vertices[:, 2]))) / 6.0


jax.tree_util.register_dataclass(
    ClosedMirrorSurface,
    data_fields=[
        field.name
        for field in fields(ClosedMirrorSurface)
        if field.name not in {"n_reduced", "reduced_representatives", "triangle_connectivity"}
    ],
    meta_fields=["n_reduced", "reduced_representatives", "triangle_connectivity"],
)


def build_closed_mirror_surface(
    boundary: "MirrorBoundary",
    grid: "MirrorGrid",
    *,
    axisymmetric_ntheta: int = 16,
    cap_rim_grade: float = 1.0,
) -> ClosedMirrorSurface:
    """Close a star-shaped mirror LCFS with disks on both axial cuts.

    The side wall uses ``(theta, xi)`` and each cap uses the regular disk
    coordinate ``r = rho(s) a(theta)``. ``cap_rim_grade > 1`` clusters panels
    at the sharp rim. The area coordinate remains ``rho^2``, giving the regular
    element ``a(theta)^2 d(rho^2) dtheta / 2`` at the center.
    """

    radius = jnp.asarray(boundary.radius_scale)
    expected = (grid.ntheta, grid.nxi)
    if radius.shape != expected:
        raise ValueError(f"boundary shape {radius.shape} does not match {expected}")

    axisymmetric = grid.ntheta == 1
    if axisymmetric:
        # The equilibrium stores no redundant angular samples in axisymmetry,
        # but Cartesian surface moments still require an angular quadrature.
        axisymmetric_ntheta = int(axisymmetric_ntheta)
        if axisymmetric_ntheta < 4:
            raise ValueError("axisymmetric_ntheta must be at least 4")
        theta = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, axisymmetric_ntheta, endpoint=False))
        theta_weights_1d = jnp.full(theta.shape, 2.0 * jnp.pi / theta.size)
        radius = jnp.broadcast_to(radius, (theta.size, grid.nxi))
        d_radius_dtheta = jnp.zeros_like(radius)
    else:
        theta = jnp.asarray(grid.theta)
        theta_weights_1d = jnp.asarray(grid.theta_basis.weights)
        d_radius_dtheta = grid.theta_basis.differentiate(radius, axis=0)
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    z = jnp.asarray(grid.z)
    dz_dxi = float(grid.dz_dxi)
    d_radius_dxi = grid.axial_basis.differentiate(radius, axis=1)

    lateral_xyz = jnp.stack(
        [
            radius * cosine[:, None],
            radius * sine[:, None],
            jnp.broadcast_to(z[None, :], radius.shape),
        ],
        axis=-1,
    )
    lateral_area_vectors = jnp.stack(
        [
            radius * dz_dxi * cosine[:, None],
            radius * dz_dxi * sine[:, None],
            -radius * d_radius_dxi,
        ],
        axis=-1,
    )
    lateral_area_vectors = lateral_area_vectors.at[..., 0].add(d_radius_dtheta * dz_dxi * sine[:, None])
    lateral_area_vectors = lateral_area_vectors.at[..., 1].add(-d_radius_dtheta * dz_dxi * cosine[:, None])
    lateral_weights = theta_weights_1d[:, None] * jnp.asarray(grid.axial_basis.weights)[None, :]
    lateral_weighted_normals = lateral_area_vectors * lateral_weights[..., None]

    cap_rim_grade = float(cap_rim_grade)
    if not np.isfinite(cap_rim_grade) or cap_rim_grade < 1.0:
        raise ValueError("cap_rim_grade must be finite and at least 1")
    base_radius_nodes = np.sqrt(np.asarray(grid.s))
    cap_radius_nodes = 1.0 - (1.0 - base_radius_nodes) ** cap_rim_grade
    cap_area_nodes = cap_radius_nodes**2
    cap_radial_weights = np.empty_like(cap_area_nodes)
    cap_radial_weights[0] = 0.5 * (cap_area_nodes[1] - cap_area_nodes[0])
    cap_radial_weights[-1] = 0.5 * (cap_area_nodes[-1] - cap_area_nodes[-2])
    cap_radial_weights[1:-1] = 0.5 * (cap_area_nodes[2:] - cap_area_nodes[:-2])
    cap_radius_nodes = jnp.asarray(cap_radius_nodes)[:, None]
    radial_weights = jnp.asarray(cap_radial_weights)[:, None]
    theta_weights = theta_weights_1d[None, :]

    def cap(endpoint: int, orientation: float) -> tuple[Array, Array]:
        cap_radius = cap_radius_nodes * radius[:, endpoint][None, :]
        cap_xyz = jnp.stack(
            [
                cap_radius * cosine[None, :],
                cap_radius * sine[None, :],
                jnp.full_like(cap_radius, z[endpoint]),
            ],
            axis=-1,
        )
        area = 0.5 * radius[:, endpoint] ** 2
        weighted_z = orientation * radial_weights * theta_weights * area[None, :]
        cap_weighted_normals = jnp.stack(
            [jnp.zeros_like(weighted_z), jnp.zeros_like(weighted_z), weighted_z],
            axis=-1,
        )
        return cap_xyz, cap_weighted_normals

    lower_xyz, lower_normals = cap(0, -1.0)
    upper_xyz, upper_normals = cap(-1, 1.0)

    ntheta, nxi = radius.shape
    lateral_map = np.arange(ntheta * nxi).reshape(ntheta, nxi)
    next_index = lateral_map.size

    def cap_map(endpoint: int) -> tuple[np.ndarray, int]:
        nonlocal next_index
        mapping = np.empty((grid.ns, ntheta), dtype=int)
        mapping[0] = next_index
        next_index += 1
        interior_size = max(0, grid.ns - 2) * ntheta
        mapping[1:-1] = np.arange(next_index, next_index + interior_size).reshape(grid.ns - 2, ntheta)
        next_index += interior_size
        mapping[-1] = lateral_map[:, endpoint]
        return mapping, next_index

    lower_map, _ = cap_map(0)
    upper_map, _ = cap_map(-1)
    quadrature_to_collocation = jnp.asarray(
        np.concatenate([lateral_map.reshape(-1), lower_map.reshape(-1), upper_map.reshape(-1)])
    )

    lateral_normals = lateral_area_vectors / jnp.linalg.norm(lateral_area_vectors, axis=-1, keepdims=True)
    lower_collocation_xyz = jnp.concatenate([lower_xyz[0, :1], lower_xyz[1:-1].reshape(-1, 3)], axis=0)
    upper_collocation_xyz = jnp.concatenate([upper_xyz[0, :1], upper_xyz[1:-1].reshape(-1, 3)], axis=0)
    lower_collocation_normals = jnp.zeros_like(lower_collocation_xyz).at[:, 2].set(-1.0)
    upper_collocation_normals = jnp.zeros_like(upper_collocation_xyz).at[:, 2].set(1.0)
    collocation_xyz = jnp.concatenate([lateral_xyz.reshape(-1, 3), lower_collocation_xyz, upper_collocation_xyz])
    collocation_normals = jnp.concatenate(
        [
            lateral_normals.reshape(-1, 3),
            lower_collocation_normals,
            upper_collocation_normals,
        ]
    )
    if axisymmetric:
        lateral_reduced = np.tile(np.arange(nxi), ntheta)
        next_reduced = nxi

        def cap_reduced() -> np.ndarray:
            nonlocal next_reduced
            center = np.asarray([next_reduced])
            next_reduced += 1
            rings = np.repeat(np.arange(next_reduced, next_reduced + grid.ns - 2), ntheta)
            next_reduced += grid.ns - 2
            return np.concatenate([center, rings])

        collocation_to_reduced = np.concatenate([lateral_reduced, cap_reduced(), cap_reduced()])
    else:
        collocation_to_reduced = np.arange(collocation_xyz.shape[0])
    triangle_array = closed_surface_triangles(lateral_map, lower_map, upper_map)
    triangles = jnp.asarray(triangle_array)
    return ClosedMirrorSurface(
        lateral_xyz=lateral_xyz,
        lateral_weighted_normals=lateral_weighted_normals,
        lower_cap_xyz=lower_xyz,
        lower_cap_weighted_normals=lower_normals,
        upper_cap_xyz=upper_xyz,
        upper_cap_weighted_normals=upper_normals,
        collocation_xyz=collocation_xyz,
        collocation_normals=collocation_normals,
        quadrature_to_collocation=quadrature_to_collocation,
        collocation_to_reduced=jnp.asarray(collocation_to_reduced),
        triangles=triangles,
        n_reduced=int(np.max(collocation_to_reduced)) + 1,
        reduced_representatives=tuple(int(index) for index in np.unique(collocation_to_reduced, return_index=True)[1]),
        triangle_connectivity=tuple(
            (int(triangle[0]), int(triangle[1]), int(triangle[2])) for triangle in triangle_array
        ),
    )


def _normalized_interpolation_weights(nodes: Array, barycentric: Array, targets: Array) -> Array:
    """Normalize barycentric weights, including targets at source nodes."""

    targets = jnp.asarray(targets)
    difference = targets[..., None] - nodes
    exact = jnp.abs(difference) <= 8.0 * jnp.finfo(nodes.dtype).eps
    scaled = barycentric / jnp.where(exact, 1.0, difference)
    weights = scaled / jnp.sum(scaled, axis=-1, keepdims=True)
    return jnp.where(jnp.any(exact, axis=-1, keepdims=True), exact, weights)


def _cgl_interpolation_weights(targets: Array, nxi: int) -> Array:
    """Barycentric weights from increasing CGL nodes to ``targets``."""

    targets = jnp.asarray(targets)
    degree = nxi - 1
    nodes = jnp.cos(jnp.pi * jnp.arange(nxi, dtype=targets.dtype) / degree)[::-1]
    barycentric = (-1.0) ** jnp.arange(nxi, dtype=targets.dtype)
    barycentric = barycentric.at[jnp.asarray([0, nxi - 1])].multiply(0.5)
    return _normalized_interpolation_weights(nodes, barycentric, targets)


def _periodic_interpolation_weights(targets: Array, ntheta: int) -> Array:
    """Real trigonometric interpolation weights on a uniform periodic grid."""

    targets = jnp.asarray(targets)
    nodes = 2.0 * jnp.pi * jnp.arange(ntheta, dtype=targets.dtype) / ntheta
    difference = targets[..., None] - nodes
    weights = jnp.ones_like(difference)
    for mode in range(1, (ntheta + 1) // 2):
        weights = weights + 2.0 * jnp.cos(mode * difference)
    if ntheta % 2 == 0:
        weights = weights + jnp.cos((ntheta // 2) * difference)
    return weights / ntheta


def closed_surface_triangles(lateral: np.ndarray, lower_cap: np.ndarray, upper_cap: np.ndarray) -> np.ndarray:
    """Triangulate periodic side quads and polar caps with outward orientation."""

    lateral = np.asarray(lateral, dtype=int)
    lower_cap = np.asarray(lower_cap, dtype=int)
    upper_cap = np.asarray(upper_cap, dtype=int)
    ntheta, nxi = lateral.shape
    triangles: list[tuple[int, int, int]] = []

    for j in range(ntheta):
        jp = (j + 1) % ntheta
        for k in range(nxi - 1):
            triangles.append((lateral[j, k], lateral[jp, k], lateral[jp, k + 1]))
            triangles.append((lateral[j, k], lateral[jp, k + 1], lateral[j, k + 1]))

    def add_cap(mapping: np.ndarray, *, upper: bool) -> None:
        center = int(mapping[0, 0])
        rings = mapping[1:]
        for j in range(ntheta):
            jp = (j + 1) % ntheta
            triangle = (center, int(rings[0, j]), int(rings[0, jp]))
            triangles.append(triangle if upper else triangle[::-1])
        for inner, outer in zip(rings[:-1], rings[1:], strict=True):
            for j in range(ntheta):
                jp = (j + 1) % ntheta
                first = (int(inner[j]), int(outer[j]), int(outer[jp]))
                second = (int(inner[j]), int(outer[jp]), int(inner[jp]))
                triangles.extend((first, second) if upper else (first[::-1], second[::-1]))

    add_cap(lower_cap, upper=False)
    add_cap(upper_cap, upper=True)
    return np.asarray(triangles, dtype=int)


@lru_cache(maxsize=None)
def _unit_gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    if order < 1:
        raise ValueError("quadrature order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _linear_density_samples(values: Array, *, order: int) -> Array:
    """Interpolate triangle-vertex values to Duffy quadrature nodes."""

    nodes, _ = _unit_gauss_legendre(order)
    u = jnp.asarray(nodes, dtype=values.dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=values.dtype)[None, None, :]
    return (1.0 - u) * values[:, 0, None, None] + u * (
        (1.0 - v) * values[:, 1, None, None] + v * values[:, 2, None, None]
    )


def _side_parameter_data(
    triangle_indices: Array,
    *,
    ntheta: int,
    nxi: int,
    order: int,
    dtype: Any,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Map Duffy nodes to side ``(theta, xi)`` coordinates and derivatives."""

    indices = jnp.asarray(triangle_indices)
    nodes, _ = _unit_gauss_legendre(order)
    u = jnp.asarray(nodes, dtype=dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=dtype)[None, None, :]
    shape = (indices.shape[0], order, order)
    barycentric = jnp.stack(
        [
            jnp.broadcast_to(1.0 - u, shape),
            jnp.broadcast_to(u * (1.0 - v), shape),
            jnp.broadcast_to(u * v, shape),
        ],
        axis=-1,
    )
    derivative_u = jnp.stack(
        [
            -jnp.ones(shape, dtype=dtype),
            jnp.broadcast_to(1.0 - v, shape),
            jnp.broadcast_to(v, shape),
        ],
        axis=-1,
    )
    derivative_v = jnp.stack(
        [
            jnp.zeros(shape, dtype=dtype),
            -jnp.broadcast_to(u, shape),
            jnp.broadcast_to(u, shape),
        ],
        axis=-1,
    )
    axial_nodes = jnp.cos(jnp.pi * jnp.arange(nxi, dtype=dtype) / (nxi - 1))[::-1]
    triangle_axial = axial_nodes[indices % nxi]
    axial = jnp.sum(barycentric * triangle_axial[:, None, None, :], axis=-1)
    axial_u = jnp.sum(derivative_u * triangle_axial[:, None, None, :], axis=-1)
    axial_v = jnp.sum(derivative_v * triangle_axial[:, None, None, :], axis=-1)

    angular = 2.0 * jnp.pi * (indices // nxi) / ntheta
    anchor = angular[:, :1]
    angular = anchor + (angular - anchor + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    theta = jnp.sum(barycentric * angular[:, None, None, :], axis=-1)
    theta_u = jnp.sum(derivative_u * angular[:, None, None, :], axis=-1)
    theta_v = jnp.sum(derivative_v * angular[:, None, None, :], axis=-1)
    return theta, axial, theta_u, theta_v, axial_u, axial_v


def _spectral_side_density_samples(
    triangle_indices: Array,
    lateral_values: Array,
    *,
    ntheta: int,
    nxi: int,
    order: int,
    axisymmetric: bool,
) -> Array:
    """Interpolate lateral nodal data spectrally within side triangles."""

    theta, axial, *_ = _side_parameter_data(
        triangle_indices,
        ntheta=ntheta,
        nxi=nxi,
        order=order,
        dtype=lateral_values.dtype,
    )
    axial_weights = _cgl_interpolation_weights(axial, nxi)
    scalar_input = lateral_values.ndim == 1
    values = lateral_values.reshape((-1, ntheta, nxi))
    if axisymmetric:
        samples = jnp.einsum("tqrk,ak->atqr", axial_weights, values[:, 0])
        return samples[0] if scalar_input else samples

    angular_weights = _periodic_interpolation_weights(theta, ntheta)
    samples = jnp.einsum("tqrj,ajk,tqrk->atqr", angular_weights, values, axial_weights)
    return samples[0] if scalar_input else samples


@partial(jax.jit, static_argnames=("order",))
def _triangle_layer_sum(
    target: Array,
    vertices: Array,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int,
) -> Array:
    """Sum single and double layers over triangles anchored for one target."""

    nodes, weights = _unit_gauss_legendre(order)
    dtype = vertices.dtype
    u = jnp.asarray(nodes, dtype=dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=dtype)[None, None, :]
    weights_2d = jnp.asarray(weights, dtype=dtype)[None, :, None] * jnp.asarray(weights, dtype=dtype)[None, None, :]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    ray = (1.0 - v)[..., None] * edge1[:, None, None, :] + (v[..., None] * edge2[:, None, None, :])
    source = vertices[:, 0, :][:, None, None, :] + u[..., None] * ray
    displacement = target[None, None, None, :] - source
    radius_squared = jnp.sum(displacement**2, axis=-1)
    inverse_radius = jax.lax.rsqrt(radius_squared)
    area_vectors = jnp.cross(edge1, edge2)
    area_scale = jnp.linalg.norm(area_vectors, axis=-1)
    normals = area_vectors / area_scale[:, None]
    jacobian = area_scale[:, None, None] * u

    def interpolate(values: Array) -> Array:
        if values.ndim == 3:
            return values
        return (1.0 - u) * values[:, 0, None, None] + u * (
            (1.0 - v) * values[:, 1, None, None] + v * values[:, 2, None, None]
        )

    single = interpolate(neumann) * jacobian * inverse_radius / (4.0 * jnp.pi)
    normal_displacement = jnp.einsum("ti,tqri->tqr", normals, displacement)
    double = -interpolate(dirichlet) * jacobian * normal_displacement * inverse_radius**3 / (4.0 * jnp.pi)
    return jnp.sum(weights_2d * (single + double))


def panel_green_boundary_residual(
    xyz: Array,
    triangles: Array,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
    target_indices: np.ndarray | None = None,
    lateral_shape: tuple[int, int] | None = None,
    spectral_side_density: bool = False,
    axisymmetric_side: bool = False,
) -> Array:
    """Evaluate ``S(q) + K(u-u_target)`` at all mesh vertices.

    Incident triangles are reordered so the collocation vertex is Duffy's
    singular vertex. The subtraction makes constants an exact nullspace and
    avoids assuming a smooth-surface jump coefficient at cap rims.
    """

    xyz = jnp.asarray(xyz)
    triangles_np = np.asarray(triangles, dtype=int)
    dirichlet = jnp.asarray(dirichlet)
    neumann = jnp.asarray(neumann)
    nvertices = int(xyz.shape[0])
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (n, 3)")
    if dirichlet.shape != (nvertices,) or neumann.shape != (nvertices,):
        raise ValueError("dirichlet and neumann must have one value per vertex")

    if target_indices is None:
        target_indices = np.arange(nvertices)
    else:
        target_indices = np.asarray(target_indices, dtype=int)
        if target_indices.ndim != 1 or np.any((target_indices < 0) | (target_indices >= nvertices)):
            raise ValueError("target_indices must select valid vertices")

    residual = []
    for target_index in target_indices:
        target_index = int(target_index)
        ordered = np.array(triangles_np, copy=True)
        rows, positions = np.nonzero(ordered == target_index)
        for row, position in zip(rows, positions, strict=True):
            ordered[row] = np.roll(ordered[row], -int(position))
        triangle_indices = jnp.asarray(ordered)
        triangle_dirichlet = dirichlet[triangle_indices] - dirichlet[target_index]
        triangle_neumann = neumann[triangle_indices]
        if spectral_side_density:
            if lateral_shape is None:
                raise ValueError("lateral_shape is required for spectral side density")
            ntheta, nxi = lateral_shape
            lateral_size = ntheta * nxi
            side_count = 2 * ntheta * (nxi - 1)
            side_samples = _spectral_side_density_samples(
                triangle_indices[:side_count],
                jnp.stack([dirichlet[:lateral_size], neumann[:lateral_size]]),
                ntheta=ntheta,
                nxi=nxi,
                order=order,
                axisymmetric=axisymmetric_side,
            )
            triangle_dirichlet = (
                _linear_density_samples(triangle_dirichlet, order=order)
                .at[:side_count]
                .set(side_samples[0] - dirichlet[target_index])
            )
            triangle_neumann = (
                _linear_density_samples(triangle_neumann, order=order).at[:side_count].set(side_samples[1])
            )
        residual.append(
            _triangle_layer_sum(
                xyz[target_index],
                xyz[triangle_indices],
                triangle_dirichlet,
                triangle_neumann,
                order=order,
            )
        )
    return jnp.stack(residual)


def _external_field_xyz(source: Any, points_xyz: Array) -> Array:
    """Evaluate an MGRID field or vectorized ``xyz -> B`` callable."""

    if hasattr(source, "b_cyl"):
        x, y, z = jnp.moveaxis(points_xyz, -1, 0)
        radius = jnp.sqrt(x**2 + y**2)
        phi = jnp.arctan2(y, x)
        b_r, b_phi, b_z = source.b_cyl(radius, phi, z)
        cosine, sine = jnp.cos(phi), jnp.sin(phi)
        return jnp.stack(
            (b_r * cosine - b_phi * sine, b_r * sine + b_phi * cosine, b_z),
            axis=-1,
        )
    if callable(source):
        field = jnp.asarray(source(points_xyz))
        if field.shape != points_xyz.shape:
            raise ValueError(f"external field returned shape {field.shape}; expected {points_xyz.shape}")
        return field
    raise TypeError("external field must provide b_cyl or be a vectorized xyz -> B callable")


def _balance_neumann_on_caps(surface: ClosedMirrorSurface, neumann: Array, lateral_size: int) -> Array:
    """Enforce discrete Neumann compatibility without changing LCFS data."""

    weights = _reduced_quadrature_weights(surface)
    cap_weights = weights.at[:lateral_size].set(0.0)
    correction = jnp.sum(weights * neumann) / jnp.sum(cap_weights)
    return neumann.at[lateral_size:].add(-correction)


def _reduced_quadrature_weights(surface: ClosedMirrorSurface) -> Array:
    quadrature_to_reduced = surface.collocation_to_reduced[surface.quadrature_to_collocation]
    return jnp.zeros(surface.reduced_size).at[quadrature_to_reduced].add(surface.quadrature_weights)


def _neumann_compatibility_error(surface: ClosedMirrorSurface, neumann: Array) -> Array:
    weights = _reduced_quadrature_weights(surface)
    net_flux = jnp.sum(weights * neumann)
    scale = surface.area * jnp.maximum(jnp.sqrt(jnp.mean(neumann**2)), jnp.finfo(neumann.dtype).tiny)
    return jnp.abs(net_flux) / scale


@dataclass(frozen=True)
class LaplaceNeumannResult:
    """Reduced boundary potential and diagnostics for a Neumann solve."""

    boundary_potential: Array
    residual: Array
    compatibility_error: Array
    raw_compatibility_error: Array
    condition_number: Array
    gauge_error: Array


@dataclass(frozen=True)
class ExteriorVacuum:
    """Solved free-space vacuum field on a mirror boundary."""

    surface: ClosedMirrorSurface
    neumann: Array
    neumann_result: LaplaceNeumannResult
    lateral_field_xyz: Array
    lateral_b_normal: Array


jax.tree_util.register_dataclass(
    LaplaceNeumannResult,
    data_fields=[field.name for field in fields(LaplaceNeumannResult)],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    ExteriorVacuum,
    data_fields=[field.name for field in fields(ExteriorVacuum)],
    meta_fields=[],
)


def axisymmetric_plasma_external_neumann(
    surface: ClosedMirrorSurface,
    plasma_field: "ContravariantField",
    plasma_grid: "MirrorGrid",
    external_field: Any,
) -> Array:
    """Build axisymmetric closed-surface Neumann data without redundant theta."""

    if plasma_grid.ntheta != 1:
        raise ValueError("axisymmetric Neumann data requires ntheta=1")
    expected_size = plasma_grid.nxi + 2 * (plasma_grid.ns - 1)
    if surface.reduced_size != expected_size:
        raise ValueError(f"surface reduced size {surface.reduced_size} must be {expected_size}")
    external_normal = jnp.sum(
        _external_field_xyz(external_field, surface.collocation_xyz) * surface.collocation_normals,
        axis=1,
    )
    neumann = -surface.reduce_collocation_values(external_normal)
    points = surface.collocation_xyz[jnp.asarray(surface.reduced_representatives)]

    nxi = plasma_grid.nxi
    cap_size = plasma_grid.ns - 1
    lower = slice(nxi, nxi + cap_size)
    upper = slice(nxi + cap_size, nxi + 2 * cap_size)
    lower_s = jnp.sum(points[lower, :2] ** 2, axis=1) / jnp.sum(surface.lateral_xyz[0, 0, :2] ** 2)
    upper_s = jnp.sum(points[upper, :2] ** 2, axis=1) / jnp.sum(surface.lateral_xyz[0, -1, :2] ** 2)
    lower_bz = jnp.interp(
        lower_s,
        jnp.asarray(plasma_grid.s),
        plasma_field.b_sup_xi[:, 0, 0] * float(plasma_grid.dz_dxi),
    )
    upper_bz = jnp.interp(
        upper_s,
        jnp.asarray(plasma_grid.s),
        plasma_field.b_sup_xi[:, 0, -1] * float(plasma_grid.dz_dxi),
    )
    neumann = neumann.at[lower].add(-lower_bz)
    return neumann.at[upper].add(upper_bz)


def axisymmetric_exterior_lateral_field(
    surface: ClosedMirrorSurface,
    boundary_potential: Array,
    neumann: Array,
    plasma_grid: "MirrorGrid",
    external_xyz: Array,
) -> Array:
    """Reconstruct total Cartesian field on the axisymmetric lateral boundary.

    The solved Neumann data supplies the correction normal component and the
    CGL derivative of boundary potential supplies its tangential component.
    Coordinates are returned at theta zero, one value per axial node.
    """

    boundary_potential = jnp.asarray(boundary_potential)
    neumann = jnp.asarray(neumann)
    external_xyz = jnp.asarray(external_xyz)
    expected = (surface.reduced_size,)
    if boundary_potential.shape != expected or neumann.shape != expected:
        raise ValueError(f"potential and neumann must have shape {expected}")
    if external_xyz.shape != (plasma_grid.nxi, 3):
        raise ValueError(f"external_xyz must have shape ({plasma_grid.nxi}, 3)")

    radius = jnp.linalg.norm(surface.lateral_xyz[0, :, :2], axis=1)
    radius_xi = plasma_grid.axial_basis.differentiate(radius)
    tangent = jnp.stack(
        [radius_xi, jnp.zeros_like(radius_xi), jnp.full_like(radius_xi, plasma_grid.dz_dxi)],
        axis=1,
    )
    arc_xi = jnp.linalg.norm(tangent, axis=1)
    tangent_hat = tangent / arc_xi[:, None]
    normal_hat = (
        jnp.stack(
            [
                jnp.full_like(radius_xi, plasma_grid.dz_dxi),
                jnp.zeros_like(radius_xi),
                -radius_xi,
            ],
            axis=1,
        )
        / arc_xi[:, None]
    )
    potential_xi = plasma_grid.axial_basis.differentiate(boundary_potential[: plasma_grid.nxi])
    correction = neumann[: plasma_grid.nxi, None] * normal_hat + (potential_xi / arc_xi)[:, None] * tangent_hat
    return external_xyz + correction


def solve_axisymmetric_exterior_vacuum(
    boundary: "MirrorBoundary",
    plasma_field: "ContravariantField",
    plasma_grid: "MirrorGrid",
    external_field: Any,
    *,
    axisymmetric_ntheta: int = 40,
    cap_rim_grade: float = 3.5,
    order: int = 8,
    spectral_side_density: bool = False,
) -> ExteriorVacuum:
    """Solve the unbounded vacuum field and reconstruct its lateral trace.

    The two end cuts are closed geometrically by graded disks. Their Neumann
    data continue the plasma axial field into free space, while the lateral
    data cancel the supplied external normal field. The caps are not material
    interfaces. The returned trace is sampled at theta zero on the plasma
    grid's axial nodes.
    """

    surface = build_closed_mirror_surface(
        boundary,
        plasma_grid,
        axisymmetric_ntheta=axisymmetric_ntheta,
        cap_rim_grade=cap_rim_grade,
    )
    neumann = axisymmetric_plasma_external_neumann(surface, plasma_field, plasma_grid, external_field)
    result = solve_reduced_exterior_laplace_neumann(
        surface,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
    )
    external = _external_field_xyz(external_field, surface.lateral_xyz[0])
    radius = jnp.linalg.norm(surface.lateral_xyz[0, :, :2], axis=1)
    radius_xi = plasma_grid.axial_basis.differentiate(radius)
    normal = jnp.stack(
        [
            jnp.full_like(radius_xi, plasma_grid.dz_dxi),
            jnp.zeros_like(radius_xi),
            -radius_xi,
        ],
        axis=1,
    )
    normal /= jnp.linalg.norm(normal, axis=1)[:, None]
    physical_neumann = neumann.at[: plasma_grid.nxi].set(-jnp.sum(external * normal, axis=1))
    lateral = axisymmetric_exterior_lateral_field(
        surface,
        result.boundary_potential,
        physical_neumann,
        plasma_grid,
        external,
    )
    return ExteriorVacuum(
        surface=surface,
        neumann=neumann,
        neumann_result=result,
        lateral_field_xyz=lateral,
        lateral_b_normal=jnp.sum(lateral * normal, axis=1),
    )


def laplace_reduced_green_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
) -> Array:
    """Evaluate the boundary identity in the surface's symmetry basis."""

    return panel_green_boundary_residual(
        surface.collocation_xyz,
        np.asarray(surface.triangle_connectivity),
        surface.expand_reduced_values(dirichlet),
        surface.expand_reduced_values(neumann),
        order=order,
        target_indices=np.asarray(surface.reduced_representatives),
        lateral_shape=surface.lateral_xyz.shape[:2],
        spectral_side_density=spectral_side_density,
        axisymmetric_side=surface.reduced_size < surface.collocation_xyz.shape[0],
    )


def _exterior_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
) -> Array:
    """Boundary residual for a harmonic potential decaying in the exterior."""

    dirichlet = jnp.asarray(dirichlet)
    return dirichlet + laplace_reduced_green_boundary_residual(
        surface,
        dirichlet,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
    )


def solve_reduced_exterior_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
) -> LaplaceNeumannResult:
    """Solve for the unique harmonic potential decaying in the exterior."""

    neumann = jnp.asarray(neumann)
    expected = (surface.reduced_size,)
    if neumann.shape != expected:
        raise ValueError(f"neumann shape {neumann.shape} must be {expected}")
    raw_compatibility_error = _neumann_compatibility_error(surface, neumann)
    full_lateral_size = int(np.prod(surface.lateral_xyz.shape[:2]))
    lateral_size = int(np.sum(np.asarray(surface.reduced_representatives) < full_lateral_size))
    neumann = _balance_neumann_on_caps(surface, neumann, lateral_size)
    zero = jnp.zeros_like(neumann)

    def dirichlet_operator(values: Array) -> Array:
        return _exterior_boundary_residual(
            surface,
            values,
            zero,
            order=order,
            spectral_side_density=spectral_side_density,
        )

    matrix = jax.jacfwd(dirichlet_operator)(zero)
    right_hand_side = -_exterior_boundary_residual(
        surface,
        zero,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
    )
    potential = jnp.linalg.solve(matrix, right_hand_side)
    residual = matrix @ potential - right_hand_side

    return LaplaceNeumannResult(
        boundary_potential=potential,
        residual=residual,
        compatibility_error=_neumann_compatibility_error(surface, neumann),
        raw_compatibility_error=raw_compatibility_error,
        condition_number=jnp.linalg.cond(matrix),
        gauge_error=jnp.asarray(0.0, dtype=matrix.dtype),
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .geometry import ContravariantField
    from .model import MirrorBoundary
