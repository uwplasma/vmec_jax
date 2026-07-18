"""Closed mirror surface geometry required by the free-space exterior solve."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror.exterior import (  # noqa: E402
    _spectral_side_density_samples,
    _unit_gauss_legendre,
)
from vmex.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
)
from vmex.mirror.exterior import build_closed_mirror_surface  # noqa: E402
from vmex.mirror.exterior import (  # noqa: E402
    axisymmetric_plasma_external_neumann,
    axisymmetric_exterior_lateral_field,
    laplace_reduced_green_boundary_residual,
    solve_reduced_exterior_laplace_neumann,
    solve_axisymmetric_exterior_vacuum,
)
from vmex.mirror.geometry import (  # noqa: E402
    contravariant_field,
    evaluate_geometry,
)


def _grid(*, ns: int = 17, mpol: int = 0, nxi: int = 25):
    return MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=mpol, nxi=nxi),
        z_min=-1.4,
        z_max=1.4,
    ).build_grid()


def _paraxial_field(points, *, center_field: float = 0.08, curvature: float = 0.02):
    """Curl-free, divergence-free external mirror field."""

    points = jnp.asarray(points)
    x, y, z = jnp.moveaxis(points, -1, 0)
    return jnp.stack(
        (
            -curvature * x * z,
            -curvature * y * z,
            center_field + curvature * (z**2 - 0.5 * (x**2 + y**2)),
        ),
        axis=-1,
    )


def _zero_field(points):
    return jnp.zeros_like(points)


def test_closed_cylinder_has_exact_area_volume_and_orientation() -> None:
    grid = _grid()
    radius = 0.37
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(radius, grid), grid)

    expected_area = 2.0 * np.pi * radius * 2.8 + 2.0 * np.pi * radius**2
    expected_volume = np.pi * radius**2 * 2.8
    np.testing.assert_allclose(surface.area, expected_area, rtol=3.0e-14)
    np.testing.assert_allclose(surface.volume, expected_volume, rtol=3.0e-14)
    np.testing.assert_allclose(jnp.sum(surface.weighted_normals, axis=0), 0.0, atol=3.0e-15)
    assert np.all(np.asarray(surface.lower_cap_weighted_normals[..., 2]) < 0.0)
    assert np.all(np.asarray(surface.upper_cap_weighted_normals[..., 2]) > 0.0)


def test_shaped_surface_satisfies_divergence_theorem_moments() -> None:
    grid = _grid(ns=21, mpol=4, nxi=33)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    radius = 0.31 * (1.0 + 0.08 * xi**2 + 0.04 * jnp.cos(2.0 * theta) * (1.0 - xi**2))
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(radius, grid), grid)

    net_normal = jnp.sum(surface.weighted_normals, axis=0)
    first_moment = jnp.einsum("ni,nj->ij", surface.xyz, surface.weighted_normals)
    np.testing.assert_allclose(net_normal, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(first_moment, np.eye(3) * float(surface.volume), rtol=3.0e-12, atol=3.0e-13)


def test_caps_share_the_lateral_end_rings() -> None:
    grid = _grid(ns=13, mpol=3, nxi=17)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.05 * jnp.cos(theta) * (1.0 + xi)), grid)
    surface = build_closed_mirror_surface(boundary, grid)

    np.testing.assert_allclose(surface.lower_cap_xyz[-1], surface.lateral_xyz[:, 0])
    np.testing.assert_allclose(surface.upper_cap_xyz[-1], surface.lateral_xyz[:, -1])


def test_collocation_map_identifies_cap_centers_and_rims() -> None:
    grid = _grid(ns=13, mpol=3, nxi=17)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    surface = build_closed_mirror_surface(
        MirrorBoundary.from_radius(0.3 * (1.0 + 0.04 * jnp.cos(2.0 * theta) * (1.0 - xi**2)), grid),
        grid,
    )
    nquadrature = surface.xyz.shape[0]
    ncollocation = surface.collocation_xyz.shape[0]
    expected = grid.ntheta * grid.nxi + 2 * (1 + (grid.ns - 2) * grid.ntheta)
    assert ncollocation == expected
    assert ncollocation < nquadrature
    assert np.unique(np.asarray(surface.collocation_xyz), axis=0).shape[0] == ncollocation

    values = surface.collocation_xyz[:, 0] + 2.0 * surface.collocation_xyz[:, 1] - 0.5 * surface.collocation_xyz[:, 2]
    expanded = surface.expand_collocation_values(values)
    expected_values = surface.xyz[:, 0] + 2.0 * surface.xyz[:, 1] - 0.5 * surface.xyz[:, 2]
    np.testing.assert_allclose(expanded, expected_values, atol=2.0e-15)


def test_axisymmetric_reduction_preserves_ring_values() -> None:
    grid = _grid(ns=13, nxi=17)
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(0.3, grid), grid, axisymmetric_ntheta=12)
    assert surface.reduced_size == grid.nxi + 2 * (grid.ns - 1)
    reduced = jnp.linspace(-0.4, 0.7, surface.reduced_size)
    expanded = surface.expand_reduced_values(reduced)
    np.testing.assert_allclose(surface.reduce_collocation_values(expanded), reduced)

    residual = laplace_reduced_green_boundary_residual(
        surface, jnp.ones(surface.reduced_size), jnp.zeros(surface.reduced_size)
    )
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-15)


def test_cap_grading_clusters_rim_without_changing_cylinder_integrals() -> None:
    grid = _grid(ns=13, nxi=17)
    boundary = MirrorBoundary.from_radius(0.3, grid)
    uniform = build_closed_mirror_surface(boundary, grid, cap_rim_grade=1.0)
    graded = build_closed_mirror_surface(boundary, grid, cap_rim_grade=2.5)
    radii = np.linalg.norm(np.asarray(graded.lower_cap_xyz[:, 0, :2]), axis=1)
    assert radii[-1] - radii[-2] < 0.1 * (radii[1] - radii[0])
    np.testing.assert_allclose(graded.area, uniform.area, rtol=3.0e-14)
    np.testing.assert_allclose(graded.volume, uniform.volume, rtol=3.0e-14)


def test_reduced_exterior_neumann_solve_recovers_decaying_dipole() -> None:
    boundary_errors = []
    lateral_errors = []
    for ns, nxi, ntheta in (
        (9, 13, 16),
        (13, 21, 24),
        (17, 29, 32),
        (21, 37, 40),
    ):
        grid = _grid(ns=ns, nxi=nxi)
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(0.37, grid),
            grid,
            axisymmetric_ntheta=ntheta,
            cap_rim_grade=3.5,
        )
        xyz = surface.collocation_xyz
        radius_squared = jnp.sum(xyz**2, axis=1)
        exact_full = xyz[:, 2] / radius_squared**1.5
        gradient_full = jnp.stack(
            [
                -3.0 * xyz[:, 2] * xyz[:, 0] / radius_squared**2.5,
                -3.0 * xyz[:, 2] * xyz[:, 1] / radius_squared**2.5,
                1.0 / radius_squared**1.5 - 3.0 * xyz[:, 2] ** 2 / radius_squared**2.5,
            ],
            axis=1,
        )
        exact = surface.reduce_collocation_values(exact_full)
        neumann = surface.reduce_collocation_values(jnp.sum(gradient_full * surface.collocation_normals, axis=1))
        result = solve_reduced_exterior_laplace_neumann(surface, neumann)
        boundary_errors.append(float(jnp.linalg.norm(result.boundary_potential - exact) / jnp.linalg.norm(exact)))
        lateral = axisymmetric_exterior_lateral_field(
            surface,
            result.boundary_potential,
            neumann,
            grid,
            jnp.zeros((grid.nxi, 3)),
        )
        lateral_xyz = surface.lateral_xyz[0]
        lateral_radius_squared = jnp.sum(lateral_xyz**2, axis=1)
        exact_lateral = jnp.stack(
            [
                -3.0 * lateral_xyz[:, 2] * lateral_xyz[:, 0] / lateral_radius_squared**2.5,
                jnp.zeros(grid.nxi),
                1.0 / lateral_radius_squared**1.5 - 3.0 * lateral_xyz[:, 2] ** 2 / lateral_radius_squared**2.5,
            ],
            axis=1,
        )
        lateral_error = jnp.linalg.norm(lateral - exact_lateral) / jnp.linalg.norm(exact_lateral)
        lateral_errors.append(float(lateral_error))
        assert float(result.compatibility_error) < 2.0e-14
        assert float(result.condition_number) < 5.0
        assert float(jnp.linalg.norm(result.residual)) < 3.0e-14

    assert all(fine < coarse for coarse, fine in zip(boundary_errors[:-1], boundary_errors[1:], strict=True))
    assert boundary_errors[-1] < 3.5e-2
    assert all(fine < coarse for coarse, fine in zip(lateral_errors[:-1], lateral_errors[1:], strict=True))
    assert lateral_errors[-1] < 4.0e-2


def test_spectral_side_density_improves_exterior_dipole() -> None:
    grid = _grid(ns=13, nxi=21)
    surface = build_closed_mirror_surface(
        MirrorBoundary.from_radius(0.37, grid),
        grid,
        axisymmetric_ntheta=24,
        cap_rim_grade=3.5,
    )
    xyz = surface.collocation_xyz
    radius_squared = jnp.sum(xyz**2, axis=1)
    exact = surface.reduce_collocation_values(xyz[:, 2] / radius_squared**1.5)
    gradient = jnp.stack(
        [
            -3.0 * xyz[:, 2] * xyz[:, 0] / radius_squared**2.5,
            -3.0 * xyz[:, 2] * xyz[:, 1] / radius_squared**2.5,
            1.0 / radius_squared**1.5 - 3.0 * xyz[:, 2] ** 2 / radius_squared**2.5,
        ],
        axis=1,
    )
    neumann = surface.reduce_collocation_values(jnp.sum(gradient * surface.collocation_normals, axis=1))
    errors = []
    for spectral in (False, True):
        result = solve_reduced_exterior_laplace_neumann(
            surface,
            neumann,
            spectral_side_density=spectral,
        )
        boundary_error = jnp.linalg.norm(result.boundary_potential - exact) / jnp.linalg.norm(exact)
        errors.append(float(boundary_error))
        assert float(result.condition_number) < 5.0
        assert float(jnp.linalg.norm(result.residual)) < 3.0e-14

    assert errors[1] < 0.3 * errors[0]


def test_panel_mesh_is_watertight_oriented_and_convergent() -> None:
    errors = []
    for ns, nxi, ntheta in ((9, 13, 8), (17, 25, 32)):
        grid = _grid(ns=ns, nxi=nxi)
        radius = 0.37
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(radius, grid),
            grid,
            axisymmetric_ntheta=ntheta,
        )
        triangles = np.asarray(surface.triangles)
        edges = np.sort(
            np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]),
            axis=1,
        )
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(edge_counts == 2)

        vertices = np.asarray(surface.triangle_xyz)
        doubled_area = np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=1,
        )
        assert np.all(doubled_area > 0.0)
        assert float(surface.mesh_volume) > 0.0

        expected_area = 2.0 * np.pi * radius * 2.8 + 2.0 * np.pi * radius**2
        expected_volume = np.pi * radius**2 * 2.8
        errors.append(
            max(
                abs(float(surface.mesh_area) - expected_area) / expected_area,
                abs(float(surface.mesh_volume) - expected_volume) / expected_volume,
            )
        )

    assert errors[1] < 0.3 * errors[0]
    assert errors[1] < 7.0e-3


@pytest.mark.parametrize("ntheta", [5, 7])
def test_spectral_side_density_reproduces_fourier_chebyshev_data(ntheta: int) -> None:
    grid = _grid(ns=7, mpol=(ntheta - 1) // 2, nxi=7)
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(0.3, grid), grid)
    nside = 2 * grid.ntheta * (grid.nxi - 1)
    triangles = np.asarray(surface.triangles[:nside])
    theta = np.asarray(grid.theta)[:, None]
    xi = np.asarray(grid.xi)[None, :]
    values = 1.0 + 0.2 * np.cos(theta) + 0.1 * np.sin(2.0 * theta) + 0.3 * xi**3
    order = 4
    samples = _spectral_side_density_samples(
        jnp.asarray(triangles),
        jnp.asarray(values).reshape(-1),
        ntheta=grid.ntheta,
        nxi=grid.nxi,
        order=order,
        axisymmetric=False,
    )

    nodes, _ = _unit_gauss_legendre(order)
    u, v = nodes[:, None], nodes[None, :]
    barycentric = np.stack(
        [
            np.broadcast_to(1.0 - u, (nside, order, order)),
            np.broadcast_to(u * (1.0 - v), (nside, order, order)),
            np.broadcast_to(u * v, (nside, order, order)),
        ],
        axis=-1,
    )
    theta_nodes = theta[:, 0][triangles // grid.nxi]
    anchor = theta_nodes[:, :1]
    theta_nodes = anchor + (theta_nodes - anchor + np.pi) % (2.0 * np.pi) - np.pi
    theta_targets = np.sum(barycentric * theta_nodes[:, None, None, :], axis=-1)
    triangle_xi = xi[0][triangles % grid.nxi]
    xi_targets = np.sum(barycentric * triangle_xi[:, None, None, :], axis=-1)
    expected = 1.0 + 0.2 * np.cos(theta_targets) + 0.1 * np.sin(2.0 * theta_targets) + 0.3 * xi_targets**3
    np.testing.assert_allclose(samples, expected, rtol=2.0e-13, atol=2.0e-13)
    derivative = jax.grad(
        lambda data: jnp.sum(
            _spectral_side_density_samples(
                jnp.asarray(triangles),
                data,
                ntheta=grid.ntheta,
                nxi=grid.nxi,
                order=order,
                axisymmetric=False,
            )
            ** 2
        )
    )(jnp.asarray(values).reshape(-1))
    assert np.all(np.isfinite(np.asarray(derivative)))


def test_closed_surface_volume_is_differentiable() -> None:
    grid = _grid(ns=15, nxi=21)

    def volume(radius):
        boundary = MirrorBoundary.from_radius(radius, grid)
        return build_closed_mirror_surface(boundary, grid).volume

    radius = 0.29
    derivative = jax.grad(volume)(radius)
    expected = 2.0 * np.pi * radius * 2.8
    np.testing.assert_allclose(derivative, expected, rtol=4.0e-13)
    jax.make_jaxpr(volume)(radius)


def test_plasma_external_neumann_adapter_matches_uniform_field_data() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=13, mpol=0, nxi=21),
        z_min=-0.6,
        z_max=0.6,
    ).build_grid()
    radius = 0.3
    boundary = MirrorBoundary.from_radius(radius, grid)
    state = MirrorState.from_boundary(boundary, grid)
    geometry = evaluate_geometry(state, grid)
    target_bz = 0.08
    plasma_field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=0.5 * target_bz * radius**2,
    )
    surface = build_closed_mirror_surface(boundary, grid, axisymmetric_ntheta=24, cap_rim_grade=3.5)
    adapted = axisymmetric_plasma_external_neumann(
        surface, plasma_field, grid, _paraxial_field
    )
    mapping = np.asarray(surface.collocation_to_reduced)
    _, representatives = np.unique(mapping, return_index=True)
    points = surface.collocation_xyz[jnp.asarray(representatives)]
    normals = surface.collocation_normals[jnp.asarray(representatives)]
    expected = jnp.sum(
        (jnp.asarray([0.0, 0.0, target_bz]) - _paraxial_field(points)) * normals,
        axis=1,
    )

    np.testing.assert_allclose(adapted, expected, rtol=3.0e-13, atol=3.0e-14)

    vacuum = solve_axisymmetric_exterior_vacuum(
        boundary,
        plasma_field,
        grid,
        _paraxial_field,
        axisymmetric_ntheta=24,
    )
    np.testing.assert_allclose(vacuum.neumann, adapted, rtol=3.0e-13, atol=3.0e-14)
    assert vacuum.lateral_field_xyz.shape == (grid.nxi, 3)
    assert float(jnp.max(jnp.abs(vacuum.lateral_b_normal))) < 2.0e-12
    assert float(vacuum.neumann_result.compatibility_error) < 2.0e-12
    assert float(vacuum.neumann_result.condition_number) < 10.0


def test_axisymmetric_neumann_balance_changes_only_artificial_caps() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, nxi=9),
        z_min=-0.6,
        z_max=0.6,
    ).build_grid()
    radius = 0.3
    boundary = MirrorBoundary.from_radius(
        radius * (1.0 + 0.1 * (1.0 - jnp.asarray(grid.xi) ** 2)), grid
    )
    state = MirrorState.from_boundary(boundary, grid)
    geometry = evaluate_geometry(state, grid)
    target_bz = 0.08
    plasma_field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=0.5 * target_bz * radius**2,
    )
    surface = build_closed_mirror_surface(
        boundary, grid, axisymmetric_ntheta=8, cap_rim_grade=3.5
    )

    def displaced_dipole(points):
        displacement = jnp.asarray(points) - jnp.asarray([0.0, 0.0, 3.0])
        radius_squared = jnp.sum(displacement**2, axis=-1)
        moment = jnp.asarray([0.0, 0.0, 0.02])
        return (
            3.0
            * displacement
            * jnp.sum(moment * displacement, axis=-1)[..., None]
            / radius_squared[..., None] ** 2.5
            - moment / radius_squared[..., None] ** 1.5
        )

    neumann = axisymmetric_plasma_external_neumann(
        surface, plasma_field, grid, displaced_dipole
    )
    representatives = jnp.asarray(surface.reduced_representatives)
    lateral_points = surface.collocation_xyz[representatives[: grid.nxi]]
    lateral_normals = surface.collocation_normals[representatives[: grid.nxi]]
    expected_lateral = -jnp.sum(
        displaced_dipole(lateral_points) * lateral_normals, axis=1
    )
    np.testing.assert_allclose(neumann[: grid.nxi], expected_lateral, atol=2.0e-16)

    vacuum = solve_axisymmetric_exterior_vacuum(
        boundary,
        plasma_field,
        grid,
        displaced_dipole,
        axisymmetric_ntheta=8,
        cap_rim_grade=3.5,
        order=4,
    )
    assert float(vacuum.neumann_result.raw_compatibility_error) > 1.0e-10
    assert float(vacuum.neumann_result.compatibility_error) < 2.0e-15
    assert float(jnp.max(jnp.abs(vacuum.lateral_b_normal))) < 2.0e-15


def test_axisymmetric_exterior_vacuum_is_shape_differentiable() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, nxi=7),
        z_min=-0.6,
        z_max=0.6,
    ).build_grid()
    def lateral_field(radius):
        boundary = MirrorBoundary.from_radius(radius, grid)
        state = MirrorState.from_boundary(boundary, grid)
        geometry = evaluate_geometry(state, grid)
        plasma_field = contravariant_field(
            state,
            geometry,
            grid,
            axial_flux_derivative=0.0036,
        )
        return solve_axisymmetric_exterior_vacuum(
            boundary,
            plasma_field,
            grid,
            _paraxial_field,
            axisymmetric_ntheta=8,
            cap_rim_grade=3.0,
            order=6,
            spectral_side_density=True,
        ).lateral_field_xyz

    _, tangent = jax.jvp(
        lateral_field,
        (jnp.asarray(0.3),),
        (jnp.asarray(1.0),),
    )
    assert np.all(np.isfinite(np.asarray(tangent)))
    assert float(jnp.linalg.norm(tangent)) > 0.0


def test_singular_boundary_green_identity_converges_for_axial_harmonic() -> None:
    errors = []
    for ns, nxi, ntheta, order in ((7, 9, 8, 6), (13, 21, 20, 8)):
        grid = _grid(ns=ns, nxi=nxi)
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(0.37, grid),
            grid,
            axisymmetric_ntheta=ntheta,
        )
        xyz = surface.collocation_xyz
        normal = surface.collocation_normals
        dirichlet = xyz[:, 2]
        neumann = normal[:, 2]
        residual = laplace_reduced_green_boundary_residual(
            surface,
            surface.reduce_collocation_values(dirichlet),
            surface.reduce_collocation_values(neumann),
            order=order,
        )
        scale = jnp.sqrt(
            jnp.mean(dirichlet**2) + surface.area * jnp.mean(neumann**2)
        )
        errors.append(float(jnp.sqrt(jnp.mean(residual**2)) / scale))

    assert errors[1] < errors[0]
    assert errors[1] < 2.0e-3
