"""Free-space mirror free-boundary equilibrium and diagnostic tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    SplineMirrorState,
    solve_beta_scan,
)
import vmex.mirror.free_boundary as continuation  # noqa: E402
from vmex.mirror.free_boundary import (  # noqa: E402
    _build_free_equilibrium_problem,
)
from vmex.mirror.output import (  # noqa: E402
    FreeBoundaryRestart,
    load_free_boundary_restart,
    save_free_boundary_restart,
    summarize_axisymmetric_beta_scan,
)


@pytest.fixture(scope="module", autouse=True)
def _enable_solver_jit():
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def _restart_fixture(config, elements, radius, mass_scale):
    discretization = SplineMirrorDiscretization.build_cgl(config, elements=elements)
    shape = (discretization.grid.ntheta, discretization.coefficient_count)
    boundary = SplineMirrorBoundary(jnp.full(shape, radius))
    coefficients = jnp.broadcast_to(boundary.radius_coefficients, (discretization.grid.ns,) + shape)
    state = SplineMirrorState(coefficients, jnp.zeros_like(coefficients))
    return discretization, FreeBoundaryRestart(boundary, state, mass_scale)


def test_free_boundary_restart_roundtrip_is_compact_and_grid_checked(tmp_path) -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=0, nxi=9))
    discretization, restart = _restart_fixture(config, 4, 0.3, 1.25)
    path = save_free_boundary_restart(tmp_path / "beta_003", restart)
    loaded = load_free_boundary_restart(path, discretization)
    assert path.stat().st_size < 4096
    np.testing.assert_array_equal(loaded.boundary.radius_coefficients, restart.boundary.radius_coefficients)
    np.testing.assert_array_equal(loaded.plasma_state.radius_coefficients, restart.plasma_state.radius_coefficients)
    np.testing.assert_array_equal(loaded.plasma_state.lambda_coefficients, restart.plasma_state.lambda_coefficients)
    assert loaded.mass_scale == restart.mass_scale
    mismatched = SplineMirrorDiscretization.build_cgl(
        MirrorConfig(resolution=MirrorResolution(ns=9, mpol=0, nxi=9)), elements=4
    )
    with pytest.raises(ValueError, match="state coefficients"):
        load_free_boundary_restart(path, mismatched)


def _external_mirror_field(points):
    """Curl-free, divergence-free paraxial mirror field."""

    points = jnp.asarray(points)
    x, y, z = jnp.moveaxis(points, -1, 0)
    curvature = 0.02
    return jnp.stack(
        (
            -curvature * x * z,
            -curvature * y * z,
            0.08 + curvature * (z**2 - 0.5 * (x**2 + y**2)),
        ),
        axis=-1,
    )


def _on_axis_mirror_field(z, **_unused):
    return 0.08 + 0.02 * jnp.asarray(z) ** 2


def _free_case(ns, nxi, elements, max_iterations, *, mpol=0, radius=0.25):
    """Build the shared paraxial-field free-boundary fixture."""

    config = MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=mpol, nxi=nxi),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=max_iterations,
    )
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build_cgl(config, elements=elements)
    grid = discretization.grid
    on_axis = _on_axis_mirror_field(jnp.asarray(grid.z))
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * radius**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    return config, source_grid, discretization, grid, on_axis, center, flux, boundary


def test_free_coefficient_operator_matches_dense_forward_and_transpose() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, nxi=7),
        z_min=-0.8,
        z_max=0.8,
    )
    discretization = SplineMirrorDiscretization.build_cgl(config, elements=2, quadrature_order=3)
    nodes = jnp.asarray(discretization.spline.collocation_nodes)
    boundary = SplineMirrorBoundary((0.24 * (1.0 - 0.08 * nodes**2))[None])
    radius = jnp.broadcast_to(
        boundary.radius_coefficients,
        (discretization.grid.ns, discretization.grid.ntheta, discretization.coefficient_count),
    )
    state = SplineMirrorState(radius, jnp.zeros_like(radius))
    problem = _build_free_equilibrium_problem(
        boundary,
        state,
        discretization,
        _external_mirror_field,
        axial_flux_derivative=0.0024,
        mass_profile=0.0,
        current_derivative=0.0,
        solve_lambda=False,
        gamma=5.0 / 3.0,
        target_central_pressure=None,
        initial_mass_scale=1.0,
        exterior_ntheta=8,
        exterior_order=4,
        exterior_spectral_side_density=True,
    )
    point = problem.vectorizer.pack()
    dense = np.asarray(jax.jacfwd(problem.residual_function)(jnp.asarray(point)))
    direction = np.linspace(0.1, 0.7, problem.size)
    cotangent = np.linspace(-0.3, 0.4, problem.size)

    assert problem.residual(point).shape == (problem.size,)
    assert np.all(np.isfinite(dense))
    operator = problem.linear_operator(point)
    np.testing.assert_allclose(operator @ direction, dense @ direction, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(operator.rmatvec(cotangent), dense.T @ cotangent, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(
        np.vdot(operator @ direction, cotangent),
        np.vdot(direction, operator.rmatvec(cotangent)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


@pytest.mark.full
def test_unbounded_exterior_free_boundary_beta_scan_converges(_module_jit_enabled) -> None:
    config, source_grid, discretization, plasma_grid, on_axis, center, flux, initial_boundary = _free_case(5, 7, 4, 200)
    betas = jnp.asarray([0.0, 0.10, 0.25, 0.50])
    results = solve_beta_scan(
        discretization.fit_boundary(initial_boundary, source_grid),
        discretization,
        config,
        _external_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
    )

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(float(result.interface.vacuum_b_normal_rms) < 1.0e-12 for result in results)
    assert all(float(result.interface.normal_stress_rms) < 1.0e-12 for result in results)
    diagnostics = summarize_axisymmetric_beta_scan(
        results,
        betas,
        plasma_grid,
        reference_field=float(on_axis[center]),
    )
    np.testing.assert_allclose(
        [item.achieved_reference_beta for item in diagnostics],
        betas,
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    center_radii = np.asarray([item.center_radius for item in diagnostics])
    field_ratios = np.asarray([item.diamagnetic_field_ratio for item in diagnostics])
    assert np.all(np.diff(center_radii) > 0.0)
    assert np.all(np.diff(field_ratios) < 0.0)
    assert center_radii[-1] > 1.07 * center_radii[0]
    assert field_ratios[-1] < 0.78
    assert all(np.isfinite(float(item.center_vacuum_side_field)) for item in diagnostics)

    resumed = solve_beta_scan(
        discretization.fit_boundary(initial_boundary, source_grid),
        discretization,
        config,
        _external_mirror_field,
        jnp.asarray([0.50]),
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        initial_restart=FreeBoundaryRestart.from_result(results[-1]),
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
    )[0]
    np.testing.assert_allclose(
        resumed.coefficient_boundary.radius_coefficients,
        results[-1].coefficient_boundary.radius_coefficients,
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        resumed.coefficient_state.radius_coefficients,
        results[-1].coefficient_state.radius_coefficients,
        rtol=2.0e-11,
        atol=2.0e-12,
    )


@pytest.mark.full
def test_unbounded_exterior_beta_observables_converge_with_resolution(_module_jit_enabled) -> None:
    observables = []
    compatibility = []
    betas = jnp.asarray([0.0, 0.10, 0.50])
    for ns, nxi, ntheta_panel in ((5, 7, 8), (7, 13, 12), (9, 17, 16)):
        elements = {7: 4, 13: 7, 17: 9}[nxi]
        config, source_grid, discretization, plasma_grid, on_axis, center, flux, initial_boundary = _free_case(
            ns, nxi, elements, 500
        )
        results = solve_beta_scan(
            discretization.fit_boundary(initial_boundary, source_grid),
            discretization,
            config,
            _external_mirror_field,
            betas,
            axial_flux_derivative=flux,
            reference_field=float(on_axis[center]),
            exterior_ntheta=ntheta_panel,
            exterior_order=8,
        )
        assert all(result.converged for result in results)
        assert all(float(result.variational_max) <= config.ftol for result in results)
        observables.append(
            np.asarray(
                [
                    [
                        float(result.boundary.radius_scale[0, center]),
                        float(jnp.sqrt(result.plasma_b_squared[0, 0, center])),
                    ]
                    for result in results
                ]
            )
        )
        compatibility.append(
            np.asarray([float(result.vacuum_field.neumann_result.raw_compatibility_error) for result in results])
        )

    relative_change = np.abs((observables[-1] - observables[-2]) / observables[-1])
    # The supported 0/10% lane is converged below 0.2%; the 50% validation
    # continuation is required to stay bounded but is not promoted.
    assert np.max(relative_change[:2]) < 2.0e-3
    assert np.max(relative_change[2]) < 1.5e-2
    assert np.max(compatibility[-1]) < 3.0e-9
    assert np.all(compatibility[-1] < compatibility[0])
    assert float(results[-1].boundary.radius_scale[0, center]) > 1.07 * float(
        results[0].boundary.radius_scale[0, center]
    )


def test_beta_scan_propagates_restart_mass_scale(monkeypatch) -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=0, nxi=5))
    discretization, restart = _restart_fixture(config, 2, 0.31, 2.5)
    grid = discretization.grid
    shape = (grid.ntheta, discretization.coefficient_count)
    reference = SplineMirrorBoundary(jnp.full(shape, 0.3))
    received = []

    def fake_solve(boundary, *_args, **kwargs):
        received.append(
            (
                boundary,
                kwargs["initial_state"],
                kwargs["initial_mass_scale"],
                kwargs["exterior_spectral_side_density"],
            )
        )
        return SimpleNamespace(
            coefficient_boundary=boundary,
            coefficient_state=kwargs["initial_state"],
            mass_scale=jnp.asarray(kwargs["initial_mass_scale"] + 0.5),
            pressure=jnp.zeros(grid.shape),
        )

    monkeypatch.setattr(continuation, "solve_free_boundary", fake_solve)
    solve_beta_scan(
        reference,
        discretization,
        config,
        object(),
        jnp.asarray([0.0, 0.0]),
        axial_flux_derivative=0.1,
        reference_field=1.0,
        initial_restart=restart,
        exterior_spectral_side_density=True,
    )

    np.testing.assert_array_equal(
        received[0][0].radius_coefficients,
        restart.boundary.radius_coefficients,
    )
    np.testing.assert_array_equal(
        received[0][1].radius_coefficients,
        restart.plasma_state.radius_coefficients,
    )
    np.testing.assert_array_equal(
        received[0][1].lambda_coefficients,
        restart.plasma_state.lambda_coefficients,
    )
    assert received[0][2] == 2.5
    assert received[1][2] == 3.0
    assert all(item[3] is True for item in received)

    with pytest.raises(ValueError, match="mutually exclusive"):
        solve_beta_scan(
            reference,
            discretization,
            config,
            object(),
            jnp.asarray([0.0]),
            axial_flux_derivative=0.1,
            reference_field=1.0,
            initial_state=restart.plasma_state,
            initial_restart=restart,
        )
