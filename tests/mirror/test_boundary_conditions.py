"""Boundary-family directional derivatives for the mirror energy.

Each natural boundary term derived in
``docs/explanation/mirror-boundary-conditions.md`` is pinned here by the
discrete directional derivative of the energy along a variation supported on
exactly one boundary family: constrained families must be null directions of
the projected gradient, gauge shifts must leave the energy unchanged to
round-off, and the released natural terms (cut through-flux, lateral
displacement) must converge to their derived surface integrals. The axis
regularity audit demonstrates that the odd/even radius parity rule reaches
its design order and that full per-mode ``rho^|m|`` factors would break
supported finite-ellipticity states.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    SplineMirrorDiscretization,
)
from vmex.mirror.basis import ChebyshevBasis  # noqa: E402
from vmex.mirror.forces import (  # noqa: E402
    MU0,
    _half_mesh_metric,
    _half_mesh_samples,
    _interpolate_radius_scale,
    _interpolate_stream_function,
    interface_residual,
    mass_profile_from_pressure,
    mirror_energy,
)
from vmex.mirror.model import project_fixed_boundary_state  # noqa: E402
from vmex.mirror.splines import (  # noqa: E402
    SplineMirrorBoundary,
    SplineMirrorState,
    _SplineStateVectorizer,
    solve_fixed_boundary,
)


@pytest.fixture(scope="module", autouse=True)
def _enable_solver_jit():
    """Exercise the solve-backed tests in their production execution mode."""

    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def _generic_state(grid):
    """Non-equilibrium 3D state exercising every term of the energy."""

    s = jnp.asarray(grid.s)[:, None, None]
    theta = jnp.asarray(grid.theta)[None, :, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    radius = 0.3 * (1.0 + 0.05 * jnp.sqrt(s) * jnp.cos(theta) * (1.0 - xi**2) + 0.1 * s * xi)
    lam = 0.004 * s * jnp.sin(theta) * (1.0 - 0.5 * xi)
    state = MirrorState(jnp.broadcast_to(radius, grid.shape), jnp.broadcast_to(lam, grid.shape))
    kwargs = {
        "axial_flux_derivative": jnp.linspace(0.09, 0.12, grid.ns),
        "current_derivative": jnp.linspace(0.02, 0.03, grid.ns),
        "mass_profile": 800.0 * (1.0 - jnp.asarray(grid.s)),
    }
    return state, kwargs


def _cut_flux_natural_term(state, grid, kwargs, radial_profile, cut_patterns):
    """Evaluate ``-(1/mu0) [oint B_theta dlam dtheta ds]`` on the energy's own Gauss points.

    ``radial_profile`` is the scalar radial factor of the cut variation and
    ``cut_patterns = (minus, plus)`` its theta patterns at ``xi = -1, +1``.
    """

    samples = _half_mesh_samples(
        state, grid, kwargs["axial_flux_derivative"], kwargs["current_derivative"]
    )
    metric = _half_mesh_metric(samples, grid)
    b_theta = samples.field_theta_numerator / metric.jacobian
    b_xi = samples.field_xi_numerator / metric.jacobian
    b_cov_theta = metric.g_thetatheta * b_theta + metric.g_thetaxi * b_xi
    ds = float(grid.s[1] - grid.s[0])
    profile = radial_profile(samples.s)
    theta_weights = jnp.asarray(grid.theta_basis.weights)

    def surface(column, pattern):
        integrand = b_cov_theta[..., column] * profile[..., column]
        cell = 0.5 * ds * jnp.sum(integrand, axis=0)
        return jnp.sum(cell * (theta_weights * pattern)[None, :])

    minus, plus = cut_patterns
    return -(surface(-1, plus) - surface(0, minus)) / MU0


def test_gauge_family_leaves_energy_invariant_and_is_projected_out() -> None:
    grid = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=2, nxi=11)).build_grid()
    state, kwargs = _generic_state(grid)
    boundary = MirrorBoundary(state.radius_scale[-1])
    shift = jnp.asarray(grid.s)[:, None, None] ** 2 - 0.3

    base = mirror_energy(state, grid, **kwargs).total
    shifted = mirror_energy(
        MirrorState(state.radius_scale, state.lambda_stream + shift), grid, **kwargs
    ).total
    np.testing.assert_allclose(shifted, base, rtol=1.0e-14)

    gradient = jax.grad(
        lambda trial: mirror_energy(
            project_fixed_boundary_state(trial, boundary, grid), grid, **kwargs
        ).total
    )(state)
    gauge_direction = jnp.broadcast_to(shift, grid.shape)
    along_gauge = float(jnp.vdot(gradient.lambda_stream, gauge_direction))
    scale = float(
        jnp.linalg.norm(gradient.lambda_stream.ravel()) * jnp.linalg.norm(gauge_direction.ravel())
    )
    assert abs(along_gauge) <= 1.0e-12 * scale


def test_lateral_and_cut_families_are_null_directions_of_the_projected_gradient() -> None:
    grid = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=2, nxi=11)).build_grid()
    state, kwargs = _generic_state(grid)
    boundary = MirrorBoundary(state.radius_scale[-1])

    def projected_energy(trial):
        return mirror_energy(
            project_fixed_boundary_state(trial, boundary, grid), grid, **kwargs
        ).total

    gradient = jax.grad(projected_energy)(state)
    np.testing.assert_allclose(gradient.radius_scale[-1], 0.0, atol=0.0)

    # A variation supported only on the lateral row cannot change the energy.
    rng = np.random.default_rng(11)
    lateral = jnp.zeros(grid.shape).at[-1].set(jnp.asarray(rng.normal(size=(grid.ntheta, grid.nxi))))
    np.testing.assert_allclose(
        projected_energy(MirrorState(state.radius_scale + 0.05 * lateral, state.lambda_stream)),
        projected_energy(state),
        rtol=1.0e-14,
    )

    # The natural cut-geometry term exists; the constraint, not cancellation,
    # is what removes it from the solve.
    raw_gradient = jax.grad(
        lambda trial: mirror_energy(trial, grid, **kwargs).total
    )(state)
    assert float(jnp.max(jnp.abs(raw_gradient.radius_scale[1:-1, :, [0, -1]]))) > 0.0


def test_open_solve_vector_excludes_every_constrained_boundary_family() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=1, nxi=9))
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    coefficients = 0.3 * jnp.ones((config.resolution.ntheta, discretization.coefficient_count))
    boundary = SplineMirrorBoundary(coefficients)
    radius = jnp.broadcast_to(coefficients, (config.resolution.ns,) + coefficients.shape)
    state = SplineMirrorState(radius, 0.01 * jnp.ones_like(radius))
    vectorizer = _SplineStateVectorizer.build(
        state, boundary, discretization, axial_flux_derivative=0.01, solve_lambda=True
    )
    base = vectorizer.base

    # Active radius variables cover interior rows and interior axial columns only.
    assert vectorizer.radius_indices[0].min() == 1
    assert vectorizer.radius_indices[0].max() == config.resolution.ns - 2
    assert vectorizer.radius_indices[2].min() == 1
    assert vectorizer.radius_indices[2].max() == discretization.coefficient_count - 2
    np.testing.assert_array_equal(
        vectorizer.lambda_axial_indices, np.arange(1, discretization.coefficient_count - 1)
    )

    packed = vectorizer.pack()
    rng = np.random.default_rng(3)
    perturbed = vectorizer.unpack(jnp.asarray(packed + 0.01 * rng.normal(size=packed.size)))

    edges = np.asarray([0, -1])
    np.testing.assert_allclose(
        np.asarray(perturbed.radius_coefficients)[:, :, edges],
        np.asarray(base.radius_coefficients)[:, :, edges],
    )
    np.testing.assert_allclose(
        np.asarray(perturbed.radius_coefficients)[-1], np.asarray(base.radius_coefficients)[-1]
    )
    np.testing.assert_allclose(
        np.asarray(perturbed.lambda_coefficients)[1:, :, edges],
        np.asarray(base.lambda_coefficients)[1:, :, edges],
    )
    np.testing.assert_allclose(
        np.asarray(perturbed.lambda_coefficients)[0], np.asarray(perturbed.lambda_coefficients)[1]
    )

    # Axis regularity: odd poloidal radius modes stay removed from the axis row.
    axis_modes = np.fft.fft(np.asarray(perturbed.radius_coefficients)[0], axis=0)
    modes = np.rint(
        np.fft.fftfreq(config.resolution.ntheta, d=1.0 / config.resolution.ntheta)
    ).astype(int)
    np.testing.assert_allclose(axis_modes[np.abs(modes) % 2 == 1], 0.0, atol=1.0e-14)

    # Gauge: the weighted interior mean of lambda is pinned per radial surface.
    coefficient_weights = np.asarray(discretization.evaluation_matrix).T @ np.asarray(
        discretization.grid.axial_basis.weights
    )
    interior_weights = (
        np.asarray(discretization.grid.theta_basis.weights)[:, None]
        * coefficient_weights[None, vectorizer.lambda_axial_indices]
    )
    interior = np.asarray(perturbed.lambda_coefficients)[1:, :, vectorizer.lambda_axial_indices]
    weighted_mean = np.einsum("jk,ijk->i", interior_weights, interior)
    np.testing.assert_allclose(
        weighted_mean + vectorizer.lambda_fixed_weighted_sum, 0.0, atol=1.0e-15
    )


def test_periodic_hybrid_solve_vector_has_no_cut_masks() -> None:
    resolution = MirrorResolution(ns=5, mpol=1, nxi=12)
    discretization = SplineMirrorDiscretization.build_closed(resolution, coefficient_count=12)
    coefficients = 0.3 * jnp.ones((resolution.ntheta, discretization.coefficient_count))
    boundary = SplineMirrorBoundary(coefficients)
    radius = jnp.broadcast_to(coefficients, (resolution.ns,) + coefficients.shape)
    state = SplineMirrorState(radius, 0.01 * jnp.ones_like(radius))
    vectorizer = _SplineStateVectorizer.build(
        state, boundary, discretization, axial_flux_derivative=0.01, solve_lambda=True
    )

    count = discretization.coefficient_count
    assert vectorizer.radius_size == (resolution.ns - 2) * resolution.ntheta * count
    assert vectorizer.radius_indices[2].min() == 0
    assert vectorizer.radius_indices[2].max() == count - 1
    np.testing.assert_array_equal(vectorizer.lambda_axial_indices, np.arange(count))
    np.testing.assert_allclose(vectorizer.lambda_fixed_weighted_sum, 0.0, atol=0.0)


def test_axis_regularity_families_are_enforced() -> None:
    grid = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=2, nxi=11)).build_grid()
    state, kwargs = _generic_state(grid)
    boundary = MirrorBoundary(state.radius_scale[-1])

    def projected_energy(trial):
        return mirror_energy(
            project_fixed_boundary_state(trial, boundary, grid), grid, **kwargs
        ).total

    # Odd poloidal radius modes on the axis row are removed by the projection.
    theta = jnp.asarray(grid.theta)
    odd = jnp.zeros(grid.shape).at[0].set(
        (0.03 * jnp.cos(theta))[:, None] * jnp.ones((grid.ntheta, grid.nxi))
    )
    np.testing.assert_allclose(
        projected_energy(MirrorState(state.radius_scale + odd, state.lambda_stream)),
        projected_energy(state),
        rtol=1.0e-14,
    )

    # The axis stream function is fixed by single-valued axial flux, so the
    # incoming axis row cannot influence the energy.
    rng = np.random.default_rng(5)
    replaced = state.lambda_stream.at[0].set(jnp.asarray(rng.normal(size=(grid.ntheta, grid.nxi))))
    np.testing.assert_allclose(
        mirror_energy(MirrorState(state.radius_scale, replaced), grid, **kwargs).total,
        mirror_energy(state, grid, **kwargs).total,
        rtol=1.0e-14,
    )


def test_cut_flux_directional_derivative_converges_to_natural_term() -> None:
    """The released cut term is ``-(1/mu0) [oint B_theta dlam]_{xi=-1}^{+1}``."""

    def directional(nxi, cylinder):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=9, mpol=1, nxi=nxi), z_min=-1.1, z_max=0.9
        )
        grid = config.build_grid()
        if cylinder:
            state = MirrorState(0.3 * jnp.ones(grid.shape), jnp.zeros(grid.shape))
            kwargs = {
                "axial_flux_derivative": jnp.linspace(0.09, 0.12, grid.ns),
                "current_derivative": jnp.linspace(0.02, 0.03, grid.ns),
                "mass_profile": 0.0,
            }
        else:
            state, kwargs = _generic_state(grid)

        # Cut-supported variation, vanishing at the axis so the prescribed
        # axis stream function does not intercept it.
        s_column = jnp.asarray(grid.s)[:, None]
        delta = jnp.zeros(grid.shape)
        delta = delta.at[:, :, 0].set(jnp.broadcast_to(s_column, (grid.ns, grid.ntheta)))
        delta = delta.at[:, :, -1].set(-0.7 * jnp.broadcast_to(s_column, (grid.ns, grid.ntheta)))

        def energy_of(lam):
            return mirror_energy(MirrorState(state.radius_scale, lam), grid, **kwargs).total

        derivative = float(jnp.vdot(jax.grad(energy_of)(state.lambda_stream), delta))
        natural = float(
            _cut_flux_natural_term(
                state,
                grid,
                kwargs,
                lambda s_gauss: s_gauss,
                (jnp.ones(grid.ntheta), -0.7 * jnp.ones(grid.ntheta)),
            )
        )
        finite_difference = None
        if not cylinder and nxi == 17:
            step = 1.0e-6
            finite_difference = float(
                (energy_of(state.lambda_stream + step * delta) - energy_of(state.lambda_stream - step * delta))
                / (2.0 * step)
            )
        return derivative, natural, finite_difference

    # Uniform cylinder with finite current: B_theta is polynomial in s and
    # constant along xi, so the discrete identity is exact.
    derivative, natural, _ = directional(11, cylinder=True)
    np.testing.assert_allclose(derivative, natural, rtol=1.0e-12)

    # Generic 3D state: the interior Euler-Lagrange remainder decays with the
    # endpoint quadrature weight, so the derivative converges to the term.
    errors = []
    for nxi in (9, 17, 33):
        derivative, natural, finite_difference = directional(nxi, cylinder=False)
        errors.append(abs(derivative - natural) / abs(natural))
        if finite_difference is not None:
            np.testing.assert_allclose(derivative, finite_difference, rtol=1.0e-7)
    assert errors[0] / errors[1] > 2.5
    assert errors[1] / errors[2] > 2.5
    assert errors[-1] < 3.0e-4


def test_axial_boundary_term_identity_distinguishes_open_and_periodic_bases() -> None:
    # Open CGL basis: the integrated derivative is exactly the boundary pair.
    chebyshev = ChebyshevBasis.build(13)
    nodes = jnp.asarray(chebyshev.nodes)
    polynomial = 0.3 + 0.7 * nodes - 1.1 * nodes**3 + 0.25 * nodes**8
    integrated = float(
        jnp.sum(jnp.asarray(chebyshev.weights) * chebyshev.differentiate(polynomial))
    )
    np.testing.assert_allclose(integrated, float(polynomial[-1] - polynomial[0]), atol=1.0e-13)

    # Periodic spline basis: the boundary pair cancels for every nodal function.
    discretization = SplineMirrorDiscretization.build_closed(
        MirrorResolution(ns=5, mpol=0, nxi=12), coefficient_count=12
    )
    basis = discretization.grid.axial_basis
    rng = np.random.default_rng(0)
    values = jnp.asarray(rng.normal(size=discretization.grid.nxi))
    np.testing.assert_allclose(
        float(jnp.sum(jnp.asarray(basis.weights) * basis.differentiate(values))), 0.0, atol=1.0e-13
    )


def test_interface_residual_measures_total_pressure_jump() -> None:
    theta_weights = jnp.ones(3) * (2.0 * np.pi / 3.0)
    axial_weights = jnp.asarray([0.5, 1.0, 1.0, 0.5])
    pressure = jnp.asarray(2000.0)
    plasma_b_squared = 0.04 * (1.0 + 0.1 * jnp.linspace(0.0, 1.0, 12).reshape(3, 4))
    balanced_vacuum = plasma_b_squared + 2.0 * MU0 * pressure

    balanced = interface_residual(
        pressure=pressure,
        plasma_b_squared=plasma_b_squared,
        vacuum_b_squared=balanced_vacuum,
        plasma_b_normal=jnp.zeros((3, 4)),
        vacuum_b_normal=jnp.zeros((3, 4)),
        theta_weights=theta_weights,
        axial_weights=axial_weights,
    )
    np.testing.assert_allclose(balanced.normal_stress_jump, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(balanced.normal_stress_rms, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(balanced.plasma_b_normal_rms, 0.0, atol=0.0)
    np.testing.assert_allclose(balanced.vacuum_b_normal_rms, 0.0, atol=0.0)

    unbalanced = interface_residual(
        pressure=pressure,
        plasma_b_squared=plasma_b_squared,
        vacuum_b_squared=0.9 * balanced_vacuum,
        plasma_b_normal=0.01 * jnp.ones((3, 4)),
        vacuum_b_normal=jnp.zeros((3, 4)),
        theta_weights=theta_weights,
        axial_weights=axial_weights,
    )
    expected_jump = pressure + (plasma_b_squared - 0.9 * balanced_vacuum) / (2.0 * MU0)
    np.testing.assert_allclose(unbalanced.normal_stress_jump, expected_jump, rtol=1.0e-13)
    assert float(unbalanced.normal_stress_rms) > 0.0
    assert float(unbalanced.plasma_b_normal_rms) > 0.0


# ---------------------------------------------------------------------------
# Solve-backed physics: lateral natural term and the paraxial finite-beta limit
# ---------------------------------------------------------------------------


_PARAXIAL_MIDPLANE_RADIUS = 0.15
_PARAXIAL_LENGTH = 2.4
_PARAXIAL_FLARE = 0.18
_PARAXIAL_PRESSURE = 9500.0


@pytest.fixture(scope="module")
def _solved_finite_beta_mirror():
    """One converged long-thin finite-beta fixed-boundary equilibrium."""

    config = MirrorConfig(
        resolution=MirrorResolution(ns=9, mpol=0, nxi=11),
        z_min=-0.5 * _PARAXIAL_LENGTH,
        z_max=0.5 * _PARAXIAL_LENGTH,
        ftol=1.0e-10,
        max_iterations=2000,
    )
    source_grid = config.build_grid()
    xi = jnp.asarray(source_grid.xi)
    radius = _PARAXIAL_MIDPLANE_RADIUS * (1.0 + _PARAXIAL_FLARE * xi**2)
    flux = 0.01
    boundary = MirrorBoundary.from_radius(radius, source_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=4)
    grid = discretization.grid
    fitted = discretization.fit_boundary(boundary, source_grid)
    initial = discretization.fit_state(MirrorState.from_boundary(boundary, source_grid), source_grid)

    vacuum = mirror_energy(
        discretization.evaluate_state(initial), grid, axial_flux_derivative=flux
    )
    pressure = _PARAXIAL_PRESSURE * (1.0 - 0.5 * jnp.asarray(grid.s))
    mass = mass_profile_from_pressure(pressure, vacuum.volume_derivative)
    result = solve_fixed_boundary(
        initial,
        fitted,
        discretization,
        config,
        axial_flux_derivative=flux,
        mass_profile=mass,
        solve_lambda=True,
        require_convergence=True,
    )
    return {
        "grid": grid,
        "discretization": discretization,
        "boundary": discretization.evaluate_boundary(fitted),
        "state": result.evaluated.state,
        "energy": result.evaluated.energy,
        "flux": flux,
        "mass": mass,
    }


def test_solved_lateral_natural_term_is_the_total_pressure_surface_integral(
    _solved_finite_beta_mirror,
) -> None:
    """At equilibrium ``dW = -oint (p + B^2/2mu0) xi.n dA`` for boundary motion."""

    grid = _solved_finite_beta_mirror["grid"]
    state = _solved_finite_beta_mirror["state"]
    energy = _solved_finite_beta_mirror["energy"]
    boundary = _solved_finite_beta_mirror["boundary"]
    flux = _solved_finite_beta_mirror["flux"]
    mass = _solved_finite_beta_mirror["mass"]

    xi = jnp.asarray(grid.xi)
    displacement = jnp.broadcast_to((1.0 - xi**2)[None, :], (grid.ntheta, grid.nxi))

    def energy_of(epsilon):
        moved = MirrorBoundary(boundary.radius_scale + epsilon * displacement)
        projected = project_fixed_boundary_state(state, moved, grid)
        return mirror_energy(
            projected, grid, axial_flux_derivative=flux, mass_profile=mass
        ).total

    derivative = float(jax.jvp(energy_of, (0.0,), (1.0,))[1])

    total_pressure = np.asarray(energy.pressure)[-1] + np.asarray(energy.b_squared)[-1] / (
        2.0 * MU0
    )
    area_measure = (
        np.asarray(state.radius_scale)[-1]
        * float(grid.dz_dxi)
        * np.asarray(grid.theta_basis.weights)[:, None]
        * np.asarray(grid.axial_basis.weights)[None, :]
    )
    natural = -float(np.sum(total_pressure * np.asarray(displacement) * area_measure))

    assert derivative < 0.0
    np.testing.assert_allclose(derivative, natural, rtol=1.0e-3)


def test_paraxial_finite_beta_solve_matches_long_thin_pressure_balance(
    _solved_finite_beta_mirror,
) -> None:
    """Radial balance ``p + B^2/2mu0 = const`` at the midplane to ``O((a/L)^2)``."""

    grid = _solved_finite_beta_mirror["grid"]
    energy = _solved_finite_beta_mirror["energy"]

    midplane = int(np.abs(np.asarray(grid.z)).argmin())
    pressure = np.asarray(energy.pressure)
    magnetic = np.asarray(energy.b_squared)[:, 0, midplane] / (2.0 * MU0)
    total = pressure + magnetic
    magnetic_scale = float(np.mean(magnetic))
    epsilon_squared = (_PARAXIAL_MIDPLANE_RADIUS / _PARAXIAL_LENGTH) ** 2

    beta = _PARAXIAL_PRESSURE / magnetic_scale
    assert 0.02 < beta < 0.05
    deviation = float(np.max(np.abs(total - total.mean())))
    assert deviation <= 1.5 * epsilon_squared * magnetic_scale

    # Diamagnetism: the outward magnetic-pressure rise compensates the
    # pressure drop, with an O((a/L)^2) defect. A solve that ignored the
    # pressure force would leave the full drop uncompensated.
    pressure_drop = pressure[0] - pressure[-1]
    magnetic_rise = magnetic[-1] - magnetic[0]
    assert magnetic_rise > 0.0
    assert abs(magnetic_rise - pressure_drop) <= 2.0 * epsilon_squared * magnetic_scale


# ---------------------------------------------------------------------------
# Axis regularity audit (plan section 15.3)
# ---------------------------------------------------------------------------


def _full_mode_radius_interpolation(radius_scale, grid, fraction):
    """Audit variant extracting ``rho^|m|`` from every poloidal radius mode.

    This is the interpolation the full mode-dependent rule would use if it
    were applied to the radius scale. It exists only to demonstrate why the
    production odd/even parity rule is the correct choice.
    """

    radius_scale = jnp.asarray(radius_scale)
    ds = float(grid.s[1] - grid.s[0])
    modes = jnp.rint(jnp.fft.fftfreq(grid.ntheta, d=1.0 / grid.ntheta)).astype(int)
    power = 0.5 * jnp.abs(modes)[:, None].astype(radius_scale.dtype)
    radial_s = jnp.asarray(grid.s, dtype=radius_scale.dtype)
    safe_s = jnp.where(radial_s == 0.0, 1.0, radial_s)
    radius_modes = jnp.fft.fft(radius_scale, axis=1)
    regular_modes = radius_modes / safe_s[:, None, None] ** power[None]
    extrapolated = 2.0 * regular_modes[1] - regular_modes[2]
    regular_modes = regular_modes.at[0].set(
        jnp.where(power == 0.0, regular_modes[0], extrapolated)
    )
    regular_quadrature = (1.0 - fraction) * regular_modes[:-1][None] + fraction * regular_modes[1:][None]
    regular_derivative = (regular_modes[1:] - regular_modes[:-1])[None] / ds
    s_quadrature = jnp.asarray(grid.s[:-1])[None, :, None, None] + fraction * ds
    scale = s_quadrature ** power[None, None]
    scale_derivative = power[None, None] * s_quadrature ** (power[None, None] - 1.0)
    values = jnp.fft.ifft(scale * regular_quadrature, axis=2).real
    derivatives = jnp.fft.ifft(
        scale_derivative * regular_quadrature + scale * regular_derivative, axis=2
    ).real
    return values, derivatives


_GAUSS_FRACTION = jnp.asarray([0.2, 0.7])[:, None, None, None]


def _smooth_map_radius(s, theta):
    """Parity-correct modes with the smooth-map structure ``a_m = s^{m/2} g(s)``."""

    return (
        0.3 * (1.0 + 0.1 * np.exp(-s))
        + 0.05 * np.sqrt(s) * (1.0 + 0.3 * s**2) * np.cos(theta)
        + 0.04 * s * np.cos(2.0 * theta) / (1.0 + 0.5 * s)
        + 0.02 * s * np.sqrt(s) * np.cos(3.0 * theta) * (1.0 - 0.2 * s)
    )


def test_radius_parity_rule_reaches_design_order_and_full_mode_factors_add_nothing() -> None:
    parity_errors = []
    full_mode_errors = []
    for ns in (9, 17, 33):
        grid = MirrorConfig(resolution=MirrorResolution(ns=ns, mpol=3, nxi=5)).build_grid()
        s = np.asarray(grid.s)[:, None, None]
        theta = np.asarray(grid.theta)[None, :, None]
        scale = jnp.broadcast_to(jnp.asarray(_smooth_map_radius(s, theta)), grid.shape)

        ds = float(grid.s[1] - grid.s[0])
        s_gauss = np.asarray(grid.s[:-1])[None, :, None, None] + np.asarray(_GAUSS_FRACTION) * ds
        theta_gauss = np.asarray(grid.theta)[None, None, :, None]
        exact = _smooth_map_radius(s_gauss, theta_gauss)

        parity_values, _ = _interpolate_radius_scale(scale, grid, _GAUSS_FRACTION)
        full_values, _ = _full_mode_radius_interpolation(scale, grid, _GAUSS_FRACTION)
        parity_errors.append(float(np.max(np.abs(np.asarray(parity_values) - exact))))
        full_mode_errors.append(float(np.max(np.abs(np.asarray(full_values) - exact))))

    # The production parity rule converges at the O(ds^2) design order.
    for coarse, fine in zip(parity_errors[:-1], parity_errors[1:], strict=True):
        assert 3.3 < coarse / fine < 4.8
    # Full per-mode factors change constants, not the order.
    for coarse, fine in zip(full_mode_errors[:-1], full_mode_errors[1:], strict=True):
        assert 3.3 < coarse / fine < 4.8
    for parity, full in zip(parity_errors, full_mode_errors, strict=True):
        assert parity < 2.0 * full


def test_full_mode_radius_factors_break_finite_axis_ellipticity() -> None:
    """Dividing ``a_2`` by ``s`` manufactures a singular remainder on axis."""

    derivative_errors = []
    for ns in (9, 17, 33):
        grid = MirrorConfig(resolution=MirrorResolution(ns=ns, mpol=2, nxi=5)).build_grid()
        s = jnp.asarray(grid.s)[:, None, None]
        theta = jnp.asarray(grid.theta)[None, :, None]
        scale = jnp.broadcast_to(0.3 + 0.06 * jnp.cos(2.0 * theta) * (1.0 + 0.2 * s), grid.shape)

        ds = float(grid.s[1] - grid.s[0])
        s_gauss = np.asarray(grid.s[:-1])[None, :, None, None] + np.asarray(_GAUSS_FRACTION) * ds
        theta_gauss = np.asarray(grid.theta)[None, None, :, None]
        exact_values = 0.3 + 0.06 * np.cos(2.0 * theta_gauss) * (1.0 + 0.2 * s_gauss)
        exact_derivatives = 0.06 * 0.2 * np.cos(2.0 * theta_gauss) * np.ones_like(s_gauss)

        parity_values, parity_derivatives = _interpolate_radius_scale(scale, grid, _GAUSS_FRACTION)
        full_values, full_derivatives = _full_mode_radius_interpolation(scale, grid, _GAUSS_FRACTION)

        # The parity rule is exact: the state is linear in s mode by mode.
        np.testing.assert_allclose(
            np.asarray(parity_values),
            np.broadcast_to(exact_values, parity_values.shape),
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            np.asarray(parity_derivatives),
            np.broadcast_to(exact_derivatives, parity_derivatives.shape),
            atol=1.0e-12,
        )

        assert float(np.max(np.abs(np.asarray(full_values) - exact_values))) > 1.0e-2
        derivative_errors.append(
            float(np.max(np.abs(np.asarray(full_derivatives) - exact_derivatives)))
        )

    # The defect grows under refinement: the full-mode rule is inadmissible,
    # not merely inaccurate, for supported finite-ellipticity states.
    assert derivative_errors[0] < derivative_errors[1] < derivative_errors[2]


def test_stream_function_interpolation_is_exact_for_full_mode_regular_states() -> None:
    grid = MirrorConfig(resolution=MirrorResolution(ns=9, mpol=3, nxi=5)).build_grid()
    s = jnp.asarray(grid.s)[:, None, None]
    theta = jnp.asarray(grid.theta)[None, :, None]
    lam = (
        (0.2 + 0.1 * s)
        + jnp.sqrt(s) * (0.05 + 0.02 * s) * jnp.cos(theta)
        + s * (0.04 - 0.01 * s) * jnp.sin(2.0 * theta)
        + s * jnp.sqrt(s) * 0.03 * jnp.cos(3.0 * theta)
    )
    lam = jnp.broadcast_to(lam, grid.shape)

    values = _interpolate_stream_function(lam, grid, _GAUSS_FRACTION)
    ds = float(grid.s[1] - grid.s[0])
    s_gauss = np.asarray(grid.s[:-1])[None, :, None, None] + np.asarray(_GAUSS_FRACTION) * ds
    theta_gauss = np.asarray(grid.theta)[None, None, :, None]
    exact = (
        (0.2 + 0.1 * s_gauss)
        + np.sqrt(s_gauss) * (0.05 + 0.02 * s_gauss) * np.cos(theta_gauss)
        + s_gauss * (0.04 - 0.01 * s_gauss) * np.sin(2.0 * theta_gauss)
        + s_gauss * np.sqrt(s_gauss) * 0.03 * np.cos(3.0 * theta_gauss)
    )
    np.testing.assert_allclose(
        np.asarray(values), np.broadcast_to(exact, np.asarray(values).shape), atol=1.0e-13
    )
