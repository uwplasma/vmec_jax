"""Mirror energies and force diagnostics on the staggered radial mesh.

The pressure model follows VMEC's mass-conserving variational form.  A radial
mass profile ``M(s)`` defines ``p(s) = M(s) / V'(s)^gamma``, where
``V'(s) = integral J dtheta dxi``.  Varying the total energy therefore gives
the pressure force instead of treating pressure as geometry-independent.

The weak residual manually assembles the first variation on the same radial
Gauss points as the energy. The pointwise residual reconstructs
``curl(B)/mu0 x B - grad(p)`` on the full mesh and is a separate spatial-error
diagnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import pi
from typing import Any

import jax
import jax.numpy as jnp

from .geometry import (
    ClosedAxisGeometry,
    ContravariantField,
    MirrorGeometry,
    contravariant_field,
    evaluate_closed_geometry,
    evaluate_geometry,
    magnetic_field_squared,
    regularize_axis_stream_function,
)
from .model import MirrorBoundary, MirrorState, project_fixed_boundary_state

Array = Any
MU0 = 4.0e-7 * pi


def _profile(values: Array, ns: int, dtype: Any, *, name: str) -> Array:
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (ns,))
    if values.shape != (ns,):
        raise ValueError(f"{name} shape {values.shape} must be scalar or ({ns},)")
    return values


def _surface_integral(values: Array, grid: "MirrorGrid") -> Array:
    return jnp.einsum(
        "j,k,ijk->i",
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        values,
    )


@dataclass(frozen=True)
class _HalfMeshSamples:
    """Geometry and field numerators at radial Gauss points."""

    fraction: Array
    s: Array
    radius_scale: Array
    d_radius_scale_ds: Array
    radius: Array
    radius_radius_s: Array
    jacobian: Array
    d_radius_dtheta: Array
    d_radius_dxi: Array
    field_theta_numerator: Array
    field_xi_numerator: Array


@dataclass(frozen=True)
class _HalfMeshForceFields:
    """Cell-centered fields used to reconstruct force on full surfaces."""

    jacobian: Array
    b_cov_theta: Array
    b_cov_xi: Array
    field_theta_numerator: Array
    field_xi_numerator: Array
    pressure: Array


@dataclass(frozen=True)
class _HalfMeshMetric:
    """Straight-axis metric terms at radial Gauss points."""

    jacobian: Array
    g_thetatheta: Array
    g_thetaxi: Array
    g_xixi: Array


def _interpolate_stream_function(lam: Array, grid: "MirrorGrid", fraction: Array) -> Array:
    """Interpolate lambda to radial Gauss points with regular axis modes.

    A single-valued scalar has poloidal mode ``m`` proportional to
    ``rho**abs(m)`` near the magnetic axis, where ``rho = sqrt(s)``. Linear
    interpolation acts on the smooth coefficient after removing that known
    factor, then restores it at each quadrature point.
    """

    if grid.ntheta == 1:
        return (1.0 - fraction) * lam[:-1][None] + fraction * lam[1:][None]

    modes = jnp.abs(jnp.fft.fftfreq(grid.ntheta, d=1.0 / grid.ntheta))
    power = 0.5 * modes[:, None]
    radial_s = jnp.asarray(grid.s, dtype=lam.dtype)
    safe_s = jnp.where(radial_s == 0.0, 1.0, radial_s)
    lam_modes = jnp.fft.fft(lam, axis=1)
    axis_offset = jnp.where(modes[:, None] == 0.0, 0.0, lam_modes[0])
    regular_modes = (lam_modes - axis_offset[None]) / safe_s[:, None, None] ** power[None]
    extrapolated_axis = 2.0 * regular_modes[1] - regular_modes[2]
    axis_modes = jnp.where(modes[:, None] == 0.0, regular_modes[0], extrapolated_axis)
    regular_modes = regular_modes.at[0].set(axis_modes)
    regular_quadrature = (
        (1.0 - fraction) * regular_modes[:-1][None]
        + fraction * regular_modes[1:][None]
    )
    s_quadrature = jnp.asarray(grid.s[:-1])[None, :, None, None] + (
        fraction * float(grid.s[1] - grid.s[0])
    )
    lambda_modes = axis_offset[None, None] + s_quadrature**power[None, None] * regular_quadrature
    return jnp.fft.ifft(lambda_modes, axis=2).real


def _interpolate_radius_scale(
    radius_scale: Array,
    grid: "MirrorGrid",
    fraction: Array,
) -> tuple[Array, Array]:
    """Interpolate the radius scale and derivative with axis regularity.

    Odd poloidal modes translate a centered section, so their leading radial
    behavior is proportional to ``rho = sqrt(s)``. Even modes can describe
    the limiting centered ellipse and remain finite on axis.
    """

    radius_scale = jnp.asarray(radius_scale)
    ds = float(grid.s[1] - grid.s[0])
    if grid.ntheta == 1:
        left, right = radius_scale[:-1][None], radius_scale[1:][None]
        return (1.0 - fraction) * left + fraction * right, (right - left) / ds

    modes = jnp.rint(jnp.fft.fftfreq(grid.ntheta, d=1.0 / grid.ntheta)).astype(int)
    power = 0.5 * (jnp.abs(modes) % 2)[:, None]
    radial_s = jnp.asarray(grid.s, dtype=radius_scale.dtype)
    safe_s = jnp.where(radial_s == 0.0, 1.0, radial_s)
    radius_modes = jnp.fft.fft(radius_scale, axis=1)
    regular_modes = radius_modes / safe_s[:, None, None] ** power[None]
    extrapolated_axis = 2.0 * regular_modes[1] - regular_modes[2]
    regular_modes = regular_modes.at[0].set(
        jnp.where(power == 0.0, regular_modes[0], extrapolated_axis)
    )
    regular_quadrature = (
        (1.0 - fraction) * regular_modes[:-1][None]
        + fraction * regular_modes[1:][None]
    )
    regular_derivative = (regular_modes[1:] - regular_modes[:-1])[None] / ds
    s_quadrature = jnp.asarray(grid.s[:-1])[None, :, None, None] + fraction * ds
    scale = s_quadrature ** power[None, None]
    scale_derivative = power[None, None] * s_quadrature ** (power[None, None] - 1.0)
    radius_quadrature = jnp.fft.ifft(scale * regular_quadrature, axis=2).real
    derivative_quadrature = jnp.fft.ifft(
        scale_derivative * regular_quadrature + scale * regular_derivative,
        axis=2,
    ).real
    return radius_quadrature, derivative_quadrature


def _half_mesh_samples(
    state: MirrorState, grid: "MirrorGrid", axial_flux_derivative: Array, current_derivative: Array
) -> _HalfMeshSamples:
    """Evaluate the primitive half-mesh quantities used by the energy."""

    state = regularize_axis_stream_function(state, grid, axial_flux_derivative)
    a = jnp.asarray(state.radius_scale)
    ds = float(grid.s[1] - grid.s[0])
    gauss = 0.5 + jnp.asarray([-1.0, 1.0]) / (2.0 * jnp.sqrt(3.0))
    fraction = gauss[:, None, None, None]
    s_left = jnp.asarray(grid.s[:-1])[None, :, None, None]
    s_quadrature = s_left + fraction * ds
    a_quadrature, da_ds = _interpolate_radius_scale(a, grid, fraction)
    radius = jnp.sqrt(jnp.maximum(s_quadrature * a_quadrature**2, 0.0))
    radius_radius_s = 0.5 * (a_quadrature**2 + 2.0 * s_quadrature * a_quadrature * da_ds)
    jacobian = radius_radius_s * float(grid.dz_dxi)
    d_radius_dtheta = grid.theta_basis.differentiate(radius, axis=2)
    d_radius_dxi = grid.axial_basis.differentiate(radius, axis=3)

    lam = jnp.asarray(state.lambda_stream)
    lambda_quadrature = _interpolate_stream_function(lam, grid, fraction)
    psi = _profile(
        axial_flux_derivative,
        grid.ns,
        a.dtype,
        name="axial_flux_derivative",
    )
    current = _profile(
        current_derivative,
        grid.ns,
        a.dtype,
        name="current_derivative",
    )
    psi_quadrature = ((1.0 - gauss[:, None]) * psi[:-1][None] + gauss[:, None] * psi[1:][None])[:, :, None, None]
    current_quadrature = ((1.0 - gauss[:, None]) * current[:-1][None] + gauss[:, None] * current[1:][None])[
        :, :, None, None
    ]
    return _HalfMeshSamples(
        fraction=fraction,
        s=s_quadrature,
        radius_scale=a_quadrature,
        d_radius_scale_ds=da_ds,
        radius=radius,
        radius_radius_s=radius_radius_s,
        jacobian=jacobian,
        d_radius_dtheta=d_radius_dtheta,
        d_radius_dxi=d_radius_dxi,
        field_theta_numerator=current_quadrature - grid.axial_basis.differentiate(lambda_quadrature, axis=3),
        field_xi_numerator=psi_quadrature + grid.theta_basis.differentiate(lambda_quadrature, axis=2),
    )


def _half_mesh_metric(
    samples: _HalfMeshSamples,
    grid: "MirrorGrid",
    axis: ClosedAxisGeometry | None = None,
) -> _HalfMeshMetric:
    """Build the straight- or closed-axis metric for all force kernels."""

    radius, radius_theta, radius_xi = samples.radius, samples.d_radius_dtheta, samples.d_radius_dxi
    if axis is not None:
        theta = jnp.asarray(grid.theta)[None, None, :, None, None]
        normal = jnp.asarray(axis.normal)[None, None, None]
        binormal = jnp.asarray(axis.binormal)[None, None, None]
        radial = jnp.cos(theta) * normal + jnp.sin(theta) * binormal
        poloidal = -jnp.sin(theta) * normal + jnp.cos(theta) * binormal
        normal_xi = grid.axial_basis.differentiate(jnp.asarray(axis.normal), axis=0)[None, None, None]
        binormal_xi = grid.axial_basis.differentiate(jnp.asarray(axis.binormal), axis=0)[None, None, None]
        radial_xi = jnp.cos(theta) * normal_xi + jnp.sin(theta) * binormal_xi
        centerline_xi = (jnp.asarray(axis.tangent) * jnp.asarray(axis.speed)[:, None])[None, None, None]
        e_theta = radius_theta[..., None] * radial + radius[..., None] * poloidal
        e_xi = centerline_xi + radius_xi[..., None] * radial + radius[..., None] * radial_xi
        orientation = jnp.sum(radial * jnp.cross(poloidal, e_xi), axis=-1)
        return _HalfMeshMetric(
            samples.radius_radius_s * orientation,
            jnp.sum(e_theta * e_theta, axis=-1),
            jnp.sum(e_theta * e_xi, axis=-1),
            jnp.sum(e_xi * e_xi, axis=-1),
        )
    return _HalfMeshMetric(
        samples.jacobian,
        radius_theta**2 + radius**2,
        radius_theta * radius_xi,
        radius_xi**2 + float(grid.dz_dxi) ** 2,
    )


def staggered_magnetic_terms(
    state: MirrorState,
    grid: "MirrorGrid",
    axial_flux_derivative: Array,
    current_derivative: Array,
    *,
    axis: ClosedAxisGeometry | None = None,
) -> tuple[Array, Array]:
    """Return radially integrated ``(B^2, J)`` on each half cell.

    Two-point Gauss integration prevents the alternating full-mesh lambda
    mode that one-point midpoint quadrature cannot see. The returned arrays
    retain shape ``(ns-1, ntheta, nxi)``: ``J`` is the cell-average Jacobian
    and ``B^2`` is weighted so ``B^2*J`` is the cell-average magnetic density.
    """

    samples = _half_mesh_samples(
        state,
        grid,
        axial_flux_derivative,
        current_derivative,
    )
    metric = _half_mesh_metric(samples, grid, axis)
    jacobian = metric.jacobian
    b_theta = samples.field_theta_numerator / jacobian
    b_xi = samples.field_xi_numerator / jacobian
    b_squared = (
        metric.g_thetatheta * b_theta**2
        + 2.0 * metric.g_thetaxi * b_theta * b_xi
        + metric.g_xixi * b_xi**2
    )
    jacobian_cell = jnp.mean(jacobian, axis=0)
    magnetic_density_cell = jnp.mean(b_squared * jacobian, axis=0)
    return magnetic_density_cell / jacobian_cell, jacobian_cell


def _half_mesh_force_fields(
    state: MirrorState,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
    mass_profile: Array,
    current_derivative: Array,
    gamma: float,
    axis: ClosedAxisGeometry | None = None,
) -> _HalfMeshForceFields:
    """Evaluate covariant field and pressure on radial energy cells."""

    samples = _half_mesh_samples(
        state,
        grid,
        axial_flux_derivative,
        current_derivative,
    )
    metric = _half_mesh_metric(samples, grid, axis)
    jacobian = metric.jacobian
    b_theta = samples.field_theta_numerator / jacobian
    b_xi = samples.field_xi_numerator / jacobian
    b_cov_theta = metric.g_thetatheta * b_theta + metric.g_thetaxi * b_xi
    b_cov_xi = metric.g_thetaxi * b_theta + metric.g_xixi * b_xi
    jacobian_cell = jnp.mean(jacobian, axis=0)
    mass = _profile(mass_profile, grid.ns, jacobian.dtype, name="mass_profile")
    volume_derivative = _surface_integral(jacobian_cell, grid)
    pressure = 0.5 * (mass[:-1] + mass[1:]) / volume_derivative ** float(gamma)
    return _HalfMeshForceFields(
        jacobian=jacobian_cell,
        b_cov_theta=jnp.mean(b_cov_theta, axis=0),
        b_cov_xi=jnp.mean(b_cov_xi, axis=0),
        field_theta_numerator=jnp.mean(samples.field_theta_numerator, axis=0),
        field_xi_numerator=jnp.mean(samples.field_xi_numerator, axis=0),
        pressure=pressure,
    )


def staggered_field_strength(
    state: MirrorState,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
    current_derivative: Array = 0.0,
) -> Array:
    """Reconstruct full-surface ``|B|`` from the radial Gauss energy kernel."""

    b_squared, _ = staggered_magnetic_terms(
        state,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    half = jnp.sqrt(jnp.maximum(b_squared, 0.0))
    first = 1.5 * half[0] - 0.5 * half[1]
    interior = 0.5 * (half[:-1] + half[1:])
    last = 1.5 * half[-1] - 0.5 * half[-2]
    return jnp.concatenate((first[None], interior, last[None]), axis=0)


def isotropic_staggered_energy_gradient(
    state: MirrorState,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    mu0: float = float(MU0),
    axis: ClosedAxisGeometry | None = None,
) -> MirrorState:
    """Assemble the isotropic discrete first variation without autodiff.

    The calculation reverses the radial Gauss interpolation and the axial and
    poloidal derivative matrices explicitly. It is an independent check of the
    differentiated energy, not the residual used by the nonlinear solver.
    """

    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    if axis is not None:
        return jax.grad(
            lambda trial: mirror_energy(
                trial,
                grid,
                axial_flux_derivative=axial_flux_derivative,
                mass_profile=mass_profile,
                current_derivative=current_derivative,
                gamma=gamma,
                mu0=mu0,
                axis=axis,
            ).total
        )(state)
    samples = _half_mesh_samples(
        state,
        grid,
        axial_flux_derivative,
        current_derivative,
    )
    radius = samples.radius
    radius_theta = samples.d_radius_dtheta
    radius_xi = samples.d_radius_dxi
    field_theta = samples.field_theta_numerator
    field_xi = samples.field_xi_numerator
    metric = _half_mesh_metric(samples, grid)
    jacobian = metric.jacobian
    g_thetatheta = metric.g_thetatheta
    g_thetaxi = metric.g_thetaxi
    g_xixi = metric.g_xixi
    numerator = g_thetatheta * field_theta**2 + 2.0 * g_thetaxi * field_theta * field_xi + g_xixi * field_xi**2

    surface_weights = (
        jnp.asarray(grid.theta_basis.weights)[None, None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, None, :]
    )
    ds = float(grid.s[1] - grid.s[0])
    sample_weights = 0.5 * ds * surface_weights
    jacobian_cell = jnp.mean(jacobian, axis=0)
    volume_derivative_half = _surface_integral(jacobian_cell, grid)
    mass = _profile(
        mass_profile,
        grid.ns,
        numerator.dtype,
        name="mass_profile",
    )
    pressure_half = 0.5 * (mass[1:] + mass[:-1]) / (volume_derivative_half ** float(gamma))

    numerator_bar = sample_weights / (2.0 * float(mu0) * jacobian)
    jacobian_bar = sample_weights * (-numerator / (2.0 * float(mu0) * jacobian**2) - pressure_half[None, :, None, None])
    field_theta_bar = numerator_bar * (2.0 * g_thetatheta * field_theta + 2.0 * g_thetaxi * field_xi)
    field_xi_bar = numerator_bar * (2.0 * g_thetaxi * field_theta + 2.0 * g_xixi * field_xi)

    radius_bar = numerator_bar * (2.0 * radius * field_theta**2)
    radius_theta_bar = numerator_bar * (
        2.0 * radius_theta * field_theta**2 + 2.0 * radius_xi * field_theta * field_xi
    )
    radius_xi_bar = numerator_bar * (
        2.0 * radius_theta * field_theta * field_xi + 2.0 * radius_xi * field_xi**2
    )
    radius_radius_s_bar = jacobian_bar * float(grid.dz_dxi)

    def radial_primitives(trial: MirrorState) -> tuple[Array, ...]:
        trial_samples = _half_mesh_samples(trial, grid, axial_flux_derivative, current_derivative)
        return (
            trial_samples.radius, trial_samples.radius_radius_s,
            trial_samples.d_radius_dtheta, trial_samples.d_radius_dxi,
            trial_samples.field_theta_numerator,
            trial_samples.field_xi_numerator,
        )

    _, pullback = jax.vjp(radial_primitives, state)
    primitive_bars = (
        radius_bar,
        radius_radius_s_bar,
        radius_theta_bar,
        radius_xi_bar,
        field_theta_bar,
        field_xi_bar,
    )
    return pullback(primitive_bars)[0]


@dataclass(frozen=True)
class MirrorEnergy:
    """Total, magnetic, and pressure energy with radial profiles."""

    total: Array
    magnetic: Array
    pressure_energy: Array
    volume_derivative: Array
    pressure: Array
    b_squared: Array
    geometry: MirrorGeometry
    field: ContravariantField


@dataclass(frozen=True)
class IsotropicForceResidual:
    """Covariant physical force components and convergence diagnostics.

    ``normalized_rms`` and the regional norms use the minor-radius force scale
    ``B^2/(mu0 a)``, which is comparable across open and closed lanes.
    ``device_normalized_rms`` keeps the legacy device-length normalization
    ``B^2/(mu0 L)`` quoted by the recorded benchmark JSONs.
    """

    covariant_s: Array
    covariant_theta: Array
    covariant_xi: Array
    current_sup_s: Array
    current_sup_theta: Array
    current_sup_xi: Array
    physical_rms: Array
    normalized_rms: Array
    bulk_normalized_rms: Array
    axis_normalized_rms: Array
    first_row_normalized_rms: Array
    end_collar_normalized_rms: Array
    device_normalized_rms: Array
    minor_radius: Array
    axis_field_nonuniformity: Array
    component_rms: Array


@dataclass(frozen=True)
class VariationalResidual:
    """Nondimensional generalized forces used for nonlinear convergence."""

    radius_gradient: Array
    lambda_gradient: Array
    radius_rms: Array
    lambda_rms: Array
    maximum: Array


@dataclass(frozen=True)
class InterfaceResidual:
    """Lateral plasma-vacuum tangency and normal-stress residuals."""

    plasma_b_normal_rms: Array
    vacuum_b_normal_rms: Array
    normal_stress_rms: Array
    normal_stress_jump: Array


for _cls in (
    MirrorEnergy,
    IsotropicForceResidual,
    VariationalResidual,
    InterfaceResidual,
):
    jax.tree_util.register_dataclass(
        _cls,
        data_fields=[field.name for field in fields(_cls)],
        meta_fields=[],
    )


def mass_profile_from_pressure(
    pressure: Array,
    volume_derivative: Array,
    *,
    gamma: float = 5.0 / 3.0,
) -> Array:
    """Convert a reference pressure into VMEC's conserved mass profile."""

    pressure = jnp.asarray(pressure)
    volume_derivative = jnp.asarray(volume_derivative, dtype=pressure.dtype)
    return pressure * volume_derivative ** float(gamma)


def mirror_energy(
    state: MirrorState,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    mu0: float = float(MU0),
    axis: ClosedAxisGeometry | None = None,
) -> MirrorEnergy:
    """Evaluate mass-conserving isotropic mirror energy."""

    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    state = regularize_axis_stream_function(state, grid, axial_flux_derivative)
    geometry = evaluate_geometry(state, grid) if axis is None else evaluate_closed_geometry(state, grid, axis)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    b_squared = magnetic_field_squared(field, geometry)
    volume_derivative = _surface_integral(geometry.sqrt_g, grid)
    mass = _profile(mass_profile, grid.ns, b_squared.dtype, name="mass_profile")
    pressure = mass / volume_derivative ** float(gamma)

    b_squared_half, jacobian_half = staggered_magnetic_terms(
        state,
        grid,
        axial_flux_derivative,
        current_derivative,
        axis=axis,
    )
    volume_derivative_half = _surface_integral(jacobian_half, grid)
    mass_half = 0.5 * (mass[1:] + mass[:-1])
    pressure_half = mass_half / volume_derivative_half ** float(gamma)
    magnetic_surface = _surface_integral(b_squared_half * jacobian_half, grid) / (2.0 * float(mu0))
    pressure_surface = pressure_half * volume_derivative_half / (float(gamma) - 1.0)
    ds = float(grid.s[1] - grid.s[0])
    magnetic = ds * jnp.sum(magnetic_surface)
    pressure_energy = ds * jnp.sum(pressure_surface)
    return MirrorEnergy(
        total=magnetic + pressure_energy,
        magnetic=magnetic,
        pressure_energy=pressure_energy,
        volume_derivative=volume_derivative,
        pressure=pressure,
        b_squared=b_squared,
        geometry=geometry,
        field=field,
    )


def interface_residual(
    *,
    pressure: Array,
    plasma_b_squared: Array,
    vacuum_b_squared: Array,
    plasma_b_normal: Array,
    vacuum_b_normal: Array,
    theta_weights: Array,
    axial_weights: Array,
    mu0: float = MU0,
) -> InterfaceResidual:
    """Evaluate isotropic free-boundary interface conditions.

    The lateral interface requires both fields to be tangent and
    ``p + B_plasma^2/(2*mu0) = B_vacuum^2/(2*mu0)``.
    Inputs are sampled on ``(theta, xi)``.
    """

    pressure, bp2, bv2, bnp, bnv = jnp.broadcast_arrays(
        pressure,
        plasma_b_squared,
        vacuum_b_squared,
        plasma_b_normal,
        vacuum_b_normal,
    )
    weights = jnp.asarray(theta_weights)[:, None] * jnp.asarray(axial_weights)[None, :]
    denominator = jnp.sum(weights)
    plasma_b_normal_rms = jnp.sqrt(
        jnp.sum(weights * bnp**2 / jnp.maximum(bp2, jnp.finfo(bp2.dtype).tiny)) / denominator
    )
    vacuum_b_normal_rms = jnp.sqrt(
        jnp.sum(weights * bnv**2 / jnp.maximum(bv2, jnp.finfo(bv2.dtype).tiny)) / denominator
    )
    jump = pressure + bp2 / (2.0 * float(mu0)) - bv2 / (2.0 * float(mu0))
    stress_scale = jnp.abs(pressure) + bp2 / (2.0 * float(mu0)) + bv2 / (2.0 * float(mu0))
    normal_stress_rms = jnp.sqrt(
        jnp.sum(weights * (jump / jnp.maximum(stress_scale, jnp.finfo(bp2.dtype).tiny)) ** 2) / denominator
    )
    return InterfaceResidual(
        plasma_b_normal_rms=plasma_b_normal_rms,
        vacuum_b_normal_rms=vacuum_b_normal_rms,
        normal_stress_rms=normal_stress_rms,
        normal_stress_jump=jump,
    )


def isotropic_staggered_fixed_boundary_gradient(
    state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    **energy_kwargs: Any,
) -> MirrorState:
    """Return the manually assembled gradient on the constrained state space."""

    projected, pullback = jax.vjp(
        lambda trial: project_fixed_boundary_state(trial, boundary, grid),
        state,
    )
    gradient = isotropic_staggered_energy_gradient(
        projected,
        grid,
        **energy_kwargs,
    )
    return pullback(gradient)[0]


def isotropic_staggered_weak_residual(
    state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    **energy_kwargs: Any,
) -> VariationalResidual:
    """Normalize the independent staggered first variation like ``fsq``."""

    projected = project_fixed_boundary_state(state, boundary, grid)
    energy = mirror_energy(projected, grid, **energy_kwargs)
    gradient = isotropic_staggered_fixed_boundary_gradient(
        projected,
        boundary,
        grid,
        **energy_kwargs,
    )
    return _normalized_variational_residual(
        gradient,
        energy.total,
        boundary,
        grid,
        axial_flux_derivative=energy_kwargs["axial_flux_derivative"],
    )


def _normalized_variational_residual(
    gradient: MirrorState,
    energy_total: Array,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
) -> VariationalResidual:
    """Scale an energy gradient into component-wise mirror force norms."""

    energy_scale = jnp.maximum(jnp.abs(energy_total), jnp.finfo(jnp.asarray(energy_total).dtype).tiny)
    radius_scale = jnp.mean(jnp.asarray(boundary.radius_scale))
    radius_gradient = gradient.radius_scale * radius_scale / energy_scale

    psi = _profile(
        axial_flux_derivative,
        grid.ns,
        jnp.asarray(energy_total).dtype,
        name="axial_flux_derivative",
    )
    flux_scale = jnp.maximum(jnp.max(jnp.abs(psi)), jnp.finfo(psi.dtype).tiny)
    lambda_gradient = gradient.lambda_stream * flux_scale / energy_scale
    radius_free = radius_gradient[1:-1, :, 1:-1]
    radius_rms = jnp.sqrt(jnp.mean(radius_free**2))
    lambda_rms = jnp.sqrt(jnp.mean(lambda_gradient**2))
    maximum = jnp.maximum(jnp.max(jnp.abs(radius_free)), jnp.max(jnp.abs(lambda_gradient)))
    return VariationalResidual(
        radius_gradient=radius_gradient,
        lambda_gradient=lambda_gradient,
        radius_rms=radius_rms,
        lambda_rms=lambda_rms,
        maximum=maximum,
    )


def effective_minor_radius(
    state: MirrorState,
    grid: "MirrorGrid",
    *,
    closed: bool = False,
) -> Array:
    """Return the flux-equivalent LCFS minor radius of a mirror state.

    Open mirrors use the midplane LCFS cross-section; closed hybrids average
    the section around the full circuit, where no midplane exists. The
    quadratic mean ``sqrt(<r^2>)`` reproduces a circular radius exactly and
    the area-equivalent ``sqrt(A B)`` for an elliptical polar section.
    """

    lcfs = jnp.asarray(state.radius_scale)[-1]
    theta_weights = jnp.asarray(grid.theta_basis.weights)
    if closed:
        axial_weights = jnp.asarray(grid.axial_basis.weights)
        weights = theta_weights[:, None] * axial_weights[None, :]
        return jnp.sqrt(jnp.sum(weights * lcfs**2) / jnp.sum(weights))
    midplane = int(abs(grid.z - 0.5 * (grid.z[0] + grid.z[-1])).argmin())
    section = lcfs[:, midplane]
    return jnp.sqrt(jnp.sum(theta_weights * section**2) / jnp.sum(theta_weights))


def isotropic_force_residual(
    energy: MirrorEnergy,
    grid: "MirrorGrid",
    *,
    state: MirrorState,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    axis: ClosedAxisGeometry | None = None,
    closed: bool = False,
    characteristic_length: float | Array | None = None,
    minor_radius: float | Array | None = None,
    mu0: float = float(MU0),
) -> IsotropicForceResidual:
    """Reconstruct ``curl(B)/mu0 x B - grad(p)`` on full radial surfaces.

    Covariant field and pressure are first evaluated on the same radial Gauss
    cells as the energy. Radial differences then place current and force on
    interior full surfaces, following VMEC's half-to-full ``jxbforce`` layout.

    The primary ``normalized_rms`` and every regional norm divide the force
    density by the transverse magnetic force scale ``B^2/(mu0 a)``, where the
    minor radius ``a`` defaults to :func:`effective_minor_radius`. This scale
    is structural (each lane has one minor radius), so open mirrors and closed
    hybrids are gated on the same footing. The legacy device-length number,
    ``B^2/(mu0 L)`` with ``L`` from ``characteristic_length`` (open default:
    the cap-to-cap extent; closed callers pass the axis arc length), is linear
    in the arbitrary device length and is kept only as
    ``device_normalized_rms`` because recorded benchmarks quote it.
    """

    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    if closed != (axis is not None):
        raise ValueError("closed force reconstruction requires its closed axis")

    geometry, field = energy.geometry, energy.field
    half = _half_mesh_force_fields(
        state,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        mass_profile=mass_profile,
        current_derivative=current_derivative,
        gamma=gamma,
        axis=axis,
    )
    ds = float(grid.s[1] - grid.s[0])
    jacobian = 0.5 * (half.jacobian[:-1] + half.jacobian[1:])
    field_theta_numerator = 0.5 * (
        half.field_theta_numerator[:-1] + half.field_theta_numerator[1:]
    )
    field_xi_numerator = 0.5 * (half.field_xi_numerator[:-1] + half.field_xi_numerator[1:])
    b_sup_theta = field_theta_numerator / jacobian
    b_sup_xi = field_xi_numerator / jacobian
    b_cov_theta = 0.5 * (half.b_cov_theta[:-1] + half.b_cov_theta[1:])
    b_cov_xi = 0.5 * (half.b_cov_xi[:-1] + half.b_cov_xi[1:])
    b_cov_s_full = (
        geometry.g_stheta * field.b_sup_theta + geometry.g_sxi * field.b_sup_xi
    )
    b_cov_s = b_cov_s_full[1:-1]

    current_s_numerator = grid.theta_basis.differentiate(b_cov_xi, axis=1)
    current_s_numerator -= grid.axial_basis.differentiate(b_cov_theta, axis=2)
    current_theta_numerator = grid.axial_basis.differentiate(b_cov_s, axis=2)
    current_theta_numerator -= (half.b_cov_xi[1:] - half.b_cov_xi[:-1]) / ds
    current_xi_numerator = (half.b_cov_theta[1:] - half.b_cov_theta[:-1]) / ds
    current_xi_numerator -= grid.theta_basis.differentiate(b_cov_s, axis=1)

    pressure_gradient_s = (half.pressure[1:] - half.pressure[:-1])[:, None, None] / ds
    force_s_interior = (
        current_theta_numerator * b_sup_xi - current_xi_numerator * b_sup_theta
    ) / float(mu0) - pressure_gradient_s
    force_theta_interior = -current_s_numerator * b_sup_xi / float(mu0)
    force_xi_interior = current_s_numerator * b_sup_theta / float(mu0)

    def extend_radially(interior: Array) -> Array:
        return jnp.concatenate((interior[:1], interior, interior[-1:]), axis=0)

    force_s = extend_radially(force_s_interior)
    force_theta = extend_radially(force_theta_interior)
    force_xi = extend_radially(force_xi_interior)
    inverse_mu0_jacobian = 1.0 / (float(mu0) * jacobian)
    current_s = extend_radially(current_s_numerator * inverse_mu0_jacobian)
    current_theta = extend_radially(current_theta_numerator * inverse_mu0_jacobian)
    current_xi = extend_radially(current_xi_numerator * inverse_mu0_jacobian)

    metric = jnp.stack(
        [
            jnp.stack([geometry.g_ss, geometry.g_stheta, geometry.g_sxi], axis=-1),
            jnp.stack([geometry.g_stheta, geometry.g_thetatheta, geometry.g_thetaxi], axis=-1),
            jnp.stack([geometry.g_sxi, geometry.g_thetaxi, geometry.g_xixi], axis=-1),
        ],
        axis=-2,
    )
    force_covariant = jnp.stack([force_s, force_theta, force_xi], axis=-1)
    axial_slice = slice(None) if closed else slice(1, -1)
    # The axis and side are constrained. Open mirrors also constrain end cuts.
    force_active = force_covariant[1:-1, :, axial_slice]
    inverse_metric = jnp.linalg.inv(metric[1:-1, :, axial_slice])
    force_squared = jnp.einsum(
        "...i,...ij,...j->...",
        force_active,
        inverse_metric,
        force_active,
    )
    weights = (
        jnp.asarray(grid.radial_weights[1:-1])[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, axial_slice]
        * geometry.sqrt_g[1:-1, :, axial_slice]
    )
    physical_rms = jnp.sqrt(jnp.sum(weights * force_squared) / jnp.sum(weights))
    component_rms = jnp.sqrt(jnp.sum(weights[..., None] * force_active**2, axis=(0, 1, 2)) / jnp.sum(weights))
    length = (
        jnp.asarray(characteristic_length)
        if characteristic_length is not None
        else jnp.asarray(float(grid.z[-1] - grid.z[0]))
    )
    radius_scale_reference = (
        jnp.asarray(minor_radius)
        if minor_radius is not None
        else effective_minor_radius(state, grid, closed=closed)
    )
    magnetic_force_scale = energy.b_squared[1:-1, :, axial_slice] / (float(mu0) * radius_scale_reference)
    reference_rms = jnp.sqrt(jnp.sum(weights * magnetic_force_scale**2) / jnp.sum(weights))
    normalized_rms = physical_rms / jnp.maximum(reference_rms, jnp.finfo(physical_rms.dtype).tiny)
    device_reference_rms = reference_rms * radius_scale_reference / length
    device_normalized_rms = physical_rms / jnp.maximum(
        device_reference_rms, jnp.finfo(physical_rms.dtype).tiny
    )
    active_s = jnp.asarray(grid.s[1:-1])

    def regional_normalized_rms(radial_mask: Array, axial_mask: Array = 1.0) -> Array:
        axial_mask = jnp.broadcast_to(jnp.asarray(axial_mask), (force_active.shape[2],))
        mask = jnp.asarray(radial_mask)[:, None, None] * axial_mask[None, None, :]
        regional_weights = weights * mask
        denominator = jnp.maximum(jnp.sum(regional_weights), jnp.finfo(physical_rms.dtype).tiny)
        regional_force = jnp.sqrt(jnp.sum(regional_weights * force_squared) / denominator)
        regional_reference = jnp.sqrt(jnp.sum(regional_weights * magnetic_force_scale**2) / denominator)
        return regional_force / jnp.maximum(regional_reference, jnp.finfo(physical_rms.dtype).tiny)

    active_xi = jnp.arange(force_active.shape[2])
    end_collar = (
        jnp.zeros_like(active_xi, dtype=bool)
        if closed
        else jnp.abs(jnp.asarray(grid.xi)[1:-1]) >= 0.8
    )
    axial_core = ~end_collar
    bulk_normalized_rms = regional_normalized_rms(active_s >= 0.2, axial_core)
    axis_normalized_rms = regional_normalized_rms(
        (active_s < 0.2) | (jnp.arange(active_s.size) == 0), axial_core
    )
    first_row_normalized_rms = regional_normalized_rms(jnp.arange(active_s.size) == 0, axial_core)
    end_collar_normalized_rms = regional_normalized_rms(jnp.ones_like(active_s), end_collar)
    axis_field = jnp.sqrt(jnp.maximum(energy.b_squared[0], 0.0))
    theta_weights = jnp.asarray(grid.theta_basis.weights)[:, None]
    axis_mean = jnp.sum(theta_weights * axis_field, axis=0) / jnp.sum(theta_weights)
    axis_variance = jnp.sum(theta_weights * (axis_field - axis_mean[None, :]) ** 2, axis=0) / jnp.sum(theta_weights)
    axis_axial_weights = jnp.asarray(grid.axial_basis.weights).at[jnp.asarray([0, -1])].set(0.0)
    axis_field_nonuniformity = jnp.sqrt(
        jnp.sum(axis_axial_weights * axis_variance) / jnp.sum(axis_axial_weights * axis_mean**2)
    )
    return IsotropicForceResidual(
        covariant_s=force_s,
        covariant_theta=force_theta,
        covariant_xi=force_xi,
        current_sup_s=current_s,
        current_sup_theta=current_theta,
        current_sup_xi=current_xi,
        physical_rms=physical_rms,
        normalized_rms=normalized_rms,
        bulk_normalized_rms=bulk_normalized_rms,
        axis_normalized_rms=axis_normalized_rms,
        first_row_normalized_rms=first_row_normalized_rms,
        end_collar_normalized_rms=end_collar_normalized_rms,
        device_normalized_rms=device_normalized_rms,
        minor_radius=radius_scale_reference,
        axis_field_nonuniformity=axis_field_nonuniformity,
        component_rms=component_rms,
    )


@dataclass(frozen=True)
class ForceGateZones:
    """Zone breakdown of the strong-force gate under the primary normalization.

    ``bulk`` covers the unconstrained volume (``s >= 0.2``, central 80% of the
    axial coordinate for open mirrors). ``end_collar`` is the outer 20%
    nearest the two frozen cuts (empty for closed hybrids), and ``axis_row``
    and ``first_row`` cover the constrained near-axis region, so constrained
    zones are visible instead of being folded into ``all_volume``.
    ``device_all_volume`` restates the total under the legacy device-length
    normalization recorded by the benchmark JSONs.
    """

    all_volume: float
    bulk: float
    end_collar: float
    axis_row: float
    first_row: float
    device_all_volume: float
    minor_radius: float


def force_gate_zones(residual: IsotropicForceResidual) -> ForceGateZones:
    """Summarize a force residual as the zone report used by gate evaluation."""

    return ForceGateZones(
        all_volume=float(residual.normalized_rms),
        bulk=float(residual.bulk_normalized_rms),
        end_collar=float(residual.end_collar_normalized_rms),
        axis_row=float(residual.axis_normalized_rms),
        first_row=float(residual.first_row_normalized_rms),
        device_all_volume=float(residual.device_normalized_rms),
        minor_radius=float(residual.minor_radius),
    )


@dataclass(frozen=True)
class RefinementConvergence:
    """Per-step refinement behaviour of a residual sequence.

    ``ratios[i] = residuals[i] / residuals[i + 1]``, so every ratio above one
    means the residual fell at that refinement step. ``monotone`` requires a
    strict decrease at every step.
    """

    residuals: tuple[float, ...]
    ratios: tuple[float, ...]
    monotone: bool


def refinement_convergence(residuals: Sequence[float] | Array) -> RefinementConvergence:
    """Report per-step ratios and monotonicity of a coarse-to-fine sequence."""

    values = tuple(float(value) for value in residuals)
    if len(values) < 2:
        raise ValueError("refinement convergence needs residuals from at least two resolutions")
    if any(value <= 0.0 or value != value for value in values):
        raise ValueError("refinement residuals must be positive and finite")
    ratios = tuple(coarse / fine for coarse, fine in zip(values[:-1], values[1:], strict=True))
    return RefinementConvergence(
        residuals=values,
        ratios=ratios,
        monotone=all(ratio > 1.0 for ratio in ratios),
    )


def passes_promotion_gate(
    residuals: Sequence[float] | Array,
    *,
    absolute_gate: float,
) -> bool:
    """Promotion criterion: finest residual under the gate and monotone refinement."""

    if absolute_gate <= 0.0:
        raise ValueError("absolute_gate must be positive")
    convergence = refinement_convergence(residuals)
    return convergence.monotone and convergence.residuals[-1] <= absolute_gate


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
