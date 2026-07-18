"""Straight-axis mirror geometry and divergence-free magnetic field.

The initial embedding is

``x = r cos(theta), y = r sin(theta), z = z(xi)``,
``r = sqrt(s) * a(s, theta, xi)``.

Using the regular scale ``a`` keeps ``r * d(r)/ds`` and the Jacobian finite at
the magnetic axis.  The contravariant field follows the VMEC construction and
is discretely divergence-free when theta and xi derivatives commute.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = Any


def radial_derivative(values: Array, spacing: float) -> Array:
    """Second-order derivative on the uniform full radial mesh."""

    values = jnp.asarray(values)
    first = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * spacing)
    interior = (values[2:] - values[:-2]) / (2.0 * spacing)
    last = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * spacing)
    return jnp.concatenate([first[None], interior, last[None]], axis=0)


def _safe_divide(numerator: Array, denominator: Array) -> Array:
    denominator = jnp.asarray(denominator)
    mask = jnp.abs(denominator) > 32.0 * jnp.finfo(denominator.dtype).eps
    safe = jnp.where(mask, denominator, jnp.ones_like(denominator))
    return jnp.where(mask, numerator / safe, jnp.zeros_like(numerator))


@dataclass(frozen=True)
class MirrorGeometry:
    """Embedding, covariant metric, Jacobian, and volume on the full grid."""

    xyz: Array
    radius: Array
    d_radius_ds_regular: Array
    d_radius_dtheta: Array
    d_radius_dxi: Array
    g_ss: Array
    g_stheta: Array
    g_sxi: Array
    g_thetatheta: Array
    g_thetaxi: Array
    g_xixi: Array
    sqrt_g: Array
    volume: Array
    jacobian_sign_changed: Array
    e_s_xyz: Array
    e_theta_xyz: Array
    e_xi_xyz: Array


@dataclass(frozen=True)
class ContravariantField:
    """Mirror magnetic field in contravariant and flux-density form."""

    b_sup_s: Array
    b_sup_theta: Array
    b_sup_xi: Array
    jac_b_theta: Array
    jac_b_xi: Array


@dataclass(frozen=True)
class ClosedAxisGeometry:
    """Periodic centerline and its closure-corrected normal frame."""

    centerline: Array
    tangent: Array
    normal: Array
    binormal: Array
    speed: Array
    curvature: Array
    arc_length: Array
    frame_holonomy: Array
    closure_error: Array
    tangent_closure_error: Array
    frame_closure_error: Array


for _cls in (MirrorGeometry, ContravariantField, ClosedAxisGeometry):
    jax.tree_util.register_dataclass(
        _cls,
        data_fields=[field.name for field in fields(_cls)],
        meta_fields=[],
    )


def _hybrid_segments(basis: Any) -> tuple[np.ndarray, np.ndarray]:
    """Label the two straight legs and two returns at spline controls."""

    if not getattr(basis, "periodic", False):
        raise ValueError("stellarator-mirror geometry requires a periodic spline")
    if basis.size < 16 or basis.size % 4:
        raise ValueError("coefficient count must be a multiple of four and at least 16")
    start, stop = basis.domain
    if not np.isclose(stop - start, 2.0 * np.pi):
        raise ValueError("stellarator-mirror parameter must have period 2*pi")
    phase = np.mod(
        np.asarray(basis.collocation_nodes) - start + 4.0 * np.pi / basis.size,
        2.0 * np.pi,
    )
    coordinate = np.mod(phase + 0.25 * np.pi, 2.0 * np.pi) / (0.5 * np.pi)
    segment = np.floor(coordinate).astype(int)
    return segment, coordinate - segment


def stellarator_mirror_axis_coefficients(
    basis: Any,
    *,
    straight_length: float,
    return_radius: float,
) -> Array:
    """Build a closed racetrack axis with exactly straight central leg spans.

    The spline controls are divided equally between a straight leg, a curved
    return, the opposite straight leg, and the second return. Cubic local
    support makes every span backed only by collinear leg controls exactly
    straight; the neighboring spans provide a C2 transition into each return.
    """

    straight_length = float(straight_length)
    return_radius = float(return_radius)
    if straight_length <= 0.0 or return_radius <= 0.0:
        raise ValueError("axis dimensions must be positive")
    segment, fraction = _hybrid_segments(basis)
    half = 0.5 * straight_length
    top_angle = np.pi * fraction
    bottom_angle = np.pi * (1.0 + fraction)
    coefficients = np.empty((basis.size, 3))
    coefficients[:, 1] = 0.0
    coefficients[:, 0] = np.select(
        (segment == 0, segment == 1, segment == 2, segment == 3),
        (
            return_radius,
            return_radius * np.cos(top_angle),
            -return_radius,
            return_radius * np.cos(bottom_angle),
        ),
    )
    coefficients[:, 2] = np.select(
        (segment == 0, segment == 1, segment == 2, segment == 3),
        (
            straight_length * (fraction - 0.5),
            half + return_radius * np.sin(top_angle),
            half - straight_length * fraction,
            -half + return_radius * np.sin(bottom_angle),
        ),
    )
    return jnp.asarray(coefficients)


def stellarator_mirror_section_coefficients(
    basis: Any,
    theta: Array,
    *,
    semi_major: float,
    semi_minor: float,
) -> Array:
    """Build ellipse radii that rotate 90 degrees only through the returns."""

    semi_major, semi_minor = float(semi_major), float(semi_minor)
    if semi_major <= 0.0 or semi_minor <= 0.0:
        raise ValueError("ellipse semiaxes must be positive")
    theta = jnp.asarray(theta)
    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    segment, fraction = _hybrid_segments(basis)
    fraction = jnp.asarray(fraction)
    transition = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
    angle = jnp.select(
        tuple(jnp.asarray(segment) == value for value in range(4)),
        (0.0, 0.5 * jnp.pi * transition, 0.5 * jnp.pi, 0.5 * jnp.pi * (1.0 - transition)),
    )
    local_angle = theta[:, None] - angle[None]
    return semi_major * semi_minor / jnp.sqrt(
        (semi_minor * jnp.cos(local_angle)) ** 2
        + (semi_major * jnp.sin(local_angle)) ** 2
    )


def _minimal_rotation(vector: Array, tangent_from: Array, tangent_to: Array) -> Array:
    cross = jnp.cross(tangent_from, tangent_to)
    cosine = jnp.clip(jnp.dot(tangent_from, tangent_to), -1.0, 1.0)
    denominator = jnp.maximum(1.0 + cosine, 64.0 * jnp.finfo(cosine.dtype).eps)
    return vector + jnp.cross(cross, vector) + jnp.cross(cross, jnp.cross(cross, vector)) / denominator


def _rotate_about_axis(vector: Array, axis: Array, angle: Array) -> Array:
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    return (
        vector * cosine
        + jnp.cross(axis, vector) * sine
        + axis * jnp.dot(axis, vector) * (1.0 - cosine)
    )


def evaluate_closed_spline_axis(
    coefficients: Array,
    basis: Any,
    points: Array,
    *,
    initial_normal: Array | None = None,
) -> ClosedAxisGeometry:
    """Evaluate a periodic spline axis and rotation-minimizing frame.

    Parallel transport avoids the undefined Frenet frame on straight legs.
    The accumulated holonomy is spread smoothly around the period so the
    normal frame closes exactly.
    """

    if not getattr(basis, "periodic", False):
        raise ValueError("closed centerline requires a periodic spline basis")
    coefficients = jnp.asarray(coefficients)
    if coefficients.shape != (basis.size, 3):
        raise ValueError(f"centerline coefficients must have shape ({basis.size}, 3)")
    point_values = np.asarray(points, dtype=float)
    start, stop = basis.domain
    if point_values.ndim != 1 or point_values.size < 4:
        raise ValueError("closed centerline points must be a one-dimensional array of length >= 4")
    if (
        np.any(np.diff(point_values) <= 0.0)
        or point_values[0] < start
        or point_values[-1] >= stop
    ):
        raise ValueError("closed centerline points must increase within one period")

    period = stop - start
    extended = jnp.asarray(np.concatenate((point_values, [point_values[0] + period])))
    centerline = basis.evaluate(coefficients, extended, axis=0)
    first = basis.evaluate(coefficients, extended, derivative=1, axis=0)
    second = basis.evaluate(coefficients, extended, derivative=2, axis=0)
    speed = jnp.linalg.norm(first, axis=-1)
    if not isinstance(speed, jax.core.Tracer) and bool(jnp.any(speed <= 0.0)):
        raise ValueError("centerline derivative must not vanish")
    tangent = first / speed[:, None]
    if initial_normal is None:
        reference = jax.nn.one_hot(jnp.argmin(jnp.abs(tangent[0])), 3, dtype=tangent.dtype)
    else:
        reference = jnp.asarray(initial_normal, dtype=tangent.dtype)
        if reference.shape != (3,):
            raise ValueError("initial_normal must have shape (3,)")
    normal0 = reference - jnp.dot(reference, tangent[0]) * tangent[0]
    normal0 /= jnp.linalg.norm(normal0)

    def transport(carry, next_tangent):
        previous_tangent, previous_normal = carry
        next_normal = _minimal_rotation(previous_normal, previous_tangent, next_tangent)
        next_normal -= jnp.dot(next_normal, next_tangent) * next_tangent
        next_normal /= jnp.linalg.norm(next_normal)
        return (next_tangent, next_normal), next_normal

    (_, _), transported = jax.lax.scan(transport, (tangent[0], normal0), tangent[1:])
    raw_normal = jnp.concatenate((normal0[None], transported), axis=0)
    holonomy = jnp.arctan2(
        jnp.dot(tangent[0], jnp.cross(raw_normal[-1], normal0)),
        jnp.dot(raw_normal[-1], normal0),
    )
    fraction = (extended - extended[0]) / period
    normal = jax.vmap(_rotate_about_axis)(raw_normal, tangent, -holonomy * fraction)
    normal -= jnp.sum(normal * tangent, axis=-1)[:, None] * tangent
    normal /= jnp.linalg.norm(normal, axis=-1)[:, None]
    binormal = jnp.cross(tangent, normal)
    arc_length = jnp.sum(0.5 * (speed[:-1] + speed[1:]) * jnp.diff(extended))
    curvature = jnp.linalg.norm(jnp.cross(first, second), axis=-1) / speed**3
    return ClosedAxisGeometry(
        centerline[:-1],
        tangent[:-1],
        normal[:-1],
        binormal[:-1],
        speed[:-1],
        curvature[:-1],
        arc_length,
        holonomy,
        jnp.linalg.norm(centerline[-1] - centerline[0]),
        jnp.linalg.norm(tangent[-1] - tangent[0]),
        jnp.linalg.norm(normal[-1] - normal[0]),
    )


def evaluate_geometry(state: "MirrorState", grid: "MirrorGrid") -> MirrorGeometry:
    """Evaluate axisymmetric or theta-dependent straight-axis geometry."""

    state.validate_shape(grid)
    a = jnp.asarray(state.radius_scale)
    sqrt_s = jnp.sqrt(jnp.asarray(grid.s))[:, None, None]
    radius = sqrt_s * a
    d_a_dtheta = grid.theta_basis.differentiate(a, axis=1)
    d_a_dxi = grid.axial_basis.differentiate(a, axis=2)
    d_radius_dtheta = sqrt_s * d_a_dtheta
    d_radius_dxi = sqrt_s * d_a_dxi

    ds = float(grid.s[1] - grid.s[0])
    # r * r_s is regular even though r_s itself is singular at s=0.
    r_r_s = 0.5 * radial_derivative(radius * radius, ds)
    r_r_s = r_r_s.at[0].set(0.5 * a[0] ** 2)
    r_s = _safe_divide(r_r_s, radius)
    r_s = r_s.at[0].set(r_s[1])

    g_ss = r_s * r_s
    g_stheta = r_s * d_radius_dtheta
    g_sxi = r_s * d_radius_dxi
    g_thetatheta = d_radius_dtheta**2 + radius**2
    g_thetaxi = d_radius_dtheta * d_radius_dxi
    g_xixi = d_radius_dxi**2 + float(grid.dz_dxi) ** 2
    sqrt_g = r_r_s * float(grid.dz_dxi)

    theta = jnp.asarray(grid.theta)[None, :, None]
    z = jnp.asarray(grid.z)[None, None, :]
    xyz = jnp.stack(
        [
            radius * jnp.cos(theta),
            radius * jnp.sin(theta),
            jnp.broadcast_to(z, radius.shape),
        ],
        axis=-1,
    )
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    zeros = jnp.zeros_like(radius)
    e_s = jnp.stack([r_s * cosine, r_s * sine, zeros], axis=-1)
    e_theta = jnp.stack(
        [
            d_radius_dtheta * cosine - radius * sine,
            d_radius_dtheta * sine + radius * cosine,
            zeros,
        ],
        axis=-1,
    )
    e_xi = jnp.stack(
        [
            d_radius_dxi * cosine,
            d_radius_dxi * sine,
            jnp.full_like(radius, float(grid.dz_dxi)),
        ],
        axis=-1,
    )
    volume = jnp.einsum(
        "i,j,k,ijk->",
        jnp.asarray(grid.radial_weights),
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        sqrt_g,
    )
    interior = sqrt_g[1:]
    sign_changed = (jnp.min(interior) <= 0.0) | (jnp.max(interior) <= 0.0)
    return MirrorGeometry(
        xyz=xyz,
        radius=radius,
        d_radius_ds_regular=r_r_s,
        d_radius_dtheta=d_radius_dtheta,
        d_radius_dxi=d_radius_dxi,
        g_ss=g_ss,
        g_stheta=g_stheta,
        g_sxi=g_sxi,
        g_thetatheta=g_thetatheta,
        g_thetaxi=g_thetaxi,
        g_xixi=g_xixi,
        sqrt_g=sqrt_g,
        volume=volume,
        jacobian_sign_changed=sign_changed,
        e_s_xyz=e_s,
        e_theta_xyz=e_theta,
        e_xi_xyz=e_xi,
    )


def evaluate_closed_geometry(
    state: "MirrorState",
    grid: "MirrorGrid",
    axis: ClosedAxisGeometry,
) -> MirrorGeometry:
    """Embed nested surfaces around a periodic spline centerline."""

    state.validate_shape(grid)
    if axis.centerline.shape != (grid.nxi, 3):
        raise ValueError(f"closed axis shape {axis.centerline.shape} must be ({grid.nxi}, 3)")
    a = jnp.asarray(state.radius_scale)
    sqrt_s = jnp.sqrt(jnp.asarray(grid.s))[:, None, None]
    radius = sqrt_s * a
    radius_theta = sqrt_s * grid.theta_basis.differentiate(a, axis=1)
    radius_xi = sqrt_s * grid.axial_basis.differentiate(a, axis=2)
    ds = float(grid.s[1] - grid.s[0])
    radius_radius_s = 0.5 * radial_derivative(radius**2, ds)
    radius_radius_s = radius_radius_s.at[0].set(0.5 * a[0] ** 2)
    radius_s = _safe_divide(radius_radius_s, radius)
    radius_s = radius_s.at[0].set(radius_s[1])

    theta = jnp.asarray(grid.theta)[None, :, None, None]
    normal = jnp.asarray(axis.normal)[None, None]
    binormal = jnp.asarray(axis.binormal)[None, None]
    radial_direction = jnp.cos(theta) * normal + jnp.sin(theta) * binormal
    poloidal_direction = -jnp.sin(theta) * normal + jnp.cos(theta) * binormal
    normal_xi = grid.axial_basis.differentiate(jnp.asarray(axis.normal), axis=0)[None, None]
    binormal_xi = grid.axial_basis.differentiate(jnp.asarray(axis.binormal), axis=0)[None, None]
    radial_direction_xi = jnp.cos(theta) * normal_xi + jnp.sin(theta) * binormal_xi
    centerline_xi = (jnp.asarray(axis.tangent) * jnp.asarray(axis.speed)[:, None])[None, None]

    e_s = radius_s[..., None] * radial_direction
    e_theta = radius_theta[..., None] * radial_direction + radius[..., None] * poloidal_direction
    e_xi = centerline_xi + radius_xi[..., None] * radial_direction + radius[..., None] * radial_direction_xi
    xyz = jnp.asarray(axis.centerline)[None, None] + radius[..., None] * radial_direction
    g_ss = jnp.sum(e_s * e_s, axis=-1)
    g_stheta = jnp.sum(e_s * e_theta, axis=-1)
    g_sxi = jnp.sum(e_s * e_xi, axis=-1)
    g_thetatheta = jnp.sum(e_theta * e_theta, axis=-1)
    g_thetaxi = jnp.sum(e_theta * e_xi, axis=-1)
    g_xixi = jnp.sum(e_xi * e_xi, axis=-1)
    orientation = jnp.sum(
        radial_direction * jnp.cross(poloidal_direction, e_xi),
        axis=-1,
    )
    sqrt_g = radius_radius_s * orientation
    volume = jnp.einsum(
        "i,j,k,ijk->",
        jnp.asarray(grid.radial_weights),
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        sqrt_g,
    )
    interior = sqrt_g[1:]
    sign_changed = (jnp.min(interior) <= 0.0) | (jnp.max(interior) <= 0.0)
    return MirrorGeometry(
        xyz,
        radius,
        radius_radius_s,
        radius_theta,
        radius_xi,
        g_ss,
        g_stheta,
        g_sxi,
        g_thetatheta,
        g_thetaxi,
        g_xixi,
        sqrt_g,
        volume,
        sign_changed,
        e_s,
        e_theta,
        e_xi,
    )


def _radial_profile(values: Array, ns: int, dtype: Any) -> Array:
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (ns,))
    if values.shape != (ns,):
        raise ValueError(f"radial profile shape {values.shape} must be scalar or ({ns},)")
    return values[:, None, None]


def regularize_axis_stream_function(
    state: "MirrorState",
    grid: "MirrorGrid",
    axial_flux_derivative: Array,
) -> "MirrorState":
    """Set the axis stream function so the axial field is single-valued.

    Every theta node at ``s=0`` is the same physical point. The axial flux
    density must therefore be proportional to the polar-coordinate Jacobian
    there. Its theta average is fixed by ``Psi'``; the zero-mean stream
    function supplies the remaining angular variation.
    """

    lam = jnp.asarray(state.lambda_stream)
    if grid.ntheta == 1:
        return type(state)(state.radius_scale, lam.at[0].set(0.0))

    radius_scale = jnp.asarray(state.radius_scale)
    radial_jacobian = 0.5 * radius_scale[0] ** 2
    theta_weights = jnp.asarray(grid.theta_basis.weights)
    mean_jacobian = jnp.sum(theta_weights[:, None] * radial_jacobian, axis=0) / jnp.sum(theta_weights)
    psi_axis = _radial_profile(axial_flux_derivative, grid.ns, lam.dtype)[0, 0, 0]
    derivative = psi_axis * (radial_jacobian / mean_jacobian[None, :] - 1.0)
    modes = jnp.fft.fftfreq(grid.ntheta, d=1.0 / grid.ntheta)[:, None]
    safe_modes = jnp.where(modes == 0.0, 1.0, modes)
    inverse = jnp.where(modes == 0.0, 0.0, 1.0 / (1j * safe_modes))
    axis_stream = jnp.fft.ifft(jnp.fft.fft(derivative, axis=0) * inverse, axis=0).real
    return type(state)(state.radius_scale, lam.at[0].set(axis_stream))


def contravariant_field(
    state: "MirrorState",
    geometry: MirrorGeometry,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
    current_derivative: Array = 0.0,
) -> ContravariantField:
    """Evaluate the divergence-free mirror field representation.

    ``axial_flux_derivative`` is ``Psi'(s)`` and ``current_derivative`` is
    ``I'(s)``.  Both may be scalar or one value per radial surface.
    """

    state.validate_shape(grid)
    state = regularize_axis_stream_function(state, grid, axial_flux_derivative)
    lam = jnp.asarray(state.lambda_stream)
    d_lambda_dtheta = grid.theta_basis.differentiate(lam, axis=1)
    d_lambda_dxi = grid.axial_basis.differentiate(lam, axis=2)
    psi_prime = _radial_profile(axial_flux_derivative, grid.ns, lam.dtype)
    current_prime = _radial_profile(current_derivative, grid.ns, lam.dtype)
    jac_b_theta = current_prime - d_lambda_dxi
    jac_b_xi = psi_prime + d_lambda_dtheta
    return ContravariantField(
        b_sup_s=jnp.zeros_like(lam),
        b_sup_theta=_safe_divide(jac_b_theta, geometry.sqrt_g),
        b_sup_xi=_safe_divide(jac_b_xi, geometry.sqrt_g),
        jac_b_theta=jac_b_theta,
        jac_b_xi=jac_b_xi,
    )


def divergence_b(field: ContravariantField, geometry: MirrorGeometry, grid: "MirrorGrid") -> Array:
    """Return ``div(B)`` from contravariant flux densities."""

    theta_term = grid.theta_basis.differentiate(field.jac_b_theta, axis=1)
    xi_term = grid.axial_basis.differentiate(field.jac_b_xi, axis=2)
    return _safe_divide(theta_term + xi_term, geometry.sqrt_g)


def normalized_divergence_rms(
    field: ContravariantField,
    geometry: MirrorGeometry,
    grid: "MirrorGrid",
    *,
    closed: bool = False,
    characteristic_length: Array | None = None,
) -> Array:
    """Return the volume-weighted RMS of ``div(B)`` normalized by ``B/L``.

    The magnetic axis and open end cuts are constrained coordinate boundaries,
    so this diagnostic norms the active volume. Its normalization permits
    comparisons across field strengths and mirror lengths.
    """

    axial_slice = slice(None) if closed else slice(1, -1)
    divergence = divergence_b(field, geometry, grid)[1:, :, axial_slice]
    b_squared = magnetic_field_squared(field, geometry)[1:, :, axial_slice]
    weights = (
        jnp.asarray(grid.radial_weights[1:])[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, axial_slice]
        * geometry.sqrt_g[1:, :, axial_slice]
    )
    weight_sum = jnp.sum(weights)
    divergence_rms = jnp.sqrt(jnp.sum(weights * divergence**2) / weight_sum)
    length = (
        jnp.asarray(characteristic_length)
        if characteristic_length is not None
        else jnp.asarray(float(grid.z[-1] - grid.z[0]))
    )
    field_gradient_rms = jnp.sqrt(jnp.sum(weights * b_squared / length**2) / weight_sum)
    return divergence_rms / jnp.maximum(field_gradient_rms, jnp.finfo(divergence_rms.dtype).tiny)


def magnetic_field_squared(field: ContravariantField, geometry: MirrorGeometry) -> Array:
    """Contract contravariant components with the covariant metric."""

    bs, bt, bx = field.b_sup_s, field.b_sup_theta, field.b_sup_xi
    return (
        geometry.g_ss * bs**2
        + geometry.g_thetatheta * bt**2
        + geometry.g_xixi * bx**2
        + 2.0 * geometry.g_stheta * bs * bt
        + 2.0 * geometry.g_sxi * bs * bx
        + 2.0 * geometry.g_thetaxi * bt * bx
    )


def magnetic_field_xyz(field: ContravariantField, geometry: MirrorGeometry) -> Array:
    """Convert contravariant magnetic components to Cartesian components."""
    return (
        field.b_sup_s[..., None] * geometry.e_s_xyz
        + field.b_sup_theta[..., None] * geometry.e_theta_xyz
        + field.b_sup_xi[..., None] * geometry.e_xi_xyz
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .model import MirrorState
