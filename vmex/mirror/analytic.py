"""Independent paraxial fixtures for nonaxisymmetric straight mirrors.

These formulas validate equilibrium output; they do not call the mirror solver.
Lengths are in metres and magnetic fields are in tesla.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

Array = Any


class FirstOrderSection(NamedTuple):
    """First-order inverse-coordinate coefficients of an elliptical section."""

    x_cos: Array
    x_sin: Array
    y_cos: Array
    y_sin: Array


class QuadrupoleField(NamedTuple):
    """Coefficients in ``B = B0 + r^2(B20+B2c cos2a+B2s sin2a)``."""

    average: Array
    cosine: Array
    sine: Array


def _rotation(angle: Array) -> Array:
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    return jnp.asarray(((cosine, -sine), (sine, cosine)))


@dataclass(frozen=True)
class AxisymmetricPolynomialMirror:
    """Exact vacuum mirror with symmetric circular end sections.

    The scalar potential is cubic in ``z`` and quadratic in transverse
    position. Its gradient is exactly curl-free and divergence-free, while
    the quartic poloidal flux gives an analytic family of nested surfaces.
    """

    center_field: float = 1.0
    half_length: float = 1.0
    mirror_strength: float = 0.5

    def __post_init__(self) -> None:
        if self.center_field <= 0.0:
            raise ValueError("center_field must be positive")
        if self.half_length <= 0.0:
            raise ValueError("half_length must be positive")
        if self.mirror_strength < 0.0:
            raise ValueError("mirror_strength must be nonnegative")

    @property
    def curvature(self) -> float:
        """Return the quadratic axial-field coefficient in inverse metres squared."""

        return self.mirror_strength / self.half_length**2

    def potential(self, points: Array) -> Array:
        """Return the exact scalar magnetic potential."""

        x, y, z = jnp.asarray(points)
        a = self.curvature
        return self.center_field * (z + a * z**3 / 3.0 - 0.5 * a * z * (x**2 + y**2))

    def field(self, points: Array) -> Array:
        """Return ``grad(potential)`` in Cartesian coordinates."""

        return jax.grad(self.potential)(jnp.asarray(points))

    def axis_field(self, z: Array) -> Array:
        """Return the quadratic on-axis mirror field."""

        return self.center_field * (1.0 + self.curvature * jnp.asarray(z) ** 2)

    def poloidal_flux(self, radius: Array, z: Array) -> Array:
        """Return the exact axisymmetric flux labeling the nested surfaces."""

        radius_squared = jnp.asarray(radius) ** 2
        a = self.curvature
        return self.center_field * (
            0.5 * (1.0 + a * jnp.asarray(z) ** 2) * radius_squared
            - 0.125 * a * radius_squared**2
        )

    def boundary_radius(self, midplane_radius: Array, z: Array) -> Array:
        """Return the exact flux-surface radius at axial position ``z``."""

        a = self.curvature
        radius = jnp.asarray(midplane_radius)
        if a == 0.0:
            return jnp.broadcast_to(radius, jnp.shape(z))
        normalized_flux = self.poloidal_flux(radius, 0.0) / self.center_field
        axial = 1.0 + a * jnp.asarray(z) ** 2
        discriminant = jnp.sqrt(axial**2 - 2.0 * a * normalized_flux)
        return jnp.sqrt(4.0 * normalized_flux / (axial + discriminant))


@dataclass(frozen=True)
class RotatingEllipseParaxial:
    """Flux-conserving rotating ellipse from Appendix C of Rodriguez et al.

    The physical ellipse rotates by ``rotation`` between ``-half_length`` and
    ``half_length``. A counter-rotation of the field-line label enforces the
    vacuum first-order consistency equation. The default is a 90-degree turn.

    The expansion is valid for ``r / half_length << 1`` and away from a zero
    of the on-axis field.
    """

    half_length: float = 1.0
    reference_field: float = 1.0
    mirror_strength: float = 1.0
    elongation: float = 2.0
    rotation: float = 0.5 * jnp.pi

    def __post_init__(self) -> None:
        if self.half_length <= 0.0:
            raise ValueError("half_length must be positive")
        if self.reference_field <= 0.0:
            raise ValueError("reference_field must be positive")
        if self.mirror_strength < 0.0:
            raise ValueError("mirror_strength must be nonnegative")
        if self.elongation <= 0.0:
            raise ValueError("elongation must be positive")

    def axis_field(self, z: Array) -> Array:
        """Return the even quadratic on-axis mirror field ``B0(z)``."""

        normalized_z = jnp.asarray(z) / self.half_length
        return self.reference_field * (1.0 + self.mirror_strength * normalized_z**2)

    def orientation(self, z: Array) -> Array:
        """Return the physical major-axis angle in the Cartesian cross-section."""

        return 0.5 * self.rotation * jnp.asarray(z) / self.half_length

    def label_angle(self, z: Array) -> Array:
        """Return the field-line-label angle required by vacuum consistency."""

        factor = 2.0 / (self.elongation + 1.0 / self.elongation)
        return -factor * self.orientation(z)

    def section_matrix(self, z: Array) -> Array:
        """Map ``(cos(alpha), sin(alpha))`` to normalized ``(x, y)``."""

        scale = jnp.sqrt(self.reference_field / self.axis_field(z))
        axes = scale * jnp.asarray(((jnp.sqrt(self.elongation), 0.0), (0.0, 1.0 / jnp.sqrt(self.elongation))))
        return _rotation(self.orientation(z)) @ axes @ _rotation(self.label_angle(z))

    def first_order(self, z: Array) -> FirstOrderSection:
        """Return ``X1c, X1s, Y1c, Y1s`` at scalar axial position ``z``."""

        matrix = self.section_matrix(z)
        return FirstOrderSection(matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1])

    def flux_determinant(self, z: Array) -> Array:
        """Return ``X1c*Y1s-X1s*Y1c = Bbar/B0``."""

        return jnp.linalg.det(self.section_matrix(z))

    def section(self, radius: Array, alpha: Array, z: Array) -> Array:
        """Return Cartesian ``(x,y,z)`` points on a first-order flux surface."""

        direction = jnp.stack((jnp.cos(alpha), jnp.sin(alpha)), axis=0)
        xy = jnp.asarray(radius) * (self.section_matrix(z) @ direction)
        axial = jnp.broadcast_to(z, jnp.shape(alpha))
        return jnp.stack((xy[0], xy[1], axial), axis=-1)

    def field(self, points: Array) -> Array:
        """Return the divergence-free first-order field tangent to the sections.

        Transverse components follow a fixed field-line label through the
        section matrix. Terms beyond first order in distance from the axis are
        intentionally omitted, consistent with the paraxial construction.
        """

        x, y, z = jnp.asarray(points)
        matrix = self.section_matrix(z)
        derivative = jax.jacfwd(self.section_matrix)(z)
        label = jnp.linalg.solve(matrix, jnp.asarray((x, y)))
        transverse = self.axis_field(z) * (derivative @ label)
        return jnp.asarray((transverse[0], transverse[1], self.axis_field(z)))

    def boundary_radius(self, midplane_radius: Array, theta: Array, z: Array) -> Array:
        """Return the physical polar radius of the rotating elliptical tube."""

        scale = jnp.sqrt(self.reference_field / self.axis_field(z))
        semi_major = midplane_radius * jnp.sqrt(self.elongation) * scale
        semi_minor = midplane_radius / jnp.sqrt(self.elongation) * scale
        angle = jnp.asarray(theta) - self.orientation(z)
        denominator = jnp.sqrt((semi_minor * jnp.cos(angle)) ** 2 + (semi_major * jnp.sin(angle)) ** 2)
        return semi_major * semi_minor / denominator

    def consistency_residual(self, z: Array) -> Array:
        """Return the Appendix-C first-order vacuum consistency identity."""

        values = jnp.asarray(self.first_order(z))
        derivative = jax.jacfwd(lambda zz: jnp.asarray(self.first_order(zz)))(z)
        xc, xs, yc, ys = values
        xc_z, xs_z, yc_z, ys_z = derivative
        return xs * xc_z - xc * xs_z + ys * yc_z - yc * ys_z

    def riccati_residual(self, z: Array) -> Array:
        """Return the Appendix-C Riccati equation residual for ``sigma=Y1c/Y1s``."""

        values = jnp.asarray(self.first_order(z))
        derivative = jax.jacfwd(lambda zz: jnp.asarray(self.first_order(zz)))(z)
        xc, xs, yc, ys = values
        xc_z, xs_z, _, _ = derivative
        sigma = yc / ys
        sigma_z = jax.grad(lambda zz: self.first_order(zz).y_cos / self.first_order(zz).y_sin)(z)
        rhs = -((xc - xs * sigma) ** 2) * (self.axis_field(z) / self.reference_field) ** 2
        rhs *= xs * xc_z - xc * xs_z
        return sigma_z - rhs

    def _potential_coefficients(self, z: Array) -> Array:
        values = jnp.asarray(self.first_order(z))
        derivative = jax.jacfwd(lambda zz: jnp.asarray(self.first_order(zz)))(z)
        xc, xs, yc, ys = values
        xc_z, xs_z, yc_z, ys_z = derivative
        denominator = xs * yc + xc * ys
        x_norm = xc**2 + xs**2
        y_norm = yc**2 + ys**2
        mixed_x = ys * xc_z + yc * xs_z
        mixed_y = xs * yc_z + xc * ys_z
        b0 = self.axis_field(z)
        phi20 = 0.25 * b0 * (x_norm * mixed_x + y_norm * mixed_y) / denominator
        phi2c = 0.25 * b0 * ((xc**2 - xs**2) * mixed_x + (yc**2 - ys**2) * mixed_y) / denominator
        phi2s = (
            0.25
            * b0
            * (xc * yc * (2.0 * xs * xs_z + 2.0 * ys * ys_z) + xs * ys * (2.0 * xc * xc_z + 2.0 * yc * yc_z))
            / denominator
        )
        phi2s += 0.5 * self.reference_field * (xc * xs_z - xs * xc_z) / denominator
        return jnp.asarray((phi20, phi2c, phi2s))

    def quadrupole(self, z: Array) -> QuadrupoleField:
        """Return the second-order field coefficients from Appendix-C Eq. (80)."""

        values_z = jax.jacfwd(lambda zz: jnp.asarray(self.first_order(zz)))(z)
        phi_z = jax.jacfwd(self._potential_coefficients)(z)
        xc_z, xs_z, yc_z, ys_z = values_z
        b0 = self.axis_field(z)
        average = phi_z[0] - 0.25 * b0 * (xc_z**2 + xs_z**2 + yc_z**2 + ys_z**2)
        cosine = phi_z[1] - 0.25 * b0 * (xc_z**2 - xs_z**2 + yc_z**2 - ys_z**2)
        sine = phi_z[2] - 0.5 * b0 * (xc_z * xs_z + yc_z * ys_z)
        return QuadrupoleField(average, cosine, sine)

    def center_quadrupole(self) -> QuadrupoleField:
        """Return the independent closed form at the magnetic-well minimum."""

        z = jnp.asarray(0.0)
        values = jnp.asarray(self.first_order(z))
        first = jax.jacfwd(lambda zz: jnp.asarray(self.first_order(zz)))(z)
        second = jax.jacfwd(jax.jacfwd(lambda zz: jnp.asarray(self.first_order(zz))))(z)
        xc, _, _, _ = values
        xc_z, xs_z, _, _ = first
        xc_zz, xs_zz, _, _ = second
        b0 = self.axis_field(z)
        b0_zz = jax.grad(jax.grad(self.axis_field))(z)
        shear = (xc_z / xc**2) ** 2 + xs_z**2
        average = 0.25 * b0 * (-b0_zz / (b0 * xc**2) + 2.0 * shear + xc_zz * xc * (1.0 - xc**-4))
        cosine = 0.25 * b0 * (b0_zz / (b0 * xc**2) - 2.0 * shear + xc_zz * xc * (1.0 + xc**-4))
        sine = 0.5 * b0 * xc * xs_zz
        return QuadrupoleField(average, cosine, sine)

    def field_strength(self, radius: Array, alpha: Array, z: Array) -> Array:
        """Evaluate the paraxial field strength through order ``r^2``."""

        quadrupole = self.quadrupole(z)
        angular = quadrupole.average + quadrupole.cosine * jnp.cos(2.0 * alpha)
        angular += quadrupole.sine * jnp.sin(2.0 * alpha)
        return self.axis_field(z) + jnp.asarray(radius) ** 2 * angular


@dataclass(frozen=True)
class StraightFieldLineMirror:
    """Agren-Savenko marginal-minimum-B vacuum field through paraxial order.

    The scalar potential is their Eq. (2), truncated at relative order
    ``(radius/axial_scale)^4``. Use only for ``|z| < axial_scale`` and a thin
    flux tube.
    """

    center_field: float = 1.0
    axial_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.center_field <= 0.0:
            raise ValueError("center_field must be positive")
        if self.axial_scale <= 0.0:
            raise ValueError("axial_scale must be positive")

    def potential(self, points: Array) -> Array:
        """Return the paraxial scalar magnetic potential at Cartesian points."""

        x, y, z = jnp.asarray(points)
        s = z / self.axial_scale
        denominator = (1.0 - s**2) ** 2
        transverse = 0.5 * (x / self.axial_scale) ** 2 * (1.0 - s) / denominator
        transverse -= 0.5 * (y / self.axial_scale) ** 2 * (1.0 + s) / denominator
        return self.axial_scale * self.center_field * (jnp.arctanh(s) + transverse)

    def field(self, points: Array) -> Array:
        """Return ``grad(potential)`` in Cartesian coordinates."""

        return jax.grad(self.potential)(jnp.asarray(points))

    def axis_field(self, z: Array) -> Array:
        """Return the exact on-axis field ``B0/(1-z^2/c^2)``."""

        s = jnp.asarray(z) / self.axial_scale
        return self.center_field / (1.0 - s**2)

    def clebsch_labels(self, points: Array) -> Array:
        """Return leading-order straight-line labels ``(x0,y0)``."""

        x, y, z = jnp.asarray(points)
        s = z / self.axial_scale
        return jnp.asarray((x / (1.0 + s), y / (1.0 - s)))

    def field_line(self, x0: Array, y0: Array, z: Array) -> Array:
        """Return leading-order straight, nonparallel field-line points."""

        s = jnp.asarray(z) / self.axial_scale
        return jnp.stack((x0 * (1.0 + s), y0 * (1.0 - s), z), axis=-1)

    def section(self, midplane_radius: Array, alpha: Array, z: Array) -> Array:
        """Return an analytic flux-tube section from a circular midplane."""

        s = jnp.asarray(z) / self.axial_scale
        x = midplane_radius * (1.0 + s) * jnp.cos(alpha)
        y = midplane_radius * (1.0 - s) * jnp.sin(alpha)
        return jnp.stack((x, y, jnp.broadcast_to(z, jnp.shape(alpha))), axis=-1)

    def boundary_radius(self, midplane_radius: Array, theta: Array, z: Array) -> Array:
        """Return the physical polar radius of its elliptical flux tube."""

        normalized_z = jnp.asarray(z) / self.axial_scale
        semi_x = midplane_radius * (1.0 + normalized_z)
        semi_y = midplane_radius * (1.0 - normalized_z)
        denominator = jnp.sqrt((semi_y * jnp.cos(theta)) ** 2 + (semi_x * jnp.sin(theta)) ** 2)
        return semi_x * semi_y / denominator

    def ellipticity(self, z: Array) -> Array:
        """Return the major/minor axis ratio of the analytic section."""

        magnitude = jnp.abs(jnp.asarray(z) / self.axial_scale)
        return (1.0 + magnitude) / (1.0 - magnitude)


__all__ = [
    "AxisymmetricPolynomialMirror",
    "FirstOrderSection",
    "QuadrupoleField",
    "RotatingEllipseParaxial",
    "StraightFieldLineMirror",
]
