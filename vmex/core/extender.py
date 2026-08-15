"""Magnetic-field queries outside a VMEC plasma boundary.

The vacuum path evaluates the supplied coil or mgrid field directly.  When the
equilibrium carries plasma pressure or current, :mod:`virtual_casing_jax` adds
the field of currents inside the last closed flux surface.  The resulting
object follows the commonly used SIMSOPT magnetic-field interface while its
explicit-point methods remain JAX-transformable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from .mgrid import MgridField, read_mgrid

Array = Any
PlasmaMode = Literal["auto", "include", "vacuum"]

__all__ = ["MagneticField", "VmecInteriorField", "VmecExtender"]


def _check_points(points: Array, name: str = "points") -> Array:
    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {points.shape}")
    return points


def _cyl_to_cart(points: Array) -> Array:
    r, phi, z = _check_points(points, "cylindrical points").T
    return jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)


def _cart_to_cyl(points: Array) -> Array:
    x, y, z = _check_points(points).T
    return jnp.stack(
        (jnp.hypot(x, y), jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi), z),
        axis=-1,
    )


def _vectors_to_cyl(points_cyl: Array, vectors: Array) -> Array:
    phi = _check_points(points_cyl, "cylindrical points")[:, 1]
    bx, by, bz = _check_points(vectors, "vectors").T
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    return jnp.stack((cphi * bx + sphi * by, -sphi * bx + cphi * by, bz), axis=-1)


def _field_cartesian(field: Any, points: Array) -> Array:
    """Evaluate a VMEX mgrid-like field or an ``xyz -> B`` callable."""
    points = _check_points(points)
    if hasattr(field, "b_cyl"):
        x, y, z = points.T
        r = jnp.hypot(x, y)
        phi = jnp.arctan2(y, x)
        br, bphi, bz = field.b_cyl(r, phi, z)
        cphi, sphi = jnp.cos(phi), jnp.sin(phi)
        return jnp.stack(
            (br * cphi - bphi * sphi, br * sphi + bphi * cphi, bz), axis=-1
        )
    if callable(field):
        value = jnp.asarray(field(points))
        if value.shape != points.shape:
            raise ValueError(
                f"external field returned shape {value.shape}, expected {points.shape}"
            )
        return value
    raise TypeError("external_field must be callable or provide b_cyl(r, phi, z)")


class MagneticField:
    """JAX magnetic field with explicit and stored-point evaluation.

    ``gradB`` has axes ``(point, B_i, x_j)``.  The SIMSOPT-compatible
    ``dB_by_dX`` swaps the last two axes to ``(point, x_j, B_i)``.
    """

    def __init__(
        self,
        B_fn: Callable[[Array], Array],
        gradB_fn: Callable[[Array], Array] | None = None,
        gradgradB_fn: Callable[[Array], Array] | None = None,
        gradgradgradB_fn: Callable[[Array], Array] | None = None,
        *,
        parameters: Array | None = None,
        parameterized_B_fn: Callable[[Array, Array], Array] | None = None,
        parameter_data_fn: Callable[[Array], Any] | None = None,
        B_from_data: Callable[[Any, Array], Array] | None = None,
        dof_names: tuple[str, ...] | None = None,
    ) -> None:
        self._B_fn = B_fn
        self._gradB_fn = gradB_fn
        self._gradgradB_fn = gradgradB_fn
        self._gradgradgradB_fn = gradgradgradB_fn
        self._parameters = (None if parameters is None else
                            jnp.ravel(jnp.asarray(parameters)))
        self._parameterized_B_fn = parameterized_B_fn
        self._parameter_data_fn = parameter_data_fn
        self._B_from_data = B_from_data
        self._parameter_data_vjp = None
        self._data_pullbacks: dict[int, Callable[[Any, Array], Any]] = {}
        self.dof_names = tuple(dof_names or ())
        direct = parameterized_B_fn is not None
        factored = parameter_data_fn is not None or B_from_data is not None
        if factored and (parameter_data_fn is None or B_from_data is None):
            raise ValueError("parameter_data_fn and B_from_data must be provided together")
        if direct and factored:
            raise ValueError("provide parameterized_B_fn or the factored data path, not both")
        if (parameters is not None) != (direct or factored):
            raise ValueError(
                "parameters and a parameterized field path must be provided together")
        if self._parameters is not None and self.dof_names and (
                len(self.dof_names) != self._parameters.size):
            raise ValueError("dof_names must match the number of field parameters")
        self._points_cart: Array | None = None
        self._points_cyl: Array | None = None

    def set_points(self, points: Array) -> "MagneticField":
        """Store Cartesian points with shape ``(n, 3)``."""
        self._points_cart = _check_points(points)
        self._points_cyl = None
        # Spatial-derivative pullbacks close over the evaluation points.
        self._data_pullbacks.clear()
        return self

    def set_points_xyz(self, points: Array) -> "MagneticField":
        """Store Cartesian ``(x, y, z)`` points with shape ``(n, 3)``."""
        return self.set_points(points)

    set_points_cart = set_points_xyz

    def set_points_cyl(self, points: Array) -> "MagneticField":
        """Store cylindrical points ``(R, phi, Z)`` with shape ``(n, 3)``."""
        cylindrical = _check_points(points, "cylindrical points")
        self.set_points(_cyl_to_cart(cylindrical))
        self._points_cyl = cylindrical
        return self

    def get_points_cart(self) -> Array:
        """Return stored Cartesian points."""
        return self._require_points()

    def get_points_cyl(self) -> Array:
        """Return stored cylindrical points ``(R, phi, Z)``."""
        points = self._require_points()
        if self._points_cyl is None:
            self._points_cyl = _cart_to_cyl(points)
        return self._points_cyl

    def _require_points(self) -> Array:
        if self._points_cart is None:
            raise RuntimeError("call set_points() or pass points explicitly")
        return self._points_cart

    def B(self, points: Array | None = None) -> Array:
        """Return Cartesian ``B`` at explicit or stored points."""
        xyz = self._require_points() if points is None else _check_points(points)
        value = jnp.asarray(self._B_fn(xyz))
        if value.shape != xyz.shape:
            raise ValueError(f"field returned shape {value.shape}, expected {xyz.shape}")
        return value

    def B_contravariant(self, point: Array) -> Array:
        """Return Cartesian ``B`` at one point for ESSOS field-line tracing."""
        point = jnp.asarray(point)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        return self.B(point[None, :])[0]

    @staticmethod
    def to_xyz(point: Array) -> Array:
        """Return a Cartesian tracing point unchanged."""
        point = jnp.asarray(point)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        return point

    def B_cyl(self, points: Array | None = None) -> Array:
        """Return ``(B_R, B_phi, B_Z)`` at cylindrical points."""
        rphiz = self.get_points_cyl() if points is None else _check_points(
            points, "cylindrical points"
        )
        return _vectors_to_cyl(rphiz, self.B(_cyl_to_cart(rphiz)))

    def absB(self, points: Array | None = None) -> Array:
        """Return ``|B|`` with shape ``(n,)``."""
        return jnp.linalg.norm(self.B(points), axis=-1)

    def AbsB(self, points: Array | None = None) -> Array:
        """Return SIMSOPT-compatible ``|B|`` with shape ``(n, 1)``."""
        return self.absB(points)[:, None]

    def gradB(self, points: Array | None = None) -> Array:
        """Return ``dB_i/dx_j`` with shape ``(n, 3, 3)``."""
        xyz = self._require_points() if points is None else _check_points(points)
        if self._gradB_fn is not None:
            value = jnp.asarray(self._gradB_fn(xyz))
        else:
            value = jax.vmap(
                jax.jacfwd(lambda point: self._B_fn(point[None, :])[0])
            )(xyz)
        expected = xyz.shape + (3,)
        if value.shape != expected:
            raise ValueError(f"field gradient returned shape {value.shape}, expected {expected}")
        return value

    def gradgradB(self, points: Array | None = None) -> Array:
        """Return ``d²B_i/dx_j dx_k`` with shape ``(n, 3, 3, 3)``."""
        xyz = self._require_points() if points is None else _check_points(points)
        if self._gradgradB_fn is not None:
            value = jnp.asarray(self._gradgradB_fn(xyz))
        else:
            value = jax.vmap(jax.jacfwd(jax.jacfwd(
                lambda point: self._B_fn(point[None, :])[0])))(xyz)
        expected = xyz.shape + (3, 3)
        if value.shape != expected:
            raise ValueError(
                f"second field derivative returned shape {value.shape}, expected {expected}")
        return value

    def gradgradgradB(self, points: Array | None = None) -> Array:
        """Return ``d³B_i/dx_j dx_k dx_l`` with shape ``(n, 3, 3, 3, 3)``."""
        xyz = self._require_points() if points is None else _check_points(points)
        if self._gradgradgradB_fn is not None:
            value = jnp.asarray(self._gradgradgradB_fn(xyz))
        else:
            point_field = lambda point: self._B_fn(point[None, :])[0]  # noqa: E731
            value = jax.vmap(
                jax.jacfwd(jax.jacfwd(jax.jacfwd(point_field))))(xyz)
        expected = xyz.shape + (3, 3, 3)
        if value.shape != expected:
            raise ValueError(
                f"third field derivative returned shape {value.shape}, expected {expected}")
        return value

    def _parameter_vjp(self, order: int, cotangent: Array) -> Array:
        if self._parameters is None:
            raise RuntimeError(
                "this field was not constructed with optimizable parameters")
        points = self._require_points()

        def spatial_quantity(field_function):
            point_field = lambda point: field_function(point[None, :])[0]  # noqa: E731
            function = point_field
            for _ in range(order):
                function = jax.jacfwd(function)
            return jax.vmap(function)(points)

        if self._parameter_data_fn is not None:
            if self._parameter_data_vjp is None:
                self._parameter_data_vjp = jax.vjp(
                    self._parameter_data_fn, self._parameters)
            data, pullback = cast(tuple[Any, Callable[[Any], Any]],
                                  self._parameter_data_vjp)
            B_from_data = cast(Callable[[Any, Array], Array], self._B_from_data)

            def quantity_from_data(field_data):
                return spatial_quantity(lambda xyz: B_from_data(field_data, xyz))

            value = quantity_from_data(data)
            vector = jnp.asarray(cotangent)
            if vector.shape != value.shape:
                raise ValueError(
                    f"cotangent has shape {vector.shape}, expected {value.shape}")
            if order not in self._data_pullbacks:
                self._data_pullbacks[order] = jax.jit(jax.grad(
                    lambda field_data, weight: jnp.vdot(
                        quantity_from_data(field_data), weight),
                    argnums=0, allow_int=True))
            data_bar = self._data_pullbacks[order](data, vector)
            return pullback(data_bar)[0]

        parameterized_B_fn = cast(Callable[[Array, Array], Array],
                                  self._parameterized_B_fn)

        def quantity(parameters):
            return spatial_quantity(lambda xyz: parameterized_B_fn(parameters, xyz))

        value, pullback = jax.vjp(quantity, self._parameters)
        vector = jnp.asarray(cotangent)
        if vector.shape != value.shape:
            raise ValueError(
                f"cotangent has shape {vector.shape}, expected {value.shape}")
        return pullback(vector)[0]

    def B_vjp(self, vector: Array) -> Array:
        """Return ``vector.T @ dB/dp`` for this field's parameters ``p``."""
        return self._parameter_vjp(0, vector)

    def gradB_vjp(self, vector: Array) -> Array:
        """Return a VJP of :meth:`gradB` with respect to field parameters."""
        return self._parameter_vjp(1, vector)

    def gradgradB_vjp(self, vector: Array) -> Array:
        """Return a VJP of :meth:`gradgradB` with respect to field parameters."""
        return self._parameter_vjp(2, vector)

    def gradgradgradB_vjp(self, vector: Array) -> Array:
        """Return a VJP of :meth:`gradgradgradB` with respect to parameters."""
        return self._parameter_vjp(3, vector)

    def dB_by_dX(self, points: Array | None = None) -> Array:
        """Return SIMSOPT axis order ``(point, x_j, B_i)``."""
        return jnp.swapaxes(self.gradB(points), -1, -2)

    def GradAbsB(self, points: Array | None = None) -> Array:
        """Return the Cartesian gradient of ``|B|``."""
        B = self.B(points)
        gradB = self.gradB(points)
        scale = jnp.maximum(jnp.linalg.norm(B, axis=-1), jnp.finfo(B.dtype).tiny)
        return jnp.einsum("...i,...ij->...j", B, gradB) / scale[:, None]


def _radial_value_and_derivative(coefficients: Array, s: Array) -> tuple[Array, Array]:
    """Piecewise-linear full-mesh spectra and their radial derivative."""
    coefficients = jnp.asarray(coefficients)
    ns = coefficients.shape[0]
    coordinate = jnp.clip(s, 0.0, 1.0) * (ns - 1)
    index = jnp.clip(jnp.floor(coordinate).astype(int), 0, ns - 2)
    fraction = coordinate - index
    lower, upper = coefficients[index], coefficients[index + 1]
    return lower + fraction * (upper - lower), (ns - 1) * (upper - lower)


def _flux_coordinates_to_xyz(spectra: dict[str, Array], points: Array) -> Array:
    """Map VMEC ``(s, theta, phi)`` coordinates to Cartesian points."""
    s, theta, phi = _check_points(points, "flux coordinates").T
    radial_r = jax.vmap(lambda value: _radial_value_and_derivative(
        spectra["rmnc"], value)[0])(s)
    radial_z = jax.vmap(lambda value: _radial_value_and_derivative(
        spectra["zmns"], value)[0])(s)
    phase = spectra["xm"][None, :] * theta[:, None] - spectra["xn"][None, :] * phi[:, None]
    radius = jnp.sum(radial_r * jnp.cos(phase), axis=1)
    z = jnp.sum(radial_z * jnp.sin(phase), axis=1)
    return jnp.stack((radius * jnp.cos(phi), radius * jnp.sin(phi), z), axis=1)


def _B_contravariant_flux(spectra: dict[str, Array], points: Array) -> Array:
    """Return ``(B^s, B^theta, B^phi)`` at VMEC flux coordinates."""
    s, theta, phi = _check_points(points, "flux coordinates").T
    bu_full = _full_mesh_contravariant(spectra["bsupu"])
    bv_full = _full_mesh_contravariant(spectra["bsupv"])
    bu_coeff = jax.vmap(lambda value: _radial_value_and_derivative(
        bu_full, value)[0])(s)
    bv_coeff = jax.vmap(lambda value: _radial_value_and_derivative(
        bv_full, value)[0])(s)
    phase = spectra["xmn"][None, :] * theta[:, None] - spectra["xnn"][None, :] * phi[:, None]
    return jnp.stack((jnp.zeros_like(s), jnp.sum(bu_coeff * jnp.cos(phase), axis=1),
                      jnp.sum(bv_coeff * jnp.cos(phase), axis=1)), axis=1)


class _VmecFluxCoordinateField:
    """ESSOS tracing adapter whose points and vectors use ``(s, theta, phi)``."""

    def __init__(self, spectra: dict[str, Array]) -> None:
        self.spectra = spectra

    def B_contravariant(self, point: Array) -> Array:
        point = jnp.asarray(point)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        return _B_contravariant_flux(self.spectra, point[None, :])[0]

    def to_xyz(self, point: Array) -> Array:
        point = jnp.asarray(point)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        return _flux_coordinates_to_xyz(self.spectra, point[None, :])[0]


def _full_mesh_contravariant(coefficients: Array) -> Array:
    """Interpolate half-mesh VMEC spectra to every full-mesh surface."""
    coefficients = jnp.asarray(coefficients)
    interior = 0.5 * (coefficients[1:-1] + coefficients[2:])
    edge = 1.5 * coefficients[-1] - 0.5 * coefficients[-2]
    return jnp.concatenate((coefficients[1:2], interior, edge[None]), axis=0)


def _interior_coordinates_and_B(
    spectra: dict[str, Array], points: Array, *, newton_iterations: int
) -> tuple[Array, Array]:
    """Invert VMEC coordinates and synthesize the interior Cartesian field."""
    points = _check_points(points)
    xm, xn = spectra["xm"], spectra["xn"]
    xmn, xnn = spectra["xmn"], spectra["xnn"]
    rmnc, zmns = spectra["rmnc"], spectra["zmns"]
    bu_full = _full_mesh_contravariant(spectra["bsupu"])
    bv_full = _full_mesh_contravariant(spectra["bsupv"])

    def geometry(s, theta, phi):
        rc, rcs = _radial_value_and_derivative(rmnc, s)
        zs, zss = _radial_value_and_derivative(zmns, s)
        phase = xm * theta - xn * phi
        cosine, sine = jnp.cos(phase), jnp.sin(phase)
        R, Z = jnp.vdot(rc, cosine), jnp.vdot(zs, sine)
        Rs, Zs = jnp.vdot(rcs, cosine), jnp.vdot(zss, sine)
        Rt, Zt = jnp.vdot(-xm * rc, sine), jnp.vdot(xm * zs, cosine)
        Rp, Zp = jnp.vdot(xn * rc, sine), jnp.vdot(-xn * zs, cosine)
        return R, Z, Rs, Zs, Rt, Zt, Rp, Zp

    def one_point(point):
        x, y, z = point
        radius, phi = jnp.hypot(x, y), jnp.arctan2(y, x)
        axis_R, axis_Z, *_ = geometry(0.0, 0.0, phi)
        theta0 = jnp.arctan2(z - axis_Z, radius - axis_R)
        edge_R, edge_Z, *_ = geometry(1.0, theta0, phi)
        edge_distance2 = (edge_R - axis_R) ** 2 + (edge_Z - axis_Z) ** 2
        s0 = ((radius - axis_R) ** 2 + (z - axis_Z) ** 2) / jnp.maximum(
            edge_distance2, 1.0e-24)

        def update(_, coordinates):
            s, theta = coordinates
            R, Z, Rs, Zs, Rt, Zt, *_ = geometry(s, theta, phi)
            determinant = Rs * Zt - Rt * Zs
            safe = jnp.where(jnp.abs(determinant) > 1.0e-14, determinant, 1.0e-14)
            residual_R, residual_Z = R - radius, Z - z
            ds = (Zt * residual_R - Rt * residual_Z) / safe
            dt = (-Zs * residual_R + Rs * residual_Z) / safe
            return jnp.clip(s - ds, -0.05, 1.05), jnp.mod(theta - dt, 2.0 * jnp.pi)

        s, theta = jax.lax.fori_loop(
            0, int(newton_iterations), update,
            (jnp.clip(s0, 1.0e-8, 1.0), theta0))
        R, Z, _Rs, _Zs, Rt, Zt, Rp, Zp = geometry(s, theta, phi)
        bu_coeff, _ = _radial_value_and_derivative(bu_full, s)
        bv_coeff, _ = _radial_value_and_derivative(bv_full, s)
        nyquist_phase = xmn * theta - xnn * phi
        bu = jnp.vdot(bu_coeff, jnp.cos(nyquist_phase))
        bv = jnp.vdot(bv_coeff, jnp.cos(nyquist_phase))
        cphi, sphi = jnp.cos(phi), jnp.sin(phi)
        e_theta = jnp.array((Rt * cphi, Rt * sphi, Zt))
        e_phi = jnp.array((Rp * cphi - R * sphi, Rp * sphi + R * cphi, Zp))
        field = bu * e_theta + bv * e_phi
        error = jnp.hypot(R - radius, Z - z)
        valid = (s >= -1.0e-8) & (s <= 1.0 + 1.0e-8) & (error <= 1.0e-7)
        return jnp.array((s, theta, phi)), jnp.where(valid, field, jnp.nan)

    coordinates, field = jax.vmap(one_point)(points)
    return coordinates, field


class VmecInteriorField(MagneticField):
    """VMEC magnetic field at Cartesian points inside the plasma boundary.

    Cartesian points are inverted to ``(s, theta, phi)`` with a fixed-count
    differentiable Newton solve.  Spectral angular evaluation and radial
    interpolation then recover ``B``.  Points outside the last closed surface
    return NaNs; use :class:`VmecExtender` there.
    """

    def __init__(
        self,
        spectra: dict[str, Array],
        *,
        newton_iterations: int = 10,
        parameters: Array | None = None,
        parameterized_B_fn: Callable[[Array, Array], Array] | None = None,
        parameter_data_fn: Callable[[Array], Any] | None = None,
        B_from_data: Callable[[Any, Array], Array] | None = None,
        dof_names: tuple[str, ...] = (),
    ) -> None:
        self.spectra = spectra
        self.newton_iterations = int(newton_iterations)
        self._points_flux: Array | None = None

        def B_fn(points):
            return _interior_coordinates_and_B(
                spectra, points, newton_iterations=self.newton_iterations)[1]

        super().__init__(
            B_fn, parameters=parameters, parameterized_B_fn=parameterized_B_fn,
            parameter_data_fn=parameter_data_fn, B_from_data=B_from_data,
            dof_names=dof_names)

    def set_points(self, points: Array) -> "VmecInteriorField":
        """Store Cartesian points and clear any cached flux coordinates."""
        self._points_flux = None
        super().set_points(points)
        return self

    def set_points_flux(self, points: Array) -> "VmecInteriorField":
        """Map VMEC ``(s, theta, phi)`` to stored Cartesian points.

        Parameter VJPs hold these mapped physical points fixed.
        """
        self._points_flux = _check_points(points, "flux coordinates")
        super().set_points(_flux_coordinates_to_xyz(self.spectra, self._points_flux))
        return self

    def get_points_flux(self) -> Array:
        """Return stored points as VMEC ``(s, theta, phi)`` coordinates."""
        return self.flux_coordinates()

    def field_in_flux_coordinates(self) -> _VmecFluxCoordinateField:
        """Return a field-line adapter in the ``(s, theta, phi)`` basis."""
        return _VmecFluxCoordinateField(self.spectra)

    def flux_coordinates(self, points: Array | None = None) -> Array:
        """Return inverted ``(s, theta, phi)`` at explicit or stored points."""
        if points is None and self._points_flux is not None:
            return self._points_flux
        xyz = self._require_points() if points is None else _check_points(points)
        return _interior_coordinates_and_B(
            self.spectra, xyz, newton_iterations=self.newton_iterations)[0]

    @classmethod
    def from_state(cls, inp: Any, state: Any, *, runtime: Any = None,
                   newton_iterations: int = 10) -> "VmecInteriorField":
        """Construct a field from a live converged VMEX state."""
        from .freeboundary_diff import _state_field_spectra

        return cls(_state_field_spectra(inp, state, runtime),
                   newton_iterations=newton_iterations)

    @classmethod
    def from_parameterized_state(
        cls,
        inp: Any,
        state_runtime_fn: Callable[[Array], tuple[Any, Any]],
        parameters: Array,
        *,
        dof_names: tuple[str, ...] = (),
        newton_iterations: int = 10,
    ) -> "VmecInteriorField":
        """Construct an interior field with exact VJPs in problem parameters."""
        from .freeboundary_diff import _state_field_spectra

        parameters = jnp.ravel(jnp.asarray(parameters))

        def spectra_of(p):
            state, runtime = state_runtime_fn(p)
            return _state_field_spectra(inp, state, runtime)

        def B_from_spectra(spectra, points):
            return _interior_coordinates_and_B(
                spectra, points, newton_iterations=newton_iterations)[1]

        return cls(
            spectra_of(parameters), newton_iterations=newton_iterations,
            parameters=parameters, parameter_data_fn=spectra_of,
            B_from_data=B_from_spectra,
            dof_names=dof_names)


def _has_plasma_sources(wout: Any) -> bool:
    """Detect pressure or current sources, including zero-net-current cases."""
    for name in ("betatotal", "wp", "ctor"):
        value = getattr(wout, name, 0.0)
        if value is not None and abs(float(value)) > 1.0e-14:
            return True
    for name in ("presf", "currumnc", "currvmnc", "currumns", "currvmns"):
        value = getattr(wout, name, None)
        if value is not None and np.any(np.abs(np.asarray(value)) > 1.0e-14):
            return True
    return False


def _mgrid_from_wout(wout: Any, base_dir: Path | None) -> MgridField | None:
    path_text = str(getattr(wout, "mgrid_file", "")).strip()
    if not path_text or path_text.upper() == "NONE":
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    data = read_mgrid(path)
    extcur = np.asarray(getattr(wout, "extcur", ()), dtype=float).reshape(-1)
    scaled = np.zeros((data.nextcur,), dtype=float)
    scaled[: min(extcur.size, data.nextcur)] = extcur[: data.nextcur]
    if str(data.mgrid_mode).upper().startswith(("R", "N")):
        raw = np.asarray(data.raw_coil_cur, dtype=float)
        scaled = np.divide(scaled, raw, out=scaled, where=raw != 0.0)
    return MgridField.from_mgrid_data(data, extcur=scaled)


class VmecExtender(MagneticField):
    """Total field outside the last closed VMEC flux surface.

    Current-free vacuum equilibria use ``external_field`` directly.  For
    finite pressure or plasma current, the internal-current virtual-casing
    branch is added.  External coil currents must lie outside the query region;
    targets must not lie exactly on the source surface.
    """

    def __init__(self, external_field: Any, plasma_field: Any | None = None) -> None:
        if external_field is None and plasma_field is None:
            raise ValueError("at least one external or plasma field is required")
        self.external_field = external_field
        self.plasma_field = plasma_field

        def B_fn(points: Array) -> Array:
            value = jnp.zeros_like(points)
            if self.external_field is not None:
                value = value + _field_cartesian(self.external_field, points)
            if self.plasma_field is not None:
                value = value + self.plasma_field.B_plasma_xyz(points)
            return value

        # Differentiate the same Cartesian field graph used by B().  This is
        # both simpler and avoids the cylindrical-axis singularity in older
        # virtual_casing_jax ``gradB_plasma_xyz`` implementations.
        super().__init__(B_fn)

    @property
    def uses_virtual_casing(self) -> bool:
        """Whether plasma-current virtual casing contributes to the field."""
        return self.plasma_field is not None

    @classmethod
    def from_surface_data(
        cls,
        surface_data: Any,
        *,
        external_field: Any | None = None,
        digits: int = 6,
        levels: tuple[tuple[int, int], ...] | None = None,
    ) -> "VmecExtender":
        """Construct the finite-beta path from traceable VMEX surface data."""
        from . import freeboundary_diff as fbd

        fbd._require_vcj()
        nphi, ntheta = map(int, surface_data.gamma.shape[1:])
        schedule = levels or ((nphi, ntheta), (2 * nphi, 2 * ntheta))
        config = fbd.ExteriorFieldConfig(
            digits=digits,
            src_nphi=nphi,
            src_ntheta=ntheta,
            levels=schedule,
            branch="internal",
        )
        plasma_field = fbd.VirtualCasingExteriorField(surface_data, config)
        return cls(external_field, plasma_field)

    @classmethod
    def from_parameterized_surface_data(
        cls,
        surface_data_fn: Callable[[Array], Any],
        parameters: Array,
        *,
        external_field: Any | None = None,
        external_parameters: Array | None = None,
        external_field_from_parameters: Callable[[Array], Any] | None = None,
        external_dof_names: tuple[str, ...] = (),
        digits: int = 6,
        levels: tuple[tuple[int, int], ...] | None = None,
        dof_names: tuple[str, ...] = (),
    ) -> "VmecExtender":
        """Construct a virtual-casing field with VJPs in ``parameters``.

        ``surface_data_fn(parameters)`` must return traceable VMEX surface
        data.  Spatial derivatives and parameter VJPs then use the same JAX
        field graph; no finite-difference equilibrium solves are introduced.
        """
        parameters = jnp.ravel(jnp.asarray(parameters))
        parameterized_external = external_parameters is not None
        if parameterized_external != (external_field_from_parameters is not None):
            raise ValueError(
                "external_parameters and external_field_from_parameters "
                "must be provided together")
        if parameterized_external and external_field is not None:
            raise ValueError(
                "external_field is inferred from external_parameters; do not provide both")
        if not parameterized_external and external_dof_names:
            raise ValueError(
                "external_dof_names require external_parameters and "
                "external_field_from_parameters")
        external_parameters = (jnp.ravel(jnp.asarray(external_parameters))
                               if parameterized_external else jnp.empty(0))
        if len(external_dof_names) not in (0, int(external_parameters.size)):
            raise ValueError("external_dof_names must match external_parameters")
        if parameterized_external and not external_dof_names:
            external_dof_names = tuple(
                f"external[{index}]" for index in range(int(external_parameters.size)))
        n_plasma = int(parameters.size)
        all_parameters = jnp.concatenate((parameters, external_parameters))
        external_factory = cast(
            Callable[[Array], Any], external_field_from_parameters
        ) if parameterized_external else None

        def split(p):
            return p[:n_plasma], p[n_plasma:]

        def make_external(external_dofs):
            if external_factory is None:
                return external_field
            return external_factory(external_dofs)

        initial_external_field = make_external(external_parameters)

        initial_surface_data = surface_data_fn(parameters)

        def differentiable_surface_data(p: Array) -> tuple[Array, ...]:
            plasma_parameters, external_dofs = split(p)
            data = surface_data_fn(plasma_parameters)
            return data.gamma, data.B_total, data.normal, data.area_vector, external_dofs

        def B_from_surface_arrays(arrays: tuple[Array, ...], points: Array) -> Array:
            from dataclasses import replace

            gamma, B_total, normal, area_vector, external_dofs = arrays
            data = replace(
                initial_surface_data, gamma=gamma, B_total=B_total,
                normal=normal, area_vector=area_vector)
            live_external_field = make_external(external_dofs)
            return cls.from_surface_data(
                data, external_field=live_external_field, digits=digits,
                levels=levels).B(points)

        field = cls.from_surface_data(
            initial_surface_data, external_field=initial_external_field,
            digits=digits, levels=levels)
        field._parameters = all_parameters
        field._parameter_data_fn = differentiable_surface_data
        field._B_from_data = B_from_surface_arrays
        field.dof_names = tuple(dof_names) + tuple(external_dof_names)
        return field

    @classmethod
    def from_wout(
        cls,
        wout: Any,
        *,
        external_field: Any | None = None,
        plasma: PlasmaMode = "auto",
        nphi: int = 32,
        ntheta: int = 32,
        digits: int = 6,
        levels: tuple[tuple[int, int], ...] | None = None,
        base_dir: str | Path | None = None,
    ) -> "VmecExtender":
        """Construct an exterior field from a wout-like object."""
        if plasma not in ("auto", "include", "vacuum"):
            raise ValueError("plasma must be 'auto', 'include', or 'vacuum'")
        if external_field is None:
            external_field = _mgrid_from_wout(
                wout, None if base_dir is None else Path(base_dir)
            )

        include_plasma = plasma == "include" or (
            plasma == "auto" and _has_plasma_sources(wout)
        )
        plasma_field = None
        if include_plasma:
            from . import freeboundary_diff as fbd

            surface = fbd.surface_field_data_from_wout(
                wout, nphi=nphi, ntheta=ntheta
            )
            return cls.from_surface_data(
                surface,
                external_field=external_field,
                digits=digits,
                levels=levels,
            )

        if external_field is None and plasma_field is None:
            raise ValueError(
                "a vacuum extension needs an mgrid file or external_field"
            )
        return cls(external_field, plasma_field)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "VmecExtender":
        """Read a wout file and resolve a relative mgrid beside it."""
        from .wout import read_wout

        path = Path(path)
        return cls.from_wout(read_wout(path), base_dir=path.parent, **kwargs)

    @classmethod
    def from_state(
        cls,
        inp: Any,
        state: Any,
        *,
        external_field: Any | None = None,
        nphi: int = 32,
        ntheta: int = 32,
        digits: int = 6,
        levels: tuple[tuple[int, int], ...] | None = None,
    ) -> "VmecExtender":
        """Construct the differentiable finite-beta path from a live VMEX state."""
        from . import freeboundary_diff as fbd

        surface = fbd.surface_field_data_from_state(
            inp, state, nphi=nphi, ntheta=ntheta
        )
        return cls.from_surface_data(
            surface,
            external_field=external_field,
            digits=digits,
            levels=levels,
        )

    @classmethod
    def from_equilibrium(cls, equilibrium: Any, **kwargs: Any) -> "VmecExtender":
        """Construct from an equilibrium, retaining live-state derivatives."""
        external_field = kwargs.pop("external_field", None)
        if external_field is None and bool(equilibrium.inp.lfreeb):
            from .freeboundary import _external_field_from_input

            external_field = _external_field_from_input(equilibrium.inp)

        plasma = kwargs.pop("plasma", "auto")
        if plasma not in ("auto", "include", "vacuum"):
            raise ValueError("plasma must be 'auto', 'include', or 'vacuum'")
        include_plasma = plasma == "include" or (
            plasma == "auto" and _has_plasma_sources(equilibrium.wout)
        )
        if include_plasma:
            return cls.from_state(
                equilibrium.inp,
                equilibrium.state,
                external_field=external_field,
                **kwargs,
            )
        return cls.from_wout(
            equilibrium.wout,
            external_field=external_field,
            plasma="vacuum",
            **kwargs,
        )
