"""Magnetic-field queries outside a VMEC plasma boundary.

The vacuum path evaluates the supplied coil or mgrid field directly.  When the
equilibrium carries plasma pressure or current, :mod:`virtual_casing_jax` adds
the field of currents inside the last closed flux surface.  The resulting
object follows the commonly used SIMSOPT magnetic-field interface while its
explicit-point methods remain JAX-transformable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .mgrid import MgridField, read_mgrid

Array = Any
PlasmaMode = Literal["auto", "include", "vacuum"]

__all__ = ["MagneticField", "VmecExtender"]


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
    ) -> None:
        self._B_fn = B_fn
        self._gradB_fn = gradB_fn
        self._points_cart: Array | None = None
        self._points_cyl: Array | None = None

    def set_points(self, points: Array) -> "MagneticField":
        """Store Cartesian points with shape ``(n, 3)``."""
        self._points_cart = _check_points(points)
        self._points_cyl = None
        return self

    set_points_cart = set_points

    def set_points_cyl(self, points: Array) -> "MagneticField":
        """Store cylindrical points ``(R, phi, Z)`` with shape ``(n, 3)``."""
        self._points_cyl = _check_points(points, "cylindrical points")
        self._points_cart = _cyl_to_cart(self._points_cyl)
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

    def dB_by_dX(self, points: Array | None = None) -> Array:
        """Return SIMSOPT axis order ``(point, x_j, B_i)``."""
        return jnp.swapaxes(self.gradB(points), -1, -2)

    def GradAbsB(self, points: Array | None = None) -> Array:
        """Return the Cartesian gradient of ``|B|``."""
        B = self.B(points)
        gradB = self.gradB(points)
        scale = jnp.maximum(jnp.linalg.norm(B, axis=-1), jnp.finfo(B.dtype).tiny)
        return jnp.einsum("...i,...ij->...j", B, gradB) / scale[:, None]


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

        def gradB_fn(points: Array) -> Array:
            value = jnp.zeros(points.shape[:-1] + (3, 3), dtype=points.dtype)
            if self.external_field is not None:
                value = value + jax.vmap(
                    jax.jacfwd(
                        lambda point: _field_cartesian(
                            self.external_field, point[None, :]
                        )[0]
                    )
                )(points)
            if self.plasma_field is not None:
                value = value + self.plasma_field.gradB_plasma_xyz(points)
            return value

        super().__init__(B_fn, gradB_fn)

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
