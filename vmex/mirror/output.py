"""Mirror-native outputs and compact restart files.

``mout`` is deliberately separate from VMEC's toroidal ``wout`` schema. It
stores physical-grid arrays so a solved open-ended equilibrium can be plotted
or inspected without reconstructing the solver objects.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from .forces import MU0, staggered_field_strength
from .geometry import (
    contravariant_field,
    evaluate_geometry,
    magnetic_field_squared,
    magnetic_field_xyz,
)
from .model import MIRROR_OUTPUT_SCHEMA
from .splines import (
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    SplineMirrorState,
    trace_closed_field_line,
)

RESTART_SCHEMA = "vmex.mirror.free_boundary_restart/3"


@dataclass(frozen=True)
class MoutData:
    """Data-only representation of one straight-axis mirror equilibrium."""

    s: Any
    theta: Any
    xi: Any
    z: Any
    boundary_radius: Any
    radius_scale: Any
    lambda_stream: Any
    mod_b: Any
    b_xyz: Any
    pressure: Any
    history: Any
    coil_xyz: Any
    ftol: float
    iterations: int
    converged: bool
    mass_scale: float
    variational_max: float
    normal_stress_rms: float
    b_normal_rms: float
    staggered_weak_max: float = np.nan
    pointwise_force_rms: float = np.nan
    normalized_divergence_rms: float = np.nan
    message: str = ""
    schema: str = MIRROR_OUTPUT_SCHEMA


@dataclass(frozen=True)
class AxisymmetricBetaDiagnostics:
    """Scalar checks for one axisymmetric free-boundary beta point."""

    requested_beta: Any
    achieved_reference_beta: Any
    volume_averaged_beta: Any
    local_axis_beta: Any
    center_radius: Any
    center_axis_field: Any
    center_vacuum_side_field: Any
    diamagnetic_field_ratio: Any
    paraxial_field_ratio: Any
    paraxial_relative_error: Any


def _integration_measure(result: "FreeBoundaryMirrorResult", grid: "MirrorGrid"):
    geometry = result.plasma_energy.geometry
    weights = (
        jnp.asarray(grid.radial_weights)[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, :]
    )
    return weights * geometry.sqrt_g


def _volume_average(values, result: "FreeBoundaryMirrorResult", grid: "MirrorGrid"):
    measure = _integration_measure(result, grid)
    return jnp.sum(jnp.asarray(values) * measure) / jnp.sum(measure)


def summarize_axisymmetric_beta_scan(
    results: tuple["FreeBoundaryMirrorResult", ...],
    requested_betas: Any,
    grid: "MirrorGrid",
    *,
    reference_field: float,
) -> tuple[AxisymmetricBetaDiagnostics, ...]:
    """Summarize solved beta points against the beta-zero equilibrium.

    ``achieved_reference_beta`` uses the supplied vacuum reference field,
    while ``local_axis_beta`` uses the finite-beta plasma field. The paraxial
    comparison is ``B/B_vac = sqrt(1-beta)`` and is meaningful for a long,
    approximately cylindrical mirror away from beta one.
    """

    betas = jnp.asarray(requested_betas)
    if betas.ndim != 1 or betas.size != len(results):
        raise ValueError("requested_betas must have one value per result")
    if not results:
        raise ValueError("beta diagnostics require at least one result")
    if grid.ntheta != 1:
        raise ValueError("axisymmetric beta diagnostics require ntheta=1")
    center = int(np.argmin(np.abs(np.asarray(grid.z))))
    baseline_field = jnp.sqrt(results[0].plasma_b_squared[0, 0, center])
    reference_field_squared = float(reference_field) ** 2
    summaries = []
    for requested_beta, result in zip(betas, results, strict=True):
        pressure = result.pressure
        axis_field = jnp.sqrt(result.plasma_b_squared[0, 0, center])
        if hasattr(result.vacuum_field, "lateral_field_xyz"):
            vacuum_xyz = result.vacuum_field.lateral_field_xyz[center]
        else:
            vacuum_xyz = result.vacuum_field.total_xyz[0, 0, center]
        vacuum_side_field = jnp.linalg.norm(vacuum_xyz)
        achieved_beta = 2.0 * MU0 * pressure[0, 0, center] / reference_field_squared
        local_beta = 2.0 * MU0 * pressure[0, 0, center] / axis_field**2
        average_pressure = _volume_average(pressure, result, grid)
        average_b_squared = _volume_average(result.plasma_b_squared, result, grid)
        average_beta = 2.0 * MU0 * average_pressure / average_b_squared
        diamagnetic_ratio = axis_field / baseline_field
        paraxial_ratio = jnp.sqrt(jnp.maximum(1.0 - achieved_beta, 0.0))
        summaries.append(
            AxisymmetricBetaDiagnostics(
                requested_beta=requested_beta,
                achieved_reference_beta=achieved_beta,
                volume_averaged_beta=average_beta,
                local_axis_beta=local_beta,
                center_radius=result.boundary.radius_scale[0, center],
                center_axis_field=axis_field,
                center_vacuum_side_field=vacuum_side_field,
                diamagnetic_field_ratio=diamagnetic_ratio,
                paraxial_field_ratio=paraxial_ratio,
                paraxial_relative_error=(diamagnetic_ratio - paraxial_ratio)
                / jnp.maximum(paraxial_ratio, jnp.finfo(axis_field.dtype).tiny),
            )
        )
    return tuple(summaries)


def mout_from_result(
    result: Any,
    grid: Any,
    config: Any,
    *,
    axial_flux_derivative: Any,
    current_derivative: Any = 0.0,
    boundary: Any | None = None,
    pressure: Any | None = None,
    coil_xyz: Any | None = None,
) -> MoutData:
    """Collect a fixed- or free-boundary result and plotting fields."""

    state = getattr(result, "plasma_state", None)
    free_boundary = state is not None
    if state is None:
        state = getattr(result, "state", None)
    if state is None:
        raise ValueError("mirror result has no solved state")
    solved_boundary = getattr(result, "boundary", boundary)
    if solved_boundary is None:
        raise ValueError("fixed-boundary mirror output requires boundary=")
    geometry = evaluate_geometry(state, grid)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    b_xyz = np.asarray(magnetic_field_xyz(field, geometry))
    mod_b = np.asarray(
        staggered_field_strength(
            state,
            grid,
            axial_flux_derivative=axial_flux_derivative,
            current_derivative=current_derivative,
        )
    )
    shape = tuple(state.radius_scale.shape)
    if pressure is None:
        pressure = getattr(result, "pressure", None)
    if pressure is None:
        energy = getattr(result, "plasma_energy", getattr(result, "energy", None))
        if energy is not None and hasattr(energy, "pressure"):
            pressure = np.broadcast_to(np.asarray(energy.pressure)[:, None, None], shape)
    pressure = np.full(shape, np.nan) if pressure is None else np.asarray(pressure)
    if pressure.shape != shape:
        raise ValueError("pressure must match the solved state")
    coils = np.empty((0, 0, 3)) if coil_xyz is None else np.asarray(coil_xyz)
    if coils.ndim != 3 or coils.shape[-1] != 3:
        raise ValueError("coil_xyz must have shape (ncoil, npoint, 3)")
    interface = getattr(result, "interface", None)
    force = getattr(result, "plasma_force", getattr(result, "force", None))
    weak_force = getattr(
        result,
        "plasma_staggered_weak_force",
        getattr(result, "staggered_weak_force", None),
    )
    variational = getattr(result, "variational", None)
    variational_max = getattr(result, "variational_max", None)
    if variational_max is None:
        variational_max = variational.maximum
    history = np.asarray(result.history)
    if not free_boundary and history.ndim == 2 and history.shape[1] >= 5:
        history = history[:, [0, 4]]
    return MoutData(
        s=np.asarray(grid.s),
        theta=np.asarray(grid.theta),
        xi=np.asarray(grid.xi),
        z=np.asarray(grid.z),
        boundary_radius=np.asarray(solved_boundary.radius_scale),
        radius_scale=np.asarray(state.radius_scale),
        lambda_stream=np.asarray(state.lambda_stream),
        mod_b=mod_b,
        b_xyz=b_xyz,
        pressure=pressure,
        history=history,
        coil_xyz=coils,
        ftol=float(config.ftol),
        iterations=int(result.iterations),
        converged=bool(result.converged),
        mass_scale=float(getattr(result, "mass_scale", 1.0)),
        variational_max=float(variational_max),
        normal_stress_rms=(float(interface.normal_stress_rms) if interface is not None else np.nan),
        b_normal_rms=(float(interface.vacuum_b_normal_rms) if interface is not None else np.nan),
        staggered_weak_max=(float(weak_force.maximum) if weak_force is not None else np.nan),
        pointwise_force_rms=(float(force.normalized_rms) if force is not None else np.nan),
        normalized_divergence_rms=float(getattr(result, "normalized_divergence_rms", np.nan)),
        message=str(result.message),
    )


def _finite_shape(data: MoutData) -> tuple[int, int, int]:
    shape = np.asarray(data.radius_scale).shape
    if len(shape) != 3:
        raise ValueError("radius_scale must have shape (ns, ntheta, nxi)")
    ns, ntheta, nxi = shape
    expected = {
        "boundary_radius": (ntheta, nxi),
        "lambda_stream": shape,
        "mod_b": shape,
        "b_xyz": (*shape, 3),
        "pressure": shape,
        "s": (ns,),
        "theta": (ntheta,),
        "xi": (nxi,),
        "z": (nxi,),
    }
    for name, wanted in expected.items():
        if np.asarray(getattr(data, name)).shape != wanted:
            raise ValueError(f"{name} must have shape {wanted}")
    return ns, ntheta, nxi


def write_mout(path: str | Path, data: MoutData, *, overwrite: bool = True) -> Path:
    """Write a compact mirror-native NetCDF file."""

    import netCDF4

    path = Path(path)
    if path.suffix.lower() != ".nc":
        path = path.with_suffix(".nc")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ns, ntheta, nxi = _finite_shape(data)
    history = np.asarray(data.history)
    if history.ndim != 2:
        raise ValueError("history must be a two-dimensional table")
    coils = np.asarray(data.coil_xyz)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.setncattr("schema", data.schema)
        for name in (
            "ftol",
            "iterations",
            "converged",
            "mass_scale",
            "variational_max",
            "normal_stress_rms",
            "b_normal_rms",
            "staggered_weak_max",
            "pointwise_force_rms",
            "normalized_divergence_rms",
            "message",
        ):
            value = getattr(data, name)
            dataset.setncattr(name, int(value) if isinstance(value, (bool, np.bool_)) else value)
        for name, size in (
            ("s", ns),
            ("theta", ntheta),
            ("xi", nxi),
            ("xyz", 3),
            ("history_row", history.shape[0]),
            ("history_column", history.shape[1]),
            ("coil", coils.shape[0]),
            ("coil_point", coils.shape[1]),
        ):
            dataset.createDimension(name, size)
        variables = {
            "s": (("s",), data.s),
            "theta": (("theta",), data.theta),
            "xi": (("xi",), data.xi),
            "z": (("xi",), data.z),
            "boundary_radius": (("theta", "xi"), data.boundary_radius),
            "radius_scale": (("s", "theta", "xi"), data.radius_scale),
            "lambda_stream": (("s", "theta", "xi"), data.lambda_stream),
            "mod_b": (("s", "theta", "xi"), data.mod_b),
            "b_xyz": (("s", "theta", "xi", "xyz"), data.b_xyz),
            "pressure": (("s", "theta", "xi"), data.pressure),
            "history": (("history_row", "history_column"), history),
            "coil_xyz": (("coil", "coil_point", "xyz"), coils),
        }
        for name, (dimensions, values) in variables.items():
            variable = dataset.createVariable(name, "f8", dimensions, zlib=True, complevel=4)
            variable[:] = np.asarray(values)
    return path


def read_mout(path: str | Path) -> MoutData:
    """Read a :class:`MoutData` file and validate its schema."""

    import netCDF4

    with netCDF4.Dataset(Path(path)) as dataset:
        schema = str(dataset.getncattr("schema"))
        if schema != MIRROR_OUTPUT_SCHEMA:
            raise ValueError(f"unsupported mirror output schema: {schema}")
        arrays = {
            name: np.asarray(dataset[name][:])
            for name in (
                "s",
                "theta",
                "xi",
                "z",
                "boundary_radius",
                "radius_scale",
                "lambda_stream",
                "mod_b",
                "b_xyz",
                "pressure",
                "history",
                "coil_xyz",
            )
        }
        attributes = {}
        for field in fields(MoutData):
            if field.name in arrays or field.name == "schema":
                continue
            if field.name in dataset.ncattrs():
                attributes[field.name] = dataset.getncattr(field.name)
            elif field.default is not MISSING:
                attributes[field.name] = field.default
            else:
                raise ValueError(f"mout file is missing required attribute: {field.name}")
    attributes["converged"] = bool(attributes["converged"])
    attributes["iterations"] = int(attributes["iterations"])
    data = MoutData(**arrays, **attributes, schema=schema)
    _finite_shape(data)
    return data


@dataclass(frozen=True)
class FreeBoundaryRestart:
    """Coefficient-native state needed to hot-start a free-boundary solve."""

    boundary: SplineMirrorBoundary
    plasma_state: SplineMirrorState
    mass_scale: float

    @classmethod
    def from_result(cls, result: Any) -> "FreeBoundaryRestart":
        """Extract restart data from a free-boundary solve result."""

        return cls(
            boundary=result.coefficient_boundary,
            plasma_state=result.coefficient_state,
            mass_scale=float(result.mass_scale),
        )


def _finite_restart_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def save_free_boundary_restart(path: str | Path, restart: FreeBoundaryRestart | Any) -> Path:
    """Atomically write a compressed, data-only free-boundary restart."""

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"restart directory does not exist: {path.parent}")
    if not isinstance(restart, FreeBoundaryRestart):
        restart = FreeBoundaryRestart.from_result(restart)

    arrays = {
        "boundary_radius_coefficients": _finite_restart_array(
            restart.boundary.radius_coefficients,
            name="boundary_radius_coefficients",
        ),
        "radius_coefficients": _finite_restart_array(
            restart.plasma_state.radius_coefficients,
            name="radius_coefficients",
        ),
        "lambda_coefficients": _finite_restart_array(
            restart.plasma_state.lambda_coefficients,
            name="lambda_coefficients",
        ),
        "mass_scale": _finite_restart_array(restart.mass_scale, name="mass_scale"),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npz", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, schema=RESTART_SCHEMA, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_free_boundary_restart(
    path: str | Path,
    discretization: SplineMirrorDiscretization,
) -> FreeBoundaryRestart:
    """Load coefficient-native restart data for ``discretization``."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema != RESTART_SCHEMA:
            raise ValueError(f"unsupported mirror restart schema: {schema!r}")
        boundary = SplineMirrorBoundary(
            _finite_restart_array(
                data["boundary_radius_coefficients"],
                name="boundary_radius_coefficients",
            )
        )
        state = SplineMirrorState(
            _finite_restart_array(data["radius_coefficients"], name="radius_coefficients"),
            _finite_restart_array(data["lambda_coefficients"], name="lambda_coefficients"),
        )
        mass_scale = float(_finite_restart_array(data["mass_scale"], name="mass_scale"))

    expected_boundary = (discretization.grid.ntheta, discretization.coefficient_count)
    expected_state = (discretization.grid.ns,) + expected_boundary
    if tuple(np.shape(boundary.radius_coefficients)) != expected_boundary:
        raise ValueError("restart boundary coefficients do not match the requested discretization")
    if (
        tuple(np.shape(state.radius_coefficients)) != expected_state
        or tuple(np.shape(state.lambda_coefficients)) != expected_state
    ):
        raise ValueError("restart state coefficients do not match the requested discretization")
    if mass_scale <= 0.0:
        raise ValueError("restart mass_scale must be positive")
    return FreeBoundaryRestart(boundary=boundary, plasma_state=state, mass_scale=mass_scale)


_PLOT_DPI = 110


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, plt, path: Path) -> Path:
    """Write one reviewed plot and release its Matplotlib resources."""

    fig.savefig(path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _as_mout(mout):
    if hasattr(mout, "boundary_radius") and hasattr(mout, "b_xyz"):
        return mout, "mout"
    path = Path(mout)
    stem = path.stem
    return read_mout(path), stem[5:] if stem.startswith("mout_") else stem


def _theta_samples(data, values, theta_dense):
    """Periodically resample a ``(ntheta, nxi)`` mirror table."""

    values = np.asarray(values, dtype=float)
    theta = np.asarray(data.theta, dtype=float)
    if theta.size == 1:
        return np.broadcast_to(values[0], (len(theta_dense), values.shape[1]))
    order = np.argsort(np.mod(theta, 2.0 * np.pi))
    theta = np.mod(theta[order], 2.0 * np.pi)
    table = values[order]
    spacing = 2.0 * np.pi / theta.size
    if np.allclose(np.diff(np.r_[theta, theta[0] + 2.0 * np.pi]), spacing):
        modes = np.fft.fftfreq(theta.size, d=1.0 / theta.size)
        coefficients = np.fft.fft(table, axis=0) / theta.size
        phase = np.exp(1j * (np.asarray(theta_dense)[:, None] - theta[0]) * modes[None, :])
        return np.real(phase @ coefficients)
    theta_extended = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
    table_extended = np.concatenate([table, table[:1]], axis=0)
    return np.stack(
        [
            np.interp(
                np.mod(theta_dense, 2.0 * np.pi),
                theta_extended,
                table_extended[:, iz],
            )
            for iz in range(values.shape[1])
        ],
        axis=1,
    )


def _field_line(data, radial_index: int, theta0: float, z_order):
    """Trace one cap-to-cap line using the saved Cartesian field samples."""

    z = np.asarray(data.z, dtype=float)[z_order]
    theta_nodes = np.asarray(data.theta, dtype=float)
    b_xyz = np.take(np.asarray(data.b_xyz, dtype=float)[radial_index], z_order, axis=1)
    radius = np.sqrt(float(np.asarray(data.s)[radial_index])) * np.take(
        np.asarray(data.radius_scale)[radial_index], z_order, axis=1
    )
    angles = np.empty(z.size)
    angles[0] = theta0
    periodic_theta = np.r_[np.mod(theta_nodes, 2.0 * np.pi), 2.0 * np.pi]
    for iz in range(z.size - 1):
        angle = angles[iz]
        if theta_nodes.size == 1:
            vector = b_xyz[0, iz]
            local_radius = radius[0, iz]
        else:
            vector = np.asarray(
                [
                    np.interp(
                        np.mod(angle, 2.0 * np.pi),
                        periodic_theta,
                        np.r_[b_xyz[:, iz, component], b_xyz[0, iz, component]],
                    )
                    for component in range(3)
                ]
            )
            local_radius = np.interp(
                np.mod(angle, 2.0 * np.pi),
                periodic_theta,
                np.r_[radius[:, iz], radius[0, iz]],
            )
        b_theta = -np.sin(angle) * vector[0] + np.cos(angle) * vector[1]
        denominator = local_radius * vector[2]
        pitch = 0.0 if abs(denominator) < 1.0e-14 else b_theta / denominator
        angles[iz + 1] = angle + (z[iz + 1] - z[iz]) * pitch
    radius_table = np.take(np.asarray(data.radius_scale)[radial_index], z_order, axis=1)
    radius_samples = _theta_samples(data, radius_table, angles)
    radius_line = radius_samples[np.arange(z.size), np.arange(z.size)] * np.sqrt(
        float(np.asarray(data.s)[radial_index])
    )
    return z, radius_line * np.cos(angles), radius_line * np.sin(angles)


def plot_mout(
    mout: MoutData | str | Path,
    outdir: str | Path,
    *,
    name: str | None = None,
) -> dict[str, Path]:
    """Render summary, cross-section, ``|B|``, and horizontal 3D plots."""

    plt = _matplotlib()
    data, default_name = _as_mout(mout)
    label = name or default_name
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    z_order = np.argsort(np.asarray(data.z))
    z = np.asarray(data.z)[z_order]
    s = np.asarray(data.s)
    center = int(np.argmin(np.abs(z)))
    boundary = np.take(np.asarray(data.boundary_radius), z_order, axis=1)
    mod_b = np.take(np.asarray(data.mod_b), z_order, axis=2)
    pressure = np.take(np.asarray(data.pressure), z_order, axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    axes[0, 0].plot(z, np.mean(boundary, axis=0), color="#0072B2", lw=2)
    axes[0, 0].fill_between(
        z,
        np.min(boundary, axis=0),
        np.max(boundary, axis=0),
        color="#0072B2",
        alpha=0.2,
    )
    axes[0, 0].set(title="Solved LCFS", xlabel="Axial position z [m]", ylabel="Radius [m]")
    axes[0, 1].plot(z, np.mean(mod_b[0], axis=0), label="axis", color="#009E73", lw=2)
    axes[0, 1].plot(z, np.mean(mod_b[-1], axis=0), label="LCFS", color="#D55E00", lw=2)
    axes[0, 1].set(
        title="Magnetic-field strength",
        xlabel="Axial position z [m]",
        ylabel="|B| [T]",
    )
    axes[0, 1].legend()
    axes[1, 0].plot(
        np.sqrt(s),
        np.mean(pressure[:, :, center], axis=1) / 1.0e3,
        "o-",
        color="#CC79A7",
    )
    axes[1, 0].set(
        title="Midplane pressure",
        xlabel="Normalized radius sqrt(s)",
        ylabel="Pressure [kPa]",
    )
    history = np.asarray(data.history)
    if history.size:
        axes[1, 1].semilogy(history[:, 0], np.maximum(history[:, -1], 1.0e-18), color="#0072B2")
    axes[1, 1].axhline(float(data.ftol), color="0.25", ls="--", lw=1, label="ftol")
    if np.isfinite(data.pointwise_force_rms):
        axes[1, 1].axhline(data.pointwise_force_rms, color="#D55E00", lw=1.5, label="strong force")
    axes[1, 1].set(
        title=f"Convergence ({int(data.iterations)} iterations)",
        xlabel="Residual evaluation",
        ylabel="Maximum normalized residual",
    )
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    paths["summary"] = _save_figure(fig, plt, outdir / f"{label}_summary.png")

    theta_dense = np.linspace(0.0, 2.0 * np.pi, 129)
    indices = np.unique(np.round(np.linspace(0, len(z) - 1, 6)).astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
    radial_indices = np.unique(np.round(np.linspace(0, len(s) - 1, 7)).astype(int))
    for axis, iz in zip(axes.flat, indices, strict=False):
        for radial_index in radial_indices:
            table = np.take(np.asarray(data.radius_scale)[radial_index], z_order, axis=1)
            radius = np.sqrt(s[radial_index]) * _theta_samples(data, table, theta_dense)[:, iz]
            axis.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), lw=0.9)
        axis.set(title=f"z = {z[iz]:.3g} m", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    for axis in axes.flat[len(indices) :]:
        axis.set_visible(False)
    paths["cross_sections"] = _save_figure(fig, plt, outdir / f"{label}_cross_sections.png")

    boundary_b = _theta_samples(data, mod_b[-1], theta_dense)
    fig, axis = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    contour = axis.contour(z, theta_dense, boundary_b, 18, cmap="viridis", linewidths=0.9)
    axis.clabel(contour, inline=True, fontsize=7, fmt="%.3g")
    fig.colorbar(contour, ax=axis, label="LCFS |B| [T]")
    axis.set(
        title="Boundary magnetic-field strength",
        xlabel="Axial position z [m]",
        ylabel="Poloidal angle theta",
    )
    paths["modB"] = _save_figure(fig, plt, outdir / f"{label}_modB.png")

    radius_dense = _theta_samples(data, boundary, theta_dense)
    zz, tt = np.meshgrid(z, theta_dense)
    arrow_length = 0.65 * float(np.max(radius_dense))
    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(float(np.min(boundary_b)), float(np.max(boundary_b)))
    surface = axis.plot_surface(
        zz,
        radius_dense * np.cos(tt),
        radius_dense * np.sin(tt),
        facecolors=plt.cm.viridis(norm(boundary_b)),
        linewidth=0,
        alpha=0.52,
    )
    surface.set_rasterized(True)
    for coil_index, coil in enumerate(np.asarray(data.coil_xyz)):
        closed = np.vstack([coil, coil[0]])
        axis.plot(
            closed[:, 2],
            closed[:, 0],
            closed[:, 1],
            color="#C44E52",
            lw=2,
            label="ESSOS coils" if coil_index == 0 else None,
        )
    for line_index, theta0 in enumerate(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)):
        line_z, line_x, line_y = _field_line(data, len(s) - 1, theta0, z_order)
        axis.plot(line_z, line_x, line_y, color="black", lw=3.5, zorder=20)
        axis.plot(
            line_z,
            line_x,
            line_y,
            color="#18C3D6",
            lw=1.8,
            label="field lines" if line_index == 0 else None,
            zorder=21,
        )
        if line_index % 2 == 0:
            center_index = len(line_z) // 2
            tangent = np.asarray(
                [
                    np.gradient(line_z)[center_index],
                    np.gradient(line_x)[center_index],
                    np.gradient(line_y)[center_index],
                ]
            )
            tangent /= max(np.linalg.norm(tangent), np.finfo(float).tiny)
            axis.quiver(
                line_z[center_index],
                line_x[center_index],
                line_y[center_index],
                *tangent,
                length=arrow_length,
                color="#E66100",
                arrow_length_ratio=0.28,
                linewidth=1.4,
            )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axis,
        shrink=0.72,
        pad=0.05,
        label="LCFS |B| [T]",
    )
    axis.set(
        title="Solved mirror equilibrium",
        xlabel="z [m]",
        ylabel="x [m]",
        zlabel="y [m]",
    )
    axis.set_box_aspect((2.2, 1.0, 1.0))
    axis.view_init(elev=22, azim=-57)
    axis.legend(loc="upper left")
    paths["3d"] = _save_figure(fig, plt, outdir / f"{label}_3d.png")
    return paths


def _periodic_theta_sample(values, theta):
    """Evaluate uniform-theta samples at one angle per axial point."""

    values = np.asarray(values)
    theta = np.asarray(theta)
    modes = np.fft.fftfreq(values.shape[0], d=1.0 / values.shape[0])
    coefficients = np.fft.fft(values, axis=0) / values.shape[0]
    return np.real(np.sum(coefficients * np.exp(1j * modes[:, None] * theta[None]), axis=0))


def _closed_field_line_xyz(line, state, discretization, axis, radial_index):
    """Interpolate a traced periodic field line into Cartesian space."""

    parameter = np.mod(np.asarray(line.axial_parameter), 2.0 * np.pi)
    recovery = np.asarray(discretization.grid.axial_basis.recovery_matrix)
    radius_coefficients = np.asarray(state.radius_scale[radial_index]) @ recovery.T
    radius_table = np.asarray(discretization.spline.evaluate(radius_coefficients, parameter))
    radius = np.sqrt(float(discretization.grid.s[radial_index])) * _periodic_theta_sample(
        radius_table,
        np.asarray(line.theta),
    )

    def interpolate_axis(values):
        coefficients = recovery @ np.asarray(values)
        return np.asarray(discretization.spline.evaluate(coefficients, parameter, axis=0))

    centerline = interpolate_axis(axis.centerline)
    normal = interpolate_axis(axis.normal)
    normal = normal / np.linalg.norm(normal, axis=1)[:, None]
    binormal = interpolate_axis(axis.binormal)
    binormal = binormal - np.sum(binormal * normal, axis=1)[:, None] * normal
    binormal = binormal / np.linalg.norm(binormal, axis=1)[:, None]
    radial = np.cos(np.asarray(line.theta))[:, None] * normal + np.sin(np.asarray(line.theta))[:, None] * binormal
    return centerline + radius[:, None] * radial


def plot_stellarator_mirror_hybrid(
    result: Any,
    setup: Any,
    outdir: str | Path,
    *,
    name: str = "stellarator_mirror_hybrid",
) -> Path:
    """Plot a solved periodic two-mirror/stellarator hybrid equilibrium."""

    plt = _matplotlib()
    solved = getattr(result, "evaluated", result)
    discretization, axis = setup.discretization, setup.axis
    if not discretization.closed:
        raise ValueError("hybrid plotting requires a periodic spline discretization")
    state, geometry, field = solved.state, solved.energy.geometry, solved.energy.field
    theta = np.asarray(discretization.grid.theta)
    parameter = np.asarray(discretization.grid.z)
    theta_dense = np.linspace(0.0, 2.0 * np.pi, 97)
    phase = np.exp(1j * theta_dense[:, None] * np.fft.fftfreq(theta.size, d=1.0 / theta.size)[None])

    def dense_theta(values):
        return np.real(phase @ (np.fft.fft(np.asarray(values), axis=0) / theta.size))

    boundary_radius = dense_theta(state.radius_scale[-1])
    radial = (
        np.cos(theta_dense)[:, None, None] * np.asarray(axis.normal)[None]
        + np.sin(theta_dense)[:, None, None] * np.asarray(axis.binormal)[None]
    )
    surface_xyz = np.asarray(axis.centerline)[None] + boundary_radius[..., None] * radial
    mod_b = np.sqrt(np.maximum(np.asarray(magnetic_field_squared(field, geometry)), 0.0))
    boundary_b = dense_theta(mod_b[-1])
    b_min, b_max = float(np.min(boundary_b)), float(np.max(boundary_b))
    color_norm = plt.Normalize(b_min, b_max)

    fig = plt.figure(figsize=(14.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.18, 1.0))
    view = fig.add_subplot(grid[0, :2], projection="3d")
    surface = view.plot_surface(
        surface_xyz[..., 2],
        surface_xyz[..., 0],
        surface_xyz[..., 1],
        facecolors=plt.cm.viridis(color_norm(boundary_b)),
        linewidth=0,
        alpha=0.68,
    )
    surface.set_rasterized(True)
    view.plot(
        np.asarray(axis.centerline)[:, 2],
        np.asarray(axis.centerline)[:, 0],
        np.asarray(axis.centerline)[:, 1],
        color="white",
        lw=3.5,
        zorder=20,
    )
    view.plot(
        np.asarray(axis.centerline)[:, 2],
        np.asarray(axis.centerline)[:, 0],
        np.asarray(axis.centerline)[:, 1],
        color="#222222",
        lw=1.2,
        label="B-spline axis",
        zorder=21,
    )
    radial_index = max(1, discretization.grid.ns - 2)
    iota_values = []
    radial_samples = np.arange(1, discretization.grid.ns)
    for index in radial_samples:
        iota_values.append(
            float(
                trace_closed_field_line(
                    field,
                    discretization,
                    radial_index=int(index),
                    turns=2,
                ).iota
            )
        )
    for line_index, theta0 in enumerate(np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)):
        line = trace_closed_field_line(
            field,
            discretization,
            radial_index=radial_index,
            theta0=float(theta0),
            turns=1,
            steps_per_turn=320,
        )
        xyz = _closed_field_line_xyz(line, state, discretization, axis, radial_index)
        view.plot(xyz[:, 2], xyz[:, 0], xyz[:, 1], color="#101010", lw=2.8, zorder=30)
        view.plot(
            xyz[:, 2],
            xyz[:, 0],
            xyz[:, 1],
            color="#35D0E2",
            lw=1.35,
            label="field lines" if line_index == 0 else None,
            zorder=31,
        )
    view.set(
        title="Solved spline stellarator-mirror hybrid",
        xlabel="z [m]",
        ylabel="x [m]",
        zlabel="y [m]",
    )
    view.set_box_aspect((2.35, 1.25, 0.42))
    view.view_init(elev=24, azim=-61)
    view.legend(loc="upper left")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap="viridis"),
        ax=view,
        orientation="horizontal",
        shrink=0.58,
        pad=0.04,
        label="LCFS |B| [T]",
    )

    map_axis = fig.add_subplot(grid[0, 2])
    image = map_axis.pcolormesh(
        parameter / (2.0 * np.pi),
        theta_dense,
        boundary_b,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(image, ax=map_axis, label="LCFS |B| [T]")
    map_axis.set(
        title="Boundary field strength",
        xlabel="Circuit fraction u / 2pi",
        ylabel="Poloidal angle theta",
    )

    sections = fig.add_subplot(grid[1, 0])
    section_indices = np.asarray(
        [int(np.argmin(np.abs(parameter - 2.0 * np.pi * fraction))) for fraction in (0.125, 0.375, 0.625, 0.875)]
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for color, axial_index in zip(colors, section_indices, strict=False):
        for surface_index in radial_samples:
            radius = (
                np.sqrt(float(discretization.grid.s[surface_index]))
                * dense_theta(state.radius_scale[surface_index])[:, axial_index]
            )
            sections.plot(
                radius * np.cos(theta_dense),
                radius * np.sin(theta_dense),
                color=color,
                lw=0.8,
                alpha=0.72,
            )
        sections.plot([], [], color=color, label=f"u/2pi={parameter[axial_index] / (2 * np.pi):.2f}")
    sections.set(
        title="Solved cross-sections",
        xlabel="Local normal [m]",
        ylabel="Local binormal [m]",
        aspect="equal",
    )
    sections.legend(fontsize=8)

    profiles = fig.add_subplot(grid[1, 1])
    profiles.plot(parameter / (2.0 * np.pi), np.mean(mod_b[0], axis=0), color="#0072B2", label="axis |B|")
    profiles.plot(
        parameter / (2.0 * np.pi),
        np.mean(mod_b[-1], axis=0),
        color="#D55E00",
        label="LCFS |B|",
    )
    profiles.set(
        title="Field and transform",
        xlabel="Circuit fraction u / 2pi",
        ylabel="|B| [T]",
    )
    transform = profiles.inset_axes([0.58, 0.56, 0.38, 0.32])
    transform.plot(
        np.sqrt(np.asarray(discretization.grid.s)[radial_samples]),
        iota_values,
        "o-",
        color="#009E73",
        label="iota",
    )
    transform.set(xlabel="sqrt(s)", ylabel="iota")
    transform.tick_params(labelsize=7)
    profiles.legend(fontsize=8, loc="lower left")

    convergence = fig.add_subplot(grid[1, 2])
    history = np.asarray(solved.history)
    convergence.semilogy(history[:, 0], np.maximum(history[:, 4], 1.0e-18), color="#0072B2", label="variational")
    convergence.axhline(float(solved.force.normalized_rms), color="#D55E00", label="strong force")
    convergence.axhline(float(solved.normalized_divergence_rms), color="#009E73", label="div B")
    convergence.axhline(
        float(solved.variational.maximum),
        color="0.2",
        ls="--",
        label="final variational",
    )
    convergence.set(
        title=f"Convergence ({solved.iterations} iterations)",
        xlabel="Residual evaluation",
        ylabel="Normalized residual",
    )
    convergence.legend(fontsize=8)
    for plot_axis in (map_axis, sections, profiles, convergence):
        plot_axis.grid(alpha=0.2)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    return _save_figure(fig, plt, path)


__all__ = [
    "AxisymmetricBetaDiagnostics",
    "FreeBoundaryRestart",
    "MoutData",
    "load_free_boundary_restart",
    "mout_from_result",
    "plot_mout",
    "plot_stellarator_mirror_hybrid",
    "read_mout",
    "save_free_boundary_restart",
    "summarize_axisymmetric_beta_scan",
    "write_mout",
]


if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .free_boundary import FreeBoundaryMirrorResult
