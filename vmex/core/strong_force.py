"""High-order toroidal reconstruction and an independent strong-force oracle.

This module deliberately does not call VMEX's legacy half-mesh force kernels.
It reconstructs smooth, axis-regular Fourier amplitudes with local splines,
forms the curvilinear metric at arbitrary points, and evaluates
``curl(B) / mu0 x B - grad(p)`` from the continuous representation.

Coordinates are ``(rho, theta, zeta)`` with ``s = rho**2`` and ``zeta`` the
field-period angle.  The physical cylindrical angle is ``zeta / nfp``.
Every Fourier amplitude is represented as ``rho**abs(m) q(s)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .profiles import MU0
from .radial_basis import BSplineBasis

Array = Any

_NORMALIZATION = (
    "2*|JxB-grad(p)|/(|JxB|+|grad(p)|+force_floor), evaluated pointwise; "
    "bounded above by 2 by construction"
)

#: Disclosure carried with every certificate.  ``F = JxB - grad(p)`` gives
#: ``|F| <= |JxB| + |grad(p)|`` pointwise, so ``eps_F`` can never exceed 2 no
#: matter how badly the state violates force balance.  It reaches 2 whenever
#: the two terms stop cancelling — in vacuum ``grad(p) = 0`` makes
#: ``eps_F = 2|JxB|/(|JxB| + force_floor)``, which is 2 to machine precision
#: wherever the current is not zero.  A value near 2 therefore reports a
#: collapsed denominator, not a 200% force error.
_SATURATION = (
    "eps_F is bounded above by 2 because |F| <= |JxB| + |grad(p)| pointwise; "
    "eps_F near 2 means the denominator collapsed (vacuum, or JxB and grad(p) "
    "no longer cancelling), not a 200% force error. Read the volume-averaged "
    "normalizations and the dimensional <|F|> alongside it."
)


@dataclass(frozen=True, eq=False)
class HighOrderEquilibriumState:
    """Axis-regular continuous equilibrium reconstructed from a VMEX state.

    The six Fourier tables have shape ``(mnmax, radial_basis.size)`` and hold
    the spline coefficients of ``q(s)`` in ``rho**abs(m) q(s)``.  Lambda is
    the external straight-field-line angle displacement (the convention in a
    wout file), not VMEX's internally rescaled lambda unknown.  ``phipf`` and
    ``chipf`` are derivatives with respect to normalized toroidal flux ``s``;
    pressure is in Pa.
    """

    radial_basis: BSplineBasis
    m: np.ndarray
    n: np.ndarray
    nfp: int
    R_cos: Array
    R_sin: Array
    Z_cos: Array
    Z_sin: Array
    L_cos: Array
    L_sin: Array
    phipf: Array
    chipf: Array
    pressure: Array
    jacobian_sign: int = 1
    source: str = "continuous"
    boundary_R_cos: Array | None = None
    boundary_R_sin: Array | None = None
    boundary_Z_cos: Array | None = None
    boundary_Z_sin: Array | None = None

    def __post_init__(self) -> None:
        if self.radial_basis.periodic:
            raise ValueError("toroidal strong-force reconstruction requires a clamped radial basis")
        if int(self.nfp) < 1:
            raise ValueError("nfp must be positive")
        m = np.asarray(self.m, dtype=int)
        n = np.asarray(self.n, dtype=int)
        if m.ndim != 1 or n.shape != m.shape:
            raise ValueError("m and n must be one-dimensional arrays with matching shape")
        expected = (m.size, self.radial_basis.size)
        for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
            if np.shape(getattr(self, name)) != expected:
                raise ValueError(f"{name} has shape {np.shape(getattr(self, name))}; expected {expected}")
        for name in ("phipf", "chipf", "pressure"):
            if np.shape(getattr(self, name)) != (self.radial_basis.size,):
                raise ValueError(f"{name} must have shape {(self.radial_basis.size,)}")

    def tree_flatten(self):
        """Expose every continuous coefficient as differentiable JAX data."""

        children = (
            self.R_cos,
            self.R_sin,
            self.Z_cos,
            self.Z_sin,
            self.L_cos,
            self.L_sin,
            self.phipf,
            self.chipf,
            self.pressure,
            self.boundary_R_cos,
            self.boundary_R_sin,
            self.boundary_Z_cos,
            self.boundary_Z_sin,
        )
        metadata = (
            self.radial_basis,
            tuple(np.asarray(self.m, dtype=int)),
            tuple(np.asarray(self.n, dtype=int)),
            int(self.nfp),
            int(self.jacobian_sign),
            self.source,
        )
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        """Rebuild a state without placing NumPy mode arrays in JAX metadata."""

        radial_basis, m, n, nfp, jacobian_sign, source = metadata
        (
            R_cos,
            R_sin,
            Z_cos,
            Z_sin,
            L_cos,
            L_sin,
            phipf,
            chipf,
            pressure,
            boundary_R_cos,
            boundary_R_sin,
            boundary_Z_cos,
            boundary_Z_sin,
        ) = children
        return cls(
            radial_basis=radial_basis,
            m=np.asarray(m, dtype=int),
            n=np.asarray(n, dtype=int),
            nfp=nfp,
            R_cos=R_cos,
            R_sin=R_sin,
            Z_cos=Z_cos,
            Z_sin=Z_sin,
            L_cos=L_cos,
            L_sin=L_sin,
            phipf=phipf,
            chipf=chipf,
            pressure=pressure,
            jacobian_sign=jacobian_sign,
            source=source,
            boundary_R_cos=boundary_R_cos,
            boundary_R_sin=boundary_R_sin,
            boundary_Z_cos=boundary_Z_cos,
            boundary_Z_sin=boundary_Z_sin,
        )


@dataclass(frozen=True)
class StrongForceSamples:
    """Pointwise, dimensional output of :func:`evaluate_strong_force`."""

    rho: Array
    theta: Array
    zeta: Array
    sqrt_g: Array
    B: Array
    J: Array
    force: Array
    force_rho: Array
    force_helical: Array
    radial_force_density: Array
    helical_force_density: Array
    signed_radial_force_density: Array
    signed_helical_force_density: Array
    lorentz_norm: Array
    grad_pressure_norm: Array


@dataclass(frozen=True)
class HighOrderFieldSamples:
    """Native geometry and magnetic field at arbitrary flux coordinates."""

    rho: Array
    theta: Array
    zeta: Array
    position: Array
    dposition_drho: Array
    dposition_dtheta: Array
    dposition_dphi: Array
    sqrt_g: Array
    B_contravariant: Array
    B_covariant: Array
    B: Array
    pressure: Array


@dataclass(frozen=True)
class HighOrderSurfaceSamples:
    """Analytic native LCFS data accepted by array-based downstream codes."""

    gamma: Array
    gammadash_theta: Array
    gammadash_phi: Array
    normal: Array
    unitnormal: Array
    area_element: Array
    B_total: Array
    theta: Array
    phi: Array
    nfp: int
    ntheta: int
    nphi: int


@dataclass(frozen=True)
class ForceErrorNormalizations:
    """Volume-averaged force-error normalizations over one evaluation set.

    These are the published global normalizations, computed on the same
    quadrature nodes and the same ``|sqrt(g)|`` volume weights as the
    pointwise certificate, and they do not saturate:

    * ``relative_force_error`` is ``<|F|> / <|grad(p)|>`` — the
      volume-averaged relative force error of Panici, Conlin, Dudt, Unalmis
      and Kolemen, *J. Plasma Phys.* **89** (2023) 955890303, Eqs. (32)-(34b).
      It is genuinely undefined in vacuum, so it is reported as ``nan``
      when ``<|grad(p)|>`` is exactly zero rather than as a huge floored
      number.  A low-beta case makes it large but finite for the same
      reason; ``volume_average_grad_pressure`` is reported beside it so a
      reader can see how small the denominator was.
    * ``magnetic_relative_force_error`` and the two ``magnetic_normalized``
      entries divide by ``<|grad(B^2/2 mu0)|>`` instead, which stays finite
      and meaningful in vacuum.  That is the ``ForceBalance`` normalization
      DESC reports (``desc/objectives/_equilibrium.py``) and the form used in
      Thun et al. 2026, Eq. (42).  ``magnetic_normalized_l2`` and
      ``magnetic_normalized_linf`` keep the pointwise numerator, as DESC
      does, and divide by that single global scale.
    * ``volume_average_force`` is the dimensional ``<|F|>`` in N m^-3, which
      no normalization can hide.

    ``s_min`` and ``s_max`` state the flux window the record covers and
    ``node_count`` the number of radial quadrature nodes inside it.
    """

    volume_average_force: Array
    volume_average_grad_pressure: Array
    volume_average_magnetic_pressure_gradient: Array
    relative_force_error: Array
    magnetic_relative_force_error: Array
    magnetic_normalized_l2: Array
    magnetic_normalized_linf: Array
    node_count: int
    s_min: float
    s_max: float


@dataclass(frozen=True)
class StrongForceReport:
    """Independent overintegrated certificate for one continuous state.

    ``normalized_*`` are the pointwise certificate ``eps_F``, which is the
    acceptance criterion and is **bounded above by 2 by construction** — see
    :data:`_SATURATION`, carried on every report as ``saturation``.  Read
    ``global_normalizations`` (full domain) and ``window_normalizations``
    (the stated ``s`` window, away from the axis and the edge, where the raw
    maxima do not dominate) for numbers that cannot saturate, and
    ``near_axis_l2``/``bulk_l2``/``edge_l2`` for where the error sits.
    """

    absolute_l2: Array
    absolute_p99: Array
    absolute_linf: Array
    normalized_l2: Array
    normalized_p99: Array
    normalized_linf: Array
    radial_l2: Array
    helical_l2: Array
    radial_normalized_l2: Array
    helical_normalized_l2: Array
    near_axis_l2: Array
    bulk_l2: Array
    edge_l2: Array
    radial_nodes: Array
    flux_surface_average: Array
    flux_surface_normalized_l2: Array
    angular_spectral_tail: Array
    radial_refinement_difference: Array
    #: Smallest signed Jacobian on the sample, in machine units.
    minimum_signed_jacobian: Array
    #: The same minimum divided by the mean absolute Jacobian: scale-free, so it is
    #: comparable across devices.  See :func:`_nestedness_margin`.
    nestedness_margin: Array
    boundary_residual: Array
    gauge_residual: Array
    global_normalizations: ForceErrorNormalizations
    window_normalizations: ForceErrorNormalizations
    force_floor: float
    normalization: str = _NORMALIZATION
    saturation: str = _SATURATION
    coordinate_convention: str = "rho=sqrt(s), theta poloidal, zeta field-period; physical phi=zeta/nfp"


jax.tree_util.register_pytree_node_class(HighOrderEquilibriumState)

for _cls in (HighOrderFieldSamples, StrongForceSamples):
    jax.tree_util.register_dataclass(
        _cls,
        data_fields=[field for field in _cls.__dataclass_fields__],
        meta_fields=[],
    )

jax.tree_util.register_dataclass(
    HighOrderSurfaceSamples,
    data_fields=[
        "gamma",
        "gammadash_theta",
        "gammadash_phi",
        "normal",
        "unitnormal",
        "area_element",
        "B_total",
        "theta",
        "phi",
    ],
    meta_fields=["nfp", "ntheta", "nphi"],
)


def _series(state: HighOrderEquilibriumState, coefficients: Array, x: Array) -> Array:
    rho, theta, zeta = x
    s = rho * rho
    q = state.radial_basis.evaluate(jnp.asarray(coefficients), s, axis=-1)
    amplitudes = q * rho ** jnp.asarray(state.m, dtype=q.dtype)
    phase = jnp.asarray(state.m, dtype=q.dtype) * theta - jnp.asarray(state.n, dtype=q.dtype) * zeta
    return jnp.sum(amplitudes * jnp.cos(phase))


def _sine_series(state: HighOrderEquilibriumState, coefficients: Array, x: Array) -> Array:
    rho, theta, zeta = x
    s = rho * rho
    q = state.radial_basis.evaluate(jnp.asarray(coefficients), s, axis=-1)
    amplitudes = q * rho ** jnp.asarray(state.m, dtype=q.dtype)
    phase = jnp.asarray(state.m, dtype=q.dtype) * theta - jnp.asarray(state.n, dtype=q.dtype) * zeta
    return jnp.sum(amplitudes * jnp.sin(phase))


def _RZL(state: HighOrderEquilibriumState, x: Array) -> tuple[Array, Array, Array]:
    R = _series(state, state.R_cos, x) + _sine_series(state, state.R_sin, x)
    Z = _series(state, state.Z_cos, x) + _sine_series(state, state.Z_sin, x)
    lam = _series(state, state.L_cos, x) + _sine_series(state, state.L_sin, x)
    return R, Z, lam


def _position(state: HighOrderEquilibriumState, x: Array) -> Array:
    R, Z, _ = _RZL(state, x)
    phi = x[2] / float(state.nfp)
    return jnp.asarray([R * jnp.cos(phi), R * jnp.sin(phi), Z])


def _profile(state: HighOrderEquilibriumState, coefficients: Array, rho: Array) -> Array:
    return state.radial_basis.evaluate(jnp.asarray(coefficients), rho * rho)


def _basic_fields(state: HighOrderEquilibriumState, x: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return basis, Jacobian, B components/vector, and pressure at one point."""

    basis_vectors = jax.jacfwd(lambda y: _position(state, y))(x)
    sqrt_g = jnp.linalg.det(basis_vectors)
    lam_gradient = jax.grad(lambda y: _RZL(state, y)[2])(x)
    phip = _profile(state, state.phipf, x[0])
    chip = _profile(state, state.chipf, x[0])
    # Convert VMEC's (s, theta, physical-phi) Clebsch representation to
    # (rho, theta, zeta=nfp*phi). The new Jacobian is 2*rho/nfp times the
    # VMEC Jacobian, B^zeta=nfp*B^phi, and lambda_phi=nfp*lambda_zeta.
    flux_factor = 2.0 * x[0] / sqrt_g
    B_sup = jnp.asarray(
        [
            0.0,
            flux_factor
            * (chip / float(state.nfp) - phip * lam_gradient[2]),
            flux_factor * phip * (1.0 + lam_gradient[1]),
        ]
    )
    B = basis_vectors @ B_sup
    B_cov = basis_vectors.T @ B
    pressure = _profile(state, state.pressure, x[0])
    return basis_vectors, sqrt_g, B_sup, B_cov, B, pressure


def _point_force(state: HighOrderEquilibriumState, x: Array) -> tuple[Array, ...]:
    basis_vectors, sqrt_g, B_sup, B_cov, B, pressure = _basic_fields(state, x)

    def covariant_and_pressure(y: Array) -> Array:
        _, _, _, covariant, _, p = _basic_fields(state, y)
        return jnp.concatenate((covariant, jnp.atleast_1d(p)))

    derivatives = jax.jacfwd(covariant_and_pressure)(x)
    dB = derivatives[:3]
    dp = derivatives[3]
    curl_numerator = jnp.asarray(
        [
            dB[2, 1] - dB[1, 2],
            dB[0, 2] - dB[2, 0],
            dB[1, 0] - dB[0, 1],
        ]
    )
    J_sup = curl_numerator / (MU0 * sqrt_g)
    J = basis_vectors @ J_sup
    grad_pressure = jnp.linalg.solve(basis_vectors.T, dp)
    lorentz = jnp.cross(J, B)
    force = lorentz - grad_pressure
    force_cov = basis_vectors.T @ force
    force_helical = curl_numerator[0] / MU0
    reciprocal = jnp.linalg.inv(basis_vectors).T
    radial_force = force_cov[0] * reciprocal[:, 0]
    helical_direction = -B_sup[2] * reciprocal[:, 1] + B_sup[1] * reciprocal[:, 2]
    helical_force = force_helical * helical_direction
    signed_radial_force_density = force_cov[0] * jnp.linalg.norm(reciprocal[:, 0])
    signed_helical_force_density = force_helical * jnp.linalg.norm(helical_direction)
    return (
        sqrt_g,
        B,
        J,
        force,
        force_cov[0],
        force_helical,
        jnp.linalg.norm(radial_force),
        jnp.linalg.norm(helical_force),
        signed_radial_force_density,
        signed_helical_force_density,
        jnp.linalg.norm(lorentz),
        jnp.linalg.norm(grad_pressure),
    )


def _magnetic_pressure(state: HighOrderEquilibriumState, x: Array) -> Array:
    """Return the magnetic pressure ``B^2 / (2 mu0)`` at one point."""

    field = _basic_fields(state, x)[4]
    return jnp.vdot(field, field) / (2.0 * MU0)


def _point_magnetic_pressure_gradient(
    state: HighOrderEquilibriumState, x: Array
) -> Array:
    """Return ``grad(B^2 / 2 mu0)`` as a Cartesian vector at one point."""

    basis_vectors = jax.jacfwd(lambda y: _position(state, y))(x)
    partials = jax.grad(lambda y: _magnetic_pressure(state, y))(x)
    return jnp.linalg.solve(basis_vectors.T, partials)


@jax.jit
def evaluate_magnetic_pressure_gradient(
    state: HighOrderEquilibriumState,
    rho: Array,
    theta: Array,
    zeta: Array,
) -> Array:
    """Evaluate ``grad(B^2 / 2 mu0)`` on broadcast-compatible points.

    This is the vacuum-safe force-balance scale: unlike ``grad(p)`` it does
    not vanish identically when the pressure is flat, so a normalization
    built on its volume average stays meaningful in vacuum.  The final
    dimension of the result is Cartesian and the units are N m^-3.  It is
    evaluated independently of :func:`evaluate_strong_force` so that the
    certificate's force values and the polish residual are untouched.

    .. note::

       This sweeps every certificate point in one ``vmap``, exactly as
       :func:`evaluate_strong_force` does, and so inherits the same peak
       allocation at production 3-D resolution.  When the force sweep gains
       a batching policy, this call must be routed through the same one:
       they run on the same node set, so an unbatched companion would
       reintroduce the allocation the batched sweep exists to avoid.
    """

    rho, theta, zeta = jnp.broadcast_arrays(rho, theta, zeta)
    shape = rho.shape
    points = jnp.stack((rho.reshape(-1), theta.reshape(-1), zeta.reshape(-1)), axis=-1)
    values = jax.vmap(lambda point: _point_magnetic_pressure_gradient(state, point))(points)
    return values.reshape(shape + (3,))


@jax.jit
def evaluate_high_order_fields(
    state: HighOrderEquilibriumState,
    rho: Array,
    theta: Array,
    zeta: Array,
) -> HighOrderFieldSamples:
    """Evaluate the native position, analytic tangents, ``B``, and pressure.

    ``zeta`` spans one field period while ``phi = zeta / nfp`` is the physical
    cylindrical angle. Array inputs broadcast, and the final dimension of
    geometric vectors and ``B`` is Cartesian. Covariant and contravariant field
    components use ``(rho, theta, zeta)``; ``dposition_dphi`` is converted to
    the physical cylindrical angle. The function is pure JAX and is the
    in-memory handoff for downstream field and surface objectives.
    """

    rho, theta, zeta = jnp.broadcast_arrays(rho, theta, zeta)
    shape = rho.shape
    points = jnp.stack((rho.reshape(-1), theta.reshape(-1), zeta.reshape(-1)), axis=-1)

    def sample(point):
        basis, sqrt_g, contravariant, covariant, field, pressure = _basic_fields(
            state, point
        )
        return (
            _position(state, point),
            basis[:, 0],
            basis[:, 1],
            float(state.nfp) * basis[:, 2],
            sqrt_g,
            contravariant,
            covariant,
            field,
            pressure,
        )

    values = jax.vmap(sample)(points)
    vector_shape = shape + (3,)
    return HighOrderFieldSamples(
        rho=rho,
        theta=theta,
        zeta=zeta,
        position=values[0].reshape(vector_shape),
        dposition_drho=values[1].reshape(vector_shape),
        dposition_dtheta=values[2].reshape(vector_shape),
        dposition_dphi=values[3].reshape(vector_shape),
        sqrt_g=values[4].reshape(shape),
        B_contravariant=values[5].reshape(vector_shape),
        B_covariant=values[6].reshape(vector_shape),
        B=values[7].reshape(vector_shape),
        pressure=values[8].reshape(shape),
    )


def evaluate_high_order_surface(
    state: HighOrderEquilibriumState,
    *,
    nphi: int = 32,
    ntheta: int = 32,
) -> HighOrderSurfaceSamples:
    """Return a one-field-period LCFS view with analytic surface tangents.

    The array layout ``(nphi, ntheta, 3)`` and attribute names are accepted
    directly by ESSOS surface objectives. VMEX's virtual-casing adapter uses
    this same view, so both downstream paths share one geometry evaluation.
    """

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / int(state.nfp), int(nphi), endpoint=False)
    pp, tt = jnp.meshgrid(phi, theta, indexing="ij")
    samples = evaluate_high_order_fields(state, 1.0, tt, pp * int(state.nfp))
    normal = jnp.cross(samples.dposition_dtheta, samples.dposition_dphi)
    area_element = jnp.linalg.norm(normal, axis=-1)
    unitnormal = normal / jnp.maximum(area_element[..., None], 1.0e-300)
    mean_radial = jnp.mean(jnp.sum(samples.dposition_drho * unitnormal, axis=-1))
    flip = jnp.where(mean_radial < 0.0, -1.0, 1.0)
    normal = flip * normal
    unitnormal = flip * unitnormal
    return HighOrderSurfaceSamples(
        gamma=samples.position,
        gammadash_theta=samples.dposition_dtheta,
        gammadash_phi=samples.dposition_dphi,
        normal=normal,
        unitnormal=unitnormal,
        area_element=area_element,
        B_total=samples.B,
        theta=theta,
        phi=phi,
        nfp=int(state.nfp),
        ntheta=int(ntheta),
        nphi=int(nphi),
    )


#: Point-batch bound for the force sweep.  One flat ``vmap`` materializes
#: per-point spectral intermediates of order ``mnmax * nbasis`` under two
#: nested ``jacfwd`` levels for *every* point at once; on the W7-X standard
#: configuration (``MPOL = NTOR = 10``, certificate grid ~2.5e5 points) that
#: is a single 34 GB allocation, which is exactly the polish OOM users hit.
#: ``lax.map`` with this batch keeps the sweep vectorized within a batch and
#: sequential across batches; per-point results are independent, so values
#: are unchanged.  4096 points bound the W7-X sweep near ~1 GB.
_FORCE_POINT_BATCH = 4096

#: The batched sweep also carries a remat boundary around the per-point
#: kernel.  ``jax.linearize``/``jax.vjp`` through the sweep otherwise store
#: the whole grid's linearization residuals — ~40 GB for the W7-X polish
#: setup's 6.8e4-point chart probes, the second half of the same OOM.  With
#: the checkpoint, reverse passes replay the per-point force kernel instead
#: of storing its intermediates: memory falls to per-batch scale for one
#: extra forward evaluation per backward pass.  The flat small-grid path is
#: untouched, so optimization-loop gradients keep their exact cost.
_point_force_checkpoint = jax.checkpoint(_point_force)


@jax.jit
def evaluate_strong_force(
    state: HighOrderEquilibriumState,
    rho: Array,
    theta: Array,
    zeta: Array,
) -> StrongForceSamples:
    """Evaluate the independent continuum force on broadcast-compatible points.

    Points on the coordinate-singular magnetic axis are intentionally excluded;
    use a shifted radial quadrature as :func:`certify_strong_force` does.
    Compilation is cached by basis metadata and broadcast point shape.
    """

    rho, theta, zeta = jnp.broadcast_arrays(rho, theta, zeta)
    shape = rho.shape
    points = jnp.stack((rho.reshape(-1), theta.reshape(-1), zeta.reshape(-1)), axis=-1)
    if points.shape[0] <= _FORCE_POINT_BATCH:
        values = jax.vmap(lambda point: _point_force(state, point))(points)
    else:
        values = jax.lax.map(
            lambda point: _point_force_checkpoint(state, point),
            points,
            batch_size=_FORCE_POINT_BATCH,
        )

    def reshape_scalar(value: Array) -> Array:
        return value.reshape(shape)

    def reshape_vector(value: Array) -> Array:
        return value.reshape(shape + (3,))

    return StrongForceSamples(
        rho=rho,
        theta=theta,
        zeta=zeta,
        sqrt_g=reshape_scalar(values[0]),
        B=reshape_vector(values[1]),
        J=reshape_vector(values[2]),
        force=reshape_vector(values[3]),
        force_rho=reshape_scalar(values[4]),
        force_helical=reshape_scalar(values[5]),
        radial_force_density=reshape_scalar(values[6]),
        helical_force_density=reshape_scalar(values[7]),
        signed_radial_force_density=reshape_scalar(values[8]),
        signed_helical_force_density=reshape_scalar(values[9]),
        lorentz_norm=reshape_scalar(values[10]),
        grad_pressure_norm=reshape_scalar(values[11]),
    )


def _constrained_spline_fit(
    basis: BSplineBasis,
    samples: np.ndarray,
    s: np.ndarray,
    *,
    mode_m: int | None = None,
    fix_axis: bool = False,
    fix_edge: bool = False,
) -> np.ndarray:
    """Host-side least-squares lift with exact clamped endpoint values."""

    values = np.asarray(samples, dtype=float)
    s = np.asarray(s, dtype=float)
    if mode_m is not None:
        radial = np.power(np.sqrt(np.maximum(s, 0.0)), abs(int(mode_m)))
        keep = radial > 1.0e-10
        values = values[keep] / radial[keep]
        nodes = s[keep]
    else:
        nodes = s
    matrix = np.asarray(basis.basis_matrix(nodes), dtype=float)
    coefficients = np.zeros((basis.size,), dtype=float)
    fixed: dict[int, float] = {}
    if fix_axis:
        fixed[0] = float(samples[0])
    if fix_edge:
        fixed[basis.size - 1] = float(samples[-1])
    free = np.asarray([index for index in range(basis.size) if index not in fixed], dtype=int)
    rhs = values.copy()
    for index, value in fixed.items():
        coefficients[index] = value
        rhs -= matrix[:, index] * value
    if free.size:
        coefficients[free] = np.linalg.lstsq(matrix[:, free], rhs, rcond=1.0e-12)[0]
    return coefficients


def lift_high_order_state(
    state: Any,
    runtime: Any,
    *,
    radial_basis: BSplineBasis | None = None,
    degree: int = 5,
    max_spans: int = 32,
    source: str = "VMEX legacy equilibrium",
) -> HighOrderEquilibriumState:
    """Lift a converged legacy :class:`SpectralState` into the smooth basis.

    The lift is a one-time host operation.  It undoes VMEX's m=1 constrained
    variables and Fourier normalization, converts internal lambda to the wout
    convention, enforces ``rho**abs(m)`` regularity, preserves the magnetic
    axis for m=0 and the fixed R/Z boundary exactly, and removes the lambda
    ``(m,n)=(0,0)`` gauge mode structurally.
    """

    from . import postprocess as _pp
    from .geometry import apply_lambda_axis_closure
    from .residuals import m1_constrained_to_physical
    from .statephysics import _field_chain
    from .transforms import physical_to_internal_scale

    setup = runtime.setup
    modes = runtime.modes
    s = np.asarray(setup.s_full, dtype=float)
    if radial_basis is None:
        # The legacy full mesh is only a first-order radial representation.
        # Giving a degree-p spline one coefficient per sample interpolates
        # that mesh noise exactly; its second derivatives then oscillate even
        # though R/Z/lambda values look innocuous.  This is catastrophic for
        # curl(B): on the ns=11 Solovev regression the interpolating degree-5
        # lift reports |JxB-grad(p)|_L2 = 4.97e7 N/m^3 and a 0.81 quadrature
        # change, whereas the overdetermined eight-coefficient fit reports
        # 9.73e1 N/m^3 and 3.8e-5.  Start with roughly two mesh samples per
        # free span.  Explicit ``radial_basis=`` remains the seam for a caller
        # that has a genuinely high-order source or has selected knots by a
        # refinement certificate.
        available_spans = max(1, s.size - int(degree))
        regularized_spans = max(1, (available_spans + 1) // 2)
        spans = max(1, min(int(max_spans), regularized_spans))
        radial_basis = BSplineBasis.clamped(
            np.linspace(0.0, 1.0, spans + 1),
            degree=degree,
            quadrature_order=degree + 3,
        )

    R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
        state.R_cos,
        state.Z_sin,
        state.R_sin,
        state.Z_cos,
        modes=modes,
        lthreed=setup.lthreed,
        lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    mode_scale = 1.0 / physical_to_internal_scale(modes, runtime.trig)
    physical = [np.asarray(block, dtype=float) * mode_scale[None, :] for block in (R_cos, R_sin, Z_cos, Z_sin)]

    _, _, _, fields, _ = _field_chain(state, runtime)
    phipf = np.asarray(setup.phipf, dtype=float)
    chipf = _pp.chipf_from_chips(np.asarray(fields.chips, dtype=float))
    pressure = _pp.full_mesh_from_half(np.asarray(fields.pressure, dtype=float)) / MU0

    lambda_sin = apply_lambda_axis_closure(state.L_sin, modes=modes, ntor=runtime.resolution.ntor)
    lambda_internal = (np.asarray(state.L_cos, dtype=float), np.asarray(lambda_sin, dtype=float))
    safe_phip = np.where(phipf != 0.0, phipf, 1.0)
    lambda_physical = [
        block * mode_scale[None, :] * float(np.asarray(setup.lamscale)) / safe_phip[:, None]
        for block in lambda_internal
    ]

    m_values = np.asarray(modes.m, dtype=int)

    def fit_modes(table: np.ndarray, *, fixed_boundary: bool) -> np.ndarray:
        fitted = np.stack(
            [
                _constrained_spline_fit(
                    radial_basis,
                    table[:, column],
                    s,
                    mode_m=int(m_values[column]),
                    fix_axis=int(m_values[column]) == 0,
                    fix_edge=fixed_boundary,
                )
                for column in range(table.shape[1])
            ],
            axis=0,
        )
        return fitted

    R_cos_q, R_sin_q, Z_cos_q, Z_sin_q = (fit_modes(table, fixed_boundary=True) for table in physical)
    L_cos_q = fit_modes(lambda_physical[0], fixed_boundary=False)
    L_sin_q = fit_modes(lambda_physical[1], fixed_boundary=False)
    gauge = (m_values == 0) & (np.asarray(modes.n, dtype=int) == 0)
    L_cos_q[gauge] = 0.0
    L_sin_q[gauge] = 0.0

    fit_profile = lambda values: _constrained_spline_fit(  # noqa: E731
        radial_basis, np.asarray(values), s, fix_axis=True, fix_edge=True
    )
    return HighOrderEquilibriumState(
        radial_basis=radial_basis,
        m=m_values,
        n=np.asarray(modes.n, dtype=int),
        nfp=int(runtime.resolution.nfp),
        R_cos=jnp.asarray(R_cos_q),
        R_sin=jnp.asarray(R_sin_q),
        Z_cos=jnp.asarray(Z_cos_q),
        Z_sin=jnp.asarray(Z_sin_q),
        L_cos=jnp.asarray(L_cos_q),
        L_sin=jnp.asarray(L_sin_q),
        phipf=jnp.asarray(fit_profile(phipf)),
        chipf=jnp.asarray(fit_profile(chipf)),
        pressure=jnp.asarray(fit_profile(pressure)),
        jacobian_sign=int(setup.signgs),
        source=source,
        boundary_R_cos=jnp.asarray(physical[0][-1]),
        boundary_R_sin=jnp.asarray(physical[1][-1]),
        boundary_Z_cos=jnp.asarray(physical[2][-1]),
        boundary_Z_sin=jnp.asarray(physical[3][-1]),
    )


def high_order_state_from_wout(
    wout: Any,
    *,
    inp: Any,
    radial_basis: BSplineBasis | None = None,
    degree: int = 5,
    max_spans: int = 32,
) -> HighOrderEquilibriumState:
    """Import a VMEX-, VMEC2000-, or VMEC++-compatible wout continuously.

    ``inp`` supplies the source equilibrium's flux/profile conventions.  Mode
    remapping and lambda half/full-mesh inversion are delegated to the tested
    hot-restart path, after which :func:`lift_high_order_state` performs the
    axis-regular constrained lift.  No external equilibrium code is a runtime
    dependency.
    """

    from .restart import state_from_wout
    from .solver import prepare_runtime, resolution_from_input
    from .wout import read_wout

    source_wout = read_wout(wout) if isinstance(wout, (str, Path)) else wout
    ns = int(source_wout.ns)
    state = state_from_wout(source_wout, inp=inp, ns=ns)
    runtime = prepare_runtime(inp, resolution_from_input(inp, ns=ns))
    source_name = getattr(source_wout, "input_extension", "") or "VMEC-compatible wout"
    return lift_high_order_state(
        state,
        runtime,
        radial_basis=radial_basis,
        degree=degree,
        max_spans=max_spans,
        source=str(source_name),
    )


def _weighted_l2(values: Array, weights: Array) -> Array:
    return jnp.sqrt(jnp.sum(weights * values * values) / jnp.sum(weights))


def _weighted_quantile(values: Array, weights: Array, quantile: float) -> Array:
    """Deterministic volume-weighted quantile for a fixed validation grid."""

    weights = jnp.broadcast_to(weights, jnp.shape(values)).reshape(-1)
    values = jnp.ravel(values)
    order = jnp.argsort(values)
    sorted_values = values[order]
    cumulative = jnp.cumsum(weights[order])
    threshold = float(quantile) * cumulative[-1]
    index = jnp.clip(jnp.searchsorted(cumulative, threshold), 0, sorted_values.size - 1)
    return sorted_values[index]


def _angular_spectral_tail(magnitude: Array) -> Array:
    """Fraction of angular power above two thirds of the resolvable harmonics.

    ``magnitude`` is sampled on ``(s, theta, zeta)``.  ``rfftn`` over the two
    angles transforms theta in full, so its rows run
    ``m = 0, 1, ..., ntheta/2, -ntheta/2, ..., -1``: a plain index cut on that
    axis selects the *lowest* ``|m|`` negative frequencies, the opposite of a
    tail.  Mask on ``|m|`` instead, and take the union with the toroidal
    condition so the high-``m``, high-``n`` corner is counted once.
    """
    power = jnp.abs(jnp.fft.rfftn(magnitude, axes=(1, 2))) ** 2
    ntheta, nzeta_half = power.shape[1], power.shape[2]
    poloidal = jnp.abs(jnp.fft.fftfreq(ntheta) * ntheta)
    toroidal = jnp.arange(nzeta_half)
    theta_cut = max(1.0, (2.0 / 3.0) * (ntheta // 2))
    zeta_cut = max(1.0, (2.0 / 3.0) * max(nzeta_half - 1, 1))
    tail_mask = (poloidal[:, None] >= theta_cut) | (toroidal[None, :] >= zeta_cut)
    tail = jnp.sum(jnp.where(tail_mask, power, 0.0))
    return jnp.sqrt(tail / jnp.maximum(jnp.sum(power), 1.0e-300))


def _nestedness_margin(signed_jacobian: Array) -> Array:
    """How close the surfaces come to touching, as a scale-free fraction.

    ``minimum_signed_jacobian`` answers the same question in machine units,
    so it says nothing on its own: a small tokamak and a reactor with equally
    healthy surfaces report numbers three decades apart.  Dividing by the mean
    magnitude removes the size of the device, since a geometry scaled by
    ``a`` scales every volume element by ``a**3``.  The margin is positive
    where the surfaces are nested, and crosses zero exactly where the
    Jacobian does.
    """
    scale = jnp.mean(jnp.abs(signed_jacobian))
    return jnp.min(signed_jacobian) / jnp.maximum(scale, 1.0e-300)


def certify_strong_force(
    state: HighOrderEquilibriumState,
    *,
    angular_multiplier: int = 2,
    radial_order_increment: int = 2,
    force_floor: float = 1.0e-12,
    window: tuple[float, float] = (0.1, 0.99),
) -> StrongForceReport:
    """Evaluate a shifted, overintegrated certificate distinct from solve nodes.

    The acceptance criterion is the pointwise ``eps_F`` volume L2
    (``normalized_l2``), which is bounded above by 2 by construction.  The
    report also carries the volume-averaged normalizations of
    :class:`ForceErrorNormalizations` twice: over the whole domain, and over
    the flux window ``window`` (``s`` in ``[0.1, 0.99]`` by default), which
    excludes the coordinate-singular axis and the boundary layer at the edge
    where the raw maxima live.
    """

    if not 0.0 <= float(window[0]) < float(window[1]) <= 1.0:
        raise ValueError("window must satisfy 0 <= s_min < s_max <= 1")
    breaks = state.radial_basis.breakpoints

    def radial_quadrature(quadrature_order: int) -> tuple[np.ndarray, np.ndarray]:
        gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(quadrature_order)
        nodes = (
            0.5 * (breaks[:-1, None] + breaks[1:, None]) + 0.5 * np.diff(breaks)[:, None] * gauss_nodes[None, :]
        ).reshape(-1)
        weights = (0.5 * np.diff(breaks)[:, None] * gauss_weights[None, :]).reshape(-1)
        return nodes, weights

    order = state.radial_basis.degree + int(radial_order_increment) + 1
    s_nodes, s_weights = radial_quadrature(order)
    rho_nodes = np.sqrt(s_nodes)
    rho_weights = s_weights / (2.0 * rho_nodes)
    max_m = int(np.max(np.asarray(state.m), initial=0))
    max_n = int(np.max(np.abs(np.asarray(state.n)), initial=0))
    ntheta = max(8, int(angular_multiplier) * 2 * (max_m + 1))
    nzeta = max(4, int(angular_multiplier) * 2 * (max_n + 1))
    theta = (np.arange(ntheta) + 0.5) * (2.0 * np.pi / ntheta)
    zeta = (np.arange(nzeta) + 0.375) * (2.0 * np.pi / nzeta)
    rr, tt, zz = jnp.meshgrid(jnp.asarray(rho_nodes), jnp.asarray(theta), jnp.asarray(zeta), indexing="ij")
    samples = evaluate_strong_force(state, rr, tt, zz)
    magnitude = jnp.linalg.norm(samples.force, axis=-1)
    weights = (
        jnp.asarray(rho_weights)[:, None, None]
        * (2.0 * jnp.pi / ntheta)
        * (2.0 * jnp.pi / nzeta)
        * jnp.abs(samples.sqrt_g)
    )
    normalized = 2.0 * magnitude / (samples.lorentz_norm + samples.grad_pressure_norm + float(force_floor))
    radial_normalized = (
        2.0 * samples.radial_force_density / (samples.lorentz_norm + samples.grad_pressure_norm + float(force_floor))
    )
    helical_normalized = (
        2.0 * samples.helical_force_density / (samples.lorentz_norm + samples.grad_pressure_norm + float(force_floor))
    )
    # The published global normalizations. They share the certificate's
    # nodes and |sqrt(g)| volume weights but never saturate, so they remain
    # informative exactly where eps_F pins at its ceiling.
    magnetic_pressure_gradient_norm = jnp.linalg.norm(
        evaluate_magnetic_pressure_gradient(state, rr, tt, zz), axis=-1
    )

    def normalizations(
        radial_mask: np.ndarray, s_min: float, s_max: float
    ) -> ForceErrorNormalizations:
        masked_weights = weights * jnp.asarray(radial_mask, dtype=weights.dtype)[:, None, None]
        total = jnp.sum(masked_weights)
        safe_total = jnp.where(total > 0.0, total, 1.0)

        def average(values: Array) -> Array:
            return jnp.where(total > 0.0, jnp.sum(masked_weights * values) / safe_total, 0.0)

        mean_force = average(magnitude)
        mean_grad_pressure = average(samples.grad_pressure_norm)
        mean_magnetic = average(magnetic_pressure_gradient_norm)
        magnetic_scale = mean_magnetic + float(force_floor)
        magnetic_normalized = magnitude / magnetic_scale
        return ForceErrorNormalizations(
            volume_average_force=mean_force,
            volume_average_grad_pressure=mean_grad_pressure,
            volume_average_magnetic_pressure_gradient=mean_magnetic,
            relative_force_error=jnp.where(
                mean_grad_pressure > 0.0,
                mean_force / jnp.where(mean_grad_pressure > 0.0, mean_grad_pressure, 1.0),
                jnp.nan,
            ),
            magnetic_relative_force_error=mean_force / magnetic_scale,
            magnetic_normalized_l2=jnp.sqrt(average(magnetic_normalized * magnetic_normalized)),
            magnetic_normalized_linf=jnp.max(
                jnp.where(
                    jnp.asarray(radial_mask)[:, None, None],
                    magnetic_normalized,
                    0.0,
                )
            ),
            node_count=int(np.count_nonzero(radial_mask)),
            s_min=float(s_min),
            s_max=float(s_max),
        )

    window_min, window_max = float(window[0]), float(window[1])
    window_mask = (s_nodes >= window_min) & (s_nodes <= window_max)
    global_normalizations = normalizations(
        np.ones_like(s_nodes, dtype=bool), float(breaks[0]), float(breaks[-1])
    )
    window_normalizations = normalizations(window_mask, window_min, window_max)

    surface_weights = jnp.abs(samples.sqrt_g)
    fsa = jnp.sum(surface_weights * magnitude, axis=(1, 2)) / jnp.sum(surface_weights, axis=(1, 2))
    surface_normalized_l2 = jnp.sqrt(
        jnp.sum(surface_weights * normalized * normalized, axis=(1, 2))
        / jnp.sum(surface_weights, axis=(1, 2))
    )

    def region_l2(mask: Array) -> Array:
        region_weights = weights * mask[:, None, None]
        denominator = jnp.sum(region_weights)
        return jnp.where(
            denominator > 0.0,
            jnp.sqrt(jnp.sum(region_weights * magnitude * magnitude) / denominator),
            0.0,
        )

    tail = _angular_spectral_tail(magnitude)
    coarse_order = max(state.radial_basis.degree + 1, order - 2)
    coarse_s, coarse_s_weights = radial_quadrature(coarse_order)
    coarse_rho = np.sqrt(coarse_s)
    coarse_rho_weights = coarse_s_weights / (2.0 * coarse_rho)
    coarse_rr, coarse_tt, coarse_zz = jnp.meshgrid(
        jnp.asarray(coarse_rho), jnp.asarray(theta), jnp.asarray(zeta), indexing="ij"
    )
    coarse_samples = evaluate_strong_force(state, coarse_rr, coarse_tt, coarse_zz)
    coarse_magnitude = jnp.linalg.norm(coarse_samples.force, axis=-1)
    coarse_weights = (
        jnp.asarray(coarse_rho_weights)[:, None, None]
        * (2.0 * jnp.pi / ntheta)
        * (2.0 * jnp.pi / nzeta)
        * jnp.abs(coarse_samples.sqrt_g)
    )
    fine_l2 = _weighted_l2(magnitude, weights)
    coarse_l2 = _weighted_l2(coarse_magnitude, coarse_weights)
    refinement = jnp.abs(fine_l2 - coarse_l2) / jnp.maximum(fine_l2, float(force_floor))
    signed_jacobian = float(state.jacobian_sign) * samples.sqrt_g / jnp.maximum(rr, 1.0e-14)

    edge = jnp.asarray(1.0)
    edge_factor = edge ** jnp.asarray(state.m)
    boundary_residual = jnp.asarray(0.0)
    for coefficients, target in (
        (state.R_cos, state.boundary_R_cos),
        (state.R_sin, state.boundary_R_sin),
        (state.Z_cos, state.boundary_Z_cos),
        (state.Z_sin, state.boundary_Z_sin),
    ):
        if target is not None:
            value = state.radial_basis.evaluate(coefficients, edge, axis=-1) * edge_factor
            boundary_residual = jnp.maximum(boundary_residual, jnp.max(jnp.abs(value - jnp.asarray(target))))
    gauge_mask = (np.asarray(state.m) == 0) & (np.asarray(state.n) == 0)
    gauge_residual = jnp.asarray(0.0)
    if np.any(gauge_mask):
        gauge_residual = jnp.maximum(
            jnp.max(jnp.abs(jnp.asarray(state.L_cos)[gauge_mask])),
            jnp.max(jnp.abs(jnp.asarray(state.L_sin)[gauge_mask])),
        )

    return StrongForceReport(
        absolute_l2=fine_l2,
        absolute_p99=_weighted_quantile(magnitude, weights, 0.99),
        absolute_linf=jnp.max(magnitude),
        normalized_l2=_weighted_l2(normalized, weights),
        normalized_p99=_weighted_quantile(normalized, weights, 0.99),
        normalized_linf=jnp.max(normalized),
        radial_l2=_weighted_l2(samples.radial_force_density, weights),
        helical_l2=_weighted_l2(samples.helical_force_density, weights),
        radial_normalized_l2=_weighted_l2(radial_normalized, weights),
        helical_normalized_l2=_weighted_l2(helical_normalized, weights),
        near_axis_l2=region_l2(jnp.asarray(rho_nodes) < 0.2),
        bulk_l2=region_l2((jnp.asarray(rho_nodes) >= 0.2) & (jnp.asarray(rho_nodes) <= 0.8)),
        edge_l2=region_l2(jnp.asarray(rho_nodes) > 0.8),
        radial_nodes=jnp.asarray(rho_nodes),
        flux_surface_average=fsa,
        flux_surface_normalized_l2=surface_normalized_l2,
        angular_spectral_tail=tail,
        radial_refinement_difference=refinement,
        minimum_signed_jacobian=jnp.min(signed_jacobian),
        nestedness_margin=_nestedness_margin(signed_jacobian),
        boundary_residual=boundary_residual,
        gauge_residual=gauge_residual,
        global_normalizations=global_normalizations,
        window_normalizations=window_normalizations,
        force_floor=float(force_floor),
    )


#: Row labels for the non-saturating force-error measures, in the order
#: :func:`force_error_measures` emits them.  ``<|.|>`` is the ``|sqrt(g)|``
#: volume average over the certificate's own quadrature nodes.
FORCE_ERROR_MEASURE_LABELS = (
    "<|F|>  [N m^-3]",
    "<|F|>/<|grad p|>",
    "<|F|>/<|grad B^2/2mu0|>",
    "|F| L2 /<|grad B^2/2mu0|>",
    "|F| L2 near axis [N m^-3]",
    "|F| L2 bulk      [N m^-3]",
    "|F| L2 edge      [N m^-3]",
)


def force_error_measures(
    initial: StrongForceReport,
    final: StrongForceReport | None = None,
) -> tuple[tuple[str, float, float | None], ...]:
    """Return the non-saturating force-error rows for one or two reports.

    ``eps_F`` cannot exceed 2, so every place that quotes it must quote
    something that can move: the dimensional ``<|F|>``, the Panici et al.
    2023 ratio ``<|F|>/<|grad p|>`` (``nan`` in vacuum, where it is
    undefined), the vacuum-safe DESC/Thun ratio ``<|F|>/<|grad(B^2/2mu0)|>``
    and its pointwise L2 companion, and the near-axis/bulk/edge split of the
    dimensional ``|F|`` that says where the residual sits.  The volume
    averages come from ``window_normalizations``, so the axis and the edge
    boundary layer do not dominate them; the region split is over the whole
    domain, which is the point of splitting it.
    """

    def values(report: StrongForceReport) -> tuple[float, ...]:
        window = report.window_normalizations
        return (
            float(window.volume_average_force),
            float(window.relative_force_error),
            float(window.magnetic_relative_force_error),
            float(window.magnetic_normalized_l2),
            float(report.near_axis_l2),
            float(report.bulk_l2),
            float(report.edge_l2),
        )

    before = values(initial)
    after = values(final) if final is not None else (None,) * len(before)
    return tuple(zip(FORCE_ERROR_MEASURE_LABELS, before, after, strict=True))


def force_error_record(report: StrongForceReport) -> dict[str, Any]:
    """Serialize a certificate's normalizations for a committed artifact.

    Both the saturating pointwise ``eps_F`` block and the volume-averaged
    normalizations are written, together with the ``saturation`` disclosure,
    so a JSON reader sees the bound at the same time as the number.
    """

    def block(normalizations: ForceErrorNormalizations) -> dict[str, Any]:
        return {
            "volume_average_force": float(normalizations.volume_average_force),
            "volume_average_grad_pressure": float(
                normalizations.volume_average_grad_pressure
            ),
            "volume_average_magnetic_pressure_gradient": float(
                normalizations.volume_average_magnetic_pressure_gradient
            ),
            "relative_force_error": float(normalizations.relative_force_error),
            "magnetic_relative_force_error": float(
                normalizations.magnetic_relative_force_error
            ),
            "magnetic_normalized_l2": float(normalizations.magnetic_normalized_l2),
            "magnetic_normalized_linf": float(normalizations.magnetic_normalized_linf),
            "node_count": int(normalizations.node_count),
            "s_min": float(normalizations.s_min),
            "s_max": float(normalizations.s_max),
        }

    return {
        "normalization": report.normalization,
        "saturation": report.saturation,
        "pointwise_eps_f": {
            "normalized_l2": float(report.normalized_l2),
            "normalized_p99": float(report.normalized_p99),
            "normalized_linf": float(report.normalized_linf),
        },
        "absolute": {
            "l2": float(report.absolute_l2),
            "p99": float(report.absolute_p99),
            "linf": float(report.absolute_linf),
            "near_axis_l2": float(report.near_axis_l2),
            "bulk_l2": float(report.bulk_l2),
            "edge_l2": float(report.edge_l2),
        },
        "global_normalizations": block(report.global_normalizations),
        "window_normalizations": block(report.window_normalizations),
    }


def plot_strong_force_report(
    reports: StrongForceReport | dict[str, StrongForceReport],
    *,
    title: str = "Independent strong-force certificate",
):
    """Plot force profiles and compact absolute/normalized comparisons.

    A mapping produces an overlaid, reviewer-facing comparison; passing one
    report is shorthand for ``{"equilibrium": report}``.  Matplotlib is
    imported lazily so force evaluation itself stays a lightweight core API.
    """

    import matplotlib.pyplot as plt

    report_map = {"equilibrium": reports} if isinstance(reports, StrongForceReport) else reports
    if not report_map:
        raise ValueError("at least one strong-force report is required")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for label, report in report_map.items():
        axes[0].semilogy(
            np.asarray(report.radial_nodes),
            np.asarray(report.flux_surface_normalized_l2),
            marker=".",
            linewidth=1.5,
            label=label,
        )
    axes[0].set(
        xlabel=r"$\rho=\sqrt{s}$",
        ylabel=r"normalized force error",
    )
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    labels = list(report_map)
    positions = np.arange(len(labels), dtype=float)
    width = 0.22
    for offset, field, legend in (
        (-width, "absolute_l2", "volume L2"),
        (0.0, "absolute_p99", "volume P99"),
        (width, "absolute_linf", "Linf"),
    ):
        axes[1].bar(
            positions + offset,
            [float(np.asarray(getattr(report_map[label], field))) for label in labels],
            width=width,
            label=legend,
        )
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, labels, rotation=15, ha="right")
    axes[1].set_ylabel(r"force density [N m$^{-3}$]")
    axes[1].grid(True, axis="y", which="both", alpha=0.25)
    axes[1].legend(frameon=False)
    figure.suptitle(title)
    return figure, axes


__all__ = [
    "FORCE_ERROR_MEASURE_LABELS",
    "ForceErrorNormalizations",
    "HighOrderEquilibriumState",
    "HighOrderFieldSamples",
    "HighOrderSurfaceSamples",
    "StrongForceReport",
    "StrongForceSamples",
    "certify_strong_force",
    "evaluate_high_order_fields",
    "evaluate_high_order_surface",
    "evaluate_magnetic_pressure_gradient",
    "evaluate_strong_force",
    "force_error_measures",
    "force_error_record",
    "high_order_state_from_wout",
    "lift_high_order_state",
    "plot_strong_force_report",
]
