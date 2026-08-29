"""The vmex-to-ESSOS field handoff, and alpha tracing on top of it.

- :func:`essos_vmec_field` — hand a solved equilibrium (or a wout file) to
  ESSOS as an ``essos.fields.Vmec``, ready for ESSOS tracing, surfaces and
  field queries.  The seam runs one way, ESSOS reading a VMEC equilibrium;
  an ESSOS coil field entering a vmex free-boundary solve goes the other way
  through :meth:`~vmex.core.mgrid.MgridField.from_coils`.
- :func:`trace_alphas` — trace fusion-born alpha particles launched from one
  flux surface of an equilibrium and return the exact loss-fraction
  diagnostics as an :class:`AlphaTracingResult`.

The tracer is ESSOS (``essos.dynamics.Tracing`` over ``essos.fields.Vmec``),
imported inside the call so vmex imports without ESSOS installed.  Only the
released ESSOS surface is used — ``Vmec(wout_filename)``, ``Particles``,
``Tracing`` and its ``loss_fractions``/``lost_times``/``trajectories``
outputs.  Consequences of that restriction:

- ``essos.fields.Vmec`` reads a wout *file*, so an in-memory equilibrium
  (:class:`~vmex.core.wout.WoutData`) takes a temporary-wout hop through
  :func:`~vmex.core.wout.write_wout`.  The file write severs any gradient;
  the differentiable loss-fraction objective is a separate feature gated on
  the ESSOS array constructor (uwplasma/ESSOS#61) and is not provided here.
- The particle energy along each orbit is reconstructed locally from
  ``0.5*m*v_par^2 + mu*|B|`` (released ESSOS exposes it inconsistently
  across versions: an eager array in 0.16, a method on the development line).
- Released ESSOS keeps no solver-failure ledger, so ``particles_failed``
  counts trajectories with non-finite samples that ESSOS did not attribute
  to a boundary loss; axis terminations are reported when the installed
  ESSOS tracks them and are zero otherwise (0.16 has no axis event).

Particles are sampled uniformly on the surface ``s``: ``theta`` over
``[0, 2*pi)``, ``phi`` over one field period ``[0, 2*pi/nfp)``, and pitch
``v_par/v`` over ``[-1, 1)``, from ``jax.random.PRNGKey(seed)``.  A particle
counts as lost when its orbit reaches ``s >= 0.99`` (the ESSOS
``loss_fraction`` criterion).
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _essos_imports():
    try:
        from essos import constants, dynamics, fields
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "alpha-particle tracing requires ESSOS (`pip install essos`)"
        ) from exc
    return constants, dynamics, fields


@dataclasses.dataclass(frozen=True)
class AlphaTracingResult:
    """Exact alpha-loss diagnostics from one :func:`trace_alphas` call.

    ``trajectories`` holds the guiding-centre coordinates
    ``(s, theta, phi, v_par)`` at each saved time; lost orbits keep the
    non-finite post-event samples ESSOS writes after the boundary crossing.
    ``lost_times`` is ``-1`` for particles that were never lost.
    """

    nparticles: int
    loss_fraction: float
    particles_lost: int
    particles_unresolved: int
    particles_failed: int
    wall_time_s: float
    particle_energy: float
    total_speed: float
    times: np.ndarray = dataclasses.field(repr=False)
    loss_fractions: np.ndarray = dataclasses.field(repr=False)
    lost_times: np.ndarray = dataclasses.field(repr=False)
    trajectories: np.ndarray = dataclasses.field(repr=False)
    trajectories_xyz: np.ndarray = dataclasses.field(repr=False)
    energies: np.ndarray = dataclasses.field(repr=False)


def essos_vmec_field(source: Any, **kwargs: Any) -> Any:
    """Return the ``essos.fields.Vmec`` field for an equilibrium or wout file.

    ``source`` is a path to a ``wout_*.nc`` file or an in-memory
    :class:`~vmex.core.wout.WoutData`.  Released ESSOS reads a wout *file*,
    so an in-memory equilibrium is written to a temporary wout; ESSOS loads
    every table eagerly in its constructor, so the file is gone by the time
    the field is returned.  That write severs the gradient — this seam is
    for diagnostics, not for differentiating through ESSOS.

    ``kwargs`` reach ``essos.fields.Vmec`` unchanged (``ntheta``, ``nphi``,
    ``close`` and ``range_torus`` on the released constructor, which set the
    resolution of the ``field.surface`` ESSOS builds alongside the field).

    Released ESSOS reads the stellarator-symmetric wout tables only, so an
    ``lasym`` equilibrium is rejected rather than silently half-transferred.
    """
    _, _, fields = _essos_imports()

    if hasattr(source, "rmnc") and hasattr(source, "xm"):  # WoutData
        if bool(source.lasym):
            raise ValueError(
                "released ESSOS reads stellarator-symmetric wout tables only; "
                "the lasym partner tables would be silently dropped"
            )
        from .wout import write_wout

        with tempfile.TemporaryDirectory(prefix="vmex_essos_") as tmp:
            wout_path = Path(tmp) / "wout_equilibrium.nc"
            write_wout(wout_path, source)
            return fields.Vmec(str(wout_path), **kwargs)

    wout_path = Path(source)
    import netCDF4

    with netCDF4.Dataset(str(wout_path)) as ds:
        if bool(int(ds.variables["lasym__logical__"][()])):
            raise ValueError(
                "released ESSOS reads stellarator-symmetric wout tables only; "
                f"{wout_path.name} is an lasym equilibrium"
            )
    return fields.Vmec(str(wout_path), **kwargs)


def trace_alphas(
    source: Any,
    *,
    tmax: float = 3e-4,
    nparticles: int = 200,
    s: float = 0.25,
    seed: int = 42,
    timestep: float = 5e-7,
    times_to_trace: int = 200,
    model: str = "GuidingCenter",
) -> AlphaTracingResult:
    """Trace fusion alphas from surface ``s`` of a wout file or equilibrium.

    Parameters
    ----------
    source:
        Path to a ``wout_*.nc`` file, or an in-memory
        :class:`~vmex.core.wout.WoutData`; handed to ESSOS by
        :func:`essos_vmec_field`.
    tmax, timestep, times_to_trace:
        Integration horizon [s], integrator step [s], and number of saved
        samples (uniform in time, including ``t = 0``).
    nparticles, s, seed:
        Ensemble size, launch surface, and sampling seed (see module notes).
    model:
        ESSOS tracing model (``"GuidingCenter"`` by default).
    """
    essos = _essos_imports()
    return _trace_vmec_field(
        essos_vmec_field(source), essos, tmax=tmax, nparticles=nparticles,
        s=s, seed=seed, timestep=timestep, times_to_trace=times_to_trace,
        model=model,
    )


def _trace_vmec_field(
    vmec, essos, *, tmax: float, nparticles: int,
    s: float, seed: int, timestep: float, times_to_trace: int, model: str,
) -> AlphaTracingResult:
    constants, dynamics, _fields = essos
    import jax
    import jax.numpy as jnp

    theta_key, phi_key, pitch_key = jax.random.split(
        jax.random.PRNGKey(int(seed)), 3)
    theta = jax.random.uniform(
        theta_key, (nparticles,), minval=0.0, maxval=2.0 * jnp.pi)
    phi = jax.random.uniform(
        phi_key, (nparticles,), minval=0.0, maxval=2.0 * jnp.pi / int(vmec.nfp))
    pitch = jax.random.uniform(
        pitch_key, (nparticles,), minval=-1.0, maxval=1.0)
    particles = dynamics.Particles(
        initial_xyz=jnp.stack(
            [jnp.full((nparticles,), float(s)), theta, phi], axis=1),
        initial_vparallel_over_v=pitch,
        mass=constants.ALPHA_PARTICLE_MASS,
        charge=constants.ALPHA_PARTICLE_CHARGE,
        energy=constants.FUSION_ALPHA_PARTICLE_ENERGY,
    )

    start = time.perf_counter()
    tracing = dynamics.Tracing(
        field=vmec, particles=particles, maxtime=float(tmax),
        timestep=float(timestep), times_to_trace=int(times_to_trace),
        model=model,
    )
    wall_time_s = time.perf_counter() - start

    trajectories = np.asarray(tracing.trajectories, dtype=float)
    lost_times = np.asarray(tracing.lost_times, dtype=float)
    loss_fractions = np.asarray(tracing.loss_fractions, dtype=float)

    # Energy along each orbit from the guiding-centre invariant
    # E = 0.5*m*v_par^2 + mu*|B|, with mu fixed by the first sample.
    mass = float(particles.mass)
    particle_energy = float(particles.energy)
    absB = np.asarray(
        jax.vmap(vmec.AbsB)(
            jnp.asarray(trajectories[:, :, :3].reshape(-1, 3))),
        dtype=float,
    ).reshape(nparticles, -1)
    vpar = trajectories[:, :, 3]
    mu = (particle_energy - 0.5 * mass * vpar[:, 0] ** 2) / absB[:, 0]
    energies = 0.5 * mass * vpar**2 + mu[:, None] * absB

    lost_mask = lost_times >= 0.0
    finite_mask = np.isfinite(trajectories).all(axis=(1, 2))
    return AlphaTracingResult(
        nparticles=int(nparticles),
        loss_fraction=float(loss_fractions[-1]),
        particles_lost=int(round(float(tracing.total_particles_lost))),
        particles_unresolved=int(
            getattr(tracing, "total_particles_unresolved", 0) or 0),
        particles_failed=int(np.sum(~finite_mask & ~lost_mask)),
        wall_time_s=float(wall_time_s),
        particle_energy=particle_energy,
        total_speed=float(particles.total_speed),
        times=np.asarray(tracing.times, dtype=float),
        loss_fractions=loss_fractions,
        lost_times=lost_times,
        trajectories=trajectories,
        trajectories_xyz=np.asarray(tracing.trajectories_xyz, dtype=float),
        energies=energies,
    )
