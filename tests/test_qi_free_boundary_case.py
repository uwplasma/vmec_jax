"""Generated quasi-isodynamic free-boundary stress case (coils + mgrid).

A portable, fully generated free-boundary case built around a QI boundary.  No
external mgrid, coil file, or equilibrium is needed: filamentary modular coils
are constructed analytically around the plasma, their Biot-Savart field is
tabulated onto a cylindrical grid with
:func:`~vmex.core.mgrid.tabulate_cartesian_field`, and the resulting table is
consumed exactly like a file-based mgrid.

Why this case exists
--------------------
Free-boundary failures reported against the solver combine several stresses at
once, and a case that isolates them individually cannot reproduce the coupling:

* a **finite pressure** profile, so the free-boundary equilibrium is genuinely
  different from the fixed-boundary one (a vacuum case would let the plasma sit
  on its prescribed boundary and hide vacuum/plasma feedback);
* a **QI** boundary, whose weak, strongly shaped fields make the Jacobian and
  the spectral constraint far more delicate than a tokamak-like case;
* a **coil field that does not exactly hold the prescribed boundary** — the
  coils are placed on a smooth winding surface offset from the LCFS rather than
  optimised against it, so the plasma must move to find its own equilibrium;
* a **radial ladder**, exercising the vacuum/NESTOR continuation across grid
  changes.

The coil geometry is deliberately simple and closed-form so the fixture is
cheap, reproducible and free of any external dependency.  It is a *stress*
fixture, not a physics benchmark: the assertions are about the solver behaving
lawfully under strain, not about the accuracy of the resulting equilibrium.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vmex.core.input import VmecInput
from vmex.core.mgrid import MgridData, MgridField, tabulate_cartesian_field

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

#: Boundary deck supplying the QI plasma shape.
QI_DECK = DATA / "input.nfp2_QI"

#: Modular-coil layout.  ``COILS_PER_PERIOD`` filaments per field period sit on
#: a circular winding surface of radius ``COIL_MINOR`` about the magnetic axis
#: circle of radius ``COIL_MAJOR``; the winding radius is comfortably outside
#: the QI LCFS so the tabulated field is smooth over the plasma volume.
COILS_PER_PERIOD = 6
COIL_MAJOR = 1.0
COIL_MINOR = 0.55
#: Filament discretisation for the Biot-Savart sum.
COIL_SEGMENTS = 96
#: Net current per filament [A]; scaled to give a field of order 1 T on axis.
COIL_CURRENT = 7.0e5

#: Cylindrical tabulation grid bracketing the plasma.
GRID = dict(rmin=0.45, rmax=1.55, zmin=-0.55, zmax=0.55, ir=48, jz=48, kp=24)

_MU0_OVER_4PI = 1.0e-7


def _coil_filaments(nfp: int) -> np.ndarray:
    """Closed modular filaments, shape ``(n_coils, COIL_SEGMENTS, 3)``.

    Planar circular loops of radius :data:`COIL_MINOR`, centred on the circle
    ``R = COIL_MAJOR`` and oriented normal to it, i.e. the simplest layout that
    produces a toroidal field with the correct field-period symmetry.  Planar
    coils cannot hold a QI plasma exactly -- that is the point: the plasma has
    to relax away from its prescribed boundary.
    """
    n_coils = COILS_PER_PERIOD * int(nfp)
    phi_centres = 2.0 * np.pi * (np.arange(n_coils) + 0.5) / n_coils
    theta = 2.0 * np.pi * np.arange(COIL_SEGMENTS) / COIL_SEGMENTS

    # Loop in the (R, Z) plane at each toroidal angle, rotated into Cartesian.
    r_loop = COIL_MAJOR + COIL_MINOR * np.cos(theta)          # (S,)
    z_loop = COIL_MINOR * np.sin(theta)                       # (S,)
    cos_p, sin_p = np.cos(phi_centres)[:, None], np.sin(phi_centres)[:, None]
    return np.stack(
        (r_loop[None, :] * cos_p, r_loop[None, :] * sin_p,
         np.broadcast_to(z_loop[None, :], (n_coils, COIL_SEGMENTS))),
        axis=-1,
    )


def _biot_savart(filaments: np.ndarray, current: float):
    """Return ``field(points) -> B`` for closed filaments carrying ``current``.

    Straight-segment Biot-Savart:  each segment contributes
    ``mu0/4pi * I * dl x r / |r|^3`` evaluated at the segment midpoint, which
    is second-order accurate in the segment length and ample for a tabulated
    external field that the solver only ever interpolates.
    """
    starts = filaments
    ends = np.roll(filaments, -1, axis=1)
    segments = (ends - starts).reshape(-1, 3)          # (N, 3)
    midpoints = (0.5 * (starts + ends)).reshape(-1, 3)  # (N, 3)

    def field(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 3)
        out = np.zeros_like(flat)
        # Chunk over evaluation points: the full outer product would be
        # (n_points x n_segments x 3) and is needlessly large for a fixture.
        for lo in range(0, flat.shape[0], 4096):
            chunk = flat[lo:lo + 4096]
            delta = chunk[:, None, :] - midpoints[None, :, :]     # (C, N, 3)
            dist = np.linalg.norm(delta, axis=-1)                 # (C, N)
            dist = np.where(dist < 1.0e-9, np.inf, dist)
            contrib = np.cross(segments[None, :, :], delta) / dist[..., None] ** 3
            out[lo:lo + 4096] = _MU0_OVER_4PI * current * contrib.sum(axis=1)
        return out.reshape(points.shape)

    return field


def qi_free_mgrid_data(nfp: int = 2) -> MgridData:
    """Tabulate the generated modular-coil field onto a cylindrical grid."""
    return tabulate_cartesian_field(
        _biot_savart(_coil_filaments(nfp), COIL_CURRENT),
        nfp=int(nfp), label="qi_modular_generated", **GRID,
    )


def qi_free_field(nfp: int = 2) -> MgridField:
    """The generated coil field as a VMEX free-boundary external field."""
    return MgridField.from_mgrid_data(qi_free_mgrid_data(nfp), extcur=np.ones(1))


def qi_free_input(
    *,
    ns_array=(9, 15),
    niter: int = 40,
    pressure_scale: float = 4.0e3,
) -> VmecInput:
    """QI boundary + finite pressure + free boundary, at stress resolution.

    ``pressure_scale`` sets a parabolic ``p(s) = PRES_SCALE (1 - s)`` profile.
    A non-zero pressure is what makes the free-boundary answer differ from the
    fixed-boundary one: the plasma expands against the coil field instead of
    resting on the prescribed boundary.
    """
    import dataclasses

    inp = VmecInput.from_file(str(QI_DECK))
    ns = [int(n) for n in ns_array]
    return dataclasses.replace(
        inp,
        lfreeb=True,
        mgrid_file="qi_modular(generated)",
        ns_array=ns,
        ftol_array=[1.0e-12] * len(ns),
        niter_array=[int(niter)] * len(ns),
        pmass_type="power_series",
        am=[1.0, -1.0] + [0.0] * 19,
        pres_scale=float(pressure_scale),
    )
