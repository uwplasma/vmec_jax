"""Portable converged LASYM free-boundary regression case.

The axisymmetric vacuum field is a degree-eight Chebyshev compression of the
21-coil-group ``mgrid_d3d_ef.nc`` field after applying the DIII-D currents in
``input.DIII-D_lasym_false``.  Enforcing the exact up/down field parity leaves
90 coefficients instead of a 12 MiB mgrid.  A small divergence-free
``B_R = 3e-4/R`` perturbation supplies a resolved asymmetric forcing rather
than testing termination-level decay of an arbitrary asymmetric seed.  The
helper tabulates the field to a standard one-group mgrid; both VMEX and
VMEC2000 consume the same generated table.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval2d

from vmex.core.input import VmecInput
from vmex.core.mgrid import MgridData, MgridField, tabulate_cartesian_field

_BR_ODD = np.asarray([
    [-0.15379965715396243, 0.005955879510258224, -0.01956813403640866,
     -0.028207093199394156, 0.06122651731779435, 0.02156859529397671,
     -0.05117367673694955, 0.03427456190248369, 0.02087734073879346],
    [0.0625684898363221, -0.10024922510139248, 0.06395074879720827,
     -0.003747162587323355, -0.00201536851552208, 0.03558282845809198,
     -0.05012011741968667, 0.025886094653241536, 0.008696790712432758],
    [0.036506402831943105, -0.037630295930214955, 0.02026938461805142,
     -0.0064743610415361865, -0.04120240535901665, 0.02802554539205232,
     -0.00224805187989305, 0.013198452128471476, 0.017712036131594066],
    [-0.03157390360011194, 0.04317346548700533, -0.050686295257507244,
     0.008567654279673043, -0.031413108695490605, 0.007151822603028063,
     0.03331282170195195, -0.001574473647923123, 0.02312119118245468],
])
_BP_EVEN = np.asarray([
    -2.0731947125253116, 1.3013678625670606, -0.40844749109871475,
    0.1281733836701595, -0.04022958749965087, 0.01259675481625877,
    -0.0039544949093219955, 0.0011934675643043222,
    -0.00037514582649129935,
])
_BZ_EVEN = np.asarray([
    [-0.1244496283876967, 0.08078169496909758, 0.09803118866701976,
     0.032619443492009514, -0.04527747610872698, 0.07340362087496946,
     -0.06903710523508322, 0.001018236463981248, -0.004011069331506127],
    [0.07003841282443365, -0.1214034192540085, 0.12041656211443061,
     -0.18661479441035686, 0.0022092654863521896, 0.016235204124678885,
     -0.028050131478305408, -0.02377631875700722, 0.027523977411596075],
    [-0.005665499794409737, 0.001543326940304009, -0.03304488368849512,
     0.02289547005403441, -0.02988872350276521, 0.09104402946627965,
     -0.010916060994154105, -0.0051642446794190515, 0.011237568464621206],
    [-0.009877866064926116, 0.024839828291842163, -0.03977074335443278,
     0.05411440056272447, -0.0544320435703102, 0.0496688981338937,
     -0.024589685459728394, 0.00030933180833038, 0.01198617885063813],
    [0.0016580288765149567, -0.010283927284954766, 0.004044515634916909,
     -0.038698389102091134, -0.006902497323189195,
     -0.053734517909433545, 0.0008850782338027559,
     -0.01826613394913359, 0.020040651912662882],
])


def lasym_free_input(data: Path) -> VmecInput:
    """DIII-D deck with a resolved nonzero asymmetric boundary seed."""
    base = VmecInput.from_file(data / "input.DIII-D_lasym_false")
    rbs, zbc = np.asarray(base.rbs).copy(), np.asarray(base.zbc).copy()
    rbs[0, 1], zbc[0, 1] = 1.0e-4, -1.0e-4
    return dataclasses.replace(
        base,
        lasym=True,
        rbs=rbs,
        zbc=zbc,
        extcur=np.ones(1),
        mgrid_file="mgrid_d3d_lasym_chebyshev.nc",
        ns_array=np.asarray([16, 32]),
        ftol_array=np.asarray([1.0e-8, 1.0e-10]),
        niter_array=np.asarray([2000, 6000]),
    )


def _cartesian_field(points):
    points = np.asarray(points)
    x, y, z = np.moveaxis(points, -1, 0)
    radius, phi = np.hypot(x, y), np.arctan2(y, x)
    r_normalized = radius - 1.75
    z_normalized = z / 1.6
    br_coefficients = np.zeros((9, 9))
    br_coefficients[1::2] = _BR_ODD
    bp_coefficients = np.zeros((9, 9))
    bp_coefficients[0] = _BP_EVEN
    bz_coefficients = np.zeros((9, 9))
    bz_coefficients[::2] = _BZ_EVEN
    br = chebval2d(z_normalized, r_normalized, br_coefficients)
    br += 3.0e-4 / radius
    bp = chebval2d(z_normalized, r_normalized, bp_coefficients)
    bz = chebval2d(z_normalized, r_normalized, bz_coefficients)
    return np.stack(
        (
            br * np.cos(phi) - bp * np.sin(phi),
            br * np.sin(phi) + bp * np.cos(phi),
            bz,
        ),
        axis=-1,
    )


def lasym_free_mgrid_data() -> MgridData:
    """Generate the shared compact one-group DIII-D regression mgrid."""
    return tabulate_cartesian_field(
        _cartesian_field,
        rmin=0.75,
        rmax=2.75,
        zmin=-1.6,
        zmax=1.6,
        ir=64,
        jz=81,
        kp=1,
        nfp=1,
        label="compressed_d3d",
    )


def lasym_free_field() -> MgridField:
    """Return the generated mgrid as a differentiable VMEX field."""
    return MgridField.from_mgrid_data(
        lasym_free_mgrid_data(), extcur=np.ones(1)
    )
