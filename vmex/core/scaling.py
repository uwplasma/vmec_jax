"""Dimensionally scale VMEC inputs, mgrid fields, and WOUT data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .input import VmecInput
from .mgrid import MgridData
from .wout import WoutData

ARIES_CS_B0 = 5.7
ARIES_CS_AMINOR = 1.7

_INPUT_LENGTHS = (
    "raxis_c", "zaxis_s", "raxis_s", "zaxis_c",
    "rbc", "zbs", "rbs", "zbc",
)

_WOUT_POWERS = {
    # Scalars and one-dimensional profiles.
    **{name: (2, 3) for name in ("wb", "wp")},
    **{name: (0, 1) for name in (
        "rmax_surf", "rmin_surf", "zmax_surf", "Aminor_p", "Rmajor_p",
        "raxis_cc", "zaxis_cs", "raxis_cs", "zaxis_cc",
        "rmnc", "zmns", "rmns", "zmnc",
    )},
    **{name: (1, 0) for name in ("b0", "volavgB", "bmnc", "bmns")},
    **{name: (1, 1) for name in (
        "rbtor0", "rbtor", "ctor", "extcur", "buco", "bvco", "jcuru",
        "jcurv", "bsubumnc", "bsubvmnc", "bsubsmns", "bsubumns",
        "bsubvmns", "bsubsmnc", "currumnc", "currvmnc", "currumns",
        "currvmns", "potsin", "potcos", "bsubumnc_sur", "bsubvmnc_sur",
        "bsubumns_sur", "bsubvmns_sur",
    )},
    **{name: (1, 2) for name in (
        "phi", "phipf", "chi", "chipf", "phips",
    )},
    **{name: (2, 0) for name in (
        "presf", "mass", "pres", "bdotb", "am_aux_f",
    )},
    **{name: (0, 3) for name in ("volume_p", "vp", "gmnc", "gmns")},
    **{name: (1, -1) for name in (
        "bdotgradv", "bsupumnc", "bsupvmnc", "bsupumns", "bsupvmns",
        "bsupumnc_sur", "bsupvmnc_sur", "bsupumns_sur", "bsupvmns_sur",
    )},
    "jdotb": (2, -1),
    "IonLarmor": (-1, 0),
    "over_r": (0, -1),
    **{name: (-2, -4) for name in (
        "DMerc", "DShear", "DWell", "DCurr", "DGeod",
    )},
}

# Pressure-profile coefficients.  ``presf``/``pres``/``mass`` scale as B**2,
# and the parameterization that produced them has to follow, or the wout's
# namelist echo re-runs to the unscaled pressure.  A wout records no
# PRES_SCALE, so the factor lands in the coefficients themselves, and only
# in the entries that multiply the profile: widths, exponents, centres, and
# mixing fractions are shape data.  Which entries are amplitudes depends on
# ``pmass_type`` (``vmex.core.profiles``): the power series and the
# pedestal polynomial are linear in every coefficient, the peaked shapes
# carry one amplitude ahead of their shape parameters, ``rational`` scales
# through its numerator, and the tabulated kinds ignore ``am`` and keep the
# pressure in ``am_aux_f``, which is linear in the knot values for every
# tabulated kind (and zero-filled otherwise), so it sits in the table above.
#
# ``ai``/``ai_aux_f`` are deliberately left alone: iota (or q under lrfp) is
# dimensionless.  ``ac``/``ac_aux_f`` are deliberately left alone for every
# ``pcurr_type``, including the ``*_i``/``*_ip`` spline kinds whose knots
# tabulate I or I': ``profil1d.f`` (``setup.flux_profiles``) divides the
# shape by its own edge value and multiplies by CURTOR,
# ``Itor = signgs*mu0*curtor/(2*pi*pcurr(1))``, and drops the profile
# entirely when ``pcurr(1)`` vanishes, so the coefficients never carry an
# ampere.  The current amplitude lives in ``ctor``, scaled as B*R above.
_PRESSURE_AMPLITUDES: dict[str, tuple[int, ...]] = {
    "power_series": tuple(range(21)),
    # c[0:16] polynomial, c[17] pedestal height; c[16] unused, c[18] centre,
    # c[19] width, c[20] the normalization VMEC recomputes.
    "pedestal": tuple(range(16)) + (17,),
    "two_power": (0,),
    "two_power_gs": (0,),
    "two_lorentz": (0,),
    "gauss_trunc": (0,),
    "rational": tuple(range(10)),   # numerator c[0:10] over denominator c[10:21]
    "cubic_spline": (),
    "akima_spline": (),
    "line_segment": (),
}


@dataclass(frozen=True)
class ScaleProbe:
    """Outcome of the bounded two-resolution solve behind :func:`probe_input`.

    A fixed-boundary or free-boundary deck does not carry ``b0``; only a
    converged equilibrium does.  :func:`probe_input` therefore solves the deck
    twice at reduced radial resolution and reports the fine-grid values
    together with the coarse-to-fine change, which is the honest uncertainty
    of the factors derived from them.

    Attributes
    ----------
    b0:
        Field on the magnetic axis of the fine probe, ``wout.b0`` in T
        (signed: its sign follows the flux direction of the deck).
    aminor:
        VMEC ``Aminor_p`` of the fine probe in m.  For a fixed-boundary deck
        :func:`aries_cs_input_scales` replaces it with the exact boundary
        quadrature of :func:`input_minor_radius`.
    b0_relative_change:
        ``|b0_coarse - b0_fine| / |b0_fine|`` — dimensionless.
    aminor_relative_change:
        The same relative change for ``Aminor_p``, and ``0.0`` when the two
        probe resolutions coincide.
    coarse_ns:
        Radial resolution of the first probe solve (``min(ns_final, 9)``).
    fine_ns:
        Radial resolution of the second probe solve (``min(ns_final, 17)``).
    """

    b0: float
    aminor: float
    b0_relative_change: float
    aminor_relative_change: float
    coarse_ns: int
    fine_ns: int


def _scales(b_scale: float, r_scale: float) -> tuple[float, float]:
    b, r = float(b_scale), float(r_scale)
    if not np.isfinite((b, r)).all() or b <= 0.0 or r <= 0.0:
        raise ValueError("B_scale and R_scale must be finite and positive")
    return b, r


def _times(value: Any, factor: float) -> Any:
    if value is None:
        return None
    scaled = np.asarray(value) * factor
    return float(scaled) if scaled.ndim == 0 else scaled


def scale_input(
    inp: VmecInput, *, b_scale: float = 1.0, r_scale: float = 1.0,
) -> VmecInput:
    """Apply the ideal-MHD similarity transform to a parsed VMEC input deck.

    Ideal MHD is invariant under an independent rescaling of every length by
    ``r_scale`` and every magnetic field by ``b_scale``, so the returned deck
    converges to the same *shape* of equilibrium at new dimensions.  The
    dimensional entries of ``&INDATA`` transform as

    - boundary and axis Fourier coefficients ``rbc``, ``zbs``, ``rbs``,
      ``zbc``, ``raxis_c``, ``zaxis_s``, ``raxis_s``, ``zaxis_c`` (m):
      ``r_scale``;
    - ``phiedge``, the total enclosed toroidal flux (Wb): ``b_scale *
      r_scale**2``;
    - ``pres_scale``, the multiplier VMEC applies on top of the ``am`` /
      ``am_aux_f`` pressure profile: ``b_scale**2``, so pressure in Pa scales
      as ``b_scale**2`` while the profile *shape* coefficients stay as
      written;
    - ``extcur``, the external coil-group currents (A): ``b_scale * r_scale``;
    - ``curtor``, the net toroidal current (A): ``b_scale * r_scale``, but
      **only when** ``ncurr == 1``.  With ``ncurr == 0`` VMEC prescribes iota
      and never reads ``curtor``, so the stored value is passed through
      untouched.

    Everything else is carried through unchanged: ``nfp``, ``mpol``/``ntor``,
    ``ns_array`` / ``ftol_array`` / ``niter_array``, the ``am``/``ai``/``ac``
    profile coefficients, every normalized spline abscissa, and all flags.
    Rotational transform, beta, aspect ratio, and the mode numbers are
    dimensionless and hence invariant by construction.

    Parameters
    ----------
    inp:
        Deck to transform.  It is not mutated; a new
        :class:`~vmex.core.input.VmecInput` is returned.
    b_scale:
        Magnetic-field factor, finite and strictly positive.  Positivity is
        what preserves the sign of the flux, hence the field direction.
    r_scale:
        Length factor, finite and strictly positive.

    Returns
    -------
    A copy of ``inp`` with the dimensional entries above rescaled.  Free
    boundary decks additionally need :func:`scale_mgrid` applied to their
    mgrid sidecar, otherwise ``extcur`` and the tabulated vacuum field no
    longer describe the same coil set.
    """
    b, r = _scales(b_scale, r_scale)
    changes = {name: _times(getattr(inp, name), r) for name in _INPUT_LENGTHS}
    changes.update(
        phiedge=inp.phiedge * b * r**2,
        pres_scale=inp.pres_scale * b**2,
        curtor=inp.curtor * b * r if inp.ncurr == 1 else inp.curtor,
        extcur=_times(inp.extcur, b * r),
    )
    return replace(inp, **changes)


def scale_mgrid(
    data: MgridData, *, b_scale: float = 1.0, r_scale: float = 1.0,
) -> MgridData:
    """Scale a MAKEGRID vacuum-field table consistently with :func:`scale_input`.

    The cylindrical grid is scaled self-similarly and the tabulated field is
    rescaled so that the pair (deck, mgrid) still describes one coil set:

    - ``rmin``, ``rmax``, ``zmin``, ``zmax`` (m): ``r_scale``.  The grid
      *dimensions* ``ir``, ``jz``, ``kp`` and the sample count are unchanged —
      the same samples simply sit at scaled cylindrical coordinates, so no
      field is re-tabulated or interpolated.
    - ``raw_coil_cur``, the currents the file was computed with (A):
      ``b_scale * r_scale``, matching the ``extcur`` factor of
      :func:`scale_input`.
    - ``br``, ``bp``, ``bz``, each of shape ``(nextcur, kp, jz, ir)``: the
      factor depends on ``mgrid_mode``.  A per-ampere table
      (``mgrid_mode`` starting with ``"S"``, units T/A) scales as
      ``1 / r_scale``, because the field it will be multiplied by scales as
      ``b_scale`` while the current scales as ``b_scale * r_scale``.  A raw
      table (``"R"`` or ``"N"``, units T, coil currents already baked in)
      scales as ``b_scale``; VMEC then divides ``extcur`` by
      ``raw_coil_cur``, and both were scaled by the same
      ``b_scale * r_scale``, so the physical field again ends up at
      ``b_scale``.

    ``nfp``, ``nextcur``, ``mgrid_mode``, and ``coil_groups`` are metadata and
    pass through unchanged.

    Parameters
    ----------
    data:
        Snapshot of an mgrid netCDF file.  It is not mutated; a new
        :class:`~vmex.core.mgrid.MgridData` is returned.
    b_scale, r_scale:
        The same finite, strictly positive factors given to
        :func:`scale_input`; using different ones here silently produces a
        deck and a vacuum field that disagree.

    Returns
    -------
    A copy of ``data`` with the extents, recorded currents, and field tables
    rescaled.
    """
    b, r = _scales(b_scale, r_scale)
    raw = np.asarray(data.raw_coil_cur) * b * r
    field_scale = 1.0 / r if data.mgrid_mode.upper().startswith("S") else b
    return replace(
        data,
        rmin=data.rmin * r,
        rmax=data.rmax * r,
        zmin=data.zmin * r,
        zmax=data.zmax * r,
        raw_coil_cur=tuple(raw),
        br=np.asarray(data.br) * field_scale,
        bp=np.asarray(data.bp) * field_scale,
        bz=np.asarray(data.bz) * field_scale,
    )


def _scale_pressure_coefficients(pmass_type: str, am: Any, factor: float) -> Any:
    """Scale the amplitude entries of ``am`` for ``pmass_type`` by ``factor``."""
    if am is None:
        return None
    kind = str(pmass_type).strip().lower()
    if kind not in _PRESSURE_AMPLITUDES:
        raise NotImplementedError(
            f"pmass_type={pmass_type!r} not implemented "
            f"(supported: {sorted(_PRESSURE_AMPLITUDES)})"
        )
    scaled = np.array(am, dtype=float).reshape(-1)
    slots = [i for i in _PRESSURE_AMPLITUDES[kind] if i < scaled.size]
    scaled[slots] *= factor
    return scaled.reshape(np.shape(am))


def scale_wout(
    data: WoutData, *, b_scale: float = 1.0, r_scale: float = 1.0,
) -> WoutData:
    """Apply the ideal-MHD similarity transform to a converged WOUT in memory.

    Each field named in the module's power table is multiplied by
    ``b_scale**p * r_scale**q``; every field *not* named is passed through
    untouched.  Applying this to the wout of the original deck must give the
    same numbers as re-converging the deck scaled by :func:`scale_input` —
    that commutation is the validation contract of the transform.

    Scaled quantities, by power of ``(b_scale, r_scale)``:

    - ``(0, 1)`` lengths in m — ``Aminor_p``, ``Rmajor_p``, ``rmin_surf``,
      ``rmax_surf``, ``zmax_surf``, the axis arrays
      ``raxis_cc``/``zaxis_cs``/``raxis_cs``/``zaxis_cc``, and the geometry
      harmonics ``rmnc``/``zmns``/``rmns``/``zmnc``;
    - ``(1, 0)`` fields in T — ``b0``, ``volavgB``, and the ``|B|`` harmonics
      ``bmnc``/``bmns`` (half mesh, Nyquist mode set);
    - ``(1, 1)`` covariant field and currents — ``buco``, ``bvco``,
      ``rbtor``, ``rbtor0`` (T m); ``ctor``, ``extcur``, ``jcuru``, ``jcurv``
      (A); the ``bsubumnc``/``bsubvmnc``/``bsubsmns`` families with their
      ``lasym`` partners and ``_sur`` surface variants; the ``currumnc`` /
      ``currvmnc`` families; and the NESTOR potential ``potsin``/``potcos``;
    - ``(1, 2)`` fluxes in Wb — ``phi``, ``chi`` and their radial derivatives
      ``phipf``, ``chipf``, ``phips``;
    - ``(2, 0)`` pressure-like quantities — ``presf``, ``pres`` (Pa),
      ``mass``, and ``bdotb`` (T^2);
    - ``(0, 3)`` volumes in m^3 — ``volume_p``, ``vp``, and the Jacobian
      harmonics ``gmnc``/``gmns``;
    - ``(2, 3)`` energies — ``wb``, ``wp``;
    - ``(1, -1)`` contravariant field — ``bdotgradv`` and the
      ``bsupumnc``/``bsupvmnc`` families with their ``_sur`` variants;
    - ``(2, -1)`` ``jdotb``, the flux-surface-averaged ``<J.B>``;
    - ``(-1, 0)`` ``IonLarmor``: at fixed particle energy a Larmor radius goes
      as ``1 / B`` and does not care about machine size — the reason this
      transform is used for 3.5 MeV alpha studies;
    - ``(0, -1)`` ``over_r``, the surface-averaged ``<1/R>``;
    - ``(-2, -4)`` the Mercier terms ``DMerc``, ``DShear``, ``DWell``,
      ``DCurr``, ``DGeod``.

    Untouched, and therefore invariant: the dimensionless physics
    (``iotaf``, ``iotas``, ``q_factor``, ``aspect``, ``betatotal`` and the
    other betas, ``beta_vol``, ``specw``, ``equif``, the ``lmns``/``lmnc``
    lambda harmonics, which are angles), the mode tables ``xm``/``xn`` and
    their Nyquist partners, the convergence record (``fsqr``, ``fsqz``,
    ``fsql``, ``fsqt``, ``ftolv``, ``niter``), and every integer or string
    of metadata.

    The pressure arrays and the ``am``/``am_aux_f`` coefficients that
    generated them scale together, so the profile the wout echoes evaluates
    to its own ``presf``; see ``_PRESSURE_AMPLITUDES`` for which entries of
    ``am`` carry the factor and why ``ac``/``ai`` do not move.

    Parameters
    ----------
    data:
        Converged equilibrium to transform.  It is not mutated; a new
        :class:`~vmex.core.wout.WoutData` is returned.  Fields that are
        ``None`` (an absent ``lasym`` partner, an absent NESTOR block) stay
        ``None``.
    b_scale:
        Magnetic-field factor, finite and strictly positive.
    r_scale:
        Length factor, finite and strictly positive.

    Returns
    -------
    A copy of ``data`` at the new dimensions.  Scalars come back as Python
    ``float``; arrays come back as ``numpy`` arrays of the same shape.
    """
    b, r = _scales(b_scale, r_scale)
    changes = {
        name: _times(getattr(data, name), b**b_power * r**r_power)
        for name, (b_power, r_power) in _WOUT_POWERS.items()
    }
    changes["am"] = _scale_pressure_coefficients(data.pmass_type, data.am, b**2)
    return replace(data, **changes)


def input_minor_radius(inp: VmecInput) -> float:
    """Compute VMEC's ``Aminor_p`` from an input boundary, without solving.

    ``Aminor_p`` is ``sqrt(A / pi)`` with ``A`` the cross-sectional area of the
    boundary in the ``(R, Z)`` plane, averaged over one field period.  The
    area is evaluated by the exact trapezoidal quadrature of
    ``A = |<R dZ/dtheta>|`` on a uniform ``(zeta, theta)`` grid: ``theta``
    spans ``[0, 2*pi)`` and ``zeta`` spans ``[0, 2*pi/nfp)``, both in radians,
    with at least ``2*mpol + 2`` and ``2*ntor + 1`` points respectively (and
    at least the deck's own ``ntheta``/``nzeta``), so the boundary harmonics
    are resolved without aliasing.

    Only the boundary tables ``rbc``, ``rbs``, ``zbs``, ``zbc`` are read; the
    axis, the profiles, and the free-boundary settings are irrelevant here.
    For a fixed-boundary deck this is exact, which is why
    :func:`aries_cs_input_scales` prefers it over the probe estimate.  It is
    *not* the right number for a free-boundary deck, whose plasma boundary is
    an output of the solve rather than the input boundary.

    Parameters
    ----------
    inp:
        Deck whose boundary is measured.

    Returns
    -------
    The minor radius in meters.
    """
    ntheta = max(int(inp.ntheta) or 0, 2 * int(inp.mpol) + 2, 16)
    nzeta = max(int(inp.nzeta) or 0, 2 * int(inp.ntor) + 1, 1)
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2.0 * np.pi * np.arange(nzeta) / (nzeta * inp.nfp)
    n = np.arange(-inp.ntor, inp.ntor + 1)
    m = np.arange(inp.mpol)
    angle = (
        m[None, None, None, :] * theta[None, :, None, None]
        - n[None, None, :, None] * inp.nfp * zeta[:, None, None, None]
    )
    r = np.sum(
        inp.rbc[None, None, :, :] * np.cos(angle)
        + inp.rbs[None, None, :, :] * np.sin(angle),
        axis=(-2, -1),
    )
    zu = np.sum(
        m[None, None, None, :] * (
            inp.zbs[None, None, :, :] * np.cos(angle)
            - inp.zbc[None, None, :, :] * np.sin(angle)
        ),
        axis=(-2, -1),
    )
    return float(np.sqrt(2.0 * abs(np.mean(r * zu))))


def aries_cs_scales(data: WoutData) -> tuple[float, float]:
    """Factors that take a converged WOUT to ARIES-CS reference magnitudes.

    The reference magnitudes are the module constants ``ARIES_CS_B0 = 5.7``
    (axis field, T) and ``ARIES_CS_AMINOR = 1.7`` (minor radius, m).  A wout
    stores both ``b0`` and ``Aminor_p``, so the factors are exact — no probe
    solve is needed, unlike :func:`aries_cs_input_scales`.

    ``b_scale`` uses ``abs(b0)``, so the factor is always positive and
    :func:`scale_wout` never flips the flux direction: a configuration with
    negative ``b0`` keeps it.

    Parameters
    ----------
    data:
        Converged equilibrium to measure.

    Returns
    -------
    ``(b_scale, r_scale) = (5.7 / abs(b0), 1.7 / Aminor_p)``, ready to pass to
    :func:`scale_wout` (or, with the matching deck, to :func:`scale_input`).

    Raises
    ------
    ValueError
        If ``b0`` is zero or ``Aminor_p`` is not positive.
    """
    if data.b0 == 0.0 or data.Aminor_p <= 0.0:
        raise ValueError("ARIES-CS scaling requires nonzero b0 and positive Aminor_p")
    return ARIES_CS_B0 / abs(data.b0), ARIES_CS_AMINOR / data.Aminor_p


def probe_input(
    inp: VmecInput,
    *,
    mgrid_path: str | Path | None = None,
    external_field: Any = None,
    device: Any = "auto",
) -> ScaleProbe:
    """Estimate ``b0`` and ``Aminor_p`` from a bounded two-resolution solve.

    An input deck has no axis field: ``b0`` only exists once the equilibrium
    is converged.  Rather than run the deck's own ``ns_array`` ladder, this
    solves it twice at reduced radial resolution — ``ns = min(ns_final, 9)``
    at ``ftol = 1e-8``, then ``ns = min(ns_final, 17)`` at ``ftol = 1e-10``,
    the second warm-started by radial interpolation of the first
    (:func:`vmex.core.multigrid.interpolate_state`) unless the two
    resolutions coincide.  Both runs use
    ``niter = max(3000, max(inp.niter_array))``, disable ``lfull3d1out``, and
    set ``raise_on_max_iterations=True``, so a probe that fails to converge
    raises instead of returning a meaningless magnitude.

    Free-boundary decks (``inp.lfreeb``) go through
    :func:`~vmex.core.multigrid.solve_free_boundary_multigrid` with the
    external field below; fixed-boundary decks go through
    :func:`~vmex.core.multigrid.solve_multigrid`, which ignores
    ``mgrid_path`` and ``external_field`` entirely.

    Parameters
    ----------
    inp:
        Deck to probe.  It is not mutated; each probe run is a copy with
        reduced resolution.
    mgrid_path:
        Path to the ``mgrid_*.nc`` vacuum-field table, for free-boundary
        decks only.
    external_field:
        In-memory alternative to ``mgrid_path`` (an
        :class:`~vmex.core.mgrid.MgridField` or equivalent), for
        free-boundary decks only.
    device:
        Device policy forwarded to the solver; ``"auto"`` keeps VMEX's
        default placement.

    Returns
    -------
    A :class:`ScaleProbe` holding the fine-grid ``b0`` (T) and ``Aminor_p``
    (m), the coarse-to-fine relative change of each, and the two radial
    resolutions used.
    """
    from .fourier import mode_table
    from .multigrid import (
        interpolate_state,
        solve_free_boundary_multigrid,
        solve_multigrid,
    )
    from .wout import wout_from_state

    final_ns = int(np.max(np.asarray(inp.ns_array)))
    coarse_ns, fine_ns = min(final_ns, 9), min(final_ns, 17)
    niter = max(3000, int(np.max(np.asarray(inp.niter_array))))

    def run(ns: int, ftol: float, initial_state=None):
        deck = replace(
            inp,
            ns_array=np.asarray([ns]),
            ftol_array=np.asarray([ftol]),
            niter_array=np.asarray([niter]),
            lfull3d1out=False,
        )
        common = dict(
            initial_state=initial_state,
            verbose=False,
            device=device,
            raise_on_max_iterations=True,
            prefetch_compile=False,
        )
        if inp.lfreeb:
            result = solve_free_boundary_multigrid(
                deck,
                mgrid_path=mgrid_path,
                external_field=external_field,
                **common,
            )
        else:
            result = solve_multigrid(deck, mode="cli", **common)
        wout = wout_from_state(
            inp=deck,
            state=result.state,
            fsqr=float(result.fsqr),
            fsqz=float(result.fsqz),
            fsql=float(result.fsql),
            niter=int(result.iterations),
            converged=bool(result.converged),
            vacuum_output=result.vacuum,
        )
        return result.state, wout

    coarse_state, coarse = run(coarse_ns, 1e-8)
    if fine_ns == coarse_ns:
        initial = coarse_state
    else:
        initial = interpolate_state(
            coarse_state,
            ns_fine=fine_ns,
            modes=mode_table(inp.mpol, inp.ntor),
        )
    _, fine = run(fine_ns, 1e-10, initial)

    relative = lambda x, y: abs(x - y) / max(abs(y), np.finfo(float).tiny)  # noqa: E731
    return ScaleProbe(
        b0=float(fine.b0),
        aminor=float(fine.Aminor_p),
        b0_relative_change=relative(float(coarse.b0), float(fine.b0)),
        aminor_relative_change=relative(
            float(coarse.Aminor_p), float(fine.Aminor_p)
        ),
        coarse_ns=coarse_ns,
        fine_ns=fine_ns,
    )


def aries_cs_input_scales(
    inp: VmecInput,
    **probe_kwargs: Any,
) -> tuple[float, float, ScaleProbe]:
    """ARIES-CS factors for an input deck, via a bounded converged probe.

    The deck itself carries no ``b0``, so :func:`probe_input` supplies it.
    For a fixed-boundary deck the probe's ``Aminor_p`` is then *discarded* and
    replaced by the exact boundary quadrature of :func:`input_minor_radius`
    (its ``aminor_relative_change`` is set to ``0.0`` accordingly); for a
    free-boundary deck the plasma boundary is an output of the solve, so the
    probe value is kept and its coarse-to-fine change is the honest
    uncertainty of ``r_scale``.

    Parameters
    ----------
    inp:
        Deck to measure.
    **probe_kwargs:
        Forwarded verbatim to :func:`probe_input` (``mgrid_path``,
        ``external_field``, ``device``).

    Returns
    -------
    ``(b_scale, r_scale, probe)`` where the factors target
    ``ARIES_CS_B0 = 5.7`` T and ``ARIES_CS_AMINOR = 1.7`` m, and ``probe`` is
    the :class:`ScaleProbe` the factors were derived from — report its
    relative changes alongside any scaled result.

    Raises
    ------
    ValueError
        If the measured ``b0`` is zero or the minor radius is not positive.
    """
    probe = probe_input(inp, **probe_kwargs)
    if not inp.lfreeb:
        probe = replace(
            probe,
            aminor=input_minor_radius(inp),
            aminor_relative_change=0.0,
        )
    if probe.b0 == 0.0 or probe.aminor <= 0.0:
        raise ValueError("ARIES-CS scaling requires nonzero b0 and positive Aminor_p")
    return (
        ARIES_CS_B0 / abs(probe.b0),
        ARIES_CS_AMINOR / probe.aminor,
        probe,
    )
