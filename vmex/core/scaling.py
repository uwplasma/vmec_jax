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
    """Low-resolution input probe used to choose ARIES-CS factors."""

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
    """Return ``inp`` at magnetic-field factor ``B_scale`` and size factor ``R_scale``."""
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
    """Scale a MAKEGRID table consistently with :func:`scale_input`."""
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
    """Return the dimensional similarity transform of a converged WOUT.

    The pressure arrays and the ``am``/``am_aux_f`` coefficients that
    generated them scale together, so the profile the wout echoes evaluates
    to its own ``presf``; see ``_PRESSURE_AMPLITUDES`` for which entries of
    ``am`` carry the factor and why ``ac``/``ai`` do not move.
    """
    b, r = _scales(b_scale, r_scale)
    changes = {
        name: _times(getattr(data, name), b**b_power * r**r_power)
        for name, (b_power, r_power) in _WOUT_POWERS.items()
    }
    changes["am"] = _scale_pressure_coefficients(data.pmass_type, data.am, b**2)
    return replace(data, **changes)


def input_minor_radius(inp: VmecInput) -> float:
    """Compute the VMEC ``Aminor_p`` convention directly from the input boundary."""
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
    """Return positive ``(B_scale, R_scale)`` factors for ARIES-CS dimensions."""
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
    """Estimate ``b0`` and final minor radius without running the requested ladder."""
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
    """Choose ARIES-CS factors from a bounded converged input probe."""
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
