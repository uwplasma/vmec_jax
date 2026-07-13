"""Differentiable free-boundary residual via virtual casing (plan.md R15.3 + R19).

This module adds a *differentiable* free-boundary path that complements — and
does not touch — the NESTOR forward solve in :mod:`vmec_jax.core.freeboundary`
(R15.1/R15.2, VMEC2000-parity).  The idea (DESC R17.8): instead of
differentiating the NESTOR fixed point, express the free-boundary condition as a
smooth objective.  At the plasma-vacuum interface the total field is tangent,

    ``B_out . n = 0``,

and (finite beta) pressure balance holds,

    ``|B_in|^2 + 2 mu0 p = |B_out|^2``,

where ``B_out = B_coil + B_plasma`` is the external coil field plus the plasma's
*own* field, obtained through the **virtual-casing principle**.  Coil / ``extcur``
dofs -> ``B_ext`` -> boundary residual -> gradients, all in JAX, with no
NESTOR-adjoint.

Reuse, not re-implementation
----------------------------
The virtual-casing math is reused verbatim from ``uwplasma/virtual_casing_jax``
(:class:`~virtual_casing_jax.VirtualCasingExteriorField`,
:class:`~virtual_casing_jax.VmecSurfaceFieldData`, and the accurate on-surface
singular-quadrature integral ``VirtualCasingJAX.compute_internal_B``).  This
module only (a) adapts a converged/trial ``vmec_jax`` boundary + total field into
the package's :class:`VmecSurfaceFieldData` and (b) wires the resulting plasma
field to an :class:`~vmec_jax.core.mgrid.MgridField` or a plain ``xyz->B``
callable (e.g. an ESSOS ``essos.coils.Coils`` Biot-Savart field) to form the
differentiable residual.

Key structural fact that makes this cheap and well-posed: for a **fixed trial
boundary** the plasma's own field on that boundary does not depend on the coil
dofs, so it is precomputed **once** via the accurate on-surface virtual-casing
integral and frozen as a constant.  The residual is then a smooth JAX function of
the external-field dofs alone — an ``MgridField``'s ``extcur``, or the dofs of a
callable coil field (e.g. an ESSOS ``Coils``' Fourier dofs / currents) — and
FD-validates to ~1e-9 (see ``tests/test_freeboundary_diff.py``).

The full single-stage piece — letting the boundary *shape* dofs vary, so the
plasma field itself depends on them through a re-solve — is now supported:
:func:`surface_field_data_from_state` rebuilds the virtual-casing surface field
traceably from a live equilibrium state, so ``jax.grad`` threads through the
implicit adjoint (boundary) and virtual casing (coils) at once.  See
``examples/single_stage_simultaneous_opt.py`` and the *True single-stage*
section of ``docs/optimization.rst``.

``virtual_casing_jax`` is an optional dependency (``pip install vmec-jax[freeb]``
or ``pip install -e /path/to/virtual_casing_jax``).  Importing this module raises
a clear error if it is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from .mgrid import MgridField

try:  # optional dependency (uwplasma/virtual_casing_jax)
    from virtual_casing_jax import (
        ExteriorFieldConfig,
        VirtualCasingExteriorField,
        VmecSurfaceFieldData,
    )

    _HAVE_VCJ = True
except Exception as _exc:  # pragma: no cover - exercised only when the dep is absent
    _HAVE_VCJ = False
    _IMPORT_ERROR = _exc


#: Vacuum permeability [T m / A] (VMEC2000 ``mu0`` convention).
MU0 = 4.0e-7 * np.pi

__all__ = [
    "MU0",
    "surface_field_data_from_wout",
    "surface_field_data_from_state",
    "plasma_field_on_boundary",
    "FreeBoundaryDiffProblem",
    "external_B_cartesian",
    "have_virtual_casing_jax",
]


def have_virtual_casing_jax() -> bool:
    """Return ``True`` iff ``virtual_casing_jax`` is importable."""

    return _HAVE_VCJ


def _require_vcj() -> None:
    if not _HAVE_VCJ:  # pragma: no cover - dependency-guard branch
        raise ImportError(
            "vmec_jax.core.freeboundary_diff requires the optional dependency "
            "'virtual_casing_jax' (uwplasma). Install it with "
            "`pip install vmec-jax[freeb]` or `pip install -e /path/to/virtual_casing_jax`."
        ) from _IMPORT_ERROR


# ---------------------------------------------------------------------------
# Boundary synthesis: converged/trial wout -> VmecSurfaceFieldData
# ---------------------------------------------------------------------------


def _mode_series(cmn, smn, xm, xn, theta, phi):
    """Value + (d/dtheta, d/dphi) of ``sum cmn cos(a) + smn sin(a)``, ``a = xm t - xn p``.

    Returns arrays of shape ``(nphi, ntheta)`` (SoA-friendly toroidal-major layout).
    Either ``cmn`` or ``smn`` may be ``None`` when that parity is absent
    (stellarator symmetry: ``R`` has no sine partner, ``Z``/``lambda`` no cosine).
    """

    if cmn is None and smn is None:
        raise ValueError("_mode_series needs at least one of cmn/smn")
    xm = jnp.asarray(xm)
    xn = jnp.asarray(xn)
    ang = theta[None, :, None] * xm[None, None, :] - phi[:, None, None] * xn[None, None, :]
    cos = jnp.cos(ang)
    sin = jnp.sin(ang)
    npt = (int(phi.shape[0]), int(theta.shape[0]))
    val = jnp.zeros(npt)
    d_t = jnp.zeros(npt)
    d_p = jnp.zeros(npt)
    if cmn is not None:
        cmn = jnp.asarray(cmn)
        val = val + jnp.einsum("ptm,m->pt", cos, cmn)
        d_t = d_t + jnp.einsum("ptm,m->pt", sin, -cmn * xm)
        d_p = d_p + jnp.einsum("ptm,m->pt", sin, cmn * xn)
    if smn is not None:
        smn = jnp.asarray(smn)
        val = val + jnp.einsum("ptm,m->pt", sin, smn)
        d_t = d_t + jnp.einsum("ptm,m->pt", cos, smn * xm)
        d_p = d_p + jnp.einsum("ptm,m->pt", cos, -smn * xn)
    return val, d_t, d_p


def _nyquist_synth(cmn, smn, xm, xn, theta, phi):
    """Value of a Nyquist (``xm_nyq``/``xn_nyq``) series, shape ``(nphi, ntheta)``."""

    xm = jnp.asarray(xm)
    xn = jnp.asarray(xn)
    ang = theta[None, :, None] * xm[None, None, :] - phi[:, None, None] * xn[None, None, :]
    val = jnp.einsum("ptm,m->pt", jnp.cos(ang), jnp.asarray(cmn))
    if smn is not None:
        val = val + jnp.einsum("ptm,m->pt", jnp.sin(ang), jnp.asarray(smn))
    return val


def _get(wout, name):
    v = getattr(wout, name, None)
    return None if v is None else jnp.asarray(np.asarray(v, dtype=float))


def surface_field_data_from_wout(
    wout,
    *,
    nphi: int = 32,
    ntheta: int = 32,
    s_index: int = -1,
    use_stellsym: bool = True,
) -> "VmecSurfaceFieldData":
    """Build a :class:`~virtual_casing_jax.VmecSurfaceFieldData` from a wout.

    The wout may be a converged :class:`~vmec_jax.core.wout.WoutData` (from
    :func:`~vmec_jax.core.wout.read_wout` or
    :func:`~vmec_jax.core.wout.wout_from_state` — i.e. a *trial* boundary too).
    The boundary (last full-mesh surface, ``s_index=-1``) is synthesised on a
    single field period ``theta in [0, 2pi)``, ``phi in [0, 2pi/nfp)``:

    - ``gamma`` = ``(R cos phi, R sin phi, Z)``,
    - ``B_total`` = ``B^theta e_theta + B^phi e_phi`` (Cartesian), with the
      half-mesh ``bsup{u,v}`` edge-extrapolated ``1.5 x[-1] - 0.5 x[-2]``,
    - ``normal`` / ``area_vector`` = ``e_theta x e_phi`` (oriented outward),

    all in the package's structure-of-arrays layout ``(3, nphi, ntheta)``.

    The construction is validated by ``|B_total . n| / |B|`` ~ 1e-16 on a
    converged equilibrium (the VMEC free-boundary condition), see the module test.
    """

    _require_vcj()
    nfp = int(wout.nfp)
    lasym = bool(getattr(wout, "lasym", False))
    ns = int(wout.ns)
    j = int(s_index % ns)

    xm = _get(wout, "xm")
    xn = _get(wout, "xn")
    xmn = _get(wout, "xm_nyq")
    xnn = _get(wout, "xn_nyq")
    rmnc = _get(wout, "rmnc")
    zmns = _get(wout, "zmns")
    rmns = _get(wout, "rmns") if lasym else None
    zmnc = _get(wout, "zmnc") if lasym else None
    bsupu = _get(wout, "bsupumnc")
    bsupv = _get(wout, "bsupvmnc")
    bsupu_s = _get(wout, "bsupumns") if lasym else None
    bsupv_s = _get(wout, "bsupvmns") if lasym else None

    return _assemble_surface_field_data(
        nfp=nfp, ns=ns, j=j, xm=xm, xn=xn, xmn=xmn, xnn=xnn,
        rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc,
        bsupu=bsupu, bsupv=bsupv, bsupu_s=bsupu_s, bsupv_s=bsupv_s,
        lasym=lasym, use_stellsym=use_stellsym,
        signgs=int(getattr(wout, "signgs", -1)),
        nphi=nphi, ntheta=ntheta, source_convention="vmec_jax_wout",
    )


def _assemble_surface_field_data(
    *, nfp, ns, j, xm, xn, xmn, xnn, rmnc, zmns, rmns, zmnc,
    bsupu, bsupv, bsupu_s, bsupv_s, lasym, use_stellsym, signgs,
    nphi, ntheta, source_convention,
) -> "VmecSurfaceFieldData":
    """Assemble a :class:`VmecSurfaceFieldData` from boundary Fourier spectra.

    Pure-``jnp`` body shared by :func:`surface_field_data_from_wout` (numpy
    spectra off a wout) and :func:`surface_field_data_from_state` (traceable
    spectra straight off a live equilibrium state).  All array inputs may be
    jax tracers, so the whole construction differentiates in the boundary.
    """
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, int(nphi), endpoint=False)

    # -- boundary geometry (full mesh) + theta/phi tangents --
    R, Ru, Rv = _mode_series(rmnc[j], None if rmns is None else rmns[j], xm, xn, theta, phi)
    Z, Zu, Zv = _mode_series(None if zmnc is None else zmnc[j], zmns[j], xm, xn, theta, phi)

    # radial derivative (only used to orient the normal outward)
    hs = 1.0 / (ns - 1)
    Rs, _, _ = _mode_series((rmnc[j] - rmnc[j - 1]) / hs,
                            None if rmns is None else (rmns[j] - rmns[j - 1]) / hs,
                            xm, xn, theta, phi)
    Zs, _, _ = _mode_series(None if zmnc is None else (zmnc[j] - zmnc[j - 1]) / hs,
                            (zmns[j] - zmns[j - 1]) / hs, xm, xn, theta, phi)

    cphi = jnp.cos(phi)[:, None]
    sphi = jnp.sin(phi)[:, None]

    def cart(vR, vP, vZ):
        return jnp.stack([vR * cphi - vP * sphi, vR * sphi + vP * cphi, vZ], axis=0)  # (3, nphi, ntheta)

    zero = jnp.zeros_like(R)
    gamma = cart(R, zero, Z)
    e_s = cart(Rs, zero, Zs)
    e_theta = cart(Ru, zero, Zu)
    e_phi = cart(Rv, R, Zv)

    # -- boundary total field (edge-extrapolated half-mesh contravariant B) --
    if j in (ns - 1, -1 % ns):
        bu_edge = 1.5 * bsupu[-1] - 0.5 * bsupu[-2]
        bv_edge = 1.5 * bsupv[-1] - 0.5 * bsupv[-2]
        bu_edge_s = None if bsupu_s is None else (1.5 * bsupu_s[-1] - 0.5 * bsupu_s[-2])
        bv_edge_s = None if bsupv_s is None else (1.5 * bsupv_s[-1] - 0.5 * bsupv_s[-2])
    else:  # interior full-mesh surface: half-mesh average
        bu_edge = 0.5 * (bsupu[j] + bsupu[j + 1])
        bv_edge = 0.5 * (bsupv[j] + bsupv[j + 1])
        bu_edge_s = None if bsupu_s is None else 0.5 * (bsupu_s[j] + bsupu_s[j + 1])
        bv_edge_s = None if bsupv_s is None else 0.5 * (bsupv_s[j] + bsupv_s[j + 1])

    bu = _nyquist_synth(bu_edge, bu_edge_s, xmn, xnn, theta, phi)
    bv = _nyquist_synth(bv_edge, bv_edge_s, xmn, xnn, theta, phi)
    B_total = bu[None, :, :] * e_theta + bv[None, :, :] * e_phi

    # -- normal / area, oriented outward --
    area_vector = jnp.cross(e_theta, e_phi, axis=0)          # (3, nphi, ntheta)
    area_norm = jnp.linalg.norm(area_vector, axis=0)
    normal = area_vector / jnp.maximum(area_norm, 1e-300)
    mean_radial = jnp.mean(jnp.sum(e_s * normal, axis=0))
    flip = jnp.where(mean_radial < 0.0, -1.0, 1.0)
    normal = flip * normal
    area_vector = flip * area_vector

    return VmecSurfaceFieldData(
        gamma=gamma,
        B_total=B_total,
        normal=normal,
        area_vector=area_vector,
        theta=theta,
        phi=phi,
        nfp=nfp,
        stellsym=(not lasym) and bool(use_stellsym),
        signgs=int(signgs),
        source_convention=source_convention,
    )


def _wrout_cos_coeffs_jax(f, modes, trig):
    """Traceable clone of :func:`nyquist.wrout_cos_coeffs`.

    Identical wrout.f Nyquist cosine analysis (``(ns, mnmax)`` coefficients),
    but with ``jnp`` einsums so the field ``f`` may be a jax tracer; the trig
    weight tables depend only on ``trig``/``modes`` and stay static numpy.
    """
    from .nyquist import _wrout_dmult, _wrout_theta_tables, _wrout_zeta_tables

    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    cosmui, sinmui = _wrout_theta_tables(trig)
    cosnv, sinnv = _wrout_zeta_tables(trig)
    fj = jnp.asarray(f)[:, : int(trig.ntheta2), :]
    f_theta_cos = jnp.einsum("sik,im->smk", fj, jnp.asarray(cosmui))
    f_theta_sin = jnp.einsum("sik,im->smk", fj, jnp.asarray(sinmui))
    cos_zeta = jnp.einsum("smk,kn->smn", f_theta_cos, jnp.asarray(cosnv))
    sin_zeta = jnp.einsum("smk,kn->smn", f_theta_sin, jnp.asarray(sinnv))
    sgn = np.where(n < 0, -1.0, 1.0)
    coeff = cos_zeta[:, m, np.abs(n)] + sgn[None, :] * sin_zeta[:, m, np.abs(n)]
    return coeff * jnp.asarray(_wrout_dmult(modes, trig))[None, :]


def surface_field_data_from_state(
    inp,
    state,
    *,
    nphi: int = 32,
    ntheta: int = 32,
    s_index: int = -1,
    use_stellsym: bool = True,
) -> "VmecSurfaceFieldData":
    """Traceable :class:`VmecSurfaceFieldData` straight from a ``SpectralState``.

    Unlike :func:`surface_field_data_from_wout` — which reads a materialised
    (numpy) wout and so cannot be differentiated in the boundary — this rebuilds
    the boundary geometry and contravariant-``B`` spectra with the *same*
    ``jnp`` recipe the wout writer uses (:func:`m1_constrained_to_physical`,
    :func:`real_space_geometry`, :func:`~vmec_jax.core.nyquist.wout_field_tables`),
    never leaving the device.  The result differentiates in ``state`` (hence in
    the boundary DOFs through the implicit adjoint), which is what makes a
    *simultaneous* plasma-boundary + coil single-stage objective possible:
    ``jax.grad`` threads through both this surface field and the coil field.

    ``inp`` supplies the static resolution / profile metadata; ``state`` the
    (possibly traced) spectral geometry.  Stellarator-symmetric only for now
    (``lasym`` uses the same path but is untested here).
    """
    _require_vcj()
    from .fields import magnetic_fields, metric_elements
    from .fourier import Resolution, mode_table, trig_tables
    from .geometry import (
        apply_lambda_axis_closure,
        half_mesh_jacobian,
        real_space_geometry,
    )
    from .nyquist import nyquist_limits
    from .residuals import m1_constrained_to_physical
    from .setup import boundary_from_input, flux_profiles, radial_grids
    from .solver import resolution_from_input
    from .transforms import physical_to_internal_scale

    ns = int(np.shape(state.R_cos)[0])
    res = resolution_from_input(inp, ns=ns)
    mpol, ntor, nfp, lasym = int(res.mpol), int(res.ntor), int(res.nfp), bool(res.lasym)
    nzeta = int(res.nzeta)
    modes = mode_table(mpol, ntor)
    mnyq_grid = max(res.ntheta1 // 2, mpol - 1)
    nnyq_grid = max(nzeta // 2, ntor)
    trig = trig_tables(Resolution(mpol=mnyq_grid + 1, ntor=nnyq_grid,
                                  ntheta=int(res.ntheta), nzeta=nzeta,
                                  nfp=nfp, lasym=lasym, ns=ns))
    grids = radial_grids(ns)
    ncurr = int(inp.ncurr)

    boundary = boundary_from_input(inp, modes=modes, trig=trig, lconm1=True)
    signgs = int(boundary.signgs)
    prof = flux_profiles(inp, grids, r00=boundary.r00, signgs=signgs, lflip=boundary.lflip)

    R_cos_p, Z_sin_p, R_sin_p, Z_cos_p = m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos,
        modes=modes, lthreed=bool(res.lthreed), lasym=lasym, lconm1=True)
    lambda_sin = apply_lambda_axis_closure(state.L_sin, modes=modes, ntor=ntor)
    geometry = real_space_geometry(
        R_cos=R_cos_p, R_sin=R_sin_p, Z_cos=Z_cos_p, Z_sin=Z_sin_p,
        lambda_cos=state.L_cos, lambda_sin=lambda_sin,
        modes=modes, trig=trig, s=grids.s_full)
    jacobian = half_mesh_jacobian(geometry, s=grids.s_full)
    metrics = metric_elements(geometry, s=grids.s_full)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=trig,
        s=grids.s_full, phips=prof["phips"], phipf=prof["phipf"],
        chips=prof["chips"], signgs=signgs, gamma=float(inp.gamma),
        mass=prof["mass"], ncurr=ncurr, enclosed_current=prof["icurv"])
    # Contravariant-B Nyquist cosine spectra (wrout.f analysis), computed
    # traceably in the field state.  nyquist.wout_field_tables materialises
    # everything to numpy; only the two B^u/B^v transforms are needed here, so
    # a jnp clone of wrout_cos_coeffs keeps the whole path differentiable.
    mnyq, nnyq = nyquist_limits(trig)
    nyq_modes = mode_table(max(mnyq, max(mpol - 1, 0)) + 1, max(nnyq, ntor))
    xm_nyq = jnp.asarray(nyq_modes.m, dtype=float)
    xn_nyq = jnp.asarray(nyq_modes.n, dtype=float) * float(nfp)
    bsupumnc = _wrout_cos_coeffs_jax(fields.bsupu, nyq_modes, trig)
    bsupvmnc = _wrout_cos_coeffs_jax(fields.bsupv, nyq_modes, trig)

    mode_scale = 1.0 / physical_to_internal_scale(modes, trig)
    rmnc = R_cos_p * mode_scale[None, :]
    zmns = Z_sin_p * mode_scale[None, :]
    rmns = R_sin_p * mode_scale[None, :] if lasym else None
    zmnc = Z_cos_p * mode_scale[None, :] if lasym else None
    xm = jnp.asarray(modes.m, dtype=float)
    xn = jnp.asarray(modes.n, dtype=float) * float(nfp)

    return _assemble_surface_field_data(
        nfp=nfp, ns=ns, j=int(s_index % ns), xm=xm, xn=xn,
        xmn=xm_nyq, xnn=xn_nyq, rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc,
        bsupu=bsupumnc, bsupv=bsupvmnc, bsupu_s=None, bsupv_s=None,
        lasym=lasym, use_stellsym=use_stellsym, signgs=signgs,
        nphi=nphi, ntheta=ntheta, source_convention="vmec_jax_state")


# ---------------------------------------------------------------------------
# Plasma field on the boundary (accurate on-surface virtual casing)
# ---------------------------------------------------------------------------


def _default_levels(nphi: int, ntheta: int) -> tuple[tuple[int, int], ...]:
    base = max(int(nphi), int(ntheta))
    return ((base, base), (2 * base, 2 * base))


def plan_vc_precision(
    surface_data: "VmecSurfaceFieldData",
    *,
    digits: int = 4,
    chunk_size: int = 1024,
    quad_nt: int | None = None,
    quad_np: int | None = None,
):
    """Select virtual-casing precision from a *concrete* surface, once.

    Returns a :class:`virtual_casing_jax.PrecisionPlan` (quadrature grid sizes +
    singular-patch indices).  Pass it to :func:`plasma_field_on_boundary` or
    :meth:`FreeBoundaryDiffProblem.from_surface_data` via ``precision=`` so the
    plasma field stays differentiable in the boundary geometry — the adaptive
    precision selection (which concretizes surface-derived values) then runs
    here, outside the traced region, on the current concrete boundary.
    """
    _require_vcj()
    cfg = ExteriorFieldConfig(
        digits=int(digits),
        levels=_default_levels(int(surface_data.gamma.shape[1]), int(surface_data.gamma.shape[2])),
        chunk_size=chunk_size,
        target_chunk_size=8,
        dtype="float64",
    )
    field = VirtualCasingExteriorField(surface_data, cfg)
    return field._vc.plan_precision(digits=int(digits), quad_nt=quad_nt, quad_np=quad_np)


def plasma_field_on_boundary(
    surface_data: "VmecSurfaceFieldData",
    *,
    digits: int = 4,
    chunk_size: int = 1024,
    quad_nt: int | None = None,
    quad_np: int | None = None,
    precision=None,
) -> jax.Array:
    """Plasma's own Cartesian field on its boundary via on-surface virtual casing.

    Uses the accurate singular-quadrature on-surface integral
    ``VirtualCasingJAX.compute_internal_B`` (the same routine the package's parity
    tests exercise), which is well-behaved on the source surface — unlike the
    off-surface schedule, which is near-singular there.  Returns ``(3, nphi,
    ntheta)`` in the same layout as ``surface_data.B_total``.

    This is the ``internal`` virtual-casing branch (currents inside the LCFS =
    the plasma current), i.e. the SIMSOPT ``VirtualCasing.B_external_normal``
    convention: the coils must supply ``-B_plasma . n`` for ``B_out . n = 0``.
    """

    _require_vcj()
    cfg = ExteriorFieldConfig(
        digits=int(digits),
        levels=_default_levels(int(surface_data.gamma.shape[1]), int(surface_data.gamma.shape[2])),
        chunk_size=chunk_size,
        target_chunk_size=8,
        dtype="float64",
    )
    field = VirtualCasingExteriorField(surface_data, cfg)
    kwargs: dict[str, Any] = dict(digits=int(digits), chunk_size=int(chunk_size))
    if quad_nt is not None:
        kwargs["quad_nt"] = int(quad_nt)
    if quad_np is not None:
        kwargs["quad_np"] = int(quad_np)
    if precision is not None:
        kwargs["precision"] = precision
    return field._vc.compute_internal_B(field.B_total, **kwargs)


# ---------------------------------------------------------------------------
# External-field evaluation on the boundary (differentiable in the dofs)
# ---------------------------------------------------------------------------


def external_B_cartesian(
    external_field: Any,
    gamma: jax.Array,
    phi_grid: jax.Array | None = None,
) -> jax.Array:
    """Cartesian external field at boundary points ``gamma`` = ``(3, nphi, ntheta)``.

    Dispatches on the external-field type, staying differentiable in its dofs:

    - :class:`~vmec_jax.core.mgrid.MgridField` -> trilinear mgrid (diff. in
      ``extcur``),
    - a plain callable ``xyz(..., 3) -> B(..., 3)`` (e.g. an ESSOS ``Coils``
      Biot-Savart field, ``lambda pts: coils.B(pts)``; diff. in its own dofs).

    Returns ``(3, nphi, ntheta)``.
    """

    x, y, z = gamma[0], gamma[1], gamma[2]

    if isinstance(external_field, MgridField):
        r = jnp.sqrt(x * x + y * y)
        phi = jnp.arctan2(y, x) if phi_grid is None else phi_grid[:, None]
        Br, Bphi, Bz = external_field.b_cyl(r, phi, z)
        cphi, sphi = jnp.cos(phi), jnp.sin(phi)
        Bx = Br * cphi - Bphi * sphi
        By = Br * sphi + Bphi * cphi
        return jnp.stack([Bx, By, Bz], axis=0)

    if callable(external_field):
        pts = jnp.stack([x, y, z], axis=-1)
        B = jnp.asarray(external_field(pts))
        return jnp.moveaxis(B, -1, 0) if B.shape[-1] == 3 else B

    raise TypeError(
        f"external_field must be an MgridField or callable, got {type(external_field).__name__}"
    )


# ---------------------------------------------------------------------------
# The differentiable free-boundary problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeBoundaryDiffProblem:
    """A differentiable free-boundary residual for a fixed trial boundary.

    Bundles the (constant, coil-dof-independent) boundary geometry and plasma
    field, precomputed once via virtual casing, with objective methods that are
    smooth JAX functions of an external field's dofs.  Build with
    :meth:`from_wout` (or :meth:`from_equilibrium`).

    Attributes
    ----------
    gamma, normal:
        Boundary position and outward unit normal, ``(3, nphi, ntheta)``.
    weights:
        Area-element weights (surface Jacobian), normalised to sum 1.
    phi_grid:
        Per-toroidal-plane physical angle ``(nphi,)`` (for the mgrid path).
    Bn_plasma:
        Plasma's own normal field ``B_plasma . n`` on the boundary ``(nphi,
        ntheta)`` — constant w.r.t. the coil dofs.
    B_plasma:
        Plasma's own Cartesian field on the boundary ``(3, nphi, ntheta)``.
    Bin_mag2:
        ``|B_total|^2`` (internal side) on the boundary ``(nphi, ntheta)``.
    p_edge:
        Edge pressure [Pa] for the pressure-balance residual (0 at a true LCFS).
    """

    gamma: jax.Array
    normal: jax.Array
    weights: jax.Array
    phi_grid: jax.Array
    Bn_plasma: jax.Array
    B_plasma: jax.Array
    Bin_mag2: jax.Array
    p_edge: float
    nfp: int

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_surface_data(
        cls,
        surface_data: "VmecSurfaceFieldData",
        *,
        p_edge: float = 0.0,
        digits: int = 4,
        chunk_size: int = 1024,
        quad_nt: int | None = None,
        quad_np: int | None = None,
        precision=None,
    ) -> "FreeBoundaryDiffProblem":
        """Precompute the constants (virtual-casing plasma field) from surface data.

        Pass ``precision`` (from :func:`plan_vc_precision`, selected once from a
        concrete surface) when this runs inside a ``jax.grad`` over the boundary
        geometry, so the virtual-casing plasma field is differentiable in the
        surface rather than tripping the precision auto-selection's concretization.
        """

        _require_vcj()
        gamma = jnp.asarray(surface_data.gamma)
        normal = jnp.asarray(surface_data.normal)
        area = jnp.linalg.norm(jnp.asarray(surface_data.area_vector), axis=0)
        weights = area / jnp.sum(area)
        phi_grid = jnp.asarray(surface_data.phi)

        B_plasma = plasma_field_on_boundary(
            surface_data, digits=digits, chunk_size=chunk_size,
            quad_nt=quad_nt, quad_np=quad_np, precision=precision,
        )
        Bn_plasma = jnp.sum(B_plasma * normal, axis=0)
        Bin_mag2 = jnp.sum(jnp.asarray(surface_data.B_total) ** 2, axis=0)
        return cls(
            gamma=gamma,
            normal=normal,
            weights=weights,
            phi_grid=phi_grid,
            Bn_plasma=Bn_plasma,
            B_plasma=B_plasma,
            Bin_mag2=Bin_mag2,
            p_edge=float(p_edge),
            nfp=int(surface_data.nfp),
        )

    @classmethod
    def from_wout(
        cls,
        wout,
        *,
        nphi: int = 32,
        ntheta: int = 32,
        s_index: int = -1,
        p_edge: float = 0.0,
        digits: int = 4,
        chunk_size: int = 1024,
        quad_nt: int | None = None,
        quad_np: int | None = None,
    ) -> "FreeBoundaryDiffProblem":
        """Build from a converged/trial wout (see :func:`surface_field_data_from_wout`)."""

        sd = surface_field_data_from_wout(wout, nphi=nphi, ntheta=ntheta, s_index=s_index)
        return cls.from_surface_data(
            sd, p_edge=p_edge, digits=digits, chunk_size=chunk_size, quad_nt=quad_nt, quad_np=quad_np
        )

    @classmethod
    def from_equilibrium(cls, eq, **kwargs) -> "FreeBoundaryDiffProblem":
        """Build from an :class:`~vmec_jax.core.optimize.Equilibrium` (uses ``eq.wout``)."""

        wout = eq.wout if hasattr(eq, "wout") else eq
        return cls.from_wout(wout, **kwargs)

    # -- field evaluation -----------------------------------------------------

    def external_B(self, external_field: Any) -> jax.Array:
        """External Cartesian field ``B_ext`` on the boundary, ``(3, nphi, ntheta)``."""

        return external_B_cartesian(external_field, self.gamma, self.phi_grid)

    def external_Bn(self, external_field: Any) -> jax.Array:
        """External normal field ``B_ext . n`` on the boundary, ``(nphi, ntheta)``."""

        return jnp.sum(self.external_B(external_field) * self.normal, axis=0)

    def total_B_out(self, external_field: Any) -> jax.Array:
        """Vacuum-side total field ``B_out = B_plasma + B_ext``, ``(3, nphi, ntheta)``."""

        return self.B_plasma + self.external_B(external_field)

    # -- residuals ------------------------------------------------------------

    def bnormal_residual(self, external_field: Any) -> jax.Array:
        """Free-boundary normal residual ``(B_plasma + B_ext) . n``, ``(nphi, ntheta)``.

        Zero for a self-consistent free-boundary equilibrium; equals
        ``B_ext . n - B_external_normal`` in the SIMSOPT stage-2 convention.
        """

        return self.Bn_plasma + self.external_Bn(external_field)

    def pressure_balance_residual(self, external_field: Any) -> jax.Array:
        """Pressure-balance residual ``|B_out|^2 - |B_in|^2 - 2 mu0 p``, ``(nphi, ntheta)``."""

        Bout2 = jnp.sum(self.total_B_out(external_field) ** 2, axis=0)
        return Bout2 - self.Bin_mag2 - 2.0 * MU0 * self.p_edge

    # -- scalar objectives (weighted mean squares) ----------------------------

    def bnormal_objective(self, external_field: Any) -> jax.Array:
        """Area-weighted mean square of :meth:`bnormal_residual` (a scalar)."""

        r = self.bnormal_residual(external_field)
        return jnp.sum(self.weights * r * r)

    def pressure_balance_objective(self, external_field: Any) -> jax.Array:
        """Area-weighted mean square of :meth:`pressure_balance_residual` (a scalar)."""

        r = self.pressure_balance_residual(external_field)
        return jnp.sum(self.weights * r * r)

    def objective(self, external_field: Any, *, pressure_weight: float = 0.0) -> jax.Array:
        """Combined residual ``J_bn + pressure_weight * J_pressure`` (a scalar)."""

        j = self.bnormal_objective(external_field)
        if pressure_weight:
            j = j + float(pressure_weight) * self.pressure_balance_objective(external_field)
        return j


def value_and_grad_bnormal(
    problem: "FreeBoundaryDiffProblem",
    external_field: Any,
) -> tuple[jax.Array, Any]:
    """``(J, dJ/d external_field)`` of the normal-field objective via ``jax.value_and_grad``.

    ``external_field`` is a pytree (an ``MgridField``, or a callable closing over
    coil dofs); the gradient has the same structure (``extcur``, or the coil dofs).
    """

    def fun(ef: Any) -> jax.Array:
        return problem.bnormal_objective(ef)

    return jax.value_and_grad(fun)(external_field)
