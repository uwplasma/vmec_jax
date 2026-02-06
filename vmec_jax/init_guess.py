"""Initial guess construction for step-1.

VMEC has a fairly elaborate procedure to build an initial nested set of
surfaces from the boundary (and/or axis) Fourier coefficients.

For step-1 we implement a *regularity-aware* but intentionally simple guess that
is good enough to exercise the full (s,theta,zeta) geometry kernel:

- For m>0 harmonics, scale boundary coefficients like rho**m with rho = sqrt(s)
  to enforce regularity at the magnetic axis (matches VMEC/VMEC++).
- For m=0 harmonics, scale with s and, if axis coefficients are provided,
  linearly blend between the axis and the boundary.
- lambda coefficients are initialized to zero.

This guess is not intended to match VMEC's exact internal initial guess yet.
It is a stable, differentiable starting point that we can later improve while
keeping the geometry/transform kernels unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._compat import jnp, has_jax
from .boundary import BoundaryCoeffs
from .fourier import build_helical_basis, eval_fourier, eval_fourier_dtheta
from .namelist import InData
from .state import StateLayout, VMECState
from .static import VMECStatic
from .vmec_realspace import (
    vmec_realspace_analysis,
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
)
from .vmec_tomnsp import vmec_trig_tables


def _read_axis_coeffs(indata: InData) -> dict[str, float | list[float]]:
    """Read axis arrays if present.

    VMEC supports axis series in a few naming conventions. For step-1 we only
    look for the common modern VMEC names:

    - RAXIS_CC, RAXIS_CS
    - ZAXIS_CC, ZAXIS_CS

    Each may be a scalar or a list. We return the raw values.
    """
    out: dict[str, float | list[float]] = {}
    for key in ("RAXIS_CC", "RAXIS_CS", "ZAXIS_CC", "ZAXIS_CS"):
        v = indata.get(key, None)
        if v is None:
            continue
        out[key] = v
    return out


def _axis_array(values: float | list[float] | None, ntor: int, *, dtype):
    if values is None:
        return None
    if isinstance(values, list):
        arr = [float(v) for v in values]
    else:
        arr = [float(values)]
    if len(arr) < ntor + 1:
        arr = arr + [0.0] * (ntor + 1 - len(arr))
    else:
        arr = arr[: ntor + 1]
    return jnp.asarray(arr, dtype=dtype)


def _guess_axis_from_boundary(static: VMECStatic, boundary: BoundaryCoeffs):
    """Guess magnetic axis from boundary geometry.

    We use a simple VMEC-like heuristic: in each toroidal plane, set the axis
    to the midpoint of the (R,Z) bounding box of the LCFS cross-section, then
    Fourier-fit the resulting axis curve in zeta.
    """
    grid = static.grid
    basis = build_helical_basis(static.modes, grid)
    Rb = np.asarray(eval_fourier(jnp.asarray(boundary.R_cos), jnp.asarray(boundary.R_sin), basis))
    Zb = np.asarray(eval_fourier(jnp.asarray(boundary.Z_cos), jnp.asarray(boundary.Z_sin), basis))

    R_axis = 0.5 * (Rb.min(axis=0) + Rb.max(axis=0))
    Z_axis = 0.5 * (Zb.min(axis=0) + Zb.max(axis=0))

    zeta = np.asarray(grid.zeta)
    n = np.arange(static.cfg.ntor + 1)
    cos_nz = np.cos(np.outer(n, zeta))
    sin_nz = np.sin(np.outer(n, zeta))

    delta = 2.0 / float(zeta.size)
    raxis_c = delta * (cos_nz @ R_axis)
    raxis_c[0] *= 0.5
    zaxis_s = delta * (sin_nz @ Z_axis)
    return jnp.asarray(raxis_c), jnp.asarray(zaxis_s)


def _boundary_cross_section_areas(static: VMECStatic, boundary: BoundaryCoeffs) -> np.ndarray:
    basis = build_helical_basis(static.modes, static.grid)
    Rb = np.asarray(eval_fourier(jnp.asarray(boundary.R_cos), jnp.asarray(boundary.R_sin), basis))
    Zb = np.asarray(eval_fourier(jnp.asarray(boundary.Z_cos), jnp.asarray(boundary.Z_sin), basis))
    areas = []
    for k in range(Rb.shape[1]):
        R = Rb[:, k]
        Z = Zb[:, k]
        # signed polygon area, periodic closure
        area = 0.5 * np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)
        areas.append(area)
    return np.asarray(areas)


def _flip_boundary_theta(static: VMECStatic, boundary: BoundaryCoeffs) -> BoundaryCoeffs:
    """Flip theta -> -theta for boundary coefficients (m>0), matching VMEC."""
    m = np.asarray(static.modes.m)
    n = np.asarray(static.modes.n)
    key_to_k = {(int(mm), int(nn)): k for k, (mm, nn) in enumerate(zip(m, n))}

    R_cos = np.asarray(boundary.R_cos).copy()
    R_sin = np.asarray(boundary.R_sin).copy()
    Z_cos = np.asarray(boundary.Z_cos).copy()
    Z_sin = np.asarray(boundary.Z_sin).copy()

    R_cos_new = R_cos.copy()
    R_sin_new = R_sin.copy()
    Z_cos_new = Z_cos.copy()
    Z_sin_new = Z_sin.copy()

    for k, (mm, nn) in enumerate(zip(m, n)):
        if int(mm) == 0:
            continue
        k2 = key_to_k.get((int(mm), int(-nn)))
        if k2 is None:
            continue
        R_cos_new[k] = R_cos[k2]
        R_sin_new[k] = -R_sin[k2]
        Z_cos_new[k] = Z_cos[k2]
        Z_sin_new[k] = -Z_sin[k2]

    return BoundaryCoeffs(R_cos=R_cos_new, R_sin=R_sin_new, Z_cos=Z_cos_new, Z_sin=Z_sin_new)


def _apply_m1_constraint(static: VMECStatic, boundary: BoundaryCoeffs) -> BoundaryCoeffs:
    """Apply VMEC m=1 constraint to boundary coefficients (symmetric runs)."""
    if not bool(getattr(static.cfg, "lconm1", True)):
        return boundary
    if int(static.cfg.ntor) == 0:
        return boundary

    R_cos = np.asarray(boundary.R_cos).copy()
    R_sin = np.asarray(boundary.R_sin).copy()
    Z_cos = np.asarray(boundary.Z_cos).copy()
    Z_sin = np.asarray(boundary.Z_sin).copy()

    for k, m in enumerate(static.modes.m):
        if int(m) != 1:
            continue
        rbs = R_sin[k]
        zbc = Z_cos[k]
        R_sin[k] = 0.5 * (rbs + zbc)
        Z_cos[k] = 0.5 * (rbs - zbc)

    return BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)


def _undo_m1_constraint_for_recompute(static: VMECStatic, boundary: BoundaryCoeffs) -> BoundaryCoeffs:
    """Undo VMEC's m=1 constraint for the boundary (used in axis recompute)."""
    if not bool(getattr(static.cfg, "lconm1", True)):
        return boundary
    if int(static.cfg.ntor) == 0:
        return boundary

    R_cos = np.asarray(boundary.R_cos).copy()
    R_sin = np.asarray(boundary.R_sin).copy()
    Z_cos = np.asarray(boundary.Z_cos).copy()
    Z_sin = np.asarray(boundary.Z_sin).copy()

    for k, m in enumerate(static.modes.m):
        if int(m) != 1:
            continue
        rbs = R_sin[k]
        zbc = Z_cos[k]
        R_sin[k] = rbs + zbc
        Z_cos[k] = rbs - zbc

    return BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)


def _blend_axis_m0(
    *,
    static: VMECStatic,
    s,
    Rcos,
    Zsin,
    Rcos_b,
    Zsin_b,
    raxis_cc,
    zaxis_cs,
):
    """Blend m=0 modes between axis and boundary (VMEC convention)."""
    m0_mask = static.modes.m == 0
    for n in range(static.cfg.ntor + 1):
        k_candidates = jnp.where(m0_mask & (static.modes.n == n))[0]
        if k_candidates.size == 0:
            continue
        k = int(k_candidates[0])
        blend = s
        new_R = (1.0 - blend) * raxis_cc[n] + blend * Rcos_b[0, k]
        # Z uses sin(-n zeta) for m=0, so Zsin coefficients are -zaxis_cs.
        new_Z = (1.0 - blend) * (-zaxis_cs[n]) + blend * Zsin_b[0, k]
        if has_jax():
            Rcos = Rcos.at[:, k].set(new_R)
            Zsin = Zsin.at[:, k].set(new_Z)
        else:
            Rcos = jnp.array(Rcos)
            Zsin = jnp.array(Zsin)
            Rcos[:, k] = new_R
            Zsin[:, k] = new_Z
    return Rcos, Zsin


def _recompute_axis_from_boundary(
    static: VMECStatic,
    boundary: BoundaryCoeffs,
    *,
    raxis_cc: np.ndarray,
    zaxis_cs: np.ndarray,
    signgs: int,
    n_grid: int = 61,
) -> tuple[np.ndarray, np.ndarray]:
    """VMEC++-style axis recompute to maximize min Jacobian in each toroidal plane."""
    cfg = static.cfg
    boundary_recompute = _undo_m1_constraint_for_recompute(static, boundary)

    trig = vmec_trig_tables(
        ntheta=cfg.ntheta,
        nzeta=cfg.nzeta,
        nfp=cfg.nfp,
        mmax=cfg.mpol - 1,
        nmax=cfg.ntor,
        lasym=cfg.lasym,
    )
    ntheta1, ntheta2, ntheta3 = int(trig.ntheta1), int(trig.ntheta2), int(trig.ntheta3)

    coeff_cos = np.asarray(boundary_recompute.R_cos)[None, :]
    coeff_sin = np.asarray(boundary_recompute.R_sin)[None, :]
    R_lcfs_red = np.asarray(
        vmec_realspace_synthesis(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=static.modes, trig=trig)
    )[0]
    coeff_cos = np.asarray(boundary_recompute.Z_cos)[None, :]
    coeff_sin = np.asarray(boundary_recompute.Z_sin)[None, :]
    Z_lcfs_red = np.asarray(
        vmec_realspace_synthesis(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=static.modes, trig=trig)
    )[0]

    coeff_cos = np.asarray(boundary_recompute.R_cos)[None, :]
    coeff_sin = np.asarray(boundary_recompute.R_sin)[None, :]
    dR_dtheta_lcfs_red = np.asarray(
        vmec_realspace_synthesis_dtheta(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=static.modes, trig=trig)
    )[0]
    coeff_cos = np.asarray(boundary_recompute.Z_cos)[None, :]
    coeff_sin = np.asarray(boundary_recompute.Z_sin)[None, :]
    dZ_dtheta_lcfs_red = np.asarray(
        vmec_realspace_synthesis_dtheta(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=static.modes, trig=trig)
    )[0]

    ns = int(cfg.ns)
    ns12 = (ns + 1) // 2 - 1
    s_mid = ns12 / (ns - 1.0)
    delta_s = (ns - 1 - ns12) / (ns - 1.0)
    sqrt_s_mid = np.sqrt(s_mid)

    m = np.asarray(static.modes.m)
    scale_r = np.where(m > 0, sqrt_s_mid**m, 1.0)
    scale_other = np.where(m > 0, sqrt_s_mid**m, s_mid)

    Rcos_mid = scale_r * np.asarray(boundary.R_cos)
    Rsin_mid = scale_other * np.asarray(boundary.R_sin)
    Zcos_mid = scale_other * np.asarray(boundary.Z_cos)
    Zsin_mid = scale_other * np.asarray(boundary.Z_sin)

    # Blend m=0 using axis guess
    m0_mask = static.modes.m == 0
    for n in range(cfg.ntor + 1):
        k_candidates = np.where(m0_mask & (static.modes.n == n))[0]
        if k_candidates.size == 0:
            continue
        k = int(k_candidates[0])
        Rcos_mid[k] = (1.0 - s_mid) * raxis_cc[n] + s_mid * Rcos_mid[k]
        Zsin_mid[k] = (1.0 - s_mid) * (-zaxis_cs[n]) + s_mid * Zsin_mid[k]

    R_half_red = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=Rcos_mid[None, :],
            coeff_sin=Rsin_mid[None, :],
            modes=static.modes,
            trig=trig,
        )
    )[0]
    Z_half_red = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=Zcos_mid[None, :],
            coeff_sin=Zsin_mid[None, :],
            modes=static.modes,
            trig=trig,
        )
    )[0]
    dR_dtheta_half_red = np.asarray(
        vmec_realspace_synthesis_dtheta(
            coeff_cos=Rcos_mid[None, :],
            coeff_sin=Rsin_mid[None, :],
            modes=static.modes,
            trig=trig,
        )
    )[0]
    dZ_dtheta_half_red = np.asarray(
        vmec_realspace_synthesis_dtheta(
            coeff_cos=Zcos_mid[None, :],
            coeff_sin=Zsin_mid[None, :],
            modes=static.modes,
            trig=trig,
        )
    )[0]

    dR_dtheta_half_red = 0.5 * (dR_dtheta_lcfs_red + dR_dtheta_half_red)
    dZ_dtheta_half_red = 0.5 * (dZ_dtheta_lcfs_red + dZ_dtheta_half_red)

    # Expand to full theta grid (ntheta1) for symmetric cases.
    R_lcfs = np.zeros((ntheta1, cfg.nzeta))
    Z_lcfs = np.zeros((ntheta1, cfg.nzeta))
    R_half = np.zeros((ntheta1, cfg.nzeta))
    Z_half = np.zeros((ntheta1, cfg.nzeta))
    dR_dtheta_lcfs = np.zeros((ntheta1, cfg.nzeta))
    dZ_dtheta_lcfs = np.zeros((ntheta1, cfg.nzeta))
    dR_dtheta_half = np.zeros((ntheta1, cfg.nzeta))
    dZ_dtheta_half = np.zeros((ntheta1, cfg.nzeta))

    # Fill reduced theta portion [0,pi].
    R_lcfs[:ntheta3, :] = R_lcfs_red
    Z_lcfs[:ntheta3, :] = Z_lcfs_red
    R_half[:ntheta3, :] = R_half_red
    Z_half[:ntheta3, :] = Z_half_red
    dR_dtheta_lcfs[:ntheta3, :] = dR_dtheta_lcfs_red
    dZ_dtheta_lcfs[:ntheta3, :] = dZ_dtheta_lcfs_red
    dR_dtheta_half[:ntheta3, :] = dR_dtheta_half_red
    dZ_dtheta_half[:ntheta3, :] = dZ_dtheta_half_red

    if not cfg.lasym:
        for iv in range(cfg.nzeta):
            ivminus = (cfg.nzeta - iv) % cfg.nzeta
            for iu in range(ntheta2, ntheta1):
                iu_r = ntheta1 - iu
                R_lcfs[iu, iv] = R_lcfs_red[iu_r, ivminus]
                Z_lcfs[iu, iv] = -Z_lcfs_red[iu_r, ivminus]
                R_half[iu, iv] = R_half_red[iu_r, ivminus]
                Z_half[iu, iv] = -Z_half_red[iu_r, ivminus]
                dR_dtheta_lcfs[iu, iv] = -dR_dtheta_lcfs_red[iu_r, ivminus]
                dZ_dtheta_lcfs[iu, iv] = dZ_dtheta_lcfs_red[iu_r, ivminus]
                dR_dtheta_half[iu, iv] = -dR_dtheta_half_red[iu_r, ivminus]
                dZ_dtheta_half[iu, iv] = dZ_dtheta_half_red[iu_r, ivminus]

    cosnv = np.asarray(trig.cosnv)  # (nzeta, nmax+1)
    sinnv = np.asarray(trig.sinnv)
    nscale = np.asarray(trig.nscale)
    r_axis = (raxis_cc[None, :] * cosnv / nscale[None, :]).sum(axis=1)
    z_axis = -(zaxis_cs[None, :] * sinnv / nscale[None, :]).sum(axis=1)

    new_r_axis = np.zeros_like(r_axis)
    new_z_axis = np.zeros_like(z_axis)

    ntheta, nzeta = R_lcfs.shape
    for k in range(nzeta // 2 + 1):
        rmin, rmax = float(R_lcfs[:, k].min()), float(R_lcfs[:, k].max())
        zmin, zmax = float(Z_lcfs[:, k].min()), float(Z_lcfs[:, k].max())

        dr = (rmax - rmin) / (n_grid - 1)
        dz = (zmax - zmin) / (n_grid - 1)

        r_guess = 0.5 * (rmax + rmin)
        z_guess = 0.5 * (zmax + zmin)

        dR_ds_half = (R_lcfs[:, k] - R_half[:, k]) / delta_s + r_axis[k]
        dZ_ds_half = (Z_lcfs[:, k] - Z_half[:, k]) / delta_s + z_axis[k]

        tau0 = dR_dtheta_half[:, k] * dZ_ds_half - dZ_dtheta_half[:, k] * dR_ds_half

        min_tau_best = -np.inf

        for iz in range(n_grid):
            z_grid = zmin + iz * dz
            if not cfg.lasym and (k == 0 or k == nzeta // 2):
                z_grid = 0.0
                if iz > 0:
                    break
            for ir in range(n_grid):
                r_grid = rmin + ir * dr
                tau = signgs * (tau0 - dR_dtheta_half[:, k] * z_grid + dZ_dtheta_half[:, k] * r_grid)
                min_tau = float(np.min(tau))
                if min_tau > min_tau_best:
                    min_tau_best = min_tau
                    r_guess = r_grid
                    z_guess = z_grid
                elif min_tau == min_tau_best and not cfg.lasym:
                    if abs(z_guess) > abs(z_grid):
                        z_guess = z_grid

        new_r_axis[k] = r_guess
        new_z_axis[k] = z_guess

    if not cfg.lasym:
        for k in range(1, nzeta // 2):
            k_rev = (nzeta - k) % nzeta
            new_r_axis[k_rev] = new_r_axis[k]
            new_z_axis[k_rev] = -new_z_axis[k]

    delta_v = 2.0 / float(nzeta)
    new_raxis_c = delta_v * (cosnv.T @ new_r_axis) / nscale
    new_zaxis_s = -delta_v * (sinnv.T @ new_z_axis) / nscale
    new_raxis_c[0] *= 0.5

    return new_raxis_c, new_zaxis_s


def initial_guess_from_boundary(
    static: VMECStatic,
    boundary: BoundaryCoeffs,
    indata: InData | None = None,
    *,
    dtype=None,
    vmec_project: bool = False,
) -> VMECState:
    """Build a VMECState initial guess from boundary coefficients.

    Parameters
    ----------
    static:
        Precomputed modes/grid/basis and radial coordinate.
    boundary:
        Boundary coefficients aligned with `static.modes`.
    indata:
        If provided, used to read optional axis specification. If absent or if
        the axis arrays are all zero, the axis is inferred from boundary m=0
        coefficients.
    dtype:
        Optional dtype for the returned arrays.
    vmec_project:
        If True, re-project the initial guess through VMEC's internal real-space
        grid (via ``vmec_realspace_synthesis`` + ``vmec_realspace_analysis``).
        This matches the VMEC grid/weighting used in parity diagnostics.
    """
    cfg = static.cfg
    K = static.modes.K
    layout = StateLayout(ns=cfg.ns, K=K, lasym=cfg.lasym)

    m = jnp.asarray(static.modes.m)
    s = jnp.asarray(static.s)
    if dtype is None:
        # Choose a dtype that avoids JAX warning spam.
        # VMEC expects float64; we default to float64 when x64 is enabled.
        if has_jax():
            try:
                import jax

                x64 = bool(jax.config.read("jax_enable_x64"))
            except Exception:
                x64 = True
            dtype = jnp.float64 if x64 else jnp.float32
        else:
            # numpy fallback: use float64 for VMEC parity
            import numpy as _np

            dtype = _np.float64

    boundary_use = boundary
    areas = _boundary_cross_section_areas(static, boundary_use)
    if np.median(areas) < 0.0:
        boundary_use = _flip_boundary_theta(static, boundary_use)
    boundary_use = _apply_m1_constraint(static, boundary_use)

    # Base: broadcast boundary vectors to (ns,K)
    Rcos_b = jnp.asarray(boundary_use.R_cos, dtype=dtype)[None, :]
    Rsin_b = jnp.asarray(boundary_use.R_sin, dtype=dtype)[None, :]
    Zcos_b = jnp.asarray(boundary_use.Z_cos, dtype=dtype)[None, :]
    Zsin_b = jnp.asarray(boundary_use.Z_sin, dtype=dtype)[None, :]

    # Regularity scaling: use rho**m with rho = sqrt(s) for m>0 (VMEC/VMEC++).
    # For m=0, keep Rcos constant unless we blend with axis; other components
    # use s to ensure regularity at the axis.
    rho = jnp.sqrt(s)
    scale_r = jnp.where(m[None, :] > 0, rho[:, None] ** m[None, :], 1.0)
    scale_other = jnp.where(m[None, :] > 0, rho[:, None] ** m[None, :], s[:, None])
    Rcos = scale_r * Rcos_b
    Rsin = scale_other * Rsin_b
    Zcos = scale_other * Zcos_b
    Zsin = scale_other * Zsin_b

    # If user supplied a non-trivial axis spec, blend m=0 coefficients between
    # axis and boundary (linear in s), matching VMEC/VMEC++ conventions.
    if indata is not None:
        ax = _read_axis_coeffs(indata)
        raxis_cc = _axis_array(ax.get("RAXIS_CC", None), cfg.ntor, dtype=dtype)
        zaxis_cs = _axis_array(ax.get("ZAXIS_CS", None), cfg.ntor, dtype=dtype)

        # If axis arrays are all zero or missing, fall back to boundary-based axis.
        have_axis = False
        if raxis_cc is not None and np.any(np.asarray(raxis_cc) != 0.0):
            have_axis = True
        if zaxis_cs is not None and np.any(np.asarray(zaxis_cs) != 0.0):
            have_axis = True
        axis_from_indata = bool(have_axis)

        if not have_axis:
            raxis_cc, zaxis_cs = _guess_axis_from_boundary(static, boundary_use)
            raxis_cc, zaxis_cs = _recompute_axis_from_boundary(
                static,
                boundary_use,
                raxis_cc=np.asarray(raxis_cc),
                zaxis_cs=np.asarray(zaxis_cs),
                signgs=1 if np.median(areas) >= 0.0 else -1,
            )
            raxis_cc = raxis_cc.astype(dtype)
            zaxis_cs = zaxis_cs.astype(dtype)
            have_axis = True

        if have_axis:
            if raxis_cc is None:
                raxis_cc = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
            if zaxis_cs is None:
                zaxis_cs = jnp.zeros((cfg.ntor + 1,), dtype=dtype)

            Rcos, Zsin = _blend_axis_m0(
                static=static,
                s=s,
                Rcos=Rcos,
                Zsin=Zsin,
                Rcos_b=Rcos_b,
                Zsin_b=Zsin_b,
                raxis_cc=raxis_cc,
                zaxis_cs=zaxis_cs,
            )

            # If the user explicitly requests it, re-run the VMEC++ axis recompute
            # even when axis coefficients are supplied in the input.
            if axis_from_indata and bool(indata.get_bool("LRECOMPUTE", False)):
                signgs_guess = 1 if np.median(areas) >= 0.0 else -1
                new_raxis_cc, new_zaxis_cs = _recompute_axis_from_boundary(
                    static,
                    boundary_use,
                    raxis_cc=np.asarray(raxis_cc),
                    zaxis_cs=np.asarray(zaxis_cs),
                    signgs=signgs_guess,
                )
                raxis_cc = jnp.asarray(new_raxis_cc, dtype=dtype)
                zaxis_cs = jnp.asarray(new_zaxis_cs, dtype=dtype)
                Rcos, Zsin = _blend_axis_m0(
                    static=static,
                    s=s,
                    Rcos=Rcos,
                    Zsin=Zsin,
                    Rcos_b=Rcos_b,
                    Zsin_b=Zsin_b,
                    raxis_cc=raxis_cc,
                    zaxis_cs=zaxis_cs,
                )

    if vmec_project:
        if cfg.lasym:
            # Defer asymmetric support for the VMEC-grid projection.
            vmec_project = False
        else:
            trig = vmec_trig_tables(
                ntheta=cfg.ntheta,
                nzeta=cfg.nzeta,
                nfp=cfg.nfp,
                mmax=cfg.mpol - 1,
                nmax=cfg.ntor,
                lasym=cfg.lasym,
                dtype=dtype,
            )
            R_real = vmec_realspace_synthesis(coeff_cos=Rcos, coeff_sin=Rsin, modes=static.modes, trig=trig)
            Z_real = vmec_realspace_synthesis(coeff_cos=Zcos, coeff_sin=Zsin, modes=static.modes, trig=trig)
            Rcos, Rsin = vmec_realspace_analysis(f=R_real, modes=static.modes, trig=trig, parity="cos")
            Zcos, Zsin = vmec_realspace_analysis(f=Z_real, modes=static.modes, trig=trig, parity="sin")
            Rcos = Rcos.astype(dtype)
            Rsin = Rsin.astype(dtype)
            Zcos = Zcos.astype(dtype)
            Zsin = Zsin.astype(dtype)

    Lcos = jnp.zeros((cfg.ns, K), dtype=dtype)
    Lsin = jnp.zeros((cfg.ns, K), dtype=dtype)

    return VMECState(layout=layout, Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin, Lcos=Lcos, Lsin=Lsin)
