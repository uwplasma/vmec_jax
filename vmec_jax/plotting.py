"""Plotting helpers for VMEC wout data.

These utilities are adapted from the standalone `vmecPlot2.py` script, but
refactored to use vmec_jax's vectorized Fourier evaluation for speed and
consistency. The functions return NumPy arrays; any plotting backend (e.g.
matplotlib) can be layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .fourier import build_helical_basis, eval_fourier
from .geom import _eval_geom_jit
from .grids import AngleGrid
from .modes import ModeTable
from .field import b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda, lamscale_from_phips
from .vmec_jacobian import vmec_half_mesh_jacobian_from_state
from .vmec_realspace import vmec_realspace_geom_from_state
from .vmec_tomnsp import vmec_trig_tables
from .energy import flux_profiles_from_indata
from .field import signgs_from_sqrtg


@dataclass(frozen=True)
class SurfaceData:
    """Surface data on a (theta, zeta) grid."""

    R: np.ndarray
    Z: np.ndarray
    B: np.ndarray | None = None


def fix_matplotlib_3d(ax):
    """Fix 3D matplotlib aspect so structures do not look distorted."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def _mode_table_from_wout(wout, *, nyq: bool, physical: bool = False) -> ModeTable:
    if nyq:
        m = np.asarray(wout.xm_nyq, dtype=int)
        n_raw = np.asarray(wout.xn_nyq, dtype=int)
    else:
        m = np.asarray(wout.xm, dtype=int)
        n_raw = np.asarray(wout.xn, dtype=int)
    if physical:
        n = n_raw
    else:
        n = n_raw // int(wout.nfp)
    return ModeTable(m=m, n=n)


def _basis_from_wout(wout, theta: np.ndarray, zeta: np.ndarray, *, nyq: bool, physical: bool = False) -> AngleGrid:
    modes = _mode_table_from_wout(wout, nyq=nyq, physical=physical)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=int(wout.nfp))
    basis = build_helical_basis(modes, grid)
    return basis


def surface_rz_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
    nyq: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,Z on a surface from wout Fourier coefficients."""
    basis = _basis_from_wout(wout, theta, zeta, nyq=nyq, physical=False)
    rmnc = np.asarray(wout.rmnc)
    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(rmnc)))
    zmns = np.asarray(wout.zmns)
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(zmns)))

    R = np.asarray(eval_fourier(rmnc[s_index], rmns[s_index], basis))
    Z = np.asarray(eval_fourier(zmnc[s_index], zmns[s_index], basis))
    return R, Z


def surface_rz_from_wout_physical(
    wout,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    s_index: int,
    nyq: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,Z on a surface using physical toroidal angle phi.

    This matches vmecPlot2's convention: phase = m*theta - xn*phi, where
    `xn` already includes the nfp factor.
    """
    basis = _basis_from_wout(wout, theta, phi, nyq=nyq, physical=True)
    rmnc = np.asarray(wout.rmnc)
    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(rmnc)))
    zmns = np.asarray(wout.zmns)
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(zmns)))

    R = np.asarray(eval_fourier(rmnc[s_index], rmns[s_index], basis))
    Z = np.asarray(eval_fourier(zmnc[s_index], zmns[s_index], basis))
    return R, Z


def surface_rz_from_state(
    state,
    modes: ModeTable,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,Z on a surface from a VMECState on a field-period grid."""
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=int(nfp))
    basis = build_helical_basis(modes, grid)
    R = np.asarray(eval_fourier(state.Rcos[s_index], state.Rsin[s_index], basis))
    Z = np.asarray(eval_fourier(state.Zcos[s_index], state.Zsin[s_index], basis))
    return R, Z


def surface_rz_from_state_physical(
    state,
    modes: ModeTable,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    s_index: int,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,Z on a surface using physical toroidal angle phi."""
    zeta = np.asarray(phi) * float(nfp)
    return surface_rz_from_state(state, modes, theta=theta, zeta=zeta, s_index=s_index, nfp=nfp)


def axis_rz_from_state_physical(
    state,
    modes: ModeTable,
    *,
    phi: np.ndarray,
    nfp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis curve from state using physical toroidal angle phi."""
    theta0 = np.zeros((1,), dtype=float)
    R, Z = surface_rz_from_state_physical(
        state,
        modes,
        theta=theta0,
        phi=phi,
        s_index=0,
        nfp=nfp,
    )
    return R[0], Z[0]


def bmag_from_state_physical(
    state,
    static,
    indata=None,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    s_index: int,
    signgs: int | None = None,
    phipf: np.ndarray | None = None,
    chipf: np.ndarray | None = None,
    lamscale: float | None = None,
    sqrtg_floor: float | None = None,
    eps: float = 1e-14,
) -> np.ndarray:
    """Compute B magnitude on a surface using physical toroidal angle phi.

    Notes
    -----
    - If ``phipf/chipf`` are provided, they override ``indata``-derived profiles.
    - If ``indata`` is None and no flux profiles are provided, this raises.
    """
    nfp = int(static.cfg.nfp)
    zeta = np.asarray(phi) * float(nfp)
    grid = AngleGrid(theta=np.asarray(theta), zeta=zeta, nfp=nfp)
    basis = build_helical_basis(static.modes, grid)
    geom = _eval_geom_jit(state, basis, static.s, grid.zeta)
    if signgs is None:
        signgs = signgs_from_sqrtg(geom.sqrtg)

    if phipf is None or chipf is None:
        if indata is None:
            raise ValueError("indata must be provided when phipf/chipf are not supplied")
        flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
        phipf_use = flux.phipf
        chipf_use = flux.chipf
        lamscale_use = flux.lamscale
    else:
        phipf_use = np.asarray(phipf)
        chipf_use = np.asarray(chipf)
        if lamscale is None:
            phips = (signgs * np.asarray(phipf_use)) / (2.0 * np.pi)
            lamscale_use = float(np.asarray(lamscale_from_phips(phips, static.s)))
        else:
            lamscale_use = float(lamscale)

    if sqrtg_floor is None:
        bsupu, bsupv = bsup_from_geom(
            geom,
            phipf=phipf_use,
            chipf=chipf_use,
            nfp=nfp,
            signgs=signgs,
            lamscale=lamscale_use,
            eps=eps,
        )
    else:
        sqrtg = np.asarray(geom.sqrtg)
        sqrtg_use = np.sign(sqrtg) * np.maximum(np.abs(sqrtg), float(sqrtg_floor))
        bsupu, bsupv = bsup_from_sqrtg_lambda(
            sqrtg=sqrtg_use,
            lam_u=geom.L_theta,
            lam_v=geom.L_phi,
            phipf=phipf_use,
            chipf=chipf_use,
            signgs=signgs,
            lamscale=lamscale_use,
            eps=eps,
        )
    B2 = b2_from_bsup(geom, bsupu, bsupv)
    B = np.sqrt(np.maximum(np.asarray(B2), 0.0))
    return B[s_index]


def bmag_from_state_vmec_realspace(
    state,
    static,
    indata=None,
    *,
    s_index: int,
    signgs: int | None = None,
    phipf: np.ndarray | None = None,
    chipf: np.ndarray | None = None,
    lamscale: float | None = None,
    sqrtg_floor: float | None = None,
) -> np.ndarray:
    """Compute |B| using VMEC real-space synthesis + half-mesh Jacobian."""
    nfp = int(static.cfg.nfp)
    trig = vmec_trig_tables(
        ntheta=static.cfg.ntheta,
        nzeta=static.cfg.nzeta,
        nfp=nfp,
        mmax=static.cfg.mpol - 1,
        nmax=static.cfg.ntor,
        lasym=static.cfg.lasym,
    )
    geom = vmec_realspace_geom_from_state(state=state, modes=static.modes, trig=trig)
    if signgs is None:
        signgs = 1

    if phipf is None or chipf is None:
        if indata is None:
            raise ValueError("indata must be provided when phipf/chipf are not supplied")
        flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
        phipf_use = flux.phipf
        chipf_use = flux.chipf
        lamscale_use = flux.lamscale
    else:
        phipf_use = np.asarray(phipf)
        chipf_use = np.asarray(chipf)
        if lamscale is None:
            phips = (signgs * np.asarray(phipf_use)) / (2.0 * np.pi)
            lamscale_use = float(np.asarray(lamscale_from_phips(phips, static.s)))
        else:
            lamscale_use = float(lamscale)

    jac = vmec_half_mesh_jacobian_from_state(
        state=state,
        modes=static.modes,
        trig=trig,
        s=np.asarray(static.s),
    )
    sqrtg = np.asarray(jac.sqrtg)
    if sqrtg_floor is not None:
        sqrtg = np.sign(sqrtg) * np.maximum(np.abs(sqrtg), float(sqrtg_floor))
    lam_u = np.asarray(geom["Lu"]) if geom["Lu"] is not None else np.zeros_like(sqrtg)
    lam_v = np.asarray(geom["Lv"]) if geom["Lv"] is not None else np.zeros_like(sqrtg)

    bsupu, bsupv = bsup_from_sqrtg_lambda(
        sqrtg=sqrtg,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=phipf_use,
        chipf=chipf_use,
        signgs=signgs,
        lamscale=lamscale_use,
    )

    Ru = np.asarray(geom["Ru"])
    Zu = np.asarray(geom["Zu"])
    Rv = np.asarray(geom["Rv"])
    Zv = np.asarray(geom["Zv"])
    R = np.asarray(geom["R"])

    g_tt = Ru * Ru + Zu * Zu
    g_tp = Ru * Rv + Zu * Zv
    g_pp = Rv * Rv + Zv * Zv + R * R

    B2 = g_tt * bsupu**2 + 2.0 * g_tp * bsupu * bsupv + g_pp * bsupv**2
    B = np.sqrt(np.maximum(B2, 0.0))
    return B[s_index]


def bmag_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
) -> np.ndarray:
    """Return B magnitude on a surface from wout Nyquist Fourier coefficients."""
    basis = _basis_from_wout(wout, theta, zeta, nyq=True, physical=False)
    bmnc = np.asarray(wout.bmnc)
    bmns = np.asarray(getattr(wout, "bmns", np.zeros_like(bmnc)))
    B = np.asarray(eval_fourier(bmnc[s_index], bmns[s_index], basis))
    return B


def bmag_from_wout_physical(
    wout,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    s_index: int,
) -> np.ndarray:
    """Return B magnitude on a surface using physical toroidal angle phi."""
    basis = _basis_from_wout(wout, theta, phi, nyq=True, physical=True)
    bmnc = np.asarray(wout.bmnc)
    bmns = np.asarray(getattr(wout, "bmns", np.zeros_like(bmnc)))
    B = np.asarray(eval_fourier(bmnc[s_index], bmns[s_index], basis))
    return B


def axis_rz_from_wout(wout, *, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis curve from wout Fourier coefficients."""
    zeta = np.asarray(zeta)
    if not hasattr(wout, "raxis_cc") or not hasattr(wout, "zaxis_cs"):
        # Fallback: use the m=0,n=0 mode from rmnc for a constant axis estimate.
        r0 = float(np.asarray(wout.rmnc)[0, 0]) if np.asarray(wout.rmnc).size else 0.0
        return np.full_like(zeta, r0, dtype=float), np.zeros_like(zeta, dtype=float)

    n = np.arange(len(wout.raxis_cc), dtype=float)
    angle = (-n[:, None] * float(wout.nfp)) * zeta[None, :]
    raxis_cc = np.asarray(wout.raxis_cc, dtype=float)[:, None]
    raxis_cs = np.asarray(getattr(wout, "raxis_cs", np.zeros_like(wout.raxis_cc)), dtype=float)[:, None]
    zaxis_cs = np.asarray(wout.zaxis_cs, dtype=float)[:, None]
    zaxis_cc = np.asarray(getattr(wout, "zaxis_cc", np.zeros_like(wout.zaxis_cs)), dtype=float)[:, None]

    R = np.sum(raxis_cc * np.cos(angle) + raxis_cs * np.sin(angle), axis=0)
    Z = np.sum(zaxis_cs * np.sin(angle) + zaxis_cc * np.cos(angle), axis=0)
    return R, Z


def axis_rz_from_wout_physical(wout, *, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis curve using physical toroidal angle phi (vmecPlot2 convention)."""
    if not hasattr(wout, "raxis_cc") or not hasattr(wout, "zaxis_cs"):
        r0 = float(np.asarray(wout.rmnc)[0, 0]) if np.asarray(wout.rmnc).size else 0.0
        return np.full_like(phi, r0, dtype=float), np.zeros_like(phi, dtype=float)

    phi = np.asarray(phi)
    n = np.arange(len(wout.raxis_cc), dtype=float)
    angle = (-n[:, None] * float(wout.nfp)) * phi[None, :]
    raxis_cc = np.asarray(wout.raxis_cc, dtype=float)[:, None]
    raxis_cs = np.asarray(getattr(wout, "raxis_cs", np.zeros_like(wout.raxis_cc)), dtype=float)[:, None]
    zaxis_cs = np.asarray(wout.zaxis_cs, dtype=float)[:, None]
    zaxis_cc = np.asarray(getattr(wout, "zaxis_cc", np.zeros_like(wout.zaxis_cs)), dtype=float)[:, None]

    R = np.sum(raxis_cc * np.cos(angle) + raxis_cs * np.sin(angle), axis=0)
    Z = np.sum(zaxis_cs * np.sin(angle) + zaxis_cc * np.cos(angle), axis=0)
    return R, Z


def profiles_from_wout(wout) -> dict[str, np.ndarray]:
    """Return common radial profiles from wout."""
    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    s_half = (np.arange(1, ns, dtype=float) - 0.5) / float(ns - 1)
    return {
        "s": s,
        "s_half": s_half,
        "iotaf": np.asarray(wout.iotaf),
        "iotas": np.asarray(wout.iotas),
        "presf": np.asarray(wout.presf),
        "pres": np.asarray(wout.pres),
        "buco": np.asarray(getattr(wout, "buco", np.zeros_like(wout.presf))),
        "bvco": np.asarray(getattr(wout, "bvco", np.zeros_like(wout.presf))),
        "jcuru": np.asarray(getattr(wout, "jcuru", np.zeros_like(wout.presf))),
        "jcurv": np.asarray(getattr(wout, "jcurv", np.zeros_like(wout.presf))),
    }


def surface_data_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
    with_bmag: bool = False,
) -> SurfaceData:
    """Convenience wrapper returning R/Z (and optionally B magnitude) on a surface."""
    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index, nyq=False)
    if with_bmag:
        B = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index)
    else:
        B = None
    return SurfaceData(R=R, Z=Z, B=B)


def closed_theta_grid(ntheta: int) -> np.ndarray:
    """Theta grid including the 2π endpoint (good for closed cross-sections)."""
    return np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=True)


def zeta_grid(nzeta: int, *, endpoint: bool = False) -> np.ndarray:
    """Uniform zeta grid over one field period."""
    return np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=bool(endpoint))


def zeta_grid_field_period(nzeta: int, *, nfp: int) -> np.ndarray:
    """Uniform zeta grid over one field period (0..2π/nfp)."""
    nfp = max(1, int(nfp))
    return np.linspace(0.0, 2.0 * np.pi / float(nfp), int(nzeta), endpoint=False)


def vmecplot2_cross_section_indices(nzeta: int) -> np.ndarray:
    """Indices used by vmecPlot2 for cross sections (0,2,4,6)."""
    if nzeta < 7:
        raise ValueError("vmecPlot2 cross sections expect nzeta>=8")
    return np.asarray([0, 2, 4, 6], dtype=int)


def select_zeta_slices(zeta: np.ndarray, *, n: int) -> np.ndarray:
    """Pick evenly spaced zeta indices from a zeta grid."""
    zeta = np.asarray(zeta)
    if n <= 0:
        raise ValueError("n must be positive")
    idx = np.linspace(0, len(zeta) - 1, num=n).round().astype(int)
    return zeta[idx]


def surface_stack(
    wout,
    *,
    theta: np.ndarray,
    zeta_list: Iterable[float],
    s_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack R,Z slices for multiple zeta values."""
    zeta = np.asarray(list(zeta_list), dtype=float)
    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=s_index, nyq=False)
    return R, Z
