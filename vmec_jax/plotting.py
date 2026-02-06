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
from .grids import AngleGrid
from .modes import ModeTable


@dataclass(frozen=True)
class SurfaceData:
    """Surface data on a (theta, zeta) grid."""

    R: np.ndarray
    Z: np.ndarray
    B: np.ndarray | None = None


def _mode_table_from_wout(wout, *, nyq: bool) -> ModeTable:
    if nyq:
        m = np.asarray(wout.xm_nyq, dtype=int)
        n = np.asarray(wout.xn_nyq, dtype=int) // int(wout.nfp)
    else:
        m = np.asarray(wout.xm, dtype=int)
        n = np.asarray(wout.xn, dtype=int) // int(wout.nfp)
    return ModeTable(m=m, n=n)


def _basis_from_wout(wout, theta: np.ndarray, zeta: np.ndarray, *, nyq: bool) -> AngleGrid:
    modes = _mode_table_from_wout(wout, nyq=nyq)
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
    basis = _basis_from_wout(wout, theta, zeta, nyq=nyq)
    rmnc = np.asarray(wout.rmnc)
    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(rmnc)))
    zmns = np.asarray(wout.zmns)
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(zmns)))

    R = np.asarray(eval_fourier(rmnc[s_index], rmns[s_index], basis))
    Z = np.asarray(eval_fourier(zmnc[s_index], zmns[s_index], basis))
    return R, Z


def bmag_from_wout(
    wout,
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    s_index: int,
) -> np.ndarray:
    """Return B magnitude on a surface from wout Nyquist Fourier coefficients."""
    basis = _basis_from_wout(wout, theta, zeta, nyq=True)
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


def zeta_grid(nzeta: int) -> np.ndarray:
    """Uniform zeta grid over one field period."""
    return np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=False)


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
