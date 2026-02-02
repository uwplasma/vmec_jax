"""Minimal `wout_*.nc` reader helpers.

This module is intentionally small and only depends on `netCDF4` when used.
It is meant for regression comparisons against VMEC2000 outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .modes import vmec_mode_table
from .state import StateLayout, VMECState


MU0 = 4e-7 * np.pi  # N/A^2


@dataclass(frozen=True)
class WoutData:
    path: Path
    ns: int
    mpol: int
    ntor: int
    nfp: int
    lasym: bool
    signgs: int

    # main mode table
    xm: np.ndarray
    xn: np.ndarray

    # nyquist mode table
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray

    # geometry coefficients (full mesh)
    rmnc: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray
    zmns: np.ndarray
    lmnc: np.ndarray
    lmns: np.ndarray

    # flux functions / profiles
    phipf: np.ndarray
    chipf: np.ndarray
    phips: np.ndarray

    # nyquist Fourier coefficients for derived fields
    gmnc: np.ndarray
    gmns: np.ndarray
    bsupumnc: np.ndarray
    bsupumns: np.ndarray
    bsupvmnc: np.ndarray
    bsupvmns: np.ndarray

    # nyquist Fourier coefficients for covariant field components (for parity checks)
    bsubumnc: np.ndarray
    bsubumns: np.ndarray
    bsubvmnc: np.ndarray
    bsubvmns: np.ndarray

    wb: float
    volume_p: float

    # pressure / energy scalars (VMEC internal units)
    gamma: float
    wp: float
    vp: np.ndarray  # (ns,) volume derivative on half mesh, normalized by (2π)^2
    pres: np.ndarray  # (ns,) pressure on half mesh in mu0*Pa (B^2 units)
    presf: np.ndarray  # (ns,) pressure on full mesh in mu0*Pa (B^2 units)

    # force residual diagnostics (VMEC scalars)
    fsqr: float  # radial force residual
    fsqz: float  # vertical force residual
    fsql: float  # lambda/constraint residual
    fsqt: np.ndarray  # force trace vs iteration (if present)


def _bool_from_nc(x: Any) -> bool:
    # VMEC stores *_logical__ as 0/1 integers in netcdf.
    try:
        return bool(int(np.asarray(x)))
    except Exception:
        return bool(x)


def read_wout(path: str | Path) -> WoutData:
    """Read a subset of `wout_*.nc` needed for step-4 regressions."""
    path = Path(path)
    try:
        import netCDF4  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("netCDF4 is required to read wout files (pip install -e .[netcdf])") from e

    with netCDF4.Dataset(path) as ds:
        ns = int(ds.variables["ns"][:])
        mpol = int(ds.variables["mpol"][:])
        ntor = int(ds.variables["ntor"][:])
        nfp = int(ds.variables["nfp"][:])
        lasym = _bool_from_nc(ds.variables.get("lasym__logical__", 0)[:])
        signgs = int(ds.variables["signgs"][:])

        xm = np.asarray(ds.variables["xm"][:], dtype=int)
        xn = np.asarray(ds.variables["xn"][:], dtype=int)
        xm_nyq = np.asarray(ds.variables["xm_nyq"][:], dtype=int)
        xn_nyq = np.asarray(ds.variables["xn_nyq"][:], dtype=int)

        rmnc = np.asarray(ds.variables["rmnc"][:])
        rmns = np.asarray(ds.variables.get("rmns", np.zeros_like(rmnc))[:])
        zmns = np.asarray(ds.variables["zmns"][:])
        zmnc = np.asarray(ds.variables.get("zmnc", np.zeros_like(zmns))[:])
        lmns = np.asarray(ds.variables["lmns"][:])
        lmnc = np.asarray(ds.variables.get("lmnc", np.zeros_like(lmns))[:])

        phipf = np.asarray(ds.variables["phipf"][:])
        chipf = np.asarray(ds.variables["chipf"][:])
        phips = np.asarray(ds.variables["phips"][:])

        gmnc = np.asarray(ds.variables["gmnc"][:])
        gmns = np.asarray(ds.variables.get("gmns", np.zeros_like(gmnc))[:])
        bsupumnc = np.asarray(ds.variables["bsupumnc"][:])
        bsupumns = np.asarray(ds.variables.get("bsupumns", np.zeros_like(bsupumnc))[:])
        bsupvmnc = np.asarray(ds.variables["bsupvmnc"][:])
        bsupvmns = np.asarray(ds.variables.get("bsupvmns", np.zeros_like(bsupvmnc))[:])

        bsubumnc = np.asarray(ds.variables.get("bsubumnc", np.zeros_like(bsupumnc))[:])
        bsubumns = np.asarray(ds.variables.get("bsubumns", np.zeros_like(bsupumnc))[:])
        bsubvmnc = np.asarray(ds.variables.get("bsubvmnc", np.zeros_like(bsupvmnc))[:])
        bsubvmns = np.asarray(ds.variables.get("bsubvmns", np.zeros_like(bsupvmnc))[:])

        wb = float(ds.variables["wb"][:])
        volume_p = float(ds.variables["volume_p"][:])
        gamma = float(ds.variables.get("gamma", 0.0)[:]) if "gamma" in ds.variables else 0.0
        wp = float(ds.variables.get("wp", 0.0)[:]) if "wp" in ds.variables else 0.0
        vp = np.asarray(ds.variables.get("vp", np.zeros((ns,), dtype=float))[:])

        # `wout` stores pres/presf divided by mu0. Convert back to VMEC internal
        # units (mu0*Pa) so it matches the energy functional.
        pres_pa = np.asarray(ds.variables.get("pres", np.zeros((ns,), dtype=float))[:])
        presf_pa = np.asarray(ds.variables.get("presf", np.zeros((ns,), dtype=float))[:])
        pres = MU0 * pres_pa
        presf = MU0 * presf_pa

        # Force residual scalars (present in most VMEC wout files).
        fsqr = float(ds.variables.get("fsqr", 0.0)[:]) if "fsqr" in ds.variables else 0.0
        fsqz = float(ds.variables.get("fsqz", 0.0)[:]) if "fsqz" in ds.variables else 0.0
        fsql = float(ds.variables.get("fsql", 0.0)[:]) if "fsql" in ds.variables else 0.0
        fsqt = np.asarray(ds.variables.get("fsqt", np.zeros((0,), dtype=float))[:])

    return WoutData(
        path=path,
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lasym=lasym,
        signgs=signgs,
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        rmnc=rmnc,
        rmns=rmns,
        zmnc=zmnc,
        zmns=zmns,
        lmnc=lmnc,
        lmns=lmns,
        phipf=phipf,
        chipf=chipf,
        phips=phips,
        gmnc=gmnc,
        gmns=gmns,
        bsupumnc=bsupumnc,
        bsupumns=bsupumns,
        bsupvmnc=bsupvmnc,
        bsupvmns=bsupvmns,
        bsubumnc=bsubumnc,
        bsubumns=bsubumns,
        bsubvmnc=bsubvmnc,
        bsubvmns=bsubvmns,
        wb=wb,
        volume_p=volume_p,
        gamma=gamma,
        wp=wp,
        vp=vp,
        pres=pres,
        presf=presf,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        fsqt=fsqt,
    )


def assert_main_modes_match_wout(*, wout: WoutData) -> None:
    """Ensure vmec_jax mode ordering matches the `wout` file (important for parity)."""
    modes = vmec_mode_table(wout.mpol, wout.ntor)
    if modes.K != int(wout.xm.size):
        raise ValueError(f"Mode count mismatch: vmec_jax K={modes.K} wout mnmax={wout.xm.size}")
    if not np.array_equal(modes.m, wout.xm.astype(int)):
        raise ValueError("wout xm ordering does not match vmec_jax vmec_mode_table")
    if not np.array_equal(modes.n, (wout.xn // wout.nfp).astype(int)):
        raise ValueError("wout xn ordering does not match vmec_jax (expected xn = n*nfp)")


def state_from_wout(wout: WoutData) -> VMECState:
    """Build a :class:`~vmec_jax.state.VMECState` from `wout` Fourier coefficients."""
    assert_main_modes_match_wout(wout=wout)
    layout = StateLayout(ns=wout.ns, K=int(wout.xm.size), lasym=bool(wout.lasym))
    return VMECState(
        layout=layout,
        Rcos=wout.rmnc,
        Rsin=wout.rmns,
        Zcos=wout.zmnc,
        Zsin=wout.zmns,
        Lcos=wout.lmnc,
        Lsin=wout.lmns,
    )
