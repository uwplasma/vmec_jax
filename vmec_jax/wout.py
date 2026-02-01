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

    wb: float
    volume_p: float


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

        wb = float(ds.variables["wb"][:])
        volume_p = float(ds.variables["volume_p"][:])

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
        wb=wb,
        volume_p=volume_p,
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
