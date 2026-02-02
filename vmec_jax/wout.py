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
    iotaf: np.ndarray  # (ns,) iota on half mesh (VMEC convention)
    iotas: np.ndarray  # (ns,) iota on full mesh (VMEC convention)

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

    # nyquist Fourier coefficients for |B|
    bmnc: np.ndarray
    bmns: np.ndarray

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
        iotaf = np.asarray(ds.variables.get("iotaf", np.zeros_like(phips))[:])
        iotas = np.asarray(ds.variables.get("iotas", np.zeros_like(phips))[:])

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

        bmnc = np.asarray(ds.variables.get("bmnc", np.zeros_like(gmnc))[:])
        bmns = np.asarray(ds.variables.get("bmns", np.zeros_like(gmnc))[:])

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
        iotaf=iotaf,
        iotas=iotas,
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
        bmnc=bmnc,
        bmns=bmns,
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


def write_wout(path: str | Path, wout: WoutData, *, overwrite: bool = False) -> None:
    """Write a minimal VMEC-style ``wout_*.nc`` file.

    This is intended for:
    - round-tripping reference ``wout`` files (read -> write -> read),
    - emitting VMEC-compatible output containers from vmec_jax as parity work progresses.

    Notes
    -----
    - Only the subset of variables represented in :class:`WoutData` is written.
    - ``pres`` and ``presf`` are written in **Pa** (VMEC convention: netCDF stores
      pressure divided by ``mu0``). Internally :class:`WoutData` stores pressure in
      VMEC internal units (``mu0*Pa``).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to overwrite)")

    try:
        import netCDF4  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("netCDF4 is required to write wout files (pip install -e .[netcdf])") from e

    # Dimensions.
    ns = int(wout.ns)
    mnmax = int(np.asarray(wout.xm).size)
    mnmax_nyq = int(np.asarray(wout.xm_nyq).size)
    nstore = int(np.asarray(wout.fsqt).size)

    # Convert pressures back to VMEC netcdf convention (Pa).
    pres_pa = np.asarray(wout.pres) / MU0
    presf_pa = np.asarray(wout.presf) / MU0

    with netCDF4.Dataset(path, mode="w", format="NETCDF4") as ds:
        ds.createDimension("ns", ns)
        ds.createDimension("mnmax", mnmax)
        ds.createDimension("mnmax_nyq", mnmax_nyq)
        ds.createDimension("nstore_seq", nstore)

        def _var_i(name: str, dims: tuple[str, ...], data: np.ndarray) -> None:
            v = ds.createVariable(name, "i4", dims)
            v[:] = np.asarray(data, dtype=np.int32)

        def _var_f(name: str, dims: tuple[str, ...], data: np.ndarray) -> None:
            v = ds.createVariable(name, "f8", dims)
            v[:] = np.asarray(data, dtype=np.float64)

        # Scalars.
        _var_i("ns", (), np.asarray(ns))
        _var_i("mpol", (), np.asarray(int(wout.mpol)))
        _var_i("ntor", (), np.asarray(int(wout.ntor)))
        _var_i("nfp", (), np.asarray(int(wout.nfp)))
        _var_i("signgs", (), np.asarray(int(wout.signgs)))
        _var_i("lasym__logical__", (), np.asarray(int(bool(wout.lasym))))

        _var_f("wb", (), np.asarray(float(wout.wb)))
        _var_f("volume_p", (), np.asarray(float(wout.volume_p)))
        _var_f("gamma", (), np.asarray(float(wout.gamma)))
        _var_f("wp", (), np.asarray(float(wout.wp)))
        _var_f("fsqr", (), np.asarray(float(wout.fsqr)))
        _var_f("fsqz", (), np.asarray(float(wout.fsqz)))
        _var_f("fsql", (), np.asarray(float(wout.fsql)))

        # Mode tables.
        _var_i("xm", ("mnmax",), np.asarray(wout.xm))
        _var_i("xn", ("mnmax",), np.asarray(wout.xn))
        _var_i("xm_nyq", ("mnmax_nyq",), np.asarray(wout.xm_nyq))
        _var_i("xn_nyq", ("mnmax_nyq",), np.asarray(wout.xn_nyq))

        # Geometry coefficients (full mesh).
        _var_f("rmnc", ("ns", "mnmax"), np.asarray(wout.rmnc))
        _var_f("rmns", ("ns", "mnmax"), np.asarray(wout.rmns))
        _var_f("zmnc", ("ns", "mnmax"), np.asarray(wout.zmnc))
        _var_f("zmns", ("ns", "mnmax"), np.asarray(wout.zmns))
        _var_f("lmnc", ("ns", "mnmax"), np.asarray(wout.lmnc))
        _var_f("lmns", ("ns", "mnmax"), np.asarray(wout.lmns))

        # Flux functions / profiles.
        _var_f("phipf", ("ns",), np.asarray(wout.phipf))
        _var_f("chipf", ("ns",), np.asarray(wout.chipf))
        _var_f("phips", ("ns",), np.asarray(wout.phips))
        _var_f("iotaf", ("ns",), np.asarray(wout.iotaf))
        _var_f("iotas", ("ns",), np.asarray(wout.iotas))

        # Nyquist Fourier fields.
        _var_f("gmnc", ("ns", "mnmax_nyq"), np.asarray(wout.gmnc))
        _var_f("gmns", ("ns", "mnmax_nyq"), np.asarray(wout.gmns))
        _var_f("bsupumnc", ("ns", "mnmax_nyq"), np.asarray(wout.bsupumnc))
        _var_f("bsupumns", ("ns", "mnmax_nyq"), np.asarray(wout.bsupumns))
        _var_f("bsupvmnc", ("ns", "mnmax_nyq"), np.asarray(wout.bsupvmnc))
        _var_f("bsupvmns", ("ns", "mnmax_nyq"), np.asarray(wout.bsupvmns))

        _var_f("bsubumnc", ("ns", "mnmax_nyq"), np.asarray(wout.bsubumnc))
        _var_f("bsubumns", ("ns", "mnmax_nyq"), np.asarray(wout.bsubumns))
        _var_f("bsubvmnc", ("ns", "mnmax_nyq"), np.asarray(wout.bsubvmnc))
        _var_f("bsubvmns", ("ns", "mnmax_nyq"), np.asarray(wout.bsubvmns))

        _var_f("bmnc", ("ns", "mnmax_nyq"), np.asarray(wout.bmnc))
        _var_f("bmns", ("ns", "mnmax_nyq"), np.asarray(wout.bmns))

        # 1D radial fields.
        _var_f("vp", ("ns",), np.asarray(wout.vp))
        _var_f("pres", ("ns",), np.asarray(pres_pa))
        _var_f("presf", ("ns",), np.asarray(presf_pa))

        # Iteration trace (optional).
        _var_f("fsqt", ("nstore_seq",), np.asarray(wout.fsqt))


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
    """Build a :class:`~vmec_jax.state.VMECState` from `wout` Fourier coefficients.

    Notes
    -----
    VMEC's ``wout`` files do **not** store the internal lambda coefficients in the
    same units VMEC uses in ``bcovar`` / ``totzsps``.

    In ``wrout.f`` VMEC writes (schematically, for each radial surface ``js``)::

        lmns_wout(:,js) = (lmns_internal(:,js) / phipf(js)) * lamscale

    to preserve an older output convention.

    For parity-style kernels that re-use VMEC's ``bcovar`` formulas, we therefore
    invert this scaling when constructing the state:

        lmns_internal = lmns_wout * phipf / lamscale
    """
    assert_main_modes_match_wout(wout=wout)
    layout = StateLayout(ns=wout.ns, K=int(wout.xm.size), lasym=bool(wout.lasym))

    # Reconstruct VMEC's internal lambda coefficients from the `wout` convention.
    # See `VMEC2000/Sources/Input_Output/wrout.f` (comment: "IF B^v ~ phip + lamu,
    # MUST DIVIDE BY phipf(js) below to maintain old-style format").
    from .field import lamscale_from_phips

    ns = int(wout.ns)
    if ns < 2:
        s = np.asarray([0.0], dtype=float)
    else:
        s = np.linspace(0.0, 1.0, ns, dtype=float)
    lamscale = float(np.asarray(lamscale_from_phips(wout.phips, s)))
    if lamscale == 0.0:
        lam_scale = np.zeros((ns,), dtype=float)
    else:
        lam_scale = np.asarray(wout.phipf, dtype=float) / lamscale  # (ns,)

    # VMEC writes lambda in a backward-compatible *half-mesh* convention (wrout.f),
    # which is not the internal full-mesh representation used by `totzsps`/`bcovar`.
    # We reproduce VMEC's own recovery logic from `load_xc_from_wout.f`:
    #   - undo the half-mesh interpolation (recurrence in `js`)
    #   - multiply by `phipf(js)` (undo old-style division)
    #   - divide by `lamscale` (undo old-style multiply)
    #
    # This yields lambda coefficients that are consistent with VMEC's internal
    # `bcovar` formulas when used with our `lamscale` scaling.
    def _lambda_full_from_wout(*, lam_wout: np.ndarray, m_modes: np.ndarray, phipf: np.ndarray, lamscale: float) -> np.ndarray:
        lam_wout = np.asarray(lam_wout, dtype=float)
        if lam_wout.ndim != 2 or lam_wout.shape[0] != ns:
            raise ValueError("Expected lam_wout with shape (ns, K)")
        m_modes = np.asarray(m_modes, dtype=int)
        if m_modes.ndim != 1 or m_modes.shape[0] != lam_wout.shape[1]:
            raise ValueError("Expected m_modes with shape (K,)")
        phipf = np.asarray(phipf, dtype=float)
        if phipf.shape != (ns,):
            raise ValueError("Expected phipf with shape (ns,)")
        if ns < 2:
            return lam_wout.copy()

        hs = float(s[1] - s[0])
        # Fortran-style 1-based arrays for sm/sp (profil1d.f).
        sqrts_f = np.zeros((ns + 1,), dtype=float)
        shalf_f = np.zeros((ns + 1,), dtype=float)
        for i in range(1, ns + 1):
            sqrts_f[i] = np.sqrt(max(hs * float(i - 1), 0.0))
            shalf_f[i] = np.sqrt(hs * abs(float(i) - 1.5))
        sqrts_f[ns] = 1.0  # avoid roundoff at boundary

        sm_f = np.zeros((ns + 1,), dtype=float)
        sp_f = np.zeros((ns + 1,), dtype=float)  # sp(0) exists in VMEC but is always 0
        for i in range(2, ns + 1):
            sm_f[i] = shalf_f[i] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
            if i < ns:
                sp_f[i] = shalf_f[i + 1] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
            else:
                sp_f[i] = 1.0 / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        sm_f[1] = 0.0
        sp_f[0] = 0.0
        sp_f[1] = sm_f[2] if ns >= 2 else 0.0

        lam_full = np.zeros_like(lam_wout)
        # Axis initialization (load_xc_from_wout.f).
        m = m_modes
        is_m0 = m == 0
        is_m1 = m == 1
        lam_full[0, is_m0] = lam_wout[1, is_m0]
        denom_m1 = sm_f[2] + sp_f[1]
        if denom_m1 != 0.0:
            lam_full[0, is_m1] = 2.0 * lam_wout[1, is_m1] / denom_m1
        lam_full[0, ~(is_m0 | is_m1)] = 0.0

        # Undo the half-mesh interpolation.
        for mval in range(0, int(np.max(m_modes)) + 1):
            mask = m_modes == mval
            if not np.any(mask):
                continue
            if (mval % 2) == 0:
                for js in range(2, ns + 1):
                    lam_full[js - 1, mask] = 2.0 * lam_wout[js - 1, mask] - lam_full[js - 2, mask]
            else:
                for js in range(2, ns + 1):
                    denom = sm_f[js]
                    if denom == 0.0:
                        lam_full[js - 1, mask] = 0.0
                    else:
                        lam_full[js - 1, mask] = (2.0 * lam_wout[js - 1, mask] - sp_f[js - 1] * lam_full[js - 2, mask]) / denom

        # Undo the old-style `phipf` division and `lamscale` multiply done in `wrout.f`.
        lam_full[1:, :] = lam_full[1:, :] * phipf[1:, None]
        if lamscale != 0.0:
            lam_full = lam_full / float(lamscale)
        return lam_full

    lmns_full = _lambda_full_from_wout(lam_wout=np.asarray(wout.lmns), m_modes=np.asarray(wout.xm), phipf=np.asarray(wout.phipf), lamscale=lamscale)
    lmnc_full = _lambda_full_from_wout(lam_wout=np.asarray(wout.lmnc), m_modes=np.asarray(wout.xm), phipf=np.asarray(wout.phipf), lamscale=lamscale)

    return VMECState(
        layout=layout,
        Rcos=wout.rmnc,
        Rsin=wout.rmns,
        Zcos=wout.zmnc,
        Zsin=wout.zmns,
        Lcos=lmnc_full,
        Lsin=lmns_full,
    )
