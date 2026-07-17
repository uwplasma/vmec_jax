"""MAKEGRID mgrid file IO and differentiable field interpolation.

This module is the clean-core home of the VMEC free-boundary external-field
inputs (§8):

- :class:`MgridData` — an immutable snapshot of a VMEC2000/MAKEGRID mgrid
  netCDF file (grid extents, dimensions, per-coil-group cylindrical field
  tables, coil-group labels, raw currents).
- :func:`read_mgrid` / :func:`write_mgrid` — netCDF round-trip in the exact
  MAKEGRID layout consumed by VMEC2000's ``read_mgrid`` (NETCDF3 classic,
  ``br_001``/``bp_001``/``bz_001``… variables on ``(phi, zee, rad)``).
- :class:`MgridField` — a JAX pytree wrapping the field tables plus external
  currents, providing extcur-scaled trilinear interpolation of
  ``B(r, phi, z)`` that is jit- and grad-compatible.

VMEC2000 counterpart: ``Sources/NESTOR_vacuum/mgrid_mod.f`` (read_mgrid,
becoil).  The IO layer is ported from the legacy parity-proven
``vmex.solvers.free_boundary.mgrid``; the interpolation kernel from
``vmex.external_fields.mgrid_jax``.

ESSOS compatibility (``essos.mgrid.MGrid``, PR#33)
--------------------------------------------------
The netCDF layout is identical (same variable names, dimensions, and
``(phi, zee, rad)`` field ordering), so files written by either package are
readable by both.  Known convention divergences:

- **Label padding**: ESSOS centers coil-group names in 30 characters and
  replaces spaces with underscores (SIMSOPT ``to_mgrid`` convention);
  MAKEGRID left-justifies and pads with blanks.  :func:`write_mgrid` follows
  MAKEGRID (blank padding); :func:`read_mgrid` strips trailing whitespace so
  either style reads back cleanly (underscore-padded ESSOS names keep their
  underscores, exactly as ESSOS' own reader returns them).
- **stringsize**: ESSOS always writes ``stringsize = 30``; :func:`write_mgrid`
  preserves the label width of the data being written (min 30) so round-trips
  of MAKEGRID files with longer labels do not truncate.
- **mgrid_mode**: ESSOS' writer hard-codes ``"N"`` and its reader exposes the
  per-group sums ``br``/``bp``/``bz`` and unit ``raw_coil_cur``;
  :class:`MgridData` keeps the mode and raw currents from the file and leaves
  scaling to :class:`MgridField` via ``extcur``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .errors import MgridNotFoundError

_FIELD_RE = re.compile(r"^(br|bp|bz)_(\d{3})$")

#: Default coil-group label width used by MAKEGRID (and ESSOS).
_STRINGSIZE = 30


@dataclass(frozen=True)
class MgridData:
    """Contents of a VMEC2000/MAKEGRID mgrid netCDF file.

    Field arrays are stored per coil group with shape ``(nextcur, kp, jz,
    ir)`` — the netCDF per-group ``(phi, zee, rad)`` layout stacked along a
    leading coil-group axis.  Fields are *unscaled* coil-group responses; the
    physical field is ``sum_g extcur[g] * B_g`` (``mgrid_mode="S"``, scaled
    per unit current) or the raw per-group field (``mgrid_mode="R"``/"N",
    raw currents baked in — VMEC then divides extcur by ``raw_coil_cur``).

    VMEC2000 counterpart: the module variables filled by ``read_mgrid`` in
    ``Sources/NESTOR_vacuum/mgrid_mod.f``.
    """

    #: Grid extents (meters), inclusive at both ends for R and Z.
    rmin: float
    rmax: float
    zmin: float
    zmax: float
    #: Grid dimensions: ir radial, jz vertical, kp toroidal planes per period.
    ir: int
    jz: int
    kp: int
    #: Number of field periods; the phi grid spans [0, 2*pi/nfp), endpoint
    #: excluded (plane kp would repeat plane 0 of the next period).
    nfp: int
    #: Number of external coil groups (independently scalable currents).
    nextcur: int
    #: "S" (scaled, per unit current), "R" (raw currents), or "N" (none/raw).
    mgrid_mode: str
    #: Coil-group labels, trailing whitespace stripped (len == nextcur).
    coil_groups: tuple[str, ...]
    #: Currents the file was computed with (len == nextcur).
    raw_coil_cur: tuple[float, ...]
    #: Cylindrical field tables, shape (nextcur, kp, jz, ir) each.
    br: np.ndarray
    bp: np.ndarray
    bz: np.ndarray

    def __post_init__(self) -> None:
        expected = (int(self.nextcur), int(self.kp), int(self.jz), int(self.ir))
        for name in ("br", "bp", "bz"):
            arr = getattr(self, name)
            if tuple(np.shape(arr)) != expected:
                raise ValueError(f"mgrid {name} shape {np.shape(arr)} != expected {expected}")
        if len(self.coil_groups) != int(self.nextcur):
            raise ValueError(
                f"coil_groups length {len(self.coil_groups)} != nextcur {self.nextcur}"
            )
        if len(self.raw_coil_cur) != int(self.nextcur):
            raise ValueError(
                f"raw_coil_cur length {len(self.raw_coil_cur)} != nextcur {self.nextcur}"
            )


def _decode_char_scalar(x: Any) -> str:
    """Decode a netCDF char scalar/1-D char array to a stripped Python str."""

    arr = np.asarray(x)
    if arr.dtype.kind == "S":
        if arr.ndim == 0:
            return arr.tobytes().decode(errors="ignore").rstrip("\x00").strip()
        return b"".join(arr.reshape(-1).tolist()).decode(errors="ignore").rstrip("\x00").strip()
    if arr.dtype.kind == "U":
        if arr.ndim == 0:
            return str(arr.item()).strip()
        return "".join(arr.reshape(-1).tolist()).strip()
    return str(arr).strip()


def read_mgrid(path: str | Path) -> MgridData:
    """Read a VMEC2000/MAKEGRID mgrid netCDF file.

    Raises
    ------
    MgridNotFoundError
        If the file does not exist or is not a readable netCDF dataset
        (host-side check; nothing here is traced).
    """

    try:
        import netCDF4  # noqa: PLC0415 - optional heavy dependency
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("netCDF4 is required for mgrid loading") from exc

    p = Path(path).expanduser()
    if not p.is_file():
        raise MgridNotFoundError(
            message=f"mgrid file not found: {p}",
            hint="check MGRID_FILE in the input deck (path is resolved relative to cwd)",
            path=str(p),
        )
    try:
        ds = netCDF4.Dataset(str(p))
    except Exception as exc:
        raise MgridNotFoundError(
            message=f"mgrid file could not be opened as netCDF: {p} ({exc})",
            hint="the file exists but is not a valid MAKEGRID netCDF mgrid",
            path=str(p),
        ) from exc

    with ds:

        def _scalar(name: str, cast):
            if name not in ds.variables:
                raise KeyError(f"Missing mgrid variable: {name}")
            return cast(np.asarray(ds.variables[name][()]).item())

        ir = _scalar("ir", int)
        jz = _scalar("jz", int)
        kp = _scalar("kp", int)
        nfp = _scalar("nfp", int)
        nextcur = _scalar("nextcur", int)
        rmin = _scalar("rmin", float)
        rmax = _scalar("rmax", float)
        zmin = _scalar("zmin", float)
        zmax = _scalar("zmax", float)

        mode = "S"
        mode_var = ds.variables.get("mgrid_mode")
        if mode_var is not None:
            mode = _decode_char_scalar(mode_var[:]) or "S"

        coil_groups: tuple[str, ...] = tuple(f"group_{i + 1:03d}" for i in range(nextcur))
        cg_var = ds.variables.get("coil_group")
        if cg_var is not None:
            cg = np.asarray(cg_var[:])
            if cg.ndim == 2:
                coil_groups = tuple(_decode_char_scalar(cg[i, :]) for i in range(cg.shape[0]))
            else:
                coil_groups = (_decode_char_scalar(cg),)
            coil_groups = coil_groups[:nextcur]

        raw_cur: tuple[float, ...] = tuple(1.0 for _ in range(nextcur))
        cur_var = ds.variables.get("raw_coil_cur")
        if cur_var is not None:
            raw_cur = tuple(float(v) for v in np.asarray(cur_var[:]).reshape(-1))[:nextcur]

        br = np.zeros((nextcur, kp, jz, ir), dtype=np.float64)
        bp = np.zeros((nextcur, kp, jz, ir), dtype=np.float64)
        bz = np.zeros((nextcur, kp, jz, ir), dtype=np.float64)
        outs = {"br": br, "bp": bp, "bz": bz}
        for name, var in ds.variables.items():
            m = _FIELD_RE.match(name)
            if not m:
                continue
            idx = int(m.group(2)) - 1
            if not (0 <= idx < nextcur):
                continue
            v = np.asarray(var[:], dtype=np.float64)
            if v.shape != (kp, jz, ir):
                raise ValueError(f"{name} shape {v.shape} != expected {(kp, jz, ir)}")
            outs[m.group(1)][idx, :, :, :] = v

    return MgridData(
        rmin=rmin,
        rmax=rmax,
        zmin=zmin,
        zmax=zmax,
        ir=ir,
        jz=jz,
        kp=kp,
        nfp=nfp,
        nextcur=nextcur,
        mgrid_mode=mode,
        coil_groups=coil_groups,
        raw_coil_cur=raw_cur,
        br=br,
        bp=bp,
        bz=bz,
    )


def write_mgrid(path: str | Path, data: MgridData) -> None:
    """Write ``data`` as a MAKEGRID-layout netCDF file (NETCDF3 classic).

    Dimension and variable names match MAKEGRID's ``write_mgrid_nc`` (and
    ESSOS ``MGrid.write``): scalars ``ir/jz/kp/nfp/nextcur/rmin/zmin/rmax/
    zmax``, char ``mgrid_mode(dim_00001)`` and ``coil_group(
    external_coil_groups, stringsize)``, ``raw_coil_cur(external_coils)``,
    and per-group ``br_XXX/bp_XXX/bz_XXX(phi, zee, rad)``.

    Divergence from ESSOS: coil-group labels are blank-padded (MAKEGRID
    style) rather than underscore-padded/centered, ``mgrid_mode`` and
    ``raw_coil_cur`` are taken from ``data`` instead of hard-coded
    ``"N"``/ones, and ``coil_group`` is always written 2-D (ESSOS writes a
    1-D char vector when there is a single group; both forms are accepted by
    :func:`read_mgrid`, ESSOS, and VMEC2000).
    """

    try:
        import netCDF4  # noqa: PLC0415 - optional heavy dependency
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("netCDF4 is required for mgrid writing") from exc

    nextcur = int(data.nextcur)
    stringsize = max(_STRINGSIZE, *(len(s) for s in data.coil_groups)) if nextcur else _STRINGSIZE

    with netCDF4.Dataset(str(path), "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", stringsize)
        ds.createDimension("external_coil_groups", nextcur)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", nextcur)
        ds.createDimension("rad", int(data.ir))
        ds.createDimension("zee", int(data.jz))
        ds.createDimension("phi", int(data.kp))

        ds.createVariable("ir", "i4")[()] = int(data.ir)
        ds.createVariable("jz", "i4")[()] = int(data.jz)
        ds.createVariable("kp", "i4")[()] = int(data.kp)
        ds.createVariable("nfp", "i4")[()] = int(data.nfp)
        ds.createVariable("nextcur", "i4")[()] = nextcur
        ds.createVariable("rmin", "f8")[()] = float(data.rmin)
        ds.createVariable("rmax", "f8")[()] = float(data.rmax)
        ds.createVariable("zmin", "f8")[()] = float(data.zmin)
        ds.createVariable("zmax", "f8")[()] = float(data.zmax)

        mode = ds.createVariable("mgrid_mode", "S1", ("dim_00001",))
        mode[:] = np.array([(data.mgrid_mode or "S")[:1].encode()], dtype="S1")

        cg = ds.createVariable("coil_group", "S1", ("external_coil_groups", "stringsize"))
        labels = np.array(
            [s[:stringsize].ljust(stringsize).encode() for s in data.coil_groups],
            dtype=f"S{stringsize}",
        )
        cg[:] = labels.view("S1").reshape(nextcur, stringsize)

        raw = ds.createVariable("raw_coil_cur", "f8", ("external_coils",))
        raw[:] = np.asarray(data.raw_coil_cur, dtype=np.float64)

        for i in range(nextcur):
            tag = f"_{i + 1:03d}"
            for name, arr in (("br", data.br), ("bp", data.bp), ("bz", data.bz)):
                var = ds.createVariable(name + tag, "f8", ("phi", "zee", "rad"))
                var[:, :, :] = np.asarray(arr[i], dtype=np.float64)


def _interpolate_bfield(
    br: Any,
    bp: Any,
    bz: Any,
    *,
    extcur: Any,
    r: Any,
    phi: Any,
    z: Any,
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    nfp: int,
) -> tuple[Any, Any, Any]:
    """Extcur-weighted trilinear interpolation of ``(nextcur, kp, jz, ir)`` tables.

    Pure jnp; R and Z are clamped to the grid, phi is periodic with period
    ``2*pi/nfp``.  Ported unchanged (modulo the vmec_kv sampling path, which
    belongs to the NESTOR becoil driver, not the generic field) from the
    legacy ``vmex.external_fields.mgrid_jax.interpolate_mgrid_bfield_jax``.
    """

    shape = tuple(jnp.shape(br))
    _, kp, jz, ir = (int(v) for v in shape)
    rr, pp, zz = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
    out_shape = rr.shape
    r_flat = jnp.clip(jnp.reshape(rr, (-1,)), float(rmin), float(rmax))
    z_flat = jnp.clip(jnp.reshape(zz, (-1,)), float(zmin), float(zmax))

    fr = (r_flat - float(rmin)) * ((ir - 1) / (float(rmax) - float(rmin)))
    fz = (z_flat - float(zmin)) * ((jz - 1) / (float(zmax) - float(zmin)))
    i0 = jnp.clip(jnp.floor(fr).astype(jnp.int32), 0, ir - 2)
    j0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, jz - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    wr = fr - i0
    wz = fz - j0

    period = (2.0 * jnp.pi) / max(1, int(nfp))
    phi_flat = jnp.mod(jnp.reshape(pp, (-1,)), period)
    fk = phi_flat * (kp / period)
    k_floor = jnp.floor(fk)
    k0 = k_floor.astype(jnp.int32) % kp
    k1 = (k0 + 1) % kp
    wk = fk - k_floor

    w0r = 1.0 - wr
    w0z = 1.0 - wz
    w0k = 1.0 - wk
    cur = jnp.reshape(jnp.asarray(extcur), (-1, 1))

    def interp_one(field: Any) -> Any:
        f = jnp.asarray(field)
        v000 = f[:, k0, j0, i0]
        v001 = f[:, k0, j0, i1]
        v010 = f[:, k0, j1, i0]
        v011 = f[:, k0, j1, i1]
        v100 = f[:, k1, j0, i0]
        v101 = f[:, k1, j0, i1]
        v110 = f[:, k1, j1, i0]
        v111 = f[:, k1, j1, i1]
        c00 = v000 * w0r + v001 * wr
        c01 = v010 * w0r + v011 * wr
        c10 = v100 * w0r + v101 * wr
        c11 = v110 * w0r + v111 * wr
        c0 = c00 * w0z + c01 * wz
        c1 = c10 * w0z + c11 * wz
        c = c0 * w0k + c1 * wk
        return jnp.reshape(jnp.sum(cur * c, axis=0), out_shape)

    return interp_one(br), interp_one(bp), interp_one(bz)


@dataclass(frozen=True)
class MgridField:
    """Extcur-scaled trilinear mgrid field ``B(r, phi, z)`` (pure JAX).

    A pytree: the field tables and ``extcur`` are differentiable leaves;
    grid extents and ``nfp`` are static metadata, so instances can flow
    through ``jax.jit``/``jax.grad`` (differentiable w.r.t. field values and
    ``extcur`` everywhere, and w.r.t. coordinates away from cell boundaries).

    VMEC2000 counterpart: ``becoil`` in ``Sources/NESTOR_vacuum/mgrid_mod.f``
    (which samples the kp planes directly; this class interpolates
    periodically in phi, matching the legacy generic path).
    """

    br: Any
    bp: Any
    bz: Any
    extcur: Any
    rmin: float
    rmax: float
    zmin: float
    zmax: float
    nfp: int

    @classmethod
    def from_mgrid_data(cls, data: MgridData, extcur: Any | None = None) -> "MgridField":
        """Build a field from :class:`MgridData`; defaults extcur to raw currents."""

        cur = data.raw_coil_cur if extcur is None else extcur
        cur_arr = jnp.atleast_1d(jnp.asarray(cur, dtype=jnp.float64)).reshape(-1)
        if int(cur_arr.shape[0]) != int(data.nextcur):
            raise ValueError(
                f"extcur length {int(cur_arr.shape[0])} does not match nextcur {data.nextcur}"
            )
        return cls(
            br=jnp.asarray(data.br),
            bp=jnp.asarray(data.bp),
            bz=jnp.asarray(data.bz),
            extcur=cur_arr,
            rmin=float(data.rmin),
            rmax=float(data.rmax),
            zmin=float(data.zmin),
            zmax=float(data.zmax),
            nfp=int(data.nfp),
        )

    @classmethod
    def from_file(cls, path: str | Path, extcur: Any | None = None) -> "MgridField":
        """Read ``path`` (raising :class:`MgridNotFoundError` if missing) and build a field."""

        return cls.from_mgrid_data(read_mgrid(path), extcur=extcur)

    def b_cyl(self, r: Any, phi: Any, z: Any) -> tuple[Any, Any, Any]:
        """Return ``(B_r, B_phi, B_z)`` at cylindrical points (broadcastable)."""

        return _interpolate_bfield(
            self.br,
            self.bp,
            self.bz,
            extcur=self.extcur,
            r=r,
            phi=phi,
            z=z,
            rmin=self.rmin,
            rmax=self.rmax,
            zmin=self.zmin,
            zmax=self.zmax,
            nfp=self.nfp,
        )

    def __call__(self, r: Any, phi: Any, z: Any) -> tuple[Any, Any, Any]:
        """Alias for :meth:`b_cyl`."""

        return self.b_cyl(r, phi, z)


# Pytree registration: field tables and extcur are leaves, geometry is static.
jax.tree_util.register_dataclass(
    MgridField,
    data_fields=["br", "bp", "bz", "extcur"],
    meta_fields=["rmin", "rmax", "zmin", "zmax", "nfp"],
)

__all__ = ["MgridData", "MgridField", "read_mgrid", "write_mgrid"]
