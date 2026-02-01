"""Visualization helpers (VTK export, field-line traces).

This module is intentionally dependency-free: it writes a small subset of VTK XML
formats in ASCII for easy inspection in ParaView:

- ``.vts``: VTK XML StructuredGrid (good for (theta,zeta) surfaces)
- ``.vtp``: VTK XML PolyData (good for field-line polylines)

The functions here are **not** performance critical; they are meant for examples
and debugging. They operate on NumPy arrays (not JAX) on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _vtk_float_array(name: str | None, data: np.ndarray, *, n_comp: int) -> str:
    data = np.asarray(data)
    if n_comp == 1:
        flat = data.reshape(-1)
    else:
        flat = data.reshape(-1, n_comp)
    # ASCII output for simplicity.
    if n_comp == 1:
        body = " ".join(f"{x:.16e}" for x in flat)
    else:
        body = "\n".join(" ".join(f"{x:.16e}" for x in row) for row in flat)
    name_attr = f' Name="{name}"' if name is not None else ""
    return (
        f'<DataArray type="Float64"{name_attr} NumberOfComponents="{n_comp}" format="ascii">\n'
        f"{body}\n"
        "</DataArray>\n"
    )


def _vtk_int_array(name: str, data: np.ndarray) -> str:
    data = np.asarray(data, dtype=np.int32).reshape(-1)
    body = " ".join(str(int(x)) for x in data)
    return (
        f'<DataArray type="Int32" Name="{name}" NumberOfComponents="1" format="ascii">\n'
        f"{body}\n"
        "</DataArray>\n"
    )


def write_vts_structured_grid(
    path: str | Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    point_data: Dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a VTK XML StructuredGrid (``.vts``) from point coordinates.

    Parameters
    ----------
    x, y, z:
        Arrays of identical shape ``(nx, ny)`` or ``(nx, ny, nz)``.
        For a single surface, use ``(ntheta, nzeta)``.
    point_data:
        Optional dict of named arrays stored as PointData. Each array must have
        either shape matching ``x`` (scalar) or shape ``x.shape + (3,)`` (vector).
    """
    path = _as_path(path)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f"x,y,z must have identical shapes, got {x.shape}, {y.shape}, {z.shape}")

    if x.ndim == 2:
        nx, ny = x.shape
        nz = 1
    elif x.ndim == 3:
        nx, ny, nz = x.shape
    else:
        raise ValueError(f"x must be 2D or 3D, got shape {x.shape}")

    pts = np.stack([x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")], axis=-1)

    point_data = point_data or {}

    # VTK extents are inclusive indices.
    extent = f"0 {nx-1} 0 {ny-1} 0 {nz-1}"

    pd_xml = ""
    if point_data:
        pieces = []
        for name, arr in point_data.items():
            a = np.asarray(arr)
            if a.shape == x.shape:
                pieces.append(_vtk_float_array(name, a.ravel(order="F"), n_comp=1))
            elif a.shape == x.shape + (3,):
                v = np.stack(
                    [a[..., 0].ravel(order="F"), a[..., 1].ravel(order="F"), a[..., 2].ravel(order="F")], axis=-1
                )
                pieces.append(_vtk_float_array(name, v, n_comp=3))
            else:
                raise ValueError(f"PointData {name!r} has shape {a.shape}, expected {x.shape} or {x.shape+(3,)}")
        pd_xml = "<PointData>\n" + "".join(pieces) + "</PointData>\n"

    pts_xml = _vtk_float_array(None, pts, n_comp=3)

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        f'  <StructuredGrid WholeExtent="{extent}">\n'
        f'    <Piece Extent="{extent}">\n'
        f"{pd_xml}"
        "      <Points>\n"
        f"        {pts_xml}"
        "      </Points>\n"
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    return path


def write_vtp_polyline(
    path: str | Path,
    *,
    points: np.ndarray,
    point_data: Dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a VTK XML PolyData (``.vtp``) containing a single polyline."""
    path = _as_path(path)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {pts.shape}")
    n = int(pts.shape[0])
    if n < 2:
        raise ValueError("polyline requires at least 2 points")

    point_data = point_data or {}
    pd_xml = ""
    if point_data:
        pieces = []
        for name, arr in point_data.items():
            a = np.asarray(arr)
            if a.shape == (n,):
                pieces.append(_vtk_float_array(name, a, n_comp=1))
            elif a.shape == (n, 3):
                pieces.append(_vtk_float_array(name, a, n_comp=3))
            else:
                raise ValueError(f"PointData {name!r} has shape {a.shape}, expected {(n,)} or {(n,3)}")
        pd_xml = "<PointData>\n" + "".join(pieces) + "</PointData>\n"

    # Polyline connectivity for a single line of length n.
    connectivity = np.arange(n, dtype=int)
    offsets = np.asarray([n], dtype=int)

    conn_xml = _vtk_int_array("connectivity", connectivity)
    off_xml = _vtk_int_array("offsets", offsets)
    pts_xml = _vtk_float_array(None, pts, n_comp=3)

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{n}" NumberOfVerts="0" NumberOfLines="1" NumberOfStrips="0" NumberOfPolys="0">\n'
        f"{pd_xml}"
        "      <Points>\n"
        f"        {pts_xml}"
        "      </Points>\n"
        "      <Lines>\n"
        f"        {conn_xml}"
        f"        {off_xml}"
        "      </Lines>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    return path
