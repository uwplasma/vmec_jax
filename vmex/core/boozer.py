"""Boozer-coordinate transform driver for the new core (§9.4).

Thin, self-contained port of the ``booz_xform_jax`` invocation from the
legacy ``vmex.booz`` module: read a ``wout_*.nc`` file, run the Boozer
transform at the requested resolution, and write a standard ``boozmn_*.nc``
netCDF file (the writer lives inside ``booz_xform_jax`` and follows the
Fortran booz_xform conventions, so downstream tools such as ``booz_plot``
and simsopt can consume the output unchanged).

Public API
----------
``run_booz_xform(wout_path, mbooz=32, nbooz=32, surfaces=None) -> Path``
    Run the transform and return the path of the written ``boozmn`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = ["run_booz_xform", "resolve_boozmn_path"]


# --------------------------------------------------------------------------
# Output-path resolution
# --------------------------------------------------------------------------

def _case_from_wout(path: Path) -> str:
    """Extract the case name from ``wout_<case>.nc`` (or fall back to stem)."""
    name = path.name
    lower = name.lower()
    if lower.startswith("wout_") and lower.endswith(".nc"):
        return path.stem.split("wout_", 1)[-1]
    if lower.startswith("input."):
        return name.split("input.", 1)[-1]
    return path.stem


def resolve_boozmn_path(
    wout_path: str | Path,
    *,
    outdir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Return the ``boozmn_<case>.nc`` path for a WOUT file.

    ``output_path`` wins outright; otherwise the file is placed in ``outdir``
    (default: next to the WOUT file) with the conventional name.
    """
    if output_path is not None:
        return Path(output_path)
    wout = Path(wout_path)
    base = Path(outdir) if outdir is not None else wout.parent
    return base / f"boozmn_{_case_from_wout(wout)}.nc"


# --------------------------------------------------------------------------
# Surface selection
# --------------------------------------------------------------------------

def _surface_indices(bx, surfaces: Sequence[float] | None) -> list[int] | None:
    """Map requested surfaces to half-mesh indices on the input WOUT grid.

    ``None`` keeps booz_xform's default (all surfaces).  Values all inside
    ``[0, 1]`` are treated as normalized toroidal flux ``s`` and matched to
    the nearest half-mesh surface; otherwise values are integer half-mesh
    indices and validated against ``0..ns_in-1``.
    """
    if surfaces is None:
        return None
    if isinstance(surfaces, (int, float)):
        surfaces = (float(surfaces),)
    values = np.asarray(list(surfaces), dtype=float)
    if values.size == 0:
        return None
    ns_in = int(getattr(bx, "ns_in", 0) or 0)
    if ns_in <= 0:
        raise ValueError("Cannot select Boozer surfaces before reading a WOUT file")
    if np.all((0.0 <= values) & (values <= 1.0)):
        s_in = np.asarray(getattr(bx, "s_in", ()), dtype=float)
        if s_in.size != ns_in:
            full = np.linspace(0.0, 1.0, ns_in + 1)
            s_in = 0.5 * (full[:-1] + full[1:])
        return [int(np.argmin(np.abs(s_in - v))) for v in values]
    indices = [int(round(v)) for v in values]
    for index in indices:
        if index < 0 or index >= ns_in:
            raise ValueError(f"Boozer surface index {index} is outside 0..{ns_in - 1}")
    return indices


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_booz_xform(
    wout_path: str | Path,
    mbooz: int = 32,
    nbooz: int = 32,
    surfaces: Iterable[float] | None = None,
    *,
    output_path: str | Path | None = None,
    outdir: str | Path | None = None,
    jit: bool = False,
    verbose: bool = False,
) -> Path:
    """Run ``booz_xform_jax`` on a WOUT file and write ``boozmn_*.nc``.

    Parameters
    ----------
    wout_path:
        Path to a VMEC ``wout_*.nc`` file (new-core or VMEC2000 output).
    mbooz, nbooz:
        Poloidal / toroidal resolution of the Boozer spectrum.
    surfaces:
        Optional surfaces to transform: values in ``[0, 1]`` select the
        nearest half-mesh surfaces by normalized flux, larger values are
        half-mesh indices.  ``None`` transforms every half-mesh surface.
    output_path, outdir:
        Explicit output file, or directory for the default
        ``boozmn_<case>.nc`` name (default: alongside the WOUT file).
    jit:
        Forwarded to ``Booz_xform.run`` (JIT-compiled transform kernels).
    verbose:
        Emit booz_xform progress output.

    Returns
    -------
    Path to the written ``boozmn`` netCDF file.
    """
    try:
        from booz_xform_jax import Booz_xform
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Boozer transforms require booz_xform_jax; "
            "run `pip install booz_xform_jax`."
        ) from exc

    wout = Path(wout_path).expanduser()
    if not wout.exists():
        raise FileNotFoundError(f"WOUT file not found: {wout}")
    out = resolve_boozmn_path(wout, outdir=outdir, output_path=output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tuple_surfaces = None if surfaces is None else tuple(
        float(v) for v in np.ravel(np.asarray(surfaces, dtype=float))
    )
    bx = Booz_xform(verbose=1 if verbose else 0, mboz=int(mbooz), nboz=int(nbooz))
    bx.read_wout(str(wout), flux=False)
    bx.compute_surfs = _surface_indices(bx, tuple_surfaces)
    bx.run(jit=bool(jit))
    bx.write_boozmn(str(out))
    return out
