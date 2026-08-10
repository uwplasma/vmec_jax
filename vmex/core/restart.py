"""Hot restart from a previous equilibrium (``wout`` file, state, or result).

VMEC++ hot restart seeds a run from a previous ``OutputQuantities`` object,
but requires matching ``mpol/ntor/ns``, drops lambda when the source is a
Fortran wout, and never seeds asymmetric geometry.  This module makes the
restart source a first-class input:

- :func:`state_from_wout` rebuilds the solver's evolved
  :class:`~vmex.core.solver.SpectralState` from **any** VMEC2000-compatible
  ``wout_*.nc`` (VMEX-, VMEC2000- or PARVMEC-written), inverting the exact
  ``wrout.f`` output maps: mode tables are matched by ``(m, n)`` pair
  (zero-fill/truncate on mpol/ntor changes), the half-mesh ``lmns`` is
  inverted back to the full-mesh internal lambda
  (:func:`vmex.core.postprocess.lambda_full_mesh_from_wout`), the m = 1
  ``residue.f90`` constraint is re-applied, and the radial grid is resampled
  with the multigrid ``interp.f`` transfer when ``ns`` differs.
- :func:`restart_state` dispatches any restart source — a path/``WoutData``,
  a previous :class:`~vmex.core.solver.SolveResult`, or a bare
  :class:`~vmex.core.solver.SpectralState` — to one seed state.

The solver entry points (:func:`vmex.core.solver.solve`,
:func:`vmex.core.multigrid.solve_multigrid`,
:func:`vmex.core.multigrid.solve_free_boundary_multigrid`) accept the raw
source via ``restart_from=...``; the multigrid ladders additionally skip
every coarse rung below the restart resolution (see :func:`skip_ladder_rungs`).
Input decks can request the same thing with the VMEX extension key
``RESTART_WOUT = 'wout_x.nc'`` inside ``&INDATA``, and the CLI with
``vmex input.x --restart wout_y.nc`` (the CLI flag wins).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from . import postprocess as _pp
from .errors import VmecInputError

if TYPE_CHECKING:
    from .input import VmecInput
    from .solver import SolveResult, SpectralState
    from .wout import WoutData

__all__ = ["RestartSource", "restart_state", "skip_ladder_rungs", "state_from_wout"]

#: Anything accepted by ``solve*(..., restart_from=...)``.
RestartSource = Union["WoutData", "SpectralState", "SolveResult", str, Path]


def _target_modes(inp: "VmecInput"):
    from .fourier import mode_table

    return mode_table(int(inp.mpol), int(inp.ntor))


def _remap_mode_columns(
    table: np.ndarray | None,
    *,
    xm: np.ndarray,
    xn_over_nfp: np.ndarray,
    lookup: dict[tuple[int, int], int],
    ns: int,
    mnmax: int,
) -> np.ndarray:
    """Map a wout ``(ns, mnmax_src)`` table onto the target mode ordering.

    Modes present in the file but absent from the target ``mpol/ntor`` range
    are truncated; target modes absent from the file stay zero.
    """
    out = np.zeros((ns, mnmax), dtype=float)
    if table is None:
        return out
    table = np.asarray(table, dtype=float)
    for col, (m, n) in enumerate(zip(xm, xn_over_nfp)):
        k = lookup.get((int(m), int(n)))
        if k is not None:
            out[:, k] = table[:, col]
    return out


def state_from_wout(
    wout: "WoutData | str | Path",
    *,
    inp: "VmecInput",
    ns: int | None = None,
) -> "SpectralState":
    """Build a hot-restart :class:`~vmex.core.solver.SpectralState` from a wout.

    ``wout`` is a :class:`~vmex.core.wout.WoutData` or a path to any
    VMEC2000-compatible ``wout_*.nc``.  ``inp`` is the **target** deck: it
    fixes the mode table (``mpol/ntor``; file modes outside it are truncated,
    missing ones zero-filled), the symmetry, and the flux profiles used to
    rebuild the internal lambda normalization.  ``ns`` resamples the radial
    grid with the multigrid ``interp.f`` transfer when it differs from the
    file's ``ns`` (default: keep the file's grid).

    Guardrails: the file's ``nfp`` must equal the deck's, and an asymmetric
    (``lasym``) wout cannot seed a stellarator-symmetric run (the sine-R /
    cosine-Z content has no representation there).  A symmetric wout may seed
    an ``LASYM = T`` run — the asymmetric blocks start at zero.

    The inverse output maps are exact (see the module docstring), so a state
    rebuilt from a converged wout of the same deck re-evaluates to force
    residuals at the file's converged level: restarting re-converges in a
    few iterations instead of the cold count.
    """
    from .fourier import trig_tables
    from .multigrid import interpolate_state
    from .residuals import m1_physical_to_constrained
    from .setup import run_setup
    from .solver import SpectralState, resolution_from_input
    from .transforms import physical_to_internal_scale
    from .wout import read_wout

    if isinstance(wout, (str, Path)):
        path = Path(wout)
        if not path.exists():
            raise VmecInputError(
                f"restart wout file not found: {path}",
                hint="check RESTART_WOUT / --restart against the input location",
            )
        wout = read_wout(path)

    if int(wout.nfp) != int(inp.nfp):
        raise VmecInputError(
            f"restart wout has NFP = {int(wout.nfp)}, the input deck NFP = "
            f"{int(inp.nfp)}",
            hint="a restart source must come from the same field-period family",
        )
    if bool(wout.lasym) and not bool(inp.lasym):
        raise VmecInputError(
            "restart wout is asymmetric (LASYM = T) but the input deck is "
            "stellarator-symmetric",
            hint="set LASYM = T in the deck or restart from a symmetric wout",
        )

    ns_src = int(wout.ns)
    resolution = resolution_from_input(inp, ns=ns_src)
    modes = _target_modes(inp)
    mnmax = int(modes.m.size)
    lookup = {
        (int(m), int(n)): k
        for k, (m, n) in enumerate(zip(modes.m, modes.n))
    }
    xm = np.asarray(wout.xm, dtype=int)
    xn_over_nfp = np.asarray(wout.xn, dtype=int) // max(int(wout.nfp), 1)
    remap = lambda table: _remap_mode_columns(  # noqa: E731
        table, xm=xm, xn_over_nfp=xn_over_nfp, lookup=lookup,
        ns=ns_src, mnmax=mnmax,
    )

    # wout tables -> internal normalization (mscale*nscale divided out).
    scale = physical_to_internal_scale(modes, trig_tables(resolution))[None, :]
    lasym = bool(inp.lasym)
    zeros = np.zeros((ns_src, mnmax), dtype=float)
    R_cos = remap(wout.rmnc) * scale
    Z_sin = remap(wout.zmns) * scale
    R_sin = remap(wout.rmns) * scale if lasym else zeros
    Z_cos = remap(wout.zmnc) * scale if lasym else zeros

    # Half-mesh wout lambda -> full-mesh internal lambda, with the TARGET
    # deck's flux normalization (phipf/lamscale) at the file's radial grid.
    setup = run_setup(inp, resolution, lconm1=True, infer_axis_if_missing=False)
    phipf_internal = np.asarray(setup.phipf, dtype=float)
    lamscale = float(np.asarray(setup.lamscale))
    s_full = np.asarray(setup.s_full, dtype=float)
    m_modes = np.asarray(modes.m, dtype=int)
    L_sin = _pp.lambda_full_mesh_from_wout(
        lmns_half=remap(wout.lmns), m_modes=m_modes, s=s_full,
        phipf_internal=phipf_internal, lamscale=lamscale,
    ) * scale
    L_cos = zeros
    if lasym:
        L_cos = _pp.lambda_full_mesh_from_wout(
            lmns_half=remap(wout.lmnc), m_modes=m_modes, s=s_full,
            phipf_internal=phipf_internal, lamscale=lamscale,
        ) * scale

    # Physical basis -> the evolved m = 1-constrained basis (residue.f90).
    R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
        R_cos, Z_sin, R_sin if lasym else None, Z_cos if lasym else None,
        modes=modes, lthreed=bool(resolution.lthreed), lasym=lasym,
        lconm1=True,
    )
    state = SpectralState(
        R_cos=np.asarray(R_cos), R_sin=np.asarray(R_sin if lasym else zeros),
        Z_cos=np.asarray(Z_cos if lasym else zeros), Z_sin=np.asarray(Z_sin),
        L_cos=L_cos, L_sin=L_sin,
    )
    if ns is not None and int(ns) != ns_src:
        state = interpolate_state(state, ns_fine=int(ns), modes=modes)
    return state


def restart_state(
    source: RestartSource,
    inp: "VmecInput",
    *,
    ns: int | None = None,
) -> "SpectralState":
    """Normalize any ``restart_from`` source to one seed state for ``inp``.

    Accepts a wout path (``str``/``Path``), a parsed
    :class:`~vmex.core.wout.WoutData`, a previous
    :class:`~vmex.core.solver.SolveResult`, or a bare
    :class:`~vmex.core.solver.SpectralState`.  State-like sources must match
    the deck's ``mpol/ntor`` mode count exactly (wout sources are remapped);
    ``ns`` optionally resamples the radial grid (``interp.f`` transfer).
    """
    from .multigrid import interpolate_state
    from .solver import SolveResult, SpectralState
    from .wout import WoutData

    if isinstance(source, (str, Path, WoutData)):
        return state_from_wout(source, inp=inp, ns=ns)
    if isinstance(source, SolveResult):
        source = source.state
    if not isinstance(source, SpectralState):
        raise VmecInputError(
            f"unsupported restart source type: {type(source).__name__}",
            hint="pass a wout path, WoutData, SolveResult, or SpectralState",
        )
    modes = _target_modes(inp)
    mnmax = int(modes.m.size)
    if int(source.R_cos.shape[1]) != mnmax:
        raise VmecInputError(
            f"restart state has {int(source.R_cos.shape[1])} modes, the deck's "
            f"MPOL/NTOR give {mnmax}",
            hint="restart from a wout file to remap the mode table",
        )
    if ns is not None and int(source.R_cos.shape[0]) != int(ns):
        source = interpolate_state(source, ns_fine=int(ns), modes=modes)
    return source


def skip_ladder_rungs(ns_arr: np.ndarray, seed_ns: int) -> int:
    """First multigrid rung to execute when restarting at ``seed_ns``.

    VMEC++ hot restart requires the first ``ns_array`` entry to equal the
    source resolution; VMEX instead **skips** every leading rung whose radial
    resolution the restart source already meets or exceeds — those rungs only
    exist to build a seed the caller already has.  Rungs at or above
    ``seed_ns`` still run (the seed is ``interp.f``-interpolated up), and if
    the seed is finer than the whole ladder only the final rung runs (the
    seed is resampled down to it).
    """
    ns_arr = np.asarray(ns_arr)
    keep = np.flatnonzero(ns_arr >= int(seed_ns))
    return int(keep[0]) if keep.size else int(ns_arr.size - 1)
