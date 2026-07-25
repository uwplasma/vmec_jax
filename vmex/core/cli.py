"""``vmex`` command-line entry point (new core, fixed-boundary).

VMEC2000 counterpart: the ``xvmec2000`` executable driver
(``Sources/TimeStep/vmec.f`` / ``runvmec.f``): parse the input deck, run the
``NS_ARRAY`` multigrid ladder with VMEC2000-format console output, write the
``wout_<case>.nc`` file, and print the ``fileout.f`` termination summary.

The solve path is the clean-room core end to end:
:class:`vmex.core.input.VmecInput` (INDATA or VMEC++-style JSON) ->
:func:`vmex.core.multigrid.solve_multigrid` (fixed boundary) or
:func:`vmex.core.multigrid.solve_free_boundary_multigrid` (free boundary) ->
:func:`vmex.core.wout.wout_from_state` -> :func:`vmex.core.wout.write_wout`,
plus the core plotting (``--plot``) and Boozer (``--booz``) drivers.

Zero-crash policy (§2.5): every failure maps to a typed
:class:`vmex.core.errors.VmecError`; the CLI prints the VMEC2000
``werror`` message plus a one-line hint and exits with the matching
``ier_flag`` code.

Free-boundary routing (``LFREEB = T``):

- a readable mgrid file goes to the full free-boundary ``NS_ARRAY`` ladder
  with the VMEC2000
  console output (``In VACUUM`` block, ``VACUUM PRESSURE TURNED ON`` banner)
  and free-boundary wout metadata (``nextcur``/``extcur``/``curlabel``/
  ``mgrid_mode``);
- a missing mgrid file falls back to a fixed-boundary solve with a warning
  (VMEC2000 behavior, dropped by VMEC++);
- ``MGRID_FILE = 'DIRECT_COILS'`` (or the ``--coils`` flag) builds the external
  field from an ESSOS coils file (``essos.coils.Coils``): the coils are tabulated
  into an in-memory mgrid (``Coils.to_mgrid``) and read back as an
  :class:`vmex.core.mgrid.MgridField`
  (``solve_free_boundary(inp, external_field=mgrid_field)``); requires ESSOS.

Free-boundary output behavior:

- Symmetric and LASYM NESTOR potential and surface-field arrays are exported
  to wout.
- ``LFULL3D1OUT = T`` requests a WOUT after NITER exhaustion for either
  boundary mode.  With the default ``F``, VMEC2000 and VMEX return
  ``ier_flag = 2`` without a WOUT.  Fatal numerical/Jacobian errors never
  produce one.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from importlib import resources
from pathlib import Path

# Suppress noisy C++ logging from XLA/PjRt before any JAX import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from .errors import (
    INPUT_ERROR_FLAG,
    WERROR_MESSAGES,
    VmecError,
    VmecInputError,
)

_TEST_INPUT_NAME = "input.nfp4_QH_warm_start"
_TEST_FTOL = 1.0e-12
_PLOT_AUTO = "__vmex_plot_auto__"


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("vmex")
    except Exception:  # pragma: no cover - metadata unavailable in-place
        return "unknown"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def case_from_input(path: Path) -> str:
    """Case name from an input path (``input.solovev`` -> ``solovev``)."""
    name = path.name
    if name.startswith("input."):
        case = name.split("input.", 1)[-1]
    elif name.startswith("input_"):
        case = name.split("input_", 1)[-1]
    else:
        case = path.stem
    case = case[: -len(".json")] if case.endswith(".json") else case
    return case or path.stem


def resolve_wout_path(*, input_path: Path, outdir: Path | None) -> Path:
    """Default ``wout_<case>.nc`` path for an input file."""
    base_dir = Path(outdir) if outdir is not None else input_path.parent
    return base_dir / f"wout_{case_from_input(input_path)}.nc"


def _is_wout_path(path: Path) -> bool:
    lower = path.name.lower()
    return path.suffix.lower() == ".nc" and not lower.startswith(("boozmn_", "mout_"))


def _is_mout_path(path: Path) -> bool:
    return path.name.lower().startswith("mout_") and path.suffix.lower() == ".nc"


def _is_boozmn_path(path: Path) -> bool:
    return path.name.lower().startswith("boozmn_") and path.suffix.lower() == ".nc"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the ``vmex`` executable (``vmec`` alias)."""
    p = argparse.ArgumentParser(
        prog="vmex",
        description=(
            "vmex equilibrium solver (fixed- and free-boundary core).\n\n"
            "  vmex input.X           — solve (INDATA or VMEC++ JSON), write wout_X.nc\n"
            "  vmex --plot wout_*.nc  — diagnostic plots from a WOUT file\n"
            "  vmex --plot mout_*.nc  — straight-axis mirror diagnostics\n"
            "  vmex --booz wout_*.nc  — run booz_xform_jax, write boozmn_*.nc\n"
            "  vmex --plot boozmn_*.nc— Boozer contour/spectrum plots\n"
            "  vmex --doctor          — installation and JAX backend diagnostics\n"
            "  vmex --test            — run and plot the bundled quick-start case\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help=(
            "VMEC input file (input.* namelist or VMEC++ .json) to solve, or a "
            "wout_*.nc/mout_*.nc/boozmn_*.nc file for --plot/--booz."
        ),
    )
    p.add_argument(
        "--plot",
        metavar="PATH",
        type=str,
        nargs="?",
        const=_PLOT_AUTO,
        default=None,
        help=(
            "Generate plots. With a wout_*.nc file, plot WOUT diagnostics; with a "
            "mout_*.nc file, plot straight-axis mirror diagnostics; with a "
            "boozmn_*.nc file, plot Boozer diagnostics; with an input file, solve "
            "first and plot the resulting WOUT. If PATH is omitted, the positional "
            "input path is used."
        ),
    )
    p.add_argument(
        "--booz",
        action="store_true",
        help=(
            "Run booz_xform_jax after solving an input file, or directly from a "
            "wout_*.nc file, and write boozmn_*.nc."
        ),
    )
    p.add_argument("--mbooz", type=int, default=32, help="Boozer poloidal resolution (default: 32).")
    p.add_argument("--nbooz", type=int, default=32, help="Boozer toroidal resolution (default: 32).")
    p.add_argument(
        "--booz-surfaces",
        type=str,
        default=None,
        help="Boozer surfaces: comma/space-separated normalized s values, or 'all' (default).",
    )
    p.add_argument("--outdir", type=str, default=None, help="Directory for wout/boozmn/figure output (default: alongside the input).")
    p.add_argument("--quiet", action="store_true", help="Silence the VMEC-style stdout.")
    p.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=("cli", "jit"),
        help=(
            "Solver lane: 'cli' (jitted blocks with host residual checks, live "
            "printing, exact-ftol exit) or 'jit' (single lax.while_loop)."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "none", "cpu", "gpu", "cuda", "rocm", "tpu"),
        help=(
            "JAX solve placement: automatic VMEX policy (default), 'none' to "
            "follow JAX, or an explicit platform."
        ),
    )
    p.add_argument("--ftol", type=float, default=None, help="Override the final-stage FTOL_ARRAY tolerance.")
    p.add_argument("--max-iter", type=int, default=None, help="Override the final-stage NITER_ARRAY iteration cap.")
    p.add_argument(
        "--jacobian-retries",
        type=int,
        default=2,
        help=(
            "Best-checkpoint retries after 75 Jacobian resets (default: 2; "
            "0 preserves the VMEC2000 fatal-stop policy)."
        ),
    )
    p.add_argument(
        "--coils",
        metavar="PATH",
        type=str,
        default=None,
        help=(
            "ESSOS coils file (.json or .npz with dofs_curves, dofs_currents, "
            "n_segments, nfp, stellsym) supplying the external field of an "
            "LFREEB = T deck: tabulated via ESSOS into an in-memory mgrid "
            "instead of a standalone mgrid file (pairs with MGRID_FILE = "
            "'DIRECT_COILS'; requires ESSOS)."
        ),
    )
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Print installation, Python, package, and JAX backend diagnostics.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run the bundled input.nfp4_QH_warm_start quick-start case: solve, "
            "write the wout file, and plot it."
        ),
    )
    p.add_argument("--version", action="version", version=f"vmex {_package_version()}")
    return p


def _parse_booz_surfaces(text: str | None):
    """Parse ``--booz-surfaces``: ``None``/'all' -> all, else float list."""
    if text is None:
        return None
    cleaned = text.strip().lower()
    if cleaned in ("", "all"):
        return None
    tokens = [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]
    try:
        return [float(tok) for tok in tokens]
    except ValueError as exc:
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint=f"--booz-surfaces expects 'all' or numbers, got {text!r}",
        ) from exc


# ---------------------------------------------------------------------------
# VMEC2000-format console output around the solve
# ---------------------------------------------------------------------------


def _preamble(case: str, *, time_slice: float = 0.0) -> str:
    """Run header block (vmec.f banner, structural match to xvmec2000)."""
    import platform

    now = time.localtime()
    date = time.strftime("%b %d,%Y", now)
    clock = time.strftime("%H:%M:%S", now)
    return (
        " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
        f"  SEQ =    1 TIME SLICE {float(time_slice):11.4E}\n"
        f"  PROCESSING INPUT.{case}\n"
        f"  THIS IS VMEX, VERSION {_package_version()}\n"
        "  Lambda: Full Radial Mesh. L-Force: hybrid full/half.\n"
        "\n"
        f"  COMPUTER: {platform.node()}   OS: {platform.system()}   "
        f"RELEASE: {platform.release()}   DATE = {date}  TIME = {clock}\n"
    )


def _threed1_summary(wout) -> str:
    """Compact threed1-style equilibrium summary from the WoutData."""
    lines = [
        "",
        f" Aspect Ratio          = {float(wout.aspect):14.6f}",
        f" Plasma Volume         = {float(wout.volume_p):14.6f} [M**3]",
        f" Major Radius          = {float(wout.Rmajor_p):14.6f} [M]",
        f" Minor Radius          = {float(wout.Aminor_p):14.6f} [M]",
        f" Volume Average B      = {float(wout.volavgB):14.6f} [T]",
        f" beta total            = {float(wout.betatotal):14.6E}",
        f" MHD Energy (wb + wp)  = {float(wout.wb) + float(wout.wp):14.6E}",
        "",
    ]
    return "\n".join(lines)


def _timing_block(read_s: float, solve_s: float, wout_s: float) -> str:
    """IO/solver timing breakdown (fileout.f layout)."""
    return (
        f"    TIME TO INPUT/OUTPUT           {read_s + wout_s:12.2f}\n"
        f"       READ IN DATA                {read_s:12.2f}\n"
        f"       WRITE OUT DATA TO WOUT      {wout_s:12.2f}\n"
        f"    TIME IN SOLVER                 {solve_s:12.2f}\n"
    )


# ---------------------------------------------------------------------------
# Solve + wout
# ---------------------------------------------------------------------------


def _read_input(input_path: Path):
    """Parse the deck into a :class:`VmecInput` (typed error on failure)."""
    from .input import VmecInput

    try:
        return VmecInput.from_file(input_path)
    except VmecError:
        raise
    except Exception as exc:
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint=f"{input_path.name}: {exc}",
        ) from exc


#: MGRID_FILE sentinel selecting the direct-coil Biot-Savart external field.
_DIRECT_COILS = "DIRECT_COILS"


class _FreeBoundaryPlan:
    """Resolved free-boundary routing: solver kwargs + wout metadata."""

    def __init__(self, *, solver_kwargs, nextcur=0, extcur=None,
                 mgrid_mode="", curlabel=None):
        self.solver_kwargs = dict(solver_kwargs)
        self.nextcur = int(nextcur)
        self.extcur = extcur
        self.mgrid_mode = str(mgrid_mode)
        self.curlabel = curlabel


def _coils_mgrid_field(path: Path, *, nr: int = 96, nphi: int = 32,
                       nz: int = 96):  # pragma: no cover  (ESSOS-only; unavailable in CI)
    """:class:`~vmex.core.mgrid.MgridField` from an ESSOS coils file.

    vmex keeps no coil code — coils live in ESSOS
    (:class:`essos.coils.Coils`).  This loads the coils from the
    ``essos.coils.Coils.to_json`` layout (``.json``, via
    ``essos.coils.Coils.from_json``) or the same keys in an ``.npz`` archive
    (``dofs_curves`` with shape ``(n_base_coils, 3, 2*order + 1)``,
    ``dofs_currents``, ``n_segments``, ``nfp``, ``stellsym``, optional
    ``currents_scale``), tabulates the coil field onto a cylindrical grid
    spanning the coil bounding box (:meth:`essos.coils.Coils.to_mgrid`), and
    reads it back with vmex's own :func:`~vmex.core.mgrid.read_mgrid` —
    yielding the very same :class:`~vmex.core.mgrid.MgridField` the mgrid-file
    lane produces.  Requires ESSOS (``pip install essos``).
    """
    import tempfile

    import numpy as np

    try:
        from essos.coils import Coils, Curves
    except ImportError as exc:
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint="--coils requires essos (pip install essos)",
        ) from exc

    from .mgrid import MgridField, read_mgrid

    try:
        if path.suffix.lower() == ".npz":
            with np.load(path) as npz:
                data = {key: npz[key] for key in npz.files}
            dofs = np.asarray(data["dofs_curves"], dtype=float)
            currents = np.asarray(data["dofs_currents"], dtype=float).reshape(-1)
            n_segments = int(np.asarray(data["n_segments"]).reshape(-1)[0])
            nfp = int(np.asarray(data.get("nfp", 1)).reshape(-1)[0])
            stellsym = bool(np.asarray(data.get("stellsym", False)).reshape(-1)[0])
            scale = float(np.asarray(data.get("currents_scale", 1.0)).reshape(-1)[0])
            coils = Coils(Curves(dofs, n_segments, nfp, stellsym), currents * scale)
        elif hasattr(Coils, "from_json"):
            coils = Coils.from_json(str(path))
        else:  # legacy ESSOS predating the Coils.from_json classmethod
            from essos.coils import Coils_from_json

            coils = Coils_from_json(str(path))
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint=(
                f"--coils {path.name}: expected an ESSOS coils file (.json/.npz "
                "with dofs_curves, dofs_currents, n_segments, nfp, stellsym): "
                f"{exc}"
            ),
        ) from exc

    # Cylindrical grid spanning the coil bounding box (10% margin); the plasma
    # boundary sits well inside the coils, so this grid brackets it.
    gamma = np.asarray(coils.gamma).reshape(-1, 3)
    r = np.hypot(gamma[:, 0], gamma[:, 1])
    z = gamma[:, 2]
    rpad = 0.1 * (float(r.max()) - float(r.min())) + 1.0e-9
    zpad = 0.1 * (float(z.max()) - float(z.min())) + 1.0e-9
    rmin, rmax = max(1.0e-2, float(r.min()) - rpad), float(r.max()) + rpad
    zmin, zmax = float(z.min()) - zpad, float(z.max()) + zpad

    if not hasattr(coils, "to_mgrid"):
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint=(
                "--coils needs an ESSOS build providing Coils.to_mgrid "
                "(coils->mgrid export); update ESSOS (pip install -U essos)"
            ),
        )

    with tempfile.TemporaryDirectory() as tmp:
        mgrid_path = Path(tmp) / "essos_coils_mgrid.nc"
        coils.to_mgrid(str(mgrid_path), nr=int(nr), nphi=int(nphi), nz=int(nz),
                       rmin=rmin, rmax=rmax, zmin=zmin, zmax=zmax)
        return MgridField.from_mgrid_data(read_mgrid(mgrid_path))


def _free_boundary_plan(args, inp, input_path: Path, *, emit):
    """Free-boundary routing policy (``None`` -> fixed-boundary solve).

    - readable mgrid -> :class:`_FreeBoundaryPlan` with ``mgrid_path`` and
      the wout coil metadata (``nextcur``/``extcur``/``curlabel``/
      ``mgrid_mode``) read from the file;
    - missing mgrid -> warning + fixed-boundary fallback (VMEC2000 policy);
    - ``--coils`` / ``MGRID_FILE = 'DIRECT_COILS'`` -> plan with
      ``external_field`` from :func:`_coils_mgrid_field` (``nextcur = 0``: the
      coil currents live in the coils file, not in EXTCUR).
    """
    import numpy as np

    coils_arg = getattr(args, "coils", None)
    if not bool(inp.lfreeb):
        if coils_arg:
            raise VmecInputError(
                WERROR_MESSAGES[INPUT_ERROR_FLAG],
                hint="--coils requires an LFREEB = T input deck",
            )
        return None

    mgrid_name = str(inp.mgrid_file or "").strip().strip("'\"")
    if coils_arg or mgrid_name.upper() == _DIRECT_COILS:  # pragma: no cover  (ESSOS-only)
        if not coils_arg:
            raise VmecInputError(
                WERROR_MESSAGES[INPUT_ERROR_FLAG],
                hint=(
                    "MGRID_FILE = 'DIRECT_COILS' needs --coils <essos_coils"
                    ".json|.npz>, or use the Python API: solve_free_boundary("
                    "inp, external_field=MgridField.from_mgrid_data(...)) with a "
                    "field tabulated from ESSOS coils (essos.coils.Coils.to_mgrid)"
                ),
            )
        coils_path = Path(coils_arg).expanduser().resolve()
        if not coils_path.exists():
            raise VmecInputError(
                WERROR_MESSAGES[INPUT_ERROR_FLAG],
                hint=f"coils file not found: {coils_path}",
            )
        return _FreeBoundaryPlan(
            solver_kwargs={"external_field": _coils_mgrid_field(coils_path)},
        )

    mgrid = Path(mgrid_name).expanduser()
    if not mgrid.is_absolute():
        mgrid = (input_path.parent / mgrid).resolve()
    if not mgrid.exists():
        emit(
            f" WARNING: mgrid file not found: {mgrid}\n"
            "          proceeding with a FIXED-BOUNDARY solve (VMEC2000 fallback)."
        )
        return None

    from .mgrid import read_mgrid

    data = read_mgrid(mgrid)
    extcur = np.zeros((int(data.nextcur),), dtype=float)
    given = np.atleast_1d(np.asarray(
        inp.extcur if inp.extcur is not None else [], dtype=float)).ravel()
    extcur[: min(given.size, extcur.size)] = given[: extcur.size]
    return _FreeBoundaryPlan(
        solver_kwargs={"mgrid_path": mgrid},
        nextcur=int(data.nextcur), extcur=extcur,
        mgrid_mode=str(data.mgrid_mode), curlabel=tuple(data.coil_groups),
    )


def _stage_overrides(inp, *, ftol: float | None, max_iter: int | None):
    """Final-stage ftol/niter overrides for the multigrid ladder."""
    import numpy as np

    ftol_array = niter_array = None
    n_stages = int(np.atleast_1d(np.asarray(inp.ns_array)).size)
    if ftol is not None:
        ftol_array = np.resize(np.asarray(inp.ftol_array, dtype=float), n_stages)
        ftol_array[-1] = float(ftol)
    if max_iter is not None:
        niter_array = np.resize(np.asarray(inp.niter_array, dtype=np.int64), n_stages)
        niter_array[-1] = int(max_iter)
    return ftol_array, niter_array


def _write_wout_from_result(inp, input_path: Path, result, wout_path: Path,
                            freeb_plan=None):
    """Build the full VMEC2000-compatible wout dataset and write it.

    ``freeb_plan`` (a :class:`_FreeBoundaryPlan`) supplies the free-boundary
    coil metadata (``nextcur``/``extcur``/``mgrid_mode``/``curlabel``).
    """
    import numpy as np

    from .wout import wout_from_state, write_wout

    # fsqt history (wrout.f nstore_seq subsampling of fsqr + fsqz).
    history = np.asarray(result.fsq_history, dtype=float)
    fsqt = None
    if history.size:
        total = history[:, 0] + history[:, 1]
        stride = total.size // 100 + 1
        fsqt = total[stride - 1 :: stride][:100]

    freeb_kwargs = {}
    if freeb_plan is not None:
        freeb_kwargs = dict(
            nextcur=freeb_plan.nextcur, extcur=freeb_plan.extcur,
            mgrid_mode=freeb_plan.mgrid_mode, curlabel=freeb_plan.curlabel,
        )
    wout = wout_from_state(
        inp=inp,
        state=result.state,
        fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
        fsqt=fsqt,
        niter=int(result.iterations),
        converged=bool(result.converged),
        input_extension=case_from_input(input_path),
        vacuum_output=result.vacuum,
        **freeb_kwargs,
    )
    write_wout(wout_path, wout)
    return wout


def _solve_input_file(args, input_path: Path, outdir: Path | None, *, emit) -> int:
    """Full solve pipeline for one input deck (fixed- or free-boundary)."""
    case = case_from_input(input_path)
    verbose = not bool(args.quiet)

    t0 = time.perf_counter()
    inp = _read_input(input_path)
    read_s = time.perf_counter() - t0

    if verbose:
        emit(_preamble(case, time_slice=float(getattr(inp, "time_slice", 0.0))))
    freeb_plan = _free_boundary_plan(args, inp, input_path, emit=emit)
    # VMEC2000 turns off LFREEB when the requested mgrid cannot be opened.
    # Use that effective mode in setup and WOUT metadata, not merely in CLI
    # routing; otherwise a fixed-boundary fallback is mislabeled free-boundary.
    effective_inp = inp
    if bool(getattr(inp, "lfreeb", False)) and freeb_plan is None:
        import dataclasses

        effective_inp = dataclasses.replace(inp, lfreeb=False)

    t1 = time.perf_counter()
    if freeb_plan is not None:
        from .multigrid import solve_free_boundary_multigrid

        ftol_array, niter_array = _stage_overrides(
            inp, ftol=args.ftol, max_iter=args.max_iter)
        result = solve_free_boundary_multigrid(
            inp, ftol_array=ftol_array, niter_array=niter_array,
            verbose=verbose,
            emit=emit,
            raise_on_max_iterations=not bool(
                getattr(inp, "lfull3d1out", False)
            ),
            device=None if args.device == "none" else args.device,
            jacobian_retries=int(args.jacobian_retries),
            **freeb_plan.solver_kwargs,
        )
    else:
        from .multigrid import solve_multigrid

        ftol_array, niter_array = _stage_overrides(inp, ftol=args.ftol, max_iter=args.max_iter)
        result = solve_multigrid(
            effective_inp,
            ftol_array=ftol_array,
            niter_array=niter_array,
            mode=str(args.mode),
            verbose=verbose,
            emit=emit,
            raise_on_max_iterations=not bool(
                getattr(inp, "lfull3d1out", False)
            ),
            device=None if args.device == "none" else args.device,
            release_stage_cache=True,
            jacobian_retries=int(args.jacobian_retries),
        )
    solve_s = time.perf_counter() - t1

    wout_path = resolve_wout_path(input_path=input_path, outdir=outdir)
    wout_path.parent.mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    wout = _write_wout_from_result(effective_inp, input_path, result, wout_path,
                                   freeb_plan=freeb_plan)
    wout_s = time.perf_counter() - t2

    if verbose:
        from .printing import termination_summary

        emit(_threed1_summary(wout), end="")
        emit(
            termination_summary(
                int(result.ier_flag), case, int(result.jacobian_resets),
                time.perf_counter() - t0,
            ),
            end="",
        )
        emit(_timing_block(read_s, solve_s, wout_s), end="")
        emit(f"\n Wrote WOUT file: {wout_path}")

    plot_dir = outdir if outdir is not None else input_path.parent
    if args.plot is not None:
        _plot_wout_file(wout_path, plot_dir, emit=emit, quiet=bool(args.quiet))
    if bool(args.booz):
        _run_booz(
            wout_path, args, plot_dir,
            plot=args.plot is not None, emit=emit, quiet=bool(args.quiet),
        )
    if not bool(result.converged):  # free-boundary NITER exhaustion (wout kept)
        return int(result.ier_flag) or 1
    return 0


# ---------------------------------------------------------------------------
# Plotting / Boozer drivers
# ---------------------------------------------------------------------------


def _plot_wout_file(wout_path: Path, outdir: Path, *, emit, quiet: bool) -> None:
    from .plotting import plot_wout

    if not quiet:
        emit(f" Plotting WOUT {wout_path.name} -> {outdir}/")
    for key, path in plot_wout(wout_path, outdir=outdir).items():
        if not quiet:
            emit(f"   Saved {key}: {path}")


def _plot_mout_file(mout_path: Path, outdir: Path, *, emit, quiet: bool) -> None:
    from vmex.mirror.output import plot_mout

    if not quiet:
        emit(f" Plotting mirror output {mout_path.name} -> {outdir}/")
    for key, path in plot_mout(mout_path, outdir=outdir).items():
        if not quiet:
            emit(f"   Saved {key}: {path}")


def _plot_boozmn_file(boozmn_path: Path, outdir: Path, *, emit, quiet: bool) -> None:
    from .plotting import plot_boozmn

    if not quiet:
        emit(f" Plotting Boozer output {boozmn_path.name} -> {outdir}/")
    for key, path in plot_boozmn(boozmn_path, outdir=outdir).items():
        if not quiet:
            emit(f"   Saved {key}: {path}")


def _run_booz(wout_path: Path, args, outdir: Path, *, plot: bool, emit, quiet: bool) -> Path:
    from .boozer import resolve_boozmn_path, run_booz_xform

    surfaces = _parse_booz_surfaces(args.booz_surfaces)
    boozmn_path = resolve_boozmn_path(wout_path, outdir=outdir)
    if not quiet:
        emit(
            f" Running booz_xform_jax (mbooz={int(args.mbooz)}, nbooz={int(args.nbooz)}, "
            f"surfaces={'all' if surfaces is None else surfaces})"
        )
    boozmn_path = run_booz_xform(
        wout_path,
        mbooz=int(args.mbooz),
        nbooz=int(args.nbooz),
        surfaces=surfaces,
        output_path=boozmn_path,
        verbose=not quiet,
    )
    if not quiet:
        emit(f" Wrote Boozer output: {boozmn_path}")
    if plot:
        _plot_boozmn_file(boozmn_path, outdir, emit=emit, quiet=quiet)
    return boozmn_path


# ---------------------------------------------------------------------------
# Bundled quick-start case (--test)
# ---------------------------------------------------------------------------


def _copy_test_input(outdir: Path) -> Path:
    """Copy the packaged quick-start deck into ``outdir`` (quick-start ftol)."""
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / _TEST_INPUT_NAME
    resource = resources.files("vmex").joinpath("resources", _TEST_INPUT_NAME)
    with resources.as_file(resource) as src:
        shutil.copyfile(src, dst)
    text = dst.read_text()
    replacement = f"FTOL_ARRAY  = {_TEST_FTOL:.0e}"
    new_text, count = re.subn(r"(?im)^\s*FTOL_ARRAY\s*=.*$", f"  {replacement}", text, count=1)
    if count:
        dst.write_text(new_text)
    return dst


def _run_bundled_test(args, parser: argparse.ArgumentParser, *, emit) -> int:
    """Solve + wout + plots for the packaged quick-start case."""
    if args.input is not None:
        parser.error("--test does not take an input path")
    if args.plot is not None:
        parser.error("--test and --plot are mutually exclusive")

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else Path.cwd().resolve() / "vmex_test"
    )
    input_path = _copy_test_input(outdir)
    wout_path = resolve_wout_path(input_path=input_path, outdir=outdir)
    plot_dir = outdir / "figures"
    quiet = bool(args.quiet)

    if not quiet:
        emit("vmec bundled test")
        emit("-----------------")
        emit("Quick-start run of the packaged fixed-boundary QH warm-start input.")
        emit(f"1. Wrote the VMEC input file to: {input_path}")
        emit("2. Solving the equilibrium and writing the WOUT file.")
        emit("")
        emit("Equivalent manual command:")
        emit(f"  vmec {input_path} --outdir {outdir}")
        emit("")

    rc = _solve_input_file(args, input_path, outdir, emit=emit)
    if rc != 0:  # pragma: no cover - _solve_input_file raises instead
        return rc

    if not quiet:
        emit("")
        emit(f"3. Wrote the WOUT file to: {wout_path}")
        emit(f"4. Plotting the WOUT file into: {plot_dir}")
        emit("")
        emit("Equivalent manual plotting command:")
        emit(f"  vmec --plot {wout_path} --outdir {plot_dir}")
        emit("")
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_wout_file(wout_path, plot_dir, emit=emit, quiet=quiet)
    if not quiet:
        emit("")
        emit("Bundled test complete: inspect the input, WOUT file, and figures above.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _dispatch(args, parser: argparse.ArgumentParser, *, emit) -> int:
    if bool(args.doctor):
        if args.input is not None:
            parser.error("--doctor does not take an input path")
        from ..doctor import main as doctor_main

        return int(doctor_main())

    if bool(args.test):
        return _run_bundled_test(args, parser, emit=emit)

    plot_requested = args.plot is not None
    target_arg = args.input
    if args.plot not in (None, _PLOT_AUTO):
        if args.input is not None:
            parser.error("provide the target either as --plot PATH or as a positional input, not both")
        target_arg = args.plot
    if args.plot == _PLOT_AUTO and target_arg is None:
        parser.error("--plot requires a PATH or a positional input")
    if target_arg is None:
        parser.error("provide a VMEC input file, a wout_*.nc/mout_*.nc/boozmn_*.nc file, or --test/--doctor")

    input_path = Path(target_arg).expanduser().resolve()
    if not input_path.exists():
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint=f"input file not found: {input_path}",
        )

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    plot_outdir = outdir if outdir is not None else input_path.parent
    quiet = bool(args.quiet)

    if _is_boozmn_path(input_path):
        if not plot_requested:
            parser.error("boozmn_*.nc inputs are plot-only; use --plot boozmn_*.nc")
        _plot_boozmn_file(input_path, plot_outdir, emit=emit, quiet=quiet)
        return 0

    if _is_mout_path(input_path):
        if not plot_requested:
            parser.error("mout_*.nc inputs are plot-only; use --plot mout_*.nc")
        if bool(args.booz):
            parser.error("Boozer transforms require toroidal wout_*.nc inputs")
        _plot_mout_file(input_path, plot_outdir, emit=emit, quiet=quiet)
        return 0

    if _is_wout_path(input_path):
        if not plot_requested and not bool(args.booz):
            parser.error("wout_*.nc inputs require --plot and/or --booz")
        if plot_requested:
            _plot_wout_file(input_path, plot_outdir, emit=emit, quiet=quiet)
        if bool(args.booz):
            _run_booz(input_path, args, plot_outdir, plot=plot_requested, emit=emit, quiet=quiet)
        return 0

    return _solve_input_file(args, input_path, outdir, emit=emit)


def main(argv: list[str] | None = None) -> int:
    """Run the ``vmec`` command-line entry point (zero-crash)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    emit = print

    try:
        return _dispatch(args, parser, emit=emit)
    except VmecError as err:
        emit(f"\n {err.message}\n")
        if err.hint:
            emit(f" HINT : {err.hint}")
        code = int(err.ier_flag)
        return code if code != 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
