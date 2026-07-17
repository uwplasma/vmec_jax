"""VMEC2000-format console output.

Replicates the iteration lines, multigrid stage banners, and termination
messages of VMEC2000 byte-for-column, so ``vmec input.x`` output can be
diffed against ``xvmec2000 input.x`` output structurally.

VMEC2000 counterparts: ``Sources/Input_Output/printout.f`` (iteration lines,
FORMATs 15/25/40/45/50/60/65/70), ``Sources/Initialization_Cleanup/
initialize_radial.f`` (FORMAT 1000, stage banner), ``Sources/TimeStep/
runvmec.f`` (FORMAT 30, force-iterations banner), and ``Sources/TimeStep/
eqsolve.f`` (FORMAT 110, vacuum banner).  The exact format strings are
recorded in Appendix B.

All functions return strings; the caller decides where they go (stdout, the
threed1 file, or a ``jax.debug.callback``).
"""

from __future__ import annotations

FORCE_ITERATIONS_BANNER = (
    " FSQR, FSQZ = Normalized Physical Force Residuals\n"
    " fsqr, fsqz = Preconditioned Force Residuals\n"
    " -----------------------\n"
    " BEGIN FORCE ITERATIONS\n"
    " -----------------------\n"
)


def stage_banner(ns: int, mnmax: int, ftol: float, niter: int) -> str:
    """Multigrid stage banner (initialize_radial.f FORMAT 1000)."""
    return f"\n  NS = {ns:4d} NO. FOURIER MODES = {mnmax:4d} FTOLV = {ftol:10.3E} NITER = {niter:6d}\n"


def vacuum_banner(iteration: int) -> str:
    """Free-boundary vacuum activation message (eqsolve.f FORMAT 110)."""
    return f"\n  VACUUM PRESSURE TURNED ON AT {iteration:4d} ITERATIONS\n"


def screen_header(lasym: bool = False, lfreeb: bool = False) -> str:
    """Column header for the per-iteration screen line (printout.f).

    Byte-exact against captured xvmec2000 output (golden fixtures):
    sym fixed, lasym fixed, and lasym free-boundary variants.
    """
    cols = "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)  "
    if lasym:
        cols += " ZAX(v=0)  "
    cols += "  DELT   "
    cols += "     WMHD      DEL-BSQ" if lfreeb else "    WMHD"
    return "\n" + cols + "\n"


def screen_line(
    iteration: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    r_axis: float,
    delt: float,
    w_mhd: float,
    *,
    z_axis: float | None = None,
    del_bsq: float | None = None,
) -> str:
    """Per-iteration screen line.

    printout.f FORMATs 45 (fixed sym), 50 (free sym), 65/70 (lasym):
    ``(i5,1p,3e10.2[,e11.3],e11.3,e10.2,e12.4[,e11.3])``.
    """
    line = f"{iteration:5d}{fsqr:10.2E}{fsqz:10.2E}{fsql:10.2E}{r_axis:11.3E}"
    if z_axis is not None:
        line += f"{z_axis:11.3E}"
    line += f"{delt:10.2E}{w_mhd:12.4E}"
    if del_bsq is not None:
        line += f"{del_bsq:11.3E}"
    return line + "\n"


def threed1_header(lfreeb: bool = False) -> str:
    """Column header for the threed1-file iteration line (printout.f 15/25)."""
    cols = (
        "  ITER    FSQR      FSQZ      FSQL   "
        "   fsqr      fsqz      fsql      DELT    "
        "RAX(v=0)       WMHD      BETA      <M>"
    )
    if lfreeb:
        cols += "   DEL-BSQ   FEDGE"
    return "\n" + cols + "\n\n"


def threed1_line(
    iteration: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    fsqr_precond: float,
    fsqz_precond: float,
    fsql_precond: float,
    delt: float,
    r_axis: float,
    w_mhd: float,
    beta_vol_avg: float,
    spectral_width: float,
    *,
    del_bsq: float | None = None,
    f_edge: float | None = None,
) -> str:
    """Threed1-file iteration line (printout.f FORMAT 40).

    ``(i6,1x,1p,7e10.2,e11.3,e12.4,e11.3,0p,f7.3,1p,2e9.2)`` — physical and
    preconditioned residuals, time step, axis position, MHD energy, volume
    beta, and spectral width <M>; plus vacuum diagnostics when free-boundary.
    """
    line = (
        f"{iteration:6d} "
        f"{fsqr:10.2E}{fsqz:10.2E}{fsql:10.2E}"
        f"{fsqr_precond:10.2E}{fsqz_precond:10.2E}{fsql_precond:10.2E}"
        f"{delt:10.2E}{r_axis:11.3E}{w_mhd:12.4E}{beta_vol_avg:11.3E}"
        f"{spectral_width:7.3f}"
    )
    if del_bsq is not None and f_edge is not None:
        line += f"{del_bsq:9.2E}{f_edge:9.2E}"
    return line + "\n"


def termination_summary(
    ier_flag: int,
    input_name: str,
    jacobian_resets: int,
    total_time_s: float,
) -> str:
    """Final termination block (fileout.f).

    Prints the ``werror`` message for ``ier_flag``, the case name, the
    Jacobian reset count, and total wall time.
    """
    from .errors import WERROR_MESSAGES

    msg = WERROR_MESSAGES.get(ier_flag, "UNKNOWN TERMINATION CODE")
    return (
        f"\n {msg}\n\n"
        f" FILE : {input_name}\n"
        f" NUMBER OF JACOBIAN RESETS = {jacobian_resets:4d}\n\n"
        f"    TOTAL COMPUTATIONAL TIME (SEC) {total_time_s:12.2f}\n"
    )
