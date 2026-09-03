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

import numpy as np

# The screen path prints only the physical residuals (printout.f FORMATs
# 45/50/65/70); the lowercase preconditioned rows exist solely in the threed1
# file (FORMAT 40, :func:`threed1_line`), so the screen legend does not
# mention them.
FORCE_ITERATIONS_BANNER = (
    " FSQR, FSQZ = Normalized Physical Force Residuals\n"
    " -----------------------\n"
    " BEGIN FORCE ITERATIONS\n"
    " -----------------------\n"
)


def _fortran_positional(value: float) -> str:
    """One double in Fortran list-directed style: 17 significant digits,
    positional (no exponent), signed zero preserved — the gfortran
    ``WRITE(*,*)`` rendering of the axis coefficients."""
    value = float(value)
    if value == 0.0:
        return "-0.0000000000000000" if np.signbit(value) else "0.0000000000000000"
    return np.format_float_positional(
        value, precision=17, unique=False, fractional=False, trim="k"
    )


def _fortran_double(value: float) -> str:
    """One double in Fortran list-directed style (the PARVMEC axis report):
    positional with 17 significant digits when ``1e-3 <= |x| < 1e4`` (or
    exactly zero), otherwise 17-significant-digit E-notation with the
    gfortran three-digit exponent (``2.3239183094066352E-002``)."""
    value = float(value)
    if value == 0.0 or 1.0e-3 <= abs(value) < 1.0e4:
        return _fortran_positional(value)
    mantissa, _, exponent = f"{value:.16E}".partition("E")
    return f"{mantissa}E{exponent[0]}{int(exponent[1:]):03d}"


def _axis_line(label: str, values) -> str:
    """One ``      RAXIS_CC = ...`` line in the gfortran list-directed layout:
    positional values right-justified in width-22 fields with 4-space
    separators; E-notation values fill the same 26-column field flush right
    (G-editing puts the trailing blanks inside the field)."""
    line = f"      {label} ="
    for v in np.atleast_1d(np.asarray(values, dtype=float)).ravel():
        text = _fortran_double(v)
        line += (text.rjust(22) + "    ") if len(text) <= 22 else text.rjust(26)
    return line.rstrip()


def improved_axis_block(
    raxis_cc, zaxis_cs, *, raxis_cs=None, zaxis_cc=None
) -> str:
    """PARVMEC-style report of the adopted re-guessed magnetic axis.

    Printed right after ``TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS``
    (``guess_axis.f``) with the axis Fourier coefficients the retry actually
    adopted.  ``raxis_cs``/``zaxis_cc`` add the two LASYM families when
    given (symmetric runs print only ``RAXIS_CC``/``ZAXIS_CS``).
    """
    lines = ["  ---- Improved AXIS Guess ----", _axis_line("RAXIS_CC", raxis_cc)]
    if raxis_cs is not None:
        lines.append(_axis_line("RAXIS_CS", raxis_cs))
    if zaxis_cc is not None:
        lines.append(_axis_line("ZAXIS_CC", zaxis_cc))
    lines.append(_axis_line("ZAXIS_CS", zaxis_cs))
    lines.append("  -----------------------------")
    return "\n".join(lines) + "\n"


def compile_notice(ns: int, *, lane: str | None = None,
                   prefetched: bool = False) -> str:
    """One-line attribution for an XLA compile pause at a rung boundary.

    Emitted when a rung's iteration executable is not already available in
    this process, so cold-start pauses in the console output are
    attributable; ``prefetched`` marks executables built ahead of time by
    the multigrid compile-overlap thread.  ``lane`` tags the free-boundary
    lanes (entry iteration, pre-vacuum loop, NESTOR full/skip updates,
    vacuum turn-on, steady vacuum loop), which compile as several distinct
    programs per rung — a large-grid free run pauses several times, and a
    file-redirected cluster log must show which program each pause builds.
    """
    suffix = " (prefetched)" if prefetched else ""
    what = f"{lane} executable" if lane else "executable"
    return f" compiling NS = {int(ns)} {what}...{suffix}\n"


def emit_flushed(*args, **kwargs) -> None:
    """``print`` that always flushes.

    The CLI's console sink: with stdout redirected to a file (cluster batch
    logs), Python block-buffers ~8 KiB, so an unflushed long free-boundary
    run shows nothing for hours.  Every emitted line must reach the file
    immediately; callers may still pass ``flush=False`` explicitly.
    """
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


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


def polish_banner(
    *,
    mode: str,
    degree: int,
    spans: int | None,
    ns: int,
    tolerance: float,
    certificate_tolerance: float,
    max_iterations: int,
) -> str:
    """Polish-phase opening banner (VMEX-native; no VMEC2000 counterpart).

    Printed in the register of :data:`FORCE_ITERATIONS_BANNER` when the
    optional force-balance polish starts, stating the resolved
    :class:`~vmex.core.polish_driver.PolishConfig`: the request mode, radial
    B-spline degree and spans (``spans=None`` means the resolution-derived
    default of ``lift_high_order_state``), the solve radial resolution
    feeding the lift (``ns``; the driver reports the actual collocation grid
    on its own lines), the Gauss--Newton relative tolerance, the independent
    certificate tolerance, and the nonlinear iteration cap.
    """
    spans_text = "AUTO" if spans is None else f"{int(spans)}"
    return (
        "\n -----------------------\n"
        " BEGIN FORCE POLISHING\n"
        " -----------------------\n"
        f"  MODE = {mode.upper()}  DEGREE = {int(degree)}"
        f"  SPANS = {spans_text}  NS = {int(ns):4d}\n"
        f"  TOL = {float(tolerance):9.3E}"
        f"  CERTIFICATE TOL = {float(certificate_tolerance):9.3E}"
        f"  MAX ITER = {int(max_iterations):4d}\n"
    )


#: Column header for the Gauss--Newton polish rows (:func:`polish_screen_line`).
POLISH_SCREEN_HEADER = (
    "\n  ITER    COST      GRAD     DAMPING     RATIO   LIN-ITS\n"
)


def polish_progress_line(
    *,
    elapsed_seconds: float,
    products: int,
    product_budget: int,
    cost: float,
) -> str:
    """One live heartbeat from inside the jitted Gauss--Newton solve.

    The Gauss--Newton loop is a single ``lax.while_loop``, so its per-step
    table (:func:`polish_screen_line`) can only print once the solve returns.
    At production stellarator resolution that window is hours long, which
    reads as a hang.  This line is emitted from a device callback on the
    matrix-free normal-equation products instead, so the console advances
    while the solve runs.  ``products`` counts those applications and
    ``product_budget`` is ``max_steps * linear_max_steps`` — the worst case,
    not a prediction, so the percentage only ever overstates what remains.
    """

    total = max(int(product_budget), 1)
    done = int(products)
    hours, remainder = divmod(max(float(elapsed_seconds), 0.0), 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    return (
        f"  polish {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        f"  {done}/{total} linear products ({100.0 * done / total:4.1f}%)"
        f"  cost {float(cost):10.3E}\n"
    )


def polish_screen_line(
    iteration: int,
    cost: float,
    gradient_norm: float,
    damping: float,
    *,
    ratio: float | None = None,
    linear_iterations: int | None = None,
    accepted: bool = True,
) -> str:
    """One Gauss--Newton polish row in the screen-line register.

    Row 0 is the initial state and carries no trial diagnostics; later rows
    add the trust ratio, the inner PCG iteration count, and a ``rejected``
    marker for trial steps the trust region refused (their damping still
    adapts, so the row is informative even without an accepted move).
    """
    line = f"{iteration:5d}{cost:10.2E}{gradient_norm:10.2E}{damping:10.2E}"
    if ratio is not None and linear_iterations is not None:
        line += f"{ratio:10.2E}{linear_iterations:8d}"
        if not accepted:
            line += "  rejected"
    return line + "\n"


#: Screen wording for the ``eps_F`` ceiling.  ``F = JxB - grad(p)`` obeys
#: ``|F| <= |JxB| + |grad(p)|`` pointwise, so ``eps_F`` cannot exceed 2 no
#: matter how badly force balance is violated; it reaches 2 wherever the
#: denominator collapses, which is everywhere in vacuum.  Printing the bound
#: next to the number stops a reader from reading "1.99" as a 200% error.
EPS_F_SATURATION_NOTICE = (
    "EPS-F IS BOUNDED BY 2 BY CONSTRUCTION: IT SATURATES WHERE ITS",
    "DENOMINATOR COLLAPSES (VACUUM). THE ROWS BELOW CANNOT SATURATE.",
)


def _measure_row(label: str, initial: float, final: float | None) -> str:
    """One ``label  before -> after`` row, or ``before`` alone."""
    cell = f"{float(initial):10.3E}" if initial == initial else "       n/a"
    if final is None:
        return f"   {label:<30s}{cell}"
    tail = f"{float(final):10.3E}" if final == final else "       n/a"
    return f"   {label:<30s}{cell} -> {tail}"


def force_error_rows(
    measures: tuple[tuple[str, float, float | None], ...],
    *,
    window: tuple[float, float] | None = None,
) -> tuple[str, ...]:
    """Render the non-saturating force-error rows for a certificate block.

    ``measures`` carries ``(label, initial, final)`` triples; pass ``None``
    as ``final`` for a single-state report.  ``window`` appends the flux
    window the volume averages cover, because a volume average over the
    whole domain is dominated by the coordinate-singular axis and the edge.
    A ``nan`` prints as ``n/a`` — the Panici ratio is undefined in vacuum
    and a floored stand-in would read as a real measurement.
    """
    rows = [_measure_row(label, initial, final) for label, initial, final in measures]
    if window is not None and rows:
        rows.append(
            f"   (volume averages over s in "
            f"[{float(window[0]):.2f}, {float(window[1]):.2f}])"
        )
    return tuple(rows)
#: ``termination_reason`` of a polish AUTO declined on predicted cost.  The
#: caller-visible marker for "measured, not attempted, not failed".
POLISH_AUTO_DECLINED = "auto-declined-cost"


def polish_cost_decline(
    *,
    seconds_per_product: float,
    products: int,
    predicted_seconds: float,
    budget_seconds: float,
    chart_size: int,
    residual_rows: int,
) -> str:
    """Explain an AUTO decline in the numbers that produced it.

    An automatic mode that silently spends eleven hours is worse than one
    that does nothing, so the decline states what was measured on this
    machine, what it implies, and every knob that changes the outcome --
    including the one that simply overrules it.
    """

    def clock(seconds: float) -> str:
        if seconds < 90.0:
            return f"{seconds:.3G} s"
        if seconds < 5400.0:
            return f"{seconds / 60.0:.0f} min"
        if seconds < 172800.0:
            return f"{seconds / 3600.0:.1f} h"
        return f"{seconds / 86400.0:.1f} days"

    return (
        "\n POLISH AUTO: DECLINED ON PREDICTED COST\n"
        f"  collocation {int(residual_rows)} rows, {int(chart_size)} unknowns;"
        f" one Gauss-Newton linear product measured at"
        f" {float(seconds_per_product):.3G} s\n"
        f"  the configured iteration limits allow {int(products)} products,"
        f" so the solve could run {clock(float(predicted_seconds))}"
        f" against an AUTO budget of {clock(float(budget_seconds))}\n"
        "  the equilibrium is returned unpolished, unchanged and uncertified\n"
        "  raise the ceiling with !@VMEX POLISH_BUDGET = <seconds>, shorten"
        " the solve with !@VMEX POLISH_MAX_ITER = <n>,\n"
        "  or run it regardless with !@VMEX POLISH = .TRUE.\n"
    )


def polish_certificate_summary(
    initial_l2: float,
    final_l2: float,
    tolerance: float,
    *,
    verdict: str,
    failed_checks: tuple[str, ...] = (),
    measures: tuple[tuple[str, float, float | None], ...] = (),
    window: tuple[float, float] | None = None,
) -> str:
    """Closing certificate block for one polish attempt.

    ``verdict`` is the human-readable outcome (``CERTIFIED``, ``ALREADY
    CERTIFIED``, ``FAILED``); ``failed_checks`` names each independent check
    that rejected the state so a failure is diagnosable from the console
    alone.  ``measures`` carries the non-saturating quantities that make the
    ``eps_F`` pair readable — the dimensional ``<|F|>`` and the
    volume-averaged normalizations — and is printed under an explicit
    statement of the ``eps_F`` ceiling.
    """
    lines = [
        "",
        f" POLISH CERTIFICATE : EPS-F {float(initial_l2):10.3E} ->"
        f" {float(final_l2):10.3E}  (TOLERANCE {float(tolerance):10.3E})",
    ]
    if measures:
        lines.extend(f"   {notice}" for notice in EPS_F_SATURATION_NOTICE)
        lines.extend(force_error_rows(measures, window=window))
    lines.append(f" POLISH {verdict}")
    lines.extend(f"   FAILED CHECK : {check}" for check in failed_checks)
    return "\n".join(lines) + "\n"


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
