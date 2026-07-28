"""Typed exception taxonomy for vmex (zero-crash policy).

Every physics or input failure maps to one of these exceptions instead of a
crash, a bare traceback, or ``sys.exit``.  Each exception carries the
diagnostic state needed to understand the failure (iteration counters, force
residuals, offending surface).  The CLI catches :class:`VmecError` and prints
the VMEC2000-style termination message from :data:`WERROR_MESSAGES` plus a
one-line remedy hint.

VMEC2000 counterpart: the ``ier_flag`` error codes defined in
``Sources/General/vmec_params.f`` and the ``werror`` message table printed by
``Sources/Input_Output/fileout.f``.  See §2.5.
"""

from __future__ import annotations

from dataclasses import dataclass

# VMEC2000 ier_flag values (Sources/General/vmec_params.f).
NORM_TERM_FLAG = 0
BAD_JACOBIAN_FLAG = 1
MORE_ITER_FLAG = 2
JAC75_FLAG = 4
INPUT_ERROR_FLAG = 5
PHIEDGE_ERROR_FLAG = 7
NS_ERROR_FLAG = 8
MISC_ERROR_FLAG = 9
SUCCESSFUL_TERM_FLAG = 11

# Internal-only loop status.  VMEC2000 has no dedicated ``ier_flag`` for a
# non-finite force evaluation; callers still receive ``MISC_ERROR_FLAG`` via
# :class:`VmecNumericalError`, while this distinct carry value lets
# ``solver._finalize`` distinguish NaN/Inf from a Jacobian-retry failure.
NONFINITE_FLAG = 90

# Internal-only eqsolve control transfer.  VMEC2000 communicates this as
# ``irst = 4`` (not an ier_flag): with ``LMOVE_AXIS=T``, a finite first force
# sum above 1e2 returns to eqsolve so ``guess_axis`` can rebuild the initial
# profiles before any momentum step is taken.  A distinct carry status lets
# the jitted VMEX loop make the same host-side control transfer.
AXIS_REGUESS_FLAG = 91

#: VMEC2000 termination messages, keyed by ier_flag
#: (Sources/Input_Output/fileout.f, ``werror`` table).
WERROR_MESSAGES: dict[int, str] = {
    NORM_TERM_FLAG: "EXECUTION TERMINATED NORMALLY",
    BAD_JACOBIAN_FLAG: "INITIAL JACOBIAN CHANGED SIGN!",
    MORE_ITER_FLAG: "MORE ITERATIONS REQUIRED",
    JAC75_FLAG: "MORE THAN 75 JACOBIAN ITERATIONS (DECREASE DELT)",
    INPUT_ERROR_FLAG: "ERROR READING INPUT FILE OR NAMELIST",
    PHIEDGE_ERROR_FLAG: "PHIEDGE HAS WRONG SIGN IN VACUUM REGION",
    NS_ERROR_FLAG: "NS ARRAY MUST NOT BE ALL ZEROES",
    MISC_ERROR_FLAG: "ERROR IN INPUT VALUES",
    SUCCESSFUL_TERM_FLAG: "EXECUTION TERMINATED NORMALLY",
}


@dataclass
class VmecError(Exception):
    """Base class for all vmex failures.

    Attributes
    ----------
    message:
        Human-readable description (VMEC2000-style where applicable).
    hint:
        One-line remedy suggestion shown by the CLI.
    ier_flag:
        The matching VMEC2000 ``ier_flag`` code, for wout/status parity.
    """

    message: str
    hint: str = ""
    ier_flag: int = MISC_ERROR_FLAG

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


@dataclass
class VmecInputError(VmecError):
    """Invalid or unreadable input (INDATA / JSON / arguments).

    VMEC2000: ``input_error_flag`` paths in ``readin.f``.
    """

    ier_flag: int = INPUT_ERROR_FLAG


@dataclass
class VmecJacobianError(VmecError):
    """The flux-surface Jacobian changed sign and could not be recovered.

    Raised after the VMEC2000 escalation ladder is exhausted (axis re-guess,
    time-step resets at ijacob = 25/50, abort at 75 → ``jac75_flag``).
    VMEC2000: ``Sources/General/jacobian.f`` (irst=2) and ``eqsolve.f``.
    """

    ier_flag: int = JAC75_FLAG
    iteration: int = 0
    jacobian_resets: int = 0
    fsq: tuple[float, float, float] | None = None  # (fsqr, fsqz, fsql)


@dataclass
class VmecConvergenceError(VmecError):
    """The force residuals did not reach ftol within the iteration budget.

    VMEC2000: ``more_iter_flag`` from ``eqsolve.f``.  Carries the residual
    history tail so callers can decide whether to continue (hot restart).
    """

    ier_flag: int = MORE_ITER_FLAG
    iteration: int = 0
    fsq: tuple[float, float, float] | None = None
    ftol: float = 0.0


@dataclass
class VmecNumericalError(VmecError):
    """A force evaluation produced NaN or infinity.

    This is intentionally a fail-fast error: once a non-finite value reaches
    the Richardson momentum state, later iterations cannot diagnose or repair
    its source.  Common first-iteration causes are zero effective toroidal
    flux (``PHIEDGE``/``APHI``), a singular or sign-changing initial geometry,
    and non-finite profile values.
    """

    ier_flag: int = MISC_ERROR_FLAG
    iteration: int = 0
    fsq: tuple[float, float, float] | None = None


@dataclass
class AdjointSolveError(VmecNumericalError):
    """The implicit-adjoint Krylov solve returned an unconverged ``lambda``.

    Raised by the reverse pass of :func:`vmex.core.implicit.solve_implicit`
    (and the multi-RHS pullback) when the GCROT(m, k) adjoint solve exhausts
    its budget with a residual above the acceptance threshold.  An
    unconverged adjoint is a *silently wrong* gradient — plausible magnitude,
    wrong value — so it is never returned: host-eager reverse passes raise
    this typed error; traced reverse passes (a ``jax.jit`` around the whole
    gradient) NaN-poison the adjoint instead, which the optimize drivers'
    finite-gradient guards catch.

    Attributes
    ----------
    iterations:
        Total inner Krylov (Arnoldi) iterations the solve performed.
    residual_norm:
        The true residual norm ``||b - A^T lambda||`` the solve reached.
    tolerance:
        The acceptance threshold it failed to meet
        (``slack * adjoint_tol * ||b||``).

    Remedies: raise ``adjoint_maxiter``/``adjoint_gcrot_m``/
    ``adjoint_gcrot_k`` (more Krylov budget) or loosen ``adjoint_tol``.
    """

    iterations: int = 0
    residual_norm: float = 0.0
    tolerance: float = 0.0


@dataclass
class MgridNotFoundError(VmecError):
    """A free-boundary run referenced an mgrid file that cannot be read.

    The solver catches this and falls back to a fixed-boundary solve with a
    warning (behavior VMEC2000 has and VMEC++ dropped — §2.5); it is
    re-raised only when the caller explicitly requires free-boundary.
    """

    ier_flag: int = INPUT_ERROR_FLAG
    path: str = ""
