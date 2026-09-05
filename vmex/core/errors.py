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

    Raised host-side, before or during setup, whenever a deck or an API
    argument cannot be turned into a well-posed run: an unparsable or missing
    INDATA namelist or ``_vmex`` JSON block, an unsupported or contradictory
    option, an out-of-range resolution or profile specification, a restart
    ``wout`` that does not match the requested run, and the ``APHI``
    toroidal-flux derivative reversing sign inside ``s`` in ``[0, 1]``
    (:func:`vmex.core.setup.validate_torflux_monotone`).  Nothing here is
    traced, so the check costs a valid deck nothing.

    It adds no attributes of its own beyond pinning ``ier_flag``; the
    ``message`` names the offending key or file and the ``hint`` says what to
    change.

    VMEC2000: ``input_error_flag`` paths in ``readin.f``.

    Attributes
    ----------
    ier_flag:
        Always ``INPUT_ERROR_FLAG`` (VMEC2000 ``ier_flag = 5``).
    """

    ier_flag: int = INPUT_ERROR_FLAG


@dataclass
class VmecJacobianError(VmecError):
    """The flux-surface Jacobian changed sign and could not be recovered.

    Raised after the VMEC2000 escalation ladder is exhausted (axis re-guess,
    time-step resets at ijacob = 25/50, abort at 75 → ``jac75_flag``).
    VMEC2000: ``Sources/General/jacobian.f`` (irst=2) and ``eqsolve.f``.
    Raised from the solver's finalize step for every terminal ``ier_flag``
    that is neither convergence nor a non-finite force, so it also carries
    ``BAD_JACOBIAN_FLAG`` (a sign change already in the initial geometry).
    The usual remedies are a smaller ``DELT`` or a better axis guess.

    Attributes
    ----------
    ier_flag:
        The terminal VMEC2000 code, ``JAC75_FLAG`` (4) for the exhausted
        ladder or ``BAD_JACOBIAN_FLAG`` (1) for an initial sign change; any
        code without a ``WERROR_MESSAGES`` entry is reported as
        ``JAC75_FLAG``.
    iteration:
        The solver iteration counter reached when the run gave up.
    jacobian_resets:
        VMEC2000's ``ijacob``: how many times the Jacobian sign change forced
        a restart with a reduced time step.
    fsq:
        The final ``(fsqr, fsqz, fsql)`` invariant force residuals, or
        ``None`` if unavailable.  These are the dimensionless normalised
        sums of squares of ``getfsq.f``, on the same scale as ``ftol``.
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
    Convergence requires ``fsqr``, ``fsqz`` and ``fsql`` to be at or below
    ``ftol`` simultaneously; this error means the iteration budget
    (``NITER``) ran out first while everything stayed finite.  Comparing
    ``fsq`` against ``ftol`` shows which of the three components -- ``R``,
    ``Z`` or ``lambda`` -- is the laggard.

    Attributes
    ----------
    ier_flag:
        Always ``MORE_ITER_FLAG`` (VMEC2000 ``ier_flag = 2``).
    iteration:
        The iteration counter at which the budget was exhausted.
    ftol:
        The convergence threshold that was in force, dimensionless.
    fsq:
        The final ``(fsqr, fsqz, fsql)`` invariant force residuals of
        ``getfsq.f`` -- normalised sums of squares of the R, Z and lambda
        forces, dimensionless and directly comparable with ``ftol`` -- or
        ``None`` if unavailable.
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

    Internally the loop carries the distinct status ``NONFINITE_FLAG`` so
    ``solver._finalize`` can tell NaN/Inf apart from a Jacobian-retry
    failure, but callers still see the generic ``MISC_ERROR_FLAG`` for
    wout/status parity with VMEC2000, which has no dedicated code for this.
    This class is also the base of the strong-force polish errors.

    Attributes
    ----------
    ier_flag:
        Always ``MISC_ERROR_FLAG`` (VMEC2000 ``ier_flag = 9``).
    iteration:
        The iteration counter at which the non-finite value was detected.
    fsq:
        The last ``(fsqr, fsqz, fsql)`` invariant force residuals of
        ``getfsq.f``, dimensionless, or ``None``.  They are themselves
        typically NaN or Inf by the time this is raised, and the polish
        subclasses never populate this field.
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
class StrongForceContinuationError(VmecNumericalError):
    """The branch-preserving strong-force correction did not reach alpha=1.

    The strong-root polish walks a homotopy parameter ``alpha`` from 0 (the
    unpolished VMEX state) to 1 (the exact strong-force root), so that the
    correction stays on the same equilibrium branch instead of jumping to a
    different one.  This is raised by
    :func:`vmex.core.polish_driver.polish_strong_root` when
    :class:`~vmex.core.polish_driver.PolishConfig` has
    ``fail_policy="raise"`` and the walk stalls, and separately by the
    pseudo-arclength stage when its bordered tangent solve fails.  With
    ``fail_policy="return_unpolished"`` the driver returns the unpolished
    state and its report instead of raising.

    Which raise site fired changes what ``residual_norm`` means, so read it
    together with ``message``.

    Attributes
    ----------
    alpha:
        The continuation parameter reached before the failure, in
        ``[0, 1]``; dimensionless.  A value close to 1 means only the last
        stretch of the branch was unreachable.
    residual_norm:
        For the stalled-continuation raise, the Euclidean norm of the
        nonlinear strong-force solve residual at that ``alpha``.  For the
        tangent-solve raise, the Krylov residual norm of the bordered
        pseudo-arclength linear system.  Both are in the polish solver's
        internal scaled variables, so only their relative size is
        meaningful.
    nonlinear_iterations:
        Total nonlinear (pseudo-transient / Newton) iterations spent across
        all continuation stages.  Left at 0 by the tangent-solve raise.
    linear_iterations:
        Total inner Krylov iterations spent by the tangent and correction
        solves.
    accepted_stages, rejected_stages:
        How many continuation steps were accepted and how many were
        rejected and retried with a smaller ``alpha`` step.  Both are left
        at 0 by the tangent-solve raise.
    iteration, fsq:
        Inherited from :class:`VmecNumericalError` and never populated on
        this path.
    """

    alpha: float = 0.0
    residual_norm: float = 0.0
    nonlinear_iterations: int = 0
    linear_iterations: int = 0
    accepted_stages: int = 0
    rejected_stages: int = 0


@dataclass
class StrongForceCertificationError(VmecNumericalError):
    """Collocation stationarity or its independent force certificate failed.

    ``solver_converged`` distinguishes failure of ``J.T r = 0`` from failure
    of either overintegrated certificate threshold.

    A polished root is accepted on the independent certificate of
    :func:`vmex.core.strong_force.certify_strong_force` -- evaluated on
    quadrature nodes the solve never touched -- and not on the solver's own
    stopping test.  This error is raised when that acceptance check fails
    and :class:`~vmex.core.polish_driver.PolishConfig` has
    ``fail_policy="raise"``; with ``fail_policy="return_unpolished"`` the
    driver returns the unpolished state and its report instead.

    Both polish routes raise it, and they check different things.
    :func:`vmex.core.polish_driver.polish_collocation_least_squares` demands
    all three of ``normalized_l2 <= certificate_tolerance``,
    ``radial_refinement_difference <= radial_refinement_tolerance`` and a
    strictly positive ``minimum_signed_jacobian``, and populates every
    attribute below.  :func:`vmex.core.polish_driver.polish_strong_root` --
    which walks the homotopy to ``alpha = 1`` first and only then certifies
    -- tests ``normalized_l2`` alone and fills only ``normalized_l2`` and
    ``tolerance``, so on that path ``solver_converged`` stays ``False`` and
    the two refinement fields keep their defaults; they say nothing about
    the run.

    Driver failures carry the solver flag and force/quadrature thresholds.
    Derivative eligibility failures carry the recomputed scaled
    ``stationarity_norm`` and its ``stationarity_tolerance``.

    Attributes
    ----------
    solver_converged:
        Whether the Gauss-Newton least-squares solve met its own internal
        stopping test on the collocation stationarity condition
        ``J.T r = 0``.  Diagnostic only: a certified state whose solver
        merely ran out of steps is still accepted.
    normalized_l2:
        The certificate's volume-weighted normalised residual, the
        dimensionless ``StrongForceReport.normalized_l2``.
    tolerance:
        The threshold it was compared against,
        ``PolishConfig.certificate_tolerance``; dimensionless.
    radial_refinement:
        ``StrongForceReport.radial_refinement_difference``: the relative
        change in the volume residual under radial quadrature refinement,
        dimensionless.  A large value means the residual is
        quadrature-limited rather than converged.
    radial_refinement_tolerance:
        The threshold for ``radial_refinement``,
        ``PolishConfig.radial_refinement_tolerance``; dimensionless.
    iteration, fsq:
        Inherited from :class:`VmecNumericalError` and never populated on
        this path.
    """

    solver_converged: bool = False
    normalized_l2: float = float("inf")
    tolerance: float = 0.0
    radial_refinement: float = float("inf")
    radial_refinement_tolerance: float = 0.0
    stationarity_norm: float = float("inf")
    stationarity_tolerance: float = 0.0


@dataclass
class StrongForceLinearSolveError(VmecNumericalError):
    """A polished-root tangent or adjoint Krylov solve did not converge.

    Raised by the derivative lanes of :mod:`vmex.core.polish_implicit`, which
    apply the implicit-function tangent and adjoint of the rectangular
    collocation stationarity equation.  An unconverged solve is a silently
    wrong derivative -- plausible magnitude, wrong value -- so it is never
    returned: with :class:`~vmex.core.polish_implicit.PolishLinearConfig`
    ``fail_policy="raise"`` (the default) the host-eager path raises this,
    and with ``fail_policy="nan"``, or whenever the solve is traced, the
    result is NaN-poisoned instead so the optimize drivers' finite-gradient
    guards catch it.

    Convergence is judged on the *true* residual ``||b - A x||`` recomputed
    after the solve, not on the Krylov method's own estimate.  The usual
    remedies are a larger ``restart``/``max_restarts`` budget, a looser
    ``rtol``/``atol``, or a refreshed polish preconditioner.

    Attributes
    ----------
    solve_kind:
        Which solve failed: ``"least-squares tangent"`` for the forward
        (JVP) lane or ``"least-squares adjoint"`` for the reverse (VJP)
        lane.  The declared default ``"tangent"`` is never used by the
        raise sites.
    iterations:
        Inner Krylov (GMRES) iterations the solve performed.
    residual_norm:
        The true residual norm ``||b - A x||`` reached, in the polish
        solver's internal scaled variables.
    tolerance:
        The acceptance threshold it failed to meet,
        ``max(atol, rtol * ||b||)``, in the same scaled variables.
    iteration, fsq:
        Inherited from :class:`VmecNumericalError` and never populated on
        this path.
    """

    solve_kind: str = "tangent"
    iterations: int = 0
    residual_norm: float = 0.0
    tolerance: float = 0.0


@dataclass
class MgridNotFoundError(VmecError):
    """A free-boundary run referenced an mgrid file that cannot be read.

    The solver catches this and falls back to a fixed-boundary solve with a
    warning (behavior VMEC2000 has and VMEC++ dropped — §2.5); it is
    re-raised only when the caller explicitly requires free-boundary.

    Raised by :func:`vmex.core.mgrid.read_mgrid` in two cases: the resolved
    path is not an existing file, or the file exists but netCDF4 cannot open
    it as a MAKEGRID dataset.  The check is host-side, so nothing here is
    traced.

    Attributes
    ----------
    ier_flag:
        Always ``INPUT_ERROR_FLAG`` (VMEC2000 ``ier_flag = 5``).
    path:
        The expanded filesystem path that was tried, as a string.  Deck
        paths from ``MGRID_FILE`` are resolved relative to the working
        directory, which is the usual reason this differs from what the
        deck says.
    """

    ier_flag: int = INPUT_ERROR_FLAG
    path: str = ""
