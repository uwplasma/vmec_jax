High-order strong-force balance
===============================

VMEX's legacy solver establishes stationarity of the discrete VMEC energy on
a staggered, uniform ``s`` mesh.  A small ``FSQR/FSQZ/FSQL`` is therefore a
certificate for those projected discrete equations; it is not, by itself, a
uniform pointwise certificate for ``J x B - grad(p)``.  The high-order lane
keeps that fast solver as the branch-finding coarse model and adds a continuous
representation and an independent strong-form certificate.

Representation and fixed constraints
------------------------------------

The continuous coordinates are ``(rho, theta, zeta)``, where ``rho=sqrt(s)``
and ``zeta`` advances from zero to ``2*pi`` over one field period.  Physical
cylindrical angle is ``phi=zeta/NFP``.  This is a module-local convention: the
legacy kernel documented in :doc:`spectral-representation` uses the physical
toroidal angle directly.  Each real Fourier amplitude is

.. math::

   X_{mn}(\rho) = \rho^{|m|} q_{mn}(s), \qquad
   q_{mn}(s) = \sum_k c_{kmn} B_k(s).

The local clamped B-splines have odd degree 3, 5, or 7.  ``PolishConfig``
selects degree 3, so a polished state is cubic unless the caller overrides
``radial_degree``; :func:`~vmex.core.strong_force.lift_high_order_state` called
on its own still defaults to degree 5.  The factor ``rho**abs(m)`` is analytic
and is never estimated from sampled surfaces.

The legacy lift first undoes VMEX's ``m=1`` constrained variables and Fourier
normalization.  It then fits ``q`` while imposing these conditions by
construction:

* all ``m>0`` amplitudes vanish with the correct magnetic-axis order;
* the ``m=0`` magnetic-axis value is exact;
* fixed-boundary R and Z coefficients are exact at ``s=1``;
* stellarator-symmetry structural zeros remain zero; and
* the lambda ``(m,n)=(0,0)`` gauge coefficient is absent.

These are affine elimination rules, not penalty terms.  A VMEC-compatible wout
from VMEX, VMEC2000, or VMEC++ enters through the same tested mode-remapping and
lambda inversion used by hot restart.  DESC is used only as an external oracle;
VMEX does not import or depend on DESC.

The legacy radial mesh is first order, so the default reconstruction is an
overdetermined fit with roughly two mesh samples per free spline span, capped
at 32 spans.  An equal-size interpolant reproduces mesh-scale noise exactly and
can turn that noise into very large second derivatives in ``curl(B)`` even when
the sampled surface coordinates look accurate.  Callers with a genuinely
high-order source may supply an explicit ``radial_basis``.

Independent continuum oracle
----------------------------

At arbitrary off-axis points, :mod:`vmex.core.strong_force` constructs the
Cartesian position, covariant basis, metric, signed Jacobian, contravariant and
covariant magnetic field, current, pressure gradient, and finally

.. math::

   \mathbf{F} = \frac{(\nabla\times\mathbf{B})\times\mathbf{B}}{\mu_0}
                - \nabla p.

The field-period coordinate transform is explicit. With
``zeta=NFP*phi`` and ``sqrt(g)`` evaluated in ``(rho,theta,zeta)``, the two
nonzero contravariant components are

.. math::

   B^\theta = \frac{2\rho}{\sqrt{g}}
      \left(\frac{\chi'}{\mathrm{NFP}}-\phi'\partial_\zeta\lambda\right),
   \qquad
   B^\zeta = \frac{2\rho\phi'}{\sqrt{g}}
      \left(1+\partial_\theta\lambda\right).

This form is invariant when an axisymmetric equilibrium is represented with a
different number of field periods.

Spline and Fourier functions are differentiated analytically by JAX.  No
legacy half-mesh force, radial finite difference, or solve collocation value is
reused.  The conventional independent components are

.. math::

   F_\rho = \mathbf{F}\cdot\partial_\rho\mathbf{r}, \qquad
   F_{\mathrm{helical}} =
   \frac{\partial_\theta B_\zeta-\partial_\zeta B_\theta}{\mu_0}.

The certificate grid is disjoint from the solve grid: composite Gauss nodes of
order ``degree+3`` per knot span, angular grids of ``max(8, 4(m_max+1))`` and
``max(4, 4(n_max+1))`` points offset by ``0.5`` and ``0.375`` of a cell, and
float64 throughout.  The volume norms are weighted by
``w = w_rho * (2*pi/ntheta) * (2*pi/nzeta) * abs(sqrt(g))`` -- the
flux-surface profile uses ``abs(sqrt(g))`` alone -- so ``normalized_l2`` is
the volume-weighted root mean square of the pointwise ratio

.. math::

   \varepsilon_F = \frac{2|\mathbf{F}|}
        {|\mathbf{J}\times\mathbf{B}|+|\nabla p|+F_{\mathrm{floor}}},
   \qquad F_{\mathrm{floor}} = 10^{-12}.

The report adds dimensional L2/P99/Linf force density, radial and helical
contributions, near-axis/bulk/edge norms, a flux-surface profile, an angular
spectral tail, a radial-quadrature difference, the signed-Jacobian margin, and
the boundary and gauge residuals.  The quadrature difference re-evaluates the
same knots at Gauss order ``degree+1`` instead of ``degree+3``, so it measures
integration-order consistency and not knot refinement.

Polishing chart and the frozen coordinate gauge
-----------------------------------------------

:func:`~vmex.core.polish.make_strong_structured_chart` builds the solve
coordinates without a global Jacobian or SVD.  It retains the constrained
``R_cos`` channels as the geometry coordinates and the constrained ``L_sin``
channels as the field-line coordinates; ``Z`` is the eliminated
poloidal-coordinate gauge.  The chart requires stellarator symmetry and raises
on ``lasym`` input.

Freezing ``Z`` is a gauge choice, and it is not a free one.  Representing a
vertical displacement ``dZ`` by a poloidal reparametrization alone needs
``delta = dZ / Z_theta``, which is singular wherever ``Z_theta`` vanishes --
the top and bottom of each cross-section.  There the chart simply cannot
produce the normal displacement, so the available correction is essentially
horizontal in the ``(R, Z)`` plane.

The one exception is the ``lconm1`` ``m=1`` constraint.  In a three-dimensional
run that group is a single internal variable that moves ``R_ss`` and ``Z_cs``
together in a fixed one-to-one ratio, so ``Z`` moves only along that constrained
direction; every other chart coordinate is pure ``R_cos`` or pure ``L_sin``.
In an axisymmetric stellarator-symmetric run the constraint is inactive and
``Z`` is frozen exactly.  Whether this gauge sets the observed residual floor
near ``1.8e-3`` has not been tested.

Rectangular collocation residual
--------------------------------

:func:`~vmex.core.polish.strong_collocation_residual` exposes both independent
physical channels at every solve point, with no angular or radial projection.
The solve grid is the tensor product of composite Gauss--Legendre nodes in
``s`` -- order ``max(3, ceil(1.5 * basis_size / spans))`` per knot span, mapped
to ``rho = sqrt(s)`` -- with uniform angular grids of ``max(4*m_max+5, 4)``
poloidal and ``1`` or ``2*|n|_max+3`` toroidal points.  The residual is

.. math::

   r = \frac{2\,|\sqrt{g}|}{D_0}
       \left(f_\rho,\; f_{\mathrm{helical}}\right),

stacked over every grid point, giving ``2 * n_rho * n_theta * n_zeta`` rows
against ``chart.size`` unknowns.  On the bundled shaped tokamak that is 1764
rows and 148 unknowns out of 212 constrained coordinates.

Three properties of this functional matter when reading a polish report.

First, it is **not** the certificate norm.  It carries ``abs(sqrt(g))``
linearly, in the DESC manner, but no quadrature weight: the radial Gauss
weights and the angular cell measures are absent, and a weighted L2 would need
``sqrt(w_rho * dtheta * dzeta * abs(sqrt(g)))`` instead.  Minimizing it is
therefore a different objective from the one that decides acceptance.

Second, the denominator ``D_0`` is frozen at the lifted state and uses a
different floor from the certificate.  It is built once in
:func:`~vmex.core.polish.make_strong_root_runtime` as
``sqrt(|JxB|^2 + f^2) + sqrt(|grad p|^2 + f^2) + f`` with ``f = 1e-30``,
against the certificate's ``1e-12``.  Freezing it means the solve cannot
manufacture a state-dependent near-null direction; the smaller floor means the
solve residual is far more sensitive than the certificate wherever both force
contributions collapse, including vacuum regions.

Third, both channels are signed densities, not ``|F|``.  A signed residual is
differentiable through its own zero, which the certificate's magnitude is not.

The quadrature is defined in normalized flux ``s`` while the oracle accepts
``rho = sqrt(s)``, so the residual evaluates physics at ``sqrt(s_quadrature)``
and fits the regularized amplitudes against the spline basis at
``s_quadrature``.  Passing flux nodes directly as rho samples over-resolves the
edge and under-resolves the axis; a regression pins the identity
``radial_nodes**2 == s_quadrature``.

Column scaling and the Gauss--Newton solve
------------------------------------------

The rows carry one scalar scale: the root-mean-square of the initial residual,
floored at ``1e-12``.  The columns are scaled by a stochastic estimate of their
norms.  ``PolishConfig.collocation_scale_probes`` (default 8) Rademacher
vectors drawn from a fixed ``numpy`` generator seeded at zero are pushed
through the transpose of the linearized residual; the root mean square response
per column estimates that column's norm, floored at
``max(1e-8 * max_norm, 1e-12)``, and the reciprocal becomes the variable scale.
The draws are deterministic, so two runs of the same case scale identically.

SOLVAX then minimizes the scaled residual with matrix-free damped
Gauss--Newton.  Each step solves ``(J^T J + mu I) p = -J^T r`` by conjugate
gradients -- ``linear_rtol=1e-3``, at most
``linear_restart * linear_max_restarts = 600`` iterations, and no
preconditioner supplied -- then accepts or rejects the trial state on a trust
ratio and adapts ``mu``.

The damping is Levenberg (a multiple of the identity), not Marquardt (a
multiple of ``diag(J^T J)``), and starts at ``1e-3``.  ``J`` and ``J^T`` are
JAX transforms of the residual, so no dense Jacobian is formed at any
resolution.  At most ``max_nonlinear_iterations`` steps are taken, 80 by
default.

Acceptance is the certificate, not solver convergence
-----------------------------------------------------

Both drivers accept a polished state only when all three certificate checks pass:
``normalized_l2 <= validation_tolerance`` (default ``1e-2``),
``radial_refinement_difference <= radial_refinement_tolerance`` (default
``1e-3``), and a strictly positive minimum signed Jacobian.  All three metrics
must be finite; both norm/difference metrics must be nonnegative.  When
``validation_tolerance=None``, the force threshold is ``tolerance``.  These
checks apply to early returns as well as final acceptance; nonfinite values
are named in the failure message.  The Gauss--Newton
solver's own relative stationarity tolerance is recorded as
``least_squares_success`` and is a diagnostic only.

This is a deliberate policy, and it has a visible consequence: a state that
merely exhausted its step budget can still be accepted.  The shipped
``shaped_tokamak_pressure`` artifact in
``benchmarks/strong_force_cases_m4.json`` ran all 80 of its 80 permitted
nonlinear iterations and reports ``least_squares_success: false``, yet is
recorded as ``converged: true`` with
``termination_reason: independently-certified``.

It moved ``normalized_l2`` from ``1.28e-2`` to ``1.79e-3``, which is what the
certificate asked for, but the Gauss--Newton iteration had not converged.
Read ``nonlinear_iterations`` against ``max_nonlinear_iterations`` before
treating a polish as converged in the solver sense.

When the certificate fails, ``fail_policy="raise"`` reports a typed
:class:`~vmex.core.errors.StrongForceCertificationError` naming each failed
check.  ``fail_policy="return_unpolished"`` returns the original lifted state
with ``report.converged=False`` and a zero correction.  Neither path reports a
failed attempt as a polished equilibrium.

Low-order operator: transfer, not preconditioner
------------------------------------------------

:func:`~vmex.core.polish.build_low_order_preconditioner` assembles and factors
the exact nearest-neighbour raw-force block system from the implicit
tangent/adjoint path.  In the shipped lane, the part that is load-bearing is
the high/low **transfer** it carries: ``T_HL`` samples every regularized spline
mode on the VMEX full mesh, restores VMEX Fourier normalization and the
internal ``m=1`` packing, and projects onto the evolved legacy degrees of
freedom, while ``T_LH`` fits back.  The chart's layout groups are built from
that transfer, so it defines which native coordinates exist at all.

Its R/Z fit has a structurally zero terminal coefficient, so a correction
cannot move the fixed boundary; symmetry zeros and the lambda gauge are
eliminated rather than penalized.  Tests certify both transfer dualities and
the complete preconditioner duality.
:func:`~vmex.core.polish.preconditioner_quality` measures the true relative
residual ``||A P r-r||/||r||`` on fixed probes; it is a library diagnostic and
the shipped lane does not call it.

The block factors are **not** applied inside the Gauss-Newton solve: the lane
calls SOLVAX with no preconditioner, so its conjugate-gradient step runs
unpreconditioned.  They are applied once during the runtime build, where a
power iteration on the low-order-solved tangent sets the equation and
coordinate scales, and their build time is reported as
``factor_build_seconds`` in every polish report.  The retired square root
below, and the preconditioner tests and benchmarks, apply them directly.

Driver sequence
---------------

:func:`~vmex.core.polish_driver.polish_legacy_solution` is the only entry point
the solver uses.  It refines the converged legacy state with the implicit
Newton anchor, lifts it into the spline basis, and evaluates the independent
certificate.  A state satisfying all three acceptance checks above returns
immediately with ``termination_reason="already-certified"`` and an empty
correction; no chart, factorization, or solve is constructed.

Otherwise it builds the low-order operator, the strong-root runtime, and the
structured chart, and calls
:func:`~vmex.core.polish_driver.polish_collocation_least_squares`.  The
runtime is built with ``balance_full_root=False``, so the full-root Ruiz
equilibration is skipped and only the single strong scale is computed.  The
fixed boundary, profile data, parity, and lambda gauge cannot drift because
they are absent from the free coordinate map.

Implicit derivatives of the polished state
------------------------------------------

The nonlinear solve is not differentiated.  Once the correction ``c`` is
stationary for native data ``q``, :mod:`vmex.core.polish_implicit` applies the
implicit-function theorem to the least-squares stationarity equation

.. math::

   g(c, q) = J(c,q)^{T} r(c,q) = 0, \qquad
   g_c\,\dot c = -g_q\,\dot q,

so a gradient costs one Krylov solve rather than a replay of Gauss--Newton
steps.

Both Jacobian actions are JAX JVPs/VJPs of ``g``; SOLVAX GMRES solves the
tangent and the transposed adjoint system, right-preconditioned by the squared
variable scales already computed for the primal solve.  A true-residual check
runs on every solve and raises a typed
:class:`~vmex.core.errors.StrongForceLinearSolveError`, or poisons the result
with NaN, when the tolerance is missed.  Forward-mode users call
:func:`~vmex.core.polish_implicit.collocation_polish_tangent` directly.

Dot-product tests cover the chain: native profiles and geometry, collocation
residual, reduced coordinate packing, and the high/low transfer.  The
collocation chart and its frozen positive normalization are local constants; at
a stationary point their parameter derivatives multiply a zero gradient and do
not affect the derivative.

Retired: the square homotopy root
---------------------------------

Earlier releases polished through a square nonlinear root rather than a
rectangular least-squares fit.  That formulation is no longer on the production
path, and this section records it so the change is visible rather than silent.
The code remains in the tree, is exercised by
``tests/test_polish_preconditioner.py``, and is measured by
``benchmarks/strong_root.py``; it is not reachable from
:func:`~vmex.core.polish_driver.polish_legacy_solution`, and so not from
``vmex --polish`` either.

The retired design projected the two physical force channels onto Fourier and
spline coefficients to obtain exactly ``N_R`` radial and ``N_lambda`` helical
equations, and closed the system with ``N_Z`` coordinate equations that set the
projection of the displacement onto the lifted poloidal tangent to zero.  That
tangential-displacement gauge produced a square Jacobian
(:func:`~vmex.core.polish.strong_root_residual`,
:func:`~vmex.core.polish.strong_physical_residual`).

:func:`~vmex.core.polish_driver.polish_strong_root` drove it with a homotopy
``H(c, alpha) = R_low(c) + alpha [R_strong(c) - R_low(c)]`` anchored on the
legacy raw-force defect, advanced by SOLVAX adaptive continuation with
pseudo-transient continuation and Eisenstat--Walker forcing, with a bordered
pseudo-arclength corrector when parameter continuation stalled.

The ``mode-block``, ``legacy``, and ``none`` values of
``PolishConfig.preconditioner``, the ``alpha_*``, ``ptc_*``, and arclength
controls, and the Arnoldi-selected equation signs all belong to that path.
``PolishConfig`` still carries them, and the collocation lane ignores them.

Its structural gate remains a useful rank test.  The five-surface Solovev case
has 23 unknowns and 23 equations, numerical rank 23 at relative SVD tolerance
``1e-8``, a finite JVP agreeing with centered differences, and an unscaled
condition number near ``2.6e5``.  ``benchmarks/strong_root_m4.json`` records
0.287 ms median warm residual and 0.427 ms median warm JVP on an Apple M4, with
1.07 s and 0.69 s first calls and a JVP error of ``9.7e-10``.  These figures
describe that gate only.

The measured reason for retirement is in the shipped artifact's
``projection_consistency`` block: on the shaped tokamak the square projection
reproduces only about six percent of the sampled residual
(``unresolved_fraction`` 0.94), because the nonlinear force is not band-limited
at the retained geometry order.  Development measurements
also rejected global equilibration, volume weighting on its own, and a dense
physical-chart factorization; those negative results are retained in the
project plan ledger rather than as one JSON artifact per experiment.
