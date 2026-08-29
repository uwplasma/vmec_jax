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
cylindrical angle is ``phi=zeta/NFP``.  Each real Fourier amplitude is

.. math::

   X_{mn}(\rho) = \rho^{|m|} q_{mn}(s), \qquad
   q_{mn}(s) = \sum_k c_{kmn} B_k(s).

The local clamped B-splines have odd degree 3, 5, or 7; degree 5 is the
production starting point because current and force require stable second
radial derivatives.  The factor ``rho**abs(m)`` is analytic and is never
estimated from sampled surfaces.

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

Independent continuum oracle
----------------------------

At arbitrary off-axis points, :mod:`vmex.core.strong_force` constructs the
Cartesian position, covariant basis, metric, signed Jacobian, contravariant and
covariant magnetic field, current, pressure gradient, and finally

.. math::

   \mathbf{F} = \frac{(\nabla\times\mathbf{B})\times\mathbf{B}}{\mu_0}
                - \nabla p.

Spline and Fourier functions are differentiated analytically by JAX.  No
legacy half-mesh force, radial finite difference, or solve collocation value is
reused.  The conventional independent components are

.. math::

   F_\rho = \mathbf{F}\cdot\partial_\rho\mathbf{r}, \qquad
   F_{\mathrm{helical}} =
   \frac{\partial_\theta B_\zeta-\partial_\zeta B_\theta}{\mu_0}.

The certificate uses shifted Gauss points, quadrature order at least two above
the representation order, doubled angular resolution by default, and float64.
It reports dimensional volume L2/P99/Linf force density, the symmetric
pointwise normalization

.. math::

   \frac{2|\mathbf{F}|}
        {|\mathbf{J}\times\mathbf{B}|+|\nabla p|+F_{\mathrm{floor}}},

radial and helical contributions, near-axis/bulk/edge norms, a flux-surface
profile, angular tail, an independently recomputed radial-quadrature
difference, signed-Jacobian margin, boundary residual, and gauge residual.

Square system and nullspace policy
----------------------------------

The polishing solve uses the explicit three-channel formulation.  In the
stellarator-symmetric chart, active ``R_cos``, ``Z_sin``, and ``L_sin``
corrections are sampled through the tested high/low transfer.  Fixed-edge,
structural-zero, axis-closure, and lambda-gauge entries are absent.  Each
constrained three-dimensional ``m=1,+/-n`` Z pair is one orthonormal
coordinate ``(z_+ + z_-)/sqrt(2)``, not two duplicate unknowns.  The reduced
vector is therefore

.. math::

   c = (c_R, c_Z, c_\lambda)
       \in \mathbb{R}^{N_R+N_Z+N_\lambda}.

At shifted radial collocation points, the independent oracle forms signed
physical radial and helical force densities.  Their symmetric normalization
uses smooth ``sqrt(|v|^2 + force_floor^2)`` norms, so the root is differentiable
even when one physical contribution vanishes.  The normalization scale is
frozen from the lifted branch root for the solve, so it conditions but cannot
create a state-dependent near-null direction; the final certificate recomputes
the symmetric normalization independently.  A third, force-free coordinate
equation projects the corrected displacement onto the lifted state's normalized
poloidal tangent and sets that tangential displacement to zero.  It fixes the
surface chart without adding physical-force content.  Full-period Fourier
projection, analytic removal of ``rho**abs(m)``, radial spline fitting, and the
high/low map produce ``N_R`` radial-force, ``N_Z`` coordinate, and
``N_\lambda`` helical-force equations.  Thus the nonlinear Jacobian is square,
while the lambda gauge remains eliminated structurally.

The low endpoint is the nonlinear legacy raw-force defect
``F_low(base+c)-F_low(base)`` evaluated at the corrected legacy coordinates.
This anchors the continuation exactly at the converged discrete VMEX state
without asking a high-order correction to remove the accepted finite legacy
stopping defect.  The stored block system's fixed row scales make its channels
dimensionless without changing its Jacobian.  With
``R_low`` and the normalized three-channel strong residual ``R_strong``, the
implemented homotopy is exactly

.. math::

   H(c,\alpha) = R_{low}(c)
       + \alpha\,[R_{strong}(c)-R_{low}(c)].

The anchored endpoint makes ``H(0,0)=0`` exactly; the continuation driver
checks that consistency endpoint and then advances ``alpha``.  At ``alpha=1``
the residual is independent strong force plus the coordinate equation.  This
is a defect-correction continuation between equations, not an interpolation
of equilibrium states and not minimization.

The signs of the radial, coordinate, and helical equation blocks are selected
from the eight possibilities using a bounded matrix-free Arnoldi diagnostic of
the low-preconditioned strong Jacobian at the lifted state.  The deterministic
choice maximizes the leftmost Ritz value.  This fixed nonsingular row scaling
cannot change the strong root, but prevents artificial early folds caused only
by inconsistent equation orientation.  The selected signs and remaining
operator balance are runtime metadata.

Small-problem tests explicitly assemble the Jacobian and reject unexplained
nullspaces.  The five-surface Solovev structural gate has 24 independent
unknowns and 24 equations, numerical rank 24 at relative SVD tolerance
``1e-8``, and a finite JVP that agrees with centered finite differences.  Its
measured condition number is about ``3.5e5`` before nonlinear-solver
preconditioning, which is recorded rather than hidden.  This deliberately
coarse case is a layout/rank test, not a polished-equilibrium accuracy claim.
A failed rank test is a hard error; the implementation does not regularize an
accidentally underdetermined system with a merit-function penalty.

The clean-commit Apple M4 record in ``benchmarks/strong_root_m4.json`` reports
0.262 ms median warm residual and 0.360 ms median warm JVP time over 20 repeats;
cold compile-plus-execute times are 1.12 s and 1.89 s.  The recorded JVP error
is ``1.0e-9`` and the runtime-build, residual, and JVP peak-RSS increments are
reported separately.  These figures describe the structural five-surface gate
only and are not extrapolated into a production-resolution solve claim.

Low-order physics preconditioner
--------------------------------

The first high-order preconditioner reuses the exact nearest-neighbour
raw-force block linearization from the implicit tangent/adjoint path.  For a
high-order residual ``r_H``, :mod:`vmex.core.polish` applies

.. math::

   P_H r_H = T_{LH} A_L^{-1} T_{HL} r_H,

where ``T_HL`` samples every regularized spline mode on the VMEX full mesh,
restores VMEX Fourier normalization and the internal ``m=1`` packing, and
projects onto the evolved legacy degrees of freedom.  ``T_LH`` fits the
regularized coefficients back to the high-order basis.  Its R/Z fit has a
structurally zero terminal coefficient, so a correction cannot move the fixed
boundary.  Symmetry zeros and the lambda gauge are also eliminated rather
than penalized.

The raw-force factors are built once and stored.  Forward and transpose
applications use the same SOLVAX block-Thomas factors; the transpose path is
the algebraic transpose of the entire transfer--solve--transfer composition,
not merely a second approximate inverse.  Tests certify both transfer
dualities and the complete preconditioner duality.  A quality monitor reports
the true relative residual ``||A P r-r||/||r||`` on fixed probes, allowing the
nonlinear driver to refresh factors only after measured degradation.

This level contains the dominant legacy radial physics but not the full
high-order angular coupling.  It is a right preconditioner, not the polishing
operator, and it never assembles a dense high-order Jacobian.  The subsequent
p-level hierarchy may wrap it as the coarse solve without changing this
contract.

The strong endpoint is first normalized by its frozen initial RMS and then by
a deterministic matrix-free estimate of the low-inverse/strong-Jacobian
stiffness.  This positive scalar leaves the target root unchanged while making
the continuation endpoints comparable.  The estimate uses a fixed probe and a
bounded number of JVP/low-inverse applications; it is stored as
``operator_balance`` for provenance.

Branch-preserving driver
------------------------

The fixed-boundary host driver first solves the ``alpha=0`` legacy consistency
endpoint, then calls SOLVAX adaptive continuation with state-dependent JVPs,
Eisenstat--Walker forcing, and the stored low-order inverse.  The measured
initial pseudo-time scale is large because the legacy endpoint has already
been row equilibrated; ``dtau=1e6`` reaches the Newton regime on the structural
Solovev gate while retaining PTC backtracking and adaptive shrink safeguards.

Every proposed state must remain finite and retain a signed-Jacobian margin
above both an absolute floor and a fixed fraction of the lifted branch margin.
The fixed boundary, profile data, parity, and lambda gauge cannot drift because
they are absent from the free coordinate map.  Rejected states never replace
the accepted branch point.

If ordinary parameter continuation exhausts its minimum step, the driver can
form a matrix-free branch tangent and invoke SOLVAX's bordered
pseudo-arclength corrector.  Its block-elimination preconditioner reuses the
same low-order inverse and an explicit scalar Schur complement; no dense
high-order Jacobian is formed.  A compact report retains accepted/rejected
stages, nonlinear and linear work, residual evaluations, arclength work,
minimum Jacobian margin, factor time, and wall time.  Failure to reach
``alpha=1`` or to pass the independent overintegrated certificate is typed and
never reported as a polished equilibrium.  ``return_unpolished`` is an
explicit opt-in policy that returns the original native state with
``report.converged=False``.

``PolishConfig.preconditioner="legacy"`` is the measured default.  An
experimental ``"mode-block"`` variant probes only bounded bands of neighboring
Fourier modes, retaining all radial and R/Z/lambda couplings inside each band,
and blends the low and strong blocks with ``alpha``.  It never assembles a
global Jacobian.  The variant remains opt-in because the hard coarse Solovev
stress case did not reduce total Krylov work; future p/h hierarchy benchmarks,
not API preference, decide whether it can become the default.
