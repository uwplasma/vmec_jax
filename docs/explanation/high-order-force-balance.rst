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

The polishing solve uses the reduced physical-displacement formulation.  Let
``N`` be the number of free, regularized scalar coefficients after the axis,
boundary, symmetry, and gauge maps above.  Pure surface relabeling is removed
before a nonlinear system is formed.  The unknown is two physical displacement
fields,

.. math::

   c = (c_\perp, c_\mathrm{binormal}) \in \mathbb{R}^{2N},

and the residual contains the same ``N`` independent tests of radial force and
``N`` independent tests of helical force.  Thus the nonlinear Jacobian is
``2N x 2N``.  Lambda and the straight-field-line map are recovered through the
same affine condensation, rather than added as a third unconstrained field.

Before this reduced operator is enabled for production, small-problem tests
must explicitly assemble its Jacobian, verify numerical rank ``2N``, and show
that the discarded coordinate tangent lies in the full formulation's
nullspace.  A failed rank test is a hard error; the implementation must not
regularize an accidentally underdetermined system with a merit-function
penalty.

Continuation then connects the converged legacy root to this square strong
root.  The low-order operator remains the preconditioner/coarse model, and the
independent certificate above is never used as the solve residual.
