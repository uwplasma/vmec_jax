Closed mirror geometry for gyrokinetics
========================================

This page defines the model exported by
:func:`vmex.mirror.gk_closed_fieldline_geometry`.  It is a local gyrokinetic
flux tube on VMEX's **periodic** stellarator--mirror hybrid.  It is not an
open-ended mirror calculation: end losses, sheaths, ambipolar-potential
formation, sources, and loss-cone replenishment are not periodic-geometry
effects and are not modeled by this interface.

Equilibrium coordinates and field
---------------------------------

Let :math:`u\in[0,2\pi)` parameterize the closed racetrack axis
:math:`\boldsymbol c(u)`.  VMEX uses the rotation-minimizing normal frame
:math:`(\boldsymbol n,\boldsymbol b)` and the embedding

.. math::

   \boldsymbol x(s,\theta,u)=\boldsymbol c(u)+\sqrt{s}\,a(s,\theta,u)
   [\cos\theta\,\boldsymbol n(u)+\sin\theta\,\boldsymbol b(u)].

The stream-function representation is discretely divergence free,

.. math::

   \sqrt g B^s=0,\qquad
   \sqrt g B^\theta=I'(s)-\partial_u\lambda,\qquad
   \sqrt g B^u=\Psi'(s)+\partial_\theta\lambda.

For the first public interface :math:`\Psi'` and :math:`I'` are scalar.  The
dimensionless Clebsch label

.. math::

   \alpha=\theta-\frac{I'}{\Psi'}u+\frac{\lambda}{\Psi'}

then satisfies :math:`\boldsymbol B\boldsymbol\cdot\nabla\alpha=0` exactly up
to the common spline/Fourier discretization.  Cartesian dual vectors are
obtained by inverting the coordinate basis
:math:`(\boldsymbol e_s,\boldsymbol e_\theta,\boldsymbol e_u)` only on the
selected non-axis surface.  This avoids manufacturing toroidal coordinates
or passing the racetrack through a WOUT representation.

Closed-tube and equal-arc contract
----------------------------------

The field line is integrated with periodic RK4 through one axis circuit.  Its
poloidal advance must be an integer multiple of :math:`2\pi` within the
requested closure tolerance.  The zero-current mirror closes exactly;
integer-transform tubes are also accepted.  A general irrational-transform
line requires a future twist-linked tube and is rejected rather than silently
made periodic.

If :math:`u` is the raw line parameter, the local parallel factor is

.. math::

   g_\parallel(u)=L_{\rm ref}\frac{|B^u|}{|B|}.

VMEX remaps it to :math:`z\in[-\pi,\pi)` using

.. math::

   z(u)=-\pi+2\pi
   \frac{\int_{u_0}^{u}du'/g_\parallel(u')}
        {\int_{u_0}^{u_0+2\pi}du'/g_\parallel(u')},\qquad
   {\tt gradpar}=\frac{2\pi}{\int du/g_\parallel}.

Thus ``gradpar`` is constant, matching the GKX/GX equal-arc contract.  The
normalizations are

.. math::

   L_{\rm ref}=\sqrt{\frac{V}{\pi L_{\rm axis}}},\qquad
   B_{\rm ref}=\frac{2|\Psi'|}{L_{\rm ref}^2},\qquad \rho=\sqrt{s}.

The scalar metadata ``epsilon`` and ``R0`` use the same definitions as the
toroidal core lane (:func:`vmex.core.turbulence.gk_fieldline_geometry`):

.. math::

   \epsilon=\frac{\max|B|-\min|B|}{\max|B|+\min|B|},\qquad
   R_0=\frac{L_{\rm axis}}{2\pi}=\frac{V}{2\pi^2L_{\rm ref}^2}.

Both are what GKX means by the keys.  Its analytic geometry is
:math:`|B|=B_0/(1+\epsilon\cos\theta)` with ``epsilon`` the inverse aspect
ratio, and the modulation depth is exactly that :math:`\epsilon` for that
model and for any :math:`1/R` tokamak field, while, unlike an aspect ratio, it
exists on a straight mirror (:func:`vmex.core.turbulence.b_modulation_depth`).
:math:`R_0` is the major radius of the circular torus with this axis length;
the same volume identity gives VMEC's ``Rmajor_p`` on the toroidal lane.  GKX
derives ``aminor = epsilon * R0`` and ``a_ref`` from the pair when it writes
run artifacts, which is why ``R0`` is not :math:`L_{\rm ref}`.  Neither
scalar enters its solver.  The field-line mirror ratio
:math:`R_m=\max|B|/\min|B|=(1+\epsilon)/(1-\epsilon)` is exported by name as
``vmex_mirror["field_line_mirror_ratio"]`` (the field-line member of the
mirror-ratio definitions in :doc:`mirror-geometry`), and
``vmex_mirror["field_line_b_modulation"]`` and ``vmex_mirror["R_major"]``
repeat :math:`\epsilon` and :math:`R_0` under their VMEX names.

At zero magnetic shear GKX interprets :math:`k_x` as the direct normalized
radial wavenumber.  The complete perpendicular metric remains

.. math::

   g_{yy}=L_{\rm ref}^2s|\nabla\alpha|^2,\qquad
   g_{xy}=\frac{L_{\rm ref}^2}{2}\nabla\alpha\boldsymbol\cdot\nabla s,
   \qquad
   g_{xx}=\frac{L_{\rm ref}^2}{4s}|\nabla s|^2,

exported as ``gds2``, ``gds21``, and ``gds22``.  Consequently

.. math::

   k_\perp^2=k_y^2g_{yy}+2k_xk_yg_{xy}+k_x^2g_{xx}

without a small-shear division.

Mirror force and magnetic drifts
--------------------------------

The magnetic-mirror coefficient consumed by the Hermite--Laguerre coupling is

.. math::

   {\tt bgrad}=L_{\rm ref}
   \frac{\boldsymbol B\boldsymbol\cdot\nabla|B|}{|B|^2}
   ={\tt gradpar}\,\partial_z\ln |B|.

The grad-B projections are evaluated in Cartesian space from
:math:`\boldsymbol B\times\nabla|B|`.  With
:math:`\sigma_\Psi={\rm sign}(\Psi')`,

.. math::

   {\tt gbdrift}=-\frac{2B_{\rm ref}L_{\rm ref}^2\sqrt{s}\,\sigma_\Psi}
   {|B|^3}(\boldsymbol B\times\nabla|B|)\boldsymbol\cdot\nabla\alpha,

.. math::

   {\tt gbdrift0}=\frac{B_{\rm ref}L_{\rm ref}^2\sigma_\Psi}
   {|B|^3\sqrt{s}}(\boldsymbol B\times\nabla|B|)\boldsymbol\cdot\nabla s.

For vacuum, curvature and grad-B coefficients coincide.  A supplied
``mu0_dp_ds`` adds the standard pressure-curvature term to ``cvdrift``;
``cvdrift0`` retains the radial grad-B projection, consistently with the
existing VMEX/GS2 geometry contract.

Differentiability and performance
---------------------------------

All dynamic evaluation is in JAX: field-line RK4, B-spline evaluation,
Fourier interpolation, equal-arc mapping, Cartesian duals, metrics, and
drifts.  The spline recovery matrix and mode numbers are static arrays.  The
mapping imports no GKX code, so VMEX remains independently installable; GKX's
adapter only converts the returned dictionary to its generic sampled geometry.

The validation ladder is deliberately ordered:

#. positive Jacobian, divergence, axis/frame closure, and field-line closure;
#. constant equal-arc ``gradpar``, positive perpendicular metric determinant,
   and the independent spectral identity for ``bgrad``;
#. JAX directional derivatives against centered finite differences;
#. CPU/GPU value and gradient parity plus cold/warm JIT and memory audits;
#. GKX constant-field and manufactured mirror-force tests;
#. linear resolution/eigenfunction convergence, then quasilinear and matched
   nonlinear audits, then held-out differentiable optimization.

Open-ended mirror roadmap
--------------------------

A physical open-mirror extension needs a nonperiodic parallel operator,
absorbing or kinetic-sheath boundaries, sources, self-consistent ambipolar
potential, collisions that resolve loss-cone replenishment, and a background
ordering suitable for strongly non-Maxwellian plasmas.  Its benchmarks are
uniform open streaming, particle magnetic-moment and bounce invariants,
loss-cone geometry, Pastukhov potential/confinement, velocity-space
convergence, and published full-f Gkeyll cases.  Force softening is acceptable
only beside an unsoftened convergence reference.  None of those claims attach
to the periodic interface documented here.

Primary references and independent inputs
-----------------------------------------

* Xanthopoulos *et al.*, `A geometry interface for gyrokinetic
  microturbulence investigations in toroidal configurations
  <https://doi.org/10.1063/1.3187907>`_ (Phys. Plasmas 16, 082303, 2009),
  defines the field-line metric/drift interface conventions used by
  GS2-family solvers.
* Rodríguez, Helander, and Goodman, `The maximum-J property in
  quasi-isodynamic stellarators
  <https://doi.org/10.1017/S0022377824000345>`_, Appendix C, supplies VMEX's
  independent rotating-ellipse paraxial oracle.
* Francisquez *et al.*, `Towards continuum gyrokinetic study of high-field
  mirrors <https://arxiv.org/abs/2305.06372>`_, and its `published Gkeyll input
  decks
  <https://github.com/gkeyllorg/gkyl-paper-inp/tree/master/2023_PoP_gkwham1x2v>`_
  define the first open-lane Pastukhov, resolution, and force-softening
  comparisons.
* Rosen *et al.*, `Gyrokinetic equilibria of high temperature superconducting
  magnetic mirrors <https://arxiv.org/abs/2604.11684>`_, and its `versioned
  simulation/analysis inputs
  <https://github.com/gkeyllorg/gkyl-paper-inp/tree/master/2026_PRE_hts_mirror_equilibria>`_
  provide kinetic-equilibrium, potential, confinement, source, and mirror-ratio
  comparisons for the future open lane.
