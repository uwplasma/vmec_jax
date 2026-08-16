Mirror geometry
===============

``vmex.mirror`` contains two spline-native scalar-pressure equilibrium
models. Open mirrors use coordinates ``(s, theta, xi)`` with a nonperiodic
axial coordinate and fixed-flux end cuts; they are not reinterpreted as thin
periodic tori. Closed stellarator-mirror hybrids use a periodic longitudinal
B-spline around two exactly straight mirror legs and two curved stellarator
returns. The axisymmetric fixed-boundary open lane is supported; the
rotating-ellipse lane is a release candidate. Axisymmetric free boundary is
supported through 10% requested beta. The
periodic hybrid has a complete fixed-boundary solve and example. Its
circular-section lane is supported: with the leg-return junction frozen as a
design parameter, its independent strong-force residual converges monotonically
under same-geometry refinement. The rotating-elliptical-section hybrid --
``section_turns`` turns the ellipse continuously around the closed circuit --
remains the single research candidate; the hybrid sections below give its
transform gain and the near-axis representation defect that blocks promotion.
Run recipes are in :doc:`/howto/mirror-machines`; the ``mout_*.nc`` output
format is :doc:`/reference/mout-file`.

Open topology and end cuts
--------------------------

The open coordinates are
:math:`(s,\theta,\xi)\in[0,1]\times[0,2\pi)\times[-1,1]`. The lateral
surface :math:`s=1` is the plasma-vacuum interface. The planes
:math:`\xi=\pm1` are prescribed computational cuts through the flux tube;
magnetic flux passes through them, so they are neither material interfaces nor
``B.n=0`` boundaries. The divergence-free representation is

.. math::

   \sqrt{g}B^\theta = I'(s)-\partial_\xi\lambda, \qquad
   \sqrt{g}B^\xi = \Psi'(s)+\partial_\theta\lambda,

with :math:`B^s=0` and a zero-surface-mean gauge for :math:`\lambda`.
All theta samples at the magnetic axis denote one physical point. Writing
:math:`q_0=\lim_{s\to0}r\,\partial_s r`, single-valued axial field requires

.. math::

   \partial_\theta\lambda(0,\theta,\xi)
   = \Psi'(0)\left[\frac{q_0(\theta,\xi)}{\langle q_0\rangle_\theta}-1\right].

The solver eliminates the axis stream function with this condition and
includes its geometry dependence in variational and implicit derivatives.
Axis ``|B|`` nonuniformity is stored as a separate axis-uniformity diagnostic.

An unbounded exterior Green solve requires a geometrically closed integration
surface, so disks temporarily close the two cuts. Their Neumann data continue
the nonzero plasma and applied-field through-flux across each cut. The disks do
not close the plasma or acquire an interface pressure-balance equation.
Tangency and total-pressure continuity are enforced only on the lateral LCFS.
The complete nested radial profile at each cut is prescribed by the initial
state and remains fixed during primal and implicit solves. A state built only
from the LCFS uses radially self-similar cuts. For a finite-radius supplied
field, callers should instead initialize nested cut surfaces from its enclosed
flux; otherwise the incompatible cut data appear as a localized strong-force
error even when the variational residual is small.

Resolution contract
-------------------

``MirrorResolution(ns=..., mpol=..., nxi=...)`` has one angular-resolution
input: ``mpol`` is the largest represented Fourier mode. The plasma
collocation size is the read-only value ``ntheta = 2*mpol + 1``; therefore
``mpol=0`` has one theta node and ``mpol=4`` has nine. Users cannot request an
inconsistent node/mode pair or accidentally introduce a Nyquist mode.
Exterior angular quadrature remains an independent argument because it
integrates the vacuum boundary rather than defining plasma unknowns.

Radial location map
-------------------

The radial mesh follows the VMEC staggering convention, adapted to the
two-point Gauss energy rule:

* geometry and stream-function unknowns are stored on full surfaces
  :math:`s_i`;
* metric terms, :math:`\sqrt{g}B^\theta`, :math:`\sqrt{g}B^\xi`, and magnetic
  energy are evaluated at two Gauss points inside each radial cell;
* Gauss averages define cell-centered covariant
  :math:`B_\theta`, :math:`B_\xi`, Jacobian, and pressure at
  :math:`s_{i+1/2}`;
* current and ``J x B - grad(p)`` are reconstructed on interior full surfaces;
* the magnetic axis, fixed LCFS, and open end cuts are reported separately
  from the unconstrained physical-volume norm.

For open mirrors, ``bulk`` and radial-axis diagnostics use the central 80% of
the axial coordinate. ``end_collar`` uses the outer 20% nearest the two fixed
cuts. The all-volume norm retains both regions.

Strong-force gate normalization
-------------------------------

The pointwise residual ``|J x B - grad(p)|`` is a force density and needs a
reference force scale to become a dimensionless gate. The primary
normalization divides by :math:`B^2/(\mu_0 a)`, where the minor radius ``a``
is the flux-equivalent LCFS radius: the midplane cross-section for open
mirrors and the circuit-averaged section for closed hybrids
(:func:`vmex.mirror.forces.effective_minor_radius`; a caller may also pass
``minor_radius`` explicitly). Transverse pressure balance acts on the
minor-radius scale, so :math:`B^2/(\mu_0 a)` is the magnitude of the
competing equilibrium forces, and every lane has one structural ``a``. The
earlier normalization used the device length ``L`` (cap-to-cap extent for
open mirrors, axis arc length for the closed racetrack). That number is
linear in an arbitrary length: open lanes have ``L/a`` near 17--20 while the
closed racetrack has ``L/a`` near 67, so one identical solved hybrid state
reads ``0.205`` arc-normalized but ``0.0030`` minor-radius-normalized, even
though its absolute force density is lower than that of the passing medium
rotating ellipse. The device-length number therefore remains available only
as the secondary ``device_normalized_rms`` diagnostic, and the recorded
``benchmarks/mirror_*.json`` files intentionally keep quoting it: they
document the historical evidence and are not rewritten under the new
normalization.

Gate evaluation reports zones rather than a single folded number:
:func:`vmex.mirror.forces.force_gate_zones` returns the all-volume, bulk,
end-collar, and near-axis/first-row norms (all minor-radius-normalized)
together with the legacy device-length total, so constrained-data regions --
the frozen end cuts and the regularized axis -- are visible next to the
unconstrained bulk. Promotion additionally requires refinement evidence:
:func:`vmex.mirror.forces.refinement_convergence` reports per-step ratios and
monotonicity for residuals from two or more resolutions, and
:func:`vmex.mirror.forces.passes_promotion_gate` combines the absolute gate
on the finest rung with that monotone-decrease requirement.

Re-measured gate numbers for the shipped cases, evaluated on the identical
solved states under both normalizations (device-length ``L``-normalized
values match the recorded evidence bit-for-bit):

.. list-table::
   :header-rows: 1
   :widths: 44 12 12 10 11 11

   * - Case
     - device norm
     - minor norm
     - bulk
     - end collar
     - axis row
   * - Rotating ellipse ``(ns,mpol,nxi,elements)`` = ``(5,4,13,5)``
     - 0.0667
     - 0.00400
     - 0.00231
     - 0.00683
     - 0.00343
   * - Rotating ellipse ``(7,6,17,6)``, the release-candidate lane
     - 0.0267
     - 0.00160
     - 0.00073
     - 0.00274
     - 0.00196
   * - Rotating ellipse ``(9,8,21,8)``
     - 0.0142
     - 0.00085
     - 0.00026
     - 0.00156
     - 0.00060
   * - SFLM ``(7,6,17,6)``, paraxial benchmark
     - 0.335
     - 0.0168
     - 0.00259
     - 0.0350
     - 0.00863
   * - SFLM ``(9,8,21,8)``, paraxial benchmark refined
     - 0.177
     - 0.00883
     - 0.00140
     - 0.0169
     - 0.00442
   * - Hybrid circular section, frozen junction, ``ns=5``, 16 controls
     - 0.204
     - 0.00304
     - 0.00304
     - (none)
     - 0.00261
   * - Hybrid circular section, frozen junction, ``ns=5``, 32 controls
     - 0.176
     - 0.00261
     - 0.00261
     - (none)
     - 0.00196
   * - Hybrid circular section, frozen junction, ``ns=5``, 64 controls
     - 0.118
     - 0.00175
     - 0.00175
     - (none)
     - 0.00084
   * - Free boundary ``(ns,nxi,elements,panels)`` = ``(5,7,4,8)``, beta 0
     - 0.0421
     - 0.00665
     - 0.00610
     - 0.00718
     - 0.00431
   * - Free boundary ``(5,7,4,8)``, beta 10%
     - 0.0269
     - 0.00430
     - 0.00490
     - 0.00361
     - 0.00588

The rotating-ellipse ladder stays monotone under the new normalization
(per-step ratios 2.50 and 1.88), and the frozen-junction circular-section
hybrid ladder in the table is likewise monotone: with the junction geometry
held fixed, exact refinement of the solve basis drives the force down instead
of plateauing.
Its minor-radius number sits between the coarse and medium open rungs instead
of appearing an order of magnitude worse: the apparent cross-lane gap was the
``L/a`` disparity, not a larger force error. The straight field-line mirror (SFLM) is a
paraxial-accuracy benchmark rather than a failed case: its unconstrained
bulk force is clean (``0.00259`` minor-normalized, below the ``0.05`` gate)
and converges under refinement -- the ``(7,6,17,6) -> (9,8,21,8)`` step
halves it, ``0.00259 -> 0.00140`` (ratio ``1.85``), and the all-volume and
end-collar norms fall in step (``0.0168 -> 0.00883`` and
``0.0350 -> 0.0169``). The elevated end collar is the expected boundary layer
where the analytic Agren--Savenko cut profile, an equilibrium only to order
``(a/c)^2``, is frozen at the two end cuts; it converges rather than plateaus.

In particular, radial curl and pressure terms use conservative cell
differences,

.. math::

   \sqrt{g}J^\theta_i = \frac{1}{\mu_0}
   \left[\partial_\xi B_{s,i}
   - \frac{B_{\xi,i+1/2}-B_{\xi,i-1/2}}{\Delta s}\right],

.. math::

   \sqrt{g}J^\xi_i = \frac{1}{\mu_0}
   \left[\frac{B_{\theta,i+1/2}-B_{\theta,i-1/2}}{\Delta s}
   - \partial_\theta B_{s,i}\right], \qquad
   p'_i = \frac{p_{i+1/2}-p_{i-1/2}}{\Delta s}.

Contravariant field on a full surface is reconstructed by averaging flux
density and Jacobian separately, rather than averaging their ratio. This is
the same placement used by VMEC2000 ``jxbforce`` and avoids differentiating an
unrelated full-mesh reconstruction of the energy field.

Current capability and limits
-----------------------------

The package currently includes:

* a supported axisymmetric and a release-candidate nonaxisymmetric
  fixed-boundary finite-current solve,
* an isotropic VMEC-style conserved-mass pressure energy with independent
  weak and pointwise force diagnostics,
* a free-space boundary-integral vacuum model with an ``xyz -> B`` field
  callable or the shared ESSOS/MAKEGRID-compatible ``MgridField``; coil
  geometry and Biot-Savart evaluation remain in ESSOS,
* coupled axisymmetric free-boundary beta continuation with compressed
  restart files,
* a closed-surface Neumann solve on the lateral LCFS and both end disks,
* component-wise nonlinear convergence checks at a requested ``ftol=1e-12``.
* a periodic B-spline racetrack with two straight mirror legs, a continuously
  rotating elliptical section, a fixed-boundary solve, and closed field-line
  tracing.

With the compact-coil configuration (0.5 m loops, vacuum ``B(0) = 0.0836 T``,
mirror ratio 4.58), a requested 50% beta continuation grows the central radius
by 7.5% and lowers the on-axis field by 22.3% from vacuum, exercising the
finite-beta coupling end to end. The axisymmetric free-boundary path is
**supported through 10% requested beta**. The 25% and 50% points are retained
as extended validation: they converge variationally, but the independent
strong-force promotion gate in
``benchmarks/mirror_free_boundary_axisymmetric.json`` fails. The
nonaxisymmetric free-boundary path is deferred because its point observables
were not monotone under spatial refinement. See :doc:`/reference/capabilities`
for the normative status.

Periodic stellarator-mirror hybrid
----------------------------------

The implemented target follows the topology proposed for `stellarators
linking axisymmetric mirrors (SLAM)
<https://downloads.regulations.gov/DOE-HQ-2023-0038-0020/attachment_1.pdf>`_:
two long straight mirror legs are joined by two curved stellarator sections.
The same topology appears in the `APS linked-mirror abstract
<https://meetings.aps.org/Meeting/DPP22/Session/NP11.24>`_. It is related to,
but geometrically distinct from, the warm-stellarator/rectilinear-mirror
fusion--fission hybrid proposed by `Moiseenko et al.
<https://doi.org/10.1017/S0022377823000442>`_. The implementation uses de
Boor's local cubic B-spline construction and `Bishop's rotation-minimizing
frame <https://doi.org/10.1080/00029890.1975.11993807>`_. The present code
addresses only ideal-MHD nested-surface geometry and a scalar-pressure
equilibrium. It does not model fusion--fission blankets, minority fast ions,
end losses, or kinetic stability.

Let :math:`u\in[0,2\pi)` parameterize the closed circuit. The coordinate map is

.. math::

   \boldsymbol x(s,\theta,u) = \boldsymbol c(u)
   + \sqrt{s}\,a(s,\theta,u)
     [\cos\theta\,\boldsymbol n(u)+\sin\theta\,\boldsymbol b(u)].

``CubicBSplineBasis.periodic_uniform`` represents :math:`\boldsymbol c`.
Control points are divided between straight leg, return, opposite straight
leg, and second return. A cubic spline has local support, so every central span
whose four active controls are collinear is exactly straight; increasing the
number of controls lengthens the exact straight region rather than merely
improving a global Fourier fit. ``evaluate_closed_spline_axis`` transports a
Bishop rotation-minimizing frame :math:`(\boldsymbol n,\boldsymbol b)` around
the curve and distributes its residual holonomy over the period. This frame is
well defined where the straight-leg curvature is exactly zero, unlike the
Frenet frame.

The LCFS is an ellipse written as a polar radius,

.. math::

   a(1,\theta,u) =
   \frac{A B}{\sqrt{[B\cos(\theta-\alpha(u))]^2
                   +[A\sin(\theta-\alpha(u))]^2}}.

The section angle is :math:`\alpha(u)=\alpha_{\mathrm{ret}}(u)+N\,u`. The
return term :math:`\alpha_{\mathrm{ret}}` is constant on each straight leg and
changes smoothly by 90 degrees through each return, so on its own it returns the
ellipse to its original orientation once per circuit. The ``section_turns``
integer :math:`N` superposes a genuine rotating ellipse: the major axis turns
continuously by :math:`N` full :math:`2\pi` turns per circuit. Because the polar
radius is :math:`2\pi`-periodic in :math:`\theta-\alpha` and :math:`N` is an
integer, the section closes on itself exactly (verified by the m=2 harmonic
winding by :math:`4\pi N`), and the straight-leg axis stays exactly straight
while the ellipse it carries keeps rotating. The radial surfaces and stream
function use the same periodic longitudinal basis. The divergence-free field
is the open expression with :math:`\xi` replaced by :math:`u`; periodicity
removes end cuts, and all longitudinal coefficients are active. A finite
:math:`I'(s)` gives visible pitch and nonzero rotational transform; at the
device aspect ratio :math:`L/a\approx 67` the rotating ellipse adds negligible
transform on its own (with :math:`I'(s)=0` the traced :math:`\iota` stays below
:math:`10^{-3}` for every ``section_turns``), so it acts by amplifying the
current-driven transform rather than by a standalone geometric one.

``build_stellarator_mirror_hybrid`` constructs the discretization, closed
axis, LCFS, and a vacuum-field initial stream function. The ordinary
``solve_fixed_boundary`` then uses the same energy, host globalization,
matrix-free Hessian actions, and separable preconditioner as open mirrors.
``trace_closed_field_line`` integrates
:math:`d\theta/du=B^\theta/B^u` with periodic RK4 steps.

The leg-return junction -- where an exactly straight leg (zero curvature)
meets a circular return (curvature :math:`1/R`) -- is rounded across the cubic
spline's local support. Building the axis directly in a finer solve basis
narrows that rounding and sharpens the junction curvature overshoot as fast as
refinement helps, so the as-built geometry family does not converge: the
circular section reads ``0.184`` device-normalized at 32 controls and ``0.218``
at 64. ``build_stellarator_mirror_hybrid(axis_coefficient_count=...)`` freezes
the junction as an explicit design parameter of the closed B-spline axis
family. The racetrack axis and rotating section are constructed at that base
control count and then exactly refined (``refine_periodic_uniform``, dyadic and
curve-preserving to roundoff) to the solve ``coefficient_count``, so the
junction-transition width is held fixed while the equilibrium resolution
increases. The default ``axis_coefficient_count=None`` keeps the legacy
behaviour of building the geometry in the solve basis.

With the junction frozen at 16 controls, the circular-section hybrid converges
monotonically under exact 16/32/64 solve refinement: device-normalized
all-volume force ``0.204 -> 0.176 -> 0.118`` (reproducing the audit ladder) and
minor-radius bulk force ``0.00304 -> 0.00261 -> 0.00175`` (per-step ratios
``1.16`` and ``1.49``), every rung below the ``0.05`` gate with variational
residual below ``3.2e-13`` and normalized ``div(B)`` below ``1e-13``. This
passes the promotion criterion (absolute gate on the finest rung and monotone
refinement), so the circular-section hybrid is a supported lane. Every hybrid
residual in this section is quoted device- (arc-length-) normalized where noted
(the legacy normalization recorded in
``benchmarks/mirror_hybrid_fixed_boundary.json``); the racetrack arc length is
about 67 minor radii, so the device numbers are roughly 67 times the
minor-radius numbers.

The rotating-elliptical-section hybrid remains the single research candidate,
now driven by the genuine toroidal rotation ``section_turns``. Its shipped
``ns=5``, ``mpol=4``, 32-control case (``semi_major=0.45``, ``semi_minor=0.25``,
``section_turns=2``) reaches variational residual ``3.19e-13`` and normalized
``div(B)=4.55e-14`` with axis closure ``8.88e-16``, and the solved
finite-current state gives ``iota=0.141`` at ``s=0.75`` -- roughly 1.7 times the
return-only ``iota=0.085`` at the same imposed current. As noted above, the
transform is current-driven and amplified by the rotating geometry, which
reshapes the metric so the same current winds field lines faster.

Under the junction-freeze contract the toroidally rotating hybrid converges at
every rung, but the near-axis defect persists. With the junction frozen at 16
controls and the section built at that base count, exact 16/32/64 refinement
(``ns=5``, ``mpol=6``, ``section_turns=2``) drives the minor-radius bulk force
``0.0445 -> 0.0056 -> 0.0046`` -- monotone, every rung below the ``0.05`` gate,
so the operational promotion gate passes -- with each rung at variational
residual below ``3.6e-13`` and normalized ``div(B)`` below ``1e-13``. The
device- (arc-length-) normalized strong force, however, plateaus
``4.07 -> 0.51 -> 0.42`` (per-step ratios ``7.97`` and ``1.21``): it does not
head toward zero like the promoted circular lane's ladder above,
and its ``~0.42`` floor is even higher than the return-only ``~0.33``. That
plateau is the same separately scoped near-axis representation defect in the
rotating section, made no better by the faster rotation, so the toroidally
rotating hybrid is kept a research candidate; finite-beta continuation and
rotating-section sensitivity claims are deferred until it is resolved.

A sensitivity study on the return-only baseline localizes the rotating-section
residual to the racetrack geometry rather than to pressure or imposed current.
At the 32-control resolution, removing current changes the return-only strong
force only from ``0.430`` to ``0.424``; replacing the rotating ellipse by a
circular section gives ``0.158`` and a fixed ellipse ``0.164``, while the
circular-axis/circular-section limit gives ``0.0083``. The periodic derivative
algorithm is validated separately on the closed circular limit below.

Source ownership is compact: periodic basis/refinement is in ``basis.py``;
axis, Bishop frame, and embedding are in ``geometry.py``; coefficient packing,
initialization, solve dispatch, and tracing are in ``splines.py``; the shared
energy and force diagnostics are in ``forces.py``; and the reviewed figure is
produced by ``output.plot_stellarator_mirror_hybrid``.

.. image:: /_static/figures/stellarator_mirror_hybrid.png
   :alt: Solved periodic B-spline stellarator-mirror hybrid with its axis, boundary magnetic field, cross-sections, transform, and residuals
   :width: 100%

QI-mirror hybrid: Fourier vs B-spline
-------------------------------------

A quasi-isodynamic (QI) stellarator has poloidally closed ``|B|`` contours and
near-straight magnetic-axis segments at its field-period-symmetric planes, so
the QI axis is the natural place to cut and insert a straight mirror cell.
``examples/mirror/qi_mirror_hybrid_fourier_vs_bspline.py`` makes that construction
concrete and compares the two ways of representing the resulting axis.

Cut locations.  The example solves ``input.nfp2_QI`` with the VMEC (Fourier)
core, reads the magnetic axis :math:`\mathbf{r}(\phi)` from the axis Fourier
arrays, and computes the 3-D curvature
:math:`\kappa = \lVert \mathbf{r}' \times \mathbf{r}'' \rVert / \lVert
\mathbf{r}' \rVert^{3}` spectrally over the torus. For this nfp=2 QI the
curvature spans ``70x``. An nfp=2 QI axis is near-straight at two planes per
field period, so the curvature has *four* minima: :math:`\kappa \approx 0.036`
:math:`\mathrm{m}^{-1}` at :math:`\phi = 0, \pi` and :math:`\approx 0.088`
:math:`\mathrm{m}^{-1}` at :math:`\phi = \pi/2, 3\pi/2` (the four
stellarator-symmetry planes, where the axis crosses the midplane), and maxima
:math:`\approx 2.5` :math:`\mathrm{m}^{-1}` at the bean tips. All four
low-curvature planes are cut, so the inserted legs are stellarator symmetric.

Cut-and-splice.  ``splice_straight_legs`` cuts the closed axis at the four minima
and inserts an exactly-straight leg at each **along the local axis tangent**, so
every leg continues the axis in its own direction (not a shared transverse
direction). The per-cut leg lengths are chosen so the inserted displacements
cancel -- the loop closes to rounding (``closure ~ 1e-16``) -- which splits the
legs into two symmetry classes (the two symmetry-plane types, here ``1.31 m`` and
``1.07 m``). One stellarator-symmetric half, between the two symmetry-fixed
planes, is built and reflected 180 degrees about the ``x`` axis, so the
four-legged racetrack is stellarator symmetric to rounding. Each leg is tangent
to the axis at its cut, so the leg/return junction is tangent-continuous
(residual break ``~0.04 deg``, versus a real corner): the seam is now a
*curvature* break -- the exactly-zero-curvature leg meeting the finite-curvature
return -- which is what separates the two representations.

Representation accuracy.  The same closed hybrid axis is fitted with a truncated
Fourier series (global, VMEC-native) and a periodic cubic B-spline (local, the
``vmex.mirror`` lane). On the straight mirror leg:

.. list-table::
   :header-rows: 1

   * - basis
     - degrees of freedom
     - deviation on the straight leg
   * - Fourier, :math:`N=8`
     - 51
     - ``2.0e-2`` (ringing)
   * - Fourier, :math:`N=32`
     - 195
     - ``2.7e-5`` (ringing)
   * - Fourier, :math:`N=64`
     - 387
     - ``2.0e-6`` (ringing, floor)
   * - B-spline, :math:`M=64`
     - 192
     - ``2.2e-6`` (leg midpoint)
   * - B-spline, :math:`M=128`
     - 384
     - ``6.0e-9`` (leg midpoint)
   * - B-spline, :math:`M=256`
     - 768
     - ``1.5e-12`` (leg midpoint)

The B-spline reproduces the straight cell to machine precision once each leg is
backed by enough collinear controls (its error is confined to a fixed few
knot-spacings around the junction, the signature of local support), while the
global Fourier series rings across the whole leg and floors near ``2e-6``. The
tangent-continuous seam lets the Fourier ringing decay faster than the old
sharp-corner case, but only the local basis reproduces the exactly-straight cell
to machine precision. The *maximum* error of both bases is instead set by the
leg/return curvature break (a cubic B-spline is :math:`C^2` and also rounds a
curvature step), so this is an honest, shared limit, not a B-spline advantage.
The comparison is the point: the local basis wins precisely on the
exactly-straight mirror cell.

Equilibrium.  ``build_qi_mirror_hybrid`` fits the spliced axis into the closed
solve basis, wraps it in a constant circular section (rotation-invariant, so the
large frame holonomy of a fully 3-D axis does not enter the boundary), and
returns a ``StellaratorMirrorSetup`` that feeds ``solve_fixed_boundary`` like the
analytic racetrack. The solved hybrid is divergence-free to ``9.4e-14`` with
``iota = 0.11`` and mirror ratio ``1.8``; its normalized strong-force residual is
``1.3e-2`` (the smooth analytic racetrack, with circular returns, converges to
``2.6e-3`` by contrast -- the QI returns' finite curvature and the four
curvature-break seams still load the force). This B-spline
equilibrium is a scalar-pressure spline model whose transform comes from a weak
axial current: it demonstrates the geometry and the exactly-straight mirror cell,
**not** a reproduction of the QI rotational transform (``|iota| ~ 0.45``).

Fourier-lane limitation.  A literal VMEC re-solve of a straight-axis QI-mirror
device is out of scope by construction, not by effort: VMEC's toroidal
coordinate is the cylindrical angle :math:`\phi`, and a straight axis segment
cannot be parameterised by :math:`\phi` (its :math:`R(\phi), Z(\phi)` are
degenerate). This is exactly why the closed-axis, arc-length-parameterised
B-spline lane exists. The VMEC solve therefore serves as the QI *reference*
(axis, ``iota``, ``|B|``, ``B_0``), and the Fourier side of the comparison is the
axis-representation accuracy above.

.. image:: /_static/figures/qi_mirror_hybrid.png
   :alt: QI-mirror hybrid comparison: QI axis coloured by curvature with cut locations, the spliced straight-leg hybrid axis, hybrid boundary magnetic field, and Fourier-versus-B-spline representation accuracy at the seam
   :width: 100%

Fixed-boundary 3D solver
------------------------

The fixed-boundary host solver jointly advances geometry and a gauge-free
stream function for helical boundaries with finite axial current. Its Newton
preconditioner combines radial, Fourier-poloidal, and CGL-axial model
stiffness. The eliminated weighted-mean stream-function node is handled by a
symmetric lift and projection, so the reduced preconditioner remains positive
definite.

The formal full-physics test refines ``(ns,nxi)`` through ``(5,5)``, ``(7,7)``,
and ``(9,9)`` at ``mpol=1`` and requires component-wise residuals below
``1e-12``. Radial magnetic energy uses two-point Gauss integration. This is
essential: the former midpoint rule admitted an alternating lambda hourglass
mode and its published refinement data were rejected. The corrected lambda
and pitch profiles are smooth, and a dedicated regression test assigns the
alternating mode finite energy.

Two manufactured pointwise-force fixtures isolate the reconstruction from the
nonlinear solve. A cylindrical finite-beta state with analytic radial
``B_z(s)`` and pressure balance converges at second order when ``ns`` doubles.
A theta-dependent self-similar tube carrying an exactly uniform Cartesian
field gives normalized Lorentz force below ``1e-12``. These fixtures show that
radial differentiation and nonaxisymmetric coordinates work independently;
they do not by themselves validate shaped solved states whose pointwise force
remains large.
The first corrected rotating-ellipse audit also exposed a missing axis
condition: the old state varied ``|B|`` by 9--20% over theta at ``s=0`` even
though those samples are one physical point. That freedom has been removed;
the canonical fixed-open records have since been regenerated with the
regular axis, while older shaped values remain invalid.

Independent nonaxisymmetric analytic fixtures
----------------------------------------------

``vmex.mirror.analytic`` contains validation data that never call an
equilibrium solve. ``RotatingEllipseParaxial`` maps a unit circle through a
flux-conserving ellipse whose major axis turns by 90 degrees from one end to
the other. A compensating field-line-label angle makes the first-order vacuum
identity vanish while preserving

.. math::

   X_{1c}Y_{1s}-X_{1s}Y_{1c}=\bar B/B_0.

It evaluates the Rodriguez-Helander-Goodman Appendix-C Riccati equation and
the independent general and magnetic-well-minimum formulas for
``(B20,B2c,B2s)``. Tests recover those coefficients from low-radius Fourier
samples and verify that the order-``r`` ``m=1`` field strength is zero. This is
the coefficient oracle for the native-spline fixed-boundary solve; it is not
itself an equilibrium.

``StraightFieldLineMirror`` implements the Agren-Savenko paraxial scalar
potential, on-axis field, Clebsch labels, straight nonparallel field lines,
and analytic elliptical sections. Its tests verify curl-free field, the
expected order-``(a/c)^2`` solenoidal and field-line truncation errors, axial
flux conservation, and

.. math::

   B_0(z)=\frac{B_0(0)}{1-z^2/c^2}, \qquad
   \mathcal E(z)=\frac{1+|z/c|}{1-|z/c|}.

Both fixtures require a thin tube and ``|z|<c``. The long-thin ordering treats
``beta`` and ``lambda=(a/L)^2`` as simultaneous small parameters;
``B/B_vac=sqrt(1-beta)`` is an asymptotic pressure-balance reference, not a
finite-beta solution or ellipticity prediction.

Native spline basis status
--------------------------

``vmex.mirror.splines.CubicBSplineBasis`` supplies the longitudinal
basis. Open mirrors use clamped knots, exact endpoint values, Gauss-Legendre
quadrature on every nonzero span, and exact Boehm knot insertion. Closed
hybrids use folded uniform periodic cubic splines and exact dyadic refinement.
Evaluation and two derivatives are JAX operations. Tests match SciPy,
reproduce cubics, preserve open and periodic curves under refinement, verify
partition of unity and C2 closure, and check JVP/VJP actions.

``SplineMirrorState`` and ``SplineMirrorBoundary`` store geometry and stream
function coefficients rather than sampled values. ``SplineMirrorDiscretization``
evaluates them on endpoint-augmented Gauss nodes before calling the shared
geometry and energy kernels, and applies side/end constraints plus the lambda
gauge in coefficient space. A quadratic flared tube uses 9 coefficients and 26
evaluation nodes versus 41 Chebyshev nodes; volume agrees to roundoff, total
energy agrees to ``5.0e-13`` relative, and an energy directional derivative
agrees with finite differences to ``1.9e-8`` relative.

The public ``solve_fixed_boundary`` minimizes the scalar-pressure energy
directly in the active spline coefficients. It fixes the side and end
coefficients, eliminates the weighted stream-function gauge, and uses the
shared host L-BFGS plus residual-Newton policy. The independent staggered first
variation is assembled on the quadrature grid and pulled back through the
spline evaluation matrix rather than reused from autodiff. The convenience
wrapper ``solve_fixed_boundary_from_radius(radius, config)`` builds a default
axisymmetric boundary of the requested LCFS radius and runs this solve in one
call.

For an ``ns=5`` finite-pressure, finite-current flared tube, both paths converge
in 59 iterations below ``ftol=1e-12``. Seven spline coefficients replace nine
Chebyshev axial nodes and reduce active variables from 45 to 31. Relative
differences are ``5.1e-7`` in energy, ``5.9e-6`` in volume, and ``3.2e-4`` in
center radius.

Independent solves at ``(ns,nxi)=(5,9),(7,13),(9,17)`` pair 7, 9, and 11
spline coefficients with the nodal grids. Energy error decreases
``5.12e-7 -> 1.46e-7 -> 5.60e-8``, volume error decreases
``6.50e-6 -> 2.24e-6 -> 9.85e-7``, and sampled radius RMS error decreases
``2.40e-4 -> 6.49e-5 -> 2.94e-5``. On the finest grid the Cartesian field and
``|B|`` differ by ``7.02e-4`` and ``3.05e-4`` RMS. Splines use roughly half the
active radius variables at all variational and independent weak residuals below
``1.3e-15``. Compact evidence is in the ``axisymmetric_refinement`` section of
``benchmarks/mirror_fixed_boundary.json``.

Open fixed-boundary systems use exact JAX Hessian-vector products and
matrix-free Newton at every size. Radius-only systems retain the inexpensive
radial/Fourier/axial tensor inverse. Finite-current nonaxisymmetric systems
freeze a sparse local Hessian at the start of Newton: columns are evaluated in
batches of at most 32, only same-field-channel terms with neighboring radial
rows and axial coefficient distance at most four are stored, and all poloidal
coupling is retained. Sparse LU is reused for every Newton step; no dense
Hessian is stored.

On the 591-variable analytic-seeded SFLM, the frozen local factor reaches true
relative residual ``9.18e-11`` in 660 Krylov iterations where the plain tensor
baseline stalls at ``7.83e-2`` after 2,000, at the same final energy and strong
force. Tangent and adjoint systems reuse the traceable separable
preconditioner; the host sparse factor remains a primal acceleration and is not
differentiated.

On the flared finite-beta case, knot refinement from 5 to 11 coefficients
reduces relative energy error against an ``nxi=17`` Chebyshev oracle from
``1.09e-6`` to ``5.14e-8`` and volume error from ``1.19e-5`` to ``2.18e-6``.
Both errors decrease at every refinement and all coefficient solves retain
variational and staggered residuals below ``9e-15``.

Nonaxisymmetric spline evidence
-------------------------------

Changing a prescribed spline boundary uses
``SplineMirrorDiscretization.transfer_boundary``. It rescales every nested
surface at spline collocation nodes before projection, instead of replacing
only the LCFS and risking crossed surfaces. The optimizer also rejects any
trial with a changed Jacobian sign, matching the regular VMEC-JAX merit policy.

The old zero-stream continuation produced a nonconvergent strong-force floor.
Supplied-field initialization provides the physical stream function and flux,
and ``impose_self_similar_cuts`` fixes each end section to scaled copies of its
LCFS. With these corrected cut semantics, the medium 90-degree rotating
ellipse has variational residual ``2.11e-16``, independent weak residual
``2.08e-16``, all-volume strong force ``2.67e-2`` device-normalized
(``1.60e-3`` under the primary minor-radius normalization), and normalized
divergence ``6.68e-15`` at LCFS radius ``0.12 m``. This is the current
release-candidate nonaxisymmetric fixed-boundary case.

``initialize_from_cartesian_field`` now keeps a supplied spline geometry fixed,
infers :math:`\Psi'(s)` from the surface-averaged axial flux, and obtains the
nonzero poloidal stream-function modes from the remaining contravariant field.
It accepts either Cartesian field samples or a point callable and performs no
coil construction or Biot--Savart integration. The independent Agren--Savenko
field is a paraxial-accuracy benchmark: it is an exact equilibrium only to
order :math:`(a/c)^2` (its solenoidal residual is :math:`O((a/c)^2)\approx
1.6\times10^{-3}`), so it is gated on its clean unconstrained bulk force and
on the refinement convergence of that force, not on a single all-volume
number. At ``(ns,mpol,elements)=(7,6,6)`` and LCFS radius ``0.10 m``, the
corrected-cut solve reaches variational residual ``1.71e-16`` and divergence
``7.04e-15``; its minor-radius-normalized bulk force is ``2.59e-3`` (well below
the ``0.05`` gate), while the all-volume and end-collar norms are ``1.68e-2``
and ``3.50e-2`` (``0.335`` and ``0.701`` device-normalized). The end collar is
the expected boundary layer where the analytic cut profile, which does not
satisfy the discrete equilibrium to machine precision, is held fixed at the two
end cuts. Refining once to ``(9,8,21,8)`` halves the bulk force to ``1.40e-3``
(ratio ``1.85``) and lowers the all-volume and end-collar norms to ``8.83e-3``
and ``1.69e-2``, so every zone converges under refinement; it therefore ships
as a validated paraxial benchmark, distinct from the release-candidate rotating
ellipse, whose section is an exact discrete flux surface.

.. image:: /_static/figures/mirror_fixed_boundary_3d.png
   :alt: Solved axisymmetric and 90-degree rotating-ellipse fixed-boundary mirrors coloured by LCFS field strength
   :width: 100%

Coefficient fixed-boundary gradients
------------------------------------

``spline_fixed_boundary_adjoint`` differentiates a scalar diagnostic through
the converged coefficient residual. Boundary and periodic-axis spline
coefficients, flux, conserved mass, and current remain differentiable. The transpose Hessian
action uses exact JAX reverse AD and the nonlinear iteration history is never
differentiated or stored. The root example
(``examples/mirror/mirror_fixed_boundary_nonaxisymmetric.py``) differentiates
rotating-ellipse volume with respect to a native boundary coefficient and
checks it against two fully reconverged equilibria.
For the corrected-cut rotating ellipse, the volume adjoint agrees with two
fully reconverged centered-difference solves to ``5.91e-10`` relative and its
transpose linear residual is ``2.30e-10``. An SFLM adjoint is not reported
because it is a paraxial-accuracy benchmark rather than a release-candidate
equilibrium: its analytic cut profile is fixed input data, not a solved exact
discrete flux surface, so a shape derivative through it is not a meaningful
optimization sensitivity.

``spline_fixed_boundary_tangent`` solves the complementary forward system
``F_u du = -F_p dp`` with exact residual JVPs and the same preconditioner. On a
nonaxisymmetric finite-current ``solve_lambda=True`` case, both radius and
stream-function tangents agree with two fully reconverged centered differences
within ``2e-4`` in relative state norm, with linear residual below ``1e-8``.
This establishes both open-spline derivative directions. On the closed
circular limit, periodic boundary and axis controls pass the parameter
JVP/VJP transpose identity and an adjoint volume derivative agrees across
three fully reconverged centered-difference steps. This validates the closed
algorithm for optimization. The supported circular-section racetrack, whose
frozen-junction strong-force reconstruction converges under refinement, is a
valid sensitivity target; the rotating-elliptical-section racetrack does not
yet carry a sensitivity claim while its independent strong-force reconstruction
plateaus on the separately scoped near-axis representation defect.

Public fixed-boundary inputs are
``SplineMirrorBoundary``, ``SplineMirrorState``, and
``SplineMirrorDiscretization``; public ``solve_fixed_boundary`` returns a
``SplineMirrorSolveResult``. CGL values remain available for quadrature and
evaluated-state parity tests, not as a second production state.

Axisymmetric free-boundary implicit gradients
---------------------------------------------

``free_boundary_adjoint`` differentiates the axisymmetric
exterior equilibrium with respect to a differentiable external-field callable,
axial flux, conserved mass profile, and axial current. The physical fixed point
contains the lateral LCFS and plasma-interior radii. The exterior Neumann BIE
eliminates vacuum unknowns, so its exact reverse-AD field and shape responses
enter the interface-stress rows directly. The transpose solve reuses the
separable primal plasma preconditioner and does not assemble a dense Jacobian
or retain nonlinear iterations.

The validation uses a differentiable curl-free paraxial mirror field and
checks both its strength/axial-curvature controls and a finite-pressure mass
direction against fully reconverged equilibria. Pressure changes through the
same conserved-mass and solved-volume relation as the primal. End-cut radii
and any central-pressure calibration target remain fixed. Coil-design
derivatives belong to the ESSOS integration layer; VMEX differentiates
only the supplied field object.

The adjoint consumes the same primal coefficient residual, coefficient
packing, and block preconditioner. It is supported through the 10% beta ceiling
and is checked against fully reconverged field and mass perturbations: the
implicit free-boundary derivative matches a reconverged finite difference to
``1.1e-10`` relative (transpose linear residual ``1.4e-9``). The nonaxisymmetric free-boundary derivative is
deliberately unavailable because
local Fourier-mode refinement failed; it will not be presented as a supported
gradient.

The one-shot mirror adjoint uses SciPy GMRES around exact JAX JVP/VJP actions
and inherits the placement of its primal arrays; no independent accelerator
solver policy is exposed because the one-shot derivative API cannot amortize a
SOLVAX solver's first compiled solve. Fixed- and free-boundary derivatives
solve the linearized converged coefficient residual and never retain or
differentiate the nonlinear iteration history.

Interpreting beta
-----------------

The equilibrium uses a VMEC-style conserved mass profile. Because geometry
changes pressure at fixed mass, the beta-scan driver adds one mass-amplitude
unknown and one central-pressure equation to the coupled nonlinear system.
Requested and achieved central beta therefore agree to the solve tolerance
without an outer sequence of complete solves. For the default profile
``p(s) = p0 (1-s)``, pressure vanishes at the LCFS, so a 10% central beta does
not imply a 10% edge-pressure jump or volume beta.

``summarize_axisymmetric_beta_scan`` reports:

* requested central beta,
* achieved central beta normalized by the reference vacuum field,
* volume-averaged beta,
* local central beta normalized by the solved plasma field,
* center radius and plasma/vacuum-side field,
* diamagnetic field ratio, and
* error against the paraxial estimate ``B/B_vac = sqrt(1-beta)``.

At the fine combined grid ``(ns,nxi,ntheta_panel)=(9,17,16)``, the default 10%
request reaches 10% central beta and 3.39% volume beta. The center radius
expands by 1.21%, while the central field falls by 4.38%. Thus field depression
is the more sensitive validation observable for this zero-edge-pressure
profile.

On that (legacy device-length-normalized) grid the 50% point reaches center
radius ``0.272554 m``, field ratio ``0.762687``, and volume beta ``0.216984``
with nonlinear residual ``8.31e-13``. The paraxial small-beta estimate is
intentionally shown but is not an accuracy reference at 50%. This coarser-grid
strong-force reconstruction is what originally held 50% at validation status;
it is superseded by the ``(13,25,13,24)`` fine-grid promotion run, which
converges every point 0–50% below the minor-radius bulk gate.

The finite-beta mirror trend follows the WHAM/Pleiades discussion in Frank et
al., `Confinement performance predictions for a high field axisymmetric tandem
mirror <https://doi.org/10.1017/S002237782510055X>`_. A checked Pleiades
Green-function reference at upstream commit ``0161abb3`` gives a 10% field
ratio of 0.952754 on a 51 by 101 grid. The production mixed-truncation
VMEX solve gives 0.956197, a 0.36% relative difference. Boundary
independent-reference curves remain the controlling diagnostics rather than
being replaced by this scalar comparison.
That study reports robust Pleiades equilibria for ``beta < 1`` and the expected
outward flux-surface expansion and diamagnetic field depression. Extending the
numerical gate to 50% therefore probes a scientifically relevant nonlinear
regime, but it is an equilibrium benchmark only: it does not establish flute,
firehose, mirror-mode, or kinetic stability.

The runnable coarse-grid continuation additionally reaches 80% requested
central beta with volume beta ``0.18755``, center-radius expansion ``15.13%``,
field ratio ``0.62075``, variational residual ``2.73e-15``, and normalized
pointwise force RMS ``1.25e-2``. Its plot compares the field ratio directly
with ``sqrt(1-beta)``. This point remains extended validation until a refined
grid campaign establishes convergence above the current 50% promotion set.

MGRID and vectorized ``xyz -> B`` callables share one external-field adapter.
MGRID interpolation tests remain in VMEX; filament sampling and
Biot-Savart parity tests live with ESSOS.

The free-space model has no artificial outer cylinder. Exterior trace order,
panel refinement, closed-surface compatibility, and memory cost remain
refinement diagnostics.

Open-exterior foundation
------------------------

:func:`vmex.mirror.build_closed_mirror_surface` closes a star-shaped
lateral LCFS with disks at both fixed-flux cuts. It stores outward ``n dA``
directly, including quadrature weights, so the disk center is regular and no
unit-normal division enters geometric identities. For axisymmetric equilibrium
grids, which intentionally store one theta node, the adapter supplies an
independent Cartesian angular quadrature; ``axisymmetric_ntheta`` controls its
resolution without adding redundant theta unknowns. ``ClosedMirrorSurface``
provides a unique collocation grid and an explicit map back to all quadrature
nodes, so repeated geometry never becomes duplicate BIE unknowns, and the same
unique nodes define a watertight outward-oriented triangular panel mesh whose
cylinder area and volume converge at second order under angular refinement.

The local Duffy primitive maps a vertex-singular triangle to a regular unit
square, interpolates density linearly, and is differentiable in panel geometry
and density. On a right triangle, orders ``2,4,8,16`` converge monotonically,
and order 16 matches the analytic constant-density single-layer integral within
``1.4e-14``. For axisymmetric data the angular panel nodes remain fully resolved
as sources but one target is evaluated per rotational orbit; at 57 unknowns and
1,762 panel vertices this reduced-target Jacobian takes 2.75 seconds instead of
67.5 seconds at unchanged 3.27% boundary error and condition number 19.1. Power
grading in the cap radius resolves the edge density without changing area or
volume, improving ``u=z`` recovery from 4.39% ungraded to 0.0216% at grade 3.5
(condition number 17.0).

The differentiable exterior solve reports net-flux compatibility, condition
number, and its full equation residual. The decaying exterior equation is
``S(q) + K(u-u_target) + u = 0``; it has no constant gauge freedom, and its
off-surface representation has the opposite sign. A zero-flux dipole MMS closes
it to ``3e-14`` with condition number below 5, boundary trace error decreases
``14.9% -> 5.44%`` over the regular two-grid test, and the exterior
field-gradient error at ``z=2`` decreases to 1.12%. Reconstructing the field
directly on the side exposes the leading limiter: the dipole lateral-field error
decreases ``48.4% -> 14.5% -> 5.28% -> 3.08%`` over four meshes, and linear
density interpolation on side triangles is the leading candidate. An opt-in
spectral side-density rule evaluates lateral Dirichlet and Neumann data with
global Fourier-Chebyshev interpolation on the same linear panel geometry; it
improves the medium dipole boundary-potential error from 5.44% to 1.19% and the
far-field gradient error from 1.12% to 0.72%, and is enabled with
``EXTERIOR_SPECTRAL_SIDE_DENSITY = True`` in the beta-scan example.

Source ownership is kept narrow: the merged ``exterior.py`` owns geometry,
reduction maps, side-panel topology, Duffy assembly, density interpolation, and
the Neumann solves. ``solve_axisymmetric_exterior_vacuum`` owns the complete
adapter: it closes the moving boundary, continues the plasma field through both
end cuts, cancels the supplied external normal field on the side, solves the
exterior Neumann problem, and returns the lateral total-field trace. Tangency
and a full shape JVP pass on the coupled adapter, which is the sole vacuum model
in the coupled axisymmetric free-boundary and beta-continuation drivers. On the
coarse ``(ns,nxi,ntheta_panel)=(5,7,8)`` two-coil case, beta 0 and 10% both
converge in seven nonlinear evaluations with maximum residuals ``7.93e-16`` and
``2.95e-15``, vacuum tangency below ``6.3e-17``, and the center radius
increasing from 0.252576 m to 0.255603 m. Restart files contain only the plasma
state, boundary, and pressure scale; the BIE potential is solved into
``result.vacuum_field.neumann_result`` rather than hot-started. The Neumann
compatibility projection is applied only on the artificial end caps, leaving
every lateral LCFS datum unchanged.

The beta-zero exterior resolution study at ``(ns,nxi,ntheta_panel)`` equal to
``(5,7,8)``, ``(7,13,12)``, and ``(9,17,16)`` gives center radii
``0.2525753, 0.2531506, 0.2531155`` m and axis fields
``0.0840027, 0.0835434, 0.0835623`` T; the last two grids agree within
``1.39e-4`` and ``2.26e-4`` relative while every force solve remains below
``5.8e-15``. Continuing every grid through 50% beta, the fine grid at beta 50%
gives center radius ``0.272554 m``, axis field ``0.063578 T``, volume beta
``0.216984``, and device-normalized all-volume/core force
``6.69e-2``/``1.50e-2``, with
medium-to-fine changes of ``0.137%`` in radius and ``1.02%`` in center field.
Those device-length force figures are a legacy diagnostic on this coarser grid.
They keep the 25% and 50% rows in extended validation rather than the supported
lane; see :doc:`/reference/capabilities`.

The coefficient solver uses a dense ``jacfwd`` only through 32 unknowns; larger
systems expose exact repeated JVP/VJP actions through a SciPy ``LinearOperator``
with no materialized Jacobian, globalized by a bounded trust region and polished
by a Newton--GMRES step using the fixed-solver spline preconditioner. Generic
interior and full-node virtual-casing adapters are intentionally left to
``virtual_casing_jax``; VMEX retains only the operator used by mirror
equilibria.

Nonaxisymmetric free-boundary mirrors are explicitly deferred. A historical
three-grid study reached roundoff nonlinear residuals, but local ``m=1``
observables changed by 73--81% between the first two grids and runtime grew from
293 to 2,995 seconds. Those states also predate the corrected magnetic-axis
regularity map, so the unsupported theta-dependent exterior has been removed
rather than presented as an equilibrium model; compact negative evidence remains
in ``benchmarks/mirror_free_boundary_nonaxisymmetric.json``. The fixed-boundary
rotating ellipse remains the release-candidate nonaxisymmetric case. The axisymmetric
``solve_beta_scan`` is the coefficient-native hot-start driver.

Cheaper boundary-limit approximations were tested and rejected: offset
collocation gave density-system condition numbers from ``1e6`` to ``1e19``, and
replacing each singular single-layer panel by an equal-area disk was stable but
only algebraic, retaining 8.3% boundary error at about 1,900 unknowns. The
implementation therefore follows local singular quadrature. Relevant numerical
foundations are Duffy's `vertex-singularity transform
<https://doi.org/10.1137/0719090>`_ and the distinction between smooth-surface
QBX error control and explicit corner treatment discussed by
`af Klinteberg and Tornberg <https://arxiv.org/abs/1603.08366>`_ and
`Helsing and Ojala <https://doi.org/10.1016/j.jcp.2008.06.022>`_.
