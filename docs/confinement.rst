Confinement physics: quasisymmetry, omnigenity, stability
=========================================================

The pages :doc:`equations` and :doc:`algorithms` derive the *solver* physics —
the energy functional whose stationary point is an equilibrium, and the
numerics that find it. This page derives the **target functionals** that a
stellarator-design campaign puts in front of that equilibrium: the confinement
and stability metrics of :mod:`vmex.core.omnigenity`,
:mod:`vmex.core.optimize`, :mod:`vmex.core.nyquist`,
:mod:`vmex.core.stability` and :mod:`vmex.core.bootstrap`. It is the
first-principles companion to :doc:`objectives`, which catalogs the same
quantities as ready-to-use optimizer terms.

The unifying object is the field strength :math:`|B|` expressed in **Boozer
coordinates**: quasisymmetry, omnigenity, and the neoclassical transport that
they control are all statements about the angular structure of :math:`|B|` at
fixed flux label, and are cleanest in the coordinates in which field lines are
straight *and* the parallel current is a flux function.

.. contents:: On this page
   :local:
   :depth: 1


Boozer coordinates
------------------

Motivation
~~~~~~~~~~

Guiding-center orbits — and therefore neoclassical transport, fast-ion
confinement, and the bootstrap current — depend on the magnetic geometry almost
entirely through the variation of :math:`|B|` along and across field lines.
Boozer coordinates :math:`(s, \theta_B, \zeta_B)` are the flux coordinates that
make this variation maximally transparent: field lines are straight,

.. math::

   \frac{d\theta_B}{d\zeta_B} = \iota(s),

and the covariant representation of :math:`\mathbf B` collapses to the two
surface functions

.. math::

   \mathbf B = \nabla\psi\times\nabla\theta_B + \iota\,\nabla\zeta_B\times\nabla\psi
             = I(s)\,\nabla\theta_B + G(s)\,\nabla\zeta_B + \beta_*\,\nabla\psi,

where :math:`2\pi I(s)` is the toroidal current inside :math:`s`, :math:`2\pi
G(s)` the poloidal current outside it (VMEC's ``buco`` and ``bvco``,
:func:`~vmex.core.fields.surface_currents`), and :math:`\psi=\Phi/2\pi` is
the toroidal flux per radian. Because :math:`I` and :math:`G` are flux
functions, the magnetic differential equation for the coordinate shift is
linear and can be solved surface by surface.

Construction
~~~~~~~~~~~~

Starting from the straight-field-line (VMEC) angles :math:`(\theta,\zeta)` with
the renormalized poloidal angle :math:`u=\theta+\lambda`, the Boozer angles
differ by a single periodic scalar :math:`\nu(s,\theta,\zeta)`:

.. math::

   \theta_B = \theta + \lambda + \iota\,\nu, \qquad
   \zeta_B  = \zeta + \nu .

The shift :math:`\nu` follows from the covariant field. Introduce the periodic
part :math:`w` of the Boozer generating potential, fixed by

.. math::

   \frac{\partial w}{\partial\theta} = B_\theta, \qquad
   \frac{\partial w}{\partial\zeta}  = B_\zeta,

with :math:`B_\theta,B_\zeta` the covariant components on the surface
(:func:`~vmex.core.fields.magnetic_fields`). In
:func:`~vmex.core.omnigenity.boozer_bmnc_state` this is inverted
spectrally: after an FFT, the non-axisymmetric (:math:`m\ne 0`) harmonics of
:math:`w` come from :math:`B_\theta` and the axisymmetric (:math:`m=0`)
harmonics from :math:`B_\zeta`, matching the mode split of the Fortran
``booz_xform``. The coordinate shift is then

.. math::

   \nu = \frac{w - I(s)\,\lambda}{G(s) + \iota(s)\,I(s)} .

Finally the Boozer :math:`|B|` spectrum is obtained by the angle-transform
quadrature, weighting :math:`|B|` (evaluated on the VMEC grid) by the Jacobian
of the angle map:

.. math::

   \hat B^{B}_{mn}(s) = \Bigl\langle\,|B|\,
      \cos\!\bigl(m\,\theta_B - n\,\zeta_B\bigr)\,
      \frac{\partial(\theta_B,\zeta_B)}{\partial(\theta,\zeta)}\Bigr\rangle,

the angle bracket being the surface average over the original
:math:`(\theta,\zeta)` grid. These are the ``bmnc_b`` coefficients (physical
mode numbers ``xm_b``, ``xn_b``) consumed by every metric below.

Two implementations share these equations. The host driver
:func:`vmex.core.boozer.run_booz_xform` calls ``booz_xform_jax`` on a
``wout_*.nc`` file and writes a standard ``boozmn_*.nc`` (used by ``vmec
--booz`` and by :func:`~vmex.core.optimize.quasi_isodynamic_residual_from_wout`).
The traceable :func:`~vmex.core.omnigenity.boozer_bmnc_state` evaluates the
*same* transform in pure ``jax.numpy`` directly from the solver's internal
half-mesh field tables, so the Boozer spectrum — and any metric built on it —
carries exact implicit gradients. The two agree to :math:`\sim10^{-6}` on the
dominant modes.


Quasisymmetry
-------------

Confinement rationale
~~~~~~~~~~~~~~~~~~~~~~

A field is **quasisymmetric** when :math:`|B|` depends on the two Boozer angles
only through a single linear combination,

.. math::

   |B| = |B|\bigl(s,\; M\theta_B - N\zeta_B\bigr),

for a fixed integer helicity :math:`(M,N)`. The magnitude, not the vector
field, need only be symmetric: this is enough for the guiding-center Lagrangian
to acquire an ignorable angle, hence a conserved canonical momentum and confined
collisionless orbits — the same confinement a true axisymmetric tokamak enjoys,
but in a compact 3D device. The helicity fixes the family
(:class:`~vmex.core.optimize.QuasisymmetryRatioResidual`, ``helicity_n`` in
units of ``nfp``):

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - family
     - :math:`(M,N)`
     - :math:`|B|` contours in Boozer angles
   * - QA (quasi-axisymmetric)
     - :math:`(1,0)`
     - close poloidally (tokamak-like)
   * - QH (quasi-helical)
     - :math:`(1,\pm\mathrm{nfp})`
     - close helically
   * - QP (quasi-poloidal)
     - :math:`(0,1)`
     - close toroidally

The two-term residual
~~~~~~~~~~~~~~~~~~~~~~

Rather than Fourier-filter the Boozer spectrum, ``vmex`` uses the
Landreman–Paul *two-term* local residual, which is an exact pointwise
diagnostic of the same condition and needs no mode truncation. On each
requested surface, sampled on a uniform :math:`(\theta,\phi)` grid,

.. math::

   f_{\mathrm{QS}} =
   \frac{(\mathbf B\times\nabla B\cdot\nabla\psi)\,(N-\iota M)
       - (\mathbf B\cdot\nabla B)\,(M G + N I)}{B^{3}},

with :math:`M=` ``helicity_m``, :math:`N=` ``helicity_n``\ :math:`\times`\
``nfp``, and :math:`G,I` the Boozer covariant averages ``bvco``/``buco``. The
key algebraic fact is that :math:`f_{\mathrm{QS}}` **vanishes identically iff**
:math:`|B|` is quasisymmetric with helicity :math:`(M,N)`; there is no residual
symmetry-breaking harmonic left to penalize. The flux-surface sum
:math:`\sum f_{\mathrm{QS}}^2`, weighted by the surface measure
:math:`\sqrt{\mathrm{nfp}\,\Delta\theta\,\Delta\phi\,|\sqrt g|/V'}`, reproduces
simsopt's ``QuasisymmetryRatioResidual`` A/B bit-for-bit. Kept in
Gauss–Newton (per-point) form, it feeds the least-squares driver as an exact
residual vector rather than a pre-summed scalar. The metric is evaluated from
the parity-proven wout tables (:mod:`vmex.core.nyquist`) and also exposes a
traceable ``residuals_state`` lane, so the *same* term optimizes under both
``jac=None`` and ``jac="implicit"`` (see :doc:`objectives`).


Omnigenity and quasi-isodynamicity
----------------------------------

Omnigenity generalizes quasisymmetry: it asks only that the **bounce-averaged
radial drift of trapped particles vanish**, without requiring a symmetry of
:math:`|B|`. Equivalently (Cary & Shasharina 1997), the second adiabatic
invariant

.. math::

   \mathcal J_\parallel(s,\alpha;B^*) = \oint v_\parallel\,d\ell,
   \qquad v_\parallel \propto \sqrt{1 - B/B^*},

is a flux function — independent of the field-line label :math:`\alpha` at every
trapping level :math:`B^*`. Every quasisymmetric field is omnigenous; the
converse is not true, which leaves omnigenity the larger (and for
poloidally-closed contours, bootstrap-suppressing) design space. A **quasi-
isodynamic** (QI) field is an omnigenous field whose :math:`|B|` contours close
**poloidally** (:math:`M=0`), so that the trapped-particle precession is purely
poloidal and the bootstrap current is small by construction — the target of the
nfp1–nfp4 decks in ``examples/data/``.

The constructed-QI target
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because :math:`\mathcal J_\parallel` uniformity is awkward to differentiate,
:class:`~vmex.core.omnigenity.QIResidual` implements the equivalent
level-set conditions of the *constructed-QI-target* method of Goodman *et al.*
(2023). On each surface, :math:`|B|` is sampled along Boozer field lines
:math:`\theta_B=\alpha+\iota\phi_B` over one field period, and three families of
residual — each an **exact zero of an exactly QI field** — are stacked:

* **Bounce-distance uniformity** (``well_weight``). For every trapping level
  :math:`B^*`, the distance :math:`\delta(\alpha,B^*)` between the two monotone
  branches of the magnetic well, minus its field-line average. This is the
  Cary–Shasharina condition (constant well width at fixed :math:`B^*` across
  field lines) that Goodman's "shuffle" step enforces; the branch envelopes are
  built from smooth running-maximum occupancy integrals so the term is
  differentiable.

* **Extremum alignment** (``extremum_weight``). The per-field-line
  :math:`B_{\min}` and :math:`B_{\max}` minus their field-line averages —
  poloidal closure of the extremal :math:`|B|` contours (Goodman's "align the
  maxima" step; also the flat-:math:`B_{\max}` condition of Dudt *et al.* 2024).

* **Single-well monotonicity** (``squash_weight``). The pointwise distance
  between :math:`|B|` and its monotone branch envelopes — Goodman's "squash"
  distance, which penalizes side wells (more than one magnetic well per period).

Every operation (sigmoid occupancies, running maxima, level-space quadrature) is
smooth or piecewise-smooth, so the residual is jit/grad/jvp-transparent and QI
optimization runs with the exact implicit adjoint, exactly like the QS residual.

.. _confinement-qi-fidelity:

Metric fidelity: report the wout-based residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The traceable :class:`~vmex.core.omnigenity.QIResidual` is built for
*gradient flow*, not for *labeling* a configuration. Its level-space envelopes
and finite sampling make it an excellent descent direction but an **optimistic
absolute scale**: driven hard, it can report values well below what an
independent Boozer analysis confirms. When quoting "how QI is this
equilibrium?", use the wout/Boozer-based
:func:`~vmex.core.optimize.quasi_isodynamic_residual_from_wout` (host
``booz_xform_jax``, finite-difference-only) — the same construction evaluated on
the fully resolved Boozer spectrum of the written ``wout``. The recommended
workflow is therefore: **optimize** with the traceable residual under
``jac="implicit"``, then **report** the wout-lane residual. The two agree in
ranking; they diverge in absolute value precisely where the traceable form is
being pushed below its trustworthy range.


Ideal MHD stability: Mercier and the magnetic well
--------------------------------------------------

Mercier criterion
~~~~~~~~~~~~~~~~~~

The Mercier criterion is the ideal-MHD stability condition for localized
interchange modes resonant on a rational surface — the toroidal generalization
of the Suydam criterion. Following Glasser–Greene–Johnson, the surface is
Mercier-**stable** where

.. math::

   D_{\mathrm{Merc}} = D_{\mathrm{shear}} + D_{\mathrm{well}}
                     + D_{\mathrm{curr}} + D_{\mathrm{geod}} > 0 ,

a decomposition into four flux-surface integrals with distinct physical origin.
:func:`~vmex.core.nyquist.mercier_and_jxb` ports VMEC2000 ``mercier.f`` term
for term (with :math:`{}' \equiv d/d\psi`, :math:`p` the pressure,
:math:`\iota` the transform, :math:`V'` the differential volume, and
:math:`I_\varphi` the enclosed toroidal current):

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - term
     - sign
     - origin
   * - :math:`D_{\mathrm{shear}} = \tfrac14(\iota')^2`
     - :math:`\ge 0`
     - magnetic shear (always stabilizing)
   * - :math:`D_{\mathrm{well}}`
     - either
     - pressure gradient :math:`\times` magnetic well :math:`V''`
   * - :math:`D_{\mathrm{curr}}`
     - either
     - parallel-current / kink drive :math:`\propto \iota'`
   * - :math:`D_{\mathrm{geod}}`
     - :math:`\le 0`
     - geodesic curvature (always destabilizing)

The well term carries the sign of :math:`p'\,(V'' - p'\langle\cdots\rangle)`:
with the pressure decreasing outward (:math:`p'<0`), a magnetic well
(:math:`V''<0`, volume that decreases toward the edge) is stabilizing. The
geodesic term is a manifestly non-positive Schwarz-inequality remainder. Because
the individual pieces involve radial derivatives of surface averages, the two
surfaces nearest the axis and the edge carry the usual numerical noise; a
practical objective penalizes ``min(DMerc[2:-1], 0)``. ``vmex`` exposes the
reporting profile as :func:`~vmex.core.optimize.d_merc`, evaluated through the
parity-proven wout engine.  The symmetric live-state counterpart
:func:`~vmex.core.stability.d_merc_state` is a pure-JAX port of the same
``jxbforce.f``/``mercier.f`` path for ``jit``/AD use and agrees with the wout
profile to floating-point round-off.  For optimization,
:func:`~vmex.core.stability.mercier_stability_residual` excludes ``[0:2]``
and the edge and returns
``smoothing * softplus((margin - DMerc) / smoothing)``; targeting it to zero
penalizes unstable (negative) ``DMerc`` with a smooth gradient.  At finite
``smoothing`` the residual is positive, rather than exactly zero, on stable
surfaces but decays exponentially with the stability margin.  Both profile
lanes retain VMEC's near-axis and edge limitations; the traceable Mercier
lane does not yet support ``lasym = True``.

Parallel current and resistive interchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~vmex.core.stability.jdotb_state` exposes the same full-mesh
``jdotb = <J.B>`` profile that VMEC2000 computes in ``jxbforce.f``, but as a
pure-JAX function of the converged state.  Its WOUT normalization and
extrapolated axis/edge entries are unchanged, so it can be used directly as
an implicit-differentiation current objective. This is a direct port of the
``jxbforce.f`` current reconstruction, distinct from the cheaper
``bootstrap.vmec_j_dot_B`` force-balance identity used by the Redl mismatch.
For optimization, :func:`~vmex.core.stability.jdotb_residual` selects
``jdotb[2:-1]`` to exclude the usual axis/edge entries.
This lane supports ``lasym = True``.  A converged finite-pressure,
up-down-asymmetric tokamak agrees with live VMEC2000 to ``1.52e-3`` relative
over ``[2:-1]``; its regression gate is ``2e-3``.

Assuming the ideal prerequisite :math:`D_{\rm Merc}>0`, the
Glasser--Greene--Johnson necessary condition for local resistive interchange
stability is :math:`D_R \leq 0`.  In the VMEC Mercier normalization, with
:math:`S=d\iota/d\Phi` and
:math:`D_{\rm shear}=S^2/4`, Landreman--Jorge's relation is

.. math::

   D_R = -D_{\rm Merc}
       + \frac{(H-S^2/2)^2}{S^2},\qquad
   H=S\left(t_{JB}-\frac{\langle\mu_0J\cdot B\rangle}
                              {\langle B^2\rangle}t_{BB}\right).

:func:`~vmex.core.stability.glasser_d_r_state` reuses the traceable Mercier
surface integrals and the VMEC-consistent ``jdotb/bdotb`` averages to evaluate
this expression.  Exact zero-shear entries are set to zero because the
criterion is undefined there; a positive ``shear_epsilon`` provides a smooth,
explicit regularization for optimization.  The
:func:`~vmex.core.stability.glasser_stability_residual` helper selects
``D_R[2:-1]`` and smoothly penalizes values above ``-margin``; it defaults to
``shear_epsilon=1e-8`` so zero-shear optimization seeds remain finite.
It must be combined with ``mercier_stability_residual`` so an
ideal-Mercier-unstable surface is never accepted based on :math:`D_R` alone.
After optimization, use ``mercier_shear_state`` to verify
``abs(S) >> shear_epsilon`` on every target surface; regularization makes the
numerics finite but cannot establish GGJ stability at zero shear.
VMEC2000 does not write ``D_R`` itself.  A live `DCON/GPEC
<https://github.com/PrincetonUniversity/GPEC>`_ evaluation independently
reproduces the symmetric VMEC normalization at ``ns=51`` (``D_I`` maximum
absolute difference ``9.10e-4`` and ``D_R`` ``8.63e-5`` over normalized
poloidal flux ``[0.1, 1)``).  The same test on an
up-down-asymmetric tokamak exposed unresolved sensitivity in :math:`H`:
at ``ns=201`` the candidate reconstruction's normalized ``D_I`` differs by
at most ``1.85e-2`` over normalized poloidal flux ``[0.2, 0.9]``, but
``D_R`` differs by ``1.49e-2`` and can change sign near marginality.
On that same input, all four geometry families agree with VMEC2000 to
``2.49e-10`` relative or better, while the interior VMEX/VMEC2000 ``DMerc``
relative difference is ``18.6`` with sign disagreements.  This isolates the
problem to the LASYM Mercier reconstruction rather than the equilibrium.
Consequently :func:`~vmex.core.stability.d_merc_state` and
:func:`~vmex.core.stability.glasser_d_r_state` still reject ``lasym = True``;
the independently validated ``jdotb`` lane is available.  A 3-D LASYM
extension additionally requires JMC or an equivalent nonaxisymmetric
reference.

Magnetic well
~~~~~~~~~~~~~~

The dominant stabilizing ingredient of :math:`D_{\mathrm{well}}` — the sign of
:math:`V''(s)` — is also useful on its own as a cheap, fully traceable proxy.
:func:`~vmex.core.optimize.magnetic_well` returns the finite-difference
vacuum-well measure

.. math::

   W = \frac{V'(0) - V'(1)}{V'(0)},

with :math:`V'=dV/ds` extrapolated from the half-mesh differential volume
:math:`vp` (VMEC ``bcovar.f``). Positive :math:`W` means :math:`V'` decreases
outward — a magnetic well, favorable for interchange stability — matching
simsopt's ``vacuum_well``. Being a pure ``(state, runtime)`` function it carries
exact implicit gradients and is a cheaper Mercier-adjacent target. Near-axis
analytic context for both measures is in Landreman–Jorge (2020) and
Kim–Jorge–Dorland (2021); see :doc:`references`.

Ideal ballooning
~~~~~~~~~~~~~~~~~

Interchange stability bounds only the :math:`n\to\infty`, radially-localized
limit. The complementary *ballooning* limit — high toroidal mode number,
extended along the field line — is provided by :mod:`vmex.core.stability`
as a fully differentiable eigenvalue objective (a JAX port of the COBRA
solve in the Gaur *et al.* formulation). It solves the self-adjoint
field-line ODE eigenproblem

.. math::

   \frac{d}{d\eta}\!\Bigl(g\,\frac{dX}{d\eta}\Bigr) + c\,X = \lambda\,f\,X,
   \qquad X(\pm\eta_b)=0,

along the straight-field-line angle :math:`\eta`, where :math:`g` is the
line-bending term, :math:`c` the pressure/curvature drive, and :math:`f>0` the
inertia; :math:`\lambda = (\gamma a_N/v_A)^2 > 0` flags instability.
:func:`~vmex.core.stability.ballooning_growth_rate` reduces the batched
eigenvalues to a smooth ``softmax`` scalar built to be driven negative as a
stable-by-construction constraint. The full coefficient definitions are in the
:mod:`~vmex.core.stability` module docstring and the usage recipe in
:doc:`objectives`.


Bootstrap current (Redl)
------------------------

The self-consistent parallel current a stellarator generates from its own
pressure gradient — the bootstrap current — sets the achievable :math:`\beta`
and, in a QI device, must be kept small. :mod:`vmex.core.bootstrap`
provides a differentiable evaluation and a self-consistency loop (reproducing
Landreman–Buller–Drevlak, arXiv:2205.02914).

Two independent estimates of :math:`\langle\mathbf J\cdot\mathbf B\rangle` are
compared. The **equilibrium** value follows from the exact MHD identity

.. math::

   \langle\mathbf J\cdot\mathbf B\rangle(s) =
   \frac{\langle B^2\rangle\,I'(s) + \mu_0\,I(s)\,p'(s)}{2\pi\,\psi_a},
   \qquad I(s) = \mathrm{signgs}\,\frac{2\pi}{\mu_0}\,\mathrm{buco}(s),

(:func:`~vmex.core.bootstrap.vmec_j_dot_B`). The **kinetic** value is the
Redl *et al.* (2021) analytic closure
(:func:`~vmex.core.bootstrap.j_dot_B_redl`), a fit in the effective trapped
fraction :math:`f_t` and the Sauter collisionalities, with the quasisymmetry
isomorphism :math:`\iota\to\iota-\mathrm{nfp}\,\mathrm{helicity\_n}` applied as
in simsopt. The trapped fraction itself uses the singularity-removing
substitution :math:`y=\sqrt{1-\lambda B_{\max}}` in

.. math::

   f_t = 1 - \tfrac34\langle B^2\rangle
         \int_0^{1/B_{\max}}\!
         \frac{\lambda\,d\lambda}{\langle\sqrt{1-\lambda B}\rangle},

evaluated with fixed-order Gauss–Legendre quadrature so the whole chain stays
differentiable. Their normalized mismatch is the residual
:class:`~vmex.core.bootstrap.RedlBootstrapMismatch` (the exact formula and
the finite-beta profile conventions are in :doc:`equations`); driving it to
zero, optionally with ``current_dofs`` freed, yields a current profile
consistent with the plasma the equilibrium describes.
