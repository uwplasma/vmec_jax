The variational problem
=======================

VMEX solves the same problem as VMEC2000: find the stationary point of the
ideal-MHD energy functional over nested flux surfaces, in the inverse
(moment) representation of Hirshman & Whitson (1983). This page states that
problem and defines the force residuals ``fsqr/fsqz/fsql`` whose norms decide
convergence; the discrete representation is in
:doc:`spectral-representation` and the iteration that finds the root is in
:doc:`iteration`.

Ideal MHD equilibrium
---------------------

The ideal MHD equilibrium is defined by:

.. math::

   \nabla p = \mathbf{J} \times \mathbf{B},

with Maxwell's equations (in magnetostatic form):

.. math::

   \nabla \cdot \mathbf{B} = 0, \qquad
   \mathbf{J} = \frac{1}{\mu_0}\nabla \times \mathbf{B}.

The pressure is a **flux function**: :math:`p = p(s)` and is specified by the
VMEC input profiles. VMEC (and VMEX) use pressure in units of
:math:`\mu_0\,\mathrm{Pa}` so that :math:`p` has the same units as :math:`B^2`.

Energy principle (VMEC formulation)
-----------------------------------

VMEC solves for a stationary point of the ideal-MHD energy functional. In
physical units,

.. math::

   W = \int \left(\frac{B^2}{2\mu_0} + \frac{p}{\gamma-1}\right) dV,

where :math:`\gamma` is the ratio of specific heats (VMEC input ``GAMMA``;
``GAMMA = 0`` selects the prescribed-pressure limit). In VMEC's internal
units (:math:`p` in :math:`\mu_0\,\mathrm{Pa}`, angles normalized by
:math:`2\pi`) this becomes

.. math::

   W = \frac{1}{(2\pi)^2}\int \left(\frac{B^2}{2} + \frac{p}{\gamma-1}\right) dV.

For fixed boundary and fixed flux profiles, the first variation of :math:`W`
with respect to a displacement :math:`\boldsymbol{\xi}` of the flux surfaces
is

.. math::

   \delta W = -\int \left(\mathbf{J}\times\mathbf{B} - \nabla p\right)
              \cdot \boldsymbol{\xi}\; dV,

so :math:`W` is stationary exactly at ideal-MHD force balance.

The Hirshman–Whitson moment method
----------------------------------

Hirshman & Whitson (1983) discretize this variational problem in *inverse*
form: the unknowns are the Fourier **moments**
:math:`R_{mn}(s), Z_{mn}(s), \lambda_{mn}(s)` of the flux-surface geometry,
not field values on a spatial grid. Varying :math:`W` with respect to each
moment gives one Euler–Lagrange equation per ``(m,n)`` mode and radial
surface — the *spectral force residuals*

.. math::

   F_{R,mn}(s) = -\frac{\delta W}{\delta R_{mn}(s)}, \qquad
   F_{Z,mn}(s) = -\frac{\delta W}{\delta Z_{mn}(s)}, \qquad
   F_{\lambda,mn}(s) = -\frac{\delta W}{\delta \lambda_{mn}(s)},

and the equilibrium is the root :math:`F = 0`. Practically, the residuals
are evaluated by synthesizing the geometry on the angular grid
(:func:`~vmex.core.transforms.fourier_to_real`), forming the real-space
force kernels (:func:`~vmex.core.forces.mhd_force_kernels`), and
projecting back onto the Fourier basis with the weighted DFT
(:func:`~vmex.core.transforms.tomnsps`); the full pipeline is
:func:`~vmex.core.forces.spectral_mhd_forces`.

The iteration is a preconditioned steepest descent on :math:`W` — a damped
second-order Richardson ("momentum") scheme

.. math::

   \ddot{\mathbf{x}} + \frac{1}{\tau}\dot{\mathbf{x}} = P^{-1} F(\mathbf{x}),

with :math:`\mathbf{x}` the stacked moments and :math:`P` the preconditioner
(:mod:`vmex.core.step`; discretization in :doc:`iteration`). Because
:math:`F = -\nabla_{\mathbf{x}} W`, every accepted step decreases :math:`W`
monotonically (up to the momentum transient) and the descent stops only at a
stationary point of the energy.
See References [1-3] in :doc:`/project/references` for the original VMEC
formulation.

Force balance in residual form
------------------------------

VMEC evaluates the force balance in real space, then transforms the residual
forces back to Fourier space. In VMEC2000, these residuals are packaged as
``tomnsps`` Fourier arrays and projected into three directions:

- :math:`F_R`: radial (R) force balance residual,
- :math:`F_Z`: vertical (Z) force balance residual,
- :math:`F_\lambda`: stream-function (lambda) residual.

Real-space force kernels (``forces.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each real-space residual is assembled from three kernels in the
Hirshman–Whitson "A/B/C" form,

.. math::

   F_R = A_R - \partial_\theta B_R + \partial_\zeta C_R, \qquad
   F_Z = A_Z - \partial_\theta B_Z + \partial_\zeta C_Z,

.. math::

   F_\lambda = -\partial_\theta B_\lambda + \partial_\zeta C_\lambda,

where the angular derivatives are taken **spectrally**: the kernels are
projected onto the Fourier basis with the derivative trig tables of
``tomnsps``, so :math:`\partial_\theta \to m` and
:math:`\partial_\zeta \to n\,\mathrm{NFP}` multiplications. In terms of the
half-mesh quantities of ``bcovar.f`` — the total pressure
:math:`\mathrm{bsq} = |B|^2/2 + p`, the interpolated radius :math:`r_{12}`,
the Jacobian factor :math:`\tau`, and the products
:math:`\sqrt{g}\,B^uB^u,\ \sqrt{g}\,B^uB^v,\ \sqrt{g}\,B^vB^v` — the
:math:`A` kernels (VMEC ``armn/azmn``) carry the radial finite difference of
the magnetic + thermal energy flux plus the toroidal-curvature term
:math:`-\sqrt{g}\,B^vB^v\,R`; the :math:`B` kernels (``brmn/bzmn``) the
poloidal-metric couplings; and the :math:`C` kernels (``crmn/czmn``) the
toroidal-metric couplings. Odd-m planes carry the internal :math:`\sqrt{s}`
representation and its chain-rule terms (the discrete
:math:`d\sqrt{s}/ds` factor ``dshalfds = 0.25``). Implemented in
:func:`~vmex.core.forces.mhd_force_kernels` (R/Z blocks) and
:func:`~vmex.core.forces.lambda_force_kernels` (the covariant
:math:`B_u, B_v` lambda-force block of ``bcovar.f``); the full real-space
pipeline is :func:`~vmex.core.forces.mhd_forces` and the projection to
spectral residuals is :func:`~vmex.core.forces.spectral_mhd_forces`.

What "converged" means (``FSQR/FSQZ/FSQL``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The residuals are combined into the scalar norms of the VMEC screen output:

.. math::

   \mathrm{FSQR} = r_1\,\mathrm{fnorm}\,\lVert F_R \rVert^2, \qquad
   \mathrm{FSQZ} = r_1\,\mathrm{fnorm}\,\lVert F_Z \rVert^2,

.. math::

   \mathrm{FSQL} = \mathrm{fnormL}\,\lVert F_\lambda \rVert^2,

where ``fnorm``, ``fnormL``, and ``r1`` are the VMEC normalization factors
computed from ``bcovar`` (half-mesh metrics + ``bsup``/``bsub``).
VMEX reproduces these scalars from the same internal quantities to
match VMEC2000's per-iteration printout. Convergence is declared when all
three simultaneously satisfy ``fsqr, fsqz, fsql <= FTOL`` (the active
``FTOL_ARRAY`` entry); the check is implemented in
:mod:`vmex.core.residuals` (``residue.f90``).

Pressure and beta
-----------------

VMEC reports thermal and magnetic energy scalars in ``wout``:

.. math::

   W_B = \frac{1}{(2\pi)^2}\int \frac{B^2}{2}\,dV, \qquad
   W_P = \frac{1}{(2\pi)^2}\int p\,dV.

The total volume-averaged beta is computed by VMEC as:

.. math::

   \beta_{\mathrm{total}} = \frac{W_P}{W_B}.

VMEX follows the same normalization when emitting ``wout`` files.

Source references
-----------------

- ``vmecpp/docs/the_numerics_of_vmecpp.pdf`` (VMEC++ numerics notes; VMEC2000
  conventions and formulas).
- VMEC2000 source in ``STELLOPT/VMEC2000/Sources``.
- Bibliography: :doc:`/project/references`.
