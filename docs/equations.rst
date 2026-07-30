Equations and derivations
=========================

This page states the **explicit equations** solved (or approximated) by
``vmex`` and connects them to the VMEC2000 formulation. The goal is to make
the physics, the coordinate conventions, and the force-balance residuals fully
transparent so that parity checks can be done equation-by-equation.

Notation and conventions
------------------------

We use VMEC's curvilinear flux coordinates:

- :math:`s \in [0,1]`: normalized toroidal flux label (VMEC radial coordinate).
- :math:`\theta \in [0,2\pi)`: poloidal angle.
- :math:`\zeta \in [0,2\pi)`: *field-period* toroidal angle.
- :math:`\phi_{\mathrm{phys}} = \zeta/\mathrm{NFP}`: physical toroidal angle.

The Fourier phase convention is:

.. math::

   \mathrm{phase}(m,n;\theta,\zeta) = m\theta - n\zeta,

where :math:`n` is the field-period toroidal mode number
(VMEC stores :math:`xn = n\,\mathrm{NFP}` in ``wout``).

VMEC weighted DFT tables (``fixaray``)
--------------------------------------

VMEC does **not** use a plain FFT for its force/residual transforms. Instead,
``fixaray`` builds weighted trig tables on a symmetry-aware grid and applies
explicit normalization factors. Let :math:`\theta_i` be the VMEC theta grid
over :math:`[0,\pi]` (with endpoint half-weights) and :math:`\zeta_k` the zeta
grid over one field period. VMEC defines

.. math::

   \mathrm{dnorm} = \frac{1}{n_\zeta\,(n_{\theta2}-1)},\qquad
   \mathrm{mscale}_0 = 1,\quad \mathrm{mscale}_{m>0}=\sqrt{2},

and the weighted cosine table

.. math::

   \mathrm{cosmui}_{i,m} = \mathrm{dnorm}\,w_i\,\mathrm{mscale}_m \cos(m\theta_i),

with :math:`w_0=w_{n_{\theta2}-1}=1/2` and :math:`w_i=1` elsewhere. The sine
table is defined analogously, with the same weights and ``mscale``. Zeta tables
use ``nscale`` (also :math:`\sqrt{2}` for :math:`n>0`) and, for derivative
terms, include the field-period multiplier :math:`n\,\mathrm{NFP}`:

.. math::

   \mathrm{cosnvn}_{k,n} = (n\,\mathrm{NFP})\,\mathrm{cosnv}_{k,n}, \qquad
   \mathrm{sinnvn}_{k,n} = -(n\,\mathrm{NFP})\,\mathrm{sinnv}_{k,n}.

``vmex`` uses these tables in ``tomnsps`` so that the Fourier-space force
arrays exactly match VMEC2000. See References [4-6] for the original VMEC2000
tables and the VMEC++ DFT/basis discussion.

Two-stage DFT for ``tomnsps``
-----------------------------

VMEC's ``tomnsps`` uses a **separable real basis** in :math:`\theta` and
:math:`\zeta`. For a real-space kernel :math:`F(\theta_i,\zeta_k)` defined on the
VMEC grid, the weighted theta projection is

.. math::

   \tilde F^{(c)}_{m}(\zeta_k) = \sum_{i=0}^{n_{\theta2}-1} F(\theta_i,\zeta_k)\,\mathrm{cosmui}_{i,m},

.. math::

   \tilde F^{(s)}_{m}(\zeta_k) = \sum_{i=0}^{n_{\theta2}-1} F(\theta_i,\zeta_k)\,\mathrm{sinmui}_{i,m}.

The zeta projection then yields the Fourier coefficients

.. math::

   F^{cc}_{m,n} = \sum_{k=0}^{n_\zeta-1} \tilde F^{(c)}_{m}(\zeta_k)\,\mathrm{cosnv}_{k,n},

.. math::

   F^{ss}_{m,n} = \sum_{k=0}^{n_\zeta-1} \tilde F^{(s)}_{m}(\zeta_k)\,\mathrm{sinnv}_{k,n}.

Derivative terms in VMEC use the scaled tables
:math:`\mathrm{cosnvn}_{k,n}=(n\,\mathrm{NFP})\,\mathrm{cosnv}_{k,n}` and
:math:`\mathrm{sinnvn}_{k,n}=-(n\,\mathrm{NFP})\,\mathrm{sinnv}_{k,n}`. In
``vmex`` we therefore compute the same base transforms and apply the
analytic factor :math:`n\,\mathrm{NFP}` after the zeta contraction for the
derivative blocks. This reduces the number of dot-product contractions while
preserving VMEC2000 parity exactly.

Implementation detail: the theta contractions for multiple force kernels are
**stacked** into a single batched ``dot_general`` call (GEMM), and the zeta
contractions are likewise stacked by basis type (cosine vs sine). This follows
the separable product identities (see Eqs. 5.55–5.56 in the VMEC++ numerics
notes) while keeping the VMEC2000 normalization and parity masks intact.

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
VMEC input profiles. VMEC (and ``vmex``) use pressure in units of
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
(:mod:`vmex.core.step`; discretization in :doc:`algorithms`). Because
:math:`F = -\nabla_{\mathbf{x}} W`, every accepted step decreases :math:`W`
monotonically (up to the momentum transient) and the descent stops only at a
stationary point of the energy.
See References [1-3] for the original VMEC formulation.

Flux coordinates and straight-field-line angle
----------------------------------------------

VMEC introduces a scalar field :math:`\lambda(s,\theta,\zeta)` to define the
straight-field-line poloidal angle:

.. math::

   u = \theta + \lambda(s,\theta,\zeta).

Field lines are straight in :math:`(u,\zeta)`:

.. math::

   \frac{du}{d\zeta} = \iota(s),

where :math:`\iota(s)` is the rotational transform.

Internal scaling and regularity (``scalxc``)
--------------------------------------------

VMEC enforces regularity at the magnetic axis by storing **odd-m** contributions
in an *internal* form that factors out :math:`\sqrt{s}`:

.. math::

   X(s,\theta,\zeta) =
   X_{\mathrm{even}}(s,\theta,\zeta)
   + \sqrt{s}\,X_{\mathrm{odd,int}}(s,\theta,\zeta).

Equivalently,

.. math::

   X_{\mathrm{odd,int}} = \frac{X_{\mathrm{odd,phys}}}{\sqrt{s}}.

VMEC implements this via the ``scalxc`` array, which is 1 for even-m harmonics
and :math:`1/\sqrt{s}` for odd-m harmonics. ``scalxc`` is applied when
interpolating coefficients between radial grids and when assembling
preconditioned residuals (VMEC2000 ``profil3d`` / ``interp`` / ``scalxc``).

On the axis, VMEC applies *odd-m* rules:

- :math:`m=1`: extrapolate the internal odd field to the axis by copying the
  first off-axis value,
- :math:`m\ge 2`: force the internal odd field to zero on-axis.

m=1 internal constraint (``lconm1``)
------------------------------------

When ``LCONM1`` is enabled (VMEC default for 3D runs), VMEC stores the *m=1*
boundary coefficients in a constrained internal basis:

.. math::

   R^{s}_{1n,\mathrm{int}} = \frac{1}{2}\left(R^{s}_{1n,\mathrm{phys}} + Z^{c}_{1n,\mathrm{phys}}\right),
   \qquad
   Z^{c}_{1n,\mathrm{int}} = \frac{1}{2}\left(R^{s}_{1n,\mathrm{phys}} - Z^{c}_{1n,\mathrm{phys}}\right).

This transformation is applied in VMEC2000 ``readin`` and inverted when
converting to physical coefficients for diagnostics. ``vmex`` uses the same
internal basis so that boundary handling and multigrid interpolation match
VMEC2000.

Magnetic field representation
-----------------------------

In VMEC's flux-coordinate representation, the magnetic field has **no radial
contravariant component**:

.. math::

   B^s = 0.

VMEC therefore stores only the contravariant components in the angular
directions:

.. math::

   \mathbf{B} = B^u \nabla u + B^v \nabla v, \qquad v \equiv \zeta.

In terms of VMEC's flux functions
:math:`\Phi(s)` (toroidal flux) and :math:`\chi(s)` (poloidal flux), we define:

.. math::

   \Phi'(s) \equiv \frac{d\Phi}{ds} \quad (\text{``phipf''}), \qquad
   \chi'(s) \equiv \frac{d\chi}{ds} \quad (\text{``chipf''}).

VMEC's **contravariant** components (``bsupu`` and ``bsupv`` in ``wout``)
are computed as:

.. math::

   B^v = \frac{\Phi'(s) + \mathrm{lamscale}\,\partial_{\theta}\lambda}
                {\mathrm{signgs}\,\sqrt{g}\,2\pi},

.. math::

   B^u = \frac{\chi'(s) - \mathrm{lamscale}\,\partial_{\zeta}\lambda}
                {\mathrm{signgs}\,\sqrt{g}\,2\pi}.

Here:

- :math:`\sqrt{g}` is the signed Jacobian,
- ``signgs`` is VMEC's sign convention such that ``signgs*sqrtg`` is positive
  away from the magnetic axis,
- ``lamscale`` is the VMEC scaling applied to :math:`\lambda` derivatives
  (stored in ``wout`` and used by ``vmex`` for parity).

The **covariant** components are defined by:

.. math::

   B_i = \mathbf{B}\cdot \mathbf{e}_i
       = \sum_{j} g_{ij} B^j, \qquad i,j \in \{s,u,v\},

where :math:`g_{ij}` is the covariant metric and
:math:`\mathbf{e}_i = \partial_i \mathbf{r}`.
VMEC stores these as ``bsub*`` in ``wout``.

From metric elements to :math:`|B|`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since :math:`B^s = 0`, only the angular metric block enters. On the half
mesh, with the even/odd-m decomposition
:math:`X = X_{\mathrm{even}} + \sqrt{s}\,X_{\mathrm{odd}}`,

.. math::

   g_{uu} = R_u^2 + Z_u^2, \qquad
   g_{uv} = R_u R_v + Z_u Z_v, \qquad
   g_{vv} = R_v^2 + Z_v^2 + R^2

(:func:`~vmex.core.fields.metric_elements`; the :math:`R^2` term is the
cylindrical toroidal metric at unit
:math:`d\phi_{\mathrm{phys}}/d\zeta`). Lowering the index and contracting,

.. math::

   B_u = g_{uu} B^u + g_{uv} B^v, \qquad
   B_v = g_{uv} B^u + g_{vv} B^v,

.. math::

   |B|^2 = B^u B_u + B^v B_v.

The chain — angular derivatives of :math:`(R, Z, \lambda)` from
:func:`~vmex.core.geometry.real_space_geometry`, half-mesh
:math:`\sqrt{g}` from :func:`~vmex.core.geometry.half_mesh_jacobian`,
metric elements, then :math:`B^u, B^v \to B_u, B_v \to |B|^2` — is assembled
in :func:`~vmex.core.fields.magnetic_fields`, which returns the
contravariant/covariant components together with the total pressure
:math:`\mathrm{bsq} = |B|^2/2 + p` and the differential volume
:math:`vp = \mathrm{signgs}\,\langle\sqrt{g}\rangle`. The ``lamscale``
normalization of the :math:`\lambda` derivatives is
:func:`~vmex.core.fields.lambda_scale`
(``lamscale`` :math:`= \sqrt{h_s \sum_{js} \mathrm{phips}^2}`, ``profil1d.f``),
and the energy scalars ``wb/wp`` with the force normalizations
``fnorm/fnorm1/fnormL`` follow in
:func:`~vmex.core.fields.energies_and_force_norms`.

``bcovar`` + ``add_fluxes`` (poloidal flux correction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VMEC updates the **contravariant** :math:`B^u` using the *full-mesh* poloidal
flux function :math:`\chi'(s)` (``chips``). In VMEC2000 ``add_fluxes``,
``chips`` is computed from force balance on each surface:

.. math::

   \chi'(s) = \frac{I_\varphi(s) - \langle g^{uu} B_u + g^{uv} B_v \rangle}
                   {\langle g^{uu}/\sqrt{g} \rangle},

where the angle brackets denote the VMEC surface quadrature, and
:math:`I_\varphi(s)` is the integrated toroidal current (``icurv``).
VMEC then applies the correction

.. math::

   B^u \leftarrow B^u + \chi'(s)\,\frac{1}{\sqrt{g}}.

VMEC stores the *half-mesh* averaged ``chipf`` in ``wout``; ``vmex``
follows VMEC’s averaging rules to convert between ``chipf`` and ``chips``.

Current density
---------------

The current density follows directly from the curl:

.. math::

   \mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}.

VMEC reports covariant current components in ``wout`` as ``jcuru`` and
``jcurv`` (poloidal and toroidal current densities on the half mesh) and uses
these in the force kernels. The parallel and perpendicular currents satisfy:

.. math::

   \mathbf{J} = \mathbf{J}_{\parallel} + \mathbf{J}_{\perp}, \qquad
   \nabla p = \mathbf{J}_{\perp} \times \mathbf{B}.

For optimization diagnostics, ``vmex`` also exposes the JXBFORCE
real-space current channels as

.. math::

   J^\theta = \frac{\texttt{itheta}}{\sqrt{g}}, \qquad
   J^\zeta = \frac{\texttt{izeta}}{\sqrt{g}},

on the full radial mesh.  The ``vj.JVector`` objective returns these
flux-coordinate components flattened over the selected surfaces and angular
grid.  ``vj.BVector`` returns the corresponding Cartesian magnetic-field vector
``(B_x,B_y,B_z)`` on one selected radial surface.

Redl Bootstrap-Current Mismatch
-------------------------------

For finite-beta stage-one studies, ``vmex`` exposes a differentiable Redl
bootstrap-current residual.  The residual follows the normalized SIMSOPT form

.. math::

   R_j =
   \frac{\langle\mathbf{J}\cdot\mathbf{B}\rangle_{\mathrm{VMEC}}(s_j)
       - \langle\mathbf{J}\cdot\mathbf{B}\rangle_{\mathrm{Redl}}(s_j)}
        {\left[\sum_k
        \left(\langle\mathbf{J}\cdot\mathbf{B}\rangle_{\mathrm{VMEC}}(s_k)
            + \langle\mathbf{J}\cdot\mathbf{B}\rangle_{\mathrm{Redl}}(s_k)
        \right)^2\right]^{1/2}}.

The Redl term uses polynomial density and temperature profiles in the same
ascending-coefficient convention as SIMSOPT ``ProfilePolynomial``.  For the
standard finite-beta stage-one examples, ``vj.standard_finite_beta_profiles``
constructs

.. math::

   n_e(s) = n_{e0}(1 - 0.99 s^5), \qquad
   T_e(s) = T_{e0}(1 - 0.99 s),

with ``ni=ne``, ``Ti=Te``, ``Zeff=1``, and
:math:`p(s)=e(n_eT_e+n_iT_i)` in Pascals.  The amplitudes use the same
scaling as the SIMSOPT finite-beta/bootstrap examples,

.. math::

   n_{e0} = 3\times 10^{20}
      \left(\frac{\beta/100}{0.05}\right)^{1/3}, \qquad
   T_{e0} = 15\,\mathrm{keV}
      \left(\frac{\beta/100}{0.05}\right)^{2/3}.

``vj.with_pressure_profile`` converts this pressure profile to VMEC ``AM`` and
``PRES_SCALE`` input fields while ``vj.RedlBootstrapMismatch`` receives the
same density/temperature coefficients.  The effective trapped-particle
fraction is evaluated with fixed Gauss-Legendre
quadrature using the substitution
:math:`y = \sqrt{1-\lambda B_{\max}}`, which removes the endpoint singularity
in the standard integral

.. math::

   f_t = 1 - \frac{3}{4}\langle B^2\rangle
         \int_0^{1/B_{\max}}
         \frac{\lambda\,d\lambda}
              {\left\langle\sqrt{1-\lambda B}\right\rangle}.

This differs from SIMSOPT's post-processing routine, which refines angular
extrema with splines.  The vmex form is intentionally fixed-shape and
differentiable for use inside exact-Jacobian optimization.

Force balance in VMEC (residual form)
-------------------------------------

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

Spectral condensation (``alias.f``, ``tcon``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The angle parameterization of each flux surface has a tangential null space
(relabeling :math:`\theta` changes no physics). VMEC fixes it by *spectral
condensation*: a constraint force that pushes the poloidal-angle freedom
toward the minimal-spectral-width parameterization. The scalar constraint
kernel is

.. math::

   z_{\mathrm{temp}} = (r_{\mathrm{con}} - r_{\mathrm{con},0})\,r_{\theta,0}
                     + (z_{\mathrm{con}} - z_{\mathrm{con},0})\,z_{\theta,0},

built from the m-profiled geometry channels ``rcon/zcon`` and their frozen
references ``rcon0/zcon0``. It is band-limited to
:math:`m \in [1, \mathrm{mpol}-2]` with the ``faccon(m)`` weights
(:func:`~vmex.core.forces.faccon`,
:func:`~vmex.core.forces.alias_constraint_force`), converted back to a
real-space force contribution
(:func:`~vmex.core.forces.constraint_force`), and scaled per surface by
the strength profile :math:`\mathrm{tcon}(s)` computed from the ratio of the
preconditioner diagonals to the angular force norms
(:func:`~vmex.core.fields.constraint_scaling`; the ``tcon`` formula is
given in :doc:`algorithms`). The constraint vanishes at convergence — it
never shifts the equilibrium, only the angle representation.

These are combined into scalar norms that appear in the VMEC screen output:

.. math::

   \mathrm{FSQR} = r_1\,\mathrm{fnorm}\,\lVert F_R \rVert^2, \qquad
   \mathrm{FSQZ} = r_1\,\mathrm{fnorm}\,\lVert F_Z \rVert^2,

.. math::

   \mathrm{FSQL} = \mathrm{fnormL}\,\lVert F_\lambda \rVert^2,

where ``fnorm``, ``fnormL``, and ``r1`` are the VMEC normalization factors
computed from ``bcovar`` (half-mesh metrics + ``bsup``/``bsub``).
``vmex`` reproduces these scalars from the same internal quantities to
match VMEC2000's per-iteration printout.

Time-step control (Garabedian update)
-------------------------------------

VMEC’s nonlinear fixed-boundary iteration uses a Garabedian-style conjugate
gradient update with a time-step control mechanism. The update computes

.. math::

   \tau_n = \min\left(\left|\ln\frac{\mathrm{fsq}_n}{\mathrm{fsq}_{n-1}}\right|,\; 0.15\right),

and maintains a moving average :math:`\overline{\tau}` over ``ndamp`` steps.
The damping factor is

.. math::

   \Delta\tau = \frac{\Delta t\,\overline{\tau}}{2}, \qquad
   b_1 = 1-\Delta\tau, \qquad
   \mathrm{fac} = \frac{1}{1+\Delta\tau}.

The update in VMEC2000 is:

.. math::

   \dot{x} \leftarrow \mathrm{fac}\,\bigl(b_1\dot{x} + \Delta t\,F\bigr), \qquad
   x \leftarrow x + \Delta t\,\dot{x},

where :math:`F` is the preconditioned residual vector (``gc``).

VMEC’s ``TimeStepControl`` tracks the minimum of the preconditioned residual
(``res0``) and the physical residual (``res1``). If either grows by a factor of
``1e4`` after 10 steps, VMEC restores the last good state and reduces
``DELT`` by a factor of 1.03. ``vmex`` mirrors this logic to reproduce the
VMEC2000 iteration trace.  If that trace reaches VMEC2000's fatal limit of 75
Jacobian resets, VMEX may perform the documented driver-level best-checkpoint
retry with a reduced initial ``DELT``.  This is a solver globalization
extension, not a change to the equilibrium residual; it is disabled with
``jacobian_retries=0``.

Multigrid interpolation (``interp.f``)
--------------------------------------

VMEC’s multigrid staging interpolates *scaled* coefficients between grids:

.. math::

   x_{\mathrm{scaled}} = x \cdot \mathrm{scalxc},

with odd-m extrapolation to the axis performed **before** interpolation.
After linear interpolation on a uniform radial grid, coefficients are unscaled:

.. math::

   x_{\mathrm{new}} = \frac{x_{\mathrm{scaled,new}}}{\mathrm{scalxc}_{\mathrm{new}}}.

``vmex`` implements this exact pipeline so that stage-to-stage coefficient
transfer matches VMEC2000.

Pressure and beta
-----------------

VMEC reports thermal and magnetic energy scalars in ``wout``:

.. math::

   W_B = \frac{1}{(2\pi)^2}\int \frac{B^2}{2}\,dV, \qquad
   W_P = \frac{1}{(2\pi)^2}\int p\,dV.

The total volume-averaged beta is computed by VMEC as:

.. math::

   \beta_{\mathrm{total}} = \frac{W_P}{W_B}.

``vmex`` follows the same normalization when emitting ``wout`` files.

Geometric embedding and Jacobian
--------------------------------

Surfaces are represented in cylindrical coordinates by Fourier series:

.. math::

   R(s,\theta,\zeta) = \sum_{m,n}\left[
     R_{mn}^c(s)\cos(m\theta-n\zeta) + R_{mn}^s(s)\sin(m\theta-n\zeta)
   \right],

.. math::

   Z(s,\theta,\zeta) = \sum_{m,n}\left[
     Z_{mn}^c(s)\cos(m\theta-n\zeta) + Z_{mn}^s(s)\sin(m\theta-n\zeta)
   \right].

We embed into Cartesian coordinates using the physical toroidal angle:

.. math::

   x = R\cos\phi_{\mathrm{phys}}, \qquad
   y = R\sin\phi_{\mathrm{phys}}, \qquad
   z = Z.

The Jacobian is:

.. math::

   \sqrt{g} = \mathbf{e}_s \cdot (\mathbf{e}_\theta \times \mathbf{e}_{\phi}).

VMEC's sign convention enforces ``signgs*sqrtg > 0`` away from the axis.

Jacobian sign check (``tau``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VMEC evaluates an auxiliary Jacobian-like scalar :math:`\tau` built from
even/odd-m real-space derivatives (see VMEC2000 ``jacobian.f``). In compact
form,

.. math::

   \tau \equiv (R_u Z_s - R_s Z_u) + \text{(odd-m corrections in } \sqrt{s}\text{)}.

If :math:`\tau` changes sign away from the axis, VMEC flags a bad Jacobian and
restarts the iteration with a refined axis guess. ``vmex`` reproduces the
same parity split, half-mesh averaging, and sign check so that Jacobian-reset
behavior matches VMEC2000.

Implementation mapping (``vmex``)
-------------------------------------

Key :mod:`vmex.core` modules that directly implement the equations above
(the full map is in :doc:`architecture`):

- :mod:`vmex.core.fourier` — mode tables, ``mscale/nscale``, trig/weight
  tables (``fixaray.f``).
- :mod:`vmex.core.transforms` — ``totzsps/tomnsps``-family transforms as
  batched matmuls.
- :mod:`vmex.core.geometry` — geometry, metric, Jacobian and the
  :math:`\tau` sign check (``jacobian.f``).
- :mod:`vmex.core.fields` — contravariant/covariant field components,
  energies, ``tcon`` (``bcovar.f``).
- :mod:`vmex.core.forces` — real-space force kernels (``A,B,C`` blocks)
  and the spectral-condensation constraint force (``forces.f``, ``alias.f``).
- :mod:`vmex.core.residuals` — scalar residuals ``fsqr/fsqz/fsql`` and the
  m=1 constraint (``residue.f90``).
- :mod:`vmex.core.preconditioner` — 1D radial preconditioner
  (``precondn.f``, ``scalfor.f``, ``lamcal.f90``).
- :mod:`vmex.core.preconditioner_2d` — 2D block preconditioner / Newton
  step (``Hessian/precon2d.f``).
- :mod:`vmex.core.step` — Richardson update and restart control
  (``evolve.f``, ``restart.f``).
- :mod:`vmex.core.solver` — the fixed-boundary iteration loop
  (``funct3d.f``, ``eqsolve.f``).
- :mod:`vmex.core.implicit` — implicit differentiation of the converged
  equilibrium (no VMEC2000 counterpart).

References (local)
------------------

- ``vmecpp/docs/the_numerics_of_vmecpp.pdf`` (VMEC++ numerics notes; VMEC2000
  conventions and formulas).
- VMEC2000 source in ``STELLOPT/VMEC2000/Sources``.
