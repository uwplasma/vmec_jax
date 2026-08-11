Spectral representation
=======================

A VMEX equilibrium is a set of nested flux surfaces stored as Fourier
moments :math:`R_{mn}(s), Z_{mn}(s), \lambda_{mn}(s)` in VMEC's exact
conventions — phase :math:`m\theta - n\zeta`, ``mscale/nscale``
normalization, internal :math:`\sqrt{s}` scaling for odd-m modes. This page
defines those conventions, the weighted DFT that replaces a plain FFT, and
the spectral-condensation constraint that fixes the poloidal-angle freedom;
the problem being solved is in :doc:`variational-problem`.

Coordinates and angles
----------------------

VMEC uses curvilinear coordinates on nested flux surfaces:

- :math:`s \in [0,1]`: normalized toroidal flux label (VMEC's "radial" coordinate).
- :math:`\theta \in [0, 2\pi)`: poloidal angle.
- :math:`\zeta \in [0, 2\pi)`: **field-period toroidal angle** (VMEC internal coordinate).
- physical toroidal angle :math:`\phi_{\mathrm{phys}} = \zeta / \mathrm{NFP}`, where
  :math:`\mathrm{NFP}` is the number of field periods.

Fourier phases are written as:

.. math::

   \mathrm{phase}(m,n;\theta,\zeta) = m\theta - n\zeta.

Here :math:`n` is the *field-period* toroidal mode number (VMEC stores
``xn = n*NFP`` in ``wout``).

Derivatives w.r.t. the physical toroidal angle satisfy:

.. math::

   \frac{\partial}{\partial \phi_{\mathrm{phys}}} =
   \mathrm{NFP}\,\frac{\partial}{\partial \zeta}.

Surface representation
----------------------

VMEC represents a surface in cylindrical coordinates using Fourier series:

.. math::

   R(s,\theta,\zeta) = \sum_{m,n} \Bigl(
      R_{mn}^c(s)\cos(m\theta-n\zeta) + R_{mn}^s(s)\sin(m\theta-n\zeta)
   \Bigr),

.. math::

   Z(s,\theta,\zeta) = \sum_{m,n} \Bigl(
      Z_{mn}^c(s)\cos(m\theta-n\zeta) + Z_{mn}^s(s)\sin(m\theta-n\zeta)
   \Bigr).

VMEX stores these coefficients in the
:class:`~vmex.core.solver.SpectralState` pytree as arrays shaped
``(ns, K)`` where ``K`` is the number of ``(m,n)`` modes in the main VMEC
ordering (see :mod:`vmex.core.fourier` for the mode bookkeeping).

Parities and stellarator symmetry (``lasym``)
---------------------------------------------

A *stellarator-symmetric* equilibrium is invariant under
:math:`(\theta,\zeta) \to (-\theta,-\zeta)` with :math:`R \to R`,
:math:`Z \to -Z`. Each field therefore keeps only one Fourier parity:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - Field
     - symmetric (``lasym = F``)
     - antisymmetric partner (``lasym = T`` only)
   * - :math:`R`
     - :math:`\cos(m\theta-n\zeta)` (``rmnc``)
     - :math:`\sin` (``rmns``)
   * - :math:`Z`
     - :math:`\sin(m\theta-n\zeta)` (``zmns``)
     - :math:`\cos` (``zmnc``)
   * - :math:`\lambda`
     - :math:`\sin(m\theta-n\zeta)` (``lmns``)
     - :math:`\cos` (``lmnc``)

:class:`~vmex.core.solver.SpectralState` always carries all six blocks
(``R_cos, R_sin, Z_cos, Z_sin, L_cos, L_sin``); in symmetric runs the
antisymmetric partners are structurally zero and do not evolve. The synthesis
:func:`~vmex.core.transforms.fourier_to_real` (VMEC ``totzsps`` +
``totzspa``) handles both parities in one signed-:math:`(m,n)` cos/sin
packing.

Stellarator symmetry also halves the angular grid: the stored poloidal extent
is :math:`\theta \in [0,\pi]` (``ntheta2``) for symmetric runs and the full
:math:`[0, 2\pi)` (``ntheta1``) when ``lasym`` — the ``ntheta3`` property of
:class:`~vmex.core.fourier.Resolution`. For ``lasym`` runs, the force
kernels are first split into symmetric/antisymmetric parts on the reduced
interval (VMEC ``symforce.f``, :func:`~vmex.core.transforms.symforce_split`,
applied by :func:`~vmex.core.forces.symmetrize_forces`) and each part is
projected with the matching analysis transform
(:func:`~vmex.core.transforms.tomnsps` /
:func:`~vmex.core.transforms.tomnspa`).

The lambda field and the straight-field-line angle
--------------------------------------------------

VMEC introduces a scalar field :math:`\lambda(s,\theta,\zeta)` to define the
straight-field-line poloidal angle:

.. math::

   u = \theta + \lambda(s,\theta,\zeta).

Field lines are straight in :math:`(u,\zeta)`:

.. math::

   \frac{du}{d\zeta} = \iota(s),

where :math:`\iota(s)` is the rotational transform.

In VMEC2000 ``wout`` files, the stored Fourier coefficients of :math:`\lambda`
are **scaled** by a run-dependent scalar ``lamscale``
(:func:`~vmex.core.fields.lambda_scale`,
``lamscale`` :math:`= \sqrt{h_s \sum_{js} \mathrm{phips}^2}`, ``profil1d.f``).
VMEC multiplies :math:`\partial\lambda/\partial\theta` and
:math:`\partial\lambda/\partial\zeta` by ``lamscale`` before using them in
the contravariant field formulas; VMEX follows the same convention so its
output validates against ``wout`` values.

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
converting to physical coefficients for diagnostics. VMEX uses the same
internal basis so that boundary handling and multigrid interpolation match
VMEC2000.

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

VMEX uses these tables in ``tomnsps`` so that the Fourier-space force
arrays exactly match VMEC2000. See References [4-6] in
:doc:`/project/references` for the original VMEC2000 tables and the VMEC++
DFT/basis discussion.

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
VMEX we therefore compute the same base transforms and apply the
analytic factor :math:`n\,\mathrm{NFP}` after the zeta contraction for the
derivative blocks. This reduces the number of dot-product contractions while
preserving VMEC2000 parity exactly.

Implementation detail: the theta contractions for multiple force kernels are
**stacked** into a single batched ``dot_general`` call (GEMM), and the zeta
contractions are likewise stacked by basis type (cosine vs sine). This follows
the separable product identities (see Eqs. 5.55–5.56 in the VMEC++ numerics
notes) while keeping the VMEC2000 normalization and parity masks intact.

Geometry, metric, and Jacobian
------------------------------

We form covariant basis vectors by embedding the surface into 3D Cartesian
coordinates using the physical toroidal angle :math:`\phi_{\mathrm{phys}}`:

.. math::

   x = R\cos\phi_{\mathrm{phys}}, \qquad
   y = R\sin\phi_{\mathrm{phys}}, \qquad
   z = Z.

The covariant basis vectors are:

.. math::

   \mathbf{e}_s = \partial_s \mathbf{r}, \qquad
   \mathbf{e}_\theta = \partial_\theta \mathbf{r}, \qquad
   \mathbf{e}_\phi = \partial_{\phi_{\mathrm{phys}}}\mathbf{r}.

The covariant metric is:

.. math::

   g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j, \qquad i,j \in \{s,\theta,\phi\},

and the signed Jacobian is:

.. math::

   \sqrt{g} = \mathbf{e}_s \cdot (\mathbf{e}_\theta \times \mathbf{e}_\phi).

VMEC stores a sign convention ``signgs = ±1`` such that ``signgs*sqrtg`` is
positive away from the axis.

Jacobian sign check (``tau``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VMEC evaluates an auxiliary Jacobian-like scalar :math:`\tau` built from
even/odd-m real-space derivatives (see VMEC2000 ``jacobian.f``). In compact
form,

.. math::

   \tau \equiv (R_u Z_s - R_s Z_u) + \text{(odd-m corrections in } \sqrt{s}\text{)}.

If :math:`\tau` changes sign away from the axis, VMEC flags a bad Jacobian and
restarts the iteration with a refined axis guess. VMEX reproduces the
same parity split, half-mesh averaging, and sign check so that Jacobian-reset
behavior matches VMEC2000 (see :doc:`iteration` for the restart rules).

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

Note that :math:`\partial_\zeta \lambda` is w.r.t. the field-period coordinate
:math:`\zeta`, while the geometry kernel returns
:math:`\partial_{\phi_{\mathrm{phys}}}\lambda`, so VMEX converts using
:math:`\partial_\zeta = (1/\mathrm{NFP})\,\partial_{\phi_{\mathrm{phys}}}`.

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
:math:`vp = \mathrm{signgs}\,\langle\sqrt{g}\rangle`. The energy scalars
``wb/wp`` with the force normalizations ``fnorm/fnorm1/fnormL`` follow in
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

VMEC stores the *half-mesh* averaged ``chipf`` in ``wout``; VMEX
follows VMEC's averaging rules to convert between ``chipf`` and ``chips``.

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

For optimization diagnostics, VMEX also exposes the JXBFORCE
real-space current channels as

.. math::

   J^\theta = \frac{\texttt{itheta}}{\sqrt{g}}, \qquad
   J^\zeta = \frac{\texttt{izeta}}{\sqrt{g}},

on the full radial mesh.  The ``vj.JVector`` objective returns these
flux-coordinate components flattened over the selected surfaces and angular
grid.  ``vj.BVector`` returns the corresponding Cartesian magnetic-field vector
``(B_x,B_y,B_z)`` on one selected radial surface.

Spectral condensation (``alias.f``, ``tcon``)
---------------------------------------------

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
(:func:`~vmex.core.fields.constraint_scaling`; the ``tcon`` formula is in
:doc:`iteration`). The constraint vanishes at convergence — it never shifts
the equilibrium, only the angle representation.

.. figure:: /_static/constraint_pipeline.svg
   :alt: Spectral-condensation constraint pipeline from rcon/zcon channels to the band-limited constraint force
   :align: center
   :width: 95%

   The constraint pipeline: m-profiled geometry channels, frozen references,
   band-limited ``faccon`` filter, and the per-surface ``tcon`` scaling.
