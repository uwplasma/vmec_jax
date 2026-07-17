Theory and conventions
======================

This page summarizes conventions used in ``vmex``. The goal is
*compatibility* with VMEC2000 (``wout_*.nc``) and with standard VMEC literature.

Coordinates and angles
----------------------

VMEC uses curvilinear coordinates on nested flux surfaces. In this repo we use:

- :math:`s \in [0,1]`: normalized toroidal flux label (VMEC’s “radial” coordinate).
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

``vmex`` stores these coefficients in the
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

Regularity and internal storage
-------------------------------

VMEC stores odd-m harmonics in an **internal** form that factors out the
radial :math:`\sqrt{s}` behavior so that coefficients remain finite on-axis.
This is implemented through the ``scalxc`` array (1 for even-m, :math:`1/\sqrt{s}`
for odd-m) and is applied whenever VMEC interpolates between radial grids or
builds preconditioned residuals.

When ``LCONM1`` is enabled, VMEC also applies a constrained internal basis for
the :math:`m=1` boundary modes:

.. math::

   R^{s}_{1n,\mathrm{int}} = \frac{1}{2}\left(R^{s}_{1n,\mathrm{phys}} + Z^{c}_{1n,\mathrm{phys}}\right),
   \qquad
   Z^{c}_{1n,\mathrm{int}} = \frac{1}{2}\left(R^{s}_{1n,\mathrm{phys}} - Z^{c}_{1n,\mathrm{phys}}\right).

``vmex`` uses the same internal storage so that its boundary mapping,
multigrid interpolation, and parity diagnostics match VMEC2000.

The lambda field
----------------

VMEC introduces a scalar field :math:`\lambda(s,\theta,\zeta)` to define the
“straight-field-line” poloidal angle:

.. math::

   u = \theta + \lambda(s,\theta,\zeta),

so that magnetic field lines are straight in the ``(u,\zeta)`` angle pair.

Important scaling convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In VMEC2000 ``wout`` files, the stored Fourier coefficients of :math:`\lambda`
are **scaled** by a run-dependent scalar ``lamscale``. VMEC multiplies
:math:`\partial\lambda/\partial\theta` and :math:`\partial\lambda/\partial\zeta`
by ``lamscale`` before using them in the contravariant field formulas.

``vmex`` follows this convention so we can validate against ``wout`` values.

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

   g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j, \qquad i,j \in \{s,\theta,\phi\}.

The signed Jacobian is:

.. math::

   \sqrt{g} = \mathbf{e}_s \cdot (\mathbf{e}_\theta \times \mathbf{e}_\phi).

VMEC stores a sign convention ``signgs = ±1`` such that ``signgs*sqrtg`` is
positive away from the axis.

Contravariant magnetic field (VMEC form)
----------------------------------------

For fixed-boundary VMEC, the magnetic field is represented in terms of
contravariant components :math:`B^u` and :math:`B^v` (VMEC’s ``bsupu`` and
``bsupv``) and 1D flux functions:

- ``phipf(s) = dΦ/ds`` (toroidal flux derivative),
- ``chipf(s) = dΧ/ds`` (poloidal flux derivative),

plus lambda derivatives.

In this repo (matching VMEC’s ``bcovar`` + ``add_fluxes`` logic), the formulas
are:

.. math::

   \mathrm{bsupv} = \frac{\mathrm{phipf} + \mathrm{lamscale}\,\partial_\theta\lambda}
                        {\mathrm{signgs}\,\sqrt{g}\,2\pi},

.. math::

   \mathrm{bsupu} = \frac{\mathrm{chipf} - \mathrm{lamscale}\,\partial_\zeta\lambda}
                        {\mathrm{signgs}\,\sqrt{g}\,2\pi}.

Note that :math:`\partial_\zeta \lambda` is w.r.t. the field-period coordinate
:math:`\zeta`, while the geometry kernel returns
:math:`\partial_{\phi_{\mathrm{phys}}}\lambda`, so we convert using
:math:`\partial_\zeta = (1/\mathrm{NFP})\,\partial_{\phi_{\mathrm{phys}}}`.

Covariant components and :math:`|B|`
------------------------------------

Because :math:`B^s = 0`, lowering the index needs only the angular metric
block:

.. math::

   B_u = g_{uu} B^u + g_{uv} B^v, \qquad
   B_v = g_{uv} B^u + g_{vv} B^v,

and the field magnitude follows without ever forming Cartesian components:

.. math::

   |B|^2 = B^u B_u + B^v B_v.

The computation chain per iteration is:

1. :func:`~vmex.core.geometry.real_space_geometry` — synthesize
   :math:`R, Z, \lambda` and their angular derivatives (even/odd-m planes);
2. :func:`~vmex.core.geometry.half_mesh_jacobian` — half-mesh
   :math:`\sqrt{g}` and the :math:`\tau` sign proxy;
3. :func:`~vmex.core.fields.metric_elements` — half-mesh
   ``guu, guv, gvv``;
4. :func:`~vmex.core.fields.magnetic_fields` — ``bsupu/bsupv`` from the
   flux profiles and :math:`\lambda` derivatives, then ``bsubu/bsubv`` and
   the total pressure :math:`|B|^2/2 + p` (VMEC ``bsq``).

Energy scalars (``wb`` and ``wp``)
----------------------------------

VMEC reports the magnetic energy scalar ``wb`` and thermal energy scalar ``wp``
in ``wout`` files. In VMEC normalization:

.. math::

   \mathrm{wb} = \frac{1}{(2\pi)^2}\int \frac{B\cdot B}{2}\,dV, \qquad
   \mathrm{wp} = \frac{1}{(2\pi)^2}\int p\,dV.

Important: VMEC treats internal pressure in units of :math:`\mu_0\,\mathrm{Pa}`
(i.e. :math:`B^2` units). ``vmex`` follows this convention for parity.
