Theory and conventions
======================

This page summarizes conventions used in ``vmec-jax``. The goal is
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

``vmec-jax`` stores these coefficients in a ``VMECState`` as arrays shaped
``(ns, K)`` where ``K`` is the number of ``(m,n)`` modes in the main VMEC
ordering.

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

``vmec-jax`` follows this convention so we can validate against ``wout`` values.

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

Energy scalars (``wb`` and ``wp``)
----------------------------------

VMEC reports the magnetic energy scalar ``wb`` and thermal energy scalar ``wp``
in ``wout`` files. In VMEC normalization:

.. math::

   \mathrm{wb} = \frac{1}{(2\pi)^2}\int \frac{B\cdot B}{2}\,dV, \qquad
   \mathrm{wp} = \frac{1}{(2\pi)^2}\int p\,dV.

Important: VMEC treats internal pressure in units of :math:`\mu_0\,\mathrm{Pa}`
(i.e. :math:`B^2` units). ``vmec-jax`` follows this convention for parity.

