The NESTOR vacuum solve
=======================

Free-boundary VMEX couples the plasma iteration to Merkel's Green's-function
vacuum solve (NESTOR, J. Comp. Phys. 66, 83 (1986)), ported from VMEC2000's
``vacuum.f`` pipeline with the same activation cadence. This page explains
the exterior Neumann problem, the full-vs-incremental update split, and
which parts of the free-boundary problem are differentiated; the run recipe
is :doc:`/howto/free-boundary`.

The exterior Neumann problem
----------------------------

For ``LFREEB = T`` decks, :mod:`vmex.core.vacuum` implements Merkel's
Green's-function method. In the vacuum region the field is curl-free, so it
is written as

.. math::

   \mathbf{B}_{\mathrm{vac}} = \mathbf{B}_{\mathrm{ext}} + \nabla\Phi,
   \qquad \nabla^2 \Phi = 0,

with :math:`\mathbf{B}_{\mathrm{ext}}` the field of the external coils
(mgrid or Biot–Savart) plus the net-toroidal-current filament, and the
plasma boundary acting as a flux surface:

.. math::

   \mathbf{n}\cdot(\mathbf{B}_{\mathrm{ext}} + \nabla\Phi) = 0
   \quad \text{on } \partial\Omega.

Green's second identity turns this exterior Neumann problem into a boundary
integral equation for the surface potential,

.. math::

   \frac{\Phi(\mathbf{x}')}{2}
   = \oint_{\partial\Omega} \Bigl[
     \Phi(\mathbf{x})\,\mathbf{n}\cdot\nabla G(\mathbf{x},\mathbf{x}')
     + G(\mathbf{x},\mathbf{x}')\,
       \mathbf{n}\cdot\mathbf{B}_{\mathrm{ext}}(\mathbf{x})
     \Bigr]\, dS, \qquad
   G = \frac{1}{4\pi\,|\mathbf{x}-\mathbf{x}'|},

which, after expanding :math:`\Phi` in Fourier harmonics
:math:`\sin(mu - nv)/\cos(mu - nv)` on the boundary, becomes a dense
``mnpd2 x mnpd2`` linear system for the potential coefficients ``potvac``.
The :math:`|\mathbf{x}-\mathbf{x}'| \to 0` singularity of :math:`G` is split
off and integrated analytically (``analyt.f``, the ``cmns`` coefficient
tables); the regular remainder is tabulated on the angular grid (``greenf`` /
``fourp``). Implementation: geometry-independent tables in
:func:`~vmex.core.vacuum.vacuum_basis`, the jitted full/incremental
solves in :func:`~vmex.core.vacuum.make_vacuum_solver`, and the surface
field :math:`B_u = \mathrm{bexu} + \partial_u\Phi` (etc.) with
:math:`\mathrm{bsqvac} = |B_{\mathrm{vac}}|^2/2` in
:func:`~vmex.core.vacuum.vacuum_channels`.

Coupling cadence (``funct3d.f``)
--------------------------------

:func:`vmex.core.freeboundary.solve_free_boundary` drives the coupling
with the VMEC2000 cadence:

- the vacuum solve activates once :math:`\mathrm{fsqr}+\mathrm{fsqz} \le 10^{-3}`;
- a **full** NESTOR solve runs when ``mod(iter2 - iter1, nvacskip) == 0``,
  factoring the dense potential matrix once; cheaper incremental updates
  reuse that LU factor (VMEC2000's ``DGETRF``/``DGETRS`` split) while only
  rebuilding the analytic right-hand side, and the cadence adapts as

  .. math::

     \mathrm{nvacskip} \leftarrow \max\!\left(\mathrm{nvskip}_0,\;
     \frac{1}{\max(0.1,\; 10^{11}\,(\mathrm{fsqr}+\mathrm{fsqz}))}\right);

- the vacuum pressure enters the edge force through
  ``rbsq = (bsqvac + presf_ns) * R(edge) / hs`` at ``js = ns``, and the
  constraint reference surfaces ``rcon0, zcon0`` ramp by 0.9 per iteration.

The multigrid form of this coupling — carried vacuum state, per-stage NESTOR
rebuilds, one activation across the ladder — is described in
:doc:`multigrid`.

External fields
---------------

The forward NESTOR solver consumes a :class:`~vmex.core.mgrid.MgridField`.
It may be loaded from an ``mgrid`` file (trilinear interpolation weighted by
``EXTCUR``) or built once with
:meth:`~vmex.core.mgrid.MgridField.from_cartesian_field`, which tabulates an
ESSOS/SIMSOPT Biot--Savart object or any ``xyz -> B`` callable.  The resulting
table and its current scale remain JAX-differentiable; tabulation itself does
not retain coil-geometry derivatives. Direct, interpolation-free ESSOS coil
derivatives use the virtual-casing residual below. VMEX carries no coil code.

On a GPU free-boundary run, the plasma iteration, mgrid interpolation, cached
vacuum arrays, and final state remain on the accelerator. The dense NESTOR
assembly/factor/solve is explicitly placed on CPU and its small boundary
inputs/outputs are bridged inside the jitted cadence loop. This follows the
VMEC++ accelerator decomposition and avoids the alternate LASYM branch seen
with accelerator dense linear algebra. An explicitly requested GPU LASYM
multigrid ladder therefore seeds only its coarsest rung on CPU, then transfers
the converged branch to all finer GPU rungs.

What is (and is not) differentiated
-----------------------------------

The NESTOR iteration above is a host-driven fixed point and is not
differentiated. For coil/current optimization,
:mod:`vmex.core.freeboundary_diff` instead expresses the free-boundary
condition as a smooth objective on a given boundary. At the plasma-vacuum
interface the total exterior field
:math:`\mathbf{B}_{\mathrm{out}} = \mathbf{B}_{\mathrm{coil}} +
\mathbf{B}_{\mathrm{plasma}}` must be tangent, and pressure balance holds:

.. math::

   \mathbf{B}_{\mathrm{out}}\cdot\mathbf{n} = 0, \qquad
   |\mathbf{B}_{\mathrm{in}}|^2 + 2\mu_0 p = |\mathbf{B}_{\mathrm{out}}|^2.

The plasma's own exterior field comes from the **virtual-casing principle**:
the field produced outside :math:`\partial\Omega` by the plasma currents
equals that of the surface current
:math:`\mathbf{K} = \mathbf{n}\times\mathbf{B}/\mu_0` on
:math:`\partial\Omega`, evaluated with an accurate on-surface singular
quadrature (reused from the optional ``virtual_casing_jax`` package,
required as ``virtual-casing-jax >= 0.0.4`` from the canonical
``uwplasma/virtual_casing_jax`` repository;
:func:`~vmex.core.freeboundary_diff.surface_field_data_from_wout`
adapts a converged boundary + field, and
:func:`~vmex.core.freeboundary_diff.plasma_field_on_boundary` evaluates
the integral). The key structural fact: for a *fixed* trial boundary,
:math:`\mathbf{B}_{\mathrm{plasma}}` on that boundary does not depend on the
coil degrees of freedom, so it is precomputed once and frozen. The residual
assembled by
:class:`~vmex.core.freeboundary_diff.FreeBoundaryDiffProblem` is then a
smooth JAX function of the external-field dofs alone (coil Fourier
coefficients/currents of a callable ESSOS coil field via
:func:`~vmex.core.freeboundary_diff.external_B_cartesian`, or
``extcur``), and its ``value_and_grad_bnormal`` helper returns gradients
validated against finite differences — no NESTOR adjoint is required.

Field-query API
---------------

:class:`~vmex.core.extender.MagneticField` provides stored Cartesian points,
``B``, ``absB``, and spatial derivatives through ``gradgradgradB``. A field
constructed from :meth:`~vmex.core.problem.VmecProblem.exterior_field` also
provides ``B_vjp`` and the three spatial-derivative VJPs in the problem's
boundary/current DOFs. The virtual-casing path applies outside the LCFS;
:class:`~vmex.core.extender.VmecInteriorField` evaluates the live VMEC
spectral field inside. Query points must stay away from the source surface and
external coil filaments.

Toward a coupled adjoint
------------------------

:class:`vmex.core.freeboundary_linear.NestorBorderedOperator` represents its
linearization as ``[[A, B], [C, D]]`` with matrix-free plasma, vacuum, and
edge-coupling actions. :func:`~vmex.core.freeboundary_linear.linearize_nestor_coupling`
builds those four actions directly from a live plasma residual and NESTOR's
unsolved ``A(x) q - b(x)`` equation; :class:`~vmex.core.vacuum.VacuumSolver`
exposes that equation through ``assemble`` without nesting a potential solve.
The operator supplies the exact generated transpose, the Schur action
:math:`D-C A^{-1}B`, and a block inverse. The live LASYM NESTOR blocks are
tested against the complete coupled JVP/VJP. The host-driven cadence above is
not yet replaced by a coupled Newton solve, so this foundation is not yet a
public implicit free-boundary adjoint (see
:doc:`/reference/capabilities`).
