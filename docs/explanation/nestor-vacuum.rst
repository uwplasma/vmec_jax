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
not retain coil-geometry derivatives. For a coupled solve,
:meth:`~vmex.core.mgrid.MgridField.from_parameterized_cartesian_field` instead
tabulates ``field(coil_parameters, xyz)`` entirely in JAX, retaining exact
shape/current derivatives through interpolation and NESTOR. Direct,
interpolation-free ESSOS derivatives use the virtual-casing residual below.
VMEX carries no coil code.

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
:mod:`vmex.core.virtual_casing` instead expresses the interface conditions as
smooth objectives on a prescribed boundary. At the plasma-vacuum
interface the total exterior field
:math:`\mathbf{B}_{\mathrm{out}} = \mathbf{B}_{\mathrm{coil}} +
\mathbf{B}_{\mathrm{plasma}}` must be tangent, and pressure balance holds:

.. math::

   \mathbf{B}_{\mathrm{out}}\cdot\mathbf{n} = 0, \qquad
   |\mathbf{B}_{\mathrm{in}}|^2 + 2\mu_0 p = |\mathbf{B}_{\mathrm{out}}|^2.

The plasma's own exterior field comes from the **virtual-casing principle**.
In the BIEST convention, its layer densities on :math:`\partial\Omega` are
:math:`\sigma=\mathbf{B}\cdot\mathbf{n}` and
:math:`\mathbf{J}=\mathbf{B}\times\mathbf{n}`. The exterior field of the
enclosed plasma currents is the internal branch
:math:`-\nabla G[\sigma]-\mathrm{BiotSavart}[\mathbf{J}]`, evaluated with an
accurate singular quadrature (reused from the optional
``virtual_casing_jax`` package,
required as ``virtual-casing-jax >= 0.0.4`` from the canonical
``uwplasma/virtual_casing_jax`` repository;
:func:`~vmex.core.virtual_casing.surface_field_data_from_wout`
adapts a converged boundary + field, and
:func:`~vmex.core.virtual_casing.plasma_field_on_boundary` evaluates
the integral). The key structural fact: for a *fixed* trial boundary,
:math:`\mathbf{B}_{\mathrm{plasma}}` on that boundary does not depend on the
coil degrees of freedom, so it is precomputed once and frozen. The residual
assembled by
:class:`~vmex.core.virtual_casing.PlasmaVacuumInterface` is then a
smooth JAX function of the external-field dofs alone (coil Fourier
coefficients/currents of a callable ESSOS coil field via
:func:`~vmex.core.virtual_casing.external_B_cartesian`, or
``extcur``), and its ``value_and_grad_bnormal`` helper returns gradients
validated against finite differences — no NESTOR adjoint is required.

The finite-beta single-stage example uses a pressure profile that vanishes at
the LCFS. It therefore needs no prescribed physical sheet current in the jump
condition; nonzero edge pressure or an imposed sheet current requires an
additional interface model.

Despite using the interface equations, that example is a **fixed-boundary**
optimization: every trial boundary is prescribed to VMEX and reconverged, and
both boundary and coil coefficients are decision variables. Virtual casing
separates the converged total VMEX field into plasma-current and external-coil
parts; it does not run NESTOR or a free-boundary equilibrium. The experimental
``single_stage_free_boundary_optimization*.py`` examples instead hold the
plasma boundary implicit and vary only coil parameters through the coupled
NESTOR derivative described below.

The reported normalized total-pressure jump is

.. math::

   \left\langle\left[
   (|\mathbf B_{\rm out}|^2-|\mathbf B_{\rm in}|^2-2\mu_0p_{\rm edge})
   / B_{\rm ref}^2\right]^2\right\rangle_A^{1/2}.

It is dimensionless and vanishes when the ideal-MHD pressure-balance
condition holds. It is not an error in the prescribed volume pressure
profile. Even when :math:`p_{\rm edge}=0`, it supplies the tangential-field
magnitude condition that ``B.n/B`` alone does not constrain.

Field-query API
---------------

:class:`~vmex.core.extender.MagneticField` provides stored Cartesian points,
``B``, ``absB``, and spatial derivatives through ``gradgradgradB``. A field
constructed from :meth:`~vmex.core.problem.VmecProblem.exterior_field` also
provides ``B_vjp`` and the three spatial-derivative VJPs in the problem's
boundary/current DOFs. The virtual-casing path applies outside the LCFS;
:class:`~vmex.core.extender.VmecInteriorField` evaluates the live VMEC
spectral field inside. Direct off-surface quadrature must stay away from the
source surface and all targets must stay away from external coil filaments.
For near-LCFS field-line tracing,
:meth:`~vmex.core.extender.VmecExtender.with_near_surface_continuation`
prepares the singular on-surface plasma field and gradient once, then uses the
first-order continuation
:math:`\mathbf B(\mathbf x_\Gamma+\delta\mathbf x)=\mathbf B_\Gamma+
\nabla\mathbf B_\Gamma\delta\mathbf x+O(|\delta\mathbf x|^2)`. This removes
the otherwise prohibitive source-grid refinement from every ODE step; direct
quadrature remains the validation path farther from the LCFS.

Virtual casing reconstructs the field produced by currents inside the plasma
surface. It does not determine the external coil field: supply an ESSOS coil
field or MGRID field and :class:`~vmex.core.extender.VmecExtender` adds the two.
This distinction matters for finite-beta exterior tracing and coil design.

See ``examples/vmex_get_B_gradB.py`` for the finite-beta interior API and
``examples/vmex_get_B_outside_plasma.py`` for the live ESSOS-coil plus
virtual-casing path, including exact equilibrium and coil VJPs. The
``vmex_fieldline_tracing_vacuum.py`` and
``vmex_fieldline_tracing_finite_beta.py`` examples use those same fields for
inside/outside tracing. The single-stage optimization examples
write both initial and optimized surface/coil VTK files; setting
``MAKE_MOVIE=True`` adds a compact animation of accepted iterates. Set the
examples' ``MOVIE_SURFACE_COLOR`` to ``None``, ``"absB"``, ``"B.n/B"``, or
a scalar-field callable to control boundary coloring without storing VTK data
for every iteration.

Coupled free-boundary adjoint
-----------------------------

Let :math:`F(z,c)=0` be the converged VMEC force residual after NESTOR has
computed the vacuum pressure on the moving edge, with equilibrium state
:math:`z` and coil parameters :math:`c`. For a scalar objective :math:`J`,
VMEX solves

.. math::

   F_z^T\lambda = J_z^T, \qquad
   \frac{dJ}{dc} = J_c - \lambda^T F_c,

:func:`~vmex.core.freeboundary_implicit.solve_free_boundary_implicit` keeps
the host-driven forward iterations off the AD tape, re-evaluates the complete
VMEC--NESTOR residual at the converged state, and solves this transpose system
with one matrix-free GCROT adjoint. A direct ESSOS ``BiotSavart.b_cyl`` field
retains coil shape/current derivatives without writing an mgrid file.

The public construction is explicit: create
:func:`~vmex.core.freeboundary_implicit.make_free_boundary_config`, map the
coil vector to a field with ``field_from_parameters``, call the implicit solve,
stack physics rows with :func:`vmex.core.optimize.residuals_from_tuples`, and
apply ``jax.value_and_grad``. ``take_free_boundary_gradients.py`` validates a
direction against independent re-solves; the two free-boundary single-stage
examples pass the same scalar pair to SciPy.

This path is currently limited to reverse mode. Its low-memory host Krylov
lane peaks near 3--5 GB on the bundled coarse examples, but the first coupled
transpose still takes about one to two minutes to compile on the reference
CPU and is not yet a practical GPU path. The next implementation step is to
eliminate the interior radial block with VMEX's block-tridiagonal solver and
solve only the NESTOR edge Schur complement. Rejected coil steps also need the
same explicit finite trial wall used by the fixed-boundary optimizer before
this experimental path is promoted to the default workflow.
