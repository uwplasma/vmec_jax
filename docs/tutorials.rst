Tutorials
=========

Every tutorial below is a runnable script in the repository's ``examples/``
directory: parameters at the top, no ``main()``, only the public API, and each
is smoke-tested in CI (``tests/test_examples.py``) so the code on this
page always runs.  Copy a script, edit the parameter block, and go.

Run any of them directly::

   python examples/fixed_boundary_run.py

The light scripts finish in a few seconds once JAX has compiled; the
free-boundary and optimization scripts are heavier (see each section).

.. contents:: On this page
   :local:
   :depth: 1


Getting started
---------------

Fixed-boundary run
~~~~~~~~~~~~~~~~~~~

Read an ``&INDATA`` deck, converge the equilibrium on the multigrid ladder with
VMEC2000-format progress printing, and write and plot the ``wout``.  This is the
three-step workflow every new user needs.

.. literalinclude:: ../examples/fixed_boundary_run.py
   :language: python

All diagnostics and the Boozer transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vmex ships its plotting and its Boozer transform in the box.  This produces
every ``plot_wout`` figure (flux-surface summary, cross-sections, ``|B|``,
profiles, 3D render) and the straight-field-line Boozer ``|B|`` spectrum on the last closed
flux surface — the view used to judge quasisymmetry.

.. literalinclude:: ../examples/plot_and_boozer.py
   :language: python

VMEC++ JSON input
~~~~~~~~~~~~~~~~~

vmex reads both the classic ``&INDATA`` namelist and the VMEC++ JSON schema,
and can write the JSON form — a drop-in for either ecosystem.  This converts a
deck, reads it back, and confirms the two representations describe one
equilibrium.

.. literalinclude:: ../examples/run_from_json.py
   :language: python


Profiles and finite-beta physics
--------------------------------

Profile representations
~~~~~~~~~~~~~~~~~~~~~~~~

Pressure and rotational transform (or current) can be given as polynomial
coefficients (``power_series``) or spline knots (``cubic_spline``).  This solves
the same equilibrium both ways and shows they agree, then points to the
``NCURR=0`` vs ``NCURR=1`` (prescribed iota vs prescribed current) switch.

.. literalinclude:: ../examples/profiles_power_and_spline.py
   :language: python

Finite-beta pressure scan
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ramp the pressure and read three diagnostics straight from the wout: the
volume-averaged beta, the Shafranov shift (outward motion of the magnetic axis),
and the Mercier ``DMerc`` interchange-stability profile.  Each step is
hot-restarted from the previous equilibrium.

.. literalinclude:: ../examples/finite_beta_scan.py
   :language: python


Performance: hot restart
------------------------

Seed each point of a parameter scan from the previous converged state.  Warm
restarts converge in about one iteration instead of hundreds, and because
vmex caches one compiled executable per solver structure, the whole scan
recompiles nothing.

.. literalinclude:: ../examples/hot_restart_scan.py
   :language: python


Differentiation
---------------

Fixed-boundary gradients (implicit differentiation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Differentiate a *converged* equilibrium.  ``jax.grad`` returns exact derivatives
of wout scalars (aspect ratio, magnetic energy, ...) with respect to the
boundary Fourier coefficients and profile parameters, computed by the implicit
function theorem — one adjoint solve, O(1) memory, no finite-difference step to
tune.  The script checks the adjoint gradient against central differences.

.. literalinclude:: ../examples/take_gradients.py
   :language: python

Free-boundary gradients (virtual casing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The differentiable complement for free boundary: the plasma–vacuum interface
mismatch is written as a smooth objective and differentiated with respect to the
coil currents (``extcur``) and coil Fourier shape, finite-difference-validated.

.. literalinclude:: ../examples/take_free_boundary_gradients.py
   :language: python


Free boundary
-------------

From an mgrid file
~~~~~~~~~~~~~~~~~~

Prescribe the *coils* instead of the boundary: the coil currents (``EXTCUR``)
and their tabulated vacuum field (an mgrid file) drive a NESTOR vacuum solve, and
VMEC finds the plasma boundary that balances against it.  The last closed flux
surface is an output, not an input.

.. literalinclude:: ../examples/free_boundary_mgrid.py
   :language: python

Free-boundary beta scan
~~~~~~~~~~~~~~~~~~~~~~~~

Ramp the pressure of the free-boundary case at fixed coil currents; the boundary
is re-solved by NESTOR at every step as the plasma pushes outward against the
external field.

.. literalinclude:: ../examples/free_boundary_beta_scan.py
   :language: python

Directly from ESSOS coils (no mgrid file)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vmex is coil-agnostic: the free-boundary solver consumes only a magnetic
field, so coils can come from ESSOS (``essos.coils.Coils``) instead of a
tabulated mgrid file.  This takes the Landreman–Paul precise-QA modular coil
set, tabulates its Biot–Savart field once into an in-memory
:class:`~vmex.core.mgrid.MgridField`, and runs a free-boundary beta scan
against it — calibrating ``PRES_SCALE`` per step so the converged wout
``betatotal`` lands on 0/1/2/3 %.

.. literalinclude:: ../examples/free_boundary_essos_coils.py
   :language: python


Mirror equilibria
-----------------

Fixed-boundary nonaxisymmetric mirrors and gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve the supported rotating ellipse and the validation-only Straight Field Line
Mirror target with native axial B-splines at ``ftol=1e-12``. The example
asserts every rotating-ellipse gate and reports the SFLM corrected-cut force
failure. Its volume derivative is checked against independently reconverged
central differences before MOUT and the standard plots are written.

.. image:: _static/figures/mirror_fixed_boundary_3d.png
   :alt: Fixed-boundary mirror geometry and magnetic field
   :width: 82%

.. literalinclude:: ../examples/mirror_fixed_boundary_nonaxisymmetric.py
   :language: python

Free-boundary mirror beta scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two circular end coils drive an open-field equilibrium whose lateral LCFS is
solved jointly with the exterior vacuum. The model is supported through 10%;
the 25% and 50% endpoints are converged but remain validation-only because
their independent force/refinement gates fail. The plots include the
horizontal mirror, coils, cap-to-cap field lines, ``|B|``, pressure,
cross-sections, and force history. The default exterior solve is intentionally
a full/nightly workflow.

.. image:: _static/figures/mirror_free_boundary_beta50_summary.png
   :alt: Free-boundary mirror refinement, field response, and force support gates
   :width: 95%

.. literalinclude:: ../examples/mirror_free_boundary_beta_scan.py
   :language: python

Periodic stellarator-mirror hybrid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Join two exactly straight mirror legs with two curved periodic B-spline
returns. The elliptical section rotates by 90 degrees through each return, and
a finite current produces visible field-line pitch. The example performs the
fixed-boundary equilibrium solve before plotting its LCFS ``|B|``, actual
field lines, cross-sections, iota, and convergence diagnostics. Its present
coarse strong-force residual is displayed as a failed validation gate.

.. image:: _static/figures/stellarator_mirror_hybrid.png
   :alt: Periodic spline stellarator-mirror hybrid
   :width: 100%

.. literalinclude:: ../examples/stellarator_mirror_hybrid.py
   :language: python

Independent Pleiades reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The resolution-qualified two-coil beta trend is independently regenerated
with Pleiades at pinned commit ``0161abb3``. Set ``PLEIADES_ROOT`` at the top;
the script writes ignored review output and never silently replaces the bundled
CSV benchmark.

.. literalinclude:: ../examples/validation/pleiades_mirror_reference.py
   :language: python


Optimization
------------

The ``examples/optimization/`` gallery drives a circular torus to precise
quasisymmetric (QA, QH, QP) and quasi-isodynamic (QI) configurations with
gradient-based least squares — user-authored ``(function, target, weight)``
objective terms with implicit-differentiation gradients
(``jac="implicit"``).  See :doc:`objectives` for the full objective library
(quasisymmetry and omnigenity residuals, Redl bootstrap, ballooning
stability, turbulence proxies, scalar targets) and :doc:`optimization` for
the differentiation machinery and the measured campaign timings.

Single-call ESS optimization (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended pattern is **one** ``least_squares`` call with *all* the
boundary harmonics released at once and Exponential Spectral Scaling
(``use_ess=True``) ordering them through the trust region — no
``max_mode`` continuation loop.  Measured: precise QA (QS 7.2e-6) in
14.5 minutes on a CPU.

.. literalinclude:: ../examples/optimization/QA_optimization_ess.py
   :language: python

``QI_optimization_ess.py`` is the quasi-isodynamic analogue: the traceable
Goodman constructed-QI residual (:class:`~vmex.core.omnigenity.QIResidual`)
plus practical targets, one call at ``max_mode = 6`` (25x residual
reduction in 17.3 minutes).

Staged ``max_mode`` continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classic ladder — one least-squares stage per ``max_mode``, each seeded
with the previous stage's boundary — remains available and is what
``QA_optimization.py``, ``QH_optimization.py``, ``QP_optimization.py``, and
``QI_optimization.py`` run (QI with a quasi-poloidal basin stage first).
It reaches the same precision class as the single-call pattern at roughly
twice the wall time; the scripts stay side by side so the comparison is
reproducible.

.. literalinclude:: ../examples/optimization/QA_optimization.py
   :language: python

These are the heaviest examples (hundreds to thousands of solves) and are
exercised in the nightly CI run.

Self-consistent bootstrap current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A different loop: instead of reshaping the boundary, regenerate the *current
profile* until it is consistent with the bootstrap current the plasma itself
drives.  ``QA_bootstrap_selfconsistent.py`` (and its sibling
``QH_bootstrap_selfconsistent.py``)
reproduces the quasi-axisymmetric configuration of Landreman–Buller–Drevlak
(arXiv:2205.02914): it erases the deck's current profile and lets the
fixed-boundary Picard loop
:func:`~vmex.core.bootstrap.self_consistent_bootstrap` rebuild it from the
Redl formula, converging to the paper's mismatch ``f_boot = 2e-6`` in a
handful of hot-restarted iterations.  The physics of the Redl closure is on
:doc:`confinement`.

.. literalinclude:: ../examples/optimization/QA_bootstrap_selfconsistent.py
   :language: python
