MOUT file reference
===================

Open straight-axis mirror solves write mirror-native ``mout_*.nc`` NetCDF
files (:mod:`vmex.mirror.output`). ``mout`` is deliberately separate from
VMEC's toroidal ``wout`` schema: it stores physical-grid arrays so a solved
open-ended equilibrium can be plotted or inspected without reconstructing the
solver objects. Open mirror data are never encoded as a toroidal WOUT file,
and a periodic MOUT schema for the closed hybrid is deliberately not inferred
from the open end-cut schema (the closed hybrid writes a reviewed PNG and
JSON summary directly from the solved objects).

Reading and writing
-------------------

- :func:`vmex.mirror.output.write_mout` — write a
  :class:`~vmex.mirror.output.MoutData` to NetCDF (schema-stamped).
- :func:`vmex.mirror.output.read_mout` — read and validate the schema.
- :func:`vmex.mirror.output.mout_from_result` — build a
  :class:`~vmex.mirror.output.MoutData` from a solve result.
- ``vmex --plot mout_*.nc`` — render the mirror figure set
  (:doc:`/howto/plot-diagnostics`).

Contents (``MoutData``)
-----------------------

Grid and geometry
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - variable
     - shape
     - meaning
   * - ``s``
     - ``(ns,)``
     - radial flux label grid
   * - ``theta``
     - ``(ntheta,)``
     - poloidal angle grid
   * - ``xi``
     - ``(nxi,)``
     - nonperiodic axial coordinate grid
   * - ``z``
     - ``(nxi,)``
     - axial position of each ``xi`` node
   * - ``boundary_radius``
     - ``(ntheta, nxi)``
     - LCFS radius
   * - ``radius_scale``
     - ``(ns, ntheta, nxi)``
     - nested-surface radial scale (defines the geometry)

Fields and profiles
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - variable
     - shape
     - meaning
   * - ``lambda_stream``
     - ``(ns, ntheta, nxi)``
     - stream function
   * - ``mod_b``
     - ``(ns, ntheta, nxi)``
     - ``|B|`` reconstructed from the same radial Gauss cells used by the
       magnetic-energy functional
   * - ``b_xyz``
     - ``(ns, ntheta, nxi, 3)``
     - Cartesian magnetic field samples (kept separate for field-line
       direction)
   * - ``pressure``
     - ``(ns, ntheta, nxi)``
     - isotropic pressure
   * - ``history``
     - ``(iterations, k)``
     - solver residual history table
   * - ``coil_xyz``
     - coil polylines
     - optional coil curves for plotting

Scalars and diagnostics
~~~~~~~~~~~~~~~~~~~~~~~

``ftol``, ``iterations``, ``converged``, ``mass_scale``,
``variational_max`` (defines nonlinear convergence), ``normal_stress_rms``,
``b_normal_rms``, ``staggered_weak_max`` (the independent staggered-weak
residual, assembled on the energy quadrature), ``pointwise_force_rms``
(the reconstructed ``J x B - grad(p)`` norm; an independent diagnostic, not
the nonlinear ``ftol``), ``normalized_divergence_rms`` (checks the field
representation), ``message``, and the ``schema`` attribute
(``vmex.mirror.model.MIRROR_OUTPUT_SCHEMA``); files with a different schema
are rejected on read.

Free-boundary restart files
---------------------------

Axisymmetric free-boundary beta scans can write one compressed ``.npz``
restart per beta point (schema ``vmex.mirror.free_boundary_restart/3``):

- :class:`vmex.mirror.output.FreeBoundaryRestart` — the coefficient-native
  state: ``boundary`` (:class:`~vmex.mirror.splines.SplineMirrorBoundary`),
  ``plasma_state`` (:class:`~vmex.mirror.splines.SplineMirrorState`), and the
  calibrated ``mass_scale``.
- :func:`vmex.mirror.output.save_free_boundary_restart` — atomic, compressed,
  data-only write (arrays: ``boundary_radius_coefficients``,
  ``radius_coefficients``, ``lambda_coefficients``, ``mass_scale``).
- :func:`vmex.mirror.output.load_free_boundary_restart` — checks the schema
  and coefficient shapes against the target discretization before returning.
  Schema 2 migration requires the original nodal grid explicitly; schema 3
  never guesses it.

Restart files contain only the plasma state, boundary, and pressure scale;
the boundary-integral potential is recomputed on load because the moving
boundary changes at every continuation point. The scan workflow is in
:doc:`/howto/mirror-machines`.
