VMEX
====

**VMEX** is a clean-room, JAX-native reimplementation of the **VMEC2000**
ideal-MHD equilibrium code for stellarators and tokamaks. It solves fixed- and
free-boundary equilibria with VMEC2000-derived numerics, writes standard
``wout_*.nc`` output, and — unlike the Fortran original — provides
differentiable fixed-boundary equilibria plus differentiable virtual-casing
external-field residuals on a specified free boundary. It runs on CPUs and
GPUs. Exact support and validation scope is in :doc:`capabilities`;
input-flag coverage is in :doc:`vmec2000_compatibility`.

Why VMEX?
---------

- **VMEC2000 parity.** The solver ports the VMEC2000 algorithms
  (steepest-descent moment method, 1D radial preconditioner, Richardson time
  stepping, spectral condensation, NESTOR vacuum solve) constant-for-constant.
  Representative benchmark decks converge in the same iteration counts as
  VMEC2000 and tested ``wout`` variables agree within their stated tolerances
  (see :doc:`performance`). An optional 2D
  block preconditioner cuts iterations 2.5–11x on stiff decks while leaving
  the default path byte-identical.
- **Differentiable.** Gradients of fixed-boundary equilibrium properties
  with respect to boundary shape and profile parameters via implicit
  differentiation of the converged fixed point
  (:mod:`vmex.core.implicit`) — no finite differences, no iteration
  unrolling — validated against central finite differences (see
  :doc:`optimization`), with an O(1)-memory adjoint. Virtual-casing residuals
  are differentiable in coil / ``extcur`` parameters on a specified boundary
  and finite-difference-validated; the host-driven NESTOR equilibrium solve
  itself is not differentiated (:mod:`vmex.core.freeboundary_diff`). A
  growing :doc:`objectives
  library <objectives>` — quasisymmetry, omnigenity, Redl bootstrap,
  ballooning stability, gyrokinetic turbulence proxies — plugs straight
  into a least-squares driver with those exact gradients, reaching precise
  QA in a single 14.5-minute CPU call (:doc:`optimization`).
- **Drop-in workflow.** The ``vmec`` command reads VMEC2000 ``input.*``
  namelists and structured JSON, prints VMEC2000-format iteration output,
  and writes ``wout_*.nc`` files that load unchanged in simsopt and
  booz_xform.
- **Batteries included.** Built-in plotting (``vmex --plot``), Boozer
  transform (``vmex --booz`` via ``booz_xform_jax``), spline profiles,
  multigrid with hot restart, free boundary from mgrid files or fields
  tabulated from coils, and typed
  zero-crash error handling. The shared linear/adjoint solver layer is
  factored out into `SOLVAX <https://pypi.org/project/solvax/>`_.

Quickstart
----------

.. code-block:: bash

   pip install vmex
   vmex --test                       # bundled QH case: solve + wout + plots
   vmex input.circular_tokamak       # run any VMEC input deck
   vmex --plot wout_circular_tokamak.nc

See :doc:`quickstart` for a full tour, including the Python API and the
Boozer-coordinate workflow.

.. figure:: _static/figures/readme_runtime_compare.png
   :alt: Runtime comparison of vmex against VMEC2000 and a reference C++ implementation
   :align: center
   :width: 95%

   Benchmark-suite runtimes: ``vmex`` (cold and warm) versus VMEC2000 and
   the reference. Warm (compiled-cache) solves are the relevant number for
   optimization loops; see :doc:`performance` for the full table.

Documentation
-------------

.. only:: not fast

   .. toctree::
      :maxdepth: 1
      :caption: Getting started

      installation
      quickstart
      scaling

   .. toctree::
      :maxdepth: 1
      :caption: Tutorials

      tutorials

   .. toctree::
      :maxdepth: 1
      :caption: Theory and numerics

      theory
      equations
      algorithms
      confinement
      architecture
      mirror_geometry

   .. toctree::
      :maxdepth: 1
      :caption: Reference

      api/index
      cli
      capabilities
      input_reference
      vmec2000_compatibility
      wout_reference

   .. toctree::
      :maxdepth: 1
      :caption: Performance and validation

      performance
      parallelization

   .. toctree::
      :maxdepth: 1
      :caption: Developer guide

      optimization
      objectives
      contributing
      references

.. only:: fast

   Fast doc builds are enabled (``SPHINX_FAST=1``). The full user guide and API
   reference are skipped to keep CI fast.
