Quickstart
==========

This page walks through a first session with ``vmex``: verify the
installation, run an equilibrium, plot it, and run the Boozer transform. All
commands work from any directory after ``pip install vmex``.

Verify the installation
-----------------------

.. code-block:: bash

   vmex --doctor
   vmex --test

``vmex --doctor`` prints the active Python, package versions, and the JAX
backend (CPU/GPU). ``vmex --test`` runs the bundled fixed-boundary QH case
end to end: it copies the packaged ``input.nfp4_QH_warm_start`` deck into
``./vmex_test/``, solves it (with ``FTOL_ARRAY = 1e-12`` for a fast first
check), writes ``wout_nfp4_QH_warm_start.nc``, and renders diagnostic figures
into ``vmex_test/figures/``. It also prints the equivalent manual
commands so you can reproduce each step yourself.

First run
---------

``vmec`` behaves like the ``xvmec2000`` executable: point it at a VMEC input
file (an ``input.*`` INDATA namelist or a VMEC++-style ``.json`` deck):

.. code-block:: bash

   vmex input.circular_tokamak

This prints the VMEC2000-format iteration table (``ITER  FSQR  FSQZ  FSQL
RAX(v=0)  DELT  WMHD``), runs the full ``NS_ARRAY`` multigrid ladder, and
writes ``wout_circular_tokamak.nc`` next to the input file. Useful flags:

- ``--outdir DIR`` — write outputs elsewhere,
- ``--quiet`` — silence the iteration table,
- ``--ftol X`` / ``--max-iter N`` — override the final-stage tolerance or
  iteration cap,
- ``--mode jit`` — run the fully traced ``lax.while_loop`` solver lane instead
  of the default host-blocked CLI lane (see :doc:`architecture`).

Free-boundary decks (``LFREEB = T``) route automatically: a readable
``MGRID_FILE`` runs the free-boundary solver; a missing mgrid file falls back
to a fixed-boundary solve with a warning (VMEC2000 behavior); and
``MGRID_FILE = 'DIRECT_COILS'`` together with ``--coils coils.json`` evaluates
the external field directly from an ESSOS-style coil set via Biot-Savart —
no mgrid interpolation at all. See :doc:`cli` for the complete reference.

Plotting
--------

Every ``wout_*.nc`` file (from ``vmex`` or from VMEC2000 itself) can be
plotted directly:

.. code-block:: bash

   vmex --plot wout_circular_tokamak.nc
   vmex input.circular_tokamak --plot     # solve, then plot in one command

This writes a set of figures next to the file (or into ``--outdir``): a
summary panel, flux-surface cross-sections at several toroidal angles,
``|B|`` on the boundary, radial profiles (pressure, iota, current), and a 3D
boundary rendering.

Boozer coordinates
------------------

The plain install includes the differentiable ``booz_xform_jax`` transform:

.. code-block:: bash

   vmex input.nfp4_QH_warm_start --booz          # solve + Boozer transform
   vmex wout_nfp4_QH_warm_start.nc --booz        # transform an existing wout
   vmex --plot boozmn_nfp4_QH_warm_start.nc      # Boozer |B| contours + spectra

``--booz`` writes a standard ``boozmn_*.nc`` file. The transform resolution
and surfaces are configurable:

.. code-block:: bash

   vmex wout_nfp4_QH_warm_start.nc --booz --mbooz 48 --nbooz 48 \
        --booz-surfaces "0.25, 0.5, 1.0"

Python API
----------

The production solver lives in :mod:`vmex.core`. A minimal solve from
Python:

.. code-block:: python

   from vmex.core.input import VmecInput
   from vmex.core import optimize as opt
   from vmex.core.wout import write_wout

   inp = VmecInput.from_file("input.circular_tokamak")
   eq = opt.solve_equilibrium(inp)        # full NS_ARRAY ladder -> Equilibrium
   r = eq.result
   print("converged:", r.converged, "in", r.iterations, "iterations")

   print("aspect ratio:", float(eq.wout.aspect))   # wout built lazily, cached
   write_wout("wout_circular_tokamak.nc", eq.wout)

:func:`~vmex.core.optimize.solve_equilibrium` bundles the converged
state with its evaluation contexts: ``eq.state`` and ``eq.runtime`` feed the
differentiable scalar targets of :mod:`~vmex.core.optimize` (e.g.
``opt.aspect_ratio(eq.state, eq.runtime)``), and ``eq.wout`` is the full
VMEC2000 wout dataset (built on first access — no manual
``wout_from_state`` plumbing needed).

``VmecInput`` is a frozen dataclass with VMEC2000 semantics and defaults —
you can also build one from scratch in Python (all INDATA fields are keyword
arguments; see :doc:`input_reference`) and round-trip it to INDATA or
VMEC++-style JSON.

Choosing an entry point
-----------------------

Four solve entry points share the same numerics; pick by what you need back:

.. list-table::
   :header-rows: 1
   :widths: 30 24 46

   * - entry point
     - returns
     - use when
   * - :func:`vmex.core.optimize.solve_equilibrium`
     - ``Equilibrium`` (state + runtime + lazy ``.wout``)
     - **Default for Python work**: analysis, objectives, anything that
       reads wout tables or the ``(state, runtime)`` scalar targets.
   * - :func:`vmex.core.multigrid.solve_multigrid`
     - ``SolveResult`` (state + convergence data)
     - You only need the converged state / iteration diagnostics — the
       engine behind the CLI and ``solve_equilibrium``.
   * - :func:`vmex.core.implicit.run`
     - ``ImplicitSolution`` (differentiable pytree, carries ``.runtime``)
     - Gradients: wrap it in ``jax.grad``/``jax.value_and_grad`` — the
       implicit-adjoint path of :doc:`optimization`.
   * - :func:`vmex.core.solver.solve`
     - ``SolveResult`` (one grid stage)
     - Low-level single-``ns`` building block (no NS_ARRAY ladder);
       mainly for solver development and tests.

For gradient-based work, wrap the solve with the implicit-differentiation
driver in :mod:`vmex.core.implicit` (:func:`~vmex.core.implicit.run`)
and the objectives in :mod:`vmex.core.optimize` — see :doc:`optimization`.

Reading wout files
------------------

.. code-block:: python

   from vmex.core.wout import read_wout

   wout = read_wout("wout_circular_tokamak.nc")
   print("aspect ratio:", float(wout.aspect))
   print("edge iota:   ", float(wout.iotaf[-1]))
   print("beta total:  ", float(wout.betatotal))

The written files carry the full VMEC2000 variable set (:doc:`wout_reference`)
and load unchanged in simsopt, booz_xform, and other VMEC-ecosystem tools.

Where to go next
----------------

- :doc:`tutorials` — worked examples (fixed boundary, free boundary,
  optimization).
- :doc:`architecture` — how the core is organized and how it maps onto
  VMEC2000 subroutines.
- :doc:`performance` — benchmarks, parity results, and GPU guidance.
