CLI reference
=============

The ``vmec`` command is a drop-in equivalent of the ``xvmec2000`` executable:
it parses the input deck, runs the ``NS_ARRAY`` multigrid ladder with
VMEC2000-format console output, writes ``wout_<case>.nc``, and prints the
termination summary.

Usage
-----

.. code-block:: text

   vmex input.X                — solve (INDATA or VMEC++ JSON), write wout_X.nc
   vmex --plot wout_*.nc       — diagnostic plots from a WOUT file
   vmex --plot mout_*.nc       — straight-axis mirror diagnostics
   vmex --booz wout_*.nc       — run booz_xform_jax, write boozmn_*.nc
   vmex --plot boozmn_*.nc     — Boozer contour/spectrum plots
   vmex --doctor               — installation and JAX backend diagnostics
   vmex --test                 — run and plot the bundled quick-start case

The positional argument is a VMEC input file (``input.*`` namelist or a
VMEC++-style ``.json`` deck), or a ``wout_*.nc``/``mout_*.nc``/``boozmn_*.nc``
file for ``--plot``/``--booz``.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Meaning
   * - ``--plot [PATH]``
     - Generate plots. With a ``wout_*.nc`` file, plot WOUT diagnostics; with
       a ``mout_*.nc`` file, plot horizontal straight-axis mirror diagnostics;
       with a ``boozmn_*.nc`` file, plot Boozer diagnostics; with an input
       file, solve first and plot the resulting WOUT.
   * - ``--booz``
     - Run ``booz_xform_jax`` after solving, or directly from a ``wout_*.nc``
       file, and write ``boozmn_*.nc``.
   * - ``--mbooz N`` / ``--nbooz N``
     - Boozer poloidal / toroidal resolution (default 32 each).
   * - ``--booz-surfaces S``
     - Boozer surfaces: comma/space-separated normalized ``s`` values, or
       ``all`` (default).
   * - ``--outdir DIR``
     - Directory for wout/boozmn/figure output (default: alongside the
       input).
   * - ``--quiet``
     - Silence the VMEC-style stdout.
   * - ``--mode {cli,jit}``
     - Solver lane: ``cli`` (jitted blocks with host residual checks, live
       printing, exact-``ftol`` exit; default) or ``jit`` (single
       ``lax.while_loop``).
   * - ``--ftol X``
     - Override the final-stage ``FTOL_ARRAY`` tolerance.
   * - ``--max-iter N``
     - Override the final-stage ``NITER_ARRAY`` iteration cap.
   * - ``--coils PATH``
     - ESSOS-style coils file (``.json`` or ``.npz`` with ``dofs_curves``,
       ``dofs_currents``, ``n_segments``, ``nfp``, ``stellsym``) supplying
       the external field of an ``LFREEB = T`` deck directly via Biot-Savart
       (pairs with ``MGRID_FILE = 'DIRECT_COILS'``).
   * - ``--doctor``
     - Print installation, Python, package, and JAX backend diagnostics.
   * - ``--test``
     - Run the bundled ``input.nfp4_QH_warm_start`` quick-start case: solve,
       write the wout file, and plot it (into ``./vmex_test/`` or
       ``--outdir``).
   * - ``--version``
     - Print the package version.

Free-boundary routing
---------------------

For ``LFREEB = T`` decks:

- a readable ``MGRID_FILE`` runs the free-boundary solver with the VMEC2000
  console output (``In VACUUM`` block, ``VACUUM PRESSURE TURNED ON`` banner)
  and free-boundary wout metadata (``nextcur``/``extcur``/``curlabel``/
  ``mgrid_mode``);
- a **missing** mgrid file falls back to a fixed-boundary solve with a
  warning (VMEC2000 behavior, dropped by VMEC++);
- ``MGRID_FILE = 'DIRECT_COILS'`` (or the ``--coils`` flag) builds the external
  field from an ESSOS coils file (``essos.coils.Coils``): the coils are tabulated
  into an in-memory mgrid (``Coils.to_mgrid``) and read back as an
  :class:`vmex.core.mgrid.MgridField` (requires ESSOS).

Known divergences of the current free-boundary lane: it is single-grid (only
the final ``NS_ARRAY`` stage runs; multi-stage decks print a note), and the
NESTOR potential is not yet exported to the wout ``potsin``/``xmpot``/
``xnpot``/``*_sur`` variables (written as netCDF fill). An NITER-exhausted
free-boundary run still writes the wout (VMEC2000 behavior) and exits with
``ier_flag = 2``.

Exit codes (zero-crash policy)
------------------------------

Every failure maps to a typed :class:`vmex.core.errors.VmecError`; the
CLI prints the VMEC2000 ``werror`` message plus a one-line hint and exits
with the matching ``ier_flag`` code (0 on success, 2 for "MORE ITERATIONS
REQUIRED", etc.). There are no raw tracebacks in normal operation.
