CLI reference
=============

The ``vmec`` command is a drop-in equivalent of the ``xvmec2000`` executable:
it parses the input deck, runs the ``NS_ARRAY`` multigrid ladder with
VMEC2000-format console output, writes ``wout_<case>.nc``, and prints the
termination summary.

Usage
-----

.. code-block:: text

   vmex input.X                — solve (INDATA or structured JSON), write wout_X.nc
   vmex --plot wout_*.nc       — diagnostic plots from a WOUT file
   vmex --plot mout_*.nc       — straight-axis mirror diagnostics
   vmex --booz wout_*.nc       — run booz_xform_jax, write boozmn_*.nc
   vmex --plot boozmn_*.nc     — Boozer contour/spectrum plots
   vmex --scale PATH [B R]     — dimensionally scale an input or WOUT
   vmex --doctor               — installation and JAX backend diagnostics
   vmex --test                 — run and plot the bundled quick-start case

The positional argument is a VMEC input file (``input.*`` namelist or a
structured-JSON ``.json`` deck), or a ``wout_*.nc``/``mout_*.nc``/``boozmn_*.nc``
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
   * - ``--scale``
     - Write a scaled input or WOUT. Optional positional factors are
       ``B_scale R_scale``; with no factors the targets are
       ``|b0| = 5.7 T`` and ``Aminor_p = 1.7 m`` (ARIES-CS).
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
   * - ``--device {auto,none,cpu,gpu,cuda,rocm,tpu}``
     - JAX solve placement. ``auto`` (default) applies VMEX's measured policy,
       ``none`` leaves placement to JAX, and the other values request a
       platform explicitly. This applies to fixed- and free-boundary solves.
   * - ``--ftol X``
     - Override the final-stage ``FTOL_ARRAY`` tolerance.
   * - ``--max-iter N``
     - Override the final-stage ``NITER_ARRAY`` iteration cap.
   * - ``--restart WOUT``
     - Hot-restart the solve from a ``wout_*.nc`` file (VMEX- or
       VMEC2000-written): the equilibrium state is rebuilt from the file,
       coarse multigrid rungs at or below its resolution are skipped, and
       radial/mode-table differences are resampled. Overrides a
       ``RESTART_WOUT`` deck entry. See
       :doc:`/howto/restart-from-previous-run`.
   * - ``--prefetch-compile`` / ``--no-prefetch-compile``
     - Opt in to or out of overlapping the next multigrid rung's compilation.
       Sequential compilation is the default and uses less peak memory;
       numerical results are identical.
   * - ``--jacobian-retries N``
     - Retry a stage from its best finite checkpoint after the VMEC2000
       75-Jacobian-reset condition, using a reduced ``DELT`` (default 2).
       Use 0 to preserve VMEC2000's immediate fatal-stop behavior.
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
  warning (retained VMEC2000 behavior);
- ``MGRID_FILE = 'DIRECT_COILS'`` (or the ``--coils`` flag) builds the external
  field from an ESSOS coils file (``essos.coils.Coils``): the coils' Biot-Savart
  field (``essos.fields.BiotSavart``) is tabulated directly into an in-memory
  :class:`vmex.core.mgrid.MgridField` via
  :meth:`~vmex.core.mgrid.MgridField.from_cartesian_field` — no temporary
  mgrid file and no mgrid-export API involved (requires ESSOS,
  ``pip install essos``).

The free-boundary path runs the complete ``NS_ARRAY`` ladder.  It interpolates
the preceding stage's final plasma state, carries VMEC2000's active-vacuum and adaptive
``NVACSKIP`` state, and selects fresh resolution-specific NESTOR programs at
each new grid.  A user-provided ``initial_state`` is also supported by the
Python API for hot restarts.

Hot restart (``--restart`` / ``RESTART_WOUT``)
----------------------------------------------

``vmex input.x --restart wout_y.nc`` seeds the solve (fixed or free
boundary) from any VMEC2000-compatible wout file; the deck can request the
same thing with the VMEX extension key ``RESTART_WOUT = 'wout_y.nc'`` inside
``&INDATA`` (resolved relative to the input file; the CLI flag wins).  The
full R/Z/lambda state is rebuilt exactly, radial/mode-table differences are
resampled, and multigrid rungs at or below the restart resolution are
skipped — see :doc:`/howto/restart-from-previous-run` for the workflow and
:doc:`/explanation/multigrid` for the mechanism.

The CLI exports the final NESTOR potential and surface fields to the wout
``potsin``/``xmpot``/``xnpot``/``*_sur`` variables. LASYM runs additionally
write ``potcos`` and the sine ``*_sur`` partners. An NITER-exhausted
fixed- or free-boundary run terminates through the normal output path —
unconverged WOUT, equilibrium summary, and the ``MORE ITERATIONS REQUIRED``
block (``fileout.f`` semantics) — and exits with the distinct
``ier_flag = 2``.  Fatal numerical/Jacobian failures never produce a WOUT.

Exit codes (zero-crash policy)
------------------------------

Every failure maps to a typed :class:`vmex.core.errors.VmecError`; the
CLI prints the VMEC2000 ``werror`` message plus a one-line hint and exits
with the matching ``ier_flag`` code (0 on success, 2 for "MORE ITERATIONS
REQUIRED", etc.). There are no raw tracebacks in normal operation.
