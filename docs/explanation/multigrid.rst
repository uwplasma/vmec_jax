The multigrid ladder
====================

``NS_ARRAY`` runs the solve as a ladder of radial resolutions: each stage
converges at its own ``ns`` with its own ``FTOL_ARRAY``/``NITER_ARRAY``
entry, and the converged coefficients seed the next finer grid through
VMEC2000's exact ``interp.f`` transfer. The same interpolation seam provides
hot restart from a previous solution — the recipe side is
:doc:`/howto/restart-from-previous-run` and :doc:`/howto/parameter-scans`.

Stage transfer (``interp.f``)
-----------------------------

:func:`vmex.core.multigrid.solve_multigrid` runs the ``NS_ARRAY``
ladder: each stage solves at its ``ns`` with its own
``FTOL_ARRAY``/``NITER_ARRAY`` entry, and the converged coefficients are
interpolated in :math:`\sqrt{s}`-internal form to the next finer grid
(``interp.f``): scale by ``scalxc``, extrapolate odd-m modes to the axis on
the scaled array, interpolate linearly in :math:`s`, unscale, and zero the
odd-m axis row (:func:`~vmex.core.multigrid.interpolate_coefficients` /
:func:`~vmex.core.multigrid.interpolate_state`). In equations, the staging
interpolates *scaled* coefficients between grids,

.. math::

   x_{\mathrm{scaled}} = x \cdot \mathrm{scalxc},

with odd-m extrapolation to the axis performed **before** interpolation;
after linear interpolation on a uniform radial grid, coefficients are
unscaled:

.. math::

   x_{\mathrm{new}} = \frac{x_{\mathrm{scaled,new}}}{\mathrm{scalxc}_{\mathrm{new}}}.

VMEX implements this exact pipeline so that stage-to-stage coefficient
transfer matches VMEC2000.

Each distinct stage structure may compile once; repeated
ladders with the same structures reuse those executables. A single
maximum-resolution masked executable is future work, not current behavior.
``--prefetch-compile`` controls whether upcoming stage executables are
compiled ahead of time or on first use (the CLI defaults to the
lower-memory on-demand path).

Carried module state
--------------------

The transfer includes VMEC2000's non-geometric module state.  In particular,
``initialize_radial.f`` resets ``fsq``, ``iter1``, ``iter2``, ``ijacob``,
and the time-step controller, but it does **not** reset
``fsqr/fsqz/fsql``.  VMEX therefore passes the previous stage's three
invariant residuals into the first force evaluation on the next grid.  This
matters in free boundary: ``residue.f90`` uses the retained
``fsqr + fsqz`` to decide whether the carried edge-force row belongs in the
first fine-grid norm.  Axis re-guess transfers and bounded JAC75 retries
retain the same residual state instead of silently becoming cold starts.

Free-boundary ladders
---------------------

:func:`vmex.core.multigrid.solve_free_boundary_multigrid` implements
``runvmec.f``'s radial ladder.  Increasing grids interpolate ``xstore`` using
the same odd-m :math:`\sqrt{s}` scaling as fixed boundary; equal grids rerun
the current state and the ladder stops at the first decreasing entry.
``ivac``, adaptive ``nvacskip``, the exact ``rbsq`` edge product, and the
three invariant residual channels are carried.  Because the free-boundary
block is guarded by ``iter2 > 1``, a new stage uses that carried edge product
on iteration 1 and performs its first full update on iteration 2.  The
resolution-specific NESTOR basis, Green-function program, axis-current
filament program, cached potential matrix, and traced cadence loop are selected
or rebuilt for the new stage.  Vacuum activates only once across the ladder
(see :doc:`nestor-vacuum`).

If a first requested grid remains invalid after ``guess_axis``, the fixed- or
free-boundary driver retries the ladder once with an ``ns=3``, ``ftol=1e-4``
stage prepended.  The coarse equilibrium is then interpolated to the user's
first grid; free-boundary vacuum activation restarts cleanly.  This narrow
recovery follows the current VMEC++ driver; it fires only for
``bad_jacobian_flag`` and does not hide convergence, non-finite, or input
failures.  ``coarse_grid_retry=False`` disables it for strict failure studies.

Hot restart
-----------

The same interpolation seam provides hot restart
(:func:`vmex.core.solver.hot_restart_state`): a previous solution (e.g.
the previous point of a parameter scan) can seed the solve directly, at the
same or a different radial resolution, via ``solve(...,
initial_state=state)`` and ``solve_multigrid(..., initial_state=state)``.
Restarting from a ``wout_*.nc`` file — including VMEC2000-written ones —
goes through :mod:`vmex.core.restart` (``restart_from`` / ``--restart`` /
the ``RESTART_WOUT`` deck key), which inverts the wout output maps back to
the internal spectral state, resamples radial/mode-table differences, and
drops every leading ``NS_ARRAY`` rung whose resolution the seed already
meets or exceeds. The recipe with measured iteration counts is
:doc:`/howto/restart-from-previous-run`.
