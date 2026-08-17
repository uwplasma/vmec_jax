Basic API
=========

The API you use daily: the lazily imported ``vmex`` top-level exports
(``import vmex as vj``) plus the three modules behind gradient-based work.
Everything else — the per-module solver internals and the mirror lane — is
in :doc:`advanced`.

Top-level package
-----------------

.. automodule:: vmex
   :no-members:

Inputs
------

.. automodule:: vmex.core.input
   :members:

Differentiation and optimization
--------------------------------

A converged :class:`vmex.core.optimize.Equilibrium` exposes
``equilibrium.solution`` (the spectral equilibrium arrays) and
``equilibrium.solver_context`` (read-only grids, profiles, and constants).
The shorter ``state`` and ``runtime`` attribute names remain compatible with
existing code; ``runtime`` never means elapsed wall-clock time.

.. automodule:: vmex.core.implicit
   :members:

.. automodule:: vmex.core.optimize
   :members:

.. automodule:: vmex.core.parallel
   :members:

Outputs
-------

.. automodule:: vmex.core.wout
   :members:

Optional neoclassical diagnostics
---------------------------------

.. automodule:: vmex.core.neoclassical
   :members:
