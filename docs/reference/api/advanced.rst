Advanced API
============

Everything below :doc:`basic`: the solver internals of :mod:`vmex.core` and
the open-field-line lane :mod:`vmex.mirror`, grouped as in
:doc:`/explanation/architecture`. Every docstring names the VMEC2000
counterpart it ports.

Profiles
--------

.. automodule:: vmex.core.profiles
   :members:

Radial basis and axis regularity
--------------------------------

.. automodule:: vmex.core.radial_basis
   :members:

High-order reconstruction and force certificate
------------------------------------------------

.. automodule:: vmex.core.strong_force
   :members:

High-order correction transfer and preconditioner
-------------------------------------------------

.. automodule:: vmex.core.polish
   :members:

.. automodule:: vmex.core.polish_driver
   :members:

.. automodule:: vmex.core.polish_implicit
   :members:

The single-grid :func:`vmex.solve` entry point accepts ``polish=False``
(unchanged behavior), ``polish=True`` (required correction), or
``polish="auto"`` (skip an already-certified state).  Polished data is attached
to the result without replacing its legacy sampled state or wout-compatible
arrays.

After the alpha=1 correction has passed the independent certificate,
:func:`vmex.implicit_polished_state` supplies an optimization-facing reverse
pass.  It differentiates the converged root with one matrix-free transposed
Krylov solve, rather than replaying continuation.  Explicit
:func:`vmex.strong_root_tangent` and :func:`vmex.strong_root_adjoint` calls
return true-residual convergence reports for forward and reverse sensitivity
work.

Spectral representation and physics kernels
-------------------------------------------

.. automodule:: vmex.core.fourier
   :members:

.. automodule:: vmex.core.transforms
   :members:

.. automodule:: vmex.core.geometry
   :members:

.. automodule:: vmex.core.fields
   :members:

.. automodule:: vmex.core.forces
   :members:

.. automodule:: vmex.core.residuals
   :members:

Solver
------

.. automodule:: vmex.core.setup
   :members:

.. automodule:: vmex.core.preconditioner
   :members:

.. automodule:: vmex.core.preconditioner_2d
   :members:

.. automodule:: vmex.core.step
   :members:

.. automodule:: vmex.core.solver
   :members:

.. automodule:: vmex.core.multigrid
   :members:

.. automodule:: vmex.core.restart
   :members:

.. automodule:: vmex.core.device
   :members:

Free boundary
-------------

.. automodule:: vmex.core.vacuum
   :members:

.. automodule:: vmex.core.freeboundary
   :members:

.. automodule:: vmex.core.freeboundary_implicit
   :members:

.. automodule:: vmex.core.freeboundary_linear
   :members:

.. automodule:: vmex.core.virtual_casing
   :members:

``vmex.core.freeboundary_diff`` remains as a compatibility name for this
prescribed-interface API. It does not differentiate through a moving-boundary
NESTOR equilibrium solve.

.. automodule:: vmex.core.mgrid
   :members:

.. automodule:: vmex.core.extender
   :members:

Physics objectives
------------------

The objective catalog with usage snippets is :doc:`/reference/objectives`.

.. automodule:: vmex.core.omnigenity
   :members:

.. automodule:: vmex.core.bounce
   :members:

.. automodule:: vmex.core.qi
   :members:

.. automodule:: vmex.core.maxj
   :members:

.. automodule:: vmex.core.gammac
   :members:

.. automodule:: vmex.core.bootstrap
   :members:

.. automodule:: vmex.core.stability
   :members:

.. automodule:: vmex.core.turbulence
   :members:

Outputs
-------

.. automodule:: vmex.core.scaling
   :members:

.. automodule:: vmex.core.nyquist
   :members:

.. automodule:: vmex.core.postprocess
   :members:

.. automodule:: vmex.core.printing
   :members:

.. automodule:: vmex.core.plotting
   :members:

.. automodule:: vmex.core.boozer
   :members:

Straight-axis mirrors
---------------------

.. automodule:: vmex.mirror.analytic
   :members:

.. automodule:: vmex.mirror.splines
   :members:

.. automodule:: vmex.mirror.model
   :members:

.. automodule:: vmex.mirror.solver
   :members:

.. automodule:: vmex.mirror.free_boundary
   :members:

.. automodule:: vmex.mirror.implicit
   :members:

.. automodule:: vmex.mirror.output
   :members:

Errors and CLI
--------------

.. automodule:: vmex.core.errors
   :members:

.. automodule:: vmex.core.cli
   :members:
