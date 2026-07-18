API reference
=============

The production API is :mod:`vmex.core`. Modules are grouped as in
:doc:`/architecture`; every docstring names the VMEC2000 counterpart it
ports.

Inputs and profiles
-------------------

.. automodule:: vmex.core.input
   :members:

.. automodule:: vmex.core.profiles
   :members:

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

.. automodule:: vmex.core.device
   :members:

Free boundary
-------------

.. automodule:: vmex.core.vacuum
   :members:

.. automodule:: vmex.core.freeboundary
   :members:

.. automodule:: vmex.core.freeboundary_diff
   :members:

.. automodule:: vmex.core.mgrid
   :members:

Differentiation and optimization
--------------------------------

.. automodule:: vmex.core.implicit
   :members:

.. automodule:: vmex.core.optimize
   :members:

.. automodule:: vmex.core.parallel
   :members:

Physics objectives
------------------

The objective catalog with usage snippets is :doc:`/objectives`.

.. automodule:: vmex.core.omnigenity
   :members:

.. automodule:: vmex.core.bootstrap
   :members:

.. automodule:: vmex.core.stability
   :members:

.. automodule:: vmex.core.turbulence
   :members:

Outputs
-------

.. automodule:: vmex.core.wout
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

Errors and CLI
--------------

.. automodule:: vmex.core.errors
   :members:

.. automodule:: vmex.core.cli
   :members:
