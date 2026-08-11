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

.. automodule:: vmex.core.freeboundary_linear
   :members:

.. automodule:: vmex.core.freeboundary_diff
   :members:

.. automodule:: vmex.core.mgrid
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
