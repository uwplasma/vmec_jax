Installation
============

Requirements
------------

- Python 3.10+ (Python 3.12+ recommended for current accelerator-enabled JAX)
- ``numpy``, ``jax`` + ``jaxlib``, ``netCDF4``, ``matplotlib``,
  ``booz_xform_jax`` (all installed automatically)

From PyPI
---------

.. code-block:: bash

   pip install vmex

JAXopt and Optax are optional because SciPy and the public problem callables
are part of the core install.  Install the external-optimizer examples with:

.. code-block:: bash

   pip install "vmex[optimizers]"

The plain install includes everything needed for solving, plotting, and the
Boozer transform — there are no user-facing extras to remember. Verify with:

.. code-block:: bash

   vmex --doctor
   vmex --test

``vmex --doctor`` diagnoses mixed-Python environments (it prints the active
interpreter, pip location, package versions, and JAX backend). If an install
misbehaves, first check that ``pip --version`` and ``python -m pip --version``
point at the same Python.

From conda-forge
----------------

.. code-block:: bash

   conda install --channel conda-forge vmex

or, with `Pixi <https://pixi.prefix.dev/>`_, ``pixi add vmex``. The
`feedstock <https://github.com/conda-forge/vmex-feedstock>`_ may lag PyPI.

From source
-----------

.. code-block:: bash

   git clone https://github.com/uwplasma/vmex
   cd vmex
   pip install -e .          # editable install, recommended for development

Float64 (required)
------------------

VMEC's numerics require double precision. ``vmex`` enables JAX x64 mode
itself when you use the CLI or the core solver entry points; if you drive JAX
directly in your own scripts, set:

.. code-block:: bash

   export JAX_ENABLE_X64=1

or ``jax.config.update("jax_enable_x64", True)`` before solving.

GPU support
-----------

GPU-enabled JAX is intentionally not forced by ``vmex`` because the right
wheel depends on your platform and CUDA/ROCm version. Install the CPU package
first, then install JAX for your accelerator following the
`official JAX installation matrix <https://docs.jax.dev/en/latest/installation.html>`_,
e.g.:

.. code-block:: bash

   pip install -U "jax[cuda13]"

CUDA 13 wheels currently require an NVIDIA driver version of at least 580
and a Python version supported by the current JAX release. On older Python
versions, package resolution can select an older JAX release whose accelerator
extras differ; always confirm the result with ``vmex --doctor``. CUDA 12,
ROCm, TPU, and platform-specific alternatives remain documented in JAX's
installation matrix.

``vmex`` then picks CPU or GPU per forward solve using a measured device
policy. High-level optimization defaults its implicit Jacobians to CPU because
they are launch-bound on the tested GPUs; low-level
:func:`vmex.core.implicit.run` follows ordinary JAX placement when ``device``
is omitted. See :ref:`performance:GPU guidance` for the policies and
persistent compilation cache. Environment variables are not required for
ordinary hardware detection.

Build the documentation locally
-------------------------------

.. code-block:: bash

   pip install ".[docs]"
   python -m sphinx -W -j auto -b html docs docs/_build/html

``SPHINX_FAST=1`` builds only a minimal landing page for quick CI checks.
