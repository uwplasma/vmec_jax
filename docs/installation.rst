Installation
============

Requirements
------------

- Python 3.10+
- ``numpy``, ``jax`` + ``jaxlib``, ``netCDF4``, ``matplotlib``,
  ``booz_xform_jax`` (all installed automatically)

From PyPI
---------

.. code-block:: bash

   pip install vmex

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

   pip install -U "jax[cuda12]"

``vmex`` then picks CPU or GPU per solve using a measured device policy —
small decks stay on the CPU, large ones move to the GPU. See
:ref:`performance:GPU guidance` for the policy, how to pin a backend with
``JAX_PLATFORMS``, and the persistent compilation cache.

Build the documentation locally
-------------------------------

.. code-block:: bash

   pip install ".[docs]"
   python -m sphinx -W -j auto -b html docs docs/_build/html

``SPHINX_FAST=1`` builds only a minimal landing page for quick CI checks.
