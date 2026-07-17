:orphan:

VMEX documentation (fast build)
===============================

Fast doc builds are enabled (``SPHINX_FAST=1``). The full user guide and API
reference are skipped to keep CI fast.

To build the full documentation locally, run:

.. code-block:: bash

   LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html
