Contributing
============

Workflow
--------

1. Add or extend a kernel with a focused API.
2. Add an example that writes an ``.npz`` artifact and prints diagnostics.
3. Add a regression test (fast) against a bundled ``wout`` reference.
4. Keep JAX gotchas in mind:

   - jitted functions should only take arrays / PyTrees,
   - avoid duplicate PyTree registration (make registration idempotent),
   - keep static objects out of jitted call signatures unless they are PyTrees.

Development installs::

  pip install -e .[dev]

