Contributing
============

Migration status
----------------

The repository recently completed a clean-room rewrite: the production
implementation is :mod:`vmex.core` (~30 focused modules, one concern per
file — see :doc:`architecture`), and the ``vmec`` CLI runs on it end to end.
The remaining top-level legacy modules are being removed in an ongoing
deletion sweep; new code, tests, and documentation should target
``vmex.core`` only. Every core module was validated by A/B equivalence
tests against the parity-proven legacy implementation
(``tests/``) and end to end against VMEC2000 golden runs.

Workflow
--------

1. Add or extend a kernel with a focused API in the matching
   :mod:`vmex.core` module; the module header docstring names the
   VMEC2000 counterpart file it ports — keep that cross-reference current.
2. Add a regression test under ``tests/`` (fast, offline where possible);
   parity-sensitive changes need a check against the golden VMEC2000
   fixtures.
3. Keep JAX gotchas in mind:

   - jitted functions should only take arrays / pytrees (static config stays
     hashable and out of traced signatures),
   - solver functions are pure ``state -> state`` — no hidden host state,
   - float64 is mandatory (enforced at solver import).

Development install and checks::

  pip install -e .[dev]
  ruff check .
  pytest -q

Documentation builds must pass strict mode::

  python -m sphinx -W -j auto -b html docs docs/_build/html

Releasing
---------

Releases are cut from ``main``:

#. Move the ``Unreleased`` entries in ``CHANGELOG.md`` under a new version
   heading and record the version and date.
#. Bump ``version`` in ``pyproject.toml`` (semantic versioning).
#. Tag the commit (``vX.Y.Z``) and publish a GitHub Release. The
   ``publish-pypi`` workflow validates that the tag matches the project version
   and uploads the wheel and sdist to PyPI.

Keep ``CHANGELOG.md`` current as part of each change rather than at release
time: add a bullet under ``Unreleased`` in the same PR that makes the change.
