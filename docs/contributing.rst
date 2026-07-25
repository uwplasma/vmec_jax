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

An installed VMEC2000 can be exercised live, outside ordinary offline CI::

  pytest -q tests/test_vmec2000_live.py --run-vmec2000 \
    --vmec2000-executable /path/to/xvmec2000

The test uses isolated output directories and compares WOUTs produced during
that invocation. It covers finite-beta current/Mercier profiles and a
converged, asymmetrically forced LASYM free-boundary case including NESTOR
potential and surface-field tables. Omitting ``--run-vmec2000`` skips it.

GPU CI
------

``GPU CI`` is a manual workflow because this is a public repository: pull
requests from forks are never run automatically on persistent self-hosted
hardware.  Its runner must carry the labels ``self-hosted``, ``linux``,
``x64``, and ``gpu``, provide an NVIDIA driver 580 or newer for CUDA 13, and
must not define ``JAX_PLATFORMS`` or ``JAX_PLATFORM_NAME``.  The workflow
installs the official ``jax[cuda13]`` distribution, verifies that JAX selects
the GPU by ordinary hardware discovery, then runs focused placement checks and
the quick nonzero-shear CPU/GPU parity audit for MHD energy, magnetic well,
DMerc, ``jdotb``, Glasser ``D_R``, quasisymmetry, and quasi-isodynamic
gradients. Timing is recorded in the uploaded ``device-parity`` artifact but
is not a pass/fail gate. The focused suite also compares the LASYM ``jdotb``
implicit Jacobian on CPU and GPU. A missing or misconfigured accelerator is
a failure, not a skipped green GPU job.

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
