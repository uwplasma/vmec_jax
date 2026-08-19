Contributing
============

Migration status
----------------

The repository recently completed a clean-room rewrite: the production
implementation is :mod:`vmex.core` (~30 focused modules, one concern per
file — see :doc:`/explanation/architecture`), and the ``vmec`` CLI runs on it end to end.
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
   fixtures. Add its module to ``tests/manifest.json``; the collection gate
   requires one owner, primary class, expected duration, device, asset bundle,
   oracle, and CI lane.
3. Keep JAX gotchas in mind:

   - jitted functions should only take arrays / pytrees (static config stays
     hashable and out of traced signatures),
   - solver functions are pure ``state -> state`` — no hidden host state,
   - float64 is mandatory (enforced at solver import).

Development install and checks::

  pip install -e .[dev]
  ruff check .
  python tools/test_manifest.py check
  pytest -q

The workflows obtain their selectors from the manifest:

- ``CI`` is the stable pull-request gate. It runs fast API checks and
  representative fixed-boundary, free-boundary/NESTOR, mirror, device, and AD
  paths. Changed executable lines must be at least 95% covered.
- ``Nightly`` owns the complete integration/oracle matrix and aggregate
  package coverage. Its matrices leave hosted capacity for pull-request
  checks.
- ``Weekly high resolution`` owns campaigns that exceed the 150-minute cold
  nightly budget.
- ``Trusted GPU physics`` is an explicit self-hosted dispatch.

Use ``pytest --vmex-report=report.json`` to record the 50 slowest tests and all
skip reasons with the same metadata.

Reference assets
----------------

Keep input decks and analytic fixtures in git. Put generated WOUT, mgrid, and
benchmark archives in a GitHub Release and add their URL, byte size, SHA-256,
source, license, generator revision, and installed paths to
``assets/manifest.json``. Fetch repository fixtures with
``python tools/fetch_assets.py`` and VMEC2000 goldens with
``python tools/fetch_assets.py --bundle golden-v1``. CI rejects any tracked
file larger than 1 MiB.

**Pull-request lanes must not depend on a released bundle.** A release can be
deleted, and when ``assets-20260316-nc`` was, every pull request went red on
the download step while the lane it fed lost exactly one test. PR selectors run
on git-tracked decks, generated fixtures, and ``tests/golden_digests.json``;
only the nightly and weekly lanes fetch ``reference-nc``. Prefer a generated
fixture to a released one whenever the assertion is about solver behaviour
rather than a specific machine: ``tests/test_lasym_free_case.py`` (a converged
free-boundary case as 90 Chebyshev coefficients) and
``tests/test_qi_free_boundary_case.py`` (analytic modular coils tabulated with
``tabulate_cartesian_field``) are the two that PR lanes build on, and modules
that need neither declare ``"asset": "generated"`` in ``tests/manifest.json``.

Cut a new bundle with ``python tools/pack_reference_assets.py``, which packs
the git-ignored netCDF files under each bundle's roots into a byte-reproducible
tarball and prints the manifest fields. It applies two slimming rules, both
measured lossless: MAKEGRID's ``ar_``/``ap_``/``az_`` vector potential is
dropped (neither VMEX nor ``xvmec2000`` reads it back — half of
``mgrid_cth_like.nc``), and duplicate ``single_grid/`` copies are re-created on
extraction from the manifest's ``mirrors`` rule instead of shipped. Mirroring
is also what keeps the tracked ``mgrid_cth_like_lasym_small.nc`` and its
``single_grid/`` copy in step.

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

``Trusted GPU physics`` is a manual workflow because this is a public repository: pull
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

#. Bump ``version`` in ``pyproject.toml`` (semantic versioning).
#. Tag the commit (``vX.Y.Z``) and publish a GitHub Release. The
   ``publish-pypi`` workflow validates that the tag matches the project version
   and uploads the wheel and sdist to PyPI.

Release notes are written on the GitHub Release itself, summarising the
merged PRs since the previous tag; the repository does not keep a changelog
file.
