# Test Suite Map

The suite tests the `vmex.core` package (the only code in the repo):

- `tests/`: one file per core concern — input parsing/round-trip,
  Fourier transforms, setup, geometry/fields, forces/residuals,
  preconditioner, solver end-to-end, multigrid interpolation and ladder,
  mgrid/coils, free boundary, wout goldens, parity breadth, implicit
  gradients, optimize, plotting/Boozer, CLI, asset fetcher, and packaging
  metadata.
- `tests/conftest.py` resolves the VMEC2000 golden parity fixtures
  (`VMEX_GOLDEN_DIR` env var, `~/vmex_notes/golden`, or a one-time
  sha256-verified download of the `golden-v1` GitHub release).

Markers: `full` tests are skipped unless `RUN_FULL=1` is set.  The root
`tests/conftest.py` disables jit globally for speed; tests that exercise the
jit lane re-enable it explicitly.
