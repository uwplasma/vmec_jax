# Test Suite Map

The suite tests the `vmex.core` and `vmex.mirror` packages:

- `tests/`: one file per core concern — input parsing/round-trip,
  Fourier transforms, setup, geometry/fields, forces/residuals,
  preconditioner, solver end-to-end, multigrid interpolation and ladder,
  mgrid/coils, free boundary, wout goldens, parity breadth, implicit
  gradients, optimize, plotting/Boozer, CLI, asset fetcher, and packaging
  metadata.
- `tests/mirror/`: analytic, geometry/field, fixed/free-boundary, spline,
  implicit-derivative, hybrid, output, and exterior-vacuum coverage.
- `tests/conftest.py` resolves the VMEC2000 golden parity fixtures
  (`VMEX_GOLDEN_DIR` env var, `~/vmex_notes/golden`, or a one-time
  sha256-verified download of the `golden-v1` GitHub release).

Markers: `full` tests are skipped unless `RUN_FULL=1` is set.
`vmec2000_live` tests additionally require `--run-vmec2000` and accept
`--vmec2000-executable PATH`; they are never part of ordinary offline CI. The
live lane includes fixed-boundary current/Mercier profiles and a converged
LASYM free-boundary DIII-D regression with generated compact mgrid data and
all NESTOR WOUT potential/surface-field tables. The
root `tests/conftest.py` disables jit globally for speed; tests that exercise
the jit lane re-enable it explicitly.
