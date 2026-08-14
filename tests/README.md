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
- `tests/conftest.py` resolves the VMEC2000 golden parity fixtures from an
  explicit local override, `~/vmex_notes/golden`, or the verified user cache.
  Populate the cache with
  `python tools/fetch_assets.py --bundle golden-v1`.
- `tests/manifest.json` assigns every collected test module one owner,
  primary class, duration class, device, asset bundle, oracle, and CI lanes.
  Run `python tools/test_manifest.py check` after adding or moving a test.
  CI selects files from this manifest rather than maintaining path lists in
  workflow YAML.

Markers: `full` tests are skipped unless `RUN_FULL=1` is set. The `weekly`
marker identifies high-resolution campaigns that also stay out of the nightly
matrix. Pull requests run fast API tests plus representative fixed-boundary,
free-boundary/NESTOR, mirror, device, and AD selectors. Nightly runs the
complete integration/oracle matrix; weekly runs the high-mode free-boundary
ladder and exterior-mirror resolution study. PR lanes fetch only the 2 MB
`golden-v1` cache: they run on tracked decks, `asset: "generated"` fixtures,
and `tests/golden_digests.json`, so a deleted release cannot redden a pull
request. Modules declaring `asset: "reference-nc"` belong to nightly/weekly.
`vmec2000_live` tests additionally require `--run-vmec2000` and accept
`--vmec2000-executable PATH`; they are never part of ordinary offline CI. The
live lane includes fixed-boundary current/Mercier profiles and a converged
LASYM free-boundary DIII-D regression with generated compact mgrid data and
all NESTOR WOUT potential/surface-field tables. The
root `tests/conftest.py` disables jit globally for speed; tests that exercise
the jit lane re-enable it explicitly.

Pass `--vmex-report=PATH` to pytest to save the 50 slowest collected tests and
every skip reason with inherited manifest metadata.
