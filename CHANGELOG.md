# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once past 1.0.

## [Unreleased]

### Changed
- **QI-mirror hybrid: tangent-aligned legs at all four symmetry planes.**
  `splice_straight_legs` now inserts each straight mirror leg *along the local
  axis tangent* (rather than a shared transverse "bisector" that produced a
  ~36-degree corner), and cuts the nfp=2 QI axis at all **four** low-curvature
  symmetry planes (rather than two). Per-cut leg lengths are chosen so the
  inserted displacements cancel (splitting into two symmetry classes), and one
  half is reflected 180 degrees about the x axis, so the four-legged racetrack is
  exactly stellarator symmetric and every leg/return junction is
  tangent-continuous (residual break ~0.04 deg). `QIMirrorSplice` gains
  `leg_lengths` / `leg_directions` (per cut) and its `leg_windows` / `cut_*`
  fields are now length-`N`; `build_qi_mirror_hybrid` accepts N cuts.

## [0.3.0] — 2026-07-20

### Added
- **QI-mirror hybrid (Fourier vs B-spline).** `vmex.mirror.splice_straight_legs`
  cuts a closed magnetic axis at its curvature minima and inserts exactly-straight
  mirror legs (closing the loop to rounding); `build_qi_mirror_hybrid` fits the
  spliced axis into the closed-spline solve basis with a circular section and
  returns a solvable `StellaratorMirrorSetup`. The new example
  `examples/qi_mirror_hybrid_fourier_vs_bspline.py` cuts the nfp=2 QI axis at its
  low-curvature symmetry planes and compares the two representations: a global
  Fourier series rings at the straight↔curved seam and decays only ~1/N, while
  the local B-spline reproduces the straight mirror cell to machine precision
  (~1e-12) once each leg is backed by enough collinear controls. See
  `docs/mirror_geometry.rst`.
- **Toroidally rotating ellipse for the stellarator-mirror hybrid.** A new
  `section_turns` parameter on `build_stellarator_mirror_hybrid` /
  `stellarator_mirror_section_coefficients` turns the elliptical cross-section
  continuously around the closed circuit by that many full turns, superposed on
  the return-only 90-degree rotation, while the legs keep an exactly straight
  axis. Two turns lift the traced transform from the return-only `iota=0.085` to
  `iota=0.141` at `s=0.75`. The default `section_turns=0` reproduces the prior
  return-only geometry exactly. The rotating-elliptical-section hybrid stays a
  research candidate: the toroidal rotation passes the minor-radius bulk
  promotion gate but its device-normalized strong force still plateaus on the
  scoped near-axis representation defect.

### Changed
- **Axisymmetric free-boundary mirror validated through 50 % β** (was 25 %). A
  size-scaled Krylov span in the Newton-GMRES polish clears the fine-grid restart
  starvation, and a fine grid (`ns=13, nxi=25, elements=13, exterior_ntheta=24`)
  converges every β point from 0 through 50 % (≤ 44 Newton-GMRES iterations) with
  bulk minor-radius force `1.21e-4 → 2.41e-3`, far below the `0.05` promotion
  gate. The older per-grid device-length force figures are reframed as a
  coarser-grid legacy diagnostic. See `benchmarks/mirror_free_boundary_axisymmetric.json`
  (`fine_grid_promotion.fine_grid_50`).
- **CI timeout headroom** for the borderline parity goldens (25 → 35 min) and
  implicit-gradient (10 → 15 min) jobs, so slow shared runners no longer cancel
  otherwise-passing suites.

### Fixed
- **ESSOS `Coils.from_json`.** Current ESSOS renamed the coils-JSON loader to the
  `Coils.from_json` classmethod and removed the old `Coils_from_json` free
  function; the `--coils` CLI path and the two ESSOS coil examples now use it
  (with a `hasattr` fallback to the legacy name).
- **ESSOS `Coils.to_mgrid` guard.** The `--coils` coils→mgrid path now raises a
  clear `VmecInputError` (instead of an opaque `AttributeError`) on ESSOS builds
  that predate `Coils.to_mgrid`, and the matching test skips rather than fails.
- **`tools/fetch_assets.py` fixture download.** The fetch constants pointed at
  non-existent `vmex_*.tar.gz` release tarballs (the published bundles kept their
  pre-rename `vmec_jax_*.tar.gz` names), 404-ing every fixture download; repointed
  at the real filenames (SHA256 unchanged).
- **Stale lasym-rejection test + docstrings.** `test_..._rejects_lasym_decks`
  and two `optimize.py` docstrings claimed `jac="implicit"` requires
  `lasym = False`; the implicit lane has supported (FD-validated) lasym since the
  4-family boundary map + traceable `readin.f` delta rotation landed. Replaced the
  test with a fast 4-family boundary-map round-trip and corrected the docstrings.

## [0.2.0] — 2026-07-18

First release under the **VMEX** name (formerly `vmec-jax`). Highlights:
the `vmec_jax → vmex` rename, the `vmex.mirror` open/closed magnetic-mirror
equilibrium package, `vmex.parallel` concurrent ensembles, the traceable
`l_grad_b` objective and objectives showcase, the cold-start single-stage
plasma+coil benchmark, NESTOR free-boundary speedups (warm now faster than
VMEC2000 on every benchmark row), and typed errors through the callback
boundary. See the entries below.

### Changed
- **Renamed `vmec_jax` → VMEX.** The import package is now `vmex`, the PyPI
  distribution is `vmex`, and the primary CLI command is `vmex`. The `vmec`
  command is kept as an alias. Environment variables use the `VMEX_*` prefix;
  the compilation-cache knobs still accept their legacy `VMEC_JAX_*` names for
  one release. The persistent compilation cache moved to `~/.cache/vmex`.
  VMEC file-format conventions are unchanged: `wout_*.nc` / `boozmn_*.nc`
  outputs, wout variable names, and the `Vmec*` class names all stay the same.

### Added
- **`vmex.parallel`** — concurrent ensembles of independent equilibrium solves
  (`solve_ensemble`, `map_ensemble`). Thread-based, results bit-identical to
  serial; measured ~3.3× at 8 workers on a balanced ensemble. Multi-GPU is
  documented as a design sketch in `docs/parallelization.rst`.
- **Traceable `l_grad_b_state`** — an implicit-adjoint-compatible magnetic
  gradient scale-length objective (soft-min for optimization, hard-min for
  reporting); machine-precision parity with the wout-lane `l_grad_b`.
- **`examples/optimization/objectives_showcase.py`** — five one-objective
  refinement campaigns off the precise-QA deck (L∇B, magnetic well, iota,
  aspect ratio via the implicit adjoint; Mercier `DMerc` via finite
  differences), each holding quasisymmetry.
- **`examples/single_stage_vs_two_stage.py`** — cold-start single-stage vs
  two-stage plasma+coil benchmark, with the single-stage "polish" of the
  two-stage result, in vacuum and at finite β.
- **`ImplicitSolution.runtime`** — the solve's runtime rides on the result, so
  objectives no longer rebuild it per evaluation.
- A `vmec_jax` compatibility shim: `import vmec_jax` re-exports `vmex` (and its
  submodules) with a `DeprecationWarning`, for one release.

### Fixed
- Typed `VmecConvergenceError` / `VmecJacobianError` now propagate through the
  `jax.pure_callback` boundary instead of surfacing as an opaque multi-kilobyte
  `JaxRuntimeError`.
- NESTOR free-boundary iteration loop fused into jitted lanes (exact parity;
  ~6–24 % warm speedup). VMEX warm solves are now faster than VMEC2000 on every
  benchmark row.

### Removed
- Repository planning documents (`plan.md`, `plan_pre_vmex.md`, `notes_*.md`)
  moved to a private archive; they no longer ship with the package.

## [0.1.0]

Initial public release as `vmec-jax`: a clean-room, JAX-native reimplementation
of VMEC2000 with iteration-for-iteration parity on the benchmark suite,
implicit-differentiation gradients for fixed-boundary equilibria, a
differentiable virtual-casing free-boundary path, a Boozer transform, and the
`vmec` command-line interface.
