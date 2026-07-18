# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once past 1.0.

## [Unreleased]

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
