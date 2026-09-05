# VMEX 0.8+ research-grade implementation plan

**Audit snapshot:** 2026-08-30  
**Primary repository:** https://github.com/uwplasma/vmex  
**Audited VMEX main:** `bc2f89c9a3a96e0af759502ca59bbb2f020d599b` (`0.7.1`)  
**Open VMEX integration PR:** https://github.com/uwplasma/vmex/pull/192 (`37a8ee39b1d985259b8afb8ec87a2e92948a1bd4`)  
**Required SOLVAX baseline for PR #192:** `0.20.0`  
**Companion repositories:**

- https://github.com/uwplasma/SOLVAX
- https://github.com/uwplasma/booz_xform_jax
- https://github.com/uwplasma/virtual_casing_jax
- https://github.com/uwplasma/ESSOS
- https://github.com/uwplasma/GKX
- https://github.com/uwplasma/DKX
- https://github.com/PlasmaControl/DESC
- https://github.com/proximafusion/vmecpp
- a local VMEC2000/STELLOPT checkout

This file replaces the previous root `plan.md`. The previous file was a valuable historical ledger, but it mixed completed release work, stale worktree paths, old benchmark snapshots, and future research tasks. This replacement preserves the unfinished requirements while using merged pull requests, benchmark JSON files, and Git history as the permanent record of completed work.

---

## 1. Mission

Turn VMEX into one coherent, high-accuracy, differentiable equilibrium and optimization system for:

1. closed toroidal equilibria;
2. fixed- and free-boundary mirrors;
3. periodic stellarator-mirror hybrids;
4. downstream coil, orbit, neoclassical, and gyrokinetic calculations;
5. fast repeated solves and gradients in design campaigns.

The immediate engineering goal is not to replace VMEC-compatible numerics. It is to keep the existing VMEX solve as the robust branch finder and compatibility layer, then attach a certified, high-order force-balance correction and native continuous equilibrium representation. The mirror and hybrid lanes should reuse the same numerical ideas where their topology permits, without forcing toroidal assumptions onto open systems.

The final system must be:

- **accurate:** independently certified strong force balance, boundary conditions, regularity, and resolution convergence;
- **fast:** measured cold, warm, persistent-cache, optimization, and gradient performance;
- **memory bounded:** no hidden all-surfaces/all-modes tensors when streaming or local support is sufficient;
- **differentiable:** implicit derivatives of converged equations, not taped nonlinear iterations;
- **simple to use:** one input-file option and one Python keyword for polishing;
- **simple to maintain:** explicit ownership between VMEX, SOLVAX, BOOZ_XFORM_JAX, ESSOS, VIRTUAL_CASING_JAX, GKX, and DKX;
- **honest:** no performance or accuracy claim without checked-in raw evidence and a reproducible command.

---

## 2. Definition of done

The program is complete only when all of the following are true.

### 2.1 Toroidal equilibrium and polishing

- A legacy-compatible VMEC `&INDATA` file can request polishing through a VMEX comment directive that VMEC2000 ignores.
- Structured JSON can request the same through a reserved `_vmex` section.
- Python users can write either:

```python
result = vmex.solve(inp, polish="auto")
```

or:

```python
result = vmex.solve_file("input.case", polish="auto")
```

- The default remains unchanged: no polishing unless requested.
- A successful polished result carries:
  - the original VMEC-compatible state;
  - the native high-order equilibrium;
  - a strong-force certificate on an independent grid;
  - convergence, admissibility, derivative, timing, and memory reports;
  - a VMEC-compatible WOUT sampled from the polished state;
  - an optional native VMEX NetCDF representation that preserves the high-order coefficients.
- Failure is typed and controlled by `fail="error" | "fallback" | "warn"`; it is never silently presented as a polished solution.

### 2.2 Performance

- Every flagship workflow has separated measurements for tracing, lowering, compilation, execution, host-device transfer, output/postprocessing, and peak memory.
- Cold-process, persistent-cache reload, and warm same-process numbers are never mixed.
- Repeated optimization evaluations at unchanged static shapes do not recompile unexpectedly.
- The profiling suite demonstrates which changes improve:
  - one cold fixed-boundary solve;
  - one cold multigrid solve;
  - polished solve;
  - scalar value and gradient;
  - vector residual and Jacobian;
  - single-stage plasma-and-coil optimization;
  - free-boundary solves and gradients;
  - LASYM solves and optimization;
  - mirror and hybrid solves;
  - BOOZ_XFORM_JAX.
- Any Pallas/custom-kernel work is justified by an XProf or Nsight trace showing an XLA fusion or memory-layout limitation.

### 2.3 Mirrors

- The fixed-cut open-mirror model has a documented variational problem and boundary-condition contract.
- Isotropic fixed-boundary mirrors pass analytic, manufactured, refinement, weak/strong residual, and independent field tests.
- An anisotropic model solves

```text
J x B = div(P),
P = p_perp I + (p_parallel - p_perp) b b,
```

with an equilibrium-consistent pressure closure rather than arbitrary independent `p_parallel` and `p_perp` arrays.
- The isotropic limit recovers the existing mirror solver.
- Axisymmetric anisotropic free-boundary equilibria driven by circular ESSOS coils are validated against analytic/paraxial limits and literature-grade reference cases.
- A nonaxisymmetric fixed-boundary mirror is optimized and independently certified.
- A nonaxisymmetric free-boundary mirror is solved with a full 3-D exterior response, not an axisymmetric surrogate.

### 2.4 Stellarator-mirror hybrids

- The periodic closed hybrid has a converged equilibrium, high-order representation, force certificate, field-line closure certificate, and implicit derivatives.
- A reproducible optimization varies at least:
  - straight mirror length;
  - mirror ratio;
  - return radius/shape;
  - cross-section rotation;
  - QI/omnigenity controls;
  - iota or current controls;
  - coil feasibility controls.
- The same in-memory result feeds:
  - ESSOS coil construction and alpha-loss calculations;
  - GKX local gyrokinetic geometry and turbulence;
  - DKX Boozer geometry and neoclassical transport;
  - BOOZ_XFORM_JAX diagnostics where closed-surface Boozer coordinates are meaningful.

### 2.5 Confinement objectives

- The summary pressure/ripple panel also shows `Gamma_c` on the same confinement axis, with caching and a bounded diagnostic resolution.
- Hard-topology `Gamma_c` remains a validation metric until its derivative is proven convergent.
- A tracked-well and/or smooth surrogate provides stable optimization derivatives with explicit branch-event diagnostics.
- One concise example optimizes a weighted combination of:
  - epsilon effective;
  - a derivative-safe `Gamma_c` objective;
  - maximum-J;
  - force-balance and geometric constraints.
- The final configuration is validated with hard `Gamma_c`, independent effective-ripple/neoclassical calculations, and ESSOS orbit losses.

### 2.6 LASYM

- `LASYM = TRUE` has measured value, gradient, Jacobian, and optimization performance.
- All asymmetric Fourier families are retained through equilibrium, WOUT, Boozer, stability, field, and downstream adapters.
- A flagship LASYM optimization produces a physically distinct asymmetric equilibrium, not numerical noise or a gauge change.
- The asymmetric result improves a declared physical objective under the same constraints as the symmetric reference and passes independent VMEC2000/VMEX and force-balance checks.

### 2.7 Documentation and code quality

- Changed/new code has at least 95% line coverage and all physics branches have direct tests.
- Repository coverage does not decrease.
- Examples follow the compact SIMSOPT pattern described below.
- Every performance and physics figure is regenerated from checked-in scripts and machine-readable results.
- No abandoned experimental implementation survives beside the promoted path.

---

## 3. Execution contract for the local agent

### 3.1 First actions in every repository

Before editing:

1. read `AGENTS.md`, `CONTRIBUTING`, `pyproject.toml`, CI workflows, and package API tests;
2. record `git status`, branch, commit, Python/JAX/jaxlib versions, platform, device, and optional dependencies;
3. update the audited hashes in a short work log if repositories have moved;
4. run the smallest relevant test set before changing code;
5. inspect open PRs so work is not duplicated.

Never overwrite user-owned untracked files or benchmark assets. Never infer completion from a local branch when the owning PR is open.

### 3.2 Pull-request discipline

- One coherent capability per PR.
- Avoid mega-PRs after PR #192 is merged.
- PR descriptions state:
  - exact problem;
  - equations/algorithm;
  - public API impact;
  - tests and oracles;
  - cold/warm/memory evidence;
  - known limits.
- Do not add a generic abstraction until two real call sites need it.
- Feature PRs may add lines. Refactor PRs should have non-positive net source LOC unless a measured simplification requires a small adapter.
- No performance change merges without before/after raw JSON on the same host and environment.
- No numerical tolerance is loosened solely to make CI pass.

### 3.3 Code style

- Prefer pure functions, frozen dataclasses, pytrees, and explicit static configuration.
- Keep public docstrings short but precise about units, coordinate conventions, derivative semantics, and failure modes.
- Comments explain non-obvious physics or numerical invariants, not the syntax of the next line.
- Do not create many one-function files. Split a large module only when ownership becomes clearer and imports become simpler.
- Keep optional dependencies behind narrow adapters.
- Avoid `__main__`, argument parsing, and hidden global state inside examples.
- Keep shape-changing options static and visible.
- Never use environment variables as the only API for numerical or memory behavior.

### 3.4 Evidence hierarchy

Use, in descending order:

1. analytic/manufactured solution;
2. independent derivation or implementation;
3. VMEC2000, VMEC++, DESC, ANIMEC, NEO, ESSOS, GKX, DKX, or VIRTUAL_CASING_JAX comparison;
4. resolution and tolerance convergence;
5. internal consistency.

A test that compares two functions sharing the same kernel is not an independent physics oracle.

---

## 4. Audited repository state

### 4.1 What is already complete on VMEX main

Do not reimplement these items.

- VMEC-compatible fixed- and free-boundary equilibrium core.
- VMEC2000-style 1-D radial tridiagonal preconditioner and lambda preconditioner.
- Optional matrix-free 2-D/JVP-GMRES Newton acceleration.
- Hot restart, multigrid, direct ESSOS-coil external fields, and typed failure handling.
- Implicit tangent/adjoint infrastructure through SOLVAX.
- Block-tridiagonal raw-force Jacobian factorization and multi-RHS response machinery.
- Shared axis-regular B-spline basis of degrees 3, 5, and 7 (`vmex/core/radial_basis.py`).
- High-order strong-force evaluation and independent certification (`vmex/core/strong_force.py`).
- Low-order strong-force preconditioner (`vmex/core/polish.py` and associated tests/benchmarks).
- WOUT field-line geometry adapter.
- Periodic stellarator-mirror gyrokinetic geometry (`vmex/mirror/turbulence.py`, merged PR #194).
- Isotropic mirror energy, weak residual, independent strong pointwise force, B-spline fixed-boundary solves, periodic hybrid geometry, and an axisymmetric free-boundary lane.
- Gamma-c value implementation with LASYM support and large warm-kernel speedup.
- LASYM Mercier/Glasser, bootstrap, Boozer, stability, and many optimization fixes.
- In-process BOOZ_XFORM_JAX and NEO_JAX effective-ripple diagnostics in the summary plot.
- Existing performance tooling for hot paths, GPU/device parity, startup, cache reload, radial basis, force, and preconditioners.

### 4.2 Open PR #192: treat as integration, not a new design

PR #192 already implements the production direction proposed by the former plan:

- axis-regular high-order reconstruction;
- overintegrated physical strong-force residual;
- overdetermined collocation least-squares polishing;
- SOLVAX matrix-free damped Gauss-Newton;
- normal-operator preconditioning;
- exact tangent, adjoint, and custom-VJP stationarity derivatives;
- native polished field/surface/Boozer/virtual-casing/ESSOS adapters;
- cross-code force-balance comparisons and README assets.

The branch reports, subject to reproduction during review:

| Case / quantity | Reported result on PR #192 |
|---|---:|
| Solovev normalized volume force L2, legacy VMEX | 0.122381 |
| Solovev normalized volume force L2, VMEC2000 | 0.122399 |
| Solovev normalized volume force L2, VMEC++ | 0.122399 |
| Solovev normalized volume force L2, DESC | 0.014405 |
| Solovev normalized volume force L2, polished VMEX | 0.002759 |
| Polished radial-refinement difference | 1.55e-4 |
| Polished cold solve / end-to-end | 13.32 s / 57.64 s |
| Finite-beta QA normalized L2, legacy VMEX / DESC | 0.873927 / 0.964975 |
| Finite-beta cold VMEX / DESC | 14.48 s / 157.66-186.32 s |
| Tangent-adjoint duality mismatch | 1.90e-10 |
| Re-polished centered-FD gradient relative difference | 5.11e-5 |
| Changed-line coverage | 95.7% |

These values must be reproduced from the branch scripts before they appear in a release claim. The finite-beta comparison is case- and normalization-specific and must not be generalized without a broader suite.

### 4.3 Lessons from the closed polishing experiments

Preserve these conclusions.

- Solving a square retained-mode strong-force root can drive projected equations down while moving force into unresolved harmonics.
- The promoted formulation is an overdetermined physical collocation residual with an independent validation grid.
- A global dense SVD coordinate chart is too expensive; use the structural local chart.
- Continuation can encounter folds; identity-preconditioned JFNK alone is not sufficient.
- Volume weighting alone does not guarantee a uniformly accurate solution.
- A derivative can be accurate for the discrete stationarity system while the physical certificate is inadequate; both are required.
- Warm-JIT timings cannot be used to claim cold end-to-end superiority.

### 4.4 Gamma-c state

The current `Gamma_c` implementation is a useful value diagnostic, but merged PR #155 documented nonconvergent boundary derivatives caused by hard well detection, hard topology changes, and branch-sensitive extrema. Reported examples include sign-changing radial-resolution gradients. Therefore:

- do not expose the current hard `Gamma_c` as a production gradient objective;
- do not weaken the warning in the documentation;
- build tracked-well or smooth derivative semantics before adding the requested optimization example.

### 4.5 Mirror state

The mirror lane already distinguishes three topologies that must remain separate:

1. **Open fixed-cut mirror:** a finite axial domain with two prescribed, flux-carrying cuts. Field lines cross the cuts. Sheaths, sources, and end-loss kinetics are not part of the equilibrium model.
2. **Free lateral boundary with open cuts:** the lateral surface is a plasma-vacuum interface. The axial caps are a mathematical closure of the Green surface, not plasma-vacuum interfaces.
3. **Periodic stellarator-mirror hybrid:** no open cuts; the axis and surfaces close periodically.

The code already has a general 3-D closed sidewall-plus-cap boundary-integral geometry in `vmex/mirror/exterior.py`. The current coupled free-boundary solver is restricted by an axisymmetric gate. Nonaxisymmetric free-boundary work should extend the existing exterior response and remove this gate after verification; it should not create a second exterior solver.

### 4.6 Boozer ownership state

- `vmex/core/qi.py` is not the main duplication; it consumes the transform in `vmex/core/omnigenity.py`.
- `vmex/core/omnigenity.py` contains a lightweight traceable Boozer implementation for the symmetric path and delegates LASYM to BOOZ_XFORM_JAX.
- `vmex/core/boozer_tables.py` creates pure-JAX WOUT-convention input spectra.
- BOOZ_XFORM_JAX contains the validated general transform.

The ownership boundary is currently blurred. It must be resolved by parity and performance measurements, not by deleting the smaller implementation on sight.

### 4.7 Large-module maintenance targets

Approximate current sizes identify review targets, not automatic split points:

- `vmex/core/optimize.py`: about 155 kB;
- `vmex/core/implicit.py`: about 122 kB;
- `vmex/core/freeboundary.py`: about 104 kB;
- `vmex/core/plotting.py`: about 82 kB;
- `vmex/mirror/splines.py`: about 62 kB.

First create an ownership/call graph and identify duplicate policy, data conversion, and solver orchestration. Split only where a stable internal interface emerges.

### 4.8 Audited PR and commit disposition

The local agent should recheck status before acting, but use this table as the 2026-08-30 baseline.

| PR / commit group | Audited contribution | Disposition in this plan |
|---|---|---|
| #192 | Integrated high-order force polishing, collocation Gauss-Newton, stationarity derivatives, native downstream adapters, cross-code evidence | Reproduce, review, clean, and merge in Phase 0; do not reimplement |
| #194 | Closed periodic stellarator-mirror GK geometry, Clebsch label, dual metrics/drifts, equal-arc remap | Foundation for hybrid/GKX work |
| #193 and #160 | 0.7.1 and 0.7.0 release work | Preserve packaging/release contracts |
| #190 | WOUT field-line geometry adapter and LASYM/symmetric oracle | Reuse in downstream parity tests |
| #177 | QA startup/memory reductions, analytic fixed-boundary mask, chunked raw-block VJPs, one scalar reverse solve | Baseline for Phase 3; do not regress |
| #173 | Cold compile, persistent-cache reload, and warm benchmark infrastructure | Consolidate into Phase 2 harness |
| #163 | Topology-independent B-splines, degrees 3/5/7, derivatives, refinement, regularity | Shared foundation for polish and mirrors |
| #164 | High-order state and independent strong-force oracle | Promoted by #192; retain independent certificate |
| #165 | Low-order strong-force preconditioner | Benchmark/reuse in #192 and Phase 3 |
| #166 | Square strong-force root/homotopy | Diagnostic lesson; not production because unresolved force can escape retained modes |
| #167-#191 experiments | Collocation charts, SVD experiments, continuation folds, preconditioner variants, derivative paths | Preserve conclusions/tests; remove abandoned code from promoted branch |
| #155 | Gamma-c fixes, large warm speedup, LASYM support, documented derivative nonconvergence | Keep hard value; redesign derivative in Phase 6 |
| #152 | LASYM ballooning and external COBRAVMEC oracle | Retain oracle and shared closures |
| #151 | LASYM bootstrap/filtering fixes and major vectorized filtering speedup | Baseline; investigate remaining performance gap |
| #127 and #126 | Full-Nyquist LASYM Boozer and differentiable LASYM normalization/optimization fixes | Foundation for Phase 7 and Boozer parity |
| #118 | LASYM Mercier/Glasser validation against VMEC2000 | Preserve as independent stability gate |
| #154 | Free-boundary adjoint factor certificates and coupled-root reproducibility findings | Foundation for future polished/coupled derivatives |
| #146 | ESSOS seams, direct-coil mgrid, batched coil-field acceleration, fixed/free workflow | Reuse; eliminate file-backed gradient breaks |
| #137 and #138 | Traceable field tables, alpha tracing, CLI/ESSOS integration | Reuse, but keep derivative/topology capability explicit |
| #149 | GKX JAX floor | Coordinate with GKX PR #86 and VMEX #194 |
| #136 | QI/bootstrap optimization | Reuse objective composition and benchmark style |
| #115, #114, #107, #104 | Optimizer-neutral API/examples and stale-gradient corrections | Do not add another optimizer framework |
| #111 | Hot restart | Reuse in scans, continuation, and performance matrix |
| #110 | In-process Boozer and 3x3 summary diagnostics | Extend rather than replace in Phase 5 |
| #109 | GPU decision sweep | Preserve device-policy evidence |
| #101 and #99 | Free-boundary geometry reuse and sequential multigrid compilation/memory policy | Baseline for Phase 3; overlap remains opt-in |
| #150 | Correction of overstated performance claims | Apply the same evidence discipline to every new claim |
| #153 and #172 | CI activation and WSL2/CUDA diagnosis | Preserve platform checks and typed diagnostics |
| #125 | Former root plan | Superseded by this dated implementation plan |

Commit-level review also confirmed that the latest main contains the radial basis, strong-force oracle, polishing preconditioner, mirror gyrokinetic geometry, current performance benchmarks, and extensive capability tests. Before beginning a phase, use `git log --first-parent`, open/closed PR search, and changed-file inspection to update this disposition rather than trusting issue titles alone.

---

## 5. Non-negotiable architectural decisions

### 5.1 Polishing is a nearby branch solve, not a global design optimization

A general nonlinear MHD equilibrium does not come with a practical global-minimum guarantee. Do not promise one. The production algorithm should instead be deterministic and branch preserving:

1. converge the ordinary VMEX equilibrium;
2. construct the high-order axis-regular state;
3. solve the physical collocation least-squares system with damped Gauss-Newton;
4. use residual/load continuation and pseudo-transient globalization when needed;
5. use pseudo-arclength only when a continuation fold is detected;
6. certify the result on an independent grid.

Near a regular solution, Gauss-Newton/Newton gives fast local convergence. Farther away, the existing VMEX solve and continuation are the globalization mechanisms. There should be one promoted algorithm and one automatic policy, not a public menu of unrelated optimizers.

### 5.2 Keep VMEC compatibility and high-order physics as two representations

- `SpectralState` remains the VMEC-compatible state and restart/output contract.
- `NativeEquilibrium` is the continuous high-order representation used for force certification, accurate derivatives, and downstream physics.
- Conversions are explicit and tested in both directions.
- WOUT is a compatibility export, not the canonical differentiable in-memory object.

### 5.3 Separate physical convergence from algebraic convergence

Every solve report must distinguish:

- legacy `FSQR/FSQZ/FSQL`;
- least-squares optimality `||J^T r||`;
- strong-force residual on the solve grid;
- strong-force residual on an independent validation grid;
- refinement difference;
- Jacobian/nestedness/admissibility;
- linear-solve true residual;
- derivative certificate.

No one number substitutes for the others.

### 5.4 SOLVAX owns generic algebra; VMEX owns equilibrium physics

SOLVAX may own:

- matrix-free damped Gauss-Newton;
- GMRES/GCROT/PCG/LSMR-like Krylov methods;
- pseudo-transient continuation;
- continuation and pseudo-arclength infrastructure;
- structured direct solves and preconditioners;
- implicit differentiation wrappers;
- compile/work diagnostics that are physics agnostic.

VMEX owns:

- state charts and constraints;
- equilibrium residuals;
- force normalizations;
- admissibility;
- boundary conditions;
- physics-based preconditioner construction;
- native equilibrium and output semantics.

Do not put VMEX geometry or pressure assumptions into SOLVAX.

### 5.5 Consumer-side adapters are preferred

The producer exposes a small stable protocol and arrays. The consumer owns semantic conversion when possible:

- BOOZ_XFORM_JAX owns the Boozer transform;
- ESSOS owns coils and orbit integration;
- VIRTUAL_CASING_JAX owns virtual-casing quadrature;
- GKX owns gyrokinetic normalization and local simulation input;
- DKX owns neoclassical geometry normalization.

VMEX may provide convenience adapters, but it should not duplicate whole downstream codes.

---

## 6. Target public API

### 6.1 Input-file directive

Use a VMEX comment directive outside or inside `&INDATA`:

```fortran
!@VMEX POLISH = AUTO
!@VMEX POLISH_TOL = 1.0E-8
!@VMEX POLISH_FAIL = ERROR
!@VMEX POLISH_DEGREE = 5
&INDATA
  ... ordinary VMEC2000 input ...
/
```

VMEC2000 treats these lines as comments and solves the ordinary input. VMEX parses them before stripping comments. Do **not** add an unknown `LPOLISH` variable to `&INDATA`; legacy namelist readers may reject it.

Structured JSON uses:

```json
{
  "mpol": 8,
  "ntor": 6,
  "_vmex": {
    "polish": "auto",
    "polish_tol": 1e-8,
    "polish_fail": "error",
    "polish_degree": 5
  }
}
```

The physics schema remains VMEC++ compatible after `_vmex` is removed. Unknown `_vmex` keys fail explicitly.

### 6.2 Python API

Keep the existing PR #192 keyword and add one file-oriented entry point:

```python
import vmex

inp = vmex.VmecInput.from_file("input.case")
result = vmex.solve(inp, polish="auto")

result2 = vmex.solve_file("input.case", polish="auto")
```

Recommended signatures:

```python
def solve(
    inp,
    *,
    polish: bool | str = False,
    polish_config: PolishConfig | None = None,
    **existing_kwargs,
) -> SolveResult: ...


def solve_file(
    path,
    *,
    polish: bool | str | None = None,
    polish_config: PolishConfig | None = None,
    write_wout: bool = True,
    write_native: bool | str = "auto",
    **solve_kwargs,
) -> SolveResult: ...
```

`None` means use the file directive. Explicit Python arguments override the file. CLI arguments override both.

### 6.3 Result contract

A polished `SolveResult` should expose:

```python
result.state                 # VMEC-compatible state
result.native_equilibrium    # high-order continuous state or None
result.strong_force          # independent certificate or None
result.polish_report         # primal/linear/refinement/timing report or None
result.polish_context        # reusable static plan/cache object or None
result.wout                  # optional in-memory WoutData
```

The result must also answer:

```python
result.is_polished
result.require_polished()
result.sample_flux_surface(...)
result.boozer_inputs(...)
result.boundary_surface(...)
result.magnetic_field(...)
```

Convenience methods delegate to the native equilibrium when present and to the legacy adapter otherwise.

### 6.4 Precedence

Use exactly:

```text
CLI option > explicit Python keyword > file directive > package default.
```

Record the resolved source in the run report.

---

## 7. Phase 0 - Review, reproduce, and merge PR #192

**Purpose:** make the existing polishing implementation the trusted baseline before extending its API or performance.

### 7.1 Rebase and scope audit

- Rebase or merge current main into PR #192 without squashing away the experimental history until the final promoted code is reviewed.
- Confirm no completed main functionality was reintroduced under a second name.
- Verify that obsolete experimental modules, paths, and docs are removed from the final diff.
- Produce a compact ownership map for:
  - `radial_basis.py`;
  - `strong_force.py`;
  - `polish.py`;
  - `polish_driver.py`;
  - polished implicit differentiation;
  - native field/surface/VC/Boozer adapters.
- Ensure only the overdetermined collocation Gauss-Newton path is presented as production. If the square continuation path remains, mark it diagnostic/private and keep it out of the top-level API.

### 7.2 Reproduce every headline result

Run the branch-provided commands in fresh processes and store:

- exact input hashes;
- repository hashes;
- dependency lock/version output;
- host and device metadata;
- precision and XLA flags;
- compile/cache state;
- raw per-stage timings;
- peak RSS/device memory;
- residual normalizations and grids.

Reproduce at minimum:

1. analytic Solovev comparison;
2. finite-beta QA comparison;
3. radial-refinement certificate;
4. tangent-adjoint duality;
5. Taylor test;
6. explicit adjoint versus custom VJP;
7. re-polished centered finite difference;
8. downstream polished field, Boozer, virtual-casing, and ESSOS adapters.

### 7.3 Cross-code fairness audit

For VMEC2000, VMEC++, DESC, legacy VMEX, and polished VMEX:

- use the same physical boundary, pressure/current/iota profiles, flux, and symmetry;
- state every representation conversion;
- separate solve resolution from evaluation resolution;
- use one independent strong-force oracle where possible;
- evaluate all codes on the same off-grid physical points;
- exclude no axis/edge region without showing both the full and restricted metric;
- report cold end-to-end, cold solver-only, persistent-cache reload, and warm execution separately;
- do not use the polished VMEX oracle to evaluate only VMEX while using DESC's own objective for DESC.

### 7.4 Merge gate

PR #192 may merge when:

- all required CI jobs are green;
- changed-line coverage is at least 95%;
- no derivative test relies only on another code path sharing the same VJP;
- the independent validation residual decreases under at least one radial and one angular refinement;
- no headline claim is broader than the measured cases;
- WOUT/native representation semantics are documented;
- `polish=False` is bitwise or tightly numerically compatible with main for the existing regression suite;
- the PR no longer contains stale plan text or duplicated experiments.

**Deliverable:** merged, reviewed polishing core with a reproducible evidence directory.

---

## 8. Phase 1 - Integrate polishing into input files, CLI, multigrid, and simple Python usage

**Depends on:** Phase 0.

### 8.1 Parse VMEX run options without changing the VMEC physics schema

Add one small module, preferably `vmex/core/run_options.py`, containing:

```python
@dataclass(frozen=True)
class RunOptions:
    polish: bool | str = False
    polish_tol: float | None = None
    polish_fail: str = "error"
    polish_degree: int | None = None
    write_native: bool | str = "auto"
```

Responsibilities:

- parse `!@VMEX KEY = VALUE` directives from raw INDATA text;
- parse and validate `_vmex` from JSON;
- return the cleaned physics text/data and run options;
- reject duplicate conflicting directives;
- reject unknown keys with a value-free typed error;
- serialize directives when VMEX writes an input file;
- preserve all ordinary Fortran comment and quote semantics.

Do not add these fields to `VmecInput`; they control execution, not equilibrium physics.

### 8.2 Preserve `VmecInput.from_file`

`VmecInput.from_file(path)` should continue to return only a `VmecInput`. It should accept a reserved `_vmex` block by removing it before ordinary JSON schema validation.

Add:

```python
read_input_request(path) -> InputRequest
```

where:

```python
@dataclass(frozen=True)
class InputRequest:
    input: VmecInput
    options: RunOptions
    source: Path
```

The CLI and `solve_file` use `InputRequest`; existing code using `VmecInput.from_file` remains valid.

### 8.3 Add CLI controls

Add minimal controls:

```text
--polish[=auto|true|false]
--no-polish
--polish-tol VALUE
--polish-fail error|fallback|warn
--polish-degree 3|5|7
--native-output auto|yes|no|PATH
```

Do not expose every `PolishConfig` field on the CLI. Advanced settings remain Python-only or in `_vmex` with a documented expert section.

### 8.4 Integrate at the correct point in multigrid

- Run polishing once, after the finest fixed-boundary stage has converged.
- Never polish every multigrid rung.
- Do not run fixed-boundary polishing after a free-boundary solve until the coupled free-boundary polished residual is implemented; reject or explicitly fall back.
- Reuse the finest runtime, mode tables, static basis plan, and low-order factorization when valid.
- Preserve the ordinary VMEX state as a fallback checkpoint.
- If `polish="auto"`, activate only when:
  - the legacy solution converged;
  - the problem is in the supported topology/physics set;
  - the estimated strong-force error exceeds the requested threshold;
  - sufficient memory is available for the selected chart and collocation plan.
- Record why auto did or did not activate.

### 8.5 Output semantics

On successful polishing:

1. sample the native equilibrium onto standard VMEC full/half grids;
2. rebuild all WOUT quantities consistently from the sampled polished state;
3. recompute legacy `FSQR/FSQZ/FSQL` on that sampled state;
4. add nonstandard NetCDF attributes or a small extension group:
   - `vmex_polished`;
   - native degree and spans;
   - solve-grid and validation-grid strong residuals;
   - refinement difference;
   - polish iterations and reason;
   - source legacy residual;
   - native-file reference if written;
5. keep standard WOUT variable names and shapes unchanged.

Write `equilibrium_<case>.vmex.nc` when `write_native=True` or auto policy determines that downstream high-order use is expected. The native file stores:

- knots/basis degree;
- free coefficient chart;
- axis/boundary lifts;
- profiles and normalization;
- symmetry and coordinate conventions;
- force certificate;
- provenance hashes.

### 8.6 Simple documentation examples

Input example:

```fortran
!@VMEX POLISH = AUTO
&INDATA
  ...
/
```

Python example:

```python
from pathlib import Path
import vmex

input_path = Path("examples/data/input.solovev")
result = vmex.solve_file(input_path, polish="auto")
result.require_polished()

print(result.strong_force.volume_l2)
print(result.polish_report.total_seconds)
```

### 8.7 Tests

- directive parser around quoted `!`, repeated assignments, whitespace, case, and invalid values;
- JSON `_vmex` round trip;
- VMEC2000 executable accepts and runs a directive-bearing INDATA file unchanged;
- `VmecInput.from_file` ignores execution metadata but preserves all physics;
- precedence tests;
- CLI/Python equivalence;
- no-polish regression parity;
- fixed-boundary polished WOUT can be read by VMEC-compatible downstream readers;
- free-boundary unsupported request fails clearly;
- fallback/error/warn semantics;
- native NetCDF round trip and derivative consistency.

**Acceptance:** one documented input runs in both VMEC2000 and VMEX; only VMEX activates polishing.

### 8.8 Phase 1 follow-ups (user feedback, recorded 2026-09-02)

Field reports from a W7-X polish run; each item stays here until closed by a
merged PR:

1. Polish observability: the polish phase printed nothing after the legacy
   iteration table — no banner, no compile attribution, no per-iteration
   rows, no certificate verdict — so a run that was dying could not be told
   from one that was converging. Fix in flight on
   `feat/polish-verbosity-and-knobs`: CLI-only `BEGIN FORCE POLISHING`
   banner with the resolved config, ` compiling ... polish executable`
   notice, one row per Gauss-Newton step from the SOLVAX history, and a
   certificate summary that names any failed check.
2. Input-file control: `AUTO` was the only documented directive value and
   the deck could not set the solver knobs. Same branch adds
   `POLISH_MAX_ITER` and `POLISH_SPANS` beside `POLISH`/`POLISH_TOL`/
   `POLISH_DEGREE`/`POLISH_FAIL`, matching CLI flags, and README reference
   material stating what `AUTO` resolves to and the
   `CLI > Python > file > default` precedence.
3. The 0.8.0 report of the polish phase dying by OOM at `MPOL = NTOR = 10`
   is fixed by #234 (polish jits no longer bake per-solve constants); the
   fix shipped in v0.8.1 (2026-09-02). Close after confirming the reporter
   runs v0.8.1 or later.

---

## 9. Phase 2 - Build one complete performance and compilation observability suite

**Purpose:** establish where time and memory are actually spent before further optimization.

### 9.1 One benchmark driver

Create or consolidate into:

```text
benchmarks/profile_workflows.py
```

with a data-driven case matrix. Avoid a new script per microbenchmark unless it is a standalone external-code oracle.

The driver must emit JSON and optional profiler traces with:

- total wall time;
- input/read/setup time;
- Python tracing time where measurable;
- JAX lowering time;
- XLA compile time;
- persistent-cache lookup/hit/miss;
- device execution time with `block_until_ready()`;
- host-device transfer time;
- WOUT/native output and plotting time;
- peak host RSS;
- peak device memory;
- nonlinear iterations;
- force evaluations;
- Krylov iterations and true residual;
- number of traces/lowerings/compilations;
- executable count and approximate resident executable memory.

Use JAX's official tooling:

- `jax_log_compiles`;
- `jax_explain_cache_misses`;
- persistent compilation-cache debug logs;
- `jax.profiler.trace` / XProf;
- `jax.profiler.save_device_memory_profile`;
- Nsight Systems on NVIDIA when needed.

References:

- https://docs.jax.dev/en/latest/benchmarking.html
- https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html
- https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- https://docs.jax.dev/en/latest/profiling.html
- https://docs.jax.dev/en/latest/device_memory_profiling.html
- https://docs.jax.dev/en/latest/gpu_performance_tips.html

### 9.2 Required workflow matrix

Profile at minimum:

| ID | Workflow |
|---|---|
| F1 | fixed-boundary single-grid value |
| F2 | fixed-boundary multigrid value |
| F3 | fixed-boundary polished value |
| F4 | implicit scalar value + gradient |
| F5 | vector residual + full Jacobian |
| F6 | hot-restart parameter scan |
| F7 | scalar boundary optimization, 10 accepted steps |
| F8 | residual least-squares optimization, 5 accepted steps |
| F9 | fixed-boundary single-stage plasma + ESSOS coils |
| F10 | free-boundary value and adjoint |
| F11 | symmetric versus LASYM value/gradient/Jacobian |
| M1 | isotropic fixed-boundary mirror |
| M2 | axisymmetric free-boundary mirror |
| M3 | periodic hybrid equilibrium and GK geometry |
| B1 | BOOZ_XFORM_JAX one surface |
| B2 | BOOZ_XFORM_JAX many selected surfaces |
| C1 | epsilon-effective summary diagnostic |
| C2 | Gamma-c value and derivative-safe objective |

### 9.3 Timing regimes

Each JAX workflow must run in separate modes:

1. **cold process, empty persistent cache**;
2. **new process, populated persistent cache**;
3. **warm same process, same shapes/static arguments**;
4. **same process, changed physical parameters but same shapes**;
5. **same process, changed resolution/shape**.

Never report mode 3 as a cold solve. Never claim the persistent cache works without a logged cache hit.

### 9.4 Platforms

Store at least:

- Apple Silicon or representative laptop CPU;
- Linux x86 CPU;
- NVIDIA CUDA GPU;
- WSL2/CUDA when it remains a supported target.

GPU runs must record driver, CUDA, jaxlib, GPU model, memory, and XLA flags.

### 9.5 Artifact schema

Commit small machine-readable summaries such as:

```json
{
  "schema": 1,
  "repo": "uwplasma/vmex",
  "commit": "...",
  "case_sha256": "...",
  "platform": {...},
  "jax": {...},
  "workflow": "F4",
  "regime": "persistent_cache_reload",
  "timing_s": {...},
  "memory_bytes": {...},
  "iterations": {...},
  "accuracy": {...}
}
```

Large traces remain release/workflow artifacts, not Git blobs. Commit a manifest with hashes and retrieval instructions.

### 9.6 Baseline gate

No Phase 3 performance refactor begins until:

- the full matrix runs unattended;
- asynchronous timing is correct;
- at least one XProf trace exists for each flagship class;
- repeated compilation reasons are captured;
- current results are committed as the baseline.

---

## 10. Phase 3 - Improve VMEX solve, optimization, single-stage, and gradient performance

**Depends on:** Phase 2.

Apply changes only where the profile identifies a significant cost.

### 10.1 Stop accidental recompilation

Audit for:

- `jax.jit` around nested functions created on every call;
- lambdas or fresh `partial` objects passed as jitted callables;
- changing Python container structure;
- data-dependent static arguments;
- incidental dtype promotion;
- changing surface tuples or chunk sizes;
- closure capture of large arrays;
- repeated construction of mode/basis/runtime objects;
- host callbacks that produce new shapes.

Move hot jitted functions to module scope or cache stable callable objects. Add a regression test that counts traces/compiles for repeated same-shape evaluations.

### 10.2 Introduce explicit compiled plans

Use small static plans for repeated work:

```python
SolvePlan(resolution, symmetry, device, dtype, mode_tables, transforms)
PolishPlan(radial_basis, collocation_grid, validation_grid, chart, preconditioner)
ObjectivePlan(surfaces, Boozer resolution, field-line grid, diagnostic config)
```

Plans contain static arrays and shape policy, not mutable physical state. Cache by a deterministic key and bound cache size.

### 10.3 Share geometry and field evaluations across objectives

Multiple optimization objectives often rebuild geometry, fields, Boozer tables, and surface closures independently. Add an internal evaluation bundle:

```python
EvaluationContext(
    geometry,
    jacobian,
    metrics,
    fields,
    profiles,
    selected_surface_tables,
)
```

- Build once per equilibrium state and objective batch.
- Keep it a pytree or explicit function result; do not introduce hidden global memoization keyed by tracer identity.
- Aggregate scalar objectives before one reverse implicit solve.
- Aggregate vector residuals before one block response when their state/operator is shared.

### 10.4 Reuse linear algebra across nearby solves

- Carry GCRO-DR/GCROT recycle spaces across:
  - continuation steps;
  - optimizer trial points;
  - tangent/adjoint calls with the same operator;
  - neighboring radial surfaces when appropriate.
- Reuse block-Thomas factors while the frozen low-order operator remains within a measured drift threshold.
- Add a cheap certificate after reuse:
  - true linear residual;
  - preconditioned residual;
  - operator/factor drift estimate.
- Rebuild immediately on certificate failure.

### 10.5 Improve preconditioner refresh policy without changing parity mode

The VMEC-compatible default retains the fixed `ns4=25` update cadence.

Add an opt-in performance policy that can refresh when:

- preconditioned residual reduction degrades;
- diagonal/block coefficients drift beyond a threshold;
- Krylov iteration count rises;
- continuation changes pressure/current/load substantially.

Compare against:

- the 1991 Hirshman-Betancourt tridiagonal preconditioner;
- VMEC2000;
- VMEC++ `updateRadialPreconditioner()` behavior;
- VMEX 1-D and 2-D paths.

References:

- https://doi.org/10.1016/0021-9991(91)90267-O
- https://arxiv.org/abs/2502.04374

The adaptive policy cannot become default until it preserves converged state, VMEC parity where expected, and improves a broad benchmark set.

### 10.6 Polishing preconditioner

For the normal system, evaluate:

1. mode-block low-order factorization from PR #192;
2. radial block-Thomas with exact transpose;
3. a coarse radial/mode correction;
4. mixed-precision factors with float64 residual iterative refinement;
5. factor/recycle reuse across Gauss-Newton and continuation steps.

Track true residual and normal-equation residual. Do not judge a preconditioner only by iteration count if setup or memory dominates.

### 10.7 Fusion and allocation work

Inspect StableHLO and XProf before editing.

Candidates include:

- batched Fourier analysis/synthesis contractions;
- geometry/field intermediates materialized multiple times;
- parity splitting and LASYM transforms;
- collocation residual and transpose actions;
- repeated packing/unpacking of large pytrees;
- free-boundary vacuum response;
- Boozer input tables.

Prefer ordinary JAX refactoring and XLA fusion first. Use Pallas only for a kernel with:

- a reproduced bottleneck;
- stable shape contracts;
- CPU fallback;
- derivative tests;
- measurable end-to-end gain.

Pallas is experimental: https://docs.jax.dev/en/latest/401/pallas.html

### 10.8 Persistent cache policy

- Centralize cache setup in one documented helper/CLI policy.
- Set it before the first JAX compilation.
- Record cache directory, max size, trust warning, and hit/miss statistics.
- Do not call `jax.clear_caches()` inside normal library functions.
- Provide an explicit memory-release operation for command-line batch boundaries.
- Test cache reload in a new process.

### 10.9 Performance acceptance

For every promoted change:

- numerical result within the declared tolerance;
- no new compile for same shape and static configuration;
- no end-to-end regression greater than 5% on unaffected flagship workflows unless justified;
- at least 20% improvement in the targeted bottleneck or a clearly larger memory reduction;
- peak memory does not increase without a documented tradeoff;
- derivative cost and accuracy are measured, not assumed from primal speed.

---

## 11. Phase 4 - Make BOOZ_XFORM_JAX faster, smaller, and the canonical full transform

### 11.1 Baseline the actual kernel

Profile `surface_transform` and `run_vmec_jax` for:

- one symmetric surface;
- one LASYM surface;
- 5, 20, and all surfaces;
- low, medium, and high `mboz/nboz`;
- CPU and GPU;
- value, JVP, and VJP.

Measure the memory of:

- input phase tensor;
- output phase tensor;
- Newton Jacobian;
- vmap batch;
- surface chunking;
- returned spectra.

### 11.2 Replace environment-only behavior with explicit configuration

Add a frozen public configuration:

```python
@dataclass(frozen=True)
class BoozerConfig:
    mboz: int
    nboz: int
    surface_chunk: int | str = "auto"
    memory_budget_bytes: int | None = None
    execution: str = "auto"       # scan, host_chunks, batched
    newton_tol: float = ...
    max_newton: int = ...
```

Environment variables may supply defaults for CLI compatibility, but functions receive an explicit config.

### 11.3 Add a reusable static plan

```python
BoozerPlan(
    mode tables,
    angle grids,
    quadrature weights,
    transform blocks,
    chunk schedule,
    device/dtype,
)
```

- Build once for a static resolution.
- Reuse across surfaces, equilibrium iterates, and optimization steps.
- Cache with a bounded deterministic key.
- Keep physical coefficients out of the plan.

### 11.4 Evaluate contraction strategies

Implement benchmark branches, then keep only the winner per regime:

1. current dense point-by-mode phase tensors;
2. separable theta/zeta contractions;
3. blocked point/mode contractions with `lax.scan`;
4. FFT-based synthesis/projection where the transformed grid permits it;
5. streamed output-mode blocks;
6. custom VJP that recomputes cheap phases rather than retaining them.

The transform uses nonlinear mapped angles, so a plain 2-D FFT cannot replace every contraction. Use separability/FFT only where mathematically exact for the relevant stage.

### 11.5 Newton reuse across radial surfaces

Nearby surfaces have nearby Boozer angle shifts.

- Warm-start each selected surface from its neighbor.
- Optionally reuse a preconditioner/factorization with a residual certificate.
- Order surfaces monotonically in flux internally, then restore user order.
- Carry no warm state between unrelated equilibria unless explicitly supplied.

### 11.6 Memory-aware chunking

Compute chunk size from:

- available device memory;
- phase/Jacobian buffer model;
- requested mode and surface counts;
- derivative mode.

The auto policy must be deterministic, report its choice, and never use host RSS as a proxy for GPU capacity.

### 11.7 Refactor ownership

- Keep compatibility I/O and legacy object API in `core.py` or a clearly named compatibility layer.
- Keep the canonical JAX kernel, plan, and dataclasses together.
- Remove duplicated sign/layout conversion after tests prove one owner.
- Keep public names backward compatible through thin aliases during one deprecation cycle.
- Shorten docstrings that repeat theory already documented elsewhere, but retain conventions and derivative semantics.

### 11.8 Cross-code and derivative tests

Compare with:

- classic BOOZ_XFORM/STELLOPT;
- existing BOOZ_XFORM_JAX output;
- VMEX host WOUT path;
- analytic axisymmetric and single-helicity fields.

Require:

- spectra and coordinate shifts within existing or tighter tolerances;
- symmetry/LASYM parity;
- surface-order invariance;
- JVP/VJP duality;
- finite-difference checks on smooth cases;
- memory tests demonstrating bounded growth with surface count under chunking.

### 11.9 Decide the VMEX lightweight transform

Benchmark `vmex/core/omnigenity.py` symmetric transform against the canonical BOOZ_XFORM_JAX plan.

Keep the VMEX implementation only if all are true:

- it is materially faster for inner-loop selected-surface objectives;
- it has lower memory;
- it matches full BOOZ_XFORM_JAX for the needed outputs;
- maintaining two paths has a clear test boundary.

Otherwise:

- retain `boozer_input_tables` in VMEX;
- call BOOZ_XFORM_JAX for both symmetric and LASYM;
- delete the duplicate transform after updating QI/omnigenity tests.

**Acceptance:** one canonical full transform, explicit memory policy, lower measured peak memory, and no accuracy regression.

---

## 12. Phase 5 - Quick confinement-summary improvement: plot Gamma-c with epsilon effective

This is a deliberately small early PR after the relevant APIs are stable.

### 12.1 Current state

The top-middle summary panel already plots pressure and overlays diagnostic `epsilon_eff^(3/2)` on a right axis. Add `Gamma_c` to this same confinement axis; do not create a fourth row or a second expensive independent Boozer transform.

### 12.2 Shared diagnostic context

Replace the narrowly named epsilon cache with a bounded confinement cache:

```python
ConfinementSummary(
    surfaces,
    epsilon_effective,
    gamma_c,
    validity,
    notes,
    timing,
)
```

Key the cache by:

- WOUT/native equilibrium identity or content hash;
- selected surfaces;
- Boozer resolution;
- NEO configuration;
- Gamma-c diagnostic configuration.

Use weak references where in-memory objects are keys. Bound the number of entries.

### 12.3 Avoid duplicate work

- Reuse one in-memory Boozer result for effective ripple and any Gamma-c ingredients that actually live in Boozer coordinates.
- Reuse selected surfaces and field tables.
- Do not force Gamma-c through Boozer if its current validated real-space path is faster and more accurate; share only mathematically common work.
- Do not clear all JAX caches inside a plotting library call.
- Change `epsilon_effective_from_wout(..., clear_jax_caches=True)` to a library-safe default of `False`.
- Let the CLI explicitly release caches after all requested diagnostics when memory policy asks for it.

### 12.4 Plot semantics

- Pressure stays on the left axis.
- `epsilon_eff^(3/2)` and `Gamma_c` share the right confinement axis.
- Use distinct markers/linestyles and one combined legend.
- Use log scale only when all valid positive data and dynamic range justify it.
- If one diagnostic is unavailable, plot the other and state the reason in the panel metadata/title or log.
- Never plot a failed/nonconverged Gamma-c value as zero.

### 12.5 Tests and budget

- unit test the combined-axis metadata and line labels;
- test missing NEO_JAX, invalid Gamma-c, and one-valid/one-invalid cases;
- test cache hit on repeated summary generation;
- test that same-result plotting does not compile the same diagnostic again;
- add a bounded runtime gate for the bundled diagnostic case;
- ensure figures are closed.

**Acceptance:** `vmex --plot wout_*.nc` produces the requested combined profile without a material increase from duplicated work.

---

## 13. Phase 6 - Make Gamma-c usable in optimization without branch-noise derivatives

### 13.1 Keep three distinct semantics

Expose three clearly named objects:

1. `GammaCValue` - current hard physical diagnostic, not promised differentiable across topology events;
2. `GammaCTracked` - fixed/matched well topology with implicit bounce-point derivatives;
3. `GammaCSmooth` - smooth optimization surrogate with annealed topology regularization.

Do not keep one class whose derivative meaning changes silently with options.

### 13.2 Tracked-well formulation

At a reference equilibrium:

1. detect all complete wells for each line and pitch;
2. assign persistent well IDs and orientation;
3. store brackets for left/right bounce roots;
4. define feature vectors using center, width, `B_min`, `B_max`, bounce action, and neighboring connectivity;
5. match wells at the next iterate using a minimum-cost assignment;
6. solve bounce points inside their matched brackets;
7. differentiate the roots implicitly rather than through a discrete root-search iteration;
8. calculate Gamma-c on the fixed matched topology.

Use the graph perspective of multi-well bounce domains to represent splitting, merging, and connectivity. Reference:

- I. E. Ochs, *Bounce-averaged theory in arbitrary multi-well plasmas: solution domains and the graph structure of their connections*, J. Plasma Phys. (2025), https://doi.org/10.1017/S002237782510069X

### 13.3 Branch events

A merge, split, birth, death, bracket crossing, or assignment ambiguity is not an ordinary differentiable step.

Return a report:

```python
GammaCTopologyReport(
    matched,
    born,
    lost,
    merged,
    split,
    ambiguous,
    min_bracket_margin,
)
```

Optimization policy:

- accept a tracked derivative only when topology is certified unchanged;
- reduce the outer trust region when a branch event is imminent;
- refresh the reference topology after an accepted event;
- reject or fall back to the smooth surrogate when matching is ambiguous;
- always recompute the hard value after an accepted outer step.

### 13.4 Smooth surrogate

Develop a root-free or smoothly rooted surrogate using:

- soft extrema for pitch range;
- smooth occupancy near `1 - lambda B = 0`;
- smooth periodic partition of wells;
- softmin tangency evaluation rather than hard argmin;
- temperature tied to angular/field-line resolution;
- annealing during optimization.

Requirements:

- converges toward hard Gamma-c as temperature decreases and resolution increases on fixed topology;
- gradients converge under field-line, pitch, quadrature, and radial refinement;
- no overflow/well-count discontinuity in the optimization range;
- value correlation with hard Gamma-c and ESSOS prompt loss is documented.

### 13.5 Efficient implementation

- Keep field-line geometry and pitch grids static for a stage.
- Batch surfaces, field lines, and pitches while bounding memory.
- Cache topology and brackets outside the AD trace.
- Use SOLVAX implicit/root primitives for bounce roots if they fit; otherwise add only a generic bracketed implicit-root primitive to SOLVAX.
- Reuse spectral point-evaluation closures and avoid one `jacfwd` construction per point.
- Profile hard, tracked, and smooth variants separately.

### 13.6 Physics references

- V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold, *Poloidal motion of trapped particle orbits in real-space coordinates*, Phys. Plasmas 15, 052501 (2008), https://doi.org/10.1063/1.2912456
- J. L. Velasco et al., *A model for the fast evaluation of prompt losses of energetic ions in stellarators*, Nucl. Fusion 61, 116059 (2021), https://doi.org/10.1088/1741-4326/ac2994
- K. Unalmis et al., *Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm and its applications*, J. Plasma Phys. 92 (2026), https://doi.org/10.1017/S0022377826101652
- J. R. Cary and S. G. Shasharina, *Helical plasma confinement devices with good confinement properties*, Phys. Rev. Lett. 78, 674 (1997), https://doi.org/10.1103/PhysRevLett.78.674

### 13.7 Gradient certification

For at least three configurations:

- simple single-well analytic/synthetic field;
- bundled QA/QH equilibrium;
- multi-well equilibrium near, but not at, a topology event;

check:

- resolution convergence of value;
- resolution convergence of directional derivative;
- tracked implicit derivative versus fixed-topology finite difference;
- smooth derivative versus finite difference;
- JVP/VJP duality;
- branch-event detection;
- hard value before/after optimization.

Do not promote until the derivative has a stable sign and magnitude under the declared production resolution refinement.

### 13.8 Combined omnigenity example

Create one file:

```text
examples/optimization/omnigenity_epsilon_gammac_maxj.py
```

Structure:

1. imports;
2. all user parameters at the top;
3. input/boundary construction;
4. VMEX solve with optional polishing;
5. shared Boozer/field-line plan;
6. normalized objective terms;
7. one optimizer call;
8. before/after print table;
9. hard validation metrics;
10. plots and saved outputs.

Objective:

```text
w_eps * normalized epsilon_eff
+ w_gc * normalized derivative-safe Gamma-c
+ w_maxj * normalized maximum-J residual
+ geometric and force-balance constraints.
```

Rules:

- scale terms from the initial values or explicit physical scales;
- print each term separately;
- use one aggregate scalar adjoint;
- validate with hard Gamma-c and ESSOS orbit losses;
- include a fast CI mode and a documented research-resolution mode;
- do not imply that minimizing the surrogate guarantees zero losses.

---

## 14. Phase 7 - Make LASYM fast, accurate, and scientifically meaningful

### 14.1 Full LASYM performance anatomy

Use the Phase 2 harness to compare symmetric and LASYM at identical physical/mode resolution:

- setup;
- geometry synthesis;
- force kernels;
- parity reconstruction;
- Fourier analysis;
- preconditioner assembly/application;
- residual/Jacobian;
- WOUT;
- Boozer;
- stability and Gamma-c;
- implicit tangent/adjoint;
- outer optimization.

The existing vectorized filtering gains are not enough; identify the remaining 2.5-5x residual gap and any 8-contraction/extended-precision path.

### 14.2 Remove avoidable duplicate parity work

Candidate work, subject to profiling:

- stack symmetric/asymmetric kernels into fewer batched contractions;
- use one full-circle transform plan rather than repeated mirror/reindex operations;
- cache LASYM mode masks and partner maps;
- avoid reconstructing zeros for inactive families;
- separate validation-only long-double calculations from production when float64 is certified;
- fuse paired cosine/sine projections;
- chunk Jacobian/VJP families by memory model.

### 14.3 Accuracy matrix

For fixed/free-boundary and vacuum/finite-beta cases:

- compare VMEX and VMEC2000 WOUT tables;
- compare independently evaluated force balance;
- verify parity under controlled asymmetric perturbations;
- verify Mercier/Glasser, bootstrap, Boozer, field, and virtual-casing quantities;
- test hot restart and multigrid;
- test polished LASYM only after PR #192 explicitly supports/certifies it.

### 14.4 Meaningful asymmetric optimization

Do not merely set `LASYM=True` on a symmetric objective and celebrate a tiny sine coefficient.

Build a flagship campaign with:

- a symmetric optimized reference;
- an asymmetric design space initialized through a controlled sine-mode continuation;
- a physical objective that can benefit from asymmetry, such as confinement under asymmetric port/coil/field constraints, or demonstrably lower QI/QS/Gamma-c/loss metric under the same engineering constraints;
- a gauge-invariant asymmetry norm;
- bounded surface separation, curvature, aspect ratio, iota, and force balance;
- at least three small independent asymmetric seed perturbations.

A result is physically distinct only when:

- sine-family amplitudes exceed resolution/noise and remain under refinement;
- the asymmetry cannot be removed by a toroidal/poloidal phase shift or coordinate gauge;
- the objective improvement survives a re-solve and independent evaluation;
- multiple seeds find the same basin or comparable result;
- the final field is nested, force balanced, and coil-feasible enough for the stated claim.

### 14.5 LASYM downstream gaps

- BOOZ_XFORM_JAX already carries asymmetric spectra; keep this path canonical.
- NEO_JAX currently rejects LASYM in VMEX's effective-ripple adapter. Decide whether to:
  1. extend NEO_JAX's Boozer data contract and equations to asymmetric harmonics; or
  2. use DKX/another neoclassical oracle for LASYM effective-ripple-like transport.
- ESSOS released field readers must preserve asymmetric WOUT tables and differentiability.
- Add explicit capability tables so unsupported downstream combinations fail, not symmetrize.

### 14.6 Acceptance

- same-shape LASYM repeated evaluations do not recompile;
- targeted performance bottlenecks improve materially;
- all derivative certificates pass;
- one committed asymmetric configuration and script reproduce the scientific improvement;
- README wording states exactly which objective and constraints improved.

---

## 15. Phase 8 - Formalize mirror boundary conditions and harden the isotropic lane

### 15.1 Write the model contract before changing numerics

Add `docs/explanation/mirror-boundary-conditions.md` deriving the variational and strong equations for each topology.

#### Open fixed-cut mirror

- radial label `s in [0,1]`;
- poloidal angle `theta`;
- axial coordinate `xi in [-1,1]`;
- fixed lateral boundary at `s=1`;
- regular axis at `s=0`;
- prescribed geometry and normal magnetic flux at the two cuts;
- cuts permit through-flux and are not plasma-vacuum interfaces;
- no claim of sheath/end-loss equilibrium physics.

#### Open free-lateral-boundary mirror

- side wall is the plasma-vacuum interface;
- ideal interface has `B . n = 0` on each side;
- isotropic total-pressure continuity is `p + B^2/(2 mu0)`;
- anisotropic continuity becomes `p_perp + B^2/(2 mu0)`;
- end caps close the Green surface mathematically but receive no side-wall pressure-balance condition.

#### Periodic hybrid

- all geometry and field variables periodic;
- no end-cap conditions;
- field-line closure/rationality handled as a diagnostic/selection condition, not a fake boundary condition.

### 15.2 Derive natural boundary terms

Starting from the existing mirror energy, derive the first variation including all lateral and cut terms. For every term, show whether it vanishes because of:

- fixed geometry;
- fixed flux;
- periodicity;
- regularity;
- physical interface balance;
- gauge.

Turn these into tests by evaluating the discrete directional derivative for variations that isolate each boundary family.

### 15.3 Strengthen radial-axis regularity

The current radius interpolation uses odd/even leading factors, while the stream function uses full mode-dependent `rho^|m|` regularity.

Audit the physical polar representation and implement full mode-dependent regularity where required:

```text
coefficient_m(rho, xi) = rho^|m| * smooth_even_function(rho, xi)
```

or an equivalent smooth polar B-spline chart.

Reference the general high-order polar regularity literature, including:

- https://arxiv.org/abs/2601.17841

Do not change the stored state until interpolation/derivative tests demonstrate the need and mapping.

### 15.4 Isotropic validation suite

Add or tighten:

- straight circular vacuum mirror;
- paraxial finite-beta mirror;
- manufactured geometry/field with known curl/divergence;
- weak residual versus AD energy gradient;
- strong residual convergence;
- axis regularity;
- cut boundary terms;
- lateral fixed-boundary constraint;
- free-boundary interface residual;
- divergence-free field;
- Fourier versus B-spline axial representation;
- periodic hybrid limit.

### 15.5 Migrate only generic solver algebra

The mirror solver currently uses host SciPy GMRES/least-squares/minimize and its own bounded Newton orchestration.

Compare with SOLVAX 0.20.0:

- matrix-free Gauss-Newton;
- PTC/Newton-Krylov;
- separable/Kronecker preconditioner;
- implicit stationarity.

Migrate only if:

- same equations and convergence contract;
- lower or equal cold/warm cost at production size;
- lower memory;
- clean JIT/gradient semantics;
- dense rescue remains a small-case diagnostic, not an unbounded production fallback.

**Acceptance:** a documented, independently tested isotropic mirror baseline suitable for anisotropic extension.

---

## 16. Phase 9 - Implement a literature-backed anisotropic mirror equilibrium

### 16.1 Physics model

Use a gyrotropic pressure tensor:

```text
P = p_perp I + (p_parallel - p_perp) b b,
b = B / |B|,
J x B = div(P).
```

The parallel projection of force balance imposes:

```text
B . grad(p_parallel)
- (p_parallel - p_perp) B . grad(log B) = 0.
```

For a closure `p_parallel = p_parallel(s, B)`, enforce:

```text
p_perp = p_parallel - B * partial_B(p_parallel).
```

This constraint is essential. Do not accept arbitrary independent three-dimensional `p_parallel` and `p_perp` as a supposedly equilibrated closure.

### 16.2 Pressure-model hierarchy

Define a small protocol:

```python
class AnisotropicPressureModel(Protocol):
    def parallel(self, s, B, volume_derivative, params): ...
    def perpendicular(self, s, B, volume_derivative, params): ...
    def energy_density(self, s, B, volume_derivative, params): ...
```

Implement in this order:

1. **ANIMEC-style equilibrium-consistent bi-Maxwellian/energetic-particle model.** Use the distribution moments and invariants documented by Cooper et al.; reproduce an isotropic limit.
2. **Tabulated `p_parallel(s,B)` model.** Use differentiable splines and compute `p_perp` from the B derivative.
3. **CGL double-adiabatic model.** Keep it explicitly named and separate; do not silently mix its invariants with the ANIMEC closure.

### 16.3 Variational formulation

The ANIMEC functional `W = integral [B^2/(2 mu0) + p_parallel/(Gamma - 1)] dV`
(Cooper 1992 Eq. 2.1; Moen, Suzuki, Proll 2023 Eq. 2.1) holds only with
the adiabatic index set to ZERO, which ANIMEC does: then the thermal part is
a prescribed flux function and the pressure enters with a MINUS sign,
`W = integral [B^2/(2 mu0) - p_parallel(s, B)] dV` (Grad 1967). Using VMEX's
`gamma = 5/3` mass-conserving energy with a B-dependent `p_parallel` gives a
WRONG force (slab proof in the 2026-09-02 mirror audit, section 31.4).
Implement the split form, which keeps the shipped isotropic solver
bit-identical and makes plan item 16.8's "exact isotropic limit" literal:

```text
W = integral [B^2/(2 mu0) + p_th/(Gamma - 1)] dV - integral p_h,par(s, B) dV,
p_th = M(s)/V'^Gamma            (the shipped thermal part, unchanged),
p_par = p_th + p_h,par,   p_perp = p_par - B d_B p_h,par|_s   (Grad closure),
```

with the chosen distribution/constraint determining the geometry dependence of `p_h,par`. Both terms give `-integral xi . (J x B - div P)` at first order.

Derive the discrete weak residual by AD of the energy first. Then implement an independent strong residual:

```text
F = J x B - div(P).
```

Do not derive both from the same final expression.

### 16.4 Files

Recommended minimal changes:

- new `vmex/mirror/anisotropy.py` for pressure models, tensor, and diagnostics;
- extend `vmex/mirror/forces.py` for anisotropic energy/weak/strong residuals;
- extend `MirrorConfig` or add a small immutable pressure-model field without filling it with many scalar options;
- extend `vmex/mirror/output.py` for `p_parallel`, `p_perp`, anisotropy, and stability profiles;
- extend solver dispatch without duplicating the isotropic solver.

### 16.5 Boundary conditions

- Fixed lateral boundary: geometry Dirichlet; force certificate excludes constrained normal variations but evaluates interior strong force.
- Free lateral boundary:

```text
B . n = 0,
p_perp + B^2/(2 mu0) continuous across the interface.
```

- End cuts: fixed through-flux cuts; do not impose plasma-vacuum total-pressure continuity there.
- Periodic hybrid: periodic anisotropic state and pressure model.

### 16.6 Stability/admissibility diagnostics

Report, without conflating them with full kinetic stability:

- firehose parameter;
- mirror-instability parameter appropriate to the selected closure;
- positivity of both pressures;
- positivity/regularity of distribution moments;
- parallel-integrability residual;
- total-pressure interface residual;
- Jacobian/nestedness.

Fail or warn according to explicit thresholds.

### 16.7 Reference literature

Primary references:

- Chew, Goldberger, Low, *The Boltzmann equation and the one-fluid hydromagnetic equations in the absence of particle collisions*, Proc. R. Soc. A 236 (1956).
- W. A. Cooper et al., *3D magnetohydrodynamic equilibria with anisotropic pressure*, Comput. Phys. Commun. 72, 1-13 (1992), https://doi.org/10.1016/0010-4655(92)90002-G
- W. A. Cooper et al., *Three-dimensional anisotropic pressure free boundary equilibria*, Comput. Phys. Commun. 180, 1524-1533 (2009), https://doi.org/10.1016/j.cpc.2009.04.006
- D. Endrizzi et al., *Physics basis for the Wisconsin HTS Axisymmetric Mirror (WHAM)*, J. Plasma Phys. 89 (2023), https://doi.org/10.1017/S0022377823000806
- S. J. Frank et al., *Nonlinear anisotropic equilibrium reconstruction in axisymmetric magnetic mirrors*, arXiv:2509.17288
- D. D. Ryutov et al., *Magneto-hydrodynamically stable axisymmetric mirrors*, Phys. Plasmas 18, 092301 (2011), https://doi.org/10.1063/1.3624763 — long-thin ordering; Eq. 30 is `B/B_vac = sqrt(1 - beta)`; Eq. 4 is the parallel balance; Eqs. 26, 31-32 are the stability integrals.
- P. Merkel, *An integral equation technique for the exterior and interior Neumann problem in toroidal regions*, J. Comput. Phys. 66, 83 (1986), https://doi.org/10.1016/0021-9991(86)90055-0
- S. P. Hirshman, W. I. van Rij and P. Merkel, *Three-dimensional free boundary calculations using a spectral Green's function method*, Comput. Phys. Commun. 43, 143 (1986), https://doi.org/10.1016/0010-4655(86)90058-5
- O. Agren and N. Savenko, *Magnetic mirror minimum B field with optimal ellipticity*, Phys. Plasmas 11, 5041 (2004), https://doi.org/10.1063/1.1799351 — the SFLM paraxial potential, on-axis field, and ellipticity used by `analytic.StraightFieldLineMirror`.
- O. Agren and N. Savenko, *Rigid rotation symmetry of a marginally stable minimum B field and analytical expressions of the flux coordinates*, Phys. Plasmas 12, 042505 (2005), https://doi.org/10.1063/1.1870002 — the Clebsch flux coordinates `(x0, y0)` and the quadrupolar-symmetry proof.

Also inspect ANIMEC source/manuals, but independently verify all formulas and conventions.

**DESC as a mirror oracle: admissible only with the right profile
parameterization (31.4-R5, corrected against the DESC source).** DESC's
`ForceBalanceAnisotropic` (`desc/objectives/_equilibrium.py`,
`desc/compute/_equil.py::_F_anisotropic`) minimizes

    F = (1 - beta_a) J x B - (B.grad beta_a) B/mu0 - beta_a grad(B^2)/2mu0 - grad(p_perp)

with `beta_a = mu0 (p_par - p_perp)/B^2` from `Equilibrium.anisotropy` and
`p_perp` from `Equilibrium.pressure`. That expression is the *exact* divergence
of `P = p_perp I + (p_par - p_perp) bb` for the Grad-type tensor, so the
equations themselves are correct and general. The audit's blanket
"inadmissible" is too strong; the real restriction is on how the two profiles
are parameterized.

Project F on `b` (the `J x B` term drops out) and the parallel balance is

    b.grad p_perp + beta_a b.grad(B^2)/2mu0 + (B^2/mu0) b.grad beta_a = 0.

- If **both** `beta_a` and `p_perp` are flux functions — DESC's default
  `PowerSeriesProfile` — this collapses to `beta_a b.grad B = 0`. In a mirror
  `b.grad B != 0` everywhere, so the only solution is `beta_a == 0`: a
  flux-function DESC anisotropic run carries no mirror anisotropy at all and is
  **not** an oracle. On a closed toroidal field line the same inconsistency is
  masked because `b.grad B` averages out over a period.
- The 31.4 spec-sheet model 1, `p_par = p0(psi) + Delta(psi) B^2/2mu0`, gives
  `beta_a = Delta(psi)` (a flux function) but `p_perp = p0(psi) -
  Delta(psi) B^2/2mu0`, which is **not**. It satisfies the equation above
  identically, and is the family where a DESC cross-check is legitimate — but
  DESC must then be given `anisotropy` as a radial profile *and* `pressure` as
  a `FourierZernikeProfile` carrying the `-Delta B^2/2mu0` dependence.
- With a fully 3-D `FourierZernikeProfile` `beta_a`, DESC can satisfy the
  parallel balance for other closures too (with flux-function `p_perp` the
  solution is `beta_a B = const` along a field line), but then `beta_a` is a
  free field fixed by force balance rather than by a kinetic closure. Matching
  VMEX then requires the same closure imposed on both sides.

Any DESC comparison outside these cases is a code-to-code difference, not a
validation, and must be labelled as such.

### 16.8 Tests

- exact isotropic limit at value, residual, and derivative levels;
- analytic tensor divergence in Cartesian manufactured fields;
- parallel-integrability identity;
- `p_perp = p_parallel - B partial_B p_parallel` AD/analytic agreement;
- weak energy directional derivative versus residual;
- strong residual refinement;
- free-boundary total-pressure condition;
- comparison with published ANIMEC cases where inputs can be reproduced;
- WHAM-like axisymmetric profile comparison;
- CPU/GPU and JVP/VJP tests;
- invalid closure and instability diagnostics.

**Acceptance:** a fixed-boundary axisymmetric and nonaxisymmetric anisotropic mirror solve with independent force and closure certificates.

---

## 17. Phase 10 - Complete axisymmetric and nonaxisymmetric free-boundary mirrors

### 17.1 Axisymmetric anisotropic circular-coil case

Construct a reproducible ESSOS coil system using circular coils, not a hand-tuned external field table.

Workflow:

1. define circular coil centers, radii, currents, and symmetry;
2. evaluate the external field with ESSOS in memory;
3. build the mirror exterior response;
4. solve the coupled anisotropic plasma-boundary-vacuum equilibrium;
5. continue beta/anisotropy from a vacuum or low-beta state;
6. validate interface and strong force;
7. compare profiles and boundary displacement with paraxial/WHAM-like expectations.

Use a parameter continuation in pressure and anisotropy rather than asking one nonlinear solve to jump directly to the target state.

### 17.2 Extend the exterior response to nonaxisymmetry

`vmex/mirror/exterior.py` already represents a general 3-D lateral wall and mathematical end caps. The work is primarily in the coupled solver and numerical certification.

Tasks:

- remove the `ntheta == 1` gate only after all operators accept general theta;
- retain full theta dependence in side-wall geometry, normals, interpolation, and density;
- verify symmetry reduction does not accidentally impose axisymmetry;
- support general external ESSOS fields;
- implement matrix-free boundary response and transpose;
- add near-singular quadrature/refinement for distorted side walls;
- keep the cap flux projection (`_balance_neumann_on_caps`) general in theta.
  This is *not* a Neumann solvability or gauge condition: the exterior Neumann
  problem with decay at infinity is uniquely solvable for arbitrary data, with
  no compatibility constraint and no additive-constant freedom (hence
  `gauge_error == 0`). The projection enforces solenoidality consistency,
  `sum_S B.n dA = 0`, which the discrete lateral and cap interpolants do not
  satisfy to round-off on their own; a nonzero value is a spurious magnetic
  monopole. The equations are written out in
  `docs/explanation/mirror-boundary-conditions.md`;
- use a Schur complement that separates interior plasma variables from boundary/vacuum variables;
- precondition plasma and exterior blocks with their natural structured solvers.

### 17.3 Coupled residual

The free-boundary residual must include:

- interior weak/strong equilibrium equations;
- side-wall `B . n` condition;
- isotropic or anisotropic total-pressure balance;
- exterior Laplace/BIE equation and the cap solenoidality projection (there is
  no gauge block: see 17.2);
- fixed cut geometry/flux constraints;
- free-boundary geometry chart constraints;
- optional coil-current/shape parameters in the differentiable input.

A small residual is insufficient unless all blocks are separately normalized and reported.

Standing limitations of this residual, to be re-checked whenever the exterior
model changes:

- **Net axial plasma current.** `I'(s) != 0` is inadmissible in the
  free-boundary lane and is rejected at the entry points
  (`reject_net_axial_current`). The exterior is a single-valued scalar
  potential decaying at infinity on a topologically spherical Green surface;
  that exterior is simply connected, so the potential carries no azimuthal
  field. A net axial current gives the plasma `B_phi = mu0 I / (2 pi r)` with no
  vacuum counterpart, and the total-pressure jump would then compare
  physically inconsistent fields. NESTOR handles the toroidal analogue by
  folding a net-current filament into `B_ext`; the mirror lane has no such
  term. Lifting the limitation means adding the analytic `phi`-hat field of the
  end-electrode circuit to `lateral_field_xyz` under a stated end-electrode
  assumption, and demonstrating that the interface residual is then consistent
  (31.4-R2).
- **Nonaxisymmetry.** `ntheta == 1` is still required (17.2).
- **Anisotropy.** The lateral jump uses the isotropic `p + B^2/2mu0`; the
  anisotropic form is Phase 9 work.

### 17.4 Globalization

Use deterministic continuation:

1. vacuum field and prescribed boundary;
2. low pressure with fixed boundary;
3. release boundary with a penalty/continuation parameter;
4. increase beta;
5. increase anisotropy;
6. introduce nonaxisymmetric external-field/boundary modes.

Use SOLVAX PTC or damped Newton-Krylov for the coupled root, with pseudo-arclength if the physical branch folds. Do not expose multiple SciPy optimizers as public user choices.

### 17.5 Axisymmetric validation

- circular coil field versus analytic Biot-Savart on and off axis;
- vacuum boundary response versus direct field evaluation;
- plasma off recovers vacuum;
- isotropic limit recovers existing free-boundary mirror;
- anisotropic continuation converges under grid refinement;
- side-wall total pressure and `B . n` converge;
- cap closure does not receive a false pressure-balance residual;
- volume/flux and force checks.

### 17.6 Nonaxisymmetric validation

Use increasingly difficult cases:

1. axisymmetric coil system evaluated on a multi-theta grid: must recover the axisymmetric result;
2. small tilted-coil/asymmetric perturbation: compare linear response to finite difference;
3. nonaxisymmetric fixed-boundary equilibrium embedded in a matching external field;
4. fully free nonaxisymmetric boundary driven by a declared coil set.

Check:

- theta refinement;
- axial spline refinement;
- cap/rim grading;
- boundary-integral quadrature;
- external-field sampling;
- coupled residual;
- tangent/adjoint duality and finite differences.

### 17.7 Output and plotting

Extend MOUT/native mirror output to contain:

- free boundary and coil metadata;
- anisotropic pressure fields/profiles;
- interface residual profiles;
- external, plasma, and total fields;
- strong-force and divergence certificates;
- continuation history.

Add plots for:

- side-wall pressure balance;
- `B . n`;
- `p_parallel/p_perp`;
- firehose/mirror parameters;
- coil geometry and flux surfaces;
- nonaxisymmetric cross-sections.

**Acceptance:** one axisymmetric anisotropic circular-coil case and one nonaxisymmetric free-boundary mirror pass full independent certificates.

---

## 18. Phase 11 - Complete the stellarator-mirror hybrid equilibrium and optimization

### 18.1 Clarify the physical model

The current hybrid is a periodic closed field-line device with straight mirror-like legs and stellarator-like returns. It is not an open mirror with end loss. Documentation must state:

- periodic topology;
- axis construction and curvature transitions;
- cross-section rotation;
- flux/current/pressure model;
- definition of mirror ratio and mirror length. **Done (31.4-R3):**
  `vmex/mirror/metrics.py` defines `R_m,axis` per leg (max/min of |B| on the
  axis over that leg's |B| well), `R_m,LCFS` separately, `L_mirror,B` (arc
  distance between the two |B| maxima bounding a well) and `L_straight` (arc
  length where the axis curvature is negligible), with persistence pruning so
  ripple in a solved |B| is not reported as an extra leg. The examples, tests,
  docs and the GK field-line contract all use it. The GKX/GS2 `epsilon` key
  is the field-line modulation depth `(Bmax-Bmin)/(Bmax+Bmin)` in both lanes,
  i.e. the field-line `R_m = (1+eps)/(1-eps)` — see
  `vmex.core.turbulence.b_modulation_depth`;
- field-line closure assumptions;
- where Boozer coordinates and toroidal stellarator diagnostics remain meaningful;
- how local mirror/GK geometry is extracted.

### 18.2 Equilibrium hardening

- apply the shared high-order spline/regularity framework;
- add independent strong-force and divergence certificates for closed curved-axis geometry;
- verify straight-leg and toroidal/stellarator limits;
- verify field-line closure and equal-arc remapping from PR #194;
- support isotropic first, then anisotropic pressure after Phase 9;
- add implicit equilibrium derivatives through the promoted root/stationarity system.

### 18.3 Design variables

Build a compact, physical parameterization rather than optimizing every coefficient at first:

- straight-leg length;
- return radius and return-shape harmonics;
- major/minor cross-section radii;
- ellipticity/triangularity where supported;
- number and phase of section rotations;
- axial and toroidal current/flux controls;
- pressure/anisotropy parameters;
- selected periodic spline coefficients for local refinement;
- coil-current and coil-curve variables in the later single-stage example.

### 18.4 Objectives and constraints

Required equilibrium/design terms:

- target mirror ratio in the straight legs;
- target effective mirror length and plateau length;
- QI/omnigenity or second-invariant contour quality;
- maximum-J where appropriate;
- target iota/field-line closure;
- strong-force certificate;
- magnetic well/stability constraints;
- surface Jacobian/nestedness;
- curvature and smooth transition from legs to returns;
- minimum plasma-plasma and plasma-axis separation;
- coil feasibility and normal-field error after ESSOS coupling;
- optional loss, DKX transport, and GKX turbulence terms in staged optimization.

Normalize every residual using a physical or initial scale and print all components.

### 18.5 Optimization strategy

Do not optimize all physics at full resolution from a cold seed.

Use stages:

1. low-resolution analytic geometry and vacuum/low-beta equilibrium;
2. mirror ratio and length;
3. field-line closure/iota;
4. QI/second-invariant quality;
5. force and resolution refinement;
6. ESSOS coil construction;
7. ESSOS loss validation/optimization;
8. DKX/GKX validation and optional correction.

Use predictor warm starts from implicit tangents. Recycle linear spaces. Re-polish accepted equilibria when the outer step changes geometry enough to invalidate the certificate.

### 18.6 Research-grade flagship result

Produce one fully reproducible example and committed configuration with:

- before/after geometry;
- mirror ratio and length profile;
- QI/J contours;
- force balance;
- iota/field-line closure;
- ESSOS coil set and normal-field error;
- ESSOS alpha loss fraction;
- DKX neoclassical coefficients;
- GKX turbulence metric at selected locations;
- cold/warm/gradient performance.

The example must remain understandable. Put large campaign settings in a small data file, not hundreds of lines of Python.

### 18.7 Tests

- periodic spline refinement exactness;
- axis and section regularity;
- straight-leg analytic geometry;
- closed-axis metric identities;
- equal-arc remap;
- field-line closure;
- strong-force refinement;
- parameter JVPs;
- objective gradients;
- consumer-adapter parity.

**Acceptance:** a periodic hybrid equilibrium and optimization that can be independently reproduced and consumed by ESSOS, GKX, and DKX.

---

## 19. Phase 12 - Establish one native downstream equilibrium/geometry protocol

### 19.1 Do not make files the optimization-loop API

WOUT, Boozmn, and MOUT remain compatibility and archival formats. In-memory consumers should receive pytrees/arrays and static metadata.

Define a small structural protocol, likely in `vmex/core/interfaces.py`:

```python
class EquilibriumView(Protocol):
    def sample_flux_surface(self, s, theta, zeta, *, derivatives=()): ...
    def magnetic_field(self, xyz): ...
    def boundary_surface(self, theta, zeta): ...
    def boozer_inputs(self, surfaces, *, resolution): ...
    @property
    def provenance(self): ...
```

For mirrors/hybrids, provide topology-specific views rather than returning fake toroidal quantities.

### 19.2 BOOZ_XFORM_JAX

- Accept `boozer_inputs()` directly.
- Return a frozen JAX-native `BoozerData` carrying cosine and sine spectra, surfaces, currents, iota, conventions, and provenance.
- Avoid reconstructing a WOUT in memory merely to call the transform.
- Use the same object for QI, epsilon effective, DKX, and plots.

### 19.3 VIRTUAL_CASING_JAX

- Consume `boundary_surface()` and the required surface field/current arrays directly.
- Preserve high-order boundary derivatives from the native equilibrium.
- Validate native versus WOUT sampling and finite differences.
- Keep VMEX's convenience adapter thin; VIRTUAL_CASING_JAX owns quadrature and singular treatment.

### 19.4 ESSOS

Use separate adapters for:

- VMEX equilibrium field;
- polished native equilibrium field;
- mirror/hybrid field;
- boundary surface;
- free-boundary external coils.

Requirements:

- no file round trip in differentiable examples;
- exact or certified VJP through equilibrium and field evaluation;
- LASYM spectra preserved;
- batched field evaluation;
- unit/coordinate conventions explicit;
- coil objects remain owned by ESSOS.

Update the existing VMEX-ESSOS workflow rather than adding a parallel example family.

### 19.5 GKX

Coordinate with GKX PR #86 and VMEX PR #194.

- Replace generated coefficient-file bridges with an in-memory mirror/hybrid geometry constructor after parity is established.
- Keep the frozen baseline as an independent regression oracle.
- Expand in stages:
  1. circular straight mirror streaming case;
  2. shaped/nonaxisymmetric fixed-boundary mirror;
  3. periodic hybrid local geometry;
  4. electromagnetic/finite-gradient physics already supported by GKX;
  5. selected turbulence metric in the hybrid optimization loop.
- VMEX supplies dimensional geometry and equilibrium derivatives; GKX owns gyrokinetic normalization and simulation semantics.

### 19.6 DKX

DKX already owns `FluxSurfaceGeometry.from_fourier`.

- Convert `BoozerData` directly to the Fourier arrays expected by DKX.
- Include both symmetric and asymmetric harmonics.
- Provide radial derivatives from the native equilibrium/Boozer plan when needed.
- Validate against DKX's existing SFINCS-compatible file reader on the same equilibrium.
- Add a mirror/hybrid path only when DKX's neoclassical model is physically applicable to the topology. Do not feed an open field line into a closed-surface DKE by relabeling it.

### 19.7 Failure and capability contracts

Each adapter declares:

- supported topology;
- symmetry support;
- pressure support;
- derivative support;
- required coordinates;
- whether it is exact, sampled, or approximate.

Unsupported combinations raise typed errors.

### 19.8 End-to-end derivative tests

At least one scalar objective through each supported chain:

```text
boundary -> polished VMEX -> BOOZ_XFORM_JAX -> DKX objective
boundary -> polished VMEX -> VIRTUAL_CASING_JAX objective
boundary/coils -> VMEX -> ESSOS field/loss surrogate
hybrid parameters -> VMEX -> GKX geometry objective
```

Use directional finite differences or an independent tangent reference where the downstream model is smooth.

---

## 20. Phase 13 - Simplify and slim the code after ownership is stable

### 20.1 Build an ownership graph

Generate a lightweight report of:

- public symbols and import owners;
- repeated equations/kernels;
- wrappers that only translate data;
- modules importing optional dependencies;
- duplicated state packing/unpacking;
- duplicated Boozer, field, profile, and objective setup;
- circular imports or private cross-module calls.

Review giant modules by responsibility, not by line count.

### 20.2 High-value simplification targets

#### Boozer/QI

- make BOOZ_XFORM_JAX the canonical full transform;
- retain VMEX's lightweight symmetric path only with a demonstrated inner-loop advantage;
- consolidate mode/table/sign conversion;
- keep QI physics in VMEX and transform numerics in BOOZ_XFORM_JAX.

#### Implicit/optimization

- separate solver-independent response reports/configuration from objective-specific orchestration;
- remove duplicate scalar and vector response policies after one certified implementation serves both;
- centralize failure/retry/certificate semantics;
- avoid separate wrappers for each external optimizer when a common callable/Jacobian object suffices.

#### Free boundary

- separate external-field/vacuum response construction from iteration orchestration;
- share boundary packing, Schur, and certificate code between fixed/free/single-stage paths;
- keep topology-specific physics separate.

#### Plotting

- separate numerical diagnostic computation from Matplotlib rendering;
- cache/share confinement and Boozer diagnostics;
- keep plotting imports lazy;
- do not move every panel into its own file.

#### Mirror splines

- share the topology-independent B-spline basis already in core where possible;
- retain mirror-specific state/boundary/axis mapping in the mirror package;
- delete duplicate derivative/interpolation implementations after exact parity.

### 20.3 Refactor gate

For each refactor PR:

- public API unchanged or one documented deprecation;
- net source LOC non-positive unless a small compatibility shim is required;
- same or lower import time;
- no compile-count increase;
- performance within 5% or improved;
- all physics and derivative tests unchanged;
- docs shorter and clearer.

### 20.4 Documentation structure

Keep:

```text
docs/tutorials/       first successful workflows
docs/howto/           concrete tasks
docs/explanation/     equations, models, algorithms
docs/reference/       APIs, inputs, outputs, capabilities, performance
```

Add or revise:

- high-order force balance and input directives;
- performance methodology;
- mirror boundary conditions;
- anisotropic mirror theory;
- stellarator-mirror hybrid theory and limitations;
- Gamma-c derivative semantics;
- native downstream interfaces;
- LASYM capability/performance.

### 20.5 Examples follow one pattern

Each example should resemble SIMSOPT's deliberate scripts:

```text
1. short module docstring stating the physics result;
2. imports;
3. user-editable parameters at the top;
4. object construction;
5. solve;
6. objective/optimization if relevant;
7. concise printed results;
8. plots/save;
9. no hidden command-line parser, main(), or __main__ guard.
```

Provide `FAST = True` near the top when CI and research resolutions differ. Tests run the fast path; docs state the research settings.

---

## 21. Phase 14 - Final validation, optimized configurations, and README evidence

### 21.1 Mandatory cross-code equilibrium suite

Use at least:

1. analytic Solovev/D-shaped axisymmetric case;
2. smooth finite-beta QA;
3. finite-beta QH;
4. LASYM finite-beta stellarator;
5. high-aspect-ratio near-axis-sensitive case;
6. one free-boundary toroidal case;
7. isotropic mirror;
8. anisotropic axisymmetric mirror;
9. nonaxisymmetric mirror;
10. periodic hybrid.

Not every code supports every topology. The table must mark unsupported rather than substituting a different problem.

### 21.2 Force-balance plot

Generate one reproducible README figure with, for common supported toroidal cases:

- VMEC2000;
- VMEC++;
- DESC;
- legacy VMEX;
- polished VMEX.

Recommended panels:

1. flux-surface RMS strong-force profile;
2. near-axis zoom;
3. volume L2 bars;
4. optional refinement convergence.

Use the same independent oracle and clearly state excluded regions, normalization, solve resolution, and evaluation resolution.

**README debts (standing checklist, audited 2026-08-31).** These stay in
the plan until each is closed by a merged PR:

1. Comparison figure: accuracy-only panels (no runtime bars) - CLOSED
   (#215).
2. Comparison rows: every row shows certified polishing measurably beating
   the legacy codes' exported equilibria; the near-tie QA row was removed
   (#215) and the figure shows the shaped-tokamak row. OPEN: a
   finite-beta stellarator row joins once 3-D polishing certifies at
   production resolution (the capture stall is fixed by #234; the
   remaining gap is polish effectiveness and memory at MPOL=NTOR=10, see
   8.8).
3. Polish summary figure (``readme_polish_summary.webp``): the shown case
   must be a true tokamak or stellarator where polishing reduces the
   certified independent error MULTIPLE-FOLD, plotting the quantity the
   oracle measures - CLOSED by the shaped tokamak (26-fold on the exported
   equilibrium, both flux-surface sets overlaid; #237, #239). A stellarator
   variant remains desirable once item 2's row exists.
4. Prose: no README paragraph exceeds ~6 lines; panel-by-panel and
   option-by-option detail lives in the docs, linked, not inlined - CLOSED
   (audited 2026-09-02, every prose paragraph at or under six lines).

**README figure policy (decided 2026-08-31).** The README comparison shows
force-balance accuracy only. Runtime panels stay out of the README until
VMEX end-to-end times are competitive with the legacy codes on the shown
cases while staying differentiable and certified - runtime evidence lives in
``benchmarks/baselines/`` and PR bodies meanwhile, and the performance
program (compile reuse, persistent caching, captured-constant elimination)
keeps driving at that bar. Every row must be a case where certified
polishing measurably beats the legacy codes' exported equilibria under the
shared oracle; a row where the polished error merely matches VMEC2000 is not
evidence and must be replaced, and at least one row is a finite-beta or
finite-current stellarator (QA/QH class).

### 21.3 Claim gate

The README may say "VMEX matches or exceeds DESC force balance at a fraction of the cost" only if:

- the statement is true across the mandatory smooth fixed-boundary comparison set, not one case;
- both cold end-to-end and warm repeated costs are reported;
- peak memory is reported;
- the same force metric and region are used;
- input conversion is documented;
- all raw results and scripts are committed or available as hashed artifacts.

Otherwise use a narrower, factual statement such as:

> On the listed cases and common independent force metric, polished VMEX reduces the legacy VMEC residual and is competitive with DESC; see the exact per-case table.

### 21.4 Optimized configurations

Commit or publish reproducible small artifacts for:

- omnigenity optimization using epsilon effective, derivative-safe Gamma-c, and maximum-J;
- meaningful LASYM stellarator;
- nonaxisymmetric fixed-boundary mirror;
- anisotropic circular-coil free-boundary mirror;
- nonaxisymmetric free-boundary mirror;
- periodic hybrid with coils and downstream validation.

Each artifact includes:

- input and optimized parameters;
- equilibrium/native output;
- force/refinement certificate;
- derivative certificate for the optimization objective;
- objective history;
- machine-readable summary;
- provenance and license.

### 21.5 Performance table

README/docs table separates:

- cold empty-cache;
- cold persistent-cache reload;
- warm same-process;
- value;
- gradient;
- peak host/device memory.

Do not place machine-specific absolute numbers in prose without the platform and date.

### 21.6 Release gate

- all PRs merged in dependency order;
- no branch-only dependency;
- package versions and minimum versions correct;
- source and wheel tests;
- docs/linkcheck;
- coverage threshold;
- benchmark smoke and external oracle workflows;
- native-file schema versioned;
- capability matrix current;
- release notes state limitations, especially Gamma-c topology and mirror end physics.

---

## 22. Exact repository and file change map

This is a target map, not permission to create every listed file. Reuse an existing cohesive owner when that produces less code.

### 22.1 VMEX

#### Polishing integration

- `vmex/core/run_options.py` - VMEX comment/JSON execution metadata.
- `vmex/core/input.py` - allow reserved `_vmex` stripping; keep physics schema strict.
- `vmex/core/cli.py` - CLI controls, final-stage dispatch, reports, output.
- `vmex/core/multigrid.py` - one final polish hook and reuse of finest-grid static data.
- `vmex/core/solver.py` - retain PR #192 direct API; no second polish implementation.
- `vmex/core/wout.py` - polished sampling and nonstandard provenance attributes.
- `vmex/core/polish*.py`, `radial_basis.py`, `strong_force.py` - hardening, not redesign.
- `vmex/core/native_io.py` only if native serialization cannot remain cohesive in the polished-equilibrium owner.
- `vmex/__init__.py` - expose `solve_file`, run options, and stable polished types.

Tests:

- `tests/test_run_options.py`;
- `tests/test_cli_polish.py`;
- `tests/test_polish_output.py`;
- expand PR #192 tests rather than duplicate them.

#### Performance

- extend `tools/profile_hotpaths.py` or replace overlapping scripts with `benchmarks/profile_workflows.py`;
- `benchmarks/results/` schemas/manifests;
- narrow measured edits in `solver.py`, `multigrid.py`, `implicit.py`, `optimize.py`, `preconditioner*.py`, `freeboundary*.py`, and transform modules;
- `docs/reference/performance.rst` and one methodology page.

#### Confinement and objectives

- `vmex/core/plotting.py` - combined epsilon/Gamma-c panel and cache.
- `vmex/core/neoclassical.py` - no implicit global cache clearing; native Boozer input.
- `vmex/core/gammac.py` - keep hard value; tracked/smooth implementation or split only if the module becomes clearer.
- `vmex/core/bounce.py` - generic tracked-root/topology support if required.
- `vmex/core/maxj.py`, `omnigenity.py`, `qi.py` - shared context and example integration, not separate Boozer copies.
- `examples/optimization/omnigenity_epsilon_gammac_maxj.py`.

#### LASYM

Likely measured edits:

- `vmex/core/transforms.py`;
- `vmex/core/forces.py`;
- `vmex/core/nyquist.py`;
- `vmex/core/boozer_tables.py`;
- `vmex/core/implicit.py` / `optimize.py` chunking and response reuse;
- LASYM examples and tests.

#### Mirrors and hybrids

- `vmex/mirror/model.py` - topology and pressure-model configuration.
- `vmex/mirror/forces.py` - anisotropic weak/strong residuals.
- `vmex/mirror/anisotropy.py` - closure, tensor, energy density, diagnostics.
- `vmex/mirror/solver.py` - one promoted nonlinear path and implicit derivatives.
- `vmex/mirror/exterior.py` - general 3-D response hardening/near-singular work.
- `vmex/mirror/free_boundary.py` - remove axisymmetric restriction through a certified coupled solve.
- `vmex/mirror/splines.py` - share core basis and retain topology mapping.
- `vmex/mirror/turbulence.py` - downstream protocol and hybrid refinements.
- `vmex/mirror/output.py` - anisotropy/interface/provenance.
- mirror examples and `tests/mirror/*`.

#### Interfaces

- `vmex/core/interfaces.py` - small structural protocols/data contracts.
- existing `virtual_casing.py`, `boozer_tables.py`, `turbulence.py`, and ESSOS seams become adapters to these contracts.

### 22.2 SOLVAX

Start from 0.20.0 and merged nonlinear least-squares/PTC/continuation features.

Only add generic capabilities exposed by profiles:

- reusable Gauss-Newton normal-preconditioner/factor state;
- GCRO-DR recycle input/output for repeated normal solves;
- multi-RHS stationarity tangent/adjoint if VMEX cannot compose existing primitives efficiently;
- bracketed implicit scalar roots for tracked bounce points, if general enough;
- compile/work counters in solver reports;
- fixed-work variants where accelerator control flow requires them.

Probable owners:

- `src/solvax/least_squares.py` or current nonlinear least-squares module;
- `src/solvax/krylov.py`;
- `src/solvax/implicit.py`;
- tests, docs, benchmark cases.

Do not add mirror, Boozer, or VMEX state types.

### 22.3 BOOZ_XFORM_JAX

- `src/booz_xform_jax/jax_api.py` - stable config/plan and promoted kernel.
- optional `src/booz_xform_jax/plan.py` if plan construction dominates and separation reduces code.
- `src/booz_xform_jax/core.py` - compatibility wrapper and removal of duplicate conversions.
- `tools/profile_jax_api.py` - expand to matrix, derivatives, compile count, memory.
- tests for plan caching, chunking, LASYM, JVP/VJP, and legacy parity.
- docs for execution/memory policy.

### 22.4 ESSOS

Coordinate rather than duplicating open work:

- ensure VMEX symmetric/LASYM/native fields can be constructed in memory;
- shaped/rotated mirror surfaces and hybrid surfaces;
- profiled batched coil fields;
- loss-fraction objective/surrogate with explicit derivative semantics;
- consumer-side adapters and tests.

Inspect open PRs before editing, particularly current VMEC-asymmetry, mirror-surface, and coil-cache work.

### 22.5 VIRTUAL_CASING_JAX

- accept VMEX native boundary/field arrays directly;
- preserve high-order derivatives and LASYM;
- add native-versus-WOUT parity and derivative tests;
- use existing external benchmark/provenance infrastructure.

### 22.6 GKX

- coordinate with current VMEX mirror pilot;
- add in-memory geometry construction after file-oracle parity;
- keep GKX normalization and simulation configuration in GKX;
- add hybrid geometry and selected turbulence benchmark only after equilibrium certificates pass.

### 22.7 DKX

- add a `BoozerData` adapter around `FluxSurfaceGeometry.from_fourier`;
- preserve sine/cosine and radial derivatives;
- compare in-memory and SFINCS-compatible file paths;
- document topology limits.

### 22.8 DESC, VMEC2000, VMEC++

These are comparison/oracle repositories unless a clearly independent bug is found.

- Keep local patches minimal and documented.
- Do not modify an oracle to match VMEX conventions without retaining an unmodified reference run.
- Record exact commits.
- Use DESC mirror PR #1848 as an experimental source of cases/equations, not a dependency.

---

## 23. Test and validation matrix

### 23.1 Coverage policy

- at least 95% line coverage on every new/changed source file;
- repository threshold does not decrease;
- every typed failure branch has a test;
- every public example is import/run tested in fast mode;
- slow external oracles run in scheduled CI with small smoke subsets in ordinary CI;
- coverage is not increased with meaningless execution-only tests.

### 23.2 Polishing tests

| Category | Required test |
|---|---|
| Parsing | INDATA directive and JSON `_vmex` |
| Legacy | same deck runs in xvmec2000 with directive ignored |
| API | `solve`, `solve_file`, CLI equivalence |
| Backward compatibility | `polish=False` regression |
| Analytic | Solovev strong force |
| Cross-code | VMEC2000, VMEC++, DESC |
| Resolution | radial degree/span and angular/collocation validation |
| Admissibility | positive Jacobian, boundary, nestedness |
| Linear | true residual and preconditioner certificate |
| AD | tangent/adjoint duality, Taylor, VJP, finite difference |
| I/O | WOUT and native round trip |
| Downstream | field, Boozer, VC, ESSOS |

### 23.3 Performance tests

CI should not enforce noisy absolute wall times. Enforce deterministic proxies:

- trace count;
- compile count for repeated same-shape calls;
- no unexpected cache miss reason;
- buffer/shape model for chunking;
- maximum retained history/state size;
- no unbounded surface batching;
- benchmark schema validity.

Scheduled benchmarks track wall time/memory with control charts and alert thresholds.

### 23.4 Gamma-c tests

- hard value parity with current implementation before refactor;
- single-well analytic integral;
- tracked root derivatives;
- topology matching and events;
- smooth-temperature convergence;
- resolution convergence of value and derivative;
- correlation suite with hard value and ESSOS losses;
- LASYM symmetry/reflection invariants;
- overflow and incomplete-well behavior.

### 23.5 LASYM tests

- mode and phase normalization;
- all parity families through geometry, force, WOUT, Boozer, and adapters;
- symmetric limit;
- controlled small asymmetry linear response;
- VMEC2000 parity;
- strong force;
- stability and bootstrap;
- gradient/Jacobian;
- no same-shape recompilation;
- optimized configuration reproducibility.

### 23.6 Mirror tests

| Layer | Isotropic | Anisotropic | Free boundary | Hybrid |
|---|---:|---:|---:|---:|
| analytic geometry | yes | yes | yes | yes |
| weak energy derivative | yes | yes | yes | yes |
| independent strong force | yes | yes | yes | yes |
| divergence | yes | yes | yes | yes |
| axis regularity | yes | yes | yes | yes |
| cut BC | yes | yes | yes | n/a |
| interface BC | n/a | n/a | yes | n/a |
| pressure closure | n/a | yes | yes | optional |
| exterior field | n/a | n/a | yes | optional coils |
| JVP/VJP | yes | yes | yes | yes |
| CPU/GPU | yes | yes | yes | yes |
| refinement | yes | yes | yes | yes |
| external/literature oracle | yes | yes | yes | downstream |

### 23.7 Downstream tests

- in-memory versus file-backed numerical parity;
- coordinate and unit round trips;
- LASYM retention;
- native high-order versus sampled WOUT convergence;
- gradient through each supported chain;
- unsupported topology fails explicitly;
- no consumer imports required for base VMEX installation.

---

## 24. Benchmark and provenance rules

### 24.1 Never edit result JSON by hand

All results are emitted by scripts. Store:

- schema version;
- UTC timestamp;
- git commits and dirty status;
- input/config hash;
- dependencies;
- platform/device;
- precision;
- static shapes;
- compile/cache state;
- command;
- metrics and units.

### 24.2 Warm-up and asynchronous execution

- always call `block_until_ready()` around measured JAX work;
- separate warm-up from timed repeats;
- report median and spread for warm microbenchmarks;
- cold runs use new processes;
- persistent-cache reload uses a second new process.

### 24.3 External-code fairness

- retain original code outputs;
- record conversion scripts;
- use identical physical normalization;
- show both code-native convergence and common independent metrics;
- do not include compilation in one code and exclude it in another without a separate table;
- report CPU/GPU differences rather than comparing unlike hardware as an algorithm result.

### 24.4 Assets

- small CSV/JSON summaries may be committed;
- large WOUT/native/trace data should use release assets or the existing VMEX asset-fetch mechanism;
- every asset has SHA256, origin, license/permission, generating commit, and script.

---

## 25. PR sequence and dependencies

Keep each PR reviewable. Suggested sequence:

1. **Merge PR #192** - reproduce, narrow claims, clean production path.
2. **Polish input/API integration** - directives, `solve_file`, CLI, output semantics.
3. **Unified performance harness** - no performance edits yet.
4. **Measured VMEX core/JIT performance fixes** - split by bottleneck if needed.
5. **BOOZ_XFORM_JAX plan/chunk refactor**.
6. **VMEX Boozer ownership cleanup** after BOOZ_XFORM_JAX parity/performance.
7. **Summary Gamma-c quick plot**.
8. **Gamma-c tracked/smooth derivatives**.
9. **Combined epsilon/Gamma-c/max-J example**.
10. **LASYM performance kernels**.
11. **LASYM flagship optimization and downstream audit**.
12. **Mirror boundary-condition derivation and isotropic hardening**.
13. **Anisotropic pressure model and fixed-boundary solver**.
14. **Axisymmetric anisotropic circular-coil free boundary**.
15. **Nonaxisymmetric mirror exterior/coupled solve**.
16. **Hybrid equilibrium/implicit derivative hardening**.
17. **Hybrid optimization and ESSOS coupling**.
18. **GKX/DKX/native downstream integrations**.
19. **Measured simplification/refactor PRs**.
20. **Final benchmark/README/release PR**.

A downstream PR may proceed in parallel once its upstream data contract is frozen, but do not merge code depending on an unmerged branch without an explicit temporary pin and removal plan.

---

## 26. Milestone gates

### Milestone A - Polished toroidal equilibrium is a normal VMEX feature

- PR #192 merged;
- directive and Python API;
- WOUT/native output;
- independent force and derivative certificates;
- no-polish parity;
- fixed-boundary documented scope.

### Milestone B - Performance is understood and improved

- full profiling matrix;
- compile misses explained;
- repeated workflows do not recompile unnecessarily;
- targeted cold/warm/memory improvements;
- BOOZ_XFORM_JAX explicit memory plan.

### Milestone C - Confinement optimization is derivative safe

- combined summary plot;
- tracked/smooth Gamma-c;
- gradient convergence;
- combined omnigenity example;
- ESSOS validation.

### Milestone D - LASYM is first-class

- performance and accuracy matrix;
- meaningful asymmetric configuration;
- downstream capability table;
- independent certificates.

### Milestone E - Research-grade mirrors

- BC derivation;
- anisotropic closure;
- fixed-boundary nonaxisymmetric optimization;
- axisymmetric anisotropic circular-coil free boundary;
- nonaxisymmetric free boundary.

### Milestone F - Research-grade hybrid

- equilibrium and optimization;
- force/closure certificates;
- ESSOS coils/losses;
- GKX and DKX results;
- reproducible flagship artifact.

### Milestone G - Release evidence

- cross-code README plot;
- benchmark tables;
- optimized configurations;
- code simplification complete;
- coverage/docs/release gates.

---

## 27. Stop conditions and anti-patterns

Stop and reassess when:

- polishing lowers solve-grid residual but raises independent validation residual;
- a preconditioner lowers iterations but raises total time or memory;
- a gradient changes sign under routine resolution refinement;
- a branch event is hidden by smoothing rather than reported;
- an anisotropic closure violates parallel integrability;
- a cap is accidentally treated as a plasma-vacuum interface;
- a hybrid downstream calculation assumes a topology it does not have;
- LASYM improvement disappears after gauge/phase alignment;
- a performance win exists only after excluding compilation for VMEX but not the comparator;
- a refactor creates more wrappers than code it removes;
- a README claim depends on one favorable case.

Forbidden shortcuts:

- hard-coded case-specific coefficients in production;
- silent symmetrization;
- optimization against the same implementation used as the only validation oracle;
- finite differences through unconverged roots;
- differentiating taped nonlinear iterations when an implicit equation exists;
- global cache clearing in reusable library functions;
- unbounded dense Jacobians/tensors at production resolution;
- arbitrary independent anisotropic pressures;
- claiming global optimality for a nonconvex stellarator design.

---

## 28. Disposition of the former plan

Carry forward:

- independent force certificates;
- fail-closed derivative semantics;
- high-order native equilibrium;
- SOLVAX structured algebra;
- VMEC2000/VMEC++/DESC comparisons;
- CPU/GPU/memory evidence;
- native ESSOS/VC/Boozer connections;
- more than 95% changed-code coverage;
- concise examples and deliberate code;
- provenance and claim gates.

Replace with current reality:

- polishing is already largely implemented in PR #192;
- SOLVAX already has the required Gauss-Newton/PTC/continuation foundations;
- mirror B-splines and independent pointwise force already exist;
- periodic hybrid GK geometry is merged;
- Gamma-c's derivative failure is known and must be redesigned;
- LASYM has broad physics support but still needs measured performance and a flagship result;
- the main remaining work is integration, hardening, performance, anisotropic/free-boundary mirrors, hybrid optimization, and downstream consolidation.

Do not copy the old historical checkbox ledger into this file. Link merged PRs and benchmark artifacts instead.

---

## 29. Primary references and required reading

### 29.1 VMEC, force balance, radial accuracy, and preconditioning

- S. P. Hirshman and J. C. Whitson, *Steepest-descent moment method for three-dimensional magnetohydrodynamic equilibria*, Phys. Fluids 26, 3553 (1983), https://doi.org/10.1063/1.864116
- S. P. Hirshman and O. Betancourt, *Preconditioned descent algorithm for rapid calculations of magnetohydrodynamic equilibria*, J. Comput. Phys. 96, 99-109 (1991), https://doi.org/10.1016/0021-9991(91)90267-O
- J. Schilling, *The Numerics of VMEC++*, arXiv:2502.04374, https://arxiv.org/abs/2502.04374
- D. Panici et al., *The DESC stellarator code suite. Part 1. Quick and accurate equilibria computations*, J. Plasma Phys. 89, 955890303 (2023), https://doi.org/10.1017/S0022377823000272
- F. Hindenlang et al., *GVEC: A flexible 3D MHD equilibrium solver*, JOSS 11, 9670 (2026), https://doi.org/10.21105/joss.09670
- VMEC2000/STELLOPT source and documentation.
- VMEC++ source and numerics notes.

### 29.2 Nonlinear solves and differentiation

- C. T. Kelley and D. E. Keyes, *Convergence analysis of pseudo-transient continuation*, SIAM J. Numer. Anal. 35, 508-523 (1998), https://doi.org/10.1137/S0036142996304796
- S. C. Eisenstat and H. F. Walker, *Choosing the forcing terms in an inexact Newton method*, SIAM J. Sci. Comput. 17 (1996), https://doi.org/10.1137/0917003
- PETSc pseudo-transient method notes: https://petsc.org/release/manualpages/TS/TSPSEUDO/
- SOLVAX documentation and current tests.

### 29.3 Anisotropic equilibrium and mirrors

- G. F. Chew, M. L. Goldberger, F. E. Low, *The Boltzmann equation and the one-fluid hydromagnetic equations in the absence of particle collisions*, Proc. R. Soc. A 236 (1956).
- W. A. Cooper et al., *3D magnetohydrodynamic equilibria with anisotropic pressure*, Comput. Phys. Commun. 72, 1-13 (1992), https://doi.org/10.1016/0010-4655(92)90002-G
- W. A. Cooper et al., *Three-dimensional anisotropic pressure free boundary equilibria*, Comput. Phys. Commun. 180, 1524-1533 (2009), https://doi.org/10.1016/j.cpc.2009.04.006
- D. Endrizzi et al., *Physics basis for the Wisconsin HTS Axisymmetric Mirror (WHAM)*, J. Plasma Phys. 89 (2023), https://doi.org/10.1017/S0022377823000806
- S. J. Frank et al., *Nonlinear anisotropic equilibrium reconstruction in axisymmetric magnetic mirrors*, arXiv:2509.17288
- ANIMEC source, manuals, and reference cases.
- DESC mirror PR #1848: https://github.com/PlasmaControl/DESC/pull/1848

### 29.4 Omnigenity, Gamma-c, and fast ions

- J. R. Cary and S. G. Shasharina, *Helical plasma confinement devices with good confinement properties*, Phys. Rev. Lett. 78, 674 (1997), https://doi.org/10.1103/PhysRevLett.78.674
- V. V. Nemov et al., *Poloidal motion of trapped particle orbits in real-space coordinates*, Phys. Plasmas 15, 052501 (2008), https://doi.org/10.1063/1.2912456
- J. L. Velasco et al., *A model for the fast evaluation of prompt losses of energetic ions in stellarators*, Nucl. Fusion 61, 116059 (2021), https://doi.org/10.1088/1741-4326/ac2994
- K. Unalmis et al., *Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm and its applications*, J. Plasma Phys. 92 (2026), https://doi.org/10.1017/S0022377826101652
- I. E. Ochs, *Bounce-averaged theory in arbitrary multi-well plasmas*, J. Plasma Phys. (2025), https://doi.org/10.1017/S002237782510069X
- E. Sanchez et al., *A quasi-isodynamic configuration with good confinement of fast ions at low plasma beta*, Nucl. Fusion 63, 066037 (2023).
- Piecewise-omnigenity and current direct-J optimization literature available at implementation time; record exact versions used.

### 29.5 Boozer coordinates and downstream tools

- BOOZ_XFORM/STELLOPT source and Hirshman-Breslau documentation.
- BOOZ_XFORM_JAX repository/docs: https://github.com/uwplasma/booz_xform_jax
- ESSOS: https://github.com/uwplasma/ESSOS
- VIRTUAL_CASING_JAX: https://github.com/uwplasma/virtual_casing_jax
- GKX: https://github.com/uwplasma/GKX
- DKX: https://github.com/uwplasma/DKX

### 29.6 JAX performance

- Benchmarking: https://docs.jax.dev/en/latest/benchmarking.html
- Slow tracing/compilation: https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html
- Persistent compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- Profiling/XProf: https://docs.jax.dev/en/latest/profiling.html
- Device memory: https://docs.jax.dev/en/latest/device_memory_profiling.html
- GPU performance tips: https://docs.jax.dev/en/latest/gpu_performance_tips.html
- Pallas: https://docs.jax.dev/en/latest/401/pallas.html

---

## 30. Final instructions to the implementing agent

Work until the current phase's acceptance criteria are met, not merely until code exists.

At the end of every PR:

1. run focused tests;
2. run full repository tests appropriate to the change;
3. verify changed-code coverage above 95%;
4. run the declared physics oracle;
5. run value and derivative certificates;
6. record cold/warm/memory evidence when performance is affected;
7. update docs, capability tables, examples, and release notes;
8. remove superseded experiments and dead code;
9. summarize remaining limitations plainly.

The final outcome should not be a larger collection of optional algorithms. It should be a smaller number of trusted paths:

- one robust VMEC-compatible branch finder;
- one certified high-order toroidal polish;
- one coherent mirror equilibrium framework with isotropic and anisotropic closures;
- one periodic hybrid framework;
- one canonical full Boozer transform;
- one derivative-safe confinement-objective stack;
- one native downstream data contract;
- one reproducible performance and physics evidence system.

---

## 31. Final review ledger (2026-09-02) - read this first if you are the next agent

This section records the state at the end of the 2026-09-02 review: what
landed, what is in flight, and every finding of the four literature and
publication audits (core equilibrium/polish physics, diagnostics and
objectives, mirror/hybrid physics, publication artifacts), with the concrete
fix for each. Findings are numbered so pull requests can cite them
(`31.x-Rn`). Nothing here is done until a merged PR closes it.

### 31.1 State of the repository

Merged since v0.8.1 (2026-09-02): #226 guard repairs; #227 cold-start
restoration; #228 CI cache off on runners; #229 iteration body traced once,
ns4 cadence; #230 eager field-chain lanes; #231 recompile keys; #232 scalar
non-finite guard; #233 compile budgets + auto chunk; #234 3-D polish capture
fix; #236 manifest row; #237/#239 README polish figures and extender islands;
#238 trace-budget carry; #240 optimization startup (62 -> 6.5 s cold to
first output; `_refine_step_core`); #241 staging idioms universal + CI pin
on optimization cold start; #242 Codecov best-effort so diff-cover runs;
#243 polish banner, GN rows, certificate verdict, `POLISH_*` directives;
#244 polish phase notices. v0.8.1 is on PyPI (the 0.8.0 MPOL=NTOR=10
polish OOM is fixed there; users must upgrade).

Open PRs: #245 (CITATION.cff, CHANGELOG, README prose policy, section 21
debts), #246 (benchmark harness records gates instead of aborting `--all`;
`--regimes` defaults to `warm`, pass all five for the section 9 matrix),
#247 (fresh-deck xvmec2000 table with the hashed artifact
`benchmarks/fresh_decks_vs_vmec2000_2026-09-02.json` and a numerical
reproducibility section), #197 (contributor's scalar-adjoint examples;
verdict in 31.6).

Branches in flight (agents were working on them when this ledger was
written; check `git log origin/<branch>` before assuming completion):
`perf/polish-3d-effectiveness` (W7-X polish memory: the un-chunked
certificate allocates 34.5 GB at MPOL=NTOR=10; a batched certificate ran at
7.8 GB peak; a rematerialized GN run was in progress on the office box
under `ulimit -v 50 GB`, log `~/polish3d/run5_w7x_remat.log`);
`perf/booz-xform-jax-0-2` (BoozerConfig/BoozerPlan adoption to recover the
5x adapter cost from #224); `docs/publication-hygiene` (31.5 items A4-A8,
A24-A27); `docs/api-reference-completeness` (31.5 A11-A12);
`docs/diagnostics-citations` (31.3); `docs/figure-provenance` (31.5 A16,
section C claims, the validation page). Parked, measured neutral, do not
merge: `perf/fused-constraint-synthesis` (bit-identical, no runtime gain;
row fusion into the batched contraction breaks parity at mnmax >= 162).

Measured baselines to cite: cold CLI on Apple M4 (in-process): QA_lowres
12.3-12.6 s, li383 4.0 s, solovev 3.1 s (0.3.0: ~15.3/5.6/2.9; 0.8.0: ~21
at QA); office x86 subprocess: QA 43.1 (0.3.0) -> 32.4 s, solovev 12.1 ->
7.0; six fresh decks vs xvmec2000 in #247 (warm parity to 2x faster, wout
maxima 1.4e-10 relative). These are in PR/release bodies; #247 is the first
hashed artifact; the section 9 matrix must be rerun on 0.8.1+ with all five
regimes after #246 merges.

### 31.2 Core equilibrium and polishing physics audit (REQUIRED)

- 31.2-R1 The certificate metric `eps_F = 2|F| / (|J x B| + |grad p| + F_floor)`
  (`strong_force.py:28,765-771`) saturates pointwise at 2 wherever
  `|J x B| + |grad p| << |F|` and is undefined in vacuum (grad p = 0 gives
  eps = 2 identically wherever J != 0). Evidence in
  `benchmarks/strong_force_cases_m4.json`: nfp2_QA_finite_beta VMEC2000 row
  has normalized_linf 1.99999999, p99 1.9993, near_axis_l2 85x bulk_l2. Fix:
  add the published global normalizations - Panici et al. 2023 Eqs. 32-34b
  `<|F|>_vol / <|grad p|>_vol` on s in [0.1, 0.99], and the vacuum-safe DESC
  form `|F| / <|grad(B^2/2mu0)|>_vol` (DESC `_equil.py:786-802`; Thun et al.
  2026 Eq. 42) - report the dimensional `<|F|>_vol`, disclose the saturation,
  and report the near-axis/bulk/edge split. Files: `strong_force.py`,
  README polish section, `docs/explanation/high-order-force-balance.rst`.
- 31.2-R2 The DESC row of the comparison figure measures VMEX's spline lift
  of a DESC-re-exported 129-surface wout (`run_external_equilibrium.py:46-76`,
  `strong_certificate.py:139-143`), not DESC's force balance (Panici 2023
  Fig. 7, Thun 2026 Sec. 4.2 put DESC orders of magnitude below VMEC).
  Evaluate DESC natively (`compute("F","|F|","sqrt(g)")` with the same
  denominator) or drop the row; caption VMEC2000/VMEC++ rows as "wout lifted
  by VMEX splines at degree d, k spans"; show the radial profile.
- 31.2-R3 The "26-fold" (5.05e-2 -> 1.91e-3) mixes ~7-9x polish gain with a
  2-4x export-mesh/lift reconstruction difference: VMEC2000 and VMEC++ wouts
  of the same deck certify at 1.71e-2, VMEX's own lifted native state at
  1.28e-2 (initial certificate), and the solve-mesh export at 3.3e-2 vs the
  ns=129 export at 1.9e-3 (`polish_driver.py:281-297`). Recompute with
  identical lift and export settings for before/after, or quote the native
  pair 1.28e-2 -> 1.79e-3 and name the metric.
- 31.2-R4 `docs/explanation/high-order-force-balance.rst:105-289` describes
  the retired square homotopy/PTC/pseudo-arclength root with a
  tangential-displacement gauge; production is
  `polish_legacy_solution -> polish_collocation_least_squares`
  (`polish_driver.py:1151-1447`): rectangular residual on composite Gauss
  nodes, Gauss-Newton/LM via solvax, 8 random transpose probes for column
  scales, and a chart that freezes Z_sin entirely ("Z is the eliminated
  poloidal-coordinate gauge", `polish.py:1551-1595`; the map delta = dZ/Z_theta
  is singular where Z_theta = 0, so corrections are purely horizontal - test
  whether that sets the ~1.8e-3 floor). The collocation functional carries
  |sqrt g| but no quadrature weights (`polish.py:1344-1354`), so it is not the
  certificate norm (DESC-style; state it or add sqrt(w)); the denominator is
  frozen with floor 1e-30 vs the certificate's 1e-12; the tokamak artifact
  exhausted its 80-iteration budget and was accepted by certificate alone.
  Rewrite the page to the shipped method and mark the homotopy material
  retired.
- 31.2-R5 No exact-solution validation of the oracle: only a vacuum 1/R field
  (`tests/test_strong_force.py:76-96`); `input.solovev` is a VMEC solve, not
  the analytic Solov'ev. Add an analytic Solov'ev equilibrium with known
  J x B = grad p and a spline/Fourier refinement convergence plot.
- 31.2-R6 Selection bias: README "the figure shows only cases where polishing
  demonstrably wins" and the section 21.2 policy. Section 21.3 already
  forbids paper claims outside the mandatory set; reword the README now and
  never present a selected-row figure as evidence.
- 31.2-R7 Possible LASYM parity defect: VMEC2000 halves the whole `tcon`
  array for LASYM (`bcovar.f:452,887`, `IF (lasym) tcon = p5*tcon`) in
  addition to `alias.f:69-75`; VMEX reproduces only the alias factor
  (`forces.py:565`; `fields.constraint_scaling` has no lasym argument,
  `solver.py:1198-1201`). Run `up_down_asymmetric_tokamak` with and without
  the halving against the VMEC2000 threed1 trajectory; fix or document in
  `docs/reference/vmec2000-compatibility.rst`.
- 31.2-R8 Convention-page errors: `spectral-representation.rst:329-337` puts
  2pi/signgs on the wrong term in B^u/B^v (code: `fields.py:360-372`,
  `bcovar.f:168-169`); `:339-342` claims a 1/NFP zeta conversion that does
  not exist (zeta tables are already physical-angle derivatives,
  `fourier.py:215-216,338-339`); `:282` "signgs = +-1" (VMEC2000/VMEC++/VMEX
  fix signgs = -1 and flip theta); `:187` omits the lasym dnorm;
  `variational-problem.rst:94-97` "every accepted step decreases W
  monotonically" is false with Richardson momentum and the condensation
  force (Hirshman-Whitson 1983 guarantees it only for first-order descent).
- 31.2-R9 `docs/howto/plot-diagnostics.md:71-72` claims the equif
  normalization is valid in vacuum; it is O(1) noise there
  (`postprocess.py:428-430`). Delete, and say on README lines ~40 and ~192
  that the radial force-error panel is meaningless for vacuum.

Recommended (31.2-C): cite Hirshman-Betancourt 1991 (preconditioner),
Hirshman-Meier 1985 (spectral condensation), Hirshman-Breslau 1998 (m=1
constraint), HvRM 1986 Sec. 2 (odd-m sqrt(s)), BCYCLIC 2010, Lewis-Bellan
1990 (analyticity rho^|m|), Dudt-Kolemen 2020, Panici 2023, Conlin 2023,
GVEC 2025 (radial B-splines, energy Galerkin - the closest precedent),
SIESTA 2011 (Hirshman's own "polish a VMEC state"), Thun 2026 (PINN
post-improvement with the standard normalization), and cite VMEC++ notes
(Schilling 2025) and VMEC2000 source by section/line for every heuristic
(tcon ramp, pdamp, FThreshold, jmin/jlam); one conventions page (signgs,
handedness, v = physical phi vs zeta = NFP phi in the high-order module,
lambda internal/wout scaling and sign, full/half-mesh table incl. lambda
jlam=2 and lmns export, pressure units, Phi' normalization); lift weighting
consistent with `polish.py:1855-1866`; one degree default (3 in
PolishConfig vs 5 elsewhere); one F_floor; rename
`radial_refinement_difference` (it is Gauss-order consistency, not
h/p-refinement) and add a true re-polish refinement check; quantify README
line ~88 (max 0.40%, median 0.06% per-iteration deviation vs VMEC2000,
VMEC++ 0.39%); "IFT-exact" not "exact" derivatives; "and Z untouched".
Verified faithful (no action): ns4, signgs/flip, lconm1 rotation and
FThreshold, mscale/nscale/dnorm/faccon, fnorm/fnormL, lamscale, tcon ramp,
precondn/lamcal factors, Jacobian tau/dshalfds, pdamp, Richardson damping,
internal energy normalization.

### 31.3 Diagnostics and objectives audit (formulas verified correct; docs REQUIRED)

- 31.3-R1 `docs/explanation/confinement.rst:144-152` inverts QA/QP contour
  topology: |B| = B(theta_B) contours wind TOROIDALLY (tokamak-like); |B| =
  B(zeta_B) contours close POLOIDALLY (QI-like). `objectives.rst:79-90` is
  right.
- 31.3-R2 "Gamma_c^2 scales the prompt-loss fraction" (`gammac.py:6,803-805`,
  `objectives.rst:336-337`, `confinement.rst:688-689`) is unsupported:
  Velasco 2021 Eq. 20 is linear in Gamma_c; Eqs. 20-21 relate the
  prompt-loss fraction approximately linearly to Gamma_c. Sum Gamma_c^2 is
  only the least-squares cost.
- 31.3-R3 `gammac.py:885-887` "smoothing biases the value slightly"
  contradicts `tests/test_gammac.py:225-231` (smooth/hard 0.02, 0.004, 0.42):
  the surrogate is a gradient objective only, never a reported number.
- 31.3-R4 `stability.py:281` cites Landreman-Jorge 2020 "Eqs. 51, 53"; the
  paper numbers by section: D_R (5.1), H (5.4), the relation (5.6); the
  implementation matches (5.6) exactly.
- 31.3-R5 `docs/project/references.rst:93-95` Redl 2021 title says
  "stellarators"; it is "... in tokamaks" (Phys. Plasmas 28, 022502).
- 31.3-R6 `confinement.rst:175` "reproduces simsopt ... bit-for-bit" has no
  simsopt oracle in tests; pin one simsopt `QuasisymmetryRatioResidual.total()`
  value with a measured tolerance, or reword to "identical formula, grid and
  weighting (Landreman-Paul 2022 Eq. 1); traceable lane gated against the
  wout lane in tests/test_optimize_traceable_qs.py".
- 31.3-R7 README roadmap line "a Gamma_c objective whose boundary derivative
  is well-posed under refinement" is stale (GammaCSmooth shipped); replace
  with a pinned DESC Gamma_c oracle assertion (2.8% agreement is only a
  docstring note, `test_gammac.py:90-94`).
- 31.3-C equation-number citations at every definition (QS: Landreman-Paul
  2022 Eq. 1, Helander 2014 for "vanishes iff"; bootstrap: Landreman-Buller-
  Drevlak 2022 Eqs. 9, 10, 11 [= Lin-Liu & Miller 1995], 12, 15, A1, A13-A14
  [= Sauter 1999 18b-18e] and Redl 2021 Eqs. 10-16, 19-21, replacing the
  nine "spec section 6.x" references and the dangling
  `notes_r26g_redl_spec.md`; Gamma_c: Velasco 2021 Eq. 16 [Nemov 2008 Eq. 61],
  footnote after Eq. 15; Boozer: Boozer 1981, Sanchez 2000, booz_xform note);
  "KNOSOS/CIEMAT-QI form" -> "Nemov/DESC form (tangency factor at B_min);
  KNOSOS uses gamma_c* without it"; `optimize.py:786` "VMEC/simsopt
  magnetic-well proxy" -> "simsopt Vmec.vacuum_well"; one `n_lambda`
  constant for `wout.py:479,630`; wout-file.rst states grid extrema and the
  measured simsopt parity (1e-3 B extrema, 3e-3 f_t); add the missing
  bibliography entries (Nemov 2008, Velasco 2021, Sauter 1999, Lin-Liu &
  Miller 1995, Boozer 1981, Sanchez 2000, Helander 2014, Glasser-Greene-
  Johnson 1975, Bader 2019/2021, Landreman-Buller-Drevlak 2022) with
  verified DOIs. Branch `docs/diagnostics-citations` was assigned all of
  31.3.
- 31.3-B Bibliography verified against Crossref/arXiv on 2026-09-02 (use
  these, the draft lists circulating in earlier prompts had errors):
  Nemov, Kasilov, Kernbichler, Leitold, Phys. Plasmas 15, 052501 (2008),
  10.1063/1.2912456. The Gamma_c prompt-loss paper is Velasco, Calvo,
  Mulas, Sanchez, Parra, Cappa, W7-X Team, "A model for the fast evaluation
  of prompt losses of energetic ions in stellarators", Nucl. Fusion 61,
  116059 (2021), 10.1088/1741-4326/ac2994 (NOT "Robust stellarator
  optimization via flat mirror magnetic fields", which is Nucl. Fusion 63,
  126038 (2023), 10.1088/1741-4326/acfe8a). Sauter, Angioni, Lin-Liu, Phys.
  Plasmas 6, 2834 (1999), 10.1063/1.873240 (erratum 9, 5140 (2002)).
  Lin-Liu & Miller, Phys. Plasmas 2, 1666 (1995), 10.1063/1.871315. Boozer,
  Phys. Fluids 24, 1999 (1981), 10.1063/1.863297. Sanchez, Hirshman, Ware,
  Berry, Spong, "Ballooning stability optimization of low-aspect-ratio
  stellarators", PPCF 42, 641 (2000), 10.1088/0741-3335/42/6/303 (the COBRA
  code paper is Sanchez, Hirshman, Whitson, Ware, JCP 161, 576 (2000),
  10.1006/jcph.2000.6514). Helander, Rep. Prog. Phys. 77, 087001 (2014),
  10.1088/0034-4885/77/8/087001. Glasser, Greene, Johnson, Phys. Fluids 18,
  875 (1975), 10.1063/1.861224. Bader et al., J. Plasma Phys. 85, 905850508
  (2019), 10.1017/S0022377819000680. Bader, Anderson, Drevlak, Faber, Hegna,
  Henneberg, Landreman, Schmitt, Suzuki, Ware, "Modeling of energetic
  particle transport in optimized stellarators", Nucl. Fusion 61, 116060
  (2021), 10.1088/1741-4326/ac2991. Landreman, Buller, Drevlak, Phys.
  Plasmas 29, 082501 (2022), 10.1063/5.0098166. Redl, Angioni, Belli,
  Sauter, Phys. Plasmas 28, 022502 (2021), 10.1063/5.0012664 (second author
  is Angioni, not Beidler). Landreman & Jorge, J. Plasma Phys. 86, 905860510
  (2020), 10.1017/S002237782000121X, equations numbered by section: D_R is
  (5.1), H is (5.4), the D_R/D_Merc relation is (5.6). Landreman & Paul,
  Phys. Rev. Lett. 128, 035001 (2022), 10.1103/PhysRevLett.128.035001; its
  Eq. (1) is the surface-summed flux-surface average of the SQUARED two-term
  residual, so cite "the residual inside Eq. (1)" for the pointwise formula.

### 31.4 Mirror and hybrid audit (REQUIRED; Phase 9 spec sheet)

- 31.4-R1 Phase 9 energy functional corrected in 16.3 above (Gamma = 0 /
  minus sign; split form). Everything downstream in 16.x must use the split
  form.
- 31.4-R2 `mirror/free_boundary.py:555,917` and `mirror/implicit.py:86,455`
  accept `current_derivative != 0`, but the exterior (single-valued decaying
  potential on a topologically spherical Green surface,
  `exterior.py:795-855,900-946`) cannot carry the azimuthal field of a net
  axial current; the interface residual then compares inconsistent fields.
  Raise on nonzero current in those entry points (or add the analytic
  phi-hat term to `lateral_field_xyz` with a stated end-electrode
  assumption); add "net axial current" to 17.3's residual list.
  **CLOSED (2026-09-03).** Confirmed a real defect, not a documentation gap:
  the interface residual is `p + |B_plasma|^2/2mu0 - |B_vac|^2/2mu0`, and the
  exterior correction field is exactly meridional at every lateral node
  (`test_exterior_vacuum_has_no_azimuthal_field_while_a_net_current_adds_one`),
  while the plasma `B^2` gains a `B_phi` term as soon as `I'(s) != 0`.
  `reject_net_axial_current` now guards `solve_free_boundary`,
  `solve_beta_scan`, `_build_free_equilibrium_problem` and
  `free_boundary_adjoint`; the limitation is listed in 17.3. `implicit.py:86`
  (`spline_fixed_boundary_parameters`) is deliberately *not* guarded - it is
  the fixed-boundary lane, solves no exterior problem, and `I'(s) != 0` there
  is legitimate (end-plate closure, or the hybrid's transform).
- 31.4-R3 Mirror ratio and mirror length are used with four inconsistent
  meanings (cut-plane on-axis ratio in `mirror_fixed_boundary_nonaxisymmetric.py:302`;
  grid max on-axis in `mirror_free_boundary_beta_scan.py:200` and
  mirror-geometry.rst:266; LCFS max/min in `qi_mirror_hybrid_fourier_vs_bspline.py:271`
  and rst:503; field-line max/min in `tests/mirror/test_turbulence.py:68`);
  `turbulence.py:293` exports std/mean of bmag under the GX/GS2 "epsilon" key
  whose meaning there is different. Define and report: R_m,axis(leg) =
  max/min of B_axis within each leg's |B| well; R_m,LCFS separately;
  L_straight = arc length where axis curvature < tol (`geometry.py:298`);
  L_mirror,B = distance between the |B| maxima bounding the well. Required by
  18.1.
  **CLOSED (2026-09-03).** `vmex/mirror/metrics.py` implements the four
  quantities once (plus persistence pruning, so ripple in a solved |B| is not
  reported as extra legs) and the three mirror examples, `test_turbulence.py`
  and the docs use it. A *fifth* meaning was found beyond the four listed:
  `vmex.core.optimize.mirror_ratio` returns the modulation depth
  `(Bmax-Bmin)/(Bmax+Bmin)`, not `R_m`; both its docstring and
  `docs/reference/objectives.rst` now say so and give `R_m = (1+m)/(1-m)`.
  The `epsilon` export was **documented, not renamed**: `vmex/core/turbulence.py`
  ships the identical `std/mean` quantity under the same key, so changing only
  the mirror lane would create a new split definition. Verified against the
  installed GKX source that GKX means the inverse aspect ratio by `epsilon`
  (`geometry/analytic.py`: `bmag = 1/(1 + eps cos theta)`;
  `artifacts/nonlinear_netcdf.py`: `aminor = eps * R0`), which is recorded in
  the `gk_closed_fieldline_geometry` docstring; the correct field-line
  quantities are exported as `vmex_mirror["field_line_mirror_ratio"]` and
  `["field_line_b_modulation"]`. Aligning both lanes' `epsilon` value is a
  follow-up that must touch the core lane too.
  **Aligned (2026-09-05, #271):** both lanes now export
  `epsilon = (Bmax-Bmin)/(Bmax+Bmin)` along the tube
  (`vmex.core.turbulence.b_modulation_depth`: exactly GKX's `epsilon` for its
  `1/(1 + eps cos theta)` model, and `r/R0` on a `1/R` field) and
  `R0 = Rmajor_p = volume_p/(2 pi <area>)` (`L_axis/(2 pi)` on the mirror,
  the same identity) instead of `L_ref`, so GKX's derived `aminor = eps * R0`
  is a length. No benchmark or golden artifact pinned the old `std/mean`
  value; only `tests/mirror/test_turbulence.py` did, and it now pins the
  modulation depth, as does `tests/test_turbulence.py` (which also checks the
  depth against `sqrt(s) L_ref/R0` on the circular vacuum tokamak).
- 31.4-R4 The hybrid has no mirror throats: `stellarator_mirror_section_coefficients`
  (`geometry.py:165-212`) uses constant semi-axes along the leg; all |B|
  variation comes from the returns. 18.3's "target mirror ratio in the
  straight legs" needs leg-radius modulation a(u) (paraxial B ~ 1/a^2). Say
  in docs that the present legs are throat-less. Precedent: linked mirror
  (Feng, Yu, Jiang, Fu, Nucl. Fusion 61, 086014, 2021).
- 31.4-R5 DESC's shipped `ForceBalanceAnisotropic` closure (flux-function
  beta_a and p_perp) leaves a nonzero parallel force wherever b.grad B != 0,
  i.e. everywhere in a mirror; it is inadmissible as a mirror oracle. Keep
  DESC only for the consistent special closure p_par = p0(psi) +
  Delta(psi) B^2/2mu0 (31.4 model 1). Amend 16.7.
  **CLOSED (2026-09-03), with a correction.** Checked against the DESC source:
  `_F_anisotropic` is the *exact* divergence of the Grad-type tensor and is
  not restricted to flux functions - `ForceBalanceAnisotropic`'s own docstring
  tells the user to supply `FourierZernikeProfile` for 3-D anisotropy. The
  inadmissible case is the default *parameterization*, both `beta_a` and
  `p_perp` radial, which forces `beta_a == 0` in a mirror. Model 1 has
  `beta_a = Delta(psi)` (radial) but `p_perp = p0 - Delta B^2/2mu0` (not
  radial), so even that cross-check needs a 3-D DESC pressure profile. 16.7
  carries the derivation.
- 31.4-R6 The Agren-Savenko SFLM benchmark (`analytic.py:288-292`,
  mirror-geometry.rst:576-590,686-703) has no citation: Agren & Savenko,
  Phys. Plasmas 11, 5041 (2004) and 12, 042505 (2005); state which paper's
  Eq. 2 the potential is.
  **CLOSED (2026-09-03).** Both papers are now cited with DOIs
  (10.1063/1.1799351 and 10.1063/1.1870002) in `analytic.py`,
  mirror-geometry.rst and 16.7. The potential is the second-order paraxial
  potential of the **2004** paper (*Magnetic mirror minimum B field with
  optimal ellipticity*), together with its on-axis field, ellipticity and
  straight field lines; the Clebsch labels `(x0, y0)` in `clebsch_labels` and
  the quadrupolar/rigid-rotation proof are the **2005** paper. The published
  equation *number* could not be verified without journal access, so the
  unsupported "Eq. (2)" was dropped rather than repeated; Savenko's thesis
  reproduces the same expression as its Eq. (4.3) and attributes it to the
  2004 paper.
- 31.4-C1 "small-beta estimate" is the wrong name for sqrt(1 - beta)
  (mirror-geometry.rst:804,815; output.py:109; README polish/mirror text): it
  is Ryutov et al. 2011 Eq. 30, leading order in (a/L)^2 at any beta < 1 -
  "long-thin estimate, O((a/L)^2)"; the shipped two-coil case has
  (a/L)^2 ~ 6%, the size of the observed 50%-beta deviation.
  **CLOSED (2026-09-03).** Fixed in mirror-geometry.rst (both sites), the
  `summarize_axisymmetric_beta_scan` docstring and the README, with the Ryutov
  DOI. Numbers as shipped: `a = CENTER_RADIUS = 0.25` m and `L = 1.0` m (half
  the 2.0 m coil separation, the axial scale of the vacuum field) give
  `(a/L)^2 = 6.3%`; the 50% point's solved ratio 0.762687 sits 7.9% above
  `sqrt(1-beta) = 0.707107`. Note the SFLM paragraph in the same page also
  called `beta` a second small parameter of the long-thin ordering; corrected.
- 31.4-C2 Write the exterior BVP as equations in
  mirror-boundary-conditions.md (Laplace, Neumann data on side wall and caps,
  decay, direct-BIE collocation with Duffy quadrature); 17.2's "enforce the
  Neumann solvability/gauge condition" is wrong - the exterior decaying
  Neumann problem is uniquely solvable; the cap projection is a
  solenoidality-consistency correction. Cite Merkel 1986 and HvRM 1986 for
  the physical problem; the numerics differ from NESTOR.
  **CLOSED (2026-09-03).** mirror-boundary-conditions.md has a new "The
  exterior boundary-value problem" section: Laplace plus decay, the lateral
  and cap Neumann data, the uniqueness statement, the collocated direct BIE
  with the constant-subtracted double layer (verified against the code's own
  sign convention and the unit-sphere monopole), Duffy quadrature, and the
  NESTOR comparison including the missing net-current filament that motivates
  R2. 17.2's bullet is rewritten: the projection is solenoidality consistency
  (`sum_S B.n dA = 0`), not solvability or gauge - `gauge_error` is an
  identical zero in the code.
- 31.4-C3 State that Dirichlet geometry plus Dirichlet flux at the cuts
  over-determines a flux-carrying plane (the end-collar boundary layer is
  the consequence) and that I'(s) != 0 in an open mirror is current closed
  through the end plates.
- 31.4-C4 The validation map labels a uniform-field test as "straight
  circular vacuum mirror"; point it at the paraxial two-coil flux-tube test
  (`test_mirror_geometry_fields.py:178-245`, Ryutov Eqs. 19-21) and the exact
  quartic-flux mirror (`analytic.py:41-105`); extend the paraxial finite-beta
  test to all z at fixed psi.
- 31.4-C5 mirror-boundary-conditions.md:27-28 cites Hirshman-Whitson 1983
  for the plasma-vacuum interface; cite HvRM 1986 and Merkel 1986.
- 31.4-C6 Add the periodic/no-end-loss sentence to the two hybrid example
  headers; state (18.1) that Boozer coordinates are well defined on the
  hybrid but any single-field-line surface diagnostic (Gamma_c, eps_eff,
  NEO averages) is invalid on the rational closed-line hybrid and must use
  explicit (theta, u) surface quadrature.
- 31.4-C7 Hybrid literature for 18: Feng 2021; Helander 2014; Landreman-Catto
  2012; Dudt et al. 2024 (general omnigenity); Goodman 2023; Velasco 2024
  (piecewise omnigenity); Ryutov 2011 Secs. V/VII (anchors) and the
  sharp-boundary stability integral Eqs. 31-32 as a cheap 18.4 constraint.
- 31.4-C8 16.2 item 3 (CGL as an equilibrium pressure model) is ill-posed
  for a static equilibrium; keep CGL only as the interchange criterion.
- 31.4-C9 16.8's ANIMEC comparison cannot validate the mirror lane (all
  published ANIMEC cases are toroidal: LHD, QAS). Either implement the same
  closure in the toroidal core and compare there, or use the mirror
  references below.

Phase 9 spec sheet (literature-backed; replaces the schematic in 16.1-16.3
where they differ). Equations (SI): P = p_perp I + (p_par - p_perp) bb;
J x B = div P; div P = grad p_perp + (p_par - p_perp) kappa + b [b.grad(p_par -
p_perp) - (p_par - p_perp) b.grad ln B]; parallel balance b.grad p_par =
(p_par - p_perp) b.grad ln B (Ryutov 2011 Eq. 4; Frank 2025 Eq. 4);
perpendicular balance grad_perp(p_perp + B^2/2mu0) = sigma (B^2/mu0) kappa,
sigma = 1 + mu0 (p_perp - p_par)/B^2; Grad closure as in 16.3 (with it the
bracket in div P vanishes identically - use as the parallel-integrability
residual test). Admissibility: firehose sigma > 0; mirror tau = 1 + (mu0/B)
d_B p_perp|_s > 0 (bi-Maxwellian: beta_perp (T_perp/T_par - 1) < 1 - unit
test); ellipticity needs both. Pressure models, all inside p_par(s, B): (1)
consistent-Delta p_h,par = Delta(s) B^2/2mu0 (closed form; the only family
where a DESC cross-check is legitimate); (2) Novatron polynomial p_par = A(s)
C (1 - (B/B1)^2)^M with p_perp = A C (1 - (B/B1)^2)^(M-1) (1 + (2M-1)(B/B1)^2)
(arXiv:2503.03387); (3) ANIMEC bi-Maxwellian moments (Cooper 2006 NF 46,
683; Moen 2023 Eq. 2.6) by (E, mu) quadrature under AD; (4) tabulated
sloshing-ion p_par(s, B/B0) (Frank 2025 Eqs. 13-17; WHAM tables) as a cubic
spline with p_perp from the AD derivative. Boundary conditions: fixed
lateral Dirichlet; free lateral B.n = 0 and p_perp + B^2/2mu0 continuous;
cuts Dirichlet geometry and lambda (normal stress p_par there); periodic
hybrid periodic p_h,par. Validation: V1 p_h = 0 reproduces the shipped
energy, residuals, JVP/VJP bit-for-bit; V2 anisotropic theta-pinch with exact
1-D balance p_perp + B_z^2/2mu0 = const, second-order convergence; V3
manufactured Cartesian div P; V4 parallel-integrability residual identically
zero per model; V5 long-thin anisotropic mirror B/B_vac = sqrt(1 -
beta_perp(psi, z)) along z to O((a/L)^2) with a sloshing peak at B/B0 = 2
(45-degree NBI) producing an off-midplane diamagnetic depression; V6
admissibility thresholds; V7 WHAM-like axisymmetric case (B0 = 0.86/0.32 T,
B_m = 17 T at z = +-0.98 m, R_m^vac ~ 20/53, a = 0.10 m, beta 0.2-0.3;
Endrizzi 2023) with R_m = R_m^vac (1 - beta)^-1/2 on axis, referenced
against the Novatron closure family or Chernoshtanov arXiv:2512.01780 /
Khristo-Beklemishev JPP 91, E3 (2025) since public Pleiades is isotropic;
V8 free-boundary anisotropic interface; V9 Ryutov Eq. 26 stability integral
reported per surface. References to add to 16.7: Grad 1967; Hall &
McNamara 1975; Taylor 1963; Newcomb 1981; Kaiser, Nevins & Pearlstein 1983;
Ryutov et al. 2011; Cooper 2006; Moen, Suzuki & Proll 2023; Novatron 2025;
Hammir arXiv:2411.06644; Khristo & Beklemishev 2025; Chernoshtanov 2025;
Feng 2021; Merkel 1986; HvRM 1986; Agren & Savenko 2004, 2005.

### 31.5 Publication-readiness audit (JOSS / CPC / FAIR4RS)

REQUIRED before submission: (1) merge #245 and then repair it - CITATION.cff
needs ORCID(s), all contributors, an `abstract`, and the Zenodo DOI under
`identifiers`; CHANGELOG backfilled to v0.1.0 (18 tags exist) with compare
links; `docs/project/contributing.rst:130-132` "no changelog" sentence
removed. (2) Zenodo archive + DOI badge + `codemeta.json`; GitHub topics and
description (no Zenodo record exists as of 2026-09-02 - a maintainer
action). (3) The validation page (outline: evidence hierarchy; VMEC2000
parity tiers with exact gates - `test_parity_breadth.py` rtol 1e-5
harmonics and a +-25% iteration window, `golden_digests.json` scalars,
nightly `test_wout_golden.py` per-variable; trajectory figure; free-boundary
ladder; #247's fresh-deck table quoted by measured maxima, never "machine
precision"; the oracle and cross-code table; derivative certificates;
mirror analytic limits; downstream parity; device/lane consistency; what is
NOT validated; how to rerun) and one numerics/reproducibility page (float64;
what "bit-identical" covers; ULP-per-iteration drift across XLA fusion,
identical iteration counts, geometry <= 1e-12; CPU/GPU rtol 1e-7 gate;
cache semantics; seeds; provenance fields). (4) Figure provenance: only 2 of
19 README/docs figures are guarded; add `docs/_static/figures/figures.json`
(path, sha256, generator, inputs, date, hardware) with a guard; generate or
remove the orphans `readme_diagnostics_summary.webp`,
`readme_diagnostics_qa_vacuum.webp`, `readme_bootstrap.webp`; pick one
provenance story for `readme_runtime_compare.webp` (delete the stale
2026-07-07 sidecar); add `_provenance` to `convergence_nfp4_ns51.json` and
`gpu_baseline.json`; make mirror/essos/extender scripts write webp
directly. (5) Unbacked numbers - remove or re-measure: README cloc table
and its two VMEC2000 shas; 1.79x/3.29x ensemble scaling; the production
workflow table in performance.rst (mixed hosts, no record - point at
`benchmarks/baselines/m4/*`); "exact match" iteration table (gate is +-25%,
relabel "observed"); 2D-preconditioner 5.4x/10.9x/9.2x typed into
`make_readme_figures.py`; "identical fp32/fp64 GPU times"; "2e-9 relative
FD" in all-of-vmex.md (gate is 1e-4); "SFINCS comparisons live in
benchmarks/". (6) Root CONTRIBUTING.md, CODE_OF_CONDUCT.md, issue templates,
a support sentence; delete the stale migration narrative in contributing.rst.
(7) pyproject: real author, jax/jaxlib floor (>= 0.4.36 for
`jax_logging_level`), Beta classifier; README rename note. (8) plan.md is
linked from the README and contains agent instructions - the maintainer
decides whether to move it out of the public tree or trim it to a
ROADMAP.md; strip "the plan" from `strong_force_comparison_m4.json:61` and
performance.rst; write the JOSS AI-usage disclosure honestly. (9) API
reference: automodule `vmex.core.problem`, `monitoring`, `run_options`,
`boozer_tables`, `statephysics`, `vmex.doctor`, `mirror.{basis,exterior,
forces,geometry,turbulence}`; 365 of 1266 public defs lack docstrings
(optimize.py 43, polish.py 32, implicit.py 31) - exported names first.
Recommended: extend `tools/check_docs_prose.py` to README/CHANGELOG and
drop the flagged register words; test `force_balance_polishing.py` and
`epsilon_effective.py` at CI budget; test or delete `simsopt_driver*.py`;
lowercase `uwplasma/VMEX` URLs in `tools/render_capabilities.py`; verify DOI
10.1063/1.3212262; settle the "clean-room" vs "port" wording and the
STELLOPT license lineage for the paper; draft the CPC Program Summary.
Branches `docs/publication-hygiene`, `docs/api-reference-completeness`,
`docs/figure-provenance` were assigned (4)-(9) and the recommended items.

### 31.6 Open decisions for the maintainer

Settled by the maintainer on 2026-09-02:

- #197 (contributor's scalar-adjoint examples): verdict posted on the PR.
  The scalar lane does not improve the two user-visible latencies (shared
  machinery, fixed by #240); it wins cold start and memory (44.7 -> 32.2 s,
  2965 -> 2574 MiB); at matched evaluation budget TRF least squares reaches
  a 3.0x lower objective. Take it in as examples with that framing, after a
  rebase (`codex/scalar-optimization-drivers-rebased` is clean) and a
  sentence per example naming the trade.
- plan.md STAYS in the public tree. 31.5 item 8 therefore reduces to
  stripping the "the plan" phrasing from user-facing text
  (`strong_force_comparison_m4.json:61`, performance.rst) and writing the
  submission's AI-usage disclosure honestly; do not move or trim this file.
- No Zenodo record yet - the code is still changing. CITATION.cff ships
  without a DOI; add `identifiers` and the badge at submission time. Do not
  block the publication pack on it.
- Codecov: deleting and re-adding the repository there did not fix the 404,
  because the Codecov project still carries the pre-rename vmec_jax
  identity. PR #249 switches the upload to the repository token; the
  `CODECOV_TOKEN` secret is the maintainer's to set, and the token pasted
  into chat on 2026-09-02 should be regenerated before use.

Still open:

- The pointwise eps_F definition (31.2-R1): keep as the internal certificate
  and add the published global normalizations for every reported number.
- GitHub topics and repository description.

### 31.7 Remaining engineering items (carried from section 8.8 and the campaign)

3-D polish effectiveness and memory at production resolution (the
stellarator README row depends on it; W7-X numbers in 31.1); v0.8.2 after
the polish memory PR; the section 9 matrix rerun with all five regimes;
GPU validation of the 0.8.1 stack (office GPUs; host RAM shared with the
polish run); persistent-cache eviction-lock behavior for concurrent
processes on one cache directory (the CI symptom was serialization to
timeout; users running job arrays on a shared home need a product answer:
per-process shard, batched writes, or floor tuning); parity lane c3d at
43-44 minutes of a 55-minute budget (repartition the implicit-campaign
modules); tridiagonal method pinning per placement (bit-identical on CPU,
needs a device-aware runtime field); booz_xform_jax 0.2.0 adoption.

### 31.8 Next-agent runbook

Local checkout `/Users/rogeriojorge/local/vmex` is shared - work in
worktrees under the session scratchpad; push after every commit; author
commits as Rogerio Jorge; never any assistant attribution; one heavy local
job at a time; the office box (`ssh office`, `~/vmex-gpu`,
`~/venvs/vmex-gpu/bin/python`, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`) takes
one heavy job and needs `rm -rf ~/.cache/vmex ~/.cache/jax` before runs
(stale cross-profile AOT entries segfault). Merge bar: local verification
plus every substantive CI lane green; the "PR gate" is red on cancelled or
guard-only failures, so read the lane list. `tools/preflight.py --static`
then the diff-affected suites; the ownership guard needs essos installed
locally. Measurement discipline: fresh process, caches cleared, subprocess
wall, two reps, never benchmark a shared checkout, and record commit,
hardware, and versions in a JSON with a guard.

### 31.9 Priority order

Work top-down. The ordering is by what a referee reads first and by what is
currently WRONG in shipped material, not by effort.

P0 - published statements that are incorrect or unsupported. Every headline
polish number depends on 31.2-R1 (the saturating metric), so fix the metric
and the reported quantities before anything else touches those figures:
31.2-R1, then 31.2-R3 (the "26-fold" recompute) and 31.2-R2 (the DESC row),
then 31.2-R4 (the high-order page describes a retired method), 31.2-R6
(selection wording), 31.2-R8/R9 and 31.3-R1..R7 (convention and diagnostic
statements that are simply wrong and cost nothing to fix), and 31.5 item 5
(numbers with no artifact behind them). None of these change solver code.

P1 - physics correctness in code, in this order: 31.2-R7 (the LASYM tcon
halving is a possible parity defect - if VMEC2000 halves and VMEX does not,
every up-down-asymmetric result is off), 31.4-R2 (free boundary silently
accepts a net axial current the exterior cannot carry - guard it), 31.2-R5
(the oracle has no exact-solution validation; add the analytic Solov'ev).

P2 - the publication package, once P0 has settled what the numbers say:
merge #245-#248; then 31.5 items 1, 3, 4, 6, 7, 9 (citation metadata,
validation and numerics pages, figure provenance manifest, community files,
packaging metadata, API reference). Items 2 and 8 wait on the maintainer.

P3 - performance and engineering, unblocked and parallel to P2: the 3-D
polish memory fix and v0.8.2; the section 9 benchmark matrix rerun on 0.8.1+
with all five regimes after #246; booz_xform_jax 0.2.0; GPU validation;
c3d lane repartition; the cache eviction-lock product answer.

P4 - Phase 9/10 mirror and hybrid work is planning only until P0-P2 land.
Section 16.3 (corrected above), 31.4-R3..R6 and the spec sheet are the
inputs; do not start implementation while publication claims are unsettled.

Not in the ordering, because they are not ours to do: Zenodo, Codecov
re-linking, the #197 comment, and whether plan.md stays in the public tree
(31.6).
