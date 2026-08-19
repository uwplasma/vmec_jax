# VMEX research-grade plan

## How to use this file (humans and agents)

This plan is self-contained: an agent (Claude, Codex, or a person) should be able to pick any
item and implement it with only this file plus the referenced repos. Conventions:

- **Context.** Main repo `github.com/uwplasma/vmex`, local checkout `~/local/vmex`. This plan
  targets the state after PR #123 (`rj/vmec-extender-field`) merges (Phase 0). Companion repos,
  all local under `~/local/` and on github.com/uwplasma unless noted: `solvax` (=`SOLVAX`,
  case-insensitive FS, installed editable), `NEO_JAX`, `booz_xform_jax`, `virtual_casing_jax`
  (branch `rj/release-0.0.5`), `ESSOS` (PR #58 pairs with vmex #123), `DESC` (PlasmaControl,
  reference only), `STELLOPT` (PrincetonUniversity, fork via rogeriojorge for PRs). Python:
  `/opt/local/bin/python3` (3.11), jax 0.9.2, scipy 1.17.1. GPU box: `ssh office`
  (pop-os, 2x RTX A4000 16 GB). Measured baselines: table below; raw profiling scripts are
  referenced per phase and should be re-run to confirm numbers on the machine at hand.
- **Item IDs.** Reference work as `P<phase>.<item>` (e.g. P3.2). Do not renumber existing items;
  append new ones.
- **Status.** Mark items inline as they change: `[TODO]` (default, unmarked), `[DOING @who]`,
  `[DONE pr#]`, `[BLOCKED reason]`.
- **Log.** Every contribution appends one entry to the `## Log` section at the bottom —
  newest last, never edit or delete prior entries. Format:
  `- YYYY-MM-DD who: P<ids> — what changed / what was measured / PR links / handoff notes.`
  Substantive design changes get a short rationale in the log, and the affected phase text is
  updated in place so the plan body always reflects the current intent.
- **Authorship.** All commits/PRs authored by `rogeriojorge` (git auth); never Claude/Codex
  attribution. PR bodies short and concrete, matching prior rogeriojorge PRs.

Working agreements that apply to every phase:

- All commits, PRs, and PR text are authored by `rogeriojorge` (git auth); never Claude/Codex
  attribution anywhere. PR bodies short, concise, in the style of prior rogeriojorge PRs.
- No scaffolds, testbeds, proxies, or "experimental" lanes survive: code is either wired in and
  certified, or deleted. Prefer fewer lines, fewer files, fewer folders — in source and tests.
- Tests are literature-anchored (papers, other codes, analytic limits), concise, and fast; CI
  stays under 30 minutes while covering >= 95% of lines and all physics/algorithm branches.
- Every performance claim in docs/README is backed by a measured number checked into the
  benchmark JSONs, never prose-only.

Measured baselines backing this plan (Apple Silicon CPU, uncontended, 2026-08-17/18; scripts in
the session scratchpad: `profile_lasym.py`, `fb_isolate.py`, `fb_forward_anatomy.py`,
`fd_tighten.py`, `profile_stall.py`):

| Measurement | Value |
|---|---|
| LASYM vs symmetric per-nfev (max_mode=2, ns=31) | 19.1 s vs 10.4 s (1.8x); jac 5.5x, compile 2.6x |
| LASYM stage-1 (20 nfev) uncontended | ~6-13 min; overnight run descended 25.0 -> 2.64 in 18 its |
| Free-boundary forward (ns=25, mpol=ntor=5) | multigrid 6.2 s; implicit wrapper warm 0.7-9.2 s |
| Free-boundary warm value+grad (136 coil dofs) | ~25 s, adjoint-dominated (unpreconditioned GCROT) |
| Coupled FD-vs-AD error (ns=16 LASYM) | 2.3e-3 warm ftol=1e-7 (current gate 2e-2); **1.5e-4 cold ftol=1e-9** |
| Compile cache | at 1 GiB cap; identical rerun recompiles everything |

---

## Phase 0 — Unblock and merge PR #123 (`rj/vmec-extender-field`)  [DONE except the merge]

Smallest possible diff to green; everything else moves to the new branch off `main`.

1. Ruff: `E701` in `examples/optimization/QA_optimization.py:65` (split the one-liner `if`);
   `F541` in `examples/optimization/QA_optimization_global.py:71` (drop the `f` prefix).
2. Add `tests/test_neoclassical.py` to `tests/manifest.json` (pick a lane that runs it so the
   changed-line coverage gate sees `vmex/core/neoclassical.py`).
3. Fix `tests/test_examples.py::test_vacuum_qs_examples_expose_trial_pressure_terms` to point at
   `QA_optimization_DMerc_vacuum.py` (where `USE_TRIAL_STABILITY` now lives); update the matching
   pointer in `docs/reference/objectives.rst`.
4. Docs linkcheck: replace `https://docs.jax.dev/en/latest/advanced-autodiff.html` (404) with
   `https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html` in `docs/project/references`.
5. Pin `scipy>=1.15` in `pyproject.toml` (all eight asymmetry examples + both maxJ continuations
   use `least_squares(..., callback=)`, added in SciPy 1.15).
6. Merge PR #123. Open the new working branch from `main`; all phases below land there in
   focused PRs.

Acceptance: PR CI fully green (quality, coverage gate, linkcheck), PR merged.

## Phase 1 — Examples run honestly (the "stall" fix)  [DONE]

Diagnosis (instrumented reproduction, `profile_stall.py`, uncontended): the examples descend
(overnight log: 18 iterations, cost 25.0 -> 2.64) and healthy iterations cost ~10-12 s
(residual re-solve 8-10 s, Jacobian 1.8-3.9 s). The stall is real and has FOUR components, now
measured:
(a) **Pathological Jacobian evaluations — the dominant cost, and it is systematic.** Full-stage
    measurement (LASYM QA, max_mode=2, 48 dofs): jac #1-2 take 1.8-3.9 s, then EVERY Jacobian
    from iterate ~3 on takes ~2000-2240 s (~35 min; ~42 s per dof column vs ~0.2 s early) while
    residual re-solves stay at 3.5 s. Stack sample: main thread blocked in a single XLA
    `Execute` (`BlockUntilReady`) — the per-dof implicit linear solves inside `jac_jit` grind
    once the iterate moves away from the reference/compile point (frozen preconditioner/tcon
    quality? hot-restart seed distance?), i.e. degradation is persistent, not an unlucky trial. All 48 dof solves
    share one operator `dF/dz(z*)`: make the factor-once amortized block-Thomas path
    (`solvax.block_thomas_factor/solve`, already documented in `optimize.py`) the default for
    ndof over a small threshold, keep per-dof GMRES as fallback, cap inner iterations with a
    typed diagnostic instead of silent grinding, and emit a heartbeat (`jax.experimental.
    io_callback`) so a long Jacobian is visibly alive. Observed blowup is ~500x (one jac
    execution > 30 min CPU-bound vs 2-4 s healthy), which exceeds any plausible GMRES-maxiter
    factor — also audit whether the jitted Jacobian program re-runs the full equilibrium
    while_loop (forward_max_iterations=2000) per dof column at unlucky iterates instead of
    reusing the converged donated state; that recomputation, 48x over, matches the magnitude.
    First step of the fix PR: an instrumented jac lane that reports per-column inner-solve
    iteration counts and solve/linearize split (io_callback), run at the captured bad iterate
    (the profiler saves x per call, `scratchpad/profile_stall.py`).
(b) **Mid-loop recompile churn**: `jit(_block_lane)` (+`jit(copy)`) recompiles ~3 s apiece
    *inside* residual re-solves (`jax_log_compiles` captured pairs of `_block_lane` compiles per
    hot-restart solve at identical shapes `f64[31,50]`) — jit identity instability
    (`solver.py:1760` lambda/closure) and/or eviction; fix the callable identity, then Phase 2.
(c) **Unflushed output**: scipy prints one row per iteration; everything else sat in the 8 KiB
    buffer (fixes below).
(d) macOS sleep/App Nap throttling long unattended runs (document `caffeinate -i`).
Then fix all of the following:

1. **Flush everywhere.** Flip the five `emit=print` defaults to the existing
   `printing.emit_flushed` (`solver.py:2119`, `multigrid.py:244`, `multigrid.py:564`,
   `freeboundary.py:1568`, `freeboundary.py:2194`); add `flush=True` at `monitoring.py:71,462,466`
   and `bootstrap.py:1039`; document `python -u` in `examples/README.md`.
2. **Per-nfev progress.** `VmecProblem` gains an opt-in progress line per residual/Jacobian call
   (timestamped, flushed) so a 40 s evaluation is visibly alive; the examples enable it. This is
   the direct answer to "stuck at iteration 1 for several minutes".
3. **Kill the monitor double-solve.** `OptimizationMonitor._term_costs` re-calls
   `problem.residual(x)` per accepted iterate (`monitoring.py:233-243`); reuse the cached residual
   from the accepted evaluation (term slices are already in `problem.metadata`).
4. **Long-run ergonomics.** Examples print a one-line budget estimate per stage (measured
   per-nfev cost x max_nfev); `examples/README.md` documents `caffeinate -i python -u ...` for
   multi-hour runs on macOS; outputs go to an ignored `results/` directory instead of the CWD.
5. **CI executes the examples.** Nightly lane runs at least QA + QI asymmetry in
   `VMEX_EXAMPLES_CI=1` smoke mode asserting descent (final cost < initial). None of the eight
   currently executes anywhere.
6. **`jax_explain_cache_misses` crash (found while profiling the stall).** Setting
   `jax.config.update("jax_explain_cache_misses", True)` deterministically kills any vmex solve
   with `ValueError: not enough values to unpack (expected at least 3, got 2)`, surfacing at the
   first jit under the flag (`solvax/tridiagonal.py:203` `lax.platform_dependent`, reached from
   `vmex/core/preconditioner.py:689`). Bisected: the flag alone is the trigger (base /
   `jax_log_compiles` / custom cache dir all pass); import order and x64 are fine
   (`_compat.py:222` env + `solver.py:71` hard-set — verified). Actions: minimal repro
   (`platform_dependent` under the flag on jax 0.9.2) -> upstream JAX issue; until fixed, Phase 2
   cache diagnosis uses `jax_log_compiles` + cache-size accounting instead of miss explanations,
   and `vmex --doctor` warns if the flag is set. Note for anyone debugging: running Python from
   `~/local` (the repo's *parent*) shadows `vmex` as an empty namespace package — imports fail
   loudly, but don't chase that as a bug.

Acceptance: interactive stage-1 run shows a flushed, timestamped line at least every ~30 s;
profile shows zero mid-stage recompiles; nightly smoke lane green.

## Phase 2 — Compilation cache policy  [PARTIAL: sizing + doctor DONE; the real cost is elsewhere]

Today: machine-scoped persistent cache (`_compat.py`) capped at 1 GiB; the cache sits exactly at
the cap and an identical rerun recompiles everything — the cap forces eviction churn, and
`pure_callback` identities may be poisoning keys.

1. Diagnose with `jax_log_compiles` + cache-directory accounting (file count/bytes/atimes before
   and after) across two identical example runs; classify misses (evicted vs key-unstable). Do
   NOT use `jax_explain_cache_misses` — it crashes vmex solves on jax 0.9.2 (Phase 1.6). If
   callbacks poison keys, hoist them so cached jits close over stable callables (module-level,
   config-keyed) rather than per-call closures.
2. Policy: default cap sized to hardware — `min(20 GiB, 10% of free disk)` with LRU eviction,
   overridable by the existing `VMEX_COMPILATION_CACHE_*` env vars; document one knob, not many.
   Rationale: single VMEX executables reach tens-hundreds of MB; a working set of one user's
   examples is several GiB; 20 GiB fits any state-of-the-art workstation, the disk-fraction guard
   protects small laptops. GPU adds its own kernels — same policy, separate per-backend fingerprint
   directory (already in place).
3. Add a `vmex --doctor` line: cache dir, size vs cap, hit rate of the last run (JAX exposes
   miss explanations; a simple counter in `_compat` suffices).
4. Regression test: build the same small problem twice in two subprocesses; assert the second
   compile time is < 25% of the first (skip on CI runners without a persistent HOME).

Acceptance: identical example rerun compiles in seconds, not 30-140 s; doctor reports cache
health; test pins it.

## Phase 3 — Free-boundary speed and accuracy (explicit plan)

Target: warm value+grad at example scale (ns=25, 136 coil dofs) from ~25 s to <= 8 s CPU, with the
exact certificate untouched, and a GPU lane that beats CPU. The Schur direct lane remains the
exact fallback; the default lane becomes preconditioned + recycled.

1. **Instrument first**: land `adjoint_matvec_count`-style counters in `_host_adjoint`
   (matvecs/gradient, mean matvec time) so before/after is a number in the PR body.
2. **Precondition the certified GCROT lane.** Pass `M ~= (A^T)^-1` to `gcrotmk` in
   `_host_adjoint` (`freeboundary_implicit.py:725`), where `A` is the frozen block-tridiagonal
   bulk already assembled by `im._raw_block_system` for the Schur lane. Factor once per gradient
   with `solvax.block_thomas_factor(store_offdiagonals=False, factor_dtype=float32)` (0.13
   reusable factors: 3-6x less factor memory); float64 refinement stays in the Krylov loop. Since
   `E = J - A` is edge-low-rank, expect O(10-30) preconditioned matvecs vs O(100+) today.
3. **Wire in `freeboundary_linear.py` as the preconditioner backbone** (disposition: wire in, not
   delete). `NestorBorderedOperator.preconditioner(plasma_solve, schur_solve)` is the block
   inverse `M`; the two adapters it needs already exist in `tests/test_freeboundary.py:162-176`
   (a `vacuum_system(x)` from `solver_vac.assemble`, and a `plasma_residual(x, q)` with explicit
   potential). This makes the bordered operator load-bearing production code with its existing
   2e-12 linearization tests as the unit certificate.
4. **Recycle Krylov subspaces across optimizer trials.** Persist the GCROT deflation space
   (scipy `CU=`/`discard_C`, or move the host lane to `solvax.gcrot` with `recycle` and surface
   `recycle_drift` in `LinearResponseReport` — resolving the stale doc claim by using it).
   Store next to the warm state in the config-keyed hot cache.
5. **SOLVAX/VMEX split** (companion solvax PR): solvax gains a generic bordered-operator type and
   a low-rank-update preconditioned/recycled GCROT policy; vmex keeps NESTOR residual assembly,
   Fourier/edge constraints (m=1 pairing), coil-to-boundary maps, and the physics certificate.
6. **Certificates unchanged in math, tightened in tolerance** (Phase 3b below). Every adjoint
   still checked against the true coupled transpose at `10 x adjoint_tol x ||rhs||`; Schur direct
   lane kept as exact fallback and as the cross-check in tests.
7. **GPU lane (office box, 2x RTX A4000 16 GB).** Profile cold compile memory and steady-state
   on GPU with the XLA profiler (`jax.profiler.trace` -> perfetto; `nsys` if kernel-level needed).
   Gate: one coupled value+grad on GPU within 16 GB and faster than CPU warm. The reusable
   float32 factors and the preconditioned lane are exactly what shrinks the GPU footprint.
8. **Forward-solve iteration budget.** Free-boundary forward runs 1193 its where fixed runs 141
   at the same size (8.5x iteration ratio, 2.2x per-iteration cost). Investigate vacuum-refresh
   cadence (`ivacskip` analogue) and preconditioner reuse across vacuum updates for a further
   forward win; any change must keep wout parity tests green.

Acceptance: matvecs/gradient reduced >= 3x with certificate green; warm value+grad <= 8 s CPU at
example scale; GPU value+grad runs in 16 GB and beats CPU; no test tolerances loosened.

## Phase 3b — Coupled FD certificate at research grade

Measured: the FD noise floor is the solver endpoint, not the adjoint. Cold re-solves + ftol=1e-9
gives 1.5e-4 agreement; warm probes at tight ftol are corrupted by hysteresis (6.8e-2); below the
reachable ftol the root itself wanders.

1. Rewrite `test_free_boundary_current_gradient_matches_resolve_finite_difference`: cold
   re-solves (pop `_FREE_HOT_CACHE` per probe), forward ftol=1e-9 (niter to reach it), h=2e-4,
   assert the forward actually attained ftol, add a noise control (two identical cold re-solves;
   require |delta objective| << h x |derivative|), gate at **rtol=1e-3** (6x margin over measured).
2. Same protocol for the boundary-Schur certificate (from 5e-2 to 1e-3).
3. Add one coil-shape-dof FD certificate (not just `extcur`): a single ESSOS geometry dof through
   `field_from_parameters`, same protocol, `full`-marked.
4. Document the endpoint-noise physics in `docs/explanation/adjoint-gradients.md` (why warm FD
   probes lie; why Richardson amplifies noise here).

Acceptance: both certificates gate at 1e-3, nightly runtime <= 10 min combined, and fail if the
forward stalls above the requested ftol instead of silently passing.

## Phase 4 — Community API: `FreeBoundaryProblem.from_tuples`

`optimize.py`/`problem.py` have zero `lfreeb` support today; the two 150-line examples are the API.

1. New `FreeBoundaryProblem` mirroring `VmecProblem`: same objective tuples, plus
   `coils=` (ESSOS `Coils` | `MgridField` | `extcur` array), `coil_dofs=` filter,
   `coil_terms=` engineering objectives, optional `boundary_max_mode=` for joint
   boundary+coil dofs (virtual-casing lane), `ns/ftol/adjoint_tol`, built-in smooth
   rejected-trial wall, unit scaling, `dof_names`, monitor term slices,
   `compile_value_and_gradient()`.
2. Rewrite both single-stage free-boundary examples to ~40 lines on top of it; keep the current
   API calls only in the how-to as the "under the hood" appendix.
3. Tests: construction/validation guards; value+grad equals the hand-rolled pipeline bit-for-bit
   on the smoke config; descent smoke (2 L-BFGS iterations); joint boundary+coil dof path;
   docs how-to `howto/optimize-free-boundary-coils.md` + tutorial `first-free-boundary.md`.
4. Retire the "experimental" label via the capability-JSON tripwire
   (`test_capability_docs` pins the exact wording — update JSON + generator in lockstep) once
   Phase 3/3b acceptance holds.

Acceptance: an end-user drives a free-boundary coil optimization in <= 40 lines; class fully
tested; capability table says supported (CPU), GPU status stated honestly.

## Phase 5 — Full LASYM

### 5a. vmex bugs (immediate, ship with Phase 0 follow-up)
- `MaximumJResidual.compute_state` (`maxj.py:543-549`) and the shared dict in
  `qi_and_maximum_j_from_boozer` (`maxj.py:387-390`) drop `bmns_b`: the maxJ certificate
  silently symmetrizes LASYM fields. Fix both; add a parametrized regression across all five
  bounce classes (nonzero `bmns_b` must change the residual; `bmns=None` == `bmns=0` bit-exact).

### 5b. vmex hard guards, in order
1. `virtual_casing._state_field_spectra` (`virtual_casing.py:346-351`): add the sine-parity
   contravariant-B spectra (jnp clone of `nyquist.wrout_sin_coeffs`, full-theta grid, LASYM
   `tmult` normalization — all patterns exist in `nyquist.py`). Geometry half already computes
   `rmns/zmnc`. This unblocks LASYM live-state virtual casing; note
   `PlasmaVacuumInterface.from_wout` already works for LASYM today.
2. `extender.py`: thread the sine families through `_flux_coordinates_to_xyz` and
   `_interior_coordinates_and_B` (currently zero `lasym` handling — would silently drop them).
3. `l_grad_b` wout lane (`optimize.py:723`) and `_lgradb_state_tables`
   (`statephysics.py:570`): plumbing only, arrays already exist.
4. Ballooning/turbulence (`stability.py:628`): larger (asymmetric-lambda PEST inversion);
   either schedule after 1-3 or keep the guard and state it as a deliberate limit in the
   capability table — no silent middle ground.
- `virtual_casing_jax` itself needs **no math changes** (vmex passes full-period grids,
  `half_period=False` hardcoded). Two small hygiene PRs there (authored rogeriojorge): honour or
  document the write-only `stellsym` field; relax the inherited simsopt-lane guard that is
  stricter than the code beneath it.

### 5c. NEO_JAX LASYM (own PR in NEO_JAX, merge when validated)
~150 lines of plumbing: add `rmns/zmnc/lmnc/bmns` + static `lasym` to `BoozerData`; ingest the
sine variables in all three `io.py` constructors (`lmnc = -pmnc_b*nfp/2pi`, `sqrtg00 = gmnc+gmns`);
forward through both drivers' coeff dicts; make the B-max tie-breaker sine-aware (or route LASYM
to the jax argmax path). The asymmetric Fourier kernel already exists and matches
`neo_fourier.f90` term for term — it is currently dead code. Validation: asymmetric boozmn
fixture + parity test against **patched** xneo (see 5d) or the STELLOPT in-memory path; the
booz_xform_jax side of the comparison uses the corrected/fixed xbooz reference. Then lift the
guard in `vmex/core/neoclassical.py:86` and add a LASYM eps_eff panel test.

### 5d. STELLOPT upstream PRs (fork `rogeriojorge/STELLOPT`, small and separate)
1. PR 1 — NEO boozmn reader: `NEO/Sources/read_booz_in.f90:143` `bmns(i,i)` -> `bmns(i,k)`
   (corrupts the asymmetric |B| spectrum; the in-memory `stellopt_neo.f90:226` copy is correct,
   proving the typo). Body: 3-4 sentences, the diff speaks.
2. PR 2 — NEO deallocation bugs: `neo_dealloc.f90:49-50` frees `pixn`/`i_n` while testing
   `pixm`/`i_m` (leaks both), and the LASYM arrays `rmns/zmnc/lmnc/bmns` are never freed.
3. Keep local patches in the fork until merged; generate all LASYM reference data with the
   patched reader only.

Acceptance: all four vmex boundary families first-class in virtual casing/extender/l_grad_b (or
explicitly gated in the capability table); NEO_JAX LASYM merged with Fortran parity at documented
tolerance; both STELLOPT PRs open.

## Phase 6 — Epsilon effective: surface-integral objective lane

Adopt the surface-integral reformulation (Paul et al. JPP 2020 Eq. 6.1; DESC and KNOSOS are both
instances): many short field-line transits x pitch grid, all independent, all fixed-shape.

1. **New `vmex/core/ripple.py`** built on what exists: `boozer_bmnc_state` (traceable Boozer,
   LASYM included) + the differentiable bounce kernel (`bounce.py`, sin-map Gauss quadrature).
   Extend `trace_boozer_field_lines` with the two dB einsums so
   `|grad psi| kappa_G = (I dB/dzeta - G dB/dtheta)/(G + iota I)` — no Boozer geometry harmonics
   needed. Pitch grid: 1/lambda uniform in B on (Bmin, Bmax), open-Simpson weights (~48-64 nodes;
   DESC `get_pitch_inv_quad` is the reference). Generalize `bounce_action` to the Nemov (H, I)
   pair sharing bounce points; assemble with `safediv`; normalize by the flux-surface-average
   line length (DESC `_neoclassical.py:225-262` pattern). `<|grad psi|>` and `R0` from vmex's own
   traceable half-mesh tables.
2. **Objective class** mirroring `QIResidual` (`residuals_state` duck type) so it drops into
   `from_tuples` and `jac="implicit"` unchanged. Register in `optimize.__all__`.
3. **Gradients fast and small**: reverse-mode works out of the box (fixed shapes); memory by
   `solvax.chunk_map` over pitch + `jax.checkpoint` per chunk (DESC's chunking-not-remat
   strategy); B-extrema roots via `solvax.root_solve` (IFT, no while_loop). For the implicit
   least-squares driver only JVPs are needed — already chunked.
4. **Smoothness for optimization**: fixed `max_wells` with NaN-honest sentinels but a smooth
   pitch/well weighting (softplus margins where a hard max would kink); verify objective
   smoothness by plotting eps_eff along a boundary-coefficient ray.
5. **Independent parity ladder (several comparisons, then claim parity):**
   - analytic tokamak limit: `B = B0(1 - eps_t cos theta)` -> eps_eff = eps_t to quadrature order;
   - STELLOPT NEO (patched xneo) at production resolution on the repo's QA/QH/QI/QP wouts,
     symmetric and LASYM, 1-3% (NEO's own acc_req bounds tighter claims);
   - NEO_JAX (post 5c) at matched `NeoConfig` — never default-vs-default (50x different problem);
   - DESC `EffectiveRipple` on a shared equilibrium (extend the existing
     `test_matches_desc_bounce1d_when_available` pattern);
   - convergence scans in (nalpha, num_transit, npitch, quad order, max_wells) with the
     num_transit x nalpha equivalence check.
6. **Gradient validation**: JVP/VJP transpose identity; jacfwd vs grad; central FD through the
   full implicit chain on one boundary coefficient (Phase 3b protocol).
7. **Example** `examples/optimization/QA_eps_eff_optimization.py` following the standard
   template (stages, monitor, report, plots), and a finite-beta variant flag.
8. **Plot fixes (land early, independent of the lane):** summary-panel eps_eff with more surfaces
   once fast (12-16), explicit `set_ylim(0.5*min, 2*max)` + minor log ticks so the minimum is
   always visible, clearer LASYM-unavailable note until 5c lands, and the same axis policy in
   `examples/epsilon_effective.py`.

Performance target: < 1 s/surface CPU, ~10-50 ms/surface GPU steady-state (DESC demonstrates the
regime); reverse-mode gradient at O(1) memory in dof count.

Acceptance: parity table (5 independent comparisons) in docs; optimization example descends and
is smoke-run in CI; gradient certificates green; wout-lane NEO_JAX diagnostic retained as the
independent cross-check, not deleted.

## Phase 7 — NEO_JAX speedups (companion PRs in NEO_JAX + solvax)

Priority order (each an independent, measurable PR):
1. Real early exit from the period scan (bounded two-pass scheme; up to 4x on converged cases).
2. Vectorize the trapped-class deposit: `segment_sum`/one-hot matmul replaces the per-step
   `fori_loop`+`cond` scatter (~1e7 serialized scalar ops per surface today).
3. Fourier as GEMM via `cos(m theta - n phi)` separability: kills the (theta x phi x mode)
   temporaries — measured 4.45 GB -> tens of MB — and obsoletes the streamed mode switch.
4. `dynamic_slice` spline gathers (16 coefficients, not a 4x4xphi_n slab, ~2.4e6 times/surface).
5. solvax offload: `splper`/`splreg` -> `cyclic_tridiagonal_solve`/`tridiagonal_solve` (deletes
   ~180 lines of ported index arithmetic); extrema Newton -> `solvax.root_solve` (restores
   reverse-mode); surface batching -> `chunk_map`; make `acc_req` traced, hoist the per-call jit.
6. Run down the NCSX 0.5% epstot discrepancy (rtol 6e-3 fast gate vs 2.5e-10 headline) **before**
   any change that reorders floating-point sums; re-baseline tolerances deliberately.
New solvax primitives (own PRs, in value order): bounded `scan_while` with reverse rule; batched
2-D spline coefficient builder; masked segment accumulator.

Acceptance: NCSX 200x200 case >= 5x faster than today at unchanged parity tolerances; memory
< 500 MB; discrepancy explained and pinned.

## Phase 8 — Performance program across all lanes + VMEC2000/VMEC++ comparisons

1. **Benchmark matrix** (one JSON, one nightly job): {fixed, free} x {lasym on/off} x
   {tokamak, stellarator} x {vacuum, finite beta} x {CPU, GPU office A4000}: wall time, its,
   ms/it, peak RSS, and for gradient lanes: s/gradient and matvecs. Extend
   `benchmarks/baseline.json` + `render_performance_docs.py` so docs numbers regenerate.
2. **VMEC2000** (local STELLOPT build) and **VMEC++** (github.com/proximafusion/vmecpp, pip
   installable) in a separate venv: run the shared input decks, compare wout parity (iota, beta,
   Mercier, |B| spectra) and wall time. Accuracy first: any VMEX-vs-VMEC2000 discrepancy beyond
   the documented parity contract is a bug before it is a benchmark. Study their speed sources —
   VMEC2000: radial-block MPI parallelism + serial hot loops in Fortran; VMEC++: C++ with
   OpenMP-style threading and zero-restart multigrid — and write down which techniques transfer
   (radial blocking maps to batched linear algebra; their vacuum refresh cadence policies map to
   Phase 3.8).
3. **Single-device speed**: profile the fixed-boundary iteration (5.2 ms/it free, 2.3 ms/it
   fixed at ns=25) to the XLA level (perfetto trace on CPU and A4000); attack the top kernels
   (fusion breaks, transposes, callback boundaries). LASYM Jacobian 5.5x -> target <= 3x via
   chunked JVP sizing and shared trig tables.
4. **Multi-CPU**: document and test `solve_ensemble` scaling; investigate radial-block sharding
   of the 1-D preconditioner (solvax block-Thomas is the natural seam) for single-solve
   multi-core strong scaling; measure, do not promise.
5. **Multi-GPU**: only after single-GPU is clean; sharded ensembles first (embarrassingly
   parallel scans are the realistic strong-scaling story), single-solve sharding recorded as an
   explicit non-goal unless the profile says otherwise.
6. Derivative cost/memory targets recorded per lane: fixed-boundary Jacobian s/dof, free-boundary
   s/gradient, eps_eff s/gradient — all in the benchmark JSON with regressions gated in nightly.

Acceptance: benchmark matrix in CI nightly with regression gates; a docs page with measured
VMEX vs VMEC2000 vs VMEC++ parity + runtime tables; at least one demonstrated strong-scaling
curve (ensembles) and honest statements elsewhere.

## Phase 9 — CI: >= 95% coverage, < 30 min, literature-anchored  [changed-line gate DONE at 96%]

1. Coverage gate moves from changed-lines to whole-repo >= 95% (line + branch on `vmex/core`),
   with per-module floors so physics modules cannot hide behind plotting.
2. Time budget: pr-fast lanes parallelized (`-n 4`) and capped at 30 min wall total. Levers:
   the Phase 2 cache (compile time dominates test wall), smaller `full`-equivalent decks (the
   FD certificates at ns=8-16 are minutes, not tens of minutes), manifest-driven sharding across
   jobs, and pruning duplicate-coverage tests when slimming (Phase 10) — fewer, sharper tests.
3. Every new physics test cites its anchor (paper/code/analytic limit) in the docstring —
   Goodman JPP 2023 (maxJ/QI), Nemov PoP 1999 + Paul JPP 2020 (eps_eff), VMEC2000/DCON/GPEC
   (Mercier), patched STELLOPT NEO (ripple parity). Edge cases enumerated per objective:
   axis limit, lasym on/off, vacuum vs finite beta, near-rational iota, degenerate |B|.
4. `ConstructedMaximumJResidual` test set (currently zero): class==functional bit-exact with a
   monkeypatched Boozer dict; `bmns_b` forwarding regression across all five bounce classes;
   input guards; symmetric-limit bit-equivalence; Goodman g_J continuation target vs the
   independent NumPy reference lineage already in `test_qi_reference_oracle.py`; grad-vs-FD +
   traced-weights JVP + one `full` implicit boundary gradient; composition consistency with
   `ConstructedQIResidual`; NaN-not-zero degenerate-regime contract.
5. Soften the eps_eff test docstring's "NEO/STELLOPT-parity" claim until Phase 6's independent
   comparisons exist; then reinstate it with the parity table as evidence.

Acceptance: coverage >= 95% enforced; fast CI wall < 30 min; every physics test names its anchor.

## Phase 10 — Slim code, docs, and repository

1. **Scaffold disposition (execute the verdicts):** wire in `freeboundary_linear.py` (Phase 3);
   delete `freeboundary_diff.py` shim after fixing its one real caller
   (`tools/build_qi_sheet_mgrid.py:86`); delete the `vmec_jax/` package shim on schedule
   (update `pyproject.toml:108`, `test_packaging_metadata`, README); delete the
   `FreeBoundaryDiffProblem` alias; strip `_compat._env`'s legacy `VMEC_JAX_*` branch; either
   consume `recycle_drift` (Phase 3.4) or delete the doc sentences; quantify the trial-pressure
   proxy (accuracy study vs re-solved finite-beta over a beta scan) so "trial" carries an error
   bar, or fold it into the standard Mercier docs as a screening tool with stated bounds.
2. **LOC/file budget:** every PR states net LOC; refactors that delete (NEO_JAX splines via
   solvax, Fourier GEMM removing the streamed mode, examples on `FreeBoundaryProblem`) are
   preferred over additions. Test suite consolidation: merge single-assert modules into their
   physics-area files where it does not hurt manifest sharding.
3. **Docs correctness sweep** (12 verified stale claims with corrected wording ready):
   the three LASYM claims (`optimize.py:54`, `confinement.rst:285`, `confinement.rst:428`),
   `all-of-vmex.md:94` denying the free-boundary adjoint, the plot-diagnostics D_R
   misattribution, `objectives.rst:164` lasym parenthetical, `objectives.rst:418` +
   `README.md:234` trial-pressure pointers, `recycle_drift`, the dead JAX URL, the pre-rename
   SVG text, the stale cloc table. README linkcheck added to the workflow paths.
4. **README restructure:** lead with "what VMEX does that VMEC2000/VMEC++ do not" (exact implicit
   derivatives, free-boundary adjoint, LASYM optimization, differentiable objectives incl.
   eps_eff, CPU+GPU); LASYM and free-boundary single-stage sections with figures; comparison
   matrix reordered differentiators-first. New how-tos: free-boundary coils, asymmetric boundary,
   effective ripple, field-line tracing, exterior field queries (add `compute/trace/query` to
   `HOWTO_VERBS` or retitle); `first-free-boundary` tutorial. Respect the 150 KB/file media gate:
   new figures as ~1600 px WebP.
5. **Repo slimming:** re-encode the 4 oversized figures (~1.26 MB saved; then remove them from
   `GRANDFATHERED_FILES` to tighten the gate); delete orphan `ess_x_scale.png`; examples write to
   `results/`.
6. **Git history rewrite:** after the above land and a release is tagged — rewrite history to
   drop dead blobs (pre-rename `vmec_jax/` trees, superseded figures; 46 MB `.git` for an 8.8 MB
   tree). Do it once, deliberately: `git filter-repo` keeping the tagged release reachable,
   force-push, announce that users must re-clone; pin the old HEAD sha in the release notes for
   provenance.

Acceptance: zero "experimental"/scaffold language in the source; net-negative LOC for the
refactor PRs; docs claims spot-checked against code in CI (`check_docs_prose` extended with the
capability cross-check); fresh clone <= ~15 MB.

## Phase 11 — Virtual-casing performance and memory (single-stage finite-beta OOM)

Symptom: `single_stage_optimization_finite_beta.py` (the specified-boundary virtual-casing lane,
`PlasmaVacuumInterface`) is slow, and users OOM when raising boundary modes, coil count, or coil
dofs. The dense virtual-casing kernel scales as (src_nt x src_np) x (trg_nt x trg_np) and the
whole graph is differentiated with plain reverse-mode, so memory grows with every mode/coil.

1. **Measure first.** Memory/runtime matrix of one value+grad over
   {max_mode 2,4,6} x {4,8,16 coils} x {nphi,ntheta 32,48,64}: peak RSS, wall, and the XLA
   allocation report (`JAX_LOG_COMPILES` + `jax.profiler.save_device_memory_profile`). Identify
   whether the OOM is the VC kernel tableau, the ESSOS Biot-Savart pullback, or XLA temporaries.
2. **Reuse quadrature plans across optimizer iterations.** virtual-casing-jax 0.0.5 ships
   "reusable quadrature plans" (vmex #123 already requires the release); audit
   `vmex/core/virtual_casing.py` + `problem.py:650` so plan/setup construction happens once per
   stage, never per evaluation (grep for per-call `VirtualCasingJAX.setup`).
3. **Chunk the kernel.** Target-point chunking via `solvax.chunk_map`/`auto_chunk_size` inside
   `plasma_field_on_boundary` and the bnormal/pressure-balance residuals so the (src x trg)
   tableau never materializes whole; same for the exterior `VmecExtender` batched queries.
4. **Adjoint-not-autodiff for the VC map (in virtual_casing_jax, own PR).** The virtual-casing
   integral operator is linear in its surface densities: its VJP is the transposed kernel applied
   to the cotangent — implement as `jax.custom_vjp` using the same chunked kernel (and the same
   quadrature plan) instead of letting JAX differentiate through plan assembly and singular-
   quadrature bookkeeping. This is the structural memory fix: forward-sized memory in the
   backward pass, no stored tableau. Certify against the existing FD tests
   (`tests/test_virtual_casing_physics.py`, rtol 1e-4..3e-4) and add a peak-RSS regression test.
5. **Precision policy.** Optional float32 kernel evaluation with float64 accumulation for the
   smooth far-field part (digits-controlled), float64 near-singular part; gate behind the
   existing `digits` knob and certify against the f64 kernel at the configured digits.
6. **Coil-side scaling.** The ESSOS Biot-Savart pullback over many coils/dofs: batch over coils
   with `chunk_map`, verify ESSOS's segment count enters linearly not quadratically, and recycle
   the boundary-quadrature phase tables across coils (coordinate with ESSOS #58 follow-up).
7. Acceptance: value+grad at max_mode=6, 16 coils, 128 curve dofs runs in < 8 GB and the
   gradient certificate stays green; memory scaling documented in the benchmark matrix (P8.1).

## Phase 12 — Minimum-|iota| objective as the default iota floor  [DONE]

Physics: with finite beta the bootstrap/driven current can carry the transform, so a mean-iota
target is satisfiable with tiny vacuum (shaping) iota — observed as the finite-beta single-stage
stall with small vacuum iota. We want most of iota from shaping. A floor on the *minimum* of
|iota(s)| over the profile pushes the whole profile up, not just its average, and (used in the
vacuum/coil-only stages and finite-beta stages alike) forces shaping transform rather than
current-carried transform.

1. **Core objective** (`vmex/core/statephysics.py`, next to `mean_iota:348`):
   `def min_abs_iota(state, rt): iotas = _iotas_half(state, rt); return jnp.min(jnp.abs(iotas[1:]))`
   — same half-mesh convention, axis excluded. Optionally add
   `soft_min_abs_iota(state, rt, tau=0.02)` (`-tau*logsumexp(-|iota|/tau)`) for a smooth min if
   least-squares progress near ties demands it; hard min is the default. Export both through
   `vmex/core/optimize.py` imports + `__all__` and document in `docs/reference/objectives.rst`
   (wout twin: `min(|wout.iotas[1:]|)` for the `jac=None` lane, mirroring `mean_iota`'s pair).
2. **Floor hinge convention** used by every script:
   `iota_floor = lambda state, rt: jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, rt), 0.0)`
   (no `jnp.abs` wrapper needed — `min_abs_iota` is already sign-free). Keep one comment line:
   `# mean-iota alternative: opt.mean_iota targets the profile average instead of its minimum.`
3. **Rollout to every optimization script** (replace the 9 existing
   `IOTA_FLOOR - jnp.abs(opt.mean_iota(...))` hinges and the `(opt.mean_iota, IOTA_TARGET, w)`
   tuples where a floor is intended): `examples/optimization/{QA,QH,QP,QI}_optimization*.py`
   (incl. `_scipy`, `_global`, `_bootstrap`, `_DMerc_vacuum`, maxJ continuations),
   `examples/optimization/stellarator_asymmetry/*.py` (8 files),
   `examples/optimization/single_stage_*.py` (fixed and free-boundary, vacuum and finite beta).
   Scripts that genuinely want a target (not a floor) keep `mean_iota` with the comment.
4. **Tests**: unit (analytic profile: min vs mean differ, sign-flip invariance, axis exclusion);
   gradient vs FD through the implicit lane (pattern of `test_optimize_traceable_qs`); one
   integration assertion in the nightly example smokes that final `min|iota| >= IOTA_FLOOR - eps`
   for the QA vacuum example.
5. **Finite-beta shaping check** (the actual physics gate): in the finite-beta single-stage
   example, log both total `min|iota|` and the vacuum-field iota proxy (re-solve the final
   boundary at `pres_scale=0`/`curtor=0` in the wout postprocess step) and assert the vacuum
   fraction exceeds a documented threshold (e.g. >= 70%). This is what "iota from the
   stellarator, not from current" means operationally, and it becomes a regression test.

## Phase 13 — Single-stage example matrix: QA and QI, vacuum and finite beta

Deliver four verified single-stage examples (fixed-boundary lane; the free-boundary pair from
Phase 4 mirrors them): `single_stage_optimization.py` (QA vacuum — exists),
`single_stage_optimization_finite_beta.py` (QA — exists, unstall via P12 + P11),
`single_stage_QI_optimization.py` (new), `single_stage_QI_optimization_finite_beta.py` (new).

1. QI variants use `ConstructedQIResidual` + the P12 iota floor + mirror/elongation hinges
   (reuse the recipe from `examples/optimization/QI_optimization.py`), coils via ESSOS exactly
   as the QA single-stage does.
2. All four adopt: P12 min-|iota| floor, P1 flushed per-nfev progress, results into `results/`,
   `VMEX_EXAMPLES_CI=1` smoke mode, and a descent assertion in the nightly lane
   (`tests/test_examples.py` entries — executed, not text-grepped).
3. "Make sure it works" = each runs end-to-end in full mode on this Mac within a documented
   budget (record wall time in the example header), descends, and the finite-beta pair passes
   the P12.5 vacuum-iota-fraction check.

## Phase 14 — L_gradB and L_gradgradB metrics (Kappel)

The magnetic gradient scale length L_gradB = sqrt(2) |B| / ||grad B||_F (Kappel, Landreman,
Dudt, PPCF 66 025018 (2024), arXiv:2309.11342 — "the magnetic gradient scale length explains
why certain plasmas require close external magnetic coils"; implemented in DESC as the
`"L_grad(B)"` compute quantity in `desc/compute/_metric.py`; simsopt-side scripts in John
Kappel's work and in github.com/rogeriojorge repos, e.g. the single-stage/omnigenity
optimization scripts — check `single_stage_optimization` and QI/omnigenity repos for the
objective wiring pattern). vmex already has the wout lane `opt.l_grad_b` (`optimize.py:702`)
and traceable `l_grad_b_state` (`statephysics.py`), both symmetric-only.

1. **Convention lock + oracle.** Match DESC's `L_grad(B)` definition exactly (Frobenius norm of
   the full Cartesian grad B tensor, sqrt(2) normalization); add a parity test vs DESC on a
   shared wout (same pattern as `test_matches_desc_bounce1d_when_available`) and vs the
   existing vmex implementation on symmetric cases.
2. **LASYM support** for `_lgradb_state_tables` / `l_grad_b` — the Phase 5b.3 item; the wout
   arrays (`bsupumns`, `bsupvmns`, `rmns`, `zmnc`) already exist, plumbing only.
3. **L_gradgradB (new).** Second-order scale length L_gradgradB = sqrt(2 |B| / ||grad grad B||_F)
   (the k=2 member of the L_grad^k B family; verify the exact normalization against DESC master
   and the Kappel paper appendix before freezing the name). Implementation: extend the
   `_lgradb_grid` tables with second radial/angular derivatives of (R, Z, B^u, B^v) — the
   Cartesian hessian assembly mirrors `extender.py`'s interior `gradgradB` (which already
   exists for point queries); a surface-grid version over the optimization surfaces is what's
   new. Traceable, jit/vmap-clean, with FD-vs-JVP tests and min-over-surface + softmin reducers.
4. **Objectives + example.** Export `l_grad_b` / `l_grad_grad_b` state objectives (min-over-
   surfaces scalar and per-surface residual forms); add to `examples/optimization/
   QA_optimization.py` as commented-out objective tuples with one-line guidance
   (`# (opt.l_grad_b, L_GRADB_TARGET, w)  # coil-simplicity proxy, Kappel PPCF 2024`), and use
   them for real in one coil-aware example once P11 lands (their whole point is coil distance).
5. **Performance**: both metrics are pointwise algebra on existing field tables — target < 0.1 s
   overhead per evaluation at example resolutions; no new solves.

---

## Sequencing and dependencies

```
Phase 0 (unblock+merge #123)
  -> Phase 1 (examples honest)  -> Phase 2 (cache)          [independent, land early]
  -> Phase 3 (FB speed) -> 3b (certificates) -> Phase 4 (from_tuples API)
  -> Phase 5a (maxJ bugs) -> 5b (vmex LASYM) -> 5c (NEO_JAX LASYM) ; 5d (STELLOPT PRs) anytime
  -> Phase 6 (eps_eff lane; needs 5c only for the LASYM parity row)
  -> Phase 7 (NEO_JAX speed; parallel to 6)
  -> Phase 8 (perf program + VMEC2000/VMEC++; benchmarks start immediately, deep work after 3)
  -> Phase 9 (CI coverage; ratchets up as each phase lands)
  -> Phase 10 (slim/docs/history; final polish + rewrite last)

Phase 12 (min-|iota|)  -> land with Phase 1 (small, unblocks the finite-beta stall physics)
Phase 11 (VC memory)   -> after Phase 2; virtual_casing_jax custom_vjp PR can start immediately
Phase 13 (QA/QI matrix)-> needs P12 (+P11 for finite-beta comfort); before Phase 4 examples rewrite
Phase 14 (L_gradB/ggB) -> convention+oracle anytime; LASYM part rides Phase 5b; example wiring last
```

Profiling infrastructure (keep, do not commit as-is): session scratchpad scripts
`profile_lasym.py`, `fb_isolate.py`, `fb_forward_anatomy.py`, `fd_tighten.py`,
`adjoint_matvec_count.py`, `profile_stall.py` — fold the useful ones into `benchmarks/` as
deliberate, minimal benchmark entries when Phase 8 lands.

## Log

Append-only; newest last; one line per contribution (see "How to use this file").

- 2026-08-18 rogeriojorge: initial plan from the two assessment/profiling sessions on
  `rj/vmec-extender-field` (measured baselines table above; stall root-caused to pathological
  Jacobian executions P1.a + `_block_lane` recompile churn P1.b; `jax_explain_cache_misses`
  crash P1.6; FD-certificate recipe P3b from the ftol/cold-probe scan; LASYM/eps_eff/NEO/docs
  audits distilled into P5-P7, P10). Added P11-P14 (virtual-casing memory, min-|iota| floor,
  QA/QI single-stage matrix, Kappel L_gradB/L_gradgradB). Plan committed as its own PR; all
  implementation PRs branch from main after PR #123 merges (P0).
- 2026-08-18 rogeriojorge: P1.a quantified with the completed instrumented stage
  (`profile_stall.py`): jac #1-2 = 1.8-3.9 s, every later Jacobian ~2000-2240 s (~42 s/dof
  column) with residuals steady at 3.5 s — the degradation is systematic once x leaves the
  reference state, so the amortized factor-once Jacobian path is priority one of Phase 1.
- 2026-08-19 rogeriojorge: P0 [DONE except merge] — ruff E701/F541, manifest entry for
  tests/test_neoclassical.py, trial-pressure test + docs pointer, JAX autodiff URL, scipy>=1.15
  pin. Two further blockers surfaced only once ruff stopped short-circuiting the job: the quality
  lane installs mypy unpinned (2.3.1) and rejects two inferred lambdas, so
  `plotting._epsilon_effective_summary` and `extender._stored_flux_quantity` now use named
  functions. Quality and docs-linkcheck jobs are green; PR #123 is ready to merge. Dev-tool
  version pinning is unresolved and belongs in P9/P10 (an unpinned major broke a gate silently).
- 2026-08-19 rogeriojorge: P12 [DONE] — `min_abs_iota` / `soft_min_abs_iota` in statephysics,
  exported through optimize, rolled out as the default floor across 20 optimization examples
  (9 existing hinges converted, 11 mean-iota targets turned into floors, reporters now print
  min |iota|), docs updated, tests added (wout-convention parity, reducer separation and
  sign-freedom, JVP-vs-FD). Design change from the plan text: the softmin uses a
  softmax-weighted mean, not log-sum-exp — the latter sits `tau log(ns)` *below* the true
  minimum (measured -0.068 where the minimum was 1e-12), which is wrong for a non-negative
  floor. P12.5 (vacuum-iota-fraction check in the finite-beta examples) is NOT done.
- 2026-08-19 rogeriojorge: P1 [PARTIAL] — items 1, 3 and 6 landed: the five `emit=print`
  defaults now flush, `monitoring` flushes its reporter/table rows, `OptimizationMonitor`
  splits the residual SciPy already hands its callback instead of re-solving the equilibrium
  per accepted iterate, and `FunctionProblem` gained `evaluation_progress` (+ `report_interval`)
  so residual and Jacobian evaluations run under the existing elapsed-time heartbeat; enabled
  in 20 examples. P1.a (the real stall) is still open and now better measured: a full
  instrumented LASYM QA stage shows jac #1-2 at 1.8-3.9 s and every later Jacobian at
  2000-2240 s with residuals steady at 3.5 s. Mechanism located: `jacobian_rows_block`
  (optimize.py:2693) factors the raw block-tridiagonal system once and then runs a
  warm-started certifying GMRES per column via `_implicit_evolved_tangent_multi_rhs`
  (implicit.py:1944-1965) against `cfg.adjoint_tol`; that certifier is the suspect, since the
  factorization stops being a good preconditioner as the iterate moves. The decisive experiment
  (same iterate, `adjoint_tol` 1e-6/1e-4 and `implicit_jacobian_method="forward_gmres"`) is
  scripted in the session scratchpad as `jac_probe.py` and was still running at hand-off — run
  it first. Note the certifier's iteration counts cannot be read with a host-side spy (they are
  traced); expose them through the existing `LinearResponseReport` instead.
- 2026-08-19 rogeriojorge: P2 [PARTIAL] — the cache bound now scales with the filesystem
  (`min(20 GiB, max(2 GiB, 10% free))`, floor on unreadable paths) instead of a fixed 1 GiB that
  both machine-fingerprint directories sat pegged at, and `--doctor` prints the directory, its
  occupancy and the bound, flagging a cache within 5% of its cap. But the measurement corrects
  this phase's premise: with a *fresh* cache directory, one `compile_residual_and_jacobian`
  wrote only **2 entries / 768 KB** and the second process was no faster. The config is applied
  correctly (verified: cache dir, enable flag, `min_compile_time_secs=1.0`, new bound), so
  eviction was never the main story — almost nothing in this workload is cacheable XLA
  compilation above the 1 s floor. The remaining "compile" wall is sub-second XLA modules
  (filtered by `jax_persistent_cache_min_compile_time_secs=1.0`) plus Python-side tracing and
  jaxpr->MLIR lowering, which the persistent cache cannot serve at all. Next steps for this
  phase, in order: (1) re-measure on a quiet machine with `jax_log_compiles` captured from
  *stdout* (the logging handler writes there, not stderr) and split total vs summed XLA vs
  summed lowering; (2) try `VMEX_CACHE_MIN_COMPILE_TIME_SECS=0.1` and see whether entry count
  and second-run time move; (3) if lowering dominates, the lever is graph size/count in
  `_least_squares_implicit`, not the cache. Raising the bound stays correct regardless.
- 2026-08-19 rogeriojorge: P0 follow-up — the changed-line coverage gate is still red at **78%**
  and this is NOT what the plan assumed. Adding `tests/test_neoclassical.py` to the manifest does
  not help, because the coverage job combines artifacts only from the `fast`, `physics-*` and
  `device` jobs, and those run curated `selectors` (`pr-physics-core`, `pr-implicit-response`,
  ...) plus the `pr-fast` lane — the `pr-parity-*` lanes never execute on a pull request. Most of
  PR #123's new physics is reachable only from `full`-marked or optional-dependency tests, so it
  contributes zero coverage: `boozer_tables.py 0%`, `omnigenity.py 4.8%`, `maxj.py 18%`,
  `neoclassical.py 27.5%`, `freeboundary_implicit.py 57.2%`, `optimize.py 78.7%` (456 changed
  lines missing). The gate was already failing this way before this session's commits. Two ways
  forward, both Phase 9 work rather than Phase 0: add fast unit tests for those lines, or add the
  relevant test ids to the CI selectors (done for the two new `min_abs_iota` certificates, which
  had the same problem — they lived in `pr-parity-d` and never ran on PRs). Until one of those
  lands, #123 merges only with an explicit exception.
- 2026-08-19 rogeriojorge: P1.a methodology note — do NOT diagnose this by timing the Jacobian
  to completion; each data point costs ~35 min and a four-way comparison runs for hours. Bound
  the work instead. An uncertified column falls back to the block-factorization solution rather
  than raising (`_implicit_evolved_tangent_multi_rhs` masks on `report.converged`), so capping
  `adjoint_maxiter` through the public `make_problem` knob is safe and makes the cost bounded by
  construction. Then the *signature* is what to read, not the wall time: at a well-conditioned
  iterate the Jacobian time is flat in the cap because the certifier converges in a couple of
  matvecs, and where it is grinding the time grows roughly linearly with the cap. Run it on a
  deliberately small deck (ns=11, mpol=4, max_mode=1, ndof=16) — the question is how convergence
  degrades with the iterate, not how cost scales with resolution. Script:
  `jac_bounded.py` in the session scratchpad. Two dead ends recorded so nobody repeats them: a
  host-side spy on `_linear_response_report` cannot read the iteration counts (they are traced
  inside jit — expose them through `LinearResponseReport` instead), and passing `jac_solver=` to
  `from_tuples` raises (the public knob is `implicit_jacobian_method`, values
  auto/block_tridiagonal/forward_gmres/reverse_adjoint).
- 2026-08-19 rogeriojorge: P8 groundwork — the office box (`ssh office`, pop-os, 2x RTX A4000
  16 GB, 36 cores, 62 GB RAM) now has the PR branch checked out at ~/local/vmex and imports it
  cleanly on CUDA. Note the version skew against this laptop: office runs jax 0.6.2, laptop jax
  0.9.2, both with solvax 0.13.0 — any CPU/GPU comparison has to say which jax produced it. First
  matrix rows (fixed-boundary solve, problem build, compile, residual, Jacobian; symmetric and
  LASYM at ns=31/mpol=5) are scripted at /tmp/gpu_bench.py on that host and write
  /tmp/bench_cpu.json and /tmp/bench_gpu.json.
- 2026-08-19 rogeriojorge: P1.a measured, and one earlier conclusion RETRACTED. The shipped
  Jacobian lanes now carry their certifier statistics out with the rows (`_certifier_summary` /
  `_record_certifier` in optimize.py; `holder["jac_certifier_iterations"]`,
  `["jac_certifier_unconverged"]`, `["jac_certifier_worst"]`), which turns this from a
  multi-hour timing hunt into one observable run. First numbers, LASYM QA vs the symmetric case
  of the same shape: **542 certifier iterations vs 23**, zero uncertified columns in both. So
  the certifier genuinely works much harder on the asymmetric problem.
  BUT the tolerance sweep at that same iterate shows the iteration count is NOT the wall-time
  driver there: adjoint_tol 1e-6/1e-5/1e-4/1e-3 gives iterations 542/66/0/0 while the Jacobian
  takes 85.4/103.8/93.8/68.6 s — flat inside compile noise, since each build recompiles. The
  accuracy price of relaxing is negligible and plateaus immediately (relative Jacobian
  difference 3.07e-5 at 1e-5, 3.24e-5 at 1e-4 and 1e-3). Do not conclude "the certifier is the
  stall" from the iteration count alone — that was my error; the count and the cost have to be
  measured separately. `jac_split.py` (two calls in one process, so the second carries no
  compilation) isolates compile from the block assembly/factorization and the certifier, and the
  instrumented run of the real stalling iterate (`jac_real.py`, jac #2 is the 2000 s one) will
  say whether the count explodes there. Both were in flight at hand-off. If the warm cost at
  1e-6 turns out to dwarf the warm cost at 1e-4, a separate looser Jacobian-certification
  tolerance is the fix and 1e-4 is defensible on the measured accuracy. If it does not, the time
  is in `_raw_block_system`'s probe assembly and factorization, and the Schur/preconditioner
  work of Phase 3 is the lever instead.
- 2026-08-19 rogeriojorge: P1.a SOLVED, with the full chain measured. The instrumented run of the
  real stalling stage (LASYM QA, ns=21, mpol=5, 48 dofs) reads:
  `jac #1 77 s, certifier iters=542, unconverged=0` then
  `jac #2 3456 s, certifier iters=9000, unconverged=47`. So at the second accepted iterate the
  per-column certifier runs to its ceiling (adjoint_maxiter 300 x 30 restarts) and still fails
  on 47 of 48 columns; those come back NaN, the whole Jacobian is discarded for the previous
  one, and the stage spends 58 minutes making no progress with nothing on screen. That is the
  stall, end to end.
  Fix shipped: `ImplicitConfig.jacobian_adjoint_tol` (default 1e-4), applied by the two Jacobian
  lanes through a `jac_cfg = dataclasses.replace(cfg, adjoint_tol=cfg.jacobian_adjoint_tol)`;
  uncertified columns now raise a RuntimeWarning naming the knob. Measured at the seed iterate,
  warm (compile excluded by calling twice in one process): **13.3 s -> 2.8 s, certifier 542 -> 0
  iterations, relative Jacobian change 3.2e-5**. The rationale is that the two tolerances have
  different consumers — a scalar gradient feeds quasi-Newton curvature accumulation, a
  least-squares Jacobian only points a trust-region step.
  IMPORTANT scoping lesson: relaxing the tolerance inside the shared
  `_implicit_evolved_tangent_multi_rhs` helper broke
  `test_block_response_forward_transpose_and_fd` (a genuine transpose/FD identity at rtol 2e-8,
  measured error 5.2e-5). The helper keeps `adjoint_tol` for its public callers; only the
  Jacobian lanes relax. Do not push the relaxation down into the helper.
- 2026-08-19 rogeriojorge: P8 warning about the office box — do NOT trust remote numbers without
  pinning the import. `~/local/vmex` is NOT what `import vmex` resolves to there: an editable
  install points at a second checkout, `/home/rjorge/vmex_profile/vmex`, and a script invoked as
  `python3 /tmp/bench.py` puts `/tmp` (not the cwd) on `sys.path[0]`, so the stale tree wins.
  Two rounds of benchmark numbers were silently produced from it, including a bogus
  `NotImplementedError: QuasisymmetryRatioResidual traceable evaluation supports lasym = False
  only` that exists in no current source, and a 575 s symmetric Jacobian. Always run remote work
  as `cd <checkout> && PYTHONPATH=<checkout> python3 ...` and assert `vmex.__file__` in the
  output. Clearing `__pycache__` does not help — it was never the cache.
- 2026-08-19 rogeriojorge: P1.a OPEN DECISION for whoever picks this up. The
  `jacobian_adjoint_tol = 1e-4` default is committed and the speedup is real and large — at the
  degraded iterate the warm Jacobian goes 395 s -> 28 s (certifier 9000 iterations and 47/48
  columns uncertified, versus 207 iterations and all certified). But two things must be settled
  before calling it finished:
  (1) `tests/test_optimize.py::test_least_squares_implicit_jac_solver_block` now FAILS. It pins
  the block lane against the per-dof GMRES lane at `rtol=1e-6`, and that guarantee genuinely
  weakens when both lanes certify to 1e-4 (each is within 1e-4 of exact, so they may differ by
  2e-4). This is the trade-off surfacing honestly, not a flaky test. Do NOT relax the assertion
  just to make it green — decide the policy first, then update the test AND its docstring to
  state the new contract.
  (2) The loose-vs-tight difference is 3.2e-5 at a clean iterate but 1.4e-1 at the degraded one.
  That large number is almost certainly comparing against a broken reference: at that iterate the
  tight solve fails its certificate, returns NaN columns, and the caller falls back to the
  previous Jacobian — so "tight" there is not a Jacobian at all. `jac_accuracy.py` in the session
  scratchpad settles it by comparing BOTH against a central finite-difference column and printing
  the uncertified/NaN counts for each; it was still running at hand-off. If loose matches FD and
  tight does not, the default is not merely faster but more correct, and the block-vs-GMRES test
  should be re-pinned at the Jacobian tolerance. If loose does NOT match FD, reconsider: keep the
  tight default and make the fix adaptive instead (start tight, relax only when the certifier
  reports uncertified columns), which the new `holder["jac_certifier_unconverged"]` counter makes
  straightforward.
  Everything else in Phase 1 (flush, heartbeat, monitor double-solve, examples) is done and green.
- 2026-08-19 rogeriojorge: P1.a — a central finite difference is NOT a usable arbiter for this
  Jacobian, do not spend time on it. Measured: at the degraded iterate, column 0 against a
  central FD came out 8.3e-1 relative for the tight Jacobian and 2.6e0 for the loose one. Both
  being of order one says the FD is wrong, not the Jacobians: each probe re-solves the
  equilibrium through the perturbation warm start, so the difference is dominated by solver
  endpoint noise rather than by the derivative (the same endpoint-noise effect already
  documented for the coupled free-boundary certificate in P3b). Same run also showed the tight
  solve reaching 9000 iterations with 9 uncertified columns and taking a derivative fallback,
  while 1e-4 certified everything in 207 iterations — i.e. at that iterate the tight result is
  the one that is not trustworthy. The arbiter that does work is a tight solve given enough
  iteration budget to actually certify every column (`jac_arbiter.py`: tol=1e-8,
  adjoint_maxiter=4000, and it checks `unconverged == 0 and derivative_fallbacks == 0` before
  accepting itself as ground truth), then comparing 1e-6/1e-4/1e-3 against it. That was running
  at hand-off; its result decides the open question above.
- 2026-08-19 rogeriojorge: P9/P0 changed-line coverage gate **78% -> 96%** (464 -> 84 missing of
  2144 changed lines), merged. The cheap wins dominated exactly as hoped: most of it came from
  running modules the pull-request lanes never selected (`test_boozer_tables`, `test_maxj`,
  `test_optimize_traceable_qs`, `test_virtual_casing_api` into `pr-physics-field`; `test_doctor`
  and `test_neoclassical` into `pr-fast`), not from new tests. Zero `full` markers were demoted —
  none of the candidates ran in under 20 s. One genuinely new certificate was worth its cost: a
  percent-level cross-check of the boundary-Schur adjoint against the certified coupled GCROT
  adjoint on a shared converged root (202 s, the two gradients agree to 0.53%), which is the
  first fast-lane coverage of that path; note it needs `adjoint_tol=1e-5`, because at 1e-9 the
  Schur lane's own certification does not converge within 3017 Krylov iterations. Estimated CI
  wall goes ~17.5 -> ~22 min, inside the 30-minute budget. When merging, the per-node maximum-J
  entries in `pr-physics-core` were dropped: the whole module now runs in `pr-physics-field`, so
  listing individual ids only duplicated it.
  Two DEAD-CODE findings for Phase 10, both confirmed unreachable rather than merely untested:
  `implicit.py` `_raw_block_apply`'s `factors is None` guard and its iterative-refinement loop
  (`refinements` is never passed non-zero anywhere in the tree), and `omnigenity.py:328,330-331`
  (the in-body LASYM mirror branch, unreachable because `boozer_bmnc_state` returns early through
  `_boozer_lasym_state` for asymmetric states — the maximum-J agent independently found the same
  thing). Delete both rather than write tests for them.
  Of the 84 lines still missing, the honest reasons are recorded: `optimize.py` closures that
  need a solve-backed `VmecProblem`, a `freeboundary_implicit.py` m=1 edge-pairing branch
  unreachable on the only free-boundary deck available (DIII-D, `ntor=0`), extender parameter-VJP
  fallbacks, `FFMpegWriter`, and the successful `import neo_jax` line.
- 2026-08-19 rogeriojorge: P1 [DONE]. The open decision is closed by measurement, and the
  earlier worry about it was based on two of my own mistakes, both now fixed.
  Final sweep at the LASYM QA iterate (warm, compile excluded by calling twice in one process),
  `jacobian_adjoint_tol` against a 1e-7 reference:

  | tol | warm jac | certifier iters | uncertified | relative vs 1e-7 |
  |---|---|---|---|---|
  | 1e-7 | 23.1 s | 962 | 0 | reference |
  | 1e-6 (old behaviour) | 6.6 s | 542 | 0 | 1.8e-5 |
  | **1e-4 (new default)** | **1.2 s** | **0** | 0 | **4.4e-5** |
  | 1e-3 | 1.2 s | 0 | 0 | 4.4e-5 |

  So the default is 19x faster than a 1e-7 Jacobian and 5.5x faster than the old 1e-6 one, for
  4.4e-5 relative error — and the error *plateaus* there, because at 1e-4 the certifier accepts
  the block backsolve unchanged (0 iterations) and 1e-3 buys nothing further. At the degraded
  iterate the same change is the difference between converging in 207 iterations and running to
  9000 with 47 of 48 columns uncertified over 58 minutes.
  MISTAKE 1 (retracted): I believed the failing
  `test_least_squares_implicit_jac_solver_block` showed the tolerance legitimately weakening
  block-vs-GMRES agreement, and nearly relaxed its assertion. Measured in that test's own case
  the two lanes agree to **1.1e-16**. The failure was unrelated.
  MISTAKE 2 (the real bug, fixed): carrying the tolerance as
  `dataclasses.replace(cfg, adjoint_tol=...)` gave the Jacobian lanes a second config identity,
  which misses every cache keyed on the original and rebuilt the runtime *inside* the traced
  Jacobian — a TracerArrayConversionError out of `setup.radial_grids`, and the CI
  implicit-response lane. A tolerance is a number: it is now threaded as `rtol=` through
  `_adjoint_solve`, `_adjoint_acceptance`, and the multi-RHS certifier. General lesson for this
  codebase: never manufacture a new `ImplicitConfig` on a traced path.
- 2026-08-19 rogeriojorge: P8 first verified remote row (office box, 36-core CPU, jax 0.6.2,
  import asserted as /home/rjorge/local/vmex): symmetric ns=31/mpol=5/max_mode=2, solve 2.9 s,
  build 56.7 s, compile 754.5 s, residual 0.52 s, **Jacobian 516.2 s**. That checkout predates
  the Jacobian-tolerance fix, so it is a pre-fix baseline — but a Jacobian that costs ~2-10 s on
  an Apple laptop costing ~500 s on a 36-core Linux box is a real finding for the performance
  program, and the first thing to re-measure there after the fix lands. Compile at 754 s on that
  machine also dwarfs the laptop's ~40 s.
- 2026-08-19 rogeriojorge: P10 partial — the two unreachable paths found during the coverage work
  are deleted (`58166f57`): `_raw_block_apply`'s iterative-refinement loop, whose `refinements`
  count no caller has ever passed, and the poloidal-mirror branch for asymmetric states in
  `boozer_bmnc_state`, unreachable because those states return through `_boozer_lasym_state`
  about thirty lines earlier. Note the distinction that matters when clearing the rest of the
  Phase 10 list: an unused *feature* is dead and goes, but a *precondition guard* is not the same
  thing even when uncovered. `_raw_block_apply`'s `factors is None` raise states a real contract
  for systems built with `factor=False`, so it stays and now has a three-line test rather than
  being deleted for the coverage number. Do not treat "uncovered" as a synonym for "dead" when
  working the remaining items.
- 2026-08-19 rogeriojorge: P6.8 plot fix done (`74cafb91`) — the eps_eff panel now picks its
  scale from the data. Measured on `wout_QA_optimized.nc`: the profile spans 1.64e-3 to 4.37e-3,
  a factor of 2.67, i.e. well under one decade, which is the normal case for an *optimized*
  configuration. Matplotlib's log autoscale snapped that to a single decade tick, flattening the
  curve against a limit and hiding the radial minimum — precisely what the panel is read for.
  The rule is now: keep the logarithm only when the profile actually spans a decade or more,
  otherwise linear with scientific tick labels, and pad the limits by 8% of the range so the
  extrema sit inside. After the change the same case gives 7 tick labels with the minimum
  comfortably inside the axis. Both branches are pinned by tests (the pre-existing test uses a
  3-decade `geomspace` and still asserts log; the new one uses a 2.7x span and asserts linear,
  the minimum inside the limits, and at least four tick labels).
  Still open in this area: more surfaces in the summary panel, which is gated on the eps_eff lane
  being fast (P6), not on plotting.
- 2026-08-19 rogeriojorge: P8 first complete verified CPU rows, office box (36-core, jax 0.6.2,
  import asserted, single process, ns=31/mpol=5/max_mode=2, checkout at 1edffddc so PRE the
  Jacobian-tolerance fix):

  | case | ndof | solve | build | compile | residual | Jacobian |
  |---|---|---|---|---|---|---|
  | symmetric | 24 | 2.9 s | 56.7 s | 754.5 s | 0.52 s | 516.2 s |
  | LASYM | 48 | 7.6 s | 32.5 s | 1055.9 s | 0.69 s | 869.4 s |

  Two readings. (1) The LASYM/symmetric Jacobian ratio is **1.68x**, which independently
  reproduces the 1.8x per-nfev ratio measured on the laptop — the asymmetric cost model holds
  across machines, so LASYM is genuinely not the problem. (2) The absolute numbers are the
  problem: the same Jacobian takes 1.8 s (symmetric) and 9.8 s (LASYM) on an Apple laptop and
  516 s / 869 s here, and compile is 754-1056 s against roughly 40 s. That is a 90-280x machine
  gap on identical code, and it is what a cluster user would actually experience.
  Hypotheses, cheapest first: **jax 0.6.2 versus 0.9.2** (three minor versions of XLA:CPU work,
  and the laptop is the newer one — most likely explanation, test by upgrading jax on that host
  and re-measuring, but that changes someone's environment so ask first); thread oversubscription
  on 36 cores for a memory-bound blocked solve (probe running now under `taskset -c 0-7`, which
  is non-invasive); and Apple Silicon's unified memory favouring this access pattern. Settle
  which before drawing any CPU-vs-GPU conclusion from that machine, and re-measure after the
  Jacobian-tolerance fix propagates there.
