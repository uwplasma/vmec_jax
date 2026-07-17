# Pre-VMEX plan (live)

**This file is the authoritative, self-contained plan for the remaining work before the
VMEX rename.** It exists so any session (or collaborator) can pick up the campaign without
prior conversation context. Update it as items complete; keep it honest — measured numbers
only, with the metric that produced them. The historical/architectural plan is `plan.md`
(untouched); this file is the *live* delta.

Last updated: 2026-07-16 (post PR #29).

---

## 0. Goal

Bring vmec_jax to the state where the only remaining step is the atomic `vmec_jax → VMEX`
rename (R21 in `plan.md` §13): research-grade differentiable VMEC with (a) honest,
polished showcases (README + docs), (b) precise-or-honestly-bounded optimization results
for all confinement classes, (c) a credible single-stage plasma+coil story, (d) a speed
and parallelization story. **The VMEX rename itself is out of scope until the user gives
an explicit go-ahead.**

## 1. Where we are (verified facts, 2026-07-16)

Main at `ed4ac7ac`. Five PRs merged this campaign:

| PR | What landed |
|----|-------------|
| #25 | **Implicit adjoint validated correct.** The collaborator's AD-vs-FD concern was a naive-FD artifact: a full re-solve at `p±h` re-forms the solver's convergence logic (preconditioner/tcon/m=1 branch/dof mask) — an O(1) path perturbation that can sign-flip FD for solver-sensitive metrics (iota at ncurr=1, mirror, well, QI). The adjoint linearizes the *frozen* fixed point, which is the physical gradient. `frozen_path_directional_fd` (exported in `vmec_jax.core.implicit`) is the correct FD reference; regression test `test_iota_edge_gradient_vs_frozen_path_fd` locks in adjoint == frozen-path FD to ~2e-6 on the li383 m=1 modes (naive FD reads +0.045 where truth is −0.773). |
| #26 | **vmec_jax is coil-agnostic.** `core/coils.py` deleted; free boundary consumes an `MgridField` or a plain `xyz→B` callable. Coils live in ESSOS (`essos.coils.Coils`, `CreateEquallySpacedCurves`, `BiotSavart`, `loss_coil_length/curvature/separation`). CLI `--coils` = optional-essos → `to_mgrid` → `read_mgrid`. Equilibrium plotting stays in vmec_jax (moving it would be a circular dep). |
| #27 | CI parity shard timeout 15→20 min; **coverage gate confirmed ≥95%** post-refactor. (Parity shard c runs 15+ min with coverage instrumentation — if it ever cancels again, the gate SKIPs; rebalancing shard c is the known follow-up.) |
| #28 | **Single-stage showcase v1 (warm start)** — `examples/single_stage_essos_coils_opt.py`, `readme_single_stage.png`, README section, full-marked smoke test. Boundary modes + ESSOS coil *currents* co-optimized from the LP-QA warm start: vacuum J 6.6×, finite-β 2.4×, FD-validated. **Known deficiency: warm start ⇒ initial vs final visually indistinguishable. Being replaced by item 2 below.** |
| #29 | **Docs completeness** — new `docs/confinement.rst` (Boozer coordinates, Landreman–Paul two-term QS residual, omnigenity/Goodman constructed-QI incl. the metric-fidelity caveat, Mercier, magnetic well, ballooning, Redl bootstrap; all with equations) + frozen-path gradient-checking section in `docs/algorithms.rst` + tutorial entries for the ESSOS-coils scan and bootstrap loop. Clean `sphinx -W` build. |

**Optimization precision ground truth** (metrics that count: `QuasisymmetryRatioResidual`
on 8–10 surfaces for QS; `quasi_isodynamic_residual_from_wout` (8 surfaces) for QI —
**never** the traceable `QIResidual.total()` for *reporting*, it over-reports up to ~3000×
below ~1e-3):

| class | bundled deck (shipped) | office-refined deck — final verdict (see Item B) |
|---|---|---|
| QA (nfp 2) | 1.05e-6 — precise | (already precise) |
| QH (nfp 4) | 5.83e-5 — precise | (already precise) |
| QP (nfp 2) | **3.3e-2** (folded 2026-07-16; 10-surface figure metric) | ACCEPTED — physical shape |
| QI nfp1 | 1.27e-2 | REJECTED (2.5e-3 but extreme elongation) |
| QI nfp2 | 7.05e-3 | REJECTED (1.7e-3 but extreme elongation) |
| QI nfp3 | 3.98e-3 | no gain — bundled kept |
| QI nfp4 | 3.16e-3 | REJECTED (1.8e-3 but R0→11 m, mirror ratio ~8) |

Infrastructure facts: office box = `ssh office` (pop-os, 36 cores, venv
`~/.venvs/vmecjax`, current source pushed to `~/vmec_jax_push/`, long jobs in tmux).
Commits authored by rogeriojorge only, **never** a Claude co-author trailer. Open PR #22
(mirror geometry) is a separate pre-existing workstream — leave it alone.

## 2. Item A — Cold-start single-stage vs two-stage comparison — **DONE (PR #31)**

**Outcome (2026-07-17, measured):** three-way README comparison shipped —
two-stage | +single-stage polish | cold single-stage, vacuum and finite β.
Polish (the arXiv:2302.10622 "stage 3" pattern) cuts ⟨|B·n|⟩/⟨B⟩ 33 % (vacuum)
/ 17 % (β≈1.5 %) below two-stage at held QS and on-target iota in 10–30 min;
the cold column is the honest 50-iteration record + the dramatic figure.
Engineering lessons recorded in the example/commits: rotating-ellipse seed
kick (a same-sign kick has iota≈0 and the pressure-loaded seed limit-cycles on
any single grid — β uses the multigrid ladder), scipy L-BFGS-B ftol is tested
against max(|f|,1), adjoint_tol 1e-8 suffices, and jit over the joint
objective is blocked by host conversions in the FBD/im.run stack (→ Item I.6).
Original design notes kept below for reference.

**Why:** the PR #28 showcase warm-starts from LP-QA, so the initial→final change is
invisible and the comparison proves little. User directive (2026-07-16): redo it as a
genuine cold start and compare against a sequential two-stage pipeline so the single-stage
advantage (smaller B·n / better coils at comparable quasisymmetry) is visible — in
**vacuum and finite β**.

**Design — aligned with the literature (§5).** The canonical protocol is R. Jorge et al.
PPCF 65, 074003 (2023), arXiv:2302.10622 (`J = J_plasma + w_coils·J_coils`); the
cold-start staged-mode-release convention is Jorge–Giuliani–Loizu arXiv:2406.07830; the
coil budget/reporting conventions are Wechsung et al. PNAS 2022 + the simsopt
`stage_two_optimization.py` example. Notably, the 2023 paper's own single-stage runs were
warm-started from two-stage results and demonstrated **vacuum only** — a general
finite-β single-stage demonstration is a near-vacant niche (§5), so the finite-β column
is the flagship.

- **Common cold seeds.** Plasma: `examples/data/input.minimal_seed_nfp2` (circular torus
  R0=1, a=0.2, nfp=2, vacuum; add the parabolic pressure + `pres_scale` for the finite-β
  variant, target ⟨β⟩ ≈ 1–2%). Coils: `essos.coils.CreateEquallySpacedCurves(n_curves=4,
  order=5, R=1.0, r=0.5, nfp=2, stellsym=True)` — the simsopt-canonical 4 base coils /
  order 5 / R1=0.5 — uniform currents ≈ 2.7e5 A (matches the phiedge-implied ~0.6 T;
  verified against `essos.fields.BiotSavart` to machine precision), current[0] fixed to
  break the vacuum scale degeneracy.
- **Two-stage baseline (same budgets as single-stage).** Stage 1: fixed-boundary QS
  optimization from the circular seed (`least_squares(jac="implicit")`, QA helicity (1,0),
  aspect 6, mean-iota target 0.42 — the existing `QA_optimization` recipe). Stage 2:
  coil-only optimization (curve dofs + currents) minimizing the virtual-casing B·n
  residual on the *fixed* stage-1 boundary + the regularization set below. Cheap — no
  equilibrium re-solves in stage 2.
- **Single-stage.** From the same seeds, one `jax.value_and_grad` over
  [boundary Fourier modes (staged release m,|n| ≤ j per arXiv:2406.07830), coil curve
  dofs, currents(1:)] of
  `J = w_qs·Σ qs.residuals_state² + w_bn·bnormal + w_asp·(A−6)² + w_iota·(⟨ι⟩−0.42)² +
  coil penalties`, scaled L-BFGS-B. Coil penalties per simsopt/PNAS conventions:
  length target (per-coil L_max with quadratic penalty), κ_max ≈ 5, MSC ≈ 5,
  d_cc ≥ 0.1, d_cs ≥ 0.3 (essos has `loss_coil_length/curvature/separation/
  surface_distance`). `w_bn` swept over ~{1e2, 1e3, 1e4} — the 2023 paper does not
  publish its weights, so publishing ours is itself a contribution.
- **Traceable pieces (all verified 2026-07-16):**
  `QuasisymmetryRatioResidual.residuals_state(state, rt)`;
  `FBD.FreeBoundaryDiffProblem.bnormal_objective(callable)` with `plan_vc_precision`
  frozen at the seed; `essos Curves(dofs)`/`Coils` constructed inside the trace;
  hand-rolled Biot-Savart == `essos.fields.BiotSavart` to 1e-16 (scratchpad
  `cold_start_probe.py`). One caution from the warm-start probe: unscaled steps can
  produce `VmecJacobianError` trial states — keep the per-dof scaling matrix `D` and
  bounds; the optimizer guard penalizes invalid trials.
- **Report (the credibility recipe from arXiv:2302.10622 §comparison + PNAS).** For BOTH
  approaches and BOTH β values, evaluate on the **coil-realized** equilibrium — a
  free-boundary vmec_jax re-solve with the final coils (stronger than the QFM-surface
  proxy the 2023 paper used): (1) QS residual on the realized surface, (2) ⟨|B·n|⟩/⟨B⟩
  and max|B·n|/|B|, (3) per-coil length / κ_max / MSC / min-distances vs budgets,
  (4) Poincaré overlay of the coil field vs the target boundary (essos tracing),
  optionally (5) Boozer |B| before/after. Figure: 2 columns (vacuum, finite-β) ×
  rows (boundary initial vs two-stage vs single-stage; final coils + LCFS 3-D; B·n or
  Poincaré). README table: two-stage | single-stage per metric. Expected qualitative
  result per the literature: single-stage trades a slightly worse *idealized* QS for a
  much better *realized-with-coils* field (in the 2023 paper: QA f_QS on the realized
  surface 1.7e-2 → 9.1e-3 two-stage → single-stage).
- **Compute plan.** Feasibility probe first (cold-start joint gradient incl. curve dofs —
  running as of 2026-07-16). Full runs on the office box in tmux (each single-stage case
  ~45 s/eval × O(200–400) evals ⇒ hours; stage 1 ~15 min; stage 2 minutes). Keep a
  `VMEC_JAX_EXAMPLES_CI=1` smoke path for the test.
- **Deliverable:** `examples/single_stage_vs_two_stage.py` (supersedes/joins
  `single_stage_essos_coils_opt.py`), updated `make_single_stage_figure`, README section
  rewrite (drop the warm-start 6.6×/2.4× framing), smoke test update — one PR.

## 3. Item B — Deck refinement fold: **outcome recorded, QI redo needs shape guards**

**Resolved 2026-07-16:** the refined **QP** deck was folded
(`benchmarks/opt_decks/input.qp_optimized`, QS 4.5e-2 → **3.3e-2** on the figure's
10-surface metric; physical shape, cleaner vertical Boozer contours). The refined
**QI nfp1/nfp2/nfp4 decks were REJECTED despite better residuals** (2.5e-3 / 1.7e-3 /
1.8e-3): visual inspection of the regenerated figure showed the refinement gamed the
residual with degenerate shapes — nfp1/nfp2 extreme elongation (pancake sections, 3-D
surface renders as a ribbon), nfp4 scale drift to R0≈11 m with |B| 2.4–18.5 T (mirror
ratio ~8). Root cause: the refinement objective had aspect+iota terms but **no elongation
penalty, no mirror-ratio cap, no R0 pin**.

**Standing lesson (also in docs/memory):** wout-metric + aspect validation is NOT enough
to accept a deck; also check elongation, mirror ratio, R0, and eyeball the regenerated
figure. **Optional QI redo** (only if better QI decks are wanted before VMEX): rerun the
office refinement with elongation ≤ ~4–5, mirror-ratio target ≈ 0.2, R0 fixed, then
re-validate. Raw rejected decks remain at `office:~/vmec_jax_push/out/` for forensics.

## 4. Item C — README + docs polish (findings list; most fixed 2026-07-16)

From the full-repo review sweep (fixed in the `deck-fold-polish` branch unless noted):

- [x] QP numbers reconciled across README table/caption (3.3e-2, provenance stated) and
  `docs/optimization.rst` (9.4e-2 = single-call budget; 3.3e-2 = shipped deck).
- [x] Garbled caption parenthetical in the README optimization figure caption.
- [x] `examples/README.md` now lists `single_stage_simultaneous_opt.py` and
  `single_stage_essos_coils_opt.py`.
- [x] `docs/references.rst`: added Landreman–Paul 2022, Goodman 2023, Cary–Shasharina
  1997, Dudt 2024, Redl 2021, Landreman–Buller–Drevlak 2022, Jorge 2023 single-stage,
  Jorge–Giuliani–Loizu 2024, Wechsung PNAS 2022 (entries 19–27).
- [x] `disabled/` (stray JAX compile cache) gitignored.
- [ ] Orphaned figures under `docs/_static/figures/` (11 Jul-8 PNGs + stray
  json/csv; ~1.2 MB) — remove in the Item A PR (repo has a 40 MiB CI size check).
- [ ] Stale merged remote branches to prune: `ci-parity-timeout-20`, `coils-to-essos`,
  `docs-completeness`, `single-stage-showcase`, `validate-implicit-adjoint-frozen-path`.
- [ ] README single-stage section still has the warm-start 6.6×/2.4× framing — replaced
  by Item A (do not patch piecemeal).
- [ ] QH table row wall-time "25.5 min (ladder)" duplicates the QA ladder number and has
  no independent provenance — re-measure or drop the number during Item A's README edit.
- [ ] `make_readme_figures.py --only` takes a free string (typos silently no-op) — add
  `choices=` when next touching that file.

## 5. Item D — Literature alignment (2026-07-16 deep dive; full report in session notes)

Key facts that shape the roadmap:

- **Canonical single-stage line** (all SIMSOPT/VMEC, vacuum): R. Jorge et al. PPCF 2023
  (arXiv:2302.10622) — `J = J₁ + ω_coils·J₂`, J₂ = quadratic flux + length/curvature/
  MSC/distance/arclength penalties; comparison via f_QS evaluated on a QFM surface built
  from the final coils; headline: QA realized f_QS 1.7e-2 (two-stage) → 9.1e-3
  (single-stage); QI: two-stage failed outright, single-stage 3.2e-3. Follow-ups:
  arXiv:2406.07830 (cold start, staged mode release, 1–3 coils per half period),
  arXiv:2603.11699 (stochastic single-stage, 2026). Giuliani et al.: near-axis and
  Boozer-surface single-stage, global QUASR database (~370k devices). QUADCOIL
  (arXiv:2408.08267, arXiv:2510.16243): winding-surface coil proxy inside stage-1
  ("quasi-single-stage") — the field is converging on differentiable coil–plasma coupling.
- **Finite-β single-stage is essentially unpublished** as a general capability: 2302.10622
  formulated the virtual-casing extension but demonstrated vacuum only; nearest neighbors
  are Smiet et al. 2025 (SPEC, island healing) and one 2026 DESC toy case
  (arXiv:2605.02139). **vmec_jax's differentiable virtual-casing free-boundary path is
  ahead of the published state of the art here — make finite-β the flagship claim.**
- **Positioning for README/paper:** (1) only *VMEC-parity* differentiable equilibrium
  (DESC is differentiable but a different discretization; VMEC++ is modern C++ but not
  differentiable, no GPU); (2) first end-to-end-differentiable single-stage including
  finite β; (3) O(1)-memory implicit adjoint vs DESC's memory-heavy AD.
- **Comparison conventions to follow:** report f_QS on the coil-*realized* surface
  (we can free-boundary re-solve — stronger than QFM), ⟨|B·n|⟩/⟨B⟩ and max|B·n|/|B|
  (PNAS 2022: 3.3e-3 at L=18 m budget for LP-QA, 4 coils/order-5), coil
  length/κ/MSC/distance tables vs budgets, Poincaré overlays. ConStellaration
  (arXiv:2506.19583) is the template for feasibility-gated benchmark scoring if we ever
  publish a benchmark suite.
- **Parallelization prior art for Item G:** DESC GPU/JIT docs; QUASR's global-to-local
  massively-parallel ensembles; GPU-batched NLP (arXiv:2606.26341) — the natural vmec_jax
  win is `vmap`-batched equilibria/optimizations (weight sweeps, multi-start), not
  intra-solve scaling.

## 6. Item E — Objectives README row (#37) — premise corrected 2026-07-16

New README row/section: from a QA-ish seed optimize combinations of L∇B, self-
consistent bootstrap (Redl) at finite β, DMerc > 0 (Mercier), magnetic well
(vacuum), higher iota at fixed aspect, lower aspect at fixed iota.

**Corrected premise (audit):** NOT all objectives are traceable. Traceable
(implicit-adjoint-ready): `magnetic_well` (optimize.py:559), Redl bootstrap
(`RedlBootstrapMismatch`), iota/aspect targets. **FD-only (wout-engine, host
NumPy)**: `l_grad_b` (optimize.py:590) and `d_merc` (optimize.py:575 →
nyquist.mercier_and_jxb) — `jac="implicit"` explicitly rejects them
(optimize.py:1439). Options per objective: (a) run those terms under `jac=None`
FD at honest cost, (b) build traceable versions (L∇B from the state-field chain
is plausible; a traceable Mercier via nyquist-in-JAX is a big lift), or
(c) scope the row to the traceable set + bootstrap. Decide when starting;
(a) for DMerc + (b) for L∇B is the likely sweet spot. After Items A–C.

## 7. Item F — Speed deep-dive (#36) — kickoff measurements done 2026-07-16

Done so far: XLA compile flags (~12% compile win), residual-jit inside adjoint, 2D
preconditioner assessment (opt-in), R25 gradient stack (15.7× vs FD Jacobian).

**Measured baseline (audit, M-series CPU, minimal_seed ns=31):** forward solve
warm 2.13 s (2.78 ms/iter); **warm implicit gradient of ONE scalar = 32.4 s eager
/ 22.5 s under `jax.jit` (~10-15 forward solves)**; host-callback overhead
negligible (cached forward 0.02 s). The adjoint GMRES (`adjoint_tol=1e-11`,
maxiter 300) dominates. Priorities, in measured-value order:
1. **Adjoint budget — tolerance ruled out (measured 2026-07-17)**: sweeping
   `adjoint_tol` 1e-13→1e-6 leaves the warm gradient wall time FLAT (solovev
   ~4 s, li383 ~7 s at every tolerance) while accuracy degrades as expected —
   the preconditioned GMRES hits near-machine residual within its first
   Arnoldi cycles, so the cost is the FIXED matvec work (each matvec = one
   residual linearization), not the convergence criterion. Remaining levers,
   in order: (a) cheaper matvecs (jit/donate the residual linearization),
   (b) fewer matvecs via cross-eval warm-starting/recycling (GCROT recycle —
   measure, default it if it wins), (c) the shipped multi-RHS batching for
   multi-objective campaigns.
2. **NESTOR loop batching** (freeboundary.py:917-973): per-iteration host
   dispatch with several device→host syncs + per-iteration runtime rebuilds; run
   the `nvacskip` iterations between vacuum updates as one jitted block and move
   the 0.9 constraint damping into the traced carry (the vacuum step itself is
   already fused).
3. **Document `jax.jit(jax.grad(...))`** — measured 30% on the implicit gradient;
   docs/examples currently don't say it.
4. Re-run `benchmarks/run_baseline.py` + refresh the performance figure/claims;
   commit a `profile.json` artifact so "last measured" exists in-tree.

## 8. Item G — Parallelization (#41)

**Reality check (audit): fully greenfield.** No pmap/shard_map/sharding anywhere
in `vmec_jax/`; the only vmap is internal to ballooning; `core/device.py` is a
single-device selection policy. Target: multi-CPU strong scaling (local
measurement) + multi-GPU for solve and optimization. Concrete first steps: (1) shard the per-dof implicit-Jacobian columns of
`least_squares(jac="implicit")` across devices (`jax.pmap`/`shard_map` over the dof axis —
embarrassingly parallel); (2) batch multi-case optimization (vmap over decks) on one GPU;
(3) strong-scaling study of a single solve (the radial dimension is the natural axis;
honest assessment — the per-iteration FFTs are small, so intra-solve scaling may be
poor and the honest story is parallelism *across* gradient columns / cases); (4) document
in README + docs. Literature notes (§5) may add relevant prior art.

## 9. Item I — Code health backlog (2026-07-16 full audit; measured findings)

Two deep audits (source code + roadmap gaps) produced these, ranked. Each is
small-to-medium and independently PR-able; batch 1-4 as one "robustness" PR.

1. [x] **Typed errors through `pure_callback`** (HIGH) — **DONE 2026-07-17**
   (robustness PR): a failing host solve inside `im.run`/`solve_implicit`
   surfaced as a raw 3.7-KB `JaxRuntimeError` with the typed
   `VmecConvergenceError`/`VmecJacobianError` lost. Now `_host_solve_and_mask`
   stashes the typed exception in the `_HOST_ERROR` slot and the shared
   `_callback_solve` call site re-raises it (`from None`); tested in
   `test_implicit_grad.py::test_typed_error_through_pure_callback` (message
   3681 -> 24 chars).
2. [x] **Test the `least_squares` zero-crash penalty paths** (HIGH) — **DONE
   2026-07-17**: `tests/test_optimize_penalty.py` covers all four
   except-bodies (FD-lane fun, implicit-lane fun, jac last-valid fallback,
   final diagnostic cold re-solve) with deterministic poisoned-solve
   campaigns on solovev.
3. [x] **mypy debt + lint gate** (MED) — **DONE 2026-07-17**: the 5
   `var-annotated` caches in implicit.py and the `dict-item` in optimize.py
   annotated, the pyproject override block deleted, and a fail-fast
   `lint` CI job (ruff + `mypy vmec_jax` on py3.12) added.
4. [x] **FD-validate the multigrid implicit gradient directly** (MED) —
   **DONE 2026-07-17**: `test_multigrid_gradient_vs_frozen_path_fd`
   (`im.run(multigrid=True)` through a genuine ns 5 -> 11 solovev ladder vs
   `frozen_path_directional_fd`, rel <= 1e-6).
5. **"Choosing an entry point" docs** (MED): `solver.solve` vs `solve_multigrid`
   vs `opt.solve_equilibrium` vs `im.run` — no when-to-use-which anywhere;
   quickstart teaches the manual `wout_from_state` plumbing while
   `solve_equilibrium` appears in no doc page.
6. **`ImplicitSolution.runtime`** (MED): callers rebuild
   `runtime_from_params(make_config(...))` per objective eval (e.g. the
   single-stage example) duplicating work `im.run` already did; also jit the
   example objectives (measured 30% win from `jit(grad)`).
7. **Consolidate duplicate physics helpers** (MED): `optimize.aspect_ratio`
   (wint quadrature) vs `implicit.aspect_ratio` (shoelace), `volume` vs
   `plasma_volume`, `edge_iota` vs `iota_edge` — same scalars, different math,
   will drift; one family in `statephysics.py`, re-exported.
8. **Dead-code prune** (LOW): `_compat.py` numpy-mode block (~250 lines; only 2
   cache helpers are used), `fourier.angle_grids`, `nyquist.
   nyquist_mode_table_from_grid` (exported, zero call sites); test-or-gate the
   untested `recycle=True` Jacobian lane (optimize.py:1806-1829).
9. **Near-axis seeding** (decide): the README/docs claimed pyQSC/pyQIC seeding
   with ZERO implementing code — claim struck 2026-07-16. Either implement it
   properly (near-axis surface → VmecInput; ESSOS has near-axis fields to
   bridge from) as a small feature PR, or leave it struck.
10. **PR #22 (mirror geometry) decision**: no longer draft — MERGEABLE,
    +15,052/−1,622 across 44 files (a whole `vmec_jax.mirror` package). Merging
    blows the plan.md §14 size budgets; deferring leaves a large branch rotting.
    DECIDED 2026-07-17: deferred — PR #22 is actively developed by its author
    and will be revisited as the last item before the VMEX rename.
11. **plan.md §14 DoD reconciliation**: size budgets already exceeded (40 files
    / 23.4k lines vs ≤35/15k; 4 files > 1000 lines; tests 9.1k vs ≤6k lines;
    fresh clone > 10 MB). Either amend the DoD numbers deliberately or schedule
    the trimming — do not let the rename ship with a false checklist.
12. **Release practice** (LOW): no CHANGELOG; add one + release-notes practice
    at the next version bump (publish workflow itself is ready — bump pyproject
    + tag + GitHub Release works). Declare an `essos` optional-extra comment in
    pyproject (mirroring the `freeb` one). Triage the three root `notes_*.md`
    (fold Redl spec into docs; archive the rest). Prune the stale
    `en/vectorized_reverse_ad_rule` branch (its content is merged as PR #24).

## 10. Item H — VMEX rename (R21) — **GATED**

Absolute last, only on explicit user go-ahead. Atomic cutover per `plan.md` §13/R21:
rename package + repo + PyPI + docs + README + CI badges in one PR, tag fresh release.

## 11. Standing rules (do not violate)

1. Commits authored by **rogeriojorge** only; never add a Claude co-author trailer.
2. Report QI with `quasi_isodynamic_residual_from_wout`; QS with
   `QuasisymmetryRatioResidual` on ≥8 surfaces. Traceable residuals are for
   *optimizing*, wout metrics are for *reporting*.
3. Never treat naive full-solve FD as gradient truth for solver-sensitive metrics —
   use `frozen_path_directional_fd`.
4. Long compute jobs → office box, tmux, log to a file, verify the process is alive
   and descending before walking away.
5. Wout-validate any "improved" deck before swapping it into the repo.
6. Keep CI green: parity shard c is the fragile one (borderline 15 min); coverage gate
   needs all four parity shards to complete.

## 12. Suggested order

A (cold-start comparison — runs in flight, harvest → figure → PR) →
I.1-4 as one robustness PR (typed errors, penalty-path tests, mypy+lint gate,
multigrid-grad FD test) → F (speed: adjoint budget first — measured 22.5-32.4 s
per gradient vs 2.1 s solve) → E (objectives row, corrected premise) →
I.5-8 (docs/API/consolidation/prune) → G (parallelization, greenfield) →
I.9-12 + hygiene (near-axis decide, PR #22 decision, DoD reconciliation,
CHANGELOG/notes/branches) → H (VMEX, gated).

Item B is DONE (QP folded, QI rejected); Item C is DONE except the QH wall-time
provenance (re-measure or drop during Item A's README edit).
