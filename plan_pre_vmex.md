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

## 2. Item A — Cold-start single-stage vs two-stage comparison (top priority)

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

## 6. Item E — Objectives README row (#37)

New README row/section: from a QA-ish seed optimize combinations of L∇B (or L_gradB
proxy), self-consistent bootstrap (Redl) at finite β, DMerc > 0 (Mercier stability),
magnetic well (vacuum), higher iota at fixed aspect, lower aspect at fixed iota. All
objectives already exist traceably (`optimize.py`, `postprocess.py`); this is
composition + runs (office box) + figure + README row. After Items A–C.

## 7. Item F — Speed deep-dive (#36)

Done so far: XLA compile flags (~12% compile win), residual-jit inside adjoint, 2D
preconditioner assessment (opt-in), R25 gradient stack (15.7× vs FD Jacobian). Remaining,
in order of expected value: (1) profile + speed the NESTOR/free-boundary path (the README
itself flags it as not speed-tuned); (2) warm-started/recycled adjoint solves across
least-squares iterations (GCROT recycle exists — measure and default it if it wins);
(3) compile-time reduction for the implicit-gradient graph (donate/remat audit);
(4) re-run `benchmarks/run_baseline.py` + refresh the performance figure/claims.

## 8. Item G — Parallelization (#41)

Target: multi-CPU strong scaling (local measurement) + multi-GPU for solve and
optimization. Concrete first steps: (1) shard the per-dof implicit-Jacobian columns of
`least_squares(jac="implicit")` across devices (`jax.pmap`/`shard_map` over the dof axis —
embarrassingly parallel); (2) batch multi-case optimization (vmap over decks) on one GPU;
(3) strong-scaling study of a single solve (the radial dimension is the natural axis;
honest assessment — the per-iteration FFTs are small, so intra-solve scaling may be
poor and the honest story is parallelism *across* gradient columns / cases); (4) document
in README + docs. Literature notes (§5) may add relevant prior art.

## 9. Item H — VMEX rename (R21) — **GATED**

Absolute last, only on explicit user go-ahead. Atomic cutover per `plan.md` §13/R21:
rename package + repo + PyPI + docs + README + CI badges in one PR, tag fresh release.

## 10. Standing rules (do not violate)

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

## 11. Suggested order

A (cold-start comparison — probe ➞ office runs ➞ PR) → B (deck fold, can overlap A's
office runs) → C (polish, folds into A/B PRs) → E (objectives row) → F (speed) →
G (parallelization) → H (VMEX, gated).
