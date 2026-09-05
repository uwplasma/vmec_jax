# VMEX research and implementation plan

Reviewed 2026-09-05 UTC, with a second literature review the same day (three
read-only reports: polishing method versus DESC/GVEC/SIESTA/SPEC/Thun and the
least-squares literature; differentiable optimization versus DESC, simsopt and
the adjoint papers; validation evidence versus every equilibrium-code paper of
2023–2026). This is the authoritative forward plan, replacing the
August plan and its appended September ledger. Completed work is linked, open
work has an owner and an acceptance gate, and unsuccessful experiments remain
evidence rather than proposed defaults. Implementation status is recorded below;
the remaining physics and distributed solver work is not yet implemented.

## Execution logbook

- **Second review (2026-09-05, this revision):** reordered the critical path
  (C0 decision gate with a kill criterion and a publishable fallback; E off
  the path; F and the first paper decoupled from C), wrote the evidence bar
  the papers actually set into J2, the derivative-verification table into
  F0, the oracles and seeding chain into F2, the LSMR/preconditioner order
  into C3, the PR merge order and required edits into section 6, the next
  ten PRs into 7, and the environment runbook into 8. Corrected: the
  tokamak polish record (80 GN / 47 308 CG / 463 s, not a 28 s run), the
  Γ_c prompt-loss statement (#272), the status of 31.2-R7 (#264: already
  correct), and the recovered `physics/force-error-normalizations` branch.
  Office run in flight: the three-arm W7-X memory artifact for #261.

- **Completed checkpoint (2026-09-05 UTC):** custom-VJP native-input provenance.
  Resume in `/Users/rogeriojorge/local/vmex-vjp-native`, branch
  `fix/polish-vjp-native-input`, based on `4c2d29a3` (PR #269). Production fix,
  focused regressions and full CPU derivative integration are validated.
  Implementation `ac759d09`, [PR #270](https://github.com/uwplasma/vmex/pull/270).
  Preserve original user worktrees. All commits use author `rogeriojorge`.
- **PR stack:** plan [#267](https://github.com/uwplasma/vmex/pull/267) →
  true linear residual [#268](https://github.com/uwplasma/vmex/pull/268) →
  shared force certificate [#269](https://github.com/uwplasma/vmex/pull/269) →
  native-input VJP [#270](https://github.com/uwplasma/vmex/pull/270).
  These remain open at this checkpoint; local checks do not imply CI/merge.
- **Completed R1:** `b2bd9da6` / #268. Finite unpreconditioned derivative
  residuals are authoritative. CPU focused plus integration: 43 tests, 374.17 s;
  GPU focused: 42 tests, 8.68 s; static preflight passed.
- **Completed R2 subset:** `02a31622` / #269. One finite force/quadrature/Jacobian
  predicate serves all four early/final acceptance points; diagnostics agree.
  Follow-up `4c2d29a3` preserves solver success when the independent force
  certificate fails; all 68 route regressions passed again (0.46 s).
  68 route tests passed (0.42 s), 3 continuation/derivative tests passed
  (358.04 s), and the public Solovev correction regression passed (285.55 s).
  That real lift passed the force threshold but failed quadrature; retaining
  the existing thresholds now triggers correction before certification.
  Static preflight and warning-strict Sphinx build passed. No new production
  API or file was added. Full A remains open.
- **Current reproducer/fix:** the custom VJP discarded primal native leaves and
  used `runtime.native` in its backward pass. For analytic `g(c,q)=c-q^2=0`,
  output `q+c`, stale runtime `q=1`, input `q=2`, stationary `c=4`, it returned
  gradient 3 instead of 5. Both eager and JIT regressions failed before the fix.
  The forward now saves its native leaves; backward replaces only the native
  state in the frozen runtime before applying the adjoint. Chart/discretization
  remain fixed. This does not solve or certify nonlinear stationarity.
- **Current validation:** CPU focused `-k 'polish_vjp_uses_primal_native_input or polish_linear'`:
  44 passed, 129 deselected, 3.06 s. The analytic regression uses the real
  adjoint/GMRES/custom-VJP chain and successive inputs through one compiled
  function; only the physics equation is substituted. It is not an MHD benchmark.
  GPU focused subset: 44 passed, 129 deselected, 12.73 s on RTX A4000
  (JAX 0.9.2, float64, actual JIT enabled). Static preflight passed (43 guards,
  3 skips, Ruff/mypy/prose); warning-strict Sphinx build passed. The existing
  CPU MHD derivative integration passed: 1 test, 172 deselected, 292.00 s.
  No full repository-suite or distributed-equilibrium claim is made.
- **Resume commands:** from this worktree use
  `VMEX_COMPILATION_CACHE=disabled python3 -m pytest -q tests/test_polish_preconditioner.py -k collocation_polish_primal_and_derivatives`,
  `python3 tools/preflight.py --static`, and a warning-strict Sphinx build.
  GPU source/test copies are isolated in `office:/home/rjorge/local/vmex-vjp-native`
  (detached `09f18464` plus current driver/implicit/test copies); use
  `/home/rjorge/venvs/vmex-gpu/bin/python`, `JAX_PLATFORMS=cuda,cpu` and the
  focused selection above. No full GPU MHD integration claim is made.
- **Consumer audit:** `core/cli.py::_write_wout_from_result` already gates native
  export on `polish_report.converged`; multigrid reports failed polish explicitly.
  `return_unpolished` retains an uncertified native lift, so callers must inspect
  the report. Direct native export and resumed-solve contracts remain to test.
- **Next:** review/merge the stacked acceptance/VJP PRs as CI permits, then
  enforce stationarity eligibility for current native data
  at derivative entry points, with explicit scaling/tolerances and eager/JIT
  failure status. Preserve the distinction between physics acceptance, nonlinear
  stationarity and linear-solve certification. The existing MHD test permits
  physics acceptance after one GN step; its duality/Taylor checks alone do not
  prove nonlinear-root derivative accuracy. For that next gate, reuse the primal
  `g` already returned by tangent `jax.linearize` and adjoint `jax.vjp` instead
  of adding another force evaluation. Match the primal scaling explicitly:
  for `c=D*y` and `r_scaled=r/a`, the scaled stationarity is `D*g/a**2`.
  The existing solver optimality is in these scaled coordinates; a raw `g`
  threshold is not interchangeable. Carry the required scale/threshold provenance
  through `PolishContext` and the custom-VJP saved data, and certify at the actual
  forward native inputs. Continue the wider A gates in order.

## 1. Outcome and order of work

Keep the fast VMEC-compatible solve as the branch finder. Build one accurate,
native high-order equilibrium path above it, with separately checked physical
residuals and implicit derivatives. Make time and memory **to a stated accuracy
and optimization result** the performance targets. Preserve standard VMEC I/O
while letting research workflows retain native geometry and derivatives.

The program must deliver:

1. Reliable fixed- and free-boundary stellarator and tokamak equilibria, including
   finite pressure, prescribed current, bootstrap workflows and LASYM.
2. A high-order path that addresses VMEC's continuous force-balance error,
   including the axis, with real radial/angular convergence and independent
   comparisons against native DESC, VMEC2000, VMEC++ and, where available, GVEC.
3. Certified derivatives and useful boundary/profile/coil optimization, measured
   through the final feasible design rather than through a cheap isolated call.
4. Measured CPU and GPU execution, independent-case parallelism, and genuine
   sharding of force/derivative work across devices without accidental gathers.
5. A coherent mirror program: isotropic foundations, consistent anisotropy,
   free-boundary coupling, and periodic stellarator–mirror hybrids whose model
   and limitations are explicit.
6. A small, understandable API, student-to-research examples, organized
   documentation, reproducible evidence and one or more publishable results.

The critical path is **A → B → C0 → (C | F) → D** below; **E is not on it**.
Documentation and measured slimming proceed with every change. Mirror research
(H) and downstream work (G) have their own prerequisites; they do not block a
well-supported toroidal release or the first methods paper. Do not launch
another multi-day W7-X polish before the frozen-operator and physical-functional
gates in C pass.

Second review, 2026-09-05, after the A-stack #268–#270 landed on this branch:
the order above is right for engineering and wrong for the calendar in two
places, and both are fixed below. C is a research bet — a new residual, chart
and linear solver for a 3-D problem that has so far certified nothing — and
nothing here said what happens if it does not pay off. C0 gives it a bounded
budget, a kill criterion and a fallback that keeps the first paper on the
evidence already in hand: certified axisymmetric polish, the analytic
Solov'ev oracle, six-deck VMEC2000 parity, the native-DESC comparison and the
honest 3-D cost. E (one-problem multi-device sharding) sat on the critical
path with no demonstrated payoff anywhere in the field — DESC's own multi-GPU
work is experimental and workload dependent, and the review's probe is a
building block — so E moves after F and out of every paper's dependencies
except an explicitly distributed claim. F does not wait for C: every F row
except "single-stage finite beta" runs on the VMEC branch finder plus the
derivatives A just repaired.

| Work package | Scope and owner | Depends on | Completion evidence |
|---|---|---|---|
| A | Acceptance, provenance and benchmark repairs — VMEX | current main | Failed roots/gradients cannot be labelled certified; honest stage records |
| B | Physics oracle, conventions and native representation — VMEX | A | Analytic and manufactured tests; matched native cross-code metrics |
| C0 | Decision gate for 3-D polish — VMEX | B | One bounded experiment says whether 3-D polish fails on representation, conditioning or physics, or C is parked with a written fallback |
| C | Recover a useful 3-D high-order solve — VMEX + SOLVAX | C0 pass | Frozen QA operator verified; certified finite-beta 3-D refinement |
| D | Profile and reduce time/memory — respective kernel owners | A; C for polish claims | Cold/warm/gradient/time-to-accuracy results and trace-backed ablations |
| F | Optimization and engineering examples — VMEX + ESSOS | A, B; C only for polished-state rows; D for cost tables | Feasible before/after designs, independent validation, total cost |
| E | CPU/GPU sharding and scaling — SOLVAX primitives, VMEX layouts | A, D, F; C for polish | Multi-device values/gradients, shard layouts and scaling curves; a dependency only of an explicitly distributed claim |
| G | Native consumers and confinement diagnostics — respective codes | B, F as appropriate | In-memory value/derivative parity and topology contracts |
| H | Mirrors, anisotropy and hybrid optimization — VMEX + ESSOS | B, C foundations | Closure, interface, force, derivative and application certificates |
| I | Documentation, API and slimming — VMEX | continuous | Shorter ownership paths, runnable tutorials and measured size/performance |
| J | Releases and publications — maintainers | gates for each claimed scope | Archived inputs, environments, evidence, papers and verified claims |

## 2. Review baseline and evidence limits

| Repository | Reviewed base | Advertised branches | PRs, all / open | Issues excluding PRs, all / open |
|---|---|---:|---:|---:|
| [VMEX](https://github.com/uwplasma/VMEX) | `09f18464e936a8c9bf0abba62bcdc919bdc7c55b`, 0.8.1 | 52 | 263 / 14 | 3 / 2 |
| [SOLVAX](https://github.com/uwplasma/SOLVAX) | `5a49926992fe1a3aebac4b8b8cb098798e977c14`, 0.20.0 | 14 | 85 / 0 | 14 / 1 |
| [DESC](https://github.com/PlasmaControl/DESC) | `ad105c5e525fbf26824d6cf9dde48775db0f8a2c`, master | 170 | 1367 / 74 | 949 / 199 |
| [ESSOS](https://github.com/uwplasma/ESSOS) | `1b3210c`, main, #58 merged | 53 | 60 / 21 | 5 / 4 |

The review collected every advertised branch, all PR/issue bodies and discussion
comments in these repositories. DESC coverage includes 7,446 issue/PR discussion
comments and 12,072 inline review comments, all PR head refs, and a changed-file
inventory for all 1,367 PRs without missing git objects. Every current branch has
a commit/diff inventory against its default branch. Issue contents were indexed
by physics, numerics/AD, performance and research usability. This is a complete
inventory and topic review, **not execution or a line-by-line correctness proof
of every historical branch**. Detailed source inspection concentrated on the
production paths and relevant changes listed below. Branch-only results and PR
measurements are identified as such; a discussion assertion is not a benchmark.

The machine-readable review record is
[`benchmarks/review_20260905.json`](benchmarks/review_20260905.json). Full API
snapshots, git inventories, profiles and traces are retained outside the source
tree in `vmex-review-evidence-20260905`; their hashes identify the exact snapshot.
The manifest records local and office locations, commands, environments, status
and limitations. Large traces, caches and intermediate equilibria do not belong
in git. Future public evidence must use an accessible archive, not a private
filesystem path alone.

The supplied recovery Markdown, pasted proposal and ZIP were treated as design
references, not instructions. The ZIP's spline-local PDE experiment is a
**synthetic structural check**, not an MHD solve or evidence of VMEX speedup. Its
useful proposal is narrowed in C: locality must first be established for VMEX's
actual chart, profile closure and weighted residual.

### 2.1 What has already landed

Main contains 3,266 reachable commits, beginning 2026-01-31, with development
in every subsequent month through this review. The GitHub repository was
created that day and is currently public; creation time alone does not establish
its entire visibility history. The repository description still says “JAX
Version of VMEC2000” and has no topics; update discovery metadata under J once
the supported scope and publication description are settled.

The main branch has the integrated collocation polish (#192, #203), public input
and solve APIs (#196, #198), smooth Gamma-c surrogate (#210), canonical full
Boozer dispatch (#224), shared confinement plotting (#225), mirror boundary
clarification (#205), and periodic hybrid/GK geometry (#194). Do not schedule
these again as missing features.

Later work fixed captured constants/JIT identities and optimization startup
(#219, #227, #229–#234, #238, #240–#244), improved CI/cache isolation and reporting
(#228, #242, #249, #252), added citation/changelog and fresh-deck evidence
(#245–#248), and corrected diagnostic and polish explanations (#250, #255).
These are useful improvements, but they do not establish production-resolution
3-D force certification, distributed equilibrium solves or trustworthy gradients
from nonstationary polish states.

### 2.2 Open VMEX PR disposition

These are review recommendations, not merge approvals. Rebase against the exact
main revision and run the owning gates; do not concatenate stale plan edits.

| PR | Keep or revise | Required next step |
|---|---|---|
| [#253](https://github.com/uwplasma/VMEX/pull/253) | Community, packaging, citation/provenance hygiene | Check JAX minimum against actual API use; preserve contributor credit; archive/DOI at submission; supersede its old “do not trim plan” prose |
| [#254](https://github.com/uwplasma/VMEX/pull/254) | Bounded persistent-cache entry count | Hostile-entry, held-lock and unparseable-bound branches now covered (diff-cover 95%). The CI-timeout mechanism stays "unproven": measured lock-held time is 0.55 s per cold solve; the measured per-write directory rescan is 0.028 ms per resident entry, 304 ms at 10 880 entries |
| [#256](https://github.com/uwplasma/VMEX/pull/256) | Figure manifest, validation page and removal of unsupported numbers | Reconcile with #260/#264 and new measurements; keep failures in comparison tables; remove orphan media after reference checks |
| [#257](https://github.com/uwplasma/VMEX/pull/257) | Public API coverage and useful docstrings | Resolve its discovered semantic defects through A/B/I; avoid presenting unused polish knobs as functional |
| [#258](https://github.com/uwplasma/VMEX/pull/258) | Correct FFT tail mask and normalized Jacobian margin | Signed modes, Nyquist, the corner union and scale invariance are tested; the `\|Jacobian\|` docutils substitution that failed `-W` is fixed. Still to say in the margin's docstring: a sampled local orientation is not global injectivity |
| [#259](https://github.com/uwplasma/VMEX/pull/259) | Native LASYM flag propagation | Check all sine/cosine families, Boozer and virtual casing, eager/JIT and explicit tracer metadata |
| [#260](https://github.com/uwplasma/VMEX/pull/260) | Native DESC measurement | Compare identical norms, grids, units and region; do not compare its native L1/pressure ratio with VMEX's bounded L2 ratio |
| [#261](https://github.com/uwplasma/VMEX/pull/261) | Chunked/checkpointed memory reduction and cost/progress reporting | Parity lane c2 fails because `tests/test_performance_docs.py` reads `benchmarks/polish_memory_w7x.json`, which the implementing agent never produced. The three-arm measurement is being regenerated on the office box from the branch head (2026-09-05); commit it, or drop the gate — no PR merges with a test that reads a file that does not exist. No certified 3-D claim from the W7-X run; C0 replaces speculative reruns. The heartbeat program carries host callbacks and is never written to the persistent cache; the lane documents that trade |
| [#262](https://github.com/uwplasma/VMEX/pull/262) | Document NaN failure under JIT | Add the true-residual acceptance repair in A; document value plus status for transformed callers |
| [#263](https://github.com/uwplasma/VMEX/pull/263) | Mirror model/metric audit | Scope axial-current rejection to the unsupported exterior; correct anisotropic oracle and exterior-BVP statements in H |
| [#264](https://github.com/uwplasma/VMEX/pull/264) | Analytic Solov'ev oracle and LASYM audit | Closes 31.2-R7 as "VMEX was already correct": the 2024 STELLOPT `tcon` halving is half of a matched pair with that tree's conditional `dnorm`; upstream STELLOPT and PARVMEC retired both, VMEC++ agrees, and VMEX tracks the modern binary to its print precision (4.9e-3) while the isolated halving matches neither convention. Green; merge first. Extend the analytic refinement to B/C |
| [#265](https://github.com/uwplasma/VMEX/pull/265) | Progress output and elimination of duplicate gradient calls | Do not promote `adjoint_fail="best_effort"` for certified optimization; a reported relative residual near 0.66 is a failed gradient |
| [#266](https://github.com/uwplasma/VMEX/pull/266) | Profile ledger, physical scaling/bounds and coil-seed investigation | Stack after the safe parts of #265; pair tolerance comparisons with derivative/feasibility checks; C addresses the remaining polish bottleneck |
| [#197](https://github.com/uwplasma/VMEX/pull/197) | Scalar-adjoint option after rebase | Preserve the measured startup/memory tradeoff and TRF's better objective per evaluation; consolidate duplicated example setup under I |

The #266 QA polish experiment reports roughly 3,096 s and 16 GiB for three GN
iterations, with all 1,800 available inner iterations consumed. The #261 W7-X
experiment reports 11 h 09 min, 17.1 GB peak RSS and only about 1.1% improvement
in dimensional L2 force, despite a much larger collocation-cost reduction. These
are PR evidence on particular machines/settings, not results rerun by this
review. They motivate fixing the functional, chart and linear system before
increasing nonlinear budgets. The old conclusion that more iterations alone
would settle effectiveness is superseded.

The two open VMEX issues also have concrete owners. [#157](https://github.com/uwplasma/VMEX/issues/157)
reports an undefined `runtime` in the GPU guide: I1 must make that snippet
self-contained and execute it. [#211](https://github.com/uwplasma/VMEX/issues/211)
reports a closed-hybrid strong-force plateau near 5e-3 despite weak residuals
near 1e-16, and a seed-to-solved change in field modulation: H4 must resolve
the physical admission criterion before downstream GKX claims are promoted.

### 2.3 Measurements made during this review

The stage audit attempted 16 workflows in a fresh process per workflow, with a
300 s budget covering setup, first calls, three warm repeats and one profiled
repeat. Local CPU: 12 completed, F7/F8/F10 timed out, F9 hit an ESSOS capability
gate. Office GPU with CPU callback support: 10 completed, F3/F7/F8/F10/C1 timed
out, F9 hit the same gate. F1 was measured separately in four cache/reuse regimes
on both hosts. M2 was excluded because its supplied field is not solenoidal.
“Completed” here means timing collection completed; it is not a physics verdict.

Representative stage medians in seconds, with first-call compilation reported
separately in the manifest:

| Stage | Local M3 Max CPU | Office RTX A4000 execution |
|---|---:|---:|
| F2 multigrid solve | 0.239 | 0.808 |
| F4 scalar gradient | 0.268 | 0.782 |
| F5 full Jacobian at repeated parameters | 0.229 | 0.698 |
| F11 LASYM gradient | 0.276 | 0.756 |
| B1 one-surface Boozer transform | 0.088 | 0.292 |
| B2 eight-surface Boozer transform | 0.431 | 1.303 |
| F3 shaped-tokamak polished value | 15.119 | workflow timeout |

These exploratory values include the shipped workflow's caches and placement
policy; the hosts differ, and no controlled CPU/GPU speedup follows. F5's warm
residual is about 1 ms because it can reuse the one-entry objective cache. That
is why A/D require changed-parameter timing and actual call counts.

F1 XProf traces expose 4,480 calls to each of two GPU tridiagonal pivot kernels
per traced solve. A separate, same-office-host li383 ablation forced the imported
SOLVAX tridiagonal backend to Thomas in a fresh process, without editing source:
warm median 0.836 s (auto GPU) → 0.406 s (Thomas GPU); office CPU auto was 0.271 s.
All used 123 iterations and passed the legacy tolerance. R/Z/lambda maximum
coefficient differences between the GPU variants were below 8.4e-14. Cold solve
times remained about 17.9/17.7 s on GPU and 7.7 s on office CPU. This is a
promising small-case backend hypothesis, not a continuous-force certificate,
large-resolution result or proposed unconditional default change.

The review also reproduced R1's false acceptance of a true residual 1.0 with
tolerance 1e-8 and of NaN when the solver flag was true. Targeted checks passed:
22 SOLVAX tests, two analytic VMEX force/coordinate tests, static preflight
(43 guards passed, three skipped) and strict Sphinx. E records the additional
actual CPU/GPU sharding probes. Production physics and algorithms are unchanged
by this plan revision.

## 3. Findings that change the implementation priorities

Paths and symbols below refer to the pinned main branch. Each finding has a
work-package owner; none is closed merely by this plan.

| ID | Finding and consequence | Owner |
|---|---|---|
| R1 | `polish_implicit._linear_report` accepts `solution.converged OR true_residual <= tolerance`. A solver flag can override a failed or nonfinite recomputed residual. | A |
| R2 | The already-certified shortcut in `polish_legacy_solution` checks normalized L2 alone; normal acceptance can pass force/quadrature/Jacobian checks without least-squares stationarity. Physics and derivative acceptance are different contracts. | A |
| R3 | `strong_collocation_residual_at_native` multiplies separate radial/helical magnitudes by `abs(sqrtg)` and a frozen denominator, without Gauss weights or the vector Gram cross term. Its least-squares cost is not the physical Cartesian volume L2 certificate. | B/C |
| R4 | The production `make_strong_structured_chart` has independent constrained R and lambda directions; Z is not independently variable, and this chart rejects LASYM. The cylindrical Z-based gauge is singular at extrema of Z. Some m=1 mixing remains, so “every Z coefficient is frozen” is too broad. Reachability must be tested. | C |
| R5 | `make_strong_root_layout` uses local-group SVDs of legacy restriction/prolongation. This can mix radial coefficients and restrict the correction to the legacy mesh image. Native spline support alone does not prove chart locality or h-refinement freedom. | B/C |
| R6 | `apply_high_order_correction` changes geometry/lambda but holds lifted `phip`, `chip` and pressure tables fixed. Prescribed current (`NCURR=1`) and finite-gamma mass/pressure constraints require a separate consistency audit as geometry changes. | B/C |
| R7 | `_gauss_newton_polish_lane` calls SOLVAX GN without `precond=` or physical `admissible=`. Legacy factors are built but not used by this lane. Exposed square-root controls do not describe production GN behavior. | A/C |
| R8 | SOLVAX GN already supports a preconditioner and admissibility hook, but does not expose/use the inner true residual, convergence or breakdown state in its trial policy. Iteration counts alone cannot distinguish an acceptable inexact step from stagnation. | C, SOLVAX |
| R9 | The continuous force kernel nests spatial `jacfwd` evaluations inside parameter AD. Repeated synthesis, residual linearization and stored intermediates create a large cost/memory target. | C/D |
| R10 | `eps_F=2|F|/(|J×B|+|grad p|+floor)` is bounded and approaches 2 for nonzero vacuum force. `radial_refinement_difference` changes quadrature on the same state; it is not solve h/p convergence. | A/B |
| R11 | Independent certificate nodes share the production force implementation. They test off-grid behavior, not implementation independence. A DESC WOUT re-lift tests the export/reconstruction chain, not native DESC accuracy. | B |
| R12 | `parallel.solve_ensemble` provides CPU task parallelism; current VMEX has no demonstrated multi-device single-solve path. SOLVAX's sharded primitives do not establish VMEX end-to-end sharding. | E |
| R13 | The profiler's generic warm fallback repeats only the first stage; new-parameter/new-shape variants exist only for F1. Repeated parameters can hit the one-entry objective cache. Compile log counts do not distinguish backend compilation from persistent-cache hits. | A/D |
| R14 | Profiling C2 uses the hard Gamma-c path under a derivative-safe description. M2's purported two-loop field is `(0,0,B_axis(z))`, which is not divergence-free when B varies. These are invalid labels/physics for those benchmark claims. | A/G/H |
| R15 | `tests/conftest.py` disables JIT globally. Ordinary unit-test success does not cover transformed failure handling, callback placement or compiled sharding. | A/E |
| R16 | Documentation already has tutorial/how-to/explanation/reference directories, but duplicates API/solver narratives and carries historical status and comparison numbers. Reorganize ownership inside that structure instead of adding a second documentation tree. | I |

Additional defects identified in #257 remain in the repair queue: public tuple
annotation disagreement in `FunctionProblem.from_tuples`; progress missing for
combined loss; inconsistent Euclidean/infinity “optimality”; parameterized
surface construction bypassing validation; scaled WOUT profiles disagreeing
with unchanged stored profile coefficients; plotting entry points bypassing the
shared backend/style setup. Triage these by a reproducer, fix in their existing
owner, and retain `doctor`'s documented exit semantics or add an explicit strict
mode. Do not silently change a diagnostic command's exit behavior.

## A. Repair acceptance and evidence first

**Owner:** VMEX reports, tests and `benchmarks/profile_workflows.py`; SOLVAX owns
only generic solve-result semantics. **Exit:** every status means what it says.

- [x] Make a finite, recomputed **unpreconditioned** tangent/adjoint residual the
  authoritative gate. A solver's flag may be diagnostic; it cannot override a
  failed true residual. Test false-positive flags, NaN/Inf, zero RHS, iteration
  exhaustion, eager behavior and real `jax.jit` behavior. Check solution and
  operator/RHS finiteness as well as the norm.
- [ ] Unify early return, normal completion, CLI, native export and resumed solve
  acceptance. Distinguish `legacy_converged`, `physics_certified`,
  `stationarity_certified`, and `derivative_certified` with explicit thresholds.
  Reuse the existing report/result owners instead of adding parallel report APIs.
- [ ] A physical certificate can accept a state before GN stationarity, but its
  stationarity-based implicit derivative must then fail or refine first. Under
  transformations return numerical values plus a usable status, with the host
  boundary raising a typed error when requested. NaN is never optimization data.
- [ ] Wire all physical invariants into trial admissibility: finite fields,
  regular/positive-oriented geometry away from the axis, axis limits, boundary
  and gauge constraints, profile/current closure and model-specific conditions.
  Reject invalid trial geometries before expensive full evaluations where a
  cheap conservative check is available. A sampled Jacobian is only a local test.
- [ ] Audit each `PolishConfig` option against the actual public call graph.
  Implement a meaningful control or deprecate it; remove unnecessary factor
  construction. Rename reports inherited from the retired continuation lane.
- [ ] Repair the profiler to time every named stage; implement meaningful changed
  parameter/shape regimes per workflow or emit “unsupported” explicitly. Record
  setup failure, dependency gate, nonconvergence, timeout, OOM and NaN separately,
  and preserve partial stage output. A process exiting zero is not success.
- [ ] Include physics and derivative status in every timing row. Fix C2's actual
  objective and replace M2's field with full analytic/ESSOS Biot–Savart before
  treating it as a physical benchmark. Add stage annotations around setup,
  forward solve, refinement, linearization, adjoint, objective and certification.
- [ ] Remove selected-winner requirements for README plots. Predeclare cases;
  retain ties, losses and failures in the table. Separate native improvement,
  export sampling and re-lift improvement. Retain the bounded diagnostic for
  compatibility, but never make it the sole accuracy or promotion criterion.

**Review measurement:** this revision profiles pinned main on the local M3 Max
and office RTX A4000 environment with fresh-process limits. It records failures
and incomplete runs, not paper-grade speedups. See the manifest for exact rows.
F1 traces already show thousands of GPU tridiagonal pivot launches. This is a
specific backend investigation for D, not evidence that all GPU solves are slow.

## B. Specify the physics, representation and independent oracle

**Owner:** VMEX `strong_force.py`, `polish.py`, `radial_basis.py`, profile/state
owners, existing certificate and comparison benchmarks. **Exit:** the oracle
passes exact solutions, and different codes are compared as the same problem.

### B1. One conventions and constraints contract

- [ ] Consolidate SI units, `MU0`, full/half mesh, normalized toroidal flux
  `s=rho^2`, angular periodicity, physical phi versus NFP-scaled angle,
  handedness/signgs, m/n ordering, lambda sign/scaling and WOUT conventions.
  Cross-reference the source formulas and verified VMEC2000/VMEC++ descriptions.
- [ ] State which flux/current/pressure quantities are independent for each
  `NCURR`, gamma and free-boundary mode. Audit the lift and correction against
  those invariants. For prescribed current, enforce/check Ampère loop integrals
  while geometry changes; freezing iota is not the same problem. For gamma/mass
  profiles, recompute the volume-dependent pressure law where required.
- [ ] Preserve exact boundary constraints, axis regularity and coordinate gauge
  in a native spline/Fourier space. Document `rho^|m| q_mn(s)` analyticity and
  handle m=0/1 axis limits explicitly. Use one declared degree default and
  precision/floor policy. Local basis support must survive constraint elimination.
- [ ] Native state serialization must retain knots, degree, all parity families,
  physical profiles/closure, units, topology and provenance. Version the schema,
  validate shapes and round-trip fields/JVPs. Sample WOUT only for compatibility;
  report export-grid dependence rather than silently discarding native accuracy.

### B2. Norms and regions

Let `F = curl(B)×B/mu0 - grad(p)` in Cartesian components, `dV` be the physical
volume measure, and `V` the evaluation volume. Report, with SI units:

- `F_L1 = integral |F| dV / V` and `F_L2 = sqrt(integral |F|^2 dV / V)`;
- a declared pointwise maximum/percentiles and flux-surface RMS profiles;
- pressure-normalized `F_L1 / <|grad p|>_V` for finite-pressure literature
  comparisons, and an explicitly named magnetic-pressure-gradient normalization
  when that denominator is nonzero;
- a vacuum-safe fixed reference `F_ref = B_ref^2/(mu0 a_ref)` and `F_L2/F_ref`,
  with B_ref/a_ref fixed for a comparison or optimization stage;
- the existing bounded `eps_F` only as an additional diagnostic, with its floor
  and saturation disclosed.

Uniform fields can make even the magnetic-pressure-gradient denominator zero;
use an unavailable status for that ratio and retain the dimensional/fixed-scale
metrics. Do not introduce an arbitrary floor that changes cross-code rankings.
Match L1 with L1 and L2 with L2; distinguish a mean of pointwise ratios from a
ratio of means. Keep the legacy radial `equif` residual distinct from vector
strong force and explain its vacuum limitation.

Use the Panici comparison region `s in [0.1,0.99]` when reproducing that paper,
**and** report full-volume, near-axis, bulk and edge results with exact region
boundaries. Do not hide difficult regions to certify the whole equilibrium.
A coordinate Jacobian vanishes at the axis by construction; check its regularized
limit there rather than applying an impossible positive lower bound at rho=0.

### B3. Verification hierarchy and benchmark cases

| Level | Required problems | What it verifies |
|---|---|---|
| Exact/local | Uniform Cartesian B, toroidal vacuum B∝1/R, manufactured divergence-free fields, circular-coil on/off-axis field | Units, curl/divergence, coordinate transforms, derivatives and vacuum metrics |
| Exact equilibrium | Analytic Solov'ev from #264, analytic theta-pinch, quartic-flux mirror | Force oracle against known nonzero pressure/current; no numerical “Solov'ev input” substituted for an exact solution |
| Representation | h- and degree-refined splines; m=0/1 and higher parity; perturbed LASYM; exact knot insertion | Approximation order, regularity, reachability and independent off-grid derivatives |
| Toroidal cross-code | Shaped tokamak; finite-beta QA; QH; LASYM finite-beta; high-aspect-ratio/near-axis case; harder W7-X case | Native force, profiles, geometry, current closure, convergence and failure boundaries |
| Free boundary | Vacuum coil/plasma-off limit, analytic axisymmetric case, finite-beta direct-coil stellarator | Interface traction/B·n, vacuum coupling, branch-local response |
| Mirror/hybrid | Isotropic fixed/free mirror; anisotropic fixed/free mirror; perturbed 3-D mirror; periodic hybrid | Topology-specific closure, interfaces, regularity and limits under H |
| Application | Stable near-axis Mercier limit, QS metrics, transport/orbit validation and constrained optimization | Whether equilibrium accuracy matters to downstream science |

- [ ] Evaluate force on shifted/oversampled quadrature distinct from solve nodes,
  then increase evaluation quadrature until integration error is below the
  claimed force improvement. The signed-frequency angular tail mask is
  corrected in #258 (a pure m=1 field had reported half its power as
  unresolved and a near-Nyquist field none), with a scale-free
  `nestedness_margin` that had been the Jacobian minimum under a second name.
- [ ] Re-solve/re-polish after radial h, spline degree and angular refinement.
  Record force, observables, current and gradient convergence separately. Rename
  the present quadrature-only difference; retain it as an integration check.
- [ ] Retain a slow independent evaluator using analytic derivatives or a
  separately implemented coordinate calculation. Test the optimized kernel
  against it before using the optimized kernel as a certificate. Manufactured
  forcing verifies operators, not existence of a physical unforced equilibrium.
- [ ] Compare native DESC spectral fields with native VMEX spline fields at
  matched physical points or equivalent volume quadrature. Record interpolation,
  flux-label correspondence and coordinate maps. WOUT comparisons form a
  separate compatibility/reconstruction experiment, including export refinement.
- [ ] Match boundary, NFP, pressure/current/iota specification, flux, beta, units,
  symmetry and solution branch. Report scalar degrees of freedom, not just a
  shared “resolution” name. Use each code's reasonable converged settings and
  sweep its own resolution. Include conversion and initialization cost explicitly.
- [ ] Retain VMEC trajectory/WOUT parity tests without confusing parity with
  continuous equilibrium accuracy. Pin the existing harmonic/iteration tolerances
  rather than describing them as machine precision.
- [ ] Separate numerical failure from model limits: nested surfaces exclude
  islands/stochastic regions, and ideal-MHD rational-surface singular currents
  can prevent smooth spectral convergence. Use SIESTA/SPEC/HINT comparisons only
  with their different topology/pressure models declared; do not promise that
  high-order polishing repairs a physically nonexistent smooth nested solution.

## C. Recover accurate and affordable 3-D force balance

**Owners:** VMEX specifies the physical residual, chart, closure, local element
blocks and acceptance. SOLVAX owns generic least-squares, factorizations,
preconditioners, Krylov algorithms, globalization and implicit linear solves.

### C0. Decision gate: representation, conditioning or physics?

Three facts are measured and none is an answer. The shaped tokamak
certifies (1.28e-2 → 1.79e-3 in the bounded metric), but the committed
record (`benchmarks/strong_force_cases_m4.json`, VMEX `polish_report`) says
how: 80 Gauss–Newton steps, **47 308 conjugate-gradient iterations**, 463 s,
`least_squares_success: false`, accepted by the certificate. That is ~590
CG iterations per step on a **148-unknown** normal system, in which exact
arithmetic terminates CG in 148 — the finite-precision signature of a badly
conditioned normal matrix, independent of any 3-D measurement. The
finite-beta QA case polishes (independent absolute L2 3.43e4 → 2.04e4,
−40%) but needed forty steps, and #266 measured 3 096 s and 16 GiB for
three of them. W7-X costs 1.75 h per step with every step at the 150-
iteration ceiling that run set (the driver default is 600) and moved the
independent L2 by 1.1% in six steps (#261).

Two things the literature settles before any experiment (method review of
2026-09-05, section C3). First, CG on the Gauss–Newton normal equations
converges at a rate set by κ(JᵀJ) = κ(J)², and reaching VMEX's
`linear_rtol` 1e-3 needs k ≈ 3.8 κ(J) iterations; the linearized strong
force is a second-order differential operator whose condition number
scales as h⁻² in the radial spacing (the Hirshman–Betancourt argument), so
κ(J) of 10³–10⁴ and k of 10⁴ per step is the natural expectation, and no
equilibrium code in the field runs an unpreconditioned Krylov solver on
it: VMEC's 2-D lane and SIESTA precondition with a block-tridiagonal
Hessian, DESC factors a dense Jacobian by QR (7.6k unknowns at its W7-X
resolution), GVEC and SPEC use descent or dense Newton. Second, there is a
**resolution floor** that no polish at fixed MPOL = NTOR = 10 can cross:
DESC's own W7-X equilibria sit at ⟨|F|⟩/⟨|∇p|⟩ = 4.8e-2 (ANSI) and 3.8e-2
(fringe) at M = N = 10 and only reach 3e-3 at M = N = 14 (Panici 2023, the
paper's notebook); Thun et al. 2026 place W7-X pruned to M_b = N_b = 10 at
~1e-2 with exponential decrease in M = N. The W7-X residual at that
resolution is angular truncation, and its saturated certificate (≈2,
absolute_l2 ≈ 4.3e6 N/m³ against a |∇p| scale of ~2e5) says the lifted
state is far from pointwise balance in the chart's own norm. So the
1.1% is not primarily a solver failure; it is the chart being asked to
remove content it cannot represent, with corrections that are purely
horizontal because Z is frozen — something no mature code does (VMEC keeps
R and Z free under spectral condensation; DESC keeps R, Z, λ free and lets
Levenberg damping absorb the gauge).

Spend **three bounded experiments**, in this order and each a few
CPU-hours at most on the office box, to decide which of three things is
wrong, because the remedies are disjoint:

0. **Resolution-floor scan (minutes).** Run the W7-X deck through VMEC++
   at ns 51/101/201 × MPOL = NTOR 10/12/16 and certify each with the
   existing `benchmarks/strong_certificate.py`, reporting absolute_l2 and
   ⟨|F|⟩/⟨|∇p|⟩. If the dimensional L2 falls by ≫ 1.1% with resolution,
   fixed-chart polish at MPOL = NTOR = 10 is provably below its ceiling of
   usefulness and the paper says so.
0′. **LSMR swap on the tokamak (minutes).** Same 148-unknown case; compare
   against 47 308 and record LSMR's condition estimate. Zero risk whatever
   the outcome, and it is the first commit in C3 regardless.

1. **Representation.** Freeze the QA linearization (C1) — or, cheaper and
   more decisive, a reduced W7-X at MPOL = NTOR = 6, ns = 25 (1–2.5k
   unknowns), dense J by batched `jacfwd` and one SVD: that gives κ(J)
   (predicting the CG count and testing the diagnosis above), the spectrum
   of the candidate block preconditioner, and from 5–10 exact steps the
   certificate floor of the chart. If exact steps reduce the *independent*
   force by much less than they reduce the collocation cost, the chart
   (Z frozen, legacy-restriction image, R–λ only) is the problem: C2 first.
   Repeat once with Z_sin unfrozen to isolate the gauge. The projection
   "unresolved fraction" is not the discriminator (0.694 W7-X, 0.734 QA);
   `sampled_rms` (0.442 vs 0.0018) is the quantity that differs.
2. **Conditioning.** If the QR step is good but PCG needs more than ~150
   iterations to reproduce it, the squared condition number of the normal
   equations is the problem: LSMR on the rectangular Jacobian with the
   existing variable scaling first, then a structured right preconditioner
   (C3).
3. **Physics.** If the QR step reduces the collocation cost but not the
   independent force, the functional is wrong (R3: |sqrt g| without
   quadrature weights, frozen denominator, separate radial/helical
   magnitudes): C2's Cartesian volume-L2 rows first.

**Budget and kill criterion.** C0 plus its first remedy get one calendar
month of one person and the office box. If at the end no 3-D case below
W7-X resolution certifies on the independent oracle with h/p convergence, C
is **parked**: the J2 numerics paper ships with the certified axisymmetric
result, the Solov'ev oracle, the native-DESC comparison (#260) and the W7-X
cost as a stated limitation, and C reopens only with a new operator-based
hypothesis. Parking is a normal outcome of this gate; the fallback is already
publishable.

The 0.8.2 release (J1) ships #261's memory fix, AUTO budget and progress
heartbeat regardless of C0, because those are user-facing bug fixes, not 3-D
accuracy claims.

### C1. Freeze one real linearization before designing the solver

Start with the modest finite-beta QA case used in #266, then the exact tokamak
and a modest LASYM case once C2 supports its chart. Save the native state,
profiles, constraints, grid,
scales, damping and random seeds in a bounded artifact. For
`r(c)=weighted physical force`, define `A=dr/dc` and test:

1. Directional finite differences over a step-size sweep and several directions;
   JVP/VJP duality; comparison against a chunked explicit Jacobian.
2. Matrix-free normal action against explicit `A.T @ A`, true residual and
   predicted reduction. Check rank, singular values and identifiable/gauge modes.
3. A trusted augmented QR/SVD solution of
   `min_delta ||A delta + r||^2 + mu ||L delta||^2`, including the actual variable
   scaling. Compare physical step, cost reduction and linear residual, not only
   coefficient norms or Krylov recurrence residuals.
4. One-factor-at-a-time ablations: old/new weighting, restricted/full R–Z chart,
   frozen/consistent profiles, and no/structured preconditioner. This separates
   unreachable force directions, poor conditioning and expensive kernel work.

Use explicit matrices only where their measured memory is affordable. The
attachment's 36,540×2,788 float64 Jacobian arithmetic is approximately 0.759 GiB
for one matrix; AD tapes, QR workspace, copies and XLA temporaries add more.
A matrix-size estimate is not a process/device-memory prediction.

**Gate C1:** Jacobian/transpose tests pass; QR gives a trustworthy reference;
the effect of each physics/chart change is measured; no unexplained rank or
operator mismatch remains. An ineffective QR step redirects work to physics,
representation or globalization instead of another Krylov variant.

### C2. Optimize the intended physical functional

Use Cartesian residual rows, or a mathematically equivalent positive metric
factorization including all cross terms. A proposed row is

`r_q = sqrt(w_q * abs(g_q) / V_ref) * F_cart(q) / F_ref`.

Here `w_q` includes the full quadrature conversion for the chosen coordinates,
`V_ref` and `F_ref` are fixed stage/comparison scales, and `g_q` is the physical
Jacobian. Decide explicitly whether geometric weights vary within the nonlinear
functional or are frozen during an outer reweighting step. Differentiate the
chosen definition consistently; a frozen-weight derivative is not automatically
the derivative of a public solve that recomputes weights for perturbed inputs.

- [ ] Introduce independent admissible R, Z and lambda directions, with a regular
  gauge based on displacement/constraint geometry rather than division by
  `Z_theta`. Include all LASYM parity families rather than merely removing the
  current symmetry guard. Test vertical shape errors and noncircular surfaces.
- [ ] Construct constraints natively, without restricting every refined correction
  to a low-order sample image. Verify exact boundary/axis constraints and the
  dimension/rank of the admissible space at each h/p level.
- [ ] Restore prescribed-current and finite-gamma closure from B1. Account for
  induced profile derivatives in both the force Jacobian and implicit response.
- [ ] Use continuation in pressure/current/resolution and regularized steps to
  reach the intended branch. Report all residual blocks and physical invariants.

### C3. Choose structured linear algebra from the actual operator

**What the literature settles (2026-09-05).** Every Krylov-based
equilibrium code preconditions the linearized operator with a
block-tridiagonal-in-radius, diagonal-in-(m, n) Hessian approximation —
VMEC's `scalfor` is the dominant ∂²_ρ, m² and (n·nfp)² terms of exactly
that operator (Hirshman–Betancourt 1991; VMEC++ numerics Eqs. 5.234–5.262),
VMEC2000's `precon2d` and SIESTA factor the full block-tridiagonal Hessian
with BCYCLIC and hand it to GMRES — and the two codes that keep a strong
collocation residual (DESC, the Thun PINN) never use Krylov at all. The
practitioner's order, each step reusing the previous one's pieces:

1. Log the inner residual history and time one residual, one JVP and one
   VJP; the tokamak product costs ≈ 9.8 ms on 1 764 × 148 and linear scaling
   predicts the W7-X 42 s, so the per-product cost is the honest size of a
   non-FFT tensor operator, not a software pathology.
2. Replace CG on JᵀJ + μI by **LSMR on [J; √μ I]** — same JVP + VJP per
   iteration, monotone ‖Jᵀr‖, safe early termination, a condition estimate,
   and the damped problem solved without forming the normal matrix (Fong &
   Saunders 2011; Nocedal & Wright §10.2). SOLVAX 0.20.0 ships `pcg`,
   GMRES/FGMRES and a `precond` hook but no LSQR/LSMR: that is the SOLVAX
   addition. In exact arithmetic LSMR has the same iteration count as CG;
   it buys stability and a meaningful stopping quantity, not a cure.
3. Build the **same-basis right preconditioner**: block-diagonal in (m, n),
   banded in the spline index (bandwidth 2·degree + 1), assembled from the
   Hirshman–Betancourt terms in the B-spline basis using metric quantities a
   residual evaluation already computes. Apply it as J M⁻¹, so the normal
   operator becomes M⁻ᵀ JᵀJ M⁻¹ and the SPD companion for CG is (MᵀM)⁻¹ — a
   solve with M followed by a solve with Mᵀ, never M⁻¹ alone. DESC's Jacobi
   column scaling (`x_scale="auto"`) and VMEX's eight Rademacher probes are
   the degenerate diagonal case; both remove coefficient disparity, neither
   touches the spectral spread of a differential operator.
4. The existing low-order block factorization goes in the same way or not
   at all: with T the high←low transfer, right-precondition by T A_low⁻¹,
   whose SPD companion is T A_low⁻¹ A_low⁻ᵀ Tᵀ (both `block_thomas_solve` and
   its transpose exist in `implicit.py`). #261 measured "B Bᵀ from the
   low-order inverse" as worse; if that applied a second-order inverse once
   against a fourth-order operator it left a second-order, non-symmetric
   product to which CG's theory does not apply, and the negative result
   says nothing about the symmetric form. Record which form was tested.
5. Only if 2–4 fail, DESC's route: a dense Jacobian by batched JVPs and a
   QR (2.3 GB at DESC's 7.6k unknowns and 2× oversampling; more on VMEX's
   denser tensor grid), a several-hour exact step instead of a 1.75 h inexact
   one.

Unfreeze Z_sin alongside: no mature code freezes a coordinate, and VMEX's
Levenberg μI damping already does what DESC relies on to absorb the
reparametrization null space, so this is cheap to try and C0's experiment 1
says whether it matters. Add quadrature weights so the polish minimizes
the certificate's norm (DESC's residual is quadrature-weighted so its sum
of squares approximates a volume L² norm; VMEX's carries |√g| but no weights
and a denominator frozen at the lifted state). The honest ceiling of all of
this is the resolution floor of C0: sell the outcome as "affordable 3-D
polish", never as "accurate W7-X at MPOL = NTOR = 10".

For a local degree-d spline chart and local closure, each quadrature row touches
at most d+1 radial basis functions. Then normal-matrix blocks couple radial
indices separated by at most d. Verify this on **VMEX's** frozen matrix. Global
profile constraints, integral normalizations or dense gauge elimination can
break that property; retain their contribution as explicit borders/low-rank
terms or revise the chart. Never discard off-band entries just to fit a solver.

- [ ] Stream local Jacobian chunks `A_e`; accumulate local `A_e.T A_e` and
  `A_e.T r_e` without retaining the full global Jacobian. Validate assembly
  against C1, including cross-element scatter and its transpose under JIT.
- [ ] For moderate angular width, compare banded/block factorizations and grouped
  radial block-Thomas elimination (group d radial blocks to obtain a block
  tridiagonal system). Estimate O(n_radial d b²) storage and factorization cost
  before promotion; increasing angular width b can make dense blocks expensive.
- [ ] Reuse SOLVAX's existing block factorization/transpose/Schur owners. Its
  generic checked solves must report pivot quality, damping and the true residual.
  Do not interpret clamped pivots as an exact SPD solve without certification.
- [ ] For larger cases, benchmark a symmetric additive Schwarz approximation
  built from the **actual** damped normal operator, with radial overlap,
  angular/mode coupling and a spline coarse correction. Exact knot insertion
  provides transfer when nested spaces apply; verify the transpose/coarse metric.
- [ ] LSMR on the rectangular damped system is the first inner-solver change
  (item 2 above), and a same-basis block preconditioner the second (item 3).
  Normal equations square the singular-value condition number. Right
  preconditioning requires a consistent transpose and damping transformation;
  arbitrary left row scaling changes the least-squares objective.
- [ ] Wire the selected preconditioner into SOLVAX GN. Return inner convergence,
  true residual, breakdown and prediction-quality diagnostics. Use an inexact
  forcing policy appropriate to GN; tighten toward stationarity. Existing
  Eisenstat–Walker support in Newton–Krylov is prior art, not already GN support.
- [ ] Reuse the accepted-point linearization/preconditioner across rejected
  damping trials when valid. Update/factor only changed terms; measure this
  against memory retained by reusable linearization closures.

The borrowed square-root `B B.T` preconditioner was measured ineffective in
#261, with a separate 3-D scatter-transpose failure. The fused legacy constraint
experiment was measured neutral and has a parity failure at larger mode counts.
Keep their negative results; do not replay them without a new operator-based
hypothesis. Do not add deflation, learned corrections, an optimizer zoo or
another generic solver framework as the first remedy.

**Gate C3:** on the fixed C1 problems, the promoted method matches the reference
step/reduction to the declared conditioning-aware tolerance, passes transpose
checks, and reduces total time/memory at matched quality. A working target is
50–100 inner iterations at a fixed requested accuracy with bounded growth under
refinement; this is a target to test, not a promised universal iteration count.
Keep one production policy and a small reference path; delete losing experiments.

### C4. Tensor kernels and nonlinear certification

- [ ] Precompute spline/Fourier values and first/second spatial derivatives for
  each static grid. Use separable/tensor contractions, local support and bounded
  quadrature chunks. Retain parameter AD but avoid reconstructing spatial
  `jacfwd` graphs point by point. Treat singular-axis limits analytically.
- [ ] Checkpoint/rematerialize only where measured AD storage dominates. Record
  residual/JVP/VJP time, compile time, executable temporary size, peak live
  memory and rejected-trial work. Do not replace memory pressure with excessive
  recomputation without a matched total-cost comparison.
- [ ] Use the same constraint/chart/closure throughout continuation. Return the
  best valid state plus explicit failure when the budget is exhausted; never
  relabel an exhausted solve as certified by a bounded force ratio alone.
- [ ] Add deterministic progress after setup and during expensive GN/linear
  work, with elapsed cost and residual trends. Callback frequency must be bounded
  and measured; a “cost estimate” is a range based on measured kernels and
  requested budgets, not a prediction of time to convergence.
- [ ] Require independent force reduction and h/p convergence on tokamak, QA,
  QH and LASYM cases before increasing to W7-X production resolution. Fix a
  public target grid/norm/accuracy band before each campaign; retain failed bands.

### C5. Correct implicit differentiation of the promoted state

For a zero root `G(x,p)=0`, use the appropriate full constrained Jacobian. For
least-squares stationarity `g=A.T r=0`, the exact state Hessian includes
`A.T A + sum_i r_i Hessian(r_i)`; GN alone is generally not the IFT operator at
nonzero residual. Current `polish_implicit.py` retains this distinction.

- [ ] Certify stationarity before IFT, including the same scales, profile closure
  and gauge as the public solve. State how reweighting and active constraints
  enter the differentiated problem.
- [ ] Check full unpreconditioned tangent/adjoint residuals, duality, multi-step
  directional finite differences and tolerance/refinement stability of gradients.
  Use conditioning/residual estimates to tie inner tolerances to objective error.
- [ ] Test the complete boundary → solve → polish → objective chain with both
  recomputed and frozen reference data as explicitly distinct experiments.
  Reject branch changes, topology events and rank loss as ordinary smooth steps.
- [ ] For free-boundary Schur methods, freeze the same coupled root for the full
  and block solve. Compare `K.T lambda - rhs` in the original system; a good
  Schur residual or a different warm-start branch is not sufficient.

## D. Profiling and performance program

**Owners:** VMEX orchestration/physics; SOLVAX linear/nonlinear algorithms;
BOOZ_XFORM_JAX transforms; ESSOS fields/coils/orbits. Profile before changing an
owner. Extend the existing harness and retire overlapping scripts after parity.

### D1. Measurement contract

For each workflow run a fresh-process cold/empty-cache case, persistent-cache
reload, warm same parameters, warm changed dynamic parameters, and changed
static resolution. Treat objective memoization separately from compute reuse.
Time and synchronize every value, gradient, Jacobian and validation stage with
`block_until_ready`; distinguish compilation, lowering, cache lookup and actual
backend compilation. Count calls to forward solve, refinement and adjoint to
catch duplicate work and unexpected host callbacks.

Record exact git and dependency revisions, input SHA256, command, x64, device
model/count, CPU affinity/threads, driver/CUDA, selected device/sharding, problem
size, seeds, tolerances, nonlinear/linear counts, achieved certificates and
failures. Record host RSS and **device allocated/live/peak** memory separately;
JAX reserved/preallocated memory is not live tensor memory. Annotate process-wide
high-water marks so they are not misread as per-stage peaks.

Use isolated runs on idle machines; record contention/thermal state and execution
order. Repeat enough for median and spread (at least five measured warm runs and
three fresh-process runs for publication), retain all samples, and include
statistical uncertainty. A 300 s censored run is “timeout at 300 s,” not a slow
successful solve. Do not compare M3 CPU with A4000 GPU as a controlled speedup;
include an office CPU baseline on the same host and a proper GPU resolution sweep.

### D2. Mandatory workload matrix

| Family | Stages and sweeps |
|---|---|
| F1–F3 fixed boundary | Single-grid, multigrid and polished value; ns, MPOL/NTOR, finite-beta/current, LASYM, cold/new-input/changed-resolution/JIT reuse |
| F4–F6 response/scan | Value, scalar gradient, vector residual/Jacobian, parameter scan; parameter count and RHS/chunk count |
| F7–F10 optimization/coupling | Actual optimizer iterations, virtual casing, coils, single-stage and any dependency gate; first result, best feasible result, full campaign |
| F11 symmetry | Matched symmetric/LASYM value and gradient at common physical resolution, parity and compile graphs |
| High-order polish | Lift, constraint chart, certificate, r/JVP/VJP, preconditioner build/apply, inner solve, GN trial, exact stationarity adjoint |
| B1/B2 Boozer | Existing one/eight-surface transforms; extend to symmetric/LASYM derivatives, modes, phase tensors, chunking and plan reuse |
| C1/C2 confinement | Effective ripple and hard/smooth/tracked Gamma-c separately, bounce topology, pitch/line/radial resolution |
| M1–M3 and H additions | Fixed/free mirror, hybrid, then anisotropy/3-D interface; valid external fields and model certificates |
| Parallel | Independent-case throughput plus single-problem strong/weak scaling and gradient communication under E |

The September review executes bounded coverage of the shipped matrix. Missing
extras, timeouts and stages without physics certificates remain visible in the
manifest. Complete the repaired matrix before drawing publication conclusions.

### D3. Tool-driven investigation

1. Use Python wall timing and cProfile for orchestration, repeats, cache lookup,
   host synchronization and file I/O. Then label the dominant JAX regions.
2. Capture JAX traces for XProf/TensorBoard and Perfetto. The current JAX docs
   recommend XProf; TensorFlow is not required merely to profile JAX. Use
   nonblocking trace capture and bounded trace windows.
3. Inspect lowering/StableHLO/optimized HLO for captured constants, graph size,
   repeated compilations, collectives, layout conversions and fusion boundaries.
   Use compiled executable memory/cost analysis alongside the trace.
4. On office GPUs use XProf kernel timelines; use Nsight Systems/Compute when
   available and needed for launch latency, occupancy, bandwidth and roofline
   analysis. Record unavailable tools instead of claiming kernel-level coverage
   from Python-only timing. Host and device inclusive durations can overlap.
5. Run one-change ablations at matched force/gradient accuracy. Save compact
   summaries and hashes; archive raw XPlane/Perfetto/HLO files outside git.

Priority investigations: tridiagonal backend/launch count on GPU; repeated
Newton refinement and adjoint work in optimization; chart SVD and nested AD in
polish; unused preconditioner setup; Boozer contraction/transpose memory;
LASYM chunk policy; mature persistent-cache rescans; shared objective setup and
callbacks. Static metadata may be cached; physical iterates must be dynamic
arguments. Bound caches by bytes **and entries**, with a documented per-job
location for shared filesystems; never routinely erase a user's global cache.

**Promotion gate:** lower time to the same certified result and/or materially
lower memory, no unexplained gradient/branch change, no compile-count regression,
and a regression budget appropriate to the size of the improvement. A noisy
5% microbenchmark difference is not a universal speed claim. Performance
refactors should normally have nonpositive source/file growth, with justified
exceptions for new capabilities and independent scientific tests.

## E. Make parallelism and sharding real on CPU and GPU

**Status after the second review:** off the critical path. Independent-case
ensembles (`parallel.solve_ensemble`) are shipped and are what design
campaigns use; single-problem sharding is a research track whose payoff no
published equilibrium code has demonstrated. Fund it after F has a feasible
optimized design to scale, and never as a paper dependency unless the claim
is distributed execution. The probes below stay as the starting point.

**Owner split:** VMEX chooses physical work partitions and reduction weights;
SOLVAX preserves shardings and implements generic communication/solves. The
current `shard_batch`, global inner products, PCG, structured solvers and Schur
operators are foundations to reuse. No VMEX physics goes into SOLVAX.

Review baseline: 22 targeted SOLVAX least-squares/sharding tests passed. An
external probe also executed VMEX's actual strong-force row kernel and its
parameter gradient on 1/2/4/8 emulated CPU devices and 1/2 real office GPUs.
Values and gradients matched the single-device reference within the declared
1e-10/2e-9 tolerances; the multi-device reverse kernel contains one all-reduce
and no all-gather. Local shard shapes were inspected. This verifies a useful
building block, not a distributed nonlinear equilibrium solver or a speedup.

- [ ] Establish four distinct modes: one-device execution, CPU independent-case
  ensembles, GPU independent-case ensembles, and one-problem multi-device work.
  Report latency, throughput and memory for the appropriate mode. Bound CPU
  workers against XLA internal threads; sweep workers/affinity to avoid
  oversubscription and memory multiplication.
- [ ] Start with independent equilibrium cases, then independent force/certificate
  quadrature rows, Boozer surfaces and particle/field-line batches. Replicate
  modest equilibrium coefficients and shard quadrature rows initially. Compute
  local JVPs and globally sum VJP/normal contributions exactly once.
- [ ] Derive the global weighted least-squares sum and adjoint reduction, including
  any normalization by global volume. Test uneven/padded batches with masks;
  never let padding change the physical norm or gradients.
- [ ] Use explicit `NamedSharding`/mesh layouts and `shard_map` where manual
  communication is needed, compatible with the pinned JAX version. Inspect
  input, output and **intermediate** layouts/HLO; multiple visible devices or
  `vmap` alone is not distributed execution.
- [ ] Test runtime matrix/diagonal operands as well as RHS, scalar nonlinear
  objectives, JVP/VJP duality and parameter gradients. This prevents constant
  folding/replication from making a broken adjoint appear collective-free.
- [ ] Extend SOLVAX CI on 2/4/8 emulated CPU devices; these test SPMD semantics,
  not physical-node scaling. On office run 1/2 real RTX A4000 GPUs, inspect local
  shards and peer topology, and compare single-device references within a
  predeclared float64 tolerance at matched residual. Record HLO collectives and
  measured communication bytes/time in both primal and reverse passes.
- [ ] Remove hidden `device_get`, NumPy conversions and single-device callbacks
  from the promoted sharded hot path. Current implicit callbacks may require
  `JAX_PLATFORMS=cuda,cpu`; report actual placement and transfers. A host callback
  stage must be marked as such until it is replaced or explicitly supported.
- [ ] Introduce distributed radial/state decomposition only when replicated-state
  memory or scaling warrants it, after C establishes local operators. Specify
  halo/border/coarse communication and transpose semantics before implementing.
- [ ] Publish strong scaling (fixed problem), weak scaling (fixed work/device),
  efficiency `T1/(n*Tn)`, time-to-accuracy and memory/device. Include small cases
  where communication makes two GPUs slower. Keep an explicit one-device fallback.

**Exit:** one real VMEX equilibrium/optimization workflow and its gradient run
correctly with the intended multi-device layouts on CPU and two GPUs; generic
SOLVAX tests and a sharded field evaluator alone are intermediate milestones.
Scaling claims are limited to measured hardware/workloads. Multi-node/MPI is a
later extension, not implied by two devices on one host.

## F. Optimization that produces useful physical designs

**Owners:** VMEX equilibrium/objectives/constraints; SOLVAX generic algorithms;
ESSOS coils, field evaluation, engineering objectives and orbit integration.

### F0. The derivative-verification table (mandatory before any optimization claim)

What the field publishes as evidence that a gradient is right, read on
2026-09-05: Antonsen–Paul–Landreman 2019 and Paul et al. 2020 compare the
adjoint against a direct finite difference of the actual objective on a real
configuration and quote the residual (1.7e-3 fixed boundary, 4–6e-2 for
coil gradients) with an error-versus-step plot that shows the round-off
plateau; Nies et al. 2022 show the error falling linearly with the FD step
and plateauing at the resolution floor; Jorge et al. 2023 (PPCF, not JPP)
show a Taylor test with first- and second-order convergence; simsopt's
test-suite criterion is a dyadic step ladder 2⁻¹⁰…2⁻¹⁹ with a contraction
factor of 0.3–0.55 per halving. DESC Parts 2–3 publish no FD-versus-AD
check at all — their evidence is perturbation accuracy against a re-solve
and agreement with STELLOPT — which means a committed table of this kind is
a differentiator, not a formality.

VMEX today: tangent/adjoint duality 1.9e-10 and custom-VJP 8.75e-21 in
`benchmarks/polish_implicit_m4.json`, single-step centered FD gates in the
tests (solovev 1e-6, li383 2e-4 at a ~3e-5 FD floor, free boundary 1e-6,
Γc within a factor 3, max-J 3e-4, single stage 5e-3…3e-2), the li383
sign-flip anecdote in `docs/explanation/adjoint-gradients.md`, and **no
Taylor test anywhere** although sections 7.2 and 23.2 of the old plan
required one. Dot tests are internal consistency; nobody publishes them as
the primary evidence.

- [ ] One committed artifact, gated like `polish_implicit_m4.json`, with one
  row per objective that will appear in a paper (QS ratio, smooth Γc, ε_eff
  if `neo_jax` is differentiable, max-J/QI, bootstrap mismatch, D_Merc,
  ballooning, coil terms; fixed and free boundary; polished state):
  centered FD from an independent cold re-solve at tightened `ftol` at three
  or more dyadic steps with the contraction factor, the duality identity, the
  adjoint residual and tolerance, and one `ftol`/resolution sweep showing the
  plateau. Include the li383 sign flip as a figure: FD through re-converged
  roots is a genuine result.
- [ ] The polished-state VJP stale-input incident (#270) and the solver-flag
  acceptance incident (#268) are exactly the failure classes an end-to-end
  multi-step Taylor test catches; the table exists so that class cannot
  recur silently.

### F1. One small optimization interface

- [ ] Keep scalar `value_and_grad` and vector `residual_and_jacobian` routes
  through one certified response/setup owner. Reuse the solve and linearization;
  callbacks receive computed values rather than recomputing gradients.
- [ ] Give every term a fixed physical or initial scale, units, target and
  independently printed value. Do not silently renormalize away degraded field
  strength, expanded volume or violated iota. Carry real bounds/constraints
  through the optimizer; soft penalties do not guarantee feasibility.
- [ ] Benchmark TRF/Gauss–Newton versus scalar quasi-Newton only on representative
  dimensionalities and objective structure. Retain #197's documented tradeoff;
  select a small default policy rather than eight duplicated driver families.
- [ ] Use pressure/current/resolution continuation, tangent predictors and warm
  starts with branch checks. Recycle generic linear spaces only while operator
  drift/true residual remain acceptable. A looser forward solve must not merely
  shift more work into root refinement and adjoints.
- [ ] After derivative failure refine/retry or reject the outer trial and record
  the reason. A line search cannot certify the direction computed from an
  unconverged adjoint. Test the effect of tolerances on the final optimum and
  engineering feasibility, not just gradient agreement at one point.
- [ ] Record initial/best/final designs, every objective and constraint, gradient
  norm definition, step acceptance, failures, physics certificates, compilation,
  total wall time and memory. Save restartable small parameter/history artifacts.

### F2. Application sequence

**Oracles that exist today and cost nothing to pin** (read 2026-09-05; every
number that reaches a paper needs one): simsopt
`QuasisymmetryRatioResidual.total()` on `wout_LandremanPaul2021_QA_lowres.nc`
(public in simsopt's tests; VMEX ships the same input) for the QS residual —
the current test pins only an internal A/B at 10%; DESC `GammaC` on li383
at s=0.25, where VMEX's 2.8% agreement is a docstring note and not an
assertion; NEO on W7-X through DESC's public `tests/inputs/neo_out.W7-X`
(VMEX has no W7-X deck in the confinement tests; add one) beside the
existing NEO/STELLOPT pins on the QA case at rtol 5e-5; simsopt
`vacuum_well` on the same wout for the magnetic-well proxy; and the
ConStellaration dataset (arXiv:2506.19583) as a large public oracle for a
Dudt-2024-form QI residual if VMEX exposes one. Bootstrap is already at
the community norm (SFINCS RMS 3.4%/0.93%, Redl curves ≤1%, DESC trapped
fraction 5e-8). Max-J has no community oracle and needs its own Taylor
test plus a physics check.

**The flagship is a reproduction, not a new design.** Reviewers of DESC
Part 3 got STELLOPT agreement to 1.3 mm; Giuliani et al. got the
Landreman–Paul coils. VMEX's flagship optimization example is therefore
Landreman–Paul precise QA from their axisymmetric circular start, their
five-step |m|,|n| ≤ j ladder, A* = 6, ι̅* = 0.42, s_j = 0, 0.1, …, 1,
reporting f_QS through the simsopt oracle, the symmetry-breaking B_mn
(≤ 50 μT in the PRL) and NEO ε_eff, with the F0 table attached. The DESC
Part 3 two-dimensional QH benchmark (data in `DESC/publications/dudt2022/`)
is the second, cheaper comparison. The current `summary.csv` numbers
(objective 4.35e-4, aspect 6.0004, ι 0.41999) are not comparable to
anything published until the same metric is computed.

| Application | Design and objectives | Required final validation |
|---|---|---|
| Student tokamak | A few boundary shape/profile variables; aspect ratio, elongation/triangularity, iota/current target | Analytic/Grad–Shafranov checks, current/force, FD gradient and feasible bounds |
| QA/QH stellarator | Increasing boundary modes; QS, aspect ratio, iota, beta/current and geometric constraints | Native force/refinement, QS oracle, bootstrap consistency and ESSOS orbit check |
| QI/omnigenity | Shared epsilon-effective, smooth Gamma-c and maximum-J terms, with weights visible | Hard Gamma-c, second-invariant/topology checks, transport and prompt losses |
| LASYM stellarator | A physical asymmetric target/perturbation with both parity families | LASYM native/Boozer/virtual-casing parity and a scientifically useful before/after result |
| Two-stage coils | Optimize boundary, then ESSOS coil curves/currents against required external normal field | Coil/plasma and coil/coil clearance, curvature/length, field-grid convergence and traced surfaces |
| Single-stage finite beta | Coupled plasma/coil optimization using moving-boundary fields and virtual casing | Full coupled/root-adjoint certificate, feasibility and independently traced total field |
| Mirror | Throat/length/shape/current and pressure parameters after H's closure gates | Mirror ratio/well geometry, force/interface, orbit/admissibility checks |
| Periodic hybrid | Leg/return geometry, throat modulation, closure/iota and confinement, later coils | H's periodic model checks, explicit surface averages, ESSOS and applicable GKX/DKX validation |

For coil cases, start from a seed with the required topology/iota and a measured
normal-field error. The literature is unanimous with #266's measurement:
no published true free-boundary single-stage run starts from circular coils.
Every one starts from coils that already confine (CNT seeds in Baillod et al.
2025; NCSX's M45 combined run only after a two-stage design existed; stage-2
output in Jorge et al. 2023) or from a near-axis or guided stage that
manufactures ι first (Giuliani et al. 2023, whose Boozer-surface problem is
degenerate at ι = 0; Jorge, Giuliani & Loizu 2024, Sec. II C). The seeding
chain is stage-1 fixed-boundary target → stage-2 coils (circular init is
fine against a fixed target) → optional fixed-boundary single stage with a
B·n penalty → free-boundary refinement with coil degrees of freedom. The
circular-coil failure (root lost before ι reaches 8e-3) is publishable as a
short negative result. On the adjoint: Henneberg et al. 2021, Sec. 6,
state that small eigenvalues of the second variation make small equilibrium
changes produce large coil changes near marginal roots — the published form
of the stagnation #265 measured — and their Eq. 5.15 is the interior
elimination that DESC's free boundary implements as an SVD pseudo-inverse
of F_x. VMEX's `boundary_schur` is that structure; the coupled Krylov
adjoint is novel and its stagnation at a marginal root is expected. Keep
`boundary_schur` as the default *and* always report the certificate against
the original coupled transpose; `adjoint_fail="best_effort"` never appears
in a certified example. ESSOS #58 is merged, but installed/released wheels must be
capability-tested; the review's installed environment failed that test. Publish
a compatible version or pinned reproducible environment before calling the
example installable. ESSOS's latest GitHub release in this snapshot is v0.16
(2025-08-23), so a merge on main must not be equated with that release. A
circular-coil zero-iota seed is not evidence that no
circular-coil arrangement can ever form a useful stellarator seed.

Coil quadrature uses physical arclength/surface area. Check parameterization,
quadrature and resolution invariance of squared flux, curvature/torsion,
clearances and coil length. State whether a length term is a target or an upper
bound and use the corresponding residual. Protect physical field/flux scale
against normalized-objective degeneracy. Topological linking diagnostics are
integer/discrete checks, not smooth shape gradients. Add perturbation/robustness
runs for coil errors and profile changes after a deterministic design is valid.

Coordinate ESSOS issues #14/#60 on named active/frozen coil DOFs and a stable
dynamic-array/static-metadata representation. Verify active-set packing, parameter
names and transpose scatter; do not implement a second coil-DOF owner in VMEX.
Use its field batching/composition work (#15/#64) for repeated sampling. QFM
surfaces (#17) can support seed/field validation, but an approximate QFM surface
is not itself a certified ideal-MHD equilibrium.

**Exit:** each promoted example reaches a declared feasible target with certified
physics/derivatives, beats its own initial design on hard validation metrics,
and reproduces on a clean documented installation. No promise that a confinement
proxy proves zero particle loss or that a magnetic-well proxy proves stability.

## G. Confinement and native downstream interfaces

### G1. One full Boozer transform and shared diagnostic work

The full transform is already dispatched to BOOZ_XFORM_JAX. Audit the optional
lightweight symmetric path only for a measured selected-surface advantage. Keep
it only with a clear accuracy/performance boundary; otherwise remove it.

- [ ] Adopt `BoozerConfig`/`BoozerPlan` only with an available compatible release
  or explicit research pin; distinguish installed 0.1.1 from unreleased APIs.
- [ ] Reuse static mode/grid/transform tables and bounded chunks across surfaces,
  objectives and summary plots. Compare separable/blocked contractions, streamed
  modes and recomputing VJPs. Nonlinearly mapped angles are not generally a
  plain FFT replacement.
- [ ] Validate cosine/sine spectra, signs, surface ordering, analytic symmetry
  limits, JVP/VJP and memory scaling against classic BOOZ_XFORM and native DESC
  where the same quantities are defined. Warm-start nearby surfaces with explicit
  ownership and residual checks; never leak state between unrelated equilibria.
- [ ] Share Boozer/field-line inputs for epsilon-effective, Gamma-c and maximum-J
  where the mathematics overlaps. Keep missing/invalid diagnostics as unavailable,
  not zero. Plot both confinement profiles without a duplicate transform.

### G2. Hard values and differentiable surrogates

`GammaCSmooth` is shipped; the missing result is convergence of its optimization
benefit to hard diagnostics. Preserve three semantic categories: hard physical
Gamma-c, a smooth optimization surrogate, and a possible tracked-topology
formulation. Implement tracked wells only if measured bias/branch behavior makes
it useful; do not require three permanent public algorithms merely for symmetry.

- [ ] Check hard Gamma-c against a pinned DESC/independent value, including
  field-line/pitch/quadrature/radial refinement. Gamma-c squared is a least-squares
  cost, not a general physical prompt-loss law.
- [ ] Validate smooth values/gradients on analytic single-well, ordinary QA/QH
  and multi-well near-event cases. Measure bias, temperature/resolution limits,
  softmin tangencies and overflow. Do not report the surrogate as the hard value.
- [ ] If tracking wells, retain brackets/IDs, match connectivity, implicitly
  differentiate bounce roots and detect births, deaths, splits, merges and
  ambiguous assignments. Refresh topology after accepted events; do not treat
  those events as ordinary differentiable steps.
- [ ] Validate epsilon-effective against NEO, maximum-J/second-invariant signs
  and conventions, and bootstrap against the appropriate Redl/Sauter/Lin-Liu
  formula and reference. Retain approximation/domain limitations (e.g. tokamak
  fit used in stellarators); connect stronger physics tests to objectives.
- [ ] Compare final optimized configurations using hard diagnostics, ESSOS
  particle losses with sampling uncertainty, and DKX/SFINCS-compatible transport
  where applicable. Use fixed random samples during smooth optimization and
  independent samples for final validation.

### G3. Native data contract, without an interface framework

Use existing state/field/surface/Boozer adapter owners to define the smallest
structural protocol needed by actual consumers. Carry arrays/pytrees and static
metadata in memory; avoid WOUT/Boozmn/MOUT file round trips inside AD.

| Consumer | VMEX provides | Consumer owns and validates |
|---|---|---|
| BOOZ_XFORM_JAX | Native geometry/field/profile tables, all parity, coordinates | Transform, nonlinear angles, spectra, chunking and transform derivatives |
| VIRTUAL_CASING_JAX | Boundary position, normals, field/current and native derivatives | Singular/near-singular quadrature and external-field response |
| ESSOS | Equilibrium field/boundary, dimensional sampling and certified response | Coils, Biot–Savart, orbit dynamics, loss estimators and engineering terms |
| DKX | Fourier/Boozer geometry with conventions and radial derivatives | DKE, normalization, collisions and transport; closed-surface applicability |
| GKX | Dimensional local mirror/hybrid/toroidal geometry | Gyrokinetic normalization, simulation and turbulence/eigenmode semantics |

Each adapter declares topology, symmetry, pressure model, derivative support and
whether sampling is exact or approximate. Preserve working file interfaces as
external regression oracles. Do not relabel an open field line as a closed-surface
DKE, or return fake toroidal quantities from a mirror. Check one end-to-end scalar
derivative through every promoted chain, plus native/file parity. GKX's JAX floor
is higher than the core review environment; test extras in separate environments.

## H. Mirrors, anisotropy and periodic hybrids

**Owner:** VMEX physical closure, topology and certificates; ESSOS external fields;
SOLVAX generic coupled solves. This is a staged physics program, not an unchecked
extension of toroidal labels.

### H1. Harden the isotropic model and boundary contract

- [ ] Retain fixed lateral geometry and through-flux end cuts. End cuts are
  computational sections, not plasma–vacuum interfaces; do not impose lateral
  total-pressure balance there. Audit simultaneous cut geometry/flux/lambda
  constraints for compatibility and resulting end boundary layers.
- [ ] On a free lateral surface require `B·n=0` and continuity of
  `p + B²/(2mu0)`. Report each interface residual separately from interior force.
- [ ] Reject net axial current only in entry points whose exterior cannot carry
  its circulation, or implement the corresponding field/circuit with explicit
  end-electrode/return-current assumptions. A fixed-boundary interior model can
  support prescribed current under a stated end circuit; do not blanket-ban it.
- [ ] Write the exterior Laplace/Neumann/decay BVP, numerical caps and BIE/Duffy
  quadrature explicitly. A decaying exterior Neumann problem is not the interior
  Neumann problem with an arbitrary constant gauge. Net physical magnetic flux
  must still be zero; cap projection is a solenoidality consistency correction.
- [ ] Validate full circular-coil fields on and off axis, quartic analytic flux,
  long-thin two-coil limits along z at fixed flux, and force/divergence under
  radial/axial/angular/end-collar refinement. Cite the exact Ågren–Savenko
  potential used. `sqrt(1-beta)` is a long-thin estimate at its stated order,
  not inherently a small-beta expansion or an exact finite-radius law.

### H2. Consistent anisotropic closure before optimization

For static, gyrotropic, flow-free MHD use

`P = p_perp I + (p_parallel-p_perp) b b`, `J×B = div P`,

`b·grad(p_parallel) = (p_parallel-p_perp) b·grad(log B)`.

For profiles `p_parallel(s,B)`, enforce the Grad relation
`p_perp = p_parallel - B partial_B p_parallel |_s`. Check the **full parallel
projection** of `div P`; the isolated b-directed term in its expanded expression
does not generally vanish by itself. Do not choose two unrelated radial
pressure profiles where B varies along a mirror field line. DESC's tensor-force
implementation can be cross-checked with a consistent profile closure; the
problem is the selected closure, not all anisotropic DESC calculations.

- [ ] Start with isotropic recovery and a consistent analytic Delta family.
  Then add a polynomial mirror pressure family on its declared B domain, and
  only then distribution moments/tabulated sloshing-ion models. Derive p_perp
  from the same smooth p_parallel representation; preserve positivity and do not
  hide a nonsmooth cutoff under AD. For tables test interpolation derivatives
  and physical-domain boundaries as well as moment values.
- [ ] Derive weak energy and boundary terms before coding. A candidate split is
  `W = integral [B²/(2mu0) + p_th/(Gamma-1)] dV - integral p_hot,parallel(s,B) dV`,
  with the thermal mass/volume law enforced. Handle isothermal/special gamma
  cases separately; do not substitute an anisotropic pressure into the isotropic
  energy formula. Verify first variations against the tensor force.
- [ ] Check normalized firehose `sigma=1+mu0(p_perp-p_parallel)/B²` and closure
  ellipticity `tau=1+(mu0/B) partial_B p_perp |_s`, both positive in the
  admissible domain. Derive the principal symbol and analytic bi-Maxwellian
  threshold as tests. Published sign/convention differences must be resolved
  from that derivation, not copied blindly from one preprint. These conditions
  are not proof of global interchange or kinetic stability.
- [ ] CGL double-adiabatic invariants belong to a specified dynamical/stability
  model; they do not by themselves specify a unique static equilibrium closure.
- [ ] Test Cartesian manufactured tensor divergence, parallel integrability,
  exact theta-pinch total pressure, energy/residual directional consistency,
  isotropic value/residual/AD limit, and anisotropic long-thin mirror profiles.
  Use published ANIMEC **toroidal** cases for that shared closure only; use
  WHAM/Novatron/Pleiades-related data for mirrors with reproducible model inputs.

### H3. Free boundary and 3-D mirrors

Use actual ESSOS circular coils for the axisymmetric baseline. Continue from
vacuum through low pressure, boundary release, pressure/anisotropy increase,
and then small nonaxisymmetric coil/boundary perturbations. Remove the present
axisymmetric gate only after the coupled solver and all geometry/quadrature
operators retain full theta dependence.

Include interior force, lateral B·n, anisotropic total-pressure balance
`p_perp+B²/(2mu0)`, exterior response, cut constraints and any circuit unknowns
in a separately scaled coupled system. Reuse SOLVAX Newton/PTC/Schur primitives;
introduce pseudo-arclength only for a diagnosed fold. Validate plasma-off,
isotropic and axisymmetric-on-a-multi-theta-grid limits, cap/rim quadrature,
near-singular response, perturbation/FD and full coupled adjoints.

MOUT/native outputs must retain anisotropic fields, external/plasma/total fields,
coil and closure metadata, interface/force/divergence certificates and continuation
history. Produce one axisymmetric anisotropic free-boundary coil case and one
small but real nonaxisymmetric case before claiming general 3-D capability.

### H4. Give the hybrid actual mirror wells and a valid optimization target

The present hybrid is periodic and closed, with straight legs and curved
returns; it has no open-end loss. Constant leg cross-sections do not create
mirror throats inside the straight legs. Add smooth leg-radius/throat modulation
and test it against the paraxial `B∝1/a²` limit before optimizing a leg mirror ratio.

Define `R_m,axis` separately for each leg's well, `R_m,LCFS` separately, straight
length by an explicit curvature threshold, and mirror length by the maxima
bounding a |B| well. Do not export `std(B)/mean(B)` as a consumer's “epsilon”
without matching its definition. Verify equal-arc remapping, periodic geometry,
axis/section regularity, field-line closure, force and straight/toroidal limits.

Reproduce #211 on a solved state, with pinned axis transition geometry, and
separate axis representation, section rotation, quadrature and force-operator
consistency under refinement. Compare local force near leg/return junctions with
bulk error; derive the appropriate regularity/weak interface condition if a
curvature discontinuity is retained. Do not simply relax the circular-limit
threshold or infer acceptable strong force from downstream growth-rate
convergence. Replace seed-based modulation assertions with solved-state physics
tests and state which metrics a downstream consumer may admit.

Optimize low-resolution geometry and mirror wells, then iota/closure, second
invariant/QI quality, force refinement, coils and loss/transport/turbulence in
stages. Use explicit surface quadrature or a justified ensemble of field lines
where a single rational closed line cannot represent a surface average. Boozer
coordinates may be valid while a single-line diagnostic is not.

Final evidence includes before/after geometry, well/length profiles, iota,
J-contours, force/refinement, actual coil feasibility and normal field, orbit
loss uncertainty, and DKX/GKX quantities only within their model scope. Stability
proxies and curvature integrals supplement, rather than replace, equilibrium
and appropriate stability analysis.

## I. Make the project easier to learn, maintain and install

### I1. Documentation and README

Keep the existing four-part documentation structure and give each fact one
owner. Tutorials teach, how-to pages solve tasks, explanation pages derive
physics/numerics, and reference pages specify APIs/options/status. Consolidate
repeated “all of VMEX”, README and optimization/polish narratives with links and
redirects. A public module index is useful; every internal helper need not be a
user-facing API.

- [ ] First path: install → doctor → bundled solve → inspect convergence and
  force → first gradient → small optimization. Explain legacy residual,
  continuous force and derivative certificate with one worked finite-beta case.
- [ ] Keep source and wheel instructions consistent, state optional capabilities
  and supported dependency ranges, and fail early with an actionable missing
  capability/version message. Exercise minimal and full-extra installations.
- [ ] Finish #257 exported API coverage, units/shapes/returns, examples and typed
  failure semantics. Generate capability status from existing tested metadata;
  include topology, LASYM, pressure/current model, native/export and AD scope.
- [ ] Organize explanations into conventions/model, branch finder, high-order
  force, differentiation, parallel execution, diagnostics and mirror physics.
  Link algorithms to SOLVAX instead of copying its solver manual into VMEX.
- [ ] Create one validation map linking claims → equations → tests → artifacts,
  and one performance methodology/results page. State unsupported/experimental
  behavior prominently, including polish stationarity, free-boundary branch
  locality, Gamma-c topology and mirror end physics.
- [ ] README: concise purpose; install and minimal runnable example; supported
  capabilities; one representative scientific result; link to measured accuracy,
  AD and performance tables; links to tutorials, research examples and citation.
  Include cold startup as well as warm cost. Remove stale version/comparison
  claims and use IFT/discretization-qualified derivative language.
- [ ] Rebuild figures from a manifest recording generator, input/revision hashes,
  units/norms, region, hardware, date and supporting JSON. Native DESC and WOUT
  reconstruction must be labelled differently. Retire unsupported/orphan media
  after link checks; keep representative losses/failures in the comparison data.
- [ ] Verify scientific source/credit lineage before using “clean-room” or “port”
  in public/paper text. Retain licenses, contributor credit and the existing
  citation/changelog rather than inventing authors, ORCIDs or an unissued DOI.

### I2. Deliberate examples, from students to research

Reuse the existing example families listed in F2. Each example has a physical
question, editable parameters at the top, visible construction/solve/objective
steps, concise output, before/after diagnostics and a small saved configuration.
Give fast teaching/CI and research settings with measured runtime/memory and
optional dependencies. Keep ordinary Python easy to run and adapt; do not hide
physical choices behind a generic launcher or many configuration layers.

- [ ] Reuse the existing first equilibrium/gradient/optimization and coil scripts.
  Avoid one file per optimizer, degree, device or scalar/vector variant.
- [ ] Demonstrate boundary, pressure/current, solver tolerance, resolution,
  device and optimizer choices where relevant, with their physical meaning.
- [ ] Fast CI checks must test a real numerical invariant or improvement, not
  only source-text presence. Research examples add hard diagnostics, refinement,
  restartability and complete provenance without putting large arrays in git.
- [ ] Student exercises: force versus solver tolerance; h/p refinement; FD versus
  IFT gradient; parameter scaling; coil-field quadrature; model limits of mirrors.
  Supply expected ranges and explanations for failure, not just a final figure.
- [ ] Ask an external researcher/student to reproduce one installation and one
  design workflow; record friction as issues and use it to simplify the path.

### I3. Reduce code and file count through ownership

The review measured roughly 58,073 text lines in 78 files under `vmex`, 40,993
under tests, 11,495 under docs and 8,308 under examples. These are filesystem
text counts, **not cloc implementation-line comparisons**. The old plan alone
was 3,411 lines / 148,272 bytes. The largest implementation owners are
`optimize.py`, `implicit.py`, `solver.py`, `freeboundary.py`, `polish.py`,
`plotting.py`, `mirror/splines.py` and `polish_driver.py`.

- [ ] Map public symbols/imports, duplicate equations, state packing, response
  caches, optional imports and private cross-owner calls. Delete obsolete paths
  after callers/tests migrate. Do not split a large module into many tiny files
  merely to improve a line-count chart.
- [ ] Consolidate scalar/vector response reports, failure/refinement policy,
  boundary packing, free-boundary Schur orchestration and plotting diagnostics.
  Share topology-independent spline basis code while retaining different topology
  constraints. Keep canonical Boozer and coil algorithms in their owning codes.
- [ ] Retire unused square-polish orchestration/controls after a migration window,
  preserving only an independent reference if it is useful. Avoid two production
  stacks with ambiguous names and partially implemented knobs.
- [ ] For each refactor record net files/code/comments, import/startup time,
  compilation count, warm value/gradient time, memory and numerical differences.
  Prefer nonpositive LOC/file growth for refactors; do not delete independent
  validation, readable equations or useful docstrings just to meet a quota.
- [ ] Keep small inputs, manifests and summaries in git; archive large traces,
  matrices, WOUT campaigns, videos and duplicate figures with hashes and download
  instructions. Preserve the existing repository-size gate. Pruning means
  removing unreferenced generated data, not rewriting history or deleting a
  user's results without a separate decision.

## J. Release and publication evidence

### J1. Tests and release gates

Use the existing test manifest and CI lane ownership. Fast tests cover equations,
contracts and failure semantics; explicitly JIT-enabled integration tests cover
compiled forward/AD behavior; optional/nightly/weekly lanes cover cross-code,
GPU, higher resolution and costly applications. Keep deterministic input/seeds,
record skips as capability gaps, and give expensive runs budgets plus partial
reports. Coverage is useful but cannot substitute for a physics oracle.

For each numerical change require the relevant exact/manufactured test,
independent force/profile/observable check, derivative certificate and matched
performance measurement. For sharding include actual multi-device execution.
Run changed/owning tests first, then required repository checks. For documentation
require strict Sphinx/navigation/link checks and runnable examples; no new test
framework is needed for this plan edit.

A release needs source/wheel installs, minimum/selected current dependency tests,
optional-extra compatibility, CPU/GPU numerical tolerance policy, docs, changelog,
capability status and reproducible small benchmark smoke. Use tolerances backed
by observed numerical accuracy: floating-point/JIT fusion changes are not
expected to be bit-identical across devices. Do not silently relax physics or
AD tolerances to make a performance change pass.

### J2. Papers with distinct scientific claims

| Output | Main question and required result | Dependencies |
|---|---|---|
| Equilibrium/numerics paper, e.g. CPC/JCP/JPP | Can a VMEC-compatible branch finder plus local high-order solve reach accurate continuous force balance efficiently? Include analytic verification, native DESC/GVEC comparison, h/p behavior, current closure, time-to-accuracy, memory and negative cases. | A–D, B's mandatory smooth toroidal set; E only for distributed claims |
| AD/optimization/performance paper | Do certified implicit gradients and distributed work reduce the cost of feasible stellarator/tokamak/coil design? Include parameter-count scaling, FD/IFT checks, adjoint residuals, compiler/kernel evidence and complete optimization histories. | C5–G, measured E |
| Mirror/hybrid physics paper | What equilibria/designs become possible with a consistent anisotropic closure and coupled coils? Include exact limits, independent references, interface/closure certificates and scientifically meaningful optimized wells. | H and relevant F/G |
| Optional JOSS software paper | Reusable software contribution, need, design, research use and community practice; no substitute for the numerical/physics papers. | Reproducible supported release and current journal eligibility |

**The evidence bar, calibrated on 2026-09-05 against the papers that set it.**
Every peer-reviewed equilibrium-code paper of the last three years (DESC
Parts 1–3, Hudson et al. 2025 on the Shafranov shift across VMEC/DESC/SPEC,
Thun et al. 2026, AGNI 2026, SPECTRE 2026) presents the same four things,
and a reviewer will look for each by name: (i) a comparison to a named
reference code on named public configurations with the metric defined in
the text; (ii) an error-versus-resolution figure with the observed rate;
(iii) timing on a named machine with the compile/warm-up policy stated;
(iv) a code commit and a data archive. JOSS adds a checklist and, as the
GVEC review shows, does not demand cross-code validation; CPC requires a
deposited sample run. VMEC++'s paper presents none of (i)–(iii) — its
validation is a CI repository of 9 configurations × β × (mpol, ntor) × ns
= 219 variants at ≤1e-6 rel-abs with iteration counts within 20%, and its
speed claim is qualitative. VMEX already exceeds that parity bar (six gated
plus six fresh decks at ≤2.5e-10 relative, iteration counts within 25%; quote
2.5e-10 and the 1e-9 gate, never 1.4e-10 alone) and already has derivative
gates that no competitor quantifies. What it lacks, in reviewer order:

1. **A code comparison that compares codes.** The force figure's DESC row
   is VMEX's lift of a 129-surface export (7.1e-2) while DESC's own error on
   the same equilibrium is 4.0e-6 (#260). Report each code's native force
   error with one shared denominator — DESC via `compute("|F|")`, VMEC2000
   and VMEC++ via the Panici Part 1 recipe on their own mesh with the
   `s∈[0.1,0.99]` exclusion disclosed — and caption lifted rows as what they
   are. Anchor externally on the one configuration both Part 1 and Thun 2026
   used, W7-X standard at M=N=12, so VMEX sits on their published axes.
2. **Headline claims main still makes that #256 withdraws** ("a fast desktop
   CPU beats the A4000 on every production workflow", the cloc table, the
   "same number of iterations" wording, the 1885→204/9.2× preconditioner row
   that measures 1885→246/7.7×): merge #256 before any paper text is drafted.
3. **A convergence-with-resolution study of VMEX solves.** Only spline-degree
   tables and the branch-only oracle projection exist. Run the Part 1 scans
   (D-shaped: M=16, NS 64…1024; W7-X 2% β: M=N 8…20, NS to 1024, FTOL
   1e-4…1e-12) with the Part 1 metric so the radial slope is directly
   comparable to their −1.02 for VMEC.
4. **Timing methodology.** Five repetitions with min and median; a
   serial-versus-serial row (XLA intra-op threads 1, `OMP_NUM_THREADS=1` for
   VMEC++, VMEC2000 `-n 1`) beside a matched-core row; compiler and flags for
   the Fortran; the compile census as its own column (already better than
   most papers); ratios per deck, never "14 of 14 rows".
5. **Parity breadth and provenance.** Run the public VMEC++ validation
   matrix inputs through the existing harness and report per-variable-class
   maxima; add one LASYM and one free-boundary deck to the fresh set; put the
   STELLOPT source commit and compiler beside the binary hash in the
   artifact; add STELLOPT's own `BENCHMARKS/VMEC_TEST` decks, whose upstream
   gate is 5% at mid-radius.
6. **One independent physics anchor** of the Hudson 2025 type: the
   rotating-ellipse vacuum case (R0=10 m, a=1, b=0.25, N_P=5, 10 coils per
   period, Ψ_edge=−1.4652431) is fully specified in that paper; reproducing
   its Tables I–II (energy error ~1e-7, δb_n ~1e-5, axis position) places
   VMEX beside VMEC, DESC and SPEC on a third party's numbers.
7. **Packaging.** A data DOI for the benchmark artifacts at submission; all
   figures in the manifest; a JOSS `paper.md` only if that venue is chosen;
   CITATION.cff must not assert "certified force-balance polishing" while
   the polish gain is under revision.
8. The Solov'ev oracle (#264) is at the field's bar; pre-empt two questions:
   say why O(h^(p−1)) is the expected order for a force built from second
   radial derivatives of a degree-p spline, and add the 1983 experiment —
   a VMEX *solve* of the Solov'ev deck scored against the closed form (axis
   R₀, ι(s), |B|) at increasing ns and mpol — because the projection test
   validates the oracle, not the solver.

Two competitor facts to carry into the positioning: VMEC++ v0.6.0 (2026-07)
shipped LASYM and v0.7.2 (2026-08) shipped adjoint boundary sensitivities
with a SIMSOPT analytic gradient, so "differentiable VMEC" is no longer a
unique claim — the implicit-derivative *certificates* and the polished-state
derivatives are; and DESC's anisotropic pressure exists in code since
v0.10.2 with only a zero-anisotropy regression test and no paper, so H's
closure work has no published competitor yet.

Predeclare the benchmark set and acceptance bands. Present accuracy/runtime/memory
Pareto curves, success/failure coverage and confidence intervals. Compare native
representations and exported compatibility separately. Show gradients versus
number of design variables **including setup/compilation**, and speedups only at
matched accuracy/hardware where a controlled comparison is possible. A single
favorable row cannot support a global “faster and more accurate than DESC” claim.

Archive release-tagged code, small inputs, exact environments, raw numerical
results, figure scripts and optimized parameters with persistent identifiers at
submission. Maintain CITATION.cff, contributor/ORCID information supplied by the
authors, license/source credit, changelog, metadata, and an honest AI-assistance
and verification disclosure. Add DOI/badges only when a record exists. Check the
current journal's requirements at submission; JOSS currently includes public
history/research-impact gates, not just an installation checklist. A DOI and
external publication are later maintainer actions, not blockers to this plan.

## 4. What to learn from DESC and other codes

These are source-grounded implementation lessons and bounded reference roles.
Open PRs are not shipped capabilities, and a closed branch may be stale even
when its name sounds relevant. Exact inventories and head SHAs are in the review
manifest/external snapshots.

| Source | Lesson for VMEX and its boundary |
|---|---|
| DESC master force/basis/objectives | Regular Fourier–Zernike axis behavior, analytic derivatives, independent resolution controls and normalized objectives are useful. Its force component weighting is also not automatically the same as VMEX's proposed Cartesian volume L2; match metrics explicitly. |
| DESC least-squares implementation | Current trust-region path supports QR/SVD/Cholesky choices; QR is the default examined here. A dense/reference factorization is valuable before scalable preconditioning, but need not become VMEX's large-case default. |
| DESC #1773, `ku/shard`, and #1495, `yge/multigpu` | Sharding/MPI work is experimental and workload dependent. Inspect current diffs, dependency versions, collectives and AD; prior JAX compatibility failures and setup costs matter. Do not claim general multi-GPU speedup from the branch name. |
| DESC #2170, #2267, #2281; issues #1686/#1872/#2171 | Sparse pullbacks, batching, releasing old Jacobian/factor buffers and shared value/gradient work guide D. Measure the complete derivative graph and avoid duplicate solves. |
| DESC #2031, #2286; issues #1569/#2000/#2087 | Proximal/QR, chunked linearization and conditioning work remain relevant. Regularization changes sensitivity; the documented difficulty differentiating public `eq.solve` motivates testing VMEX's actual public composable chain. |
| DESC #2304/#944, #2302/#2301 | Dynamic normalization and coil quadrature can change the optimization target or create resolution dependence. Include physical scales, parameterization and quadrature invariance in F. |
| DESC #1877/#1876, `dp/lambda-resolution` | Near-axis iota/current and lambda/grid resolution require dedicated constraints and tests, not an assumption that spectral code is automatically exact. |
| DESC #1848, mirror/anisotropy branches | Useful geometry and tensor-force prior art, still a research branch. Validate a consistent B-dependent pressure closure and the same open/periodic model before treating it as an oracle. |
| DESC #2309, #2215, #1676, #994 and bounce work | Smooth J/topology objectives, available energy and omnigenity research guide G, with hard diagnostics and topology-event validation. |
| DESC #2255/#2317 and documentation issues | Student workflows, scaling tutorials, reduced figure size and realistic runtime guidance belong in the main usability path. |
| SOLVAX #98/#96/#86/#32/#66/#71/#28 | GN, Schur, forcing, recycling, sharded batch/operands and collective tests already exist in varying scope. Extend those owners; do not reimplement them in VMEX. Issue #74's large complex Arnoldi work is not a prerequisite for real GN polishing. |
| ESSOS merged #58; open #61–#66 | Moving-boundary coil fields and optimization interfaces are on main; in-memory VMEC, combined fields, linking and winding-surface/objective repairs need release/source checks. Keep field/coil geometry and engineering quadrature in ESSOS. |
| GVEC | Closest spline/Fourier equilibrium precedent: arbitrary-degree radial B-splines and flexible axes. Compare radial convergence, constraints and energy formulation; do not present B-splines alone as VMEX novelty. |
| VMEC2000 and VMEC++ | Profile/normalization/constraint and free-boundary compatibility references. Refresh feature tables against pinned source; old abstract limitations may no longer describe current code. |
| SIMSOPT | Clear parameterized examples, finite-beta single-stage objectives and coil engineering/robustness checks. Use it for independent objective/field oracles, not as a source of copied solver abstractions. |
| SIESTA, SPEC, HINT; ANIMEC/Pleiades | Different topology or anisotropy assumptions bound the claim. Numerical agreement requires the same physical model; unsupported combinations must remain explicit. |

## 5. Literature mapped to concrete work

Use primary papers, source/manuals and textbooks together. Record the exact
version/equation and input data used for each new test. References below ground
the program; they are not claims that every published result has been reproduced.
The existing bibliography retains detailed diagnostic citations corrected in
#250. Bibliographic metadata alone is not sufficient to implement an equation.

| Reference | Required use |
|---|---|
| Hirshman–Whitson (1983), [10.1063/1.864116](https://doi.org/10.1063/1.864116); Hirshman–Betancourt (1991), [10.1016/0021-9991(91)90267-O](https://doi.org/10.1016/0021-9991(91)90267-O) | VMEC variational/normalization and preconditioned-descent baseline, B/D |
| Schilling, [The Numerics of VMEC++, arXiv:2502.04374v3](https://arxiv.org/abs/2502.04374v3) | Trace conventions, profiles, constraints and implementation heuristics to source, B |
| Panici et al. (2023), [DESC Part I](https://doi.org/10.1017/S0022377823000272) | Native force, near-axis/stability and time-to-accuracy methodology, B/J; reproduce the defined norm and region |
| Conlin et al. (2023), [DESC Part II](https://doi.org/10.1017/S0022377823000399); Dudt et al. (2023), [Part III](https://doi.org/10.1017/S0022377823000235) | Perturbation/continuation and constrained optimization baselines, C/F |
| Hindenlang et al. (2026), [GVEC](https://doi.org/10.21105/joss.09670), [current manual](https://gvec.readthedocs.io/develop/index.html) | Spline/Fourier and flexible-axis prior art, constraints and comparison cases, B/C/H |
| Hirshman et al. (2011), [SIESTA](https://doi.org/10.1063/1.3597155) | VMEC initialization followed by a different equilibrium method; distinguish topology/model from accuracy polishing |
| Freidberg, [Ideal MHD](https://doi.org/10.1017/CBO9780511795046); Helander (2014), [Theory of plasma confinement in non-axisymmetric magnetic fields](https://doi.org/10.1088/0034-4885/77/8/087001) | Equilibrium/stability assumptions, nested-surface limits and confinement interpretation, B/F/H |
| Fong–Saunders (2011), [LSMR](https://doi.org/10.1137/10079687X), [SOL implementation](https://web.stanford.edu/group/SOL/software/lsmr/) | Rectangular reference, conditioning, true residual and transpose requirements, C |
| Eisenstat–Walker (1996), [10.1137/0917003](https://doi.org/10.1137/0917003); Kelley–Keyes (1998), [10.1137/S0036142996304796](https://doi.org/10.1137/S0036142996304796) | Inexact forcing and PTC where justified; retain exact terminal certificates, C/F |
| Sangalli–Tani, [arXiv:1602.01636](https://arxiv.org/abs/1602.01636); Hofreither–Takacs, [arXiv:1607.05035](https://arxiv.org/abs/1607.05035); Kronbichler–Kormann, [10.1145/3325864](https://doi.org/10.1145/3325864) | Spline preconditioning, coarse spaces and tensor/matrix-free kernels; verify applicability to VMEX's operator, C/D |
| Cooper et al. (1992), [10.1016/0010-4655(92)90002-G](https://doi.org/10.1016/0010-4655(92)90002-G); Cooper et al. (2009), [10.1016/j.cpc.2009.04.006](https://doi.org/10.1016/j.cpc.2009.04.006) | Anisotropic energy/free-boundary derivation and toroidal reference cases, H |
| [Analysis of influences of pressure anisotropies on the 3-D MHD equilibrium in LHD](https://doi.org/10.1063/5.0033807), Eqs. 7–8 | Independent sigma/tau convention and sign checks for closure ellipticity, H2 |
| Endrizzi et al. (2023), [WHAM physics basis](https://doi.org/10.1017/S0022377823000806); Frank et al. (2026), [10.1063/5.0306291](https://doi.org/10.1063/5.0306291), [preprint v4](https://arxiv.org/abs/2509.17288v4) | Mirror pressure moments, reconstruction, physical inputs and closure comparisons, H; verify source equation signs independently |
| Lindvall et al., [Novatron: Equilibrium and Stability](https://arxiv.org/abs/2503.03387); Ryutov et al. (2011) and Ågren–Savenko (2004/2005), as indexed in mirror references | Polynomial closure, long-thin limits, potential tests and model-specific stability, H; identify exact equations before implementation |
| Cary–Shasharina (1997), [10.1103/PhysRevLett.78.674](https://doi.org/10.1103/PhysRevLett.78.674); Nemov et al. (2008), [10.1063/1.2912456](https://doi.org/10.1063/1.2912456); Velasco et al. (2021), [10.1088/1741-4326/ac2994](https://doi.org/10.1088/1741-4326/ac2994) | Omnigenity, Gamma-c definitions and limits of prompt-loss proxies, G |
| Unalmis et al., [differentiable bounce averaging](https://arxiv.org/abs/2412.01724), [published 2026](https://doi.org/10.1017/S0022377826101652); Ochs (2025), [multi-well domains](https://doi.org/10.1017/S002237782510069X) | Bounce quadrature/AD and topology handling, G |
| Landreman–Paul (2022), [10.1103/PhysRevLett.128.035001](https://doi.org/10.1103/PhysRevLett.128.035001); Landreman–Jorge (2020), [10.1017/S002237782000121X](https://doi.org/10.1017/S002237782000121X) | QS objective and near-axis Mercier tests with correct equation conventions, B/F |
| [Single-stage stellarator optimization](https://arxiv.org/abs/2302.10622), [SIMSOPT finite-beta example](https://github.com/hiddenSymmetries/simsopt/blob/master/docs/source/example_single_stage.rst) | Matched plasma/coil objectives and practical engineering examples, F |
| JAX [benchmarking](https://docs.jax.dev/en/latest/benchmarking.html), [XProf/profiling](https://docs.jax.dev/en/latest/201/profiling.html), [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html), [AD and sharding](https://docs.jax.dev/en/latest/301/sharding-ad.html) | Synchronization, profiler interpretation, explicit reduction/transpose semantics and version-aware device work, D/E |
| [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html), [submission requirements](https://joss.readthedocs.io/en/latest/submitting.html), [FAIR4RS](https://doi.org/10.15497/RDA00068) | Software/research evidence, attribution, reproducible archives and current submission gates, J |

Neural post-correction and deflation papers are later comparison/hypothesis
sources, not substitutes for C's residual/chart verification. Reassess literature
at each paper freeze; publication dates and arXiv versions matter.

## 6. Open pull requests: merge order and the edits each still needs

Seventeen pull requests were open on 2026-09-05. Their file overlaps were
computed against each PR's own base; the order below minimizes rebases of the
large branches and lets the API-docs guard (#257) check everything that lands
after it. "BLOCKED" on GitHub means unsigned commits and is cleared with an
admin merge once the real lanes are green; `codecov/project` is an external
status and never blocks.

| Step | PR | State on 2026-09-05 | Action before merge |
|---|---|---|---|
| 1 | #267 → #268 → #269 → #270 | green; stacked | Merge in stack order; this revision is stacked on #270 |
| 2 | #264 LASYM/Solov'ev | green | Merge |
| 3 | #263 mirror audit | green | Merge; rebase only if plan.md conflicts with step 1 |
| 4 | #262 failure policy | green | Merge |
| 5 | #260 native DESC | green | Merge |
| 6 | #258 certificate diagnostics | **merged 2026-09-05** (1716594a) | — |
| 7 | #259 symmetry flags | green except codecov | Merge |
| 8 | #254 cache entry bound | **merged 2026-09-05** (928d4fc3) | — |
| 9 | #253 hygiene | green | Rebase on 1–8 (README, performance.rst); merge |
| 10 | #256 figures/validation page | green | Rebase on 9 (README, performance.rst, manifest); merge |
| 11 | #257 API reference | green except codecov | Rebase on 1–10 (touches polish/implicit/virtual casing/omnigenity); its docstring guard then covers everything above; merge |
| 12 | #265 optimizer output + free-boundary adjoint | green except codecov | Split: the trial printing and the duplicate-gradient fix merge; `adjoint_fail="best_effort"` stays opt-in and is never used by a certified example (F1) |
| 13 | #266 coil target/seed/profile | stacked on #265 | Merge after 12 |
| 14 | #261 polish memory/AUTO/heartbeat | parity c2 red: missing artifact; branch rebased onto main with #258 (27 commits, clean) and two commits the dead agent never pushed are now on it | Commit `benchmarks/polish_memory_w7x.json` from the office rerun (first run: `measurement_dirty` true because the harness's own `arm_*.json` dirtied the checkout — use `--scratch`; second run: the `auto` arm died with SIGSEGV in the chart build at 6.85 GiB, an intermittent native crash on the office box that must be recorded with the artifact; third run in flight); then 0.8.2 |
| 15 | #197 contributor examples | behind main | Rebase (`codex/scalar-optimization-drivers-rebased` is clean); add the one-sentence trade note; merge as examples |
| 16 | #275 force-error normalizations | preflight PASS, 33 strong-force tests; rebased onto main after #258 (one appended-tests conflict resolved) | Re-verify against #258's `nestedness_margin` redefinition (its committed certificate baseline predates it), push, merge; it retires every "26-fold" number |
| 17 | #272 Γ_c prompt-loss wording | docs only | Merge; it corrects a statement #250 introduced (below) |

After step 14, tag **v0.8.2** with #261, #254, #258, #259 and #262: a
user-facing release (no OOM at W7-X scale, a priced AUTO, a heartbeat, a
cache that does not slow down with age, correct certificate diagnostics)
that makes no new 3-D accuracy claim.

## 7. The next ten pull requests, in order

1. C0 experiment on the frozen QA linearization (C1 items 1–3 on one saved
   state; augmented QR vs PCG vs LSMR; independent-force reduction per step),
   committed as an artifact with the verdict written into C0.
2. #261 artifact and 0.8.2 (step 14 above).
3. F1: the derivative-verification table across every shipped objective
   (tangent/adjoint identity, three-step directional finite differences,
   stationarity Taylor check for polished states) as one committed artifact
   and one docs table. This is what every reviewer will ask for first.
4. A/B: `eps_F` disclosure plus the B2 norms in the certificate and in every
   README/docs number. The 31.2-R1 branch `physics/force-error-normalizations`
   holds four pushed commits (2 722 insertions, 18 files, no PR yet): the
   non-saturating normalizations, the withdrawal of the 26-fold claim with
   the like-for-like numbers (7.12× in the bounded metric, 1.32× whole
   domain, 1.06× on s∈[0.1,0.99] in ⟨|F|⟩/⟨|∇p|⟩), the selection statement,
   and a baseline-regeneration tool. Its agent died before verifying it:
   rebase onto #258/#261 (all three touch `strong_force.py`), run the
   strong-force and polish suites, then open the PR.
5. B3: native DESC vs native VMEX at matched physical points on the shaped
   tokamak and the finite-beta QA, extending #260 from "what DESC measures"
   to "the same quantity for both codes".
6. `scale_wout` profile-coefficient consistency (task chip open), the
   GKX `epsilon` export alignment and the mirror figure writer (task chips).
7. F2 row 2: reproduce Landreman–Paul precise QA from the published initial
   condition with the QS residual reported in their definition, as the
   flagship optimization example, with the F1 table attached.
8. D3 tridiagonal backend on GPU (the 4 480 pivot-kernel launches and the
   Thomas ablation): promote or reject with the D1 measurement contract.
9. I1 first-path tutorial that runs end to end on a clean install, and the
   issue #157 snippet.
10. J2 paper 1 outline with the evidence map, written against the C0 outcome.

## 8. Environment and resume facts (moved from the old ledger's 31.8)

- The local checkout `/Users/rogeriojorge/local/vmex` is shared between
  sessions; work in a worktree, push after every commit, author as
  `Rogerio Jorge <rogerio.jorge@wisc.edu>`, never any assistant attribution
  in commits or PR bodies. One heavy local run at a time; the machine has been
  crashed by two.
- There is no venv inside the repository. The measurement rig venvs live at
  `/tmp/claude-501/localab/env-{0.3.0,0.8.0,dev}` (`env-dev` is editable on
  the shared checkout and has essos); `/opt/local/bin/python3` also imports
  vmex and essos. Run tests from a worktree with `PYTHONPATH=$PWD`.
- `tests/conftest.py` sets `jax_disable_jit=True` for the whole suite (R15);
  a test of traced behaviour must opt in with `jax.disable_jit(False)` or the
  `_module_jit_enabled` fixture.
- Office box: `ssh office`, 36 cores, 62 GB, 2×RTX A4000; checkout
  `~/vmex-gpu`, interpreter `~/venvs/vmex-gpu/bin/python`, run with
  `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` (or `cuda,cpu`); clear
  `~/.cache/vmex ~/.cache/jax` before a run (stale cross-profile AOT entries
  segfault); launch long jobs with `setsid nohup … < /dev/null &` and bound
  them with `ulimit -v`. A `pkill -f` pattern that matches the ssh command
  line kills the ssh session. Reference binaries: `~/bin/xvmec2000` (2024
  STELLOPT snapshot) and the modern `~/local/STELLOPT_new/bin/xvmec2000`.
- Persistent-cache facts: entries are `*-cache`/`*-atime` pairs in
  `~/.cache/vmex/jax_cache/<fingerprint>`; every write rescans the directory
  (#254); programs with host callbacks are never cached.
- The merge bar: local verification plus every substantive CI lane green;
  admin-merge over "BLOCKED" (unsigned) and over `codecov/project`; never
  over a red physics or parity lane.

## 9. Migration from the old plan and execution discipline

No unfinished physics goal is dropped by shortening the old ledger. Use this map
when closing existing PR references; do not append another contradictory ledger.

| Previous requirement | New owner / status |
|---|---|
| Integration, input/CLI polish, public results, export/native state | Landed baseline plus A/B1 hardening |
| Performance phases 1–3; startup/cache/W7-X ledgers | A, C, D, E; old timings remain historical evidence |
| Boozer, combined summary, Gamma-c phases 4–6 | Partly landed; G completes convergence and native reuse |
| LASYM phase 7 | B/C/F; #264 rejects an isolated `tcon` halving |
| Isotropic/anisotropic/free mirror and hybrid phases 8–11 | H, with explicit corrected closure/topology/metric contracts |
| Native protocol and downstream phase 12 | G3, using existing owners and exact model boundaries |
| Slimming/docs/examples phase 13 | I; no duplicated documentation tree or optimizer-file matrix |
| Validation, figures, optimized configurations and release phase 14 | B/F/H/J; selected-winner figure policy removed |
| 31.2 R1–R7 and recommendations | A/B/C; #260 native-DESC and #264 exact-Solov'ev work remain open |
| 31.2 R4/R8/R9 explanations | Corrected by #255; verify against subsequent algorithm changes |
| 31.3 diagnostic formulas/citations | #250 landed with one error of its own: it said Velasco et al. 2021 eqs. (20)–(21) relate prompt losses "approximately linearly" to Γ_c. Eq. 20 is a rescaling, eq. 21 the \|γ_c*\| variant; their section 5.4 finds the loss fraction follows the eq. 21 variant, Γ_c itself does not predict it, and Γ_α is the validated predictor. #272 corrects the three sites. Hard independent values/refinement and scientific optimization remain G |
| 31.4 mirror findings and spec sheet | H; corrected full parallel projection, closure eligibility and exterior-BVP interpretation |
| 31.5 publication items | #245/#249 landed; #253/#256/#257 pending, I/J; DOI at submission |
| 31.6 #197 decision | F/I preserve measured scalar/TRF tradeoff with fewer duplicated examples |
| New source findings and profiling gaps | R1–R16 → A–I; tracked independently of earlier PR claims |

Implement the next open gate in dependency order. A numerical PR contains one
physical or algorithmic hypothesis, a reproducer, the relevant independent
checks, a compact performance comparison and an updated capability statement.
A performance result is not complete while accuracy is unknown. A physics
feature is not complete while its public usage and failure mode are unclear.

Update the affected checkbox/table and link the merged commit/artifact in place.
Keep a compact execution logbook here with the active branch, checks, blockers
and exact next action so an independent agent can resume. Link detailed history
in git/PRs; keep this file focused on decisions still needed. Do not create a new report file for each experiment when the existing
manifest can hold a record. Commits made for this work are authored by
`rogeriojorge`; preserve existing contributors' authorship when incorporating
their changes. Release publishing, external announcements and archival deposits
are separate maintainer actions after their concrete artifacts are ready.
