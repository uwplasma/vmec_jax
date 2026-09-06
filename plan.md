# VMEX research plan: third take (2026-09-06)

Base: main `ae0e410f` (0.8.1). This is one of three plan proposals on the
table and is written so an independent reviewer can merge them into one
authoritative plan:

- [#274](https://github.com/uwplasma/vmex/pull/274): my first revision after
  a literature review (2026-09-05). Superseded by this document; its C0/F0/J2
  ideas survive here in corrected form.
- [#282](https://github.com/uwplasma/vmex/pull/282): the focused P0–P4 plan
  with fresh probes (VMEC++ 0.7.3 same-deck parity, native DESC, a frozen
  156×17 operator, a capped GVEC run) and a 220-line README.
- This take: the same P0–P4 letters so the reviewer can diff, but with a
  different answer on the high-order solver, an explicit cost discipline for
  the repository itself, and external oracles instead of self-verification.

**No release is scheduled.** One release follows integration of the agreed
plan, when its gates pass.

## 0. Where this take differs from #282

| Topic | #282 | This take | Why |
|---|---|---|---|
| High-order solver | Verify the functional and the chart, take a dense reference step, then derive one banded preconditioner for the existing Gauss-Newton loop | Same first two steps, then **stop preconditioning the normal equations**: use the dense Jacobian with QR/LM where it fits (≤ 3,000 unknowns, ≤ 2 GiB), and test variational Newton on the MHD energy for the rest | Collocation least squares on a second-order operator yields a fourth-order normal operator; its condition number scales as h⁻⁴ against h⁻² for the energy Hessian. The committed tokamak record (80 Gauss-Newton steps, 47,308 CG iterations, 463 s on 148 unknowns) is that scaling, not a tuning problem |
| Energy Hessian | "Valid only after verifying the differentiated constrained energy yields the intended force" | Agreed, and that verification is a one-day test that already has its oracle: the reverse-mode gradient of W must equal the weak projection of the certificate's Cartesian force onto the basis | The certificate evaluates F independently; the test is a duality check, not a new derivation |
| 3-D | Bounded recovery at one QA resolution with three radial and two angular refinements | Same, plus a resolution ladder in (M, N) with DESC and VMEC++ at matching resolution as the published figure, and a kill criterion for the 3-D lane as a product | Thun et al. show angular truncation is the floor; a ladder measures it, a budget does not |
| Derivatives | Independent re-solves and objective Taylor remainders | Same, plus a direct vector comparison against VMEC++ 0.7.2's adjoint and simsopt's analytic gradient on identical decks | Two independent codes agreeing to 1e-6 is stronger than any finite-difference table |
| Repository cost | Track lines per PR; retire wrappers after ownership is settled | Named deletions with line counts now (homotopy route 1,367 lines, shims, test-only public API), CI tiering to a 25-minute PR ceiling in week 1 | Yesterday's 21 merges each re-dirtied the queue behind them at 44 minutes per lane; the cost is measurable and immediate |
| jax versions | Record the failing CI context for #277 | Set the policy: floor 0.9.2 and head 0.11.x both tested, tolerances by observed accuracy, no cross-version bit claims; bisect #277 with XLA flags | The same test converges to 1e-10 relative on 0.9.2 and not on 0.11.1; that is a policy gap, not a bug hunt |

Where #282 corrected #274, it was right: finite-difference derivative tests
exist (twelve files, fixed steps); a spectral tail is a flag, not a proven
floor; a published W7-X study does not bound a different representation; the
0.8.2 tag is withdrawn. #282's README and validation-page edits should merge
whichever plan wins; one leak remains outside them (`CHANGELOG.md:57` still
carries the withdrawn 26-fold number).

## 1. Product definition

VMEX is the fast, differentiable, VMEC-compatible equilibrium solver with an
independent continuous force certificate and certified implicit derivatives.
Everything else (high-order polish, mirrors, hybrids, downstream consumers)
is either a validated scope with its own lane or a research lane with an
entry criterion. The first paper is about the product. A second paper about
accurate native 3-D force balance exists only if P2 produces it.

## 2. Evidence at the baseline

Measured on clean main at `ae0e410f` on 2026-09-06 (Apple M4, jax 0.9.2,
SOLVAX 0.20.0) and by three read-only audits of code, documentation and the
2024–2026 literature.

| Fact | Value | Source |
|---|---|---|
| Code | `vmex/` 63,342 lines (core 52,563, mirror 9,376); 1,275 top-level definitions, 93 exported; 120 public definitions (4,883 lines) consumed only by tests | AST census |
| Tests | 1,898 collected; 115 `full` decorators; PR lanes: fast 544 passed in 78 s; core 106 in 6 m 15 s; implicit-response-a/b 9 and 15 in 3 m 20 s; free-boundary-adjoint 1 in 1 m 48 s; field-api 86 in 4 m; mirror-equilibrium 49, mirror-field 31; all green | local lane runs with CI's own selectors |
| CI | e-polish 42–44 min, c3d 31–44 min, parity budget 55 min; a merge that touches the plan logbook or a rewritten docstring dirties every queued PR | run logs, 2026-09-05 |
| Where the force error lives | shaped tokamak, VMEC state lifted to the continuous basis: near axis 977 N m⁻³, bulk 163, edge 409; after polish 67 / 147 / 284 | `benchmarks/polish_force_error_2026-09-03.json` |
| What the polish costs | 80 Gauss-Newton steps, 47,308 CG iterations, 463 s, 148 unknowns, solver flag false, certificate true | `benchmarks/strong_force_cases_m4.json` |
| W7-X polish setup | certificate 3.24 GiB, chart 16.6 GiB and 1,751 s before any solve | `benchmarks/polish_memory_w7x.json` |
| Stale claims on main | README solver matrix marks VMEC++ without LASYM and derivatives (wrong since 0.6.0/0.7.1/0.7.2); withdrawn 26-fold number in `CHANGELOG.md:57`; `performance.rst:899–953` cites three artifacts that do not exist; two benchmark records cite dead plan numbering | grep |
| Test hygiene | four modules flip `jax_disable_jit` and never restore it (`test_scaling.py:131`, `test_cli_freeboundary.py:50,65`, `test_tracing.py:42`, `test_optimize.py:63,329`) | grep |
| Vocabulary | "sharding" in code and plan is single-device placement; no `Mesh` or `NamedSharding` exists | grep |
| Field | VMEC++ 0.6.0–0.7.3 shipped LASYM, multigrid in mpol/ntor, Python-driven iteration, opt-in Enzyme AD, an exact force Jacobian and a boundary adjoint with a SIMSOPT wrapper; DESC 0.17.3 has QR reuse and exponential spectral scaling and pins `jax<0.10`; Thun et al. (NF 66, 2026) define `F_norm` and show the VMEC axis spike at 2,048 surfaces and the truncation floor; GVEC is a JOSS paper with banded radial preconditioners per mode; SPECTRE succeeds SPEC | release pages and papers, checked 2026-09-06 |

## 3. The three technical decisions

### 3.1 Which functional, which solver

The high-order lane minimizes the strong residual at collocation points by
Gauss-Newton with preconditioned CG on the damped normal equations. The
force is second order in the geometry (B carries first derivatives, J
second), so the normal operator JᵀJ is fourth order and its condition number
grows as h⁻⁴ in the radial resolution. The MHD energy W = ∫(B²/2μ₀ +
p/(γ−1))√g is first order in the geometry, its Hessian is second order and
symmetric, and the classical VMEC preconditioner (Hirshman and Betancourt
1991, block-tridiagonal in the radial index per (m, n)) was built for exactly
that operator. This is why the tokamak record needs 590 CG iterations per
step and why another preconditioner for JᵀJ is the wrong target.

Three solvers can run on the existing `ρ^|m| q(s)` representation; the
experiments below decide between them, in this order, each with a kill rule.

**E1, functional consistency (one day).** For random directions v in the
chart, compare `⟨∂W/∂c, v⟩` from reverse-mode AD with `−∫ F_cart · (∂x/∂c v)
√g` from the certificate's own Cartesian force at the same quadrature. Also
check the strong residual's Jacobian against the same object by duality.
Pass: agreement to 1e-10 relative on Solov'ev and the tokamak with
prescribed iota and γ = 0. Fail: the energy route stops here and the reason
(closure, boundary term, gauge) is recorded. #282's chart-rank and
gauge-quotient audit runs alongside on the same tiny reference.

**E2, dense reference step (one week; #282's steps 1–2).** Assemble J on
the frozen tokamak and one affordable QA linearization (≤ 2 GiB, ≤ 10 min
per job) and take the augmented QR/SVD step with the actual scaling. This is
DESC's solver, not a diagnostic: at ≤ 3,000 unknowns the Jacobian is under
1 GiB and the factorization is seconds on the office box. Pass: the
independent certificate improves by the order of magnitude #282 sets as
the QA target. Fail: the obstruction is representation, reachability or
closure, and no Krylov work follows.

**E3, variational Newton (one to two weeks, only if E1 passes).** Newton on
∂W/∂c = 0 with Hessian-vector products from forward-over-reverse AD,
MINRES or CG inner solves at Eisenstat–Walker forcing, the banded per-(m, n)
radial preconditioner assembled from the second variation (GVEC's
`mhd3d_evalfunc.F90` is the template), trust-region acceptance with geometry
rejection, and the strong certificate as the judge. Pass: the certified
tokamak value at fewer than 50 Hessian products per step and 20 steps, and
h/p convergence on Solov'ev at the degree's order. Kill: an indefinite
Hessian where VMEC converges, or no 5× reduction in operator applications
against E2's dense step.

**Decision.** If E2 passes and E3 does not, the 3-D solver is dense LM with
an explicit memory-bounded resolution cap. If both pass, E3 is the scalable
mode and E2 the reference. If E2 fails, the lane keeps experimental status
and the logbook records the obstruction; no W7-X run is launched.

### 3.2 Coordinates and the axis

Keep `ρ^|m| q(s)` with B-splines in s and quadrature in ρ; it is the DESC
regularity written locally and it already removes the axis spike (977 →
67 N m⁻³ near the axis on the tokamak). A ρ-uniform mesh for the
finite-difference lane is not pursued: Thun et al. show the spike at 2,048
uniform surfaces, so it is the axis row (`X(js=1) = X(js=2)`,
`geometry.py:22`) and the half-mesh closure, not the spacing. Polar
center-splines (Jiang et al. 2026) and a generalized toroidal angle (DESC
#2282) stay conditional exactly as #282 lists them.

What changes for users now: `solve()`'s summary and the validation page
report the near-axis, bulk and edge force error separately, so the
finite-difference lane's axis error is visible without polishing. If E3
passes, the regular representation becomes a mode of the same solve
(seeded by the finite-difference lane, VMEC mesh kept for parity and
export) and the polish driver's two routes, four preconditioner variants,
AUTO pricing and heartbeat collapse into one Newton loop.

### 3.3 3-D: a ladder, not a budget

The angular truncation is a floor only resolution crosses. The 3-D
experiment is the same QA deck at (M, N) = 5, 7, 9, 11 in whichever mode
E1–E3 select, with the certificate's spectral tail as the refinement
signal, plotting Thun's `F_norm` against (M, N) and against wall time with
DESC and VMEC++ 0.7.3 at matching resolution on the same axes. Kill the 3-D
lane as a product if the floor at (M, N) = 11 sits more than 3× above
DESC's at the same resolution; keep it as research with that figure as its
honest status.

## 4. Priorities

The letters match #282 so the two can be merged line by line.

### P0. Acceptance, integration and the cost of working (weeks 1–2)

1. Fix the stale claims in §2 in one PR; extend the prose gate to `.rst`
   pages and to numbers in `CHANGELOG.md`; add a cited-path existence test
   for `benchmarks/` references.
2. Set the jax policy: test 0.9.2 and 0.11.x in CI, tolerances backed by
   observed accuracy, no cross-version bit claims. Bisect #277 with
   `--xla_cpu_use_xnnpack=false` and `--xla_allow_excess_precision=false`
   on the office box under `~/vmex_sweep/env-0.8.0` (jax 0.11.1). Its test
   should assert stationarity at the derivative gate's 1e-8 bar and let the
   derivative call be the check, not SOLVAX's 1e-10 flag.
3. Tier CI to a 25-minute PR ceiling: every test that runs a polish or a
   free-boundary implicit adjoint moves to `full`; split
   `test_polish_preconditioner.py` into Gauss-Newton (PR), homotopy
   (nightly) and linear (PR); `test_run_options.py` keeps one real solve;
   the campaign-class modules in `pr-parity-a1` move to nightly. Fix the
   four jit leaks with the restoring pattern in `test_freeboundary.py:801`.
4. #282's P0 items 2–4 as written (ordinary `_refined_state` eligibility,
   the three-status contract for polished gradients, the AUTO wording).
5. Merge order: #266 and #276 as they stand; #277 after item 2; #282's
   README and validation edits; then the plan the reviewer authorizes.

Gate: no stale claim grep-able on main; every PR lane under 25 minutes on
the shared runners; #277 green on both jax versions; failed roots and
gradients cannot be labelled certified.

### P1. One physical contract and a small matrix (weeks 2–3)

#282's contract stands. Two additions: Thun's `F_norm` with DESC's volume
average over s ∈ [0.1, 0.99] is the *published* residual and the bounded
`eps_F` is acceptance only; near-axis, bulk and edge are always reported
with it. The matrix is #282's five rows. The deliverable is one committed
script, one hashed artifact, one figure: residual versus `ns` for the VMEC
lane and versus spline count and degree for the regular representation, on
the D-shape, the Mb = Nb = 12 W7-X and precise QA, with DESC and VMEC++
points on the same axes.

Gate: the VMEC lane shows its first-order slope and axis behaviour; the
regular representation shows h/p convergence on the axisymmetric cases;
every number in the figure comes from a hashed artifact.

### P2. Formulation and solver (weeks 1–5, beside P0/P1)

E1 in week 1, E2 in weeks 2–3, E3 in weeks 3–5 if E1 passes, the ladder of
§3.3 in weeks 4–6 on the office box. Budgets: ≤ 2 GiB reference matrices,
≤ 10 minutes per diagnostic job, ≤ 30 minutes per nonlinear demonstration
including setup, enforced by the experiment runner. No new knobs on the
polish lane while this runs.

Gate: the decision of §3.1 written in the logbook with numbers, and the
ladder figure.

### P3. Derivatives and one validated design (weeks 3–7)

1. External oracle first (one week): on `input.shaped_tokamak_pressure_polished`
   and the QA deck, compare VMEX's implicit boundary gradient with VMEC++
   0.7.2's adjoint gradient and simsopt's analytic gradient as vectors
   (relative L2 and angle). Accept at 1e-6 relative on the tokamak and 1e-4
   on QA; a larger disagreement is a finding to report, never a tolerance
   to widen.
2. Then #282's Taylor test: independent nonlinear re-solves over a
   perturbation sequence, objective Taylor remainders with rates over at
   least four halvings (target 1.9–2.1), the simsopt `0.3·err_old` rule,
   duality at 1e-6, forward/reverse agreement, adjoint residual as a hard
   gate, under JIT. This replaces the fixed-step FD checks; it does not
   duplicate them.
3. One ordinary QA optimization profiled as #266 did, with the refinement
   and adjoint stages as the targets; linearization reuse and preconditioner
   update frequency before any new machinery.
4. Flagship: Landreman–Paul precise QA from simsopt's
   `input.LandremanPaul2021_QA` and DESC's `precise_QA.py`, validated by
   simsopt's QS ratio residual, a Boozer spectrum, ε_eff, ESSOS orbits and
   bootstrap consistency of the final equilibrium, with total cost including
   compilation. The withdrawn 2026 DESC free-boundary paper is the reason
   bootstrap consistency is a gate.

Gate: three-way gradient agreement published; a before/after design with
independent validation; the student example finishes in minutes on a
documented CPU.

### P4. Evidence, documentation, code size, publication (continuous)

Documentation: #282's README (220 lines) with a 300-line ceiling in the
prose gate; the plan at ≤ 400 lines with the review baseline, measurements
and literature map in `benchmarks/review_20260905.md` and
`docs/explanation`; CHANGELOG at release-note grade with a 25-line unreleased
ceiling; the validation page in the reference contracts; plan-like sections
out of `vmec2000-compatibility.rst` (690–786) and status sections out of
`mirror-geometry.rst`; the benchmark records as the single narrative owner of
every polish number; a generated `benchmarks/INDEX.md` and a test that fails
on any cited artifact that does not exist. Root prose goes from 2,228 to
about 1,050 lines.

Code: the homotopy route (`polish_driver.py` 62–85 and 804–1578 plus its
~600-line preconditioner family in `polish.py`) to `polish_homotopy.py` with
nightly tests; delete the `vmec_jax` shim, `freeboundary_diff.py`, the
`boozer_bmnc_*` aliases, `wout_field_names`, `value_and_grad_bnormal`;
retire `freeboundary_linear.py`; privatise the 120 test-only public
definitions; split `optimize.py` into objectives and drivers; merge the
duplicated `_tree_norm`, lazy-export block and matplotlib guard. About
1,600 lines leave `vmex/` with no product path touched.

Parallelism: ensembles and placement stay; no distributed single-solve work
(no equilibrium code ships it; DESC's PR is a year open); rename the
vocabulary to "placement".

Publication: paper 1 is the product (compatibility, certificate, three-way
gradient agreement, the residual figure, cross-code table, timing on named
hardware, archived inputs, CITATION.cff, AI-assistance disclosure, JOSS's
six-month history and impact statement if JOSS is chosen). Paper 2 exists
only with P2's solved finite-beta 3-D case and the ladder figure.

## 5. Set aside, with entry criteria

- Distributed single-solve sharding: a named workload that does not fit one
  device.
- Normal-equation preconditioners, tensor kernels, promoted-state
  differentiation: E1–E3 decided.
- Anisotropic closure: a mirror design question that needs it; DESC's
  released anisotropic force balance is the oracle then.
- 3-D mirrors and hybrids beyond validated scopes: paper 3 scoped.
- New diagnostics: a downstream consumer with a parity test.
- Neural correction, deflation, further Krylov variants, cross-version bit
  reproducibility, a generalized toroidal angle: not before the above.

## 6. Order and budgets

| When | Work | Budget | Gate |
|---|---|---|---|
| Week 1 | P0 items 1–3; E1 | 2 days docs/CI, 1 day E1 | stale claims gone; lanes ≤ 25 min; E1 verdict |
| Weeks 1–2 | P0 items 4–5; docs and code slimming PRs | 3 PRs | root prose ≈ 1,050 lines; 1,600 lines out of `vmex/` |
| Weeks 2–3 | E2; P1 figure and matrix | ≤ 2 GiB, ≤ 10 min per job | E2 verdict; residual figure |
| Weeks 3–5 | E3 if E1 passed; P3 items 1–2 | ≤ 30 min per demonstration | solver decision; gradient agreement |
| Weeks 4–6 | ladder on the office box; P3 item 3 | four weeks of office wall time | ladder figure; profiled optimization |
| Weeks 5–7 | flagship precise QA | one run with archived inputs | validated design |
| After | paper 1 package; one release | — | all gates above |

## 7. Open pull requests

| PR | Disposition |
|---|---|
| #266 | Merge as is; its coil-length target and seeding note are consistent with the merged #265. |
| #276 | Merge as is after its final CI. |
| #277 | Merge after P0 item 2; the gate is right, the assertion is on the wrong quantity. |
| #274 | Superseded by this document; close when the reviewer merges the plans. |
| #282 | README and validation edits merge regardless; its plan is reconciled here. |

## 8. Logbook

**2026-09-06, this take.** Branch `review/plan-third-take-20260906`, based on
`ae0e410f`, planning only. Evidence: the local lane runs and audits in §2;
release pages for VMEC++, DESC, GVEC, SPECTRE and jax checked the same day;
Thun et al. arXiv:2507.03119; Hirshman and Betancourt, J. Comput. Phys. 96
(1991); Jang, Conlin and Landreman, arXiv:2509.16320; simsopt's Taylor test
rule; pyadjoint's verification page; JOSS review criteria. The office
workstation was unreachable during this pass, so the jax 0.11.1
reproduction of #277's assertion (`~/vmex277/run277b.log`) is pending. No
production code was changed.

Every implementation PR appends one entry here: base and head, owning gate,
command and environment, result with units, time and memory, limitation,
next action. Abandoned experiments stay as evidence with their stop reason.
