# VMEX: focused research plan and execution logbook

Reviewed **2026-09-06** against main **`ae0e410f`**, version **0.8.1**.
This replaces the previous A–J schedule and reconciles the proposals in
[#274](https://github.com/uwplasma/vmex/pull/274). It is a planning change:
no new equilibrium algorithm, derivative guarantee or performance result is
being promoted. **Do not cut a release now.** Resolve the outstanding PRs and
the important gates below before proposing a release.

## 1. Decision: fewer promises, stronger results

Keep the VMEC-compatible solve as the reliable starting point for toroidal
research. Concentrate implementation on **correct acceptance, a demonstrably
better force-balance formulation, and time to a validated optimized design**.
The high-order method deserves one bounded recovery experiment. It does not
yet deserve an expanding family of solvers, coordinate systems or long runs.

The strongest near-term product is a small, documented equilibrium and
optimization API with trustworthy status and reproducible CPU/GPU evidence.
The ambitious result remains accurate native 3-D force balance with useful
implicit derivatives. Separate that research question from routine fixes:
ordinary boundary optimization should not wait for distributed polishing,
anisotropic mirrors or a replacement coordinate system.

| Priority | Deliverable | Owner | Exit decision |
|---|---|---|---|
| P0 | Honest solve/derivative status and reconciled PRs | VMEX | Failed nonlinear or linear solves cannot silently supply certified gradients |
| P1 | One physical force contract and a small benchmark matrix | VMEX | Native accuracy, discretization, reconstruction and solver error are distinguishable |
| P2 | Bounded recovery of the high-order correction | VMEX physics; SOLVAX iteration primitives | Promote only demonstrated accuracy; otherwise retain explicit experimental status and record the obstruction |
| P3 | Faster valid gradients and useful optimization examples | VMEX + SOLVAX; ESSOS coils | A feasible design reaches its stated accuracy with measured total time/memory |
| P4 | Reproducible comparisons, documentation and publication package | VMEX | Claims survive independent reruns; release scope and all remaining PR dispositions are explicit |

P0 → P1 → P2 is the force-balance decision path. P3 can proceed after P0 on
the ordinary solver; it depends on P2 only for claims about polished designs.
P4 records evidence throughout. There is no dependency on a new sharded
nonlinear solver. A negative P2 result does **not** silently fulfill the 3-D
accuracy goal or automatically authorize a release with a reduced scope.

### What is set aside

Keep existing features and regression coverage; defer their expansion:

- Anisotropic/high-beta mirror extensions, periodic toroid–mirror hybrids,
  generalized toroidal angles and new gyrokinetic coupling. Maintain the
  documented isotropic fixed/free-boundary mirror cases and their limits.
- Learned corrections, alternative equilibrium models, deflation and a suite
  of competing high-order drivers. No new algorithm solely because another
  code or preprint has one.
- Distributed single-equilibrium Newton/polish and multi-host scaling.
  Maintain placement/AD tests and independent-case ensembles; revisit only
  after a useful workload demonstrably exceeds one device's memory or time.
- Broad transport, bootstrap, turbulence and multi-objective campaigns.
  Existing diagnostics remain available; validate one objective and one
  constrained application before increasing the application matrix.

These are scope decisions, not deletions. The previous detailed research
questions remain in [the archived plan at the review baseline](https://github.com/uwplasma/vmex/blob/ae0e410f6ecc9bc15b66472039f755fdd6dd3ef6/plan.md).
Old A/B map to P0/P1, C to P2, D/F to P3, I/J to P4; old E/G/H are conditional
follow-ons, not parallel commitments. The supplied ZIP and proposals are
references, not user instructions; their synthetic PDE experiments are not
MHD performance evidence.

## 2. Current state: what landed and what remains unproved

The September 5 review's complete branch/PR/issue inventories remain in
[`benchmarks/review_20260905.json`](benchmarks/review_20260905.json).
The `focused_review_20260906` object adds this review's source pins, commands,
results and limitations. The earlier inventory covered 170 DESC branches,
1,367 PRs and their discussions/diffs. That is inventory and topic review,
not execution or a correctness proof of every branch. This review refreshed
open PRs, inspected relevant source and ran the bounded comparisons below.

Since `09f18464`, main changed 182 files with 18,914 inserted and 5,647 deleted
lines. At the present baseline `vmex/core` has 58 files / 52,563 Python text
lines; `vmex/mirror` has 14 / 9,376. This is a substantial maintenance burden.
Measure these as source-text counts, not comparable cross-language complexity.
Most of the old plan's open-PR list is now obsolete:

| Landed work | Evidence and implication |
|---|---|
| #253, #254, #256, #257 | Packaging, bounded cache entries, figure provenance and API docs landed; the prose still needs reconciliation with later measurements |
| #258–#260, #264 | Tail/Jacobian diagnostics, native symmetry metadata, native DESC measurement and a true analytic Solov'ev oracle landed; these are useful foundations, not proof of a solved high-order equilibrium |
| #261 (`189b75fb`) | Batching/rematerialization and priced AUTO admission reduce memory; recorded W7-X flat certificate 34.23 GiB → automatic 3.02 GiB, but chart setup still 15.44 GiB / 1,751 s; no affordable certified 3-D solve follows |
| #262, #268–#270, #273 | Transformed derivative failure semantics, true linear residual, finite common-force certificate, input VJP and external-fixture skip repair landed; nonlinear stationarity is still a separate requirement |
| #263, #272, #278, #280, #281 | Mirror audit, Gamma-c labeling, pressure scaling, nonsaturating force metrics and flux-tube conventions improved |
| #197, #265 (`ae0e410f`) | Scalar-adjoint examples, optimizer progress and free-boundary workflow changes landed; useful optimization cost remains much larger than a warm primal call |

### Open PR disposition at this review

These are review recommendations, not merge approvals. Recheck the head and
required CI/reviews immediately before any merge.

| PR / reviewed head | Disposition and remaining work |
|---|---|
| [#277](https://github.com/uwplasma/vmex/pull/277), `db6092b7` | Highest priority. Necessary nonlinear stationarity gate; local focused tests pass, but Linux e-polish CI fails the real-MHD `least_squares_success` assertion. Reproduce and explain before merging; do not relax the certificate to obtain a badge |
| [#266](https://github.com/uwplasma/vmex/pull/266), `daf6e2dc` | Preserve its profiling and explicit coil target. CI passes at this snapshot. Review physical target/feasibility and gradient policy, refresh stale dependencies and correct the claim that iteration counts identify conditioning or that forward and transpose operators are identical |
| [#276](https://github.com/uwplasma/vmex/pull/276), `e1714014` | CI passes; bounded format/output cleanup. Verify lossless output and figure hashes against its final head. It can be reviewed independently of force-balance research |
| [#274](https://github.com/uwplasma/vmex/pull/274), `308d39d5` | This plan supersedes its proposed schedule. Preserve the useful bounded recovery and publication-scope decisions; do not merge two competing plans. Close as superseded after the replacement is accepted |

#274 adds roughly 477 lines to an already long plan. Some statements are too
strong: Taylor-style stationarity tests already exist; a spectral projection
tail is not a proven lower bound on achievable nonlinear force error; and a
published resolution study cannot establish a universal W7-X error floor for
a different representation, norm and pressure profile. Its proposed immediate
0.8.2 tag conflicts with the requested release hold.

### New local evidence

CPU: Apple M3 Max, 36 GiB, Python 3.11.14, JAX/jaxlib 0.9.2, float64.
Compilation cache disabled for the VMEX probes. The host also had unrelated
jobs: the following timings are diagnostic samples, **not speedup rankings**.

| Execution | Result | What it establishes |
|---|---|---|
| Main radial basis + strong force suites | 67 passed, 619.75 s | Existing axis/force identities and rejection checks pass locally |
| Analytic Solov'ev + selected ordinary implicit FD/preconditioner tests | 9 passed, 148.21 s | Closed-form projection/refinement and selected derivative consistency pass |
| #277 nonlinear stationarity selection | 26 passed, 219.70 s | Local real-MHD fixture passes; portability unresolved: CI JAX 0.11.1 / Python 3.12 has 1 failed, 226 passed in 2,519.45 s |
| Fresh VMEC++ source Hessian/adjoint tests against its 0.7.3 wheel | 4 passed, 1.79 s | HVP/adjoint examples work in this environment; the wheel is not a build of fresh main |
| Same shaped-tokamak deck, VMEX / VMEC++ | Both converge; R/Z coefficient relative L2 below 1.2e-15, lambda below 3e-14 | Strong compatibility evidence on this case; iteration counts 158/159 |
| Same finite-beta QA deck, VMEX / VMEC++ | Both converge; relative L2 R 3.73e-6, Z 1.79e-5, lambda 3.87e-4, iota 4.06e-7 | Small differences, not universal roundoff agreement; iterations 730/775 |
| Fresh native DESC solve from the tokamak WOUT, L=12/M=6/N=0 | Converged in 28 iterations; native mean-force/mean-pressure-gradient 8.06e-5 | Actual native DESC execution; imported profiles/initialization and quadrature still need matching for a ranking |
| Fresh GVEC source, analytic GS elliptical example | Builds/runs; force channels reach 2.87e-9, 3.58e-9, 3.75e-8 after 3,000 iterations but the configured stopping test remains unsatisfied | Source build and bounded execution succeeded; an exit code of zero and “successfully finished” are not convergence certificates |

The VMEX first/warm and VMEC++ times were 3.33/0.0422/0.00692 s for the
tokamak and 8.42/1.16/0.258 s for QA. These are two small cases, with one warm
sample, one VMEC++ thread, and shared-host load. Retire blanket performance
claims; neither these measurements nor previous selected runs establish a
cross-code time-to-accuracy advantage. GVEC's example is a different problem
and is not included in that timing comparison. No new GPU result is claimed.

## 3. What the literature and source change about our choices

### Native force accuracy must lead the comparison

[Panici et al., DESC Part I](https://arxiv.org/abs/2203.17173) demonstrates why
near-axis force accuracy and radial resolution affect quantities such as
Mercier stability. Reproducing the VMEC discrete trajectory does not remove
that issue. Use its force definitions and evaluation-region conventions as
an explicit benchmark protocol, while also reporting the whole volume and
the magnetic-pressure normalization for near-vacuum cases.

The current validation page incorrectly ranked native DESC using an exported,
refitted WOUT. The merged native DESC result is 4.01e-6 in its pressure-gradient
ratio, whereas the refitted record is about 7.13e-2 in a different pointwise
L2 metric. Dividing these is not a measured export amplification factor.
This PR corrects that interpretation and preserves the historical numbers
as reconstruction evidence in [validation](docs/explanation/validation.md).

The recorded tokamak polish improves the bounded pointwise score 7.12 times,
but dimensional volume L2 only 1.61 times and the full-volume pressure ratio
1.32 times; near-axis dimensional L2 improves 14.5 times. These are different
claims. A projected analytic Solov'ev reaches about 7.44e-10 geometric error
with degree five and a force roundoff floor near 1e-10; that proves useful
representational capacity, not that the nonlinear solver finds that state.

### VMEC++ is a serious baseline, not a frozen historical comparator

[The Numerics of VMEC++](https://arxiv.org/abs/2502.04374) documents the
variational discretization, preconditioning and iteration machinery.
Current source retains the damped VMEC-style iteration; its improvements
include engineering of the implementation and the response interface.
[0.7.2](https://github.com/proximafusion/vmecpp/releases/tag/v0.7.2) introduced
the boundary-adjoint example; [0.7.3](https://github.com/proximafusion/vmecpp/releases/tag/v0.7.3)
adds LFORBAL and restart fixes, LTO builds and x86 dispatch improvements.
The tested Hessian product uses centered finite differences of analytic
forces, not exact automatic differentiation. Its adjoint test independently
reconverges perturbed equilibria. Adopt that verification discipline; do not
infer that a public HVP means a globally robust Newton equilibrium solver.
Pin a released binary as well as source and compare one-thread and OpenMP
costs separately; the [validation repository](https://github.com/proximafusion/vmecpp-validation)
is useful prior art for broad input compatibility.

### DESC and GVEC suggest formulation and preconditioning work

The fresh DESC checkout (`ad105c5e`) uses Fourier–Zernike regularity and a
least-squares trust-region implementation whose default subproblem method is
QR, with SVD/Cholesky alternatives. It is not evidence that replacing VMEX's
CG with LSMR alone solves the physical problem. Relevant open work includes
[#2286](https://github.com/PlasmaControl/DESC/pull/2286) on linearization reuse,
[#1773](https://github.com/PlasmaControl/DESC/pull/1773) on sharding, and
[#1877](https://github.com/PlasmaControl/DESC/pull/1877) on near-axis evaluation.
Their branch status and claims remain separate from released behavior.

[GVEC's theory](https://gvec.readthedocs.io/latest/user/theory.html) separates
the geometry map, radial B-splines/Fourier representation and energy descent.
The source at `c0dc66fe`, `src/functionals/mhd3d/mhd3d_evalfunc.F90`, constructs
mode-separated banded radial preconditioners for both geometry fields and
lambda. This is a concrete template for deriving an approximate VMEX
operator, not justification for copying GVEC's whole coordinate framework.
Its independently adjustable geometry components also motivate auditing
VMEX's present correction chart before optimizing its linear algebra.

## 4. P0 — make acceptance and PR integration reliable

Owner: VMEX (`implicit.py`, `polish_driver.py`, `polish_implicit.py`, public
optimization wrappers); SOLVAX owns generic true-residual reporting.

1. Finish #277 with the actual failing Linux/JAX 0.11.1 context. Record
   Python/JAX/SciPy versions, initial residual scaling, stopping reason and
   attained stationarity. The local JAX 0.9.2 result is not sufficient.
   Keep the tight real-MHD fixture; change an algorithm or justified stopping
   definition only after identifying the discrepancy.
2. Audit ordinary `implicit._refined_state` as well. It already applies a
   matrix-free Newton/GCROT refinement, but the implementation explicitly
   allows a best-effort nonroot to be returned. Gate derivative eligibility
   on the actual nonlinear equation, admissible geometry and true linear
   residual. Small adjoint residual alone cannot certify the implicit
   derivative at an unconverged primal. Preserve value-only fallback with
   explicit status where useful.
3. For high-order least squares, distinguish physical force acceptance,
   stationarity `g = Jᵀr`, and the true residual of the stationarity derivative
   solve. All three are needed for a certified polished gradient. Host callers
   and `jit`/`vmap` callers need consistent status; transformed NaN rejection
   must not become a finite “best effort” gradient in an optimizer.
4. Replace the CLI's misleading hard-ceiling description of AUTO. The code
   prices anticipated Gauss–Newton work after setup; it does not interrupt
   end-to-end execution at `--polish-budget`. Record setup, admitted work and
   actual elapsed time separately; an enforced timeout would require a
   separate, explicitly tested cancellation contract.

**Gate:** real converged and failed-root cases under eager/JIT, including
parameter updates and both derivative directions; true residuals recomputed
outside iterative recurrences; no missing-fixture blanket skips. Preserve
coverage and physics tolerances. Integrate #266/#276 only after their final
review, and resolve #274 through this replacement rather than additive logs.

## 5. P1 — define the physical problem and a small accuracy matrix

Owner: VMEX. Reuse the existing certificate, analytic oracle and benchmark
runner; consolidate them instead of adding another scoring implementation.

### One comparison contract

Record boundary, axis convention, NFP/LASYM, units, total toroidal flux,
pressure and prescribed-iota/current functions, GAMMA/mass closure, radial
coordinate and reference field/length. Hash the deck and native state. Compare
physical functions, not just similarly named coefficient arrays. A projected
state, a nonlinear solve and a WOUT reconstruction are different objects.

Report Cartesian `F = J × B − ∇p` with converged quadrature: dimensional
volume L1/L2/L∞, `mean(|F|)/mean(|∇p|)` where meaningful, magnetic-pressure
normalization, and a fixed physical reference `B_ref²/(mu0 L_ref)` when a
normalization vanishes. Include the axis neighborhood, bulk and edge, the
whole volume, and an explicitly named literature window. State whether a
window uses normalized flux `s` or `rho = sqrt(s)`; `[0.1,0.99]` means different
regions in these coordinates. Never hide the axis by trimming it from the
only quoted score. The bounded pointwise `eps_F ≤ 2` remains secondary.

Certify on an off-grid mesh with independently evaluated derivatives and
radial/angular refinement. Do not call two paths independent if they share
the derivative or normalization being tested. Analytic identities and
manufactured forcing test discretization; native cross-code comparison tests
an actual equilibrium. Sampled Jacobian orientation does not prove global
injectivity: also check boundary/interior geometry and crossing diagnostics.

### Minimum matrix, expanded only when a gate needs it

| Case | Purpose and acceptance evidence |
|---|---|
| Closed-form Solov'ev equilibrium from `test_strong_force_solovev.py` | Distinguish projection error from recovery of the same solution from perturbed coefficients; demonstrate radial h/p convergence, analytic axis derivatives and correct pressure/current conventions |
| `input.shaped_tokamak_pressure_polished` | Baseline same-input VMEC++/VMEC2000 parity and native high-order force/gradient test; do not substitute a merely similarly named Solov'ev input |
| `input.nfp2_QA_smooth_beta` | First genuine finite-beta 3-D recovery; native DESC with matched boundary/profiles and a quadrature-converged physical norm |
| Existing vacuum and LASYM small decks | Vanishing-denominator behavior, orientation, Fourier families and ordinary implicit derivatives |
| One prescribed-current deck and one small NESTOR/ESSOS case | Ordinary solver closure and free-boundary interface regression; admission to high-order polishing requires its own closure proof |

Start P2 with fixed boundary, stellarator symmetry, prescribed iota and
GAMMA=0. Audit `phip`, `chip` and pressure updates before admitting NCURR=1 or
finite-GAMMA/mass constraints. Freezing flux/profile arrays while geometry
changes must not silently solve a different prescribed-current problem.
Maintain those existing ordinary-solver features while high-order support
remains narrower. W7-X is a later stress case, not the first recovery test.

**Gate:** reproducible reference values with units/norms, convergence trends
under independent quadrature and physical-profile agreement. Where a native
DESC/GVEC adapter is missing, record the comparison as unavailable; do not
fill it with the WOUT-relift score. Publication comparisons require native
state archives as well as common interchange files.

## 6. P2 — correct the formulation, then choose the solver

### First question: are we minimizing the right functional in enough directions?

In `core/polish.py::_strong_collocation_residual`, the current residual packs
signed radial/helical force densities with `2*abs(sqrt_g)/normalization`.
Squaring that vector is not the Cartesian volume L2 norm: the Jacobian is
squared, Gauss weights are absent, and the physical vector metric/cross terms
are not represented by simply summing component squares.

Derive and implement a reference residual whose squared norm is
`sum_q w_q |sqrt_g(q)| |F_cart(q)|² / F_ref²`. Use square-root quadrature/volume
weights and a declared physical scaling. Geometry-dependent weights belong
to the differentiated objective. A nonorthogonal two-component formulation
must carry the full Gram metric; verify it against Cartesian evaluation.
Use one residual definition for optimization, stationarity and its HVPs;
preconditioning must not accidentally redefine the objective.

`make_strong_structured_chart` retains R/lambda directions while eliminating
Z directions as a coordinate gauge. Removing arbitrary vertical displacement
by a poloidal reparameterization requires division by `Z_theta`, which can
vanish. This is not a general global gauge proof. The current layout also
comes through the prolongation/restriction image of the low-order state, so
local B-splines do not automatically provide independent high-order freedom.

Construct a tiny reference with independent native R/Z/lambda coefficients,
fixed-boundary constraints, exact axis regularity and an explicit gauge
quotient. Check rank, nullspaces and allowed perturbations as h/p increase.
Simply adding every Z coefficient can introduce gauge singularity; simply
removing them can discard physical directions. Compare the existing chart
against this reference before designing a local preconditioner.

### Coordinate decision

| Option | Assessment and trigger |
|---|---|
| Existing `rho^|m| q(s)`, `s=rho²` | Retain first. `radial_basis.evaluate_regularized_mode` already implements this and analytic first/second axis limits. “Regularize the axis” alone is not a new remedy |
| Normalize/localize the regular basis and its constraint elimination | First candidate if rank/conditioning identifies a basis problem. Compare equal function spaces and DOFs; knot spacing in s versus rho changes near-axis resolution |
| Independent geometry components or a regular normal-displacement chart | Test against the full constrained reference. An off-axis normal parametrization still needs independent axis motion and a nonsingular axis limit |
| Smooth polar center-splines | Conditional alternative if powers of rho or global constraints cause conditioning/locality failure; test a small basis replacement, not a second equilibrium framework |
| Better nested-coordinate initialization | Conditional on poor Jacobian/admission at strongly shaped boundaries; separate initialization from the physical objective |
| Generalized toroidal angle or a new global map | Defer for normal toroidal cases. Reconsider only when laboratory-angle geometry is the demonstrated limitation, with new gauge/axis constraints and AD costs budgeted |

[Jiang et al., smooth polar B-splines](https://arxiv.org/html/2601.17841v1),
sections 3.1, 4.7 and 5.1, gives precise regularity and normalization machinery;
it is a 2026 preprint and evidence for a testable basis construction, not a
VMEX performance result. [Tecchiolli et al.](https://arxiv.org/html/2405.08173v1)
(JPP, DOI 10.1017/S0022377824001119) constructs nested initial coordinates for
strong shaping. [DESC #2282](https://github.com/PlasmaControl/DESC/pull/2282)
introduces `phi = zeta + omega`; its unfinished axis/continuation work and
additional unknowns argue against making it our critical path. Its racetrack
comparison is not a matched-boundary proof that VMEX needs that coordinate.

### Would preconditioned Newton with matrix-free Hessian products help?

Potentially, but there are three different operators:

| Problem | Operator and implication |
|---|---|
| Ordinary discrete equilibrium `F(u,p)=0` | Newton uses `F_u v`; this need not equal a symmetric energy Hessian after scaling/constraints. Existing implicit refinement already uses this route. Better late refinement can reduce gradient cost but cannot remove the finite-difference continuum error |
| High-order force least squares `min ½||r||²` | Gauss–Newton uses `JᵀJ`; exact stationarity Newton uses `H = JᵀJ + sum_i r_i ∇²r_i`. Existing polished implicit differentiation needs H. Away from small residuals H can be indefinite, so plain CG is not a general Newton solver |
| Discrete energy minimization | An energy HVP is valid only after verifying that the differentiated constrained energy yields the intended force/closure equations. Do not substitute the current force vector for that proof |

The current `polish_driver._gauss_newton_polish_lane` calls SOLVAX with variable
scaling but does not pass a physics preconditioner or admissibility callback.
This is a concrete gap. The previous BBT-style preconditioner targeted a
different operator and did not demonstrate a useful 3-D result; reusing its
name or factors is not a derivation.

The new frozen **156×17** real-MHD probe, after current scaling, has
`cond(J) ≈ 7.15e3`. At damping 1e-3, CG/diagonal-CG/LSMR took 31/30/35
iterations. At 1e-6 they took 36/32/39. LSMR used tighter stopping controls,
so these counts are not a fair runtime contest. All recovered the dense
augmented least-squares step; exact-Hessian versus Gauss–Newton relative
curvature was 2.08e-4, and a centered HVP difference reached 3.88e-9 relative
error. This is a tiny initial Solov'ev lift using the **current** functional,
not evidence of nonlinear 3-D convergence or a Newton advantage.

[Knoll & Keyes, JFNK review](https://doi.org/10.1016/j.jcp.2003.08.010) motivates
matrix-free products with problem-specific preconditioning and globalization.
[Eisenstat & Walker](https://doi.org/10.1137/0917003) motivates adaptive inner
accuracy, `||F + F_u delta|| ≤ eta ||F||`, rather than oversolving every step.
[PETSc's SNES manual](https://petsc.org/main/manual/snes/) illustrates combining
matrix-free operators with an explicit approximate preconditioner; it is
prior art, not a proposed VMEX dependency.

[LSMR](https://web.stanford.edu/group/SOL/software/lsmr/) provides a rectangular
J/Jᵀ alternative without explicitly forming normal equations. It does not
remove ill-conditioning. `cond(JᵀJ)=cond(J)²` does not imply CG needs that
many iterations; nonlinear aggregate iteration counts exceeding the unknown
count also do not estimate a condition number.

### A bounded experiment with a decision, not a solver competition

1. On the tiny analytic case, assemble J and the constrained reference H for
   verification only. Compare JVP/VJP dot products, centered HVP differences,
   spectrum/rank and augmented QR/SVD reference steps. Check the actual
   nonlinear update on an independent physical certificate.
2. Repeat at one affordable finite-beta QA resolution. Limit reference
   matrix storage to 2 GiB and each diagnostic job to 10 minutes; reduce
   resolution instead of silently exceeding those review budgets. Save a
   failed or capped attempt as such.
3. Only if the reference step is useful, derive **one** mode-block/banded
   radial approximation from the corrected VMEX functional, including axis,
   metric and closure terms. Reuse SOLVAX's existing Krylov/globalization
   APIs; compare equal true residual and objective decrease, including
   setup/update/memory. Compare CG and a rectangular solve only as needed to
   isolate normal-equation accuracy. Test exact Newton only where measured
   residual curvature or slow local convergence justifies its added cost.
4. Use trust-region/line-search acceptance with geometry rejection and
   adaptive forcing. Cap the initial nonlinear demonstration at 30 minutes
   per small case, including setup; the experiment runner must enforce this
   because AUTO currently does not. Record the final state and reason.
5. Demonstrate actual recovery under at least three radial refinements and
   two angular resolutions. Require lower independent dimensional and
   normalized force, satisfied stationarity and constraints, and repeatable
   derivatives. Freeze tolerances before the comparison. For the first QA
   target, seek at least an order-of-magnitude physical-force improvement
   over its converged ordinary solve; this is a proposed promotion target,
   not an existing achievement or an assumed attainable universal threshold.

**Decision:** if even a trusted constrained dense step cannot improve the
independent force, investigate representation, closure or admissibility;
stop tuning Krylov iterations. If it can, but the iterative solve cannot,
fix preconditioning/linear accuracy. If both work but cost is excessive,
profile the demonstrated bottleneck. If the bounded QA recovery fails,
retain experimental status and publish the obstruction in this logbook.
Do not launch W7-X or change the global coordinate model without that diagnosis.

## 7. P3 — optimize the cost that researchers actually pay

### Start with one ordinary QA optimization

[#266's profile](https://github.com/uwplasma/vmex/pull/266) reports a 7.63 s
value/gradient call with about 0.35 s in the VMEC loop, 3.86 s in Newton
refinement and 3.06 s in the adjoint. A max-mode-2 QA run takes about 265 s
including 29 s compilation, 205 s optimizer loop, 7 s finalization and 13 s
plotting. Treat this as a case-specific profile; it makes refinement and
response solves higher priority than accelerating the already cheap primal.
Loosening the forward tolerance made that reported optimization slower.

Reproduce one accepted optimization step with changed parameters, and then
one complete run with fixed seed, objective scaling and feasibility criteria.
Measure compile, primal, refinement, linearization, adjoint/Jacobian, rejected
trials and plotting separately. Repeated identical parameters can hit memoized
results and are not evidence of cheap changed-parameter gradients.

Test existing linearization reuse, preconditioner update frequency and
adaptive refinement first. Recycled forward Krylov vectors cannot be assumed
to solve the adjoint: the operator is transposed and changes with state and
parameters. Recompute the appropriate operator images, recertify the true
residual and bound memory. SOLVAX already has GCROT machinery; extend a
missing generic capability there only after a reproducible VMEX need.

Use JAX/XProf traces, Perfetto and XLA/HLO inspection when stage timing points
to compilation, transfers, allocations or kernels; use GPU counters only for
a specific unresolved kernel question. Record synchronized measurements and
trace overhead. An attractive kernel speedup that worsens total accepted
optimization time does not pass.

### Derivatives and applications

Preserve existing FD and stationarity-Taylor tests. Add the missing test:
independent nonlinear re-solves over a sequence of perturbations for boundary
and profile directions, followed by objective Taylor remainders. Resolve
both perturbed states more tightly than the derivative error being measured;
show a convergence interval before roundoff/solver noise, and check branch
continuity and geometry. Include eager/JIT and CPU/GPU once the CPU contract
passes. Differentiate the actual accepted equilibrium equations, not a finite
iteration history mislabeled as an exact equilibrium derivative.

Use [Landreman & Paul, precise quasisymmetry](https://doi.org/10.1103/PhysRevLett.128.035001)
as the first physics anchor: a reduced QA example for students, with a stated
resolution gap to the paper, and a separate research budget for reproducing
the physical target. Preserve QH/QP/QI scripts but share configuration/output
logic only where it reduces duplication without hiding physics choices.
One validated QA application precedes an exhaustive campaign.

Then validate a tokamak shape/profile example and an ESSOS coil example.
[Jorge et al., single-stage optimization](https://arxiv.org/abs/2302.10622)
provides a concrete target for coupled plasma/coil objectives and independent
field verification. Report coil length, curvature, separation, normal-field
error, geometry and equilibrium accuracy alongside QS/QI. A lower weighted
penalty with infeasible coils or an uncertified equilibrium is not success.
For finite beta, include the plasma exterior contribution consistently.
ESSOS owns Biot–Savart fields, coil geometry and their derivatives; VMEX owns
equilibrium response, interface physics and acceptance. Do not duplicate coil
kernels or move MHD closure into SOLVAX.

**Gate:** cold re-evaluation of the final design, force/gradient/feasibility
checks, preserved restart, readable convergence history and measured total
CPU/GPU cost against the same accepted baseline. Choose a practical runtime
budget from that baseline before optimizing; do not declare an unmeasured
speedup target an achievement. The student example must finish in minutes
on a documented CPU configuration; advanced settings must state their costs.

## 8. P4 — evidence, parallelism, documentation and publication

### Performance and parallelism

Keep a small case manifest in the existing benchmark framework, using P1's
inputs and P3's optimization. Compare VMEX, released VMEC++/VMEC2000 and native
DESC at common physical tolerances and separately at common input resolution.
GVEC joins numerical rankings only after a matched, converged native adapter.
Pin source, executable/wheel hash, dependencies, CPU threads, GPU model,
precision, input hash, status, DOFs, quadrature and all stopping criteria.

Use fresh-process cold, disk-cache reload, warm same-shape **changed-parameter**,
resolution-change and full optimized-design runs. Collect at least five warm
samples and three independent complete runs when affordable; report spread,
failures and censored timeouts. Separate profiling runs from timings. Archive
peak host RSS and live/allocated device-memory measures with their definitions.
The release comparison must use an otherwise idle host and matching thread
budgets, not this review's shared-host samples.

Run one-device CPU and `ssh office` GPU first. Then independent-case throughput
with bounded CPU workers and one worker per GPU, checking oversubscription,
compilation/cache races, deterministic outputs and aggregate memory. Reuse
existing placement tests for correctness. Two host-device emulation tests do
not demonstrate multi-GPU speedup; actual `NamedSharding` metadata alone does
not prove a nonlinear iteration avoids gathers. A distributed equilibrium
implementation is reconsidered only after its single-device bottleneck and
communication budget are measured.

### Documentation and deliberate code size

This PR shortens the README around installation, solve/restart, gradients,
optimization, scenario commands and supported features. Detailed numeric
polish tables and failure records belong in `docs/explanation/validation.md`.
The adjusted documentation tests preserve hashes, values and tolerance checks
while allowing that relocation. Do not require a long benchmark essay on the
landing page to retain scientific accountability.

Continue with one owner per document: tutorials for learning; how-to pages
for tasks; reference for API/model contracts; explanations for derivations;
validation for measured evidence; this plan for unfinished work and decisions.
Correct conflicting “exact gradient” and warm-primal performance language in
other pages during P0/P3. Run examples with expected outputs and bounded
budgets, including one beginner and one advanced workflow from a clean install.

Track added/deleted source lines, file count, installed size and artifact
bytes per implementation PR. Retire superseded residual/driver wrappers and
duplicate tests only after behavior and failure coverage have an owner.
Prefer extending existing modules and examples over new wrappers or framework
layers. Keep compact numeric records and useful figures in git; put native
states, traces, caches, generated intermediates and bulk benchmark output in
a public, persistent archive. Delete unreferenced artifacts only after link,
provenance and reproduction checks. Preserve scientific evidence when pruning.

### Publication and release gates

The first credible paper can center on VMEC compatibility, validated implicit
responses, research optimization and measured execution on CPU/GPU. It still
requires the P0/P1/P3/P4 evidence; current smoke tests are not a paper. A second
high-order methods claim requires P2's solved finite-beta 3-D case, convergence
and time-to-accuracy comparison. Do not inflate one incomplete story into
several promised publications.

Before proposing any release: every current PR is merged or explicitly
superseded/deferred; important acceptance and application gates pass; any
remaining 3-D limitation has an explicit scope decision; the supported
Python/JAX matrix, source/wheel install, required CI, docs and examples pass.
Freeze inputs, environments, traces and native states in an accessible archive,
record licenses/citation metadata and regenerate claims from those records.
There is no scheduled tag, automatic version bump or publication date here.

## 9. Execution logbook and handoff

**2026-09-06 — planning review, no production solver edits.** Branch
`review/focused-plan-20260906`, worktree
`/Users/rogeriojorge/local/vmex-focused-plan-20260906`, based on `ae0e410f`.
Reconciled main and four open PRs, reviewed #274/#266 and relevant DESC changes,
cloned VMEC++/GVEC/DESC, ran the tests and operator/comparison probes in §2/§6,
rewrote this plan and the README, and corrected the native/WOUT interpretation
on the validation page. Documentation validation passed: Ruff, mypy, prose/media checks, strict
Sphinx HTML, linkcheck, 59 guard tests and seven figure-provenance tests.
The README Python solve/gradient/optimization/export walkthrough ran on the
bundled circular tokamak and converged in three optimization evaluations.
The standalone 19 performance-record tests also passed (included in the
59 guards, not additional distinct tests). Planning commit `814af87a` is published in
[PR #282](https://github.com/uwplasma/vmex/pull/282); CI/review are pending at
handoff. No merge, release or production solver implementation was performed. All authored commits use `rogeriojorge`.

Raw logs, native outputs, isolated environment and the two short review probes
are in `/Users/rogeriojorge/local/vmex-review-evidence-20260906`; the compact
results and probe sources are embedded in the existing review JSON so this
plan remains usable without that filesystem. Fresh external source pins:
VMEC++ `07ef6710078e78e29ccabab0443cb3ec0ad7e375`, GVEC
`c0dc66fe2b9faa147c76a728e5cd053ce693d472`, DESC
`ad105c5e525fbf26824d6cf9dde48775db0f8a2c`; SOLVAX remains
`5a49926992fe1a3aebac4b8b8cb098798e977c14`. DESC distribution metadata came
from an inherited environment and was stale; the executed source path/SHA
identify the actual checkout. No other user's dirty checkout was changed.

Reproduction commands, from the VMEX checkout (review probes use float64):

```bash
VMEX_COMPILATION_CACHE=disabled JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_radial_basis.py tests/test_strong_force.py
VMEX_COMPILATION_CACHE=disabled JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_strong_force_solovev.py tests/test_implicit_grad.py \
  -k 'not test_implicit_grad or test_solovev_gradients_vs_fd or test_adjoint_gmres_preconditioner_value'
# In a detached worktree at PR #277 head db6092b7:
VMEX_COMPILATION_CACHE=disabled JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_polish_preconditioner.py \
  -k 'collocation_polish_primal_and_derivatives or polish_stationarity'
```

The review JSON contains commands for the pinned VMEC++ tests, same-deck
comparison, frozen operator, native DESC runner and GVEC build/capped solve.
It preserves failed attempts and environment limitations. Do not promote a
result based only on a process exit code or a private raw-log path.

**Next implementation PRs, in order:**

1. Resolve #277's CI failure and audit ordinary nonlinear derivative
   eligibility (P0); update the status contract and affected examples.
2. Complete the native physical-force/profile contract and the small
   constrained chart/functional reference (P1/P2). Make the recovery decision
   from that evidence before a new solver, coordinate rewrite or W7-X run.
3. Improve the measured refinement/response bottleneck on one accepted QA
   optimization (P3), using SOLVAX only for genuinely generic algorithm work.

After each implementation change, append one compact entry here: exact
base/head and PR, owning gate, command/environment, attained numerical result,
time/memory, limitation/failure and next action. Update the priority's status
in place. Keep abandoned experiments as evidence with a stop reason, rather
than appending new work packages. A resumed agent should first read this
section, check dirty worktrees and current remote PR status, then continue the
first unmet gate. Do not rerun the full historical inventory or a multi-day
polish unless new evidence justifies it.
